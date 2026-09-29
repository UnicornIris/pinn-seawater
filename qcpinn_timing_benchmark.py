"""
Stage-resolved timing benchmark: classical PINN vs. hybrid QCPINN.

One self-contained run that produces everything the paper's timing section needs, in one
process and one output folder, so that all numbers come from the same machine state:

  main         classical + 4 hybrid configs + 3 size-matched classical controls
               (the hybrid net with the quantum layer replaced by tanh, so it has exactly the
               classical share of the hybrid's parameters), all four stages
  layer_sweep  4 qubits, n_layers in LAYERS, all four stages
  batch_sweep  baseline (4 qubits, 2 layers), batch in BATCHES, all four stages
  qubit_sweep  2 layers, n_qubits in QUBITS; all four stages up to --max-qubits-grad,
               forward only above it (second-order graphs need a lot of RAM)
  backend      circuit-only forward on default.qubit vs. lightning.qubit
               (configs at the default batch, plus a batch sweep on the baseline circuit)
  drift_check  a few `main` cases measured again at the end; compares against `main` to show
               whether the machine's state drifted during the run

Stages (cumulative, as in qcpinn_timing_profile.py):
  forward      no-grad evaluation of the network
  first_grad   forward + d T / d(z,t)                       (create_graph=True)
  second_grad  forward + d T / d(z,t) + d^2 T / d z^2
  param_grad   second_grad's graph + PDE-residual loss + backward to the parameters
               (the optimiser's own backward pass; ~ the cost of one training step's PDE term)

Quantum vs. classical split (hybrid cases only, `main`/`layer_sweep`/`batch_sweep`/`qubit_sweep`)
  Forward-pre/post and full-backward-pre/post hooks on the net's QuantumLayer submodule bracket
  the wall time spent inside the quantum circuit (forward evaluation and, when the stage takes a
  gradient, the backward pass through it) during the very same call used for the `forward` /
  `first_grad` / ... row. Each such call therefore also emits `<stage>_quantum` (time inside the
  hooks) and `<stage>_classical` (the remainder: pre/postprocessor MLP + autograd bookkeeping
  outside the hooked module) rows, so the two add back up to the original `<stage>` row for that
  repetition. Classical-only cases (`classical`, `ctrl_*`) have no quantum component and get no
  such rows.

Measurement design
  * Cases are interleaved: in every round, each case (config x seed) is timed once, in a
    shuffled order, so slow drift of the machine (thermal throttling, background load) hits all
    cases alike instead of biasing whichever ran last.
  * Every timed call is stored (raw/*.csv, long format). Mean, std across seeds, median and
    within-seed CV are computed from those rows.
  * Repetitions adapt to cost: reps = clip(point_budget / one-round-time, min_reps, max_reps).
  * Before the first timed section, --settle seconds of untimed work bring CPU clocks and thread
    placement to steady state.
  * The drift check compares medians (end of run vs. `main`) and bases its verdict only on cells
    taking at least DRIFT_MIN_MS; faster cells are reported but are too jittery to judge drift.
  * Environment (CPU, RAM, versions, threads, power state, git commit, arguments) goes to env.json.

Usage
  python qcpinn_timing_benchmark.py --quick --out-root /tmp/tb_test   # 1-2 min smoke test
  python qcpinn_timing_benchmark.py --tag my_machine                  # full run
  python qcpinn_timing_benchmark.py --tag my_machine --resume         # continue after interruption
  python qcpinn_timing_benchmark.py --tag my_machine --summarize-only # rebuild summary/report

Before a full run: plug in the power, close other applications, disable sleep
(macOS: `caffeinate -i python qcpinn_timing_benchmark.py ...`), and do not use the machine.

Needs qcpinn_seawater.py next to this file (model definitions).
Requirements: torch, pennylane, numpy, matplotlib; pennylane-lightning for the backend section.
"""
import argparse
import csv
import datetime
import gc
import glob
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

from qcpinn_seawater import ClassicalDNN, HybridQNN, Z_MIN, Z_MAX, T_MIN, T_MAX

KZ = 0.1  # diffusivity in param_grad(); its value does not change the cost
STAGES = ["forward", "first_grad", "second_grad", "param_grad"]
SECTIONS = ["main", "layer_sweep", "batch_sweep", "qubit_sweep", "backend", "drift_check"]
HYBRIDS = {"narrow": (2, 2), "baseline": (4, 2), "wide": (6, 2), "deep": (4, 4)}  # (qubits, layers)
CONTROL_QUBITS = (2, 4, 6)
CONTROL_OF = {"narrow": "ctrl_n2", "baseline": "ctrl_n4", "wide": "ctrl_n6", "deep": "ctrl_n4"}
DRIFT_CASES = ("classical", "baseline", "wide")
DRIFT_TOL = 0.10     # largest allowed |median ratio - 1| in the drift check
DRIFT_MIN_MS = 5.0   # cells faster than this (median in `main`) are shown but not used for the verdict:
                     # sub-5 ms calls are dominated by dispatch overhead and OS scheduling jitter
RAW_FIELDS = ["section", "case", "kind", "backend", "n_qubits", "n_layers", "batch", "n_params",
              "seed", "stage", "rep", "seconds"]


# ── stages (same computations as qcpinn_timing_profile.py) ───────────────────────────────────
def forward_only(net, zt):
    with torch.no_grad():
        net(zt)


def first_grad(net, zt):
    zt = zt.detach().clone().requires_grad_(True)
    T_pred = net(zt)
    torch.autograd.grad(T_pred, zt, torch.ones_like(T_pred), create_graph=True)[0]


def second_grad(net, zt):
    zt = zt.detach().clone().requires_grad_(True)
    T_pred = net(zt)
    dT = torch.autograd.grad(T_pred, zt, torch.ones_like(T_pred), create_graph=True)[0]
    dT_dz = dT[:, 0:1]
    torch.autograd.grad(dT_dz, zt, torch.ones_like(dT_dz), create_graph=True)[0]


def param_grad(net, zt):
    net.zero_grad(set_to_none=True)
    zt = zt.detach().clone().requires_grad_(True)
    T_pred = net(zt)
    dT = torch.autograd.grad(T_pred, zt, torch.ones_like(T_pred), create_graph=True)[0]
    dT_dz, dT_dt = dT[:, 0:1], dT[:, 1:2]
    d2T_dz2 = torch.autograd.grad(dT_dz, zt, torch.ones_like(dT_dz), create_graph=True)[0][:, 0:1]
    loss = ((dT_dt - KZ * d2T_dz2) ** 2).mean()
    loss.backward()


STAGE_FNS = {"forward": forward_only, "first_grad": first_grad,
             "second_grad": second_grad, "param_grad": param_grad}


# ── models and data ──────────────────────────────────────────────────────────────────────────
def make_classical(seed):
    torch.manual_seed(seed)
    return ClassicalDNN()


def make_hybrid(n_qubits, n_layers, seed):
    torch.manual_seed(seed)
    return HybridQNN(n_qubits=n_qubits, n_qlayers=n_layers, quantum_init_std=None)


def make_control(n_qubits, seed):
    """Hybrid net with the VQC replaced by tanh: same classical layers, no quantum parameters."""
    torch.manual_seed(seed)
    net = HybridQNN(n_qubits=n_qubits, n_qlayers=1, quantum_init_std=None)
    net.quantum = nn.Tanh()
    return net


def make_batch(seed, batch):
    rng = np.random.RandomState(seed)
    z = rng.uniform(Z_MIN, Z_MAX, batch)
    t = rng.uniform(T_MIN, T_MAX, batch)
    return torch.tensor(np.stack([z, t], axis=1).astype(np.float32), requires_grad=True)


def n_params(net):
    return sum(p.numel() for p in net.parameters())


def circuit(x, params, n_qubits, n_layers):
    """Same circuit as QuantumLayer._circuit in qcpinn_seawater.py."""
    qml.templates.AngleEmbedding(x, wires=range(n_qubits), rotation="X")
    for layer_idx in range(n_layers):
        p = params[layer_idx]
        idx = 0
        for q in range(n_qubits):
            qml.RZ(p[idx], wires=q); idx += 1
            qml.RX(p[idx], wires=q); idx += 1
        for q in range(n_qubits):
            qml.CNOT(wires=[q, (q + 1) % n_qubits])
        for q in range(n_qubits):
            qml.RX(p[idx], wires=q); idx += 1
            qml.RZ(p[idx], wires=q); idx += 1
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


# ── quantum/classical split (hybrid cases only) ─────────────────────────────────────────────────
class QuantumTimer:
    """Wall time spent inside a QuantumLayer submodule's forward + backward, via module hooks.

    Forward-pre/post hooks bracket the forward call; full-backward-pre/post hooks bracket the
    backward pass through that module (fired once per `torch.autograd.grad`/`.backward()` call
    that touches it, so second-order stages accumulate over both backward passes). Hooks are
    plain wall-clock brackets around synchronous CPU ops, so nesting/ordering is not an issue.
    """

    def __init__(self, module):
        self.total = 0.0
        self._t_fwd = None
        self._t_bwd = None
        module.register_forward_pre_hook(self._fwd_pre)
        module.register_forward_hook(self._fwd_post)
        module.register_full_backward_pre_hook(self._bwd_pre)
        module.register_full_backward_hook(self._bwd_post)

    def _fwd_pre(self, module, inputs):
        self._t_fwd = time.perf_counter()

    def _fwd_post(self, module, inputs, output):
        self.total += time.perf_counter() - self._t_fwd

    def _bwd_pre(self, module, grad_output):
        self._t_bwd = time.perf_counter()

    def _bwd_post(self, module, grad_input, grad_output):
        self.total += time.perf_counter() - self._t_bwd

    def reset(self):
        self.total = 0.0


# ── cases ────────────────────────────────────────────────────────────────────────────────────
class Case:
    """One thing to time: a set of zero-argument callables, one per stage."""

    def __init__(self, section, name, kind, seed, calls, backend="default.qubit",
                 n_qubits="", n_layers="", batch="", n_params="", max_reps=None,
                 quantum_timer=None):
        self.section, self.name, self.kind, self.seed, self.calls = section, name, kind, seed, calls
        self.meta = dict(backend=backend, n_qubits=n_qubits, n_layers=n_layers, batch=batch,
                         n_params=n_params)
        self.max_reps = max_reps
        self.reps = 0
        self.quantum_timer = quantum_timer


def net_case(section, name, kind, seed, net, batch, stages, n_qubits="", n_layers=""):
    zt = make_batch(seed, batch)
    calls = {s: (lambda f=STAGE_FNS[s]: f(net, zt)) for s in stages}
    timer = QuantumTimer(net.quantum) if kind == "hybrid" else None
    return Case(section, name, kind, seed, calls, n_qubits=n_qubits, n_layers=n_layers,
                batch=batch, n_params=n_params(net), quantum_timer=timer)


def circuit_case(section, name, seed, backend, n_qubits, n_layers, batch, max_reps=None):
    torch.manual_seed(seed)
    dev = qml.device(backend, wires=n_qubits)
    qnode = qml.QNode(lambda x, p: circuit(x, p, n_qubits, n_layers), dev,
                      interface="torch", diff_method=None)
    params = torch.randn(n_layers, n_qubits * 4, dtype=torch.float32)
    x = torch.rand(batch, n_qubits, dtype=torch.float32)
    return Case(section, name, "circuit", seed, {"circuit_forward": lambda: qnode(x, params)},
                backend=backend, n_qubits=n_qubits, n_layers=n_layers, batch=batch,
                n_params=params.numel(), max_reps=max_reps)


def sec_main(a, only=None, section="main"):
    cases = []
    for seed in a.seeds:
        def add(name, kind, net, nq="", nl=""):
            if only is None or name in only:
                cases.append(net_case(section, name, kind, seed, net, a.batch, STAGES, nq, nl))
        add("classical", "classical", make_classical(seed))
        for name, (nq, nl) in HYBRIDS.items():
            add(name, "hybrid", make_hybrid(nq, nl, seed), nq, nl)
        for nq in CONTROL_QUBITS:
            add(f"ctrl_n{nq}", "control", make_control(nq, seed), nq, 0)
    return cases


def sec_drift(a):
    return sec_main(a, only=DRIFT_CASES, section="drift_check")


def sec_layer(a):
    return [net_case("layer_sweep", f"L{nl}", "hybrid", seed, make_hybrid(4, nl, seed),
                     a.batch, STAGES, 4, nl)
            for seed in a.seeds for nl in a.layers]


def sec_batch(a):
    return [net_case("batch_sweep", f"b{b}", "hybrid", seed, make_hybrid(4, 2, seed),
                     b, STAGES, 4, 2)
            for seed in a.seeds for b in a.batches]


def sec_qubit(a):
    cases = []
    for seed in a.seeds:
        for nq in a.qubits:
            stages = STAGES if nq <= a.max_qubits_grad else ["forward"]
            cases.append(net_case("qubit_sweep", f"q{nq}", "hybrid", seed,
                                  make_hybrid(nq, 2, seed), a.batch, stages, nq, 2))
    return cases


def lightning_available():
    try:
        qml.device("lightning.qubit", wires=2)
        return True
    except Exception:
        return False


def sec_backend(a):
    backends = ["default.qubit"]
    if a.skip_lightning:
        print("  (--skip-lightning: only default.qubit)")
    elif lightning_available():
        backends.append("lightning.qubit")
    else:
        print("  WARNING: lightning.qubit not available (pip install pennylane-lightning); "
              "backend section runs default.qubit only.")
    cases = []
    for seed in a.seeds:
        for be in backends:
            cap = a.lightning_max_reps if be == "lightning.qubit" else None
            for name, (nq, nl) in HYBRIDS.items():
                cases.append(circuit_case("backend", f"{name}|{be}", seed, be, nq, nl, a.batch, cap))
            for b in a.batches:
                cases.append(circuit_case("backend", f"batch{b}|{be}", seed, be, 4, 2, b, cap))
    return cases


BUILDERS = {"main": sec_main, "layer_sweep": sec_layer, "batch_sweep": sec_batch,
            "qubit_sweep": sec_qubit, "backend": sec_backend, "drift_check": sec_drift}


# ── runner ───────────────────────────────────────────────────────────────────────────────────
def peak_rss_mb():
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 1e6 if sys.platform == "darwin" else r / 1e3  # macOS: bytes, Linux: KB
    except Exception:
        return None


def settle(seconds, a):
    """Keep the CPU busy on the drift-check cases so clocks and scheduling reach steady state before
    anything is timed (otherwise the end of the run can come out faster than its start)."""
    if seconds <= 0:
        return
    print(f"\n=== settle: {seconds:.0f}s of untimed work ===", flush=True)
    cases = sec_main(a, only=DRIFT_CASES, section="settle")
    t_end = time.time() + seconds
    while time.time() < t_end:
        for c in cases:
            for call in c.calls.values():
                call()


def run_section(section, cases, a, raw_path):
    print(f"\n=== {section}: {len(cases)} cases ===", flush=True)
    t_start = time.time()
    for c in cases:  # warm-up, then pick the number of repetitions from the cost of one round
        round_t = 0.0
        for call in c.calls.values():
            for _ in range(a.warmup):
                t0 = time.perf_counter()
                call()
                dt = time.perf_counter() - t0
            round_t += dt
        cap = min(a.max_reps, c.max_reps) if c.max_reps else a.max_reps
        c.reps = int(np.clip(a.point_budget // max(round_t, 1e-9), a.min_reps, cap))
    print(f"  warm-up done ({time.time() - t_start:.0f}s); reps per case: "
          f"{min(c.reps for c in cases)}-{max(c.reps for c in cases)}", flush=True)

    rng = random.Random(0)
    n_rounds = max(c.reps for c in cases)
    with open(raw_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        w.writeheader()
        for r in range(n_rounds):
            active = [c for c in cases if r < c.reps]
            rng.shuffle(active)
            for c in active:
                gc.collect()
                for stage, call in c.calls.items():
                    if c.quantum_timer is not None:
                        c.quantum_timer.reset()
                    t0 = time.perf_counter()
                    call()
                    dt = time.perf_counter() - t0
                    w.writerow(dict(section=section, case=c.name, kind=c.kind, seed=c.seed,
                                    stage=stage, rep=r, seconds=f"{dt:.9f}", **c.meta))
                    if c.quantum_timer is not None:
                        q = c.quantum_timer.total
                        w.writerow(dict(section=section, case=c.name, kind=c.kind, seed=c.seed,
                                        stage=f"{stage}_quantum", rep=r, seconds=f"{q:.9f}", **c.meta))
                        w.writerow(dict(section=section, case=c.name, kind=c.kind, seed=c.seed,
                                        stage=f"{stage}_classical", rep=r,
                                        seconds=f"{max(dt - q, 0.0):.9f}", **c.meta))
            f.flush()
            print(f"  round {r + 1}/{n_rounds}  elapsed {time.time() - t_start:.0f}s", flush=True)
    return time.time() - t_start


# ── environment ──────────────────────────────────────────────────────────────────────────────
def _sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def env_info(a):
    info = dict(
        timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
        platform=platform.platform(), machine=platform.machine(), python=sys.version.split()[0],
        torch=torch.__version__, pennylane=qml.__version__, numpy=np.__version__,
        torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
        cpu_count_logical=os.cpu_count(), env_OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS"),
    )
    try:
        info["pennylane_lightning"] = importlib.metadata.version("pennylane-lightning")
    except Exception:
        info["pennylane_lightning"] = None
    if sys.platform == "darwin":
        info["cpu"] = _sh("sysctl -n machdep.cpu.brand_string")
        info["cpu_physical"] = _sh("sysctl -n hw.physicalcpu")
        mem = _sh("sysctl -n hw.memsize")
        info["ram_gb"] = round(int(mem) / 2**30, 1) if mem else None
        info["power"] = _sh("pmset -g batt | head -1")
        info["low_power_mode"] = _sh("pmset -g | grep -i lowpowermode")
    elif sys.platform.startswith("linux"):
        info["cpu"] = _sh("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2")
        info["cpu_physical"] = _sh("lscpu -p=core | grep -v '#' | sort -u | wc -l")
        mem = _sh("grep MemTotal /proc/meminfo | awk '{print $2}'")
        info["ram_gb"] = round(int(mem) / 2**20, 1) if mem else None
        info["cpu_governor"] = _sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    else:
        info["cpu"] = platform.processor()
    info["gpu_used"] = False  # qcpinn_seawater.DEVICE is cpu
    here = os.path.dirname(os.path.abspath(__file__))
    info["git_commit"] = _sh(f"git -C '{here}' rev-parse HEAD")
    info["git_dirty"] = bool(_sh(f"git -C '{here}' status --porcelain"))
    info["args"] = {k: v for k, v in vars(a).items()}
    return info


def param_report():
    """Parameter counts of every config, and a check that controls carry the classical share."""
    counts = {"classical": n_params(make_classical(0))}
    for name, (nq, nl) in HYBRIDS.items():
        net = make_hybrid(nq, nl, 0)
        counts[name] = n_params(net)
        counts[f"{name}_quantum_params"] = n_params(net.quantum)
    for nq in CONTROL_QUBITS:
        counts[f"ctrl_n{nq}"] = n_params(make_control(nq, 0))
    for name, ctrl in CONTROL_OF.items():
        assert counts[ctrl] == counts[name] - counts[f"{name}_quantum_params"], \
            f"control {ctrl} does not match the classical share of {name}"
    return counts


# ── summary and report ───────────────────────────────────────────────────────────────────────
def load_raw(out_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(out_dir, "raw", "*.csv"))):
        with open(path, newline="") as f:
            rows += list(csv.DictReader(f))
    return rows


def aggregate(rows):
    per_seed = defaultdict(lambda: defaultdict(list))
    meta = {}
    for r in rows:
        key = (r["section"], r["case"], r["stage"])
        per_seed[key][int(r["seed"])].append(float(r["seconds"]))
        meta[key] = r
    stats = {}
    for key, seeds in per_seed.items():
        means = [np.mean(v) for v in seeds.values()]
        allv = np.concatenate([v for v in seeds.values()])
        cvs = [np.std(v, ddof=1) / np.mean(v) for v in seeds.values() if len(v) > 1]
        stats[key] = dict(
            mean=float(np.mean(means) * 1e3),
            std=float(np.std(means, ddof=1) * 1e3) if len(means) > 1 else float("nan"),
            median=float(np.median(allv) * 1e3),
            min=float(np.min(allv) * 1e3),
            cv=float(np.mean(cvs) * 100) if cvs else float("nan"),
            n_seeds=len(means), reps=min(len(v) for v in seeds.values()), meta=meta[key])
    return stats


def write_summary_csv(stats, path):
    fields = ["section", "case", "kind", "backend", "n_qubits", "n_layers", "batch", "n_params",
              "stage", "n_seeds", "reps_per_seed", "mean_ms", "std_across_seeds_ms", "median_ms",
              "cv_within_seed_pct"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (sec, case, stage), s in sorted(stats.items()):
            m = s["meta"]
            w.writerow(dict(section=sec, case=case, kind=m["kind"], backend=m["backend"],
                            n_qubits=m["n_qubits"], n_layers=m["n_layers"], batch=m["batch"],
                            n_params=m["n_params"], stage=stage, n_seeds=s["n_seeds"],
                            reps_per_seed=s["reps"], mean_ms=f"{s['mean']:.6f}",
                            std_across_seeds_ms=f"{s['std']:.6f}", median_ms=f"{s['median']:.6f}",
                            cv_within_seed_pct=f"{s['cv']:.3f}"))


def _fmt(s):
    if s is None:
        return "-"
    d = 1 if s["mean"] >= 100 else 2
    return f"{s['mean']:.{d}f}" if np.isnan(s["std"]) else f"{s['mean']:.{d}f} ± {s['std']:.{d}f}"


def _md(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def _fit(xs, ys, mode):
    if len(xs) < 3:
        return None
    xs, ys = np.array(xs, float), np.array(ys, float)
    if mode == "exp2":      # multiplicative factor per unit x
        return 2 ** np.polyfit(xs, np.log2(ys), 1)[0]
    if mode == "loglog":    # exponent of a power law
        return np.polyfit(np.log(xs), np.log(ys), 1)[0]
    return np.polyfit(xs, ys, 1)[0]  # linear slope


def build_report(out_dir, stats, env, counts):
    S = lambda sec, case, stage: stats.get((sec, case, stage))
    L = []
    L.append("# Timing benchmark report\n")
    L.append(f"Run folder: `{out_dir}`. All times in ms; `mean ± std` = mean over seeds of the per-seed "
             "mean, and sample std (ddof=1) across seeds. Ratios use the means.\n")
    L.append("## Environment\n")
    keys = ["timestamp", "platform", "cpu", "cpu_physical", "cpu_count_logical", "ram_gb", "python",
            "torch", "pennylane", "pennylane_lightning", "torch_threads", "power", "git_commit",
            "git_dirty"]
    L.append(_md(["item", "value"], [[k, env.get(k)] for k in keys if env.get(k) is not None]))
    if counts:
        L.append("## Parameter counts\n")
        L.append(_md(["config", "total", "quantum"],
                     [[c, counts[c], counts.get(f"{c}_quantum_params", 0)]
                      for c in ["classical", *HYBRIDS, *[f"ctrl_n{n}" for n in CONTROL_QUBITS]]]))

    if any(k[0] == "main" for k in stats):
        L.append("## Main table (stages, batch = default)\n")
        order = ["classical", *HYBRIDS, *[f"ctrl_n{n}" for n in CONTROL_QUBITS]]
        rows = []
        for c in order:
            f, g1, g2, gp = (S("main", c, s) for s in STAGES)
            if f is None:
                continue
            rows.append([c, f["meta"]["n_params"], _fmt(f), _fmt(g1), _fmt(g2), _fmt(gp),
                         f"{g2['mean'] / f['mean']:.2f}", f"{gp['mean'] / f['mean']:.2f}"])
        L.append(_md(["config", "params", "forward", "first_grad", "second_grad", "param_grad",
                      "second/forward", "param/forward"], rows))
        L.append("### Slowdown\n")
        rows = []
        for c in HYBRIDS:
            r1, r2 = [], []
            for stage in STAGES:
                h, cl, ct = S("main", c, stage), S("main", "classical", stage), S("main", CONTROL_OF[c], stage)
                r1.append(f"{h['mean'] / cl['mean']:.1f}x" if h and cl else "-")
                r2.append(f"{h['mean'] / ct['mean']:.1f}x" if h and ct else "-")
            rows.append([c, *r1, *r2])
        L.append(_md(["config", *[f"{s} vs classical" for s in STAGES],
                      *[f"{s} vs matched control" for s in STAGES]], rows))
        L.append("Matched control = same classical layers, VQC replaced by tanh "
                 "(narrow -> ctrl_n2, baseline and deep -> ctrl_n4, wide -> ctrl_n6).\n")

        if any(k[2].endswith("_quantum") for k in stats if k[0] == "main"):
            L.append("### Quantum vs. classical component time\n")
            L.append("Time inside the QuantumLayer submodule (forward + backward, via module "
                     "hooks) vs. the rest of the hybrid net (pre/postprocessor MLP + autograd "
                     "bookkeeping), for the same calls as the main table above.\n")
            rows = []
            for c in HYBRIDS:
                cells = []
                for stage in STAGES:
                    tot, q = S("main", c, stage), S("main", c, f"{stage}_quantum")
                    if tot and q:
                        cells.append(f"{_fmt(q)} ({q['mean'] / tot['mean'] * 100:.0f}%)")
                    else:
                        cells.append("-")
                rows.append([c, *cells])
            L.append(_md(["config", *[f"{s} quantum (% of total)" for s in STAGES]], rows))

    for section, prefix, xlabel, title in [
            ("layer_sweep", "L", "n_layers", "Depth sweep"),
            ("batch_sweep", "b", "batch", "Batch sweep"),
            ("qubit_sweep", "q", "n_qubits", "Qubit sweep")]:
        if any(k[0] == section and k[2].endswith("_quantum") for k in stats):
            xs = sorted({int(k[1][len(prefix):]) for k in stats
                        if k[0] == section and k[2].endswith("_quantum")})
            rows = []
            for x in xs:
                case = f"{prefix}{x}"
                cells = []
                for stage in STAGES:
                    tot, q = S(section, case, stage), S(section, case, f"{stage}_quantum")
                    cells.append(f"{q['mean'] / tot['mean'] * 100:.0f}%" if tot and q else "-")
                rows.append([x, *cells])
            L.append(f"### {title}: quantum share of total time\n")
            L.append(_md([xlabel, *STAGES], rows))

    def sweep(title, section, prefix, xs, xlabel, mode, unit):
        if not any(k[0] == section for k in stats):
            return
        L.append(f"## {title}\n")
        rows = []
        for x in xs:
            cells = [S(section, f"{prefix}{x}", s) for s in STAGES]
            if all(c is None for c in cells):
                continue
            rows.append([x, *[_fmt(c) for c in cells]])
        L.append(_md([xlabel, *STAGES], rows))
        fits = []
        for s in STAGES:
            pts = [(x, S(section, f"{prefix}{x}", s)["mean"]) for x in xs if S(section, f"{prefix}{x}", s)]
            v = _fit([p[0] for p in pts], [p[1] for p in pts], mode)
            fits.append("-" if v is None else f"{v:.2f}{unit}")
        L.append(f"Fit per stage ({', '.join(STAGES)}): {', '.join(fits)}\n")

    xs_q = sorted({int(k[1][1:]) for k in stats if k[0] == "qubit_sweep"})
    sweep("Qubit sweep (2 layers)", "qubit_sweep", "q", xs_q, "n_qubits", "exp2",
          "x per added qubit (log2-linear fit)")
    xs_l = sorted({int(k[1][1:]) for k in stats if k[0] == "layer_sweep"})
    sweep("Depth sweep (4 qubits)", "layer_sweep", "L", xs_l, "n_layers", "linear",
          " ms per added layer (linear fit)")
    xs_b = sorted({int(k[1][1:]) for k in stats if k[0] == "batch_sweep"})
    sweep("Batch sweep (baseline)", "batch_sweep", "b", xs_b, "batch", "loglog",
          " = exponent of time ~ batch^k (log-log fit)")

    if any(k[0] == "backend" for k in stats):
        L.append("## Backend comparison (circuit-only forward)\n")
        be = ["default.qubit", "lightning.qubit"]
        rows = []
        for name in [*HYBRIDS, *[f"batch{b}" for b in xs_b]]:
            d, l = (S("backend", f"{name}|{b}", "circuit_forward") for b in be)
            if d:
                rows.append([name, _fmt(d), _fmt(l), f"{l['mean'] / d['mean']:.1f}x" if l else "-"])
        L.append(_md(["case", "default.qubit", "lightning.qubit", "lightning / default"], rows))
        for b in be:
            pts = [(x, S("backend", f"batch{x}|{b}", "circuit_forward")) for x in xs_b]
            pts = [(x, s["mean"]) for x, s in pts if s]
            v = _fit([p[0] for p in pts], [p[1] for p in pts], "loglog")
            if v is not None:
                L.append(f"- batch exponent on {b}: {v:.2f} (1.0 = linear in batch, 0 = flat)")
        L.append("")

    if any(k[0] == "drift_check" for k in stats):
        L.append("## Drift check (end of run vs. `main`)\n")
        rows, worst, worst_fast = [], None, None
        for c in DRIFT_CASES:
            cells = []
            for s in STAGES:
                a_, b_ = S("main", c, s), S("drift_check", c, s)
                if a_ and b_:
                    r = b_["median"] / a_["median"]
                    fast = a_["median"] < DRIFT_MIN_MS
                    cand = (abs(r - 1), f"{c} `{s}` {r:.3f}")
                    if fast:
                        worst_fast = max(worst_fast or cand, cand)
                    else:
                        worst = max(worst or cand, cand)
                    cells.append(f"{r:.3f} / {b_['mean'] / a_['mean']:.3f} / {b_['min'] / a_['min']:.3f}"
                                 + (" (fast)" if fast else ""))
                else:
                    cells.append("-")
            rows.append([c, *cells])
        L.append(_md(["config", *[f"{s} (end / main)" for s in STAGES]], rows))
        L.append("Each cell: median / mean / min ratio. Ratios near 1.000 mean the machine state was stable. "
                 f"The verdict uses the median ratio of cells taking at least {DRIFT_MIN_MS:g} ms in `main`; "
                 f"cells marked (fast) are faster than that, dominated by timing jitter, and shown for "
                 f"information only.\n")
        if worst is None:
            L.append("No drift-check cell is slow enough for a verdict.\n")
        elif worst[0] > DRIFT_TOL:
            L.append(f"**Largest deviation is above {DRIFT_TOL:.0%} ({worst[1]}): the run drifted; "
                     "repeat it on an idle machine.**\n")
        else:
            L.append(f"Largest deviation: {worst[0] * 100:.1f}% ({worst[1]}), within {DRIFT_TOL:.0%}.\n")
        if worst_fast is not None:
            L.append(f"Largest deviation among (fast) cells (not used for the verdict): "
                     f"{worst_fast[0] * 100:.1f}% ({worst_fast[1]}).\n")
    return "\n".join(L)


def summarize(out_dir):
    rows = load_raw(out_dir)
    if not rows:
        raise SystemExit(f"No raw data under {out_dir}/raw")
    stats = aggregate(rows)
    write_summary_csv(stats, os.path.join(out_dir, "summary.csv"))
    env, counts = {}, None
    if os.path.exists(os.path.join(out_dir, "env.json")):
        with open(os.path.join(out_dir, "env.json")) as f:
            j = json.load(f)
        env, counts = j.get("env", {}), j.get("param_counts")
    report = build_report(out_dir, stats, env, counts)
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write(report)
    print(f"\nWrote {out_dir}/summary.csv and {out_dir}/report.md")
    return report


# ── main ─────────────────────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default="results/timing_benchmark", help="parent folder of the run folder")
    ap.add_argument("--tag", default=None, help="run folder name (default: timestamp)")
    ap.add_argument("--resume", action="store_true", help="continue a run, skipping finished sections")
    ap.add_argument("--summarize-only", action="store_true", help="rebuild summary.csv/report.md and exit")
    ap.add_argument("--sections", nargs="+", default=SECTIONS, choices=SECTIONS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--batch", type=int, default=2540, help="default batch size (= training batch)")
    ap.add_argument("--warmup", type=int, default=3, help="warm-up calls per stage")
    ap.add_argument("--settle", type=float, default=120.0,
                    help="seconds of untimed work before the first timed section (CPU warm-up)")
    ap.add_argument("--min-reps", type=int, default=5)
    ap.add_argument("--max-reps", type=int, default=20)
    ap.add_argument("--point-budget", type=float, default=20.0,
                    help="target seconds of timed work per case; sets reps between min and max")
    ap.add_argument("--lightning-max-reps", type=int, default=10)
    ap.add_argument("--threads", type=int, default=None, help="torch.set_num_threads (default: torch's)")
    ap.add_argument("--qubits", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--batches", type=int, nargs="+", default=[100, 500, 1000, 2540, 5000])
    ap.add_argument("--max-qubits-grad", type=int, default=8,
                    help="largest qubit count timed with the gradient stages; above it: forward only "
                         "(RAM grows as 2^n x batch x gates, and twice over for second order)")
    ap.add_argument("--skip-lightning", action="store_true")
    ap.add_argument("--quick", action="store_true", help="tiny grid for a smoke test")
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.warmup, a.min_reps, a.max_reps, a.point_budget, a.settle = [0], 1, 1, 2, 1.0, 2.0
        a.qubits, a.layers, a.batches, a.max_qubits_grad, a.lightning_max_reps = [2, 4], [1, 2], [100, 500], 4, 2
        a.batch = 500
    return a


def main():
    a = parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    tag = a.tag or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(a.out_root, tag)
    if a.summarize_only:
        print(summarize(out_dir))
        return
    prog_path = os.path.join(out_dir, "progress.json")
    if os.path.exists(prog_path) and not a.resume:
        raise SystemExit(f"{out_dir} already exists. Use --resume to continue it, or choose another --tag.")
    os.makedirs(os.path.join(out_dir, "raw"), exist_ok=True)
    progress = json.load(open(prog_path)) if os.path.exists(prog_path) else {}

    counts = param_report()
    env = env_info(a)
    env_path = os.path.join(out_dir, "env.json" if not a.resume or not os.path.exists(
        os.path.join(out_dir, "env.json")) else f"env_resume_{datetime.datetime.now():%Y%m%d_%H%M%S}.json")
    with open(env_path, "w") as f:
        json.dump(dict(env=env, param_counts=counts), f, indent=2)
    print(json.dumps(env, indent=2, default=str))
    print("parameter counts:", counts)

    t_all = time.time()
    if any(s in a.sections and s not in progress for s in SECTIONS):
        settle(a.settle, a)
    for section in SECTIONS:  # fixed order regardless of the order given on the command line
        if section not in a.sections:
            continue
        if section in progress:
            print(f"\n=== {section}: already finished, skipping ===")
            continue
        cases = BUILDERS[section](a)
        secs = run_section(section, cases, a, os.path.join(out_dir, "raw", f"{section}.csv"))
        progress[section] = dict(seconds=round(secs, 1), peak_rss_mb=peak_rss_mb())
        with open(prog_path, "w") as f:
            json.dump(progress, f, indent=2)
        print(f"  {section} finished in {secs / 60:.1f} min, peak RSS so far {progress[section]['peak_rss_mb']} MB")
    print(f"\nAll sections finished in {(time.time() - t_all) / 60:.1f} min")
    print(summarize(out_dir))


if __name__ == "__main__":
    main()
