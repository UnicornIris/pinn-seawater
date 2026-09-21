"""
Timing profile for the QCPINN quantum branch.

qcpinn_ablation_phase2.py showed that full-training wall-clock time scales
with circuit size (~2.7x per +2 qubits, ~1.67x per doubling of layers) --
consistent with quantum-statevector-simulation cost. But even the smallest
circuit tested there (narrow: 2 qubits, 2 layers) was already 21-30x slower
than the classical net, and phase 2's per-(scenario, config) granularity
can't say *where* inside a training step that residual overhead comes from.

This script isolates three stages of what PINN._pde_residual() does every
training step (see qcpinn_seawater.py:242-251) on the same batch size (2540
collocation points) used in real training:
  1. forward only:            T = net(zt)
  2. + first-order grad:      dT/dz, dT/dt  (autograd.grad, create_graph=True)
  3. + second-order grad:     d2T/dz2       (a second autograd.grad call
                                             through the create_graph=True
                                             graph -- a Hessian-vector product)

Timing each stage separately, for the classical net and all four phase-2
circuit configs, tells us whether the huge constant overhead is dominated by
raw circuit simulation (forward) or blows up specifically when backprop has
to differentiate through the quantum circuit twice.

A second section repeats the forward-only measurement on PennyLane's
"lightning.qubit" C++ backend instead of the code's default "default.qubit",
to check whether the overhead is a "default.qubit is a naive Python
simulator" artifact fixable by switching backends, or something more
fundamental (statevector simulation cost that no backend swap avoids).
default.qubit vectorizes AngleEmbedding's batch dimension via torch tensor
ops; lightning.qubit's torch interface does not share that vectorized state,
so this also tests whether that batching is actually what makes the current
choice tractable at the training batch size (2540).

A third section sweeps n_qubits over a wider range (2-12, fixed 2 layers) and
fits log2(time) vs n_qubits, to turn "forward time looks exponential in
n_qubits" into an actual measured growth-rate-per-qubit instead of an eyeball
call from 3 points.

A fourth section sweeps batch size (fixed baseline circuit: 4 qubits, 2
layers) for both backends, to check directly whether default.qubit's forward
time stays roughly flat as batch grows (true vectorization) while
lightning.qubit's grows roughly linearly (per-sample looping) -- the
mechanism the backend-comparison section above argues for.

A fifth stage, "param-grad", extends stage 3 by the optimizer's own backward
pass: it builds the PDE-residual loss from the second-order gradient and calls
loss.backward() to reach the network parameters (what training does every
step, minus the boundary/data terms, which use far fewer points). Stages 1-3
only differentiate with respect to the input; this one differentiates the
input-derivatives with respect to the parameters, so it is the closest
single-number proxy for the cost of a training step.

A sixth section sweeps circuit depth (n_layers = 1, 2, 4, 8 at 4 qubits) on
default.qubit and times all four stages (forward, first-grad, second-grad,
param-grad) under the same independent seeds as the other sweeps, since the
qubit and batch sweeps above are forward-only and phase 2 has only two depths.

Usage:
    python qcpinn_timing_profile.py
    python qcpinn_timing_profile.py --only-layer-sweep --tag <tag>
"""
import argparse
import csv
import datetime
import os
import time

import numpy as np
import pennylane as qml
import torch

from qcpinn_seawater import ClassicalDNN, HybridQNN, Z_MIN, Z_MAX, T_MIN, T_MAX

N_REPS = 20
N_WARMUP = 3
BATCH = 2540
SWEEP_SEEDS = [0, 1, 2]  # independent circuit-param/data draws for the qubit and batch sweeps
KZ = 0.1  # diffusivity used in param_grad(); its value does not change the cost
LAYER_SWEEP = [1, 2, 4, 8]  # n_layers at fixed 4 qubits

# Same four circuit configs as qcpinn_ablation_phase2.py, plus classical.
CONFIGS = {
    "classical": None,
    "narrow":    dict(n_qubits=2, n_qlayers=2),
    "baseline":  dict(n_qubits=4, n_qlayers=2),
    "wide":      dict(n_qubits=6, n_qlayers=2),
    "deep":      dict(n_qubits=4, n_qlayers=4),
}


def make_net(cfg):
    if cfg is None:
        return ClassicalDNN()
    return HybridQNN(n_qubits=cfg["n_qubits"], n_qlayers=cfg["n_qlayers"],
                      quantum_init_std=None)


def make_batch(seed):
    rng = np.random.RandomState(seed)
    z = rng.uniform(Z_MIN, Z_MAX, BATCH)
    t = rng.uniform(T_MIN, T_MAX, BATCH)
    zt = np.stack([z, t], axis=1).astype(np.float32)
    return torch.tensor(zt, requires_grad=True)


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
    """PDE-residual loss + backward to the parameters.

    Same graph as second_grad, then the residual is squared-averaged and
    differentiated with respect to net.parameters(). The diffusivity value
    does not affect cost, so a constant is used.
    """
    net.zero_grad(set_to_none=True)
    zt = zt.detach().clone().requires_grad_(True)
    T_pred = net(zt)
    dT = torch.autograd.grad(T_pred, zt, torch.ones_like(T_pred), create_graph=True)[0]
    dT_dz, dT_dt = dT[:, 0:1], dT[:, 1:2]
    d2T_dz2 = torch.autograd.grad(dT_dz, zt, torch.ones_like(dT_dz), create_graph=True)[0][:, 0:1]
    loss = ((dT_dt - KZ * d2T_dz2) ** 2).mean()
    loss.backward()


def time_stage(fn, n_reps):
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times))


def _circuit(x, params, n_qubits, n_layers):
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


def backend_forward_time(device_name, n_qubits, n_layers, n_reps, batch=None, seed=None):
    batch = BATCH if batch is None else batch
    if seed is not None:
        torch.manual_seed(seed)
    dev = qml.device(device_name, wires=n_qubits)
    qnode = qml.QNode(lambda x, p: _circuit(x, p, n_qubits, n_layers),
                       dev, interface="torch", diff_method=None)
    params = torch.randn(n_layers, n_qubits * 4, dtype=torch.float32)
    x = torch.rand(batch, n_qubits, dtype=torch.float32)
    for _ in range(N_WARMUP):
        qnode(x, params)
    return time_stage(lambda: qnode(x, params), n_reps)


def layer_sweep(out_dir, ts):
    """Depth sweep at 4 qubits: all four stages, mean +/- std across SWEEP_SEEDS."""
    stages = [("forward", forward_only), ("first_grad", first_grad),
              ("second_grad", second_grad), ("param_grad", param_grad)]
    print(f"\nLayer sweep (default.qubit, 4 qubits, batch={BATCH}, "
          f"{len(SWEEP_SEEDS)} seeds, seconds):")
    print(f"{'layers':<8}{'forward':>12}{'first-grad':>13}{'second-grad':>14}"
          f"{'param-grad':>13}{'param/fwd':>11}\n" + "-" * 71)
    seed_rows, rows = [], []
    for nl in LAYER_SWEEP:
        per_stage = {name: [] for name, _ in stages}
        for seed in SWEEP_SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            net = make_net(dict(n_qubits=4, n_qlayers=nl))
            zt = make_batch(seed)
            for _ in range(N_WARMUP):
                for _, fn in stages:
                    fn(net, zt)
            row = dict(n_layers=nl, seed=seed)
            for name, fn in stages:
                mean, std = time_stage(lambda: fn(net, zt), N_REPS)
                per_stage[name].append(mean)
                row[f"{name}_mean"], row[f"{name}_std"] = mean, std
            seed_rows.append(row)
        agg = dict(n_layers=nl, n_seeds=len(SWEEP_SEEDS))
        for name, _ in stages:
            agg[f"{name}_mean"] = float(np.mean(per_stage[name]))
            agg[f"{name}_std_across_seeds"] = float(np.std(per_stage[name]))
        agg["param_over_forward_ratio"] = agg["param_grad_mean"] / agg["forward_mean"]
        rows.append(agg)
        print(f"{nl:<8}{agg['forward_mean']:>12.4f}{agg['first_grad_mean']:>13.4f}"
              f"{agg['second_grad_mean']:>14.4f}{agg['param_grad_mean']:>13.4f}"
              f"{agg['param_over_forward_ratio']:>11.1f}")

    # Growth per doubling of depth (forward and param-grad), between adjacent points.
    for name in ("forward", "param_grad"):
        growth = [rows[i + 1][f"{name}_mean"] / rows[i][f"{name}_mean"]
                  for i in range(len(rows) - 1)]
        print(f"  {name}: time ratio per doubling of layers = "
              + ", ".join(f"{g:.2f}x" for g in growth))

    for fname, data in (("timing_profile_layer_sweep", rows),
                        ("timing_profile_layer_sweep_byseed", seed_rows)):
        path = f"{out_dir}/{fname}_{ts}.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
        print(f"Wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0,
                     help="seed for the stage-breakdown and backend-comparison "
                          "sections' net weights / batch / circuit params "
                          "(run this script multiple times with different "
                          "--seed values to get independent-seed repeats "
                          "instead of repeated timing of the same draw)")
    ap.add_argument("--skip-sweeps", action="store_true",
                     help="skip the qubit-count and batch-size sweeps (sections "
                          "3-4), which are the slow part -- use this when only "
                          "refreshing the stage-breakdown/backend-comparison "
                          "tables (sections 1-2) across several --seed values")
    ap.add_argument("--tag", type=str, default=None,
                     help="shared output subfolder name under results/timing_profile/ "
                          "for this run, e.g. results/timing_profile/<tag>/ "
                          "-- pass the same --tag to multiple --seed invocations "
                          "so they land in one folder together (mirrors "
                          "qcpinn_ablation_phase2.py's one-folder-per-run layout); "
                          "default: a fresh timestamp")
    ap.add_argument("--only-layer-sweep", action="store_true",
                     help="run only the layer-depth sweep (section 6) and exit; "
                          "use with --tag to add it to an existing results folder")
    args = ap.parse_args()

    if args.only_layer_sweep:
        tag = args.tag or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f"results/timing_profile/{tag}"
        os.makedirs(out_dir, exist_ok=True)
        layer_sweep(out_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        return

    print(f"\nQCPINN timing profile (batch={BATCH}, n_reps={N_REPS}, seed={args.seed})")
    print(f"{'config':<12}{'forward(s)':>14}{'first-grad(s)':>16}{'second-grad(s)':>17}"
          f"{'2nd/fwd ratio':>16}{'param-grad(s)':>16}{'param/fwd':>11}\n" + "-" * 102)

    rows = []
    for name, cfg in CONFIGS.items():
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        net = make_net(cfg)
        zt = make_batch(args.seed)

        for _ in range(N_WARMUP):
            forward_only(net, zt)
            first_grad(net, zt)
            second_grad(net, zt)
            param_grad(net, zt)

        f_mean, f_std = time_stage(lambda: forward_only(net, zt), N_REPS)
        g1_mean, g1_std = time_stage(lambda: first_grad(net, zt), N_REPS)
        g2_mean, g2_std = time_stage(lambda: second_grad(net, zt), N_REPS)
        gp_mean, gp_std = time_stage(lambda: param_grad(net, zt), N_REPS)
        ratio = g2_mean / f_mean if f_mean > 0 else float("nan")
        pratio = gp_mean / f_mean if f_mean > 0 else float("nan")

        print(f"{name:<12}{f_mean:>14.4f}{g1_mean:>16.4f}{g2_mean:>17.4f}{ratio:>16.1f}"
              f"{gp_mean:>16.4f}{pratio:>11.1f}")
        rows.append(dict(config=name,
                          forward_mean=f_mean, forward_std=f_std,
                          first_grad_mean=g1_mean, first_grad_std=g1_std,
                          second_grad_mean=g2_mean, second_grad_std=g2_std,
                          second_over_forward_ratio=ratio,
                          param_grad_mean=gp_mean, param_grad_std=gp_std,
                          param_over_forward_ratio=pratio))

    tag = args.tag or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"results/timing_profile/{tag}"
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = f"{out_dir}/timing_profile_seed{args.seed}_{ts}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    classical_f = rows[0]["forward_mean"]
    print(f"\nForward-pass slowdown vs classical (isolates raw circuit-simulation cost):")
    for r in rows:
        print(f"  {r['config']:<12} {r['forward_mean']/classical_f:>8.1f}x")

    print(f"\nWrote {out_csv}")

    # ── Backend comparison: default.qubit (used in the code, vectorized over
    # the batch) vs lightning.qubit (C++ backend, no shared batch vectorization
    # in the torch interface) — forward-only, no gradient tracking. ──────────
    print(f"\nBackend comparison (forward-only, no grad, batch={BATCH}):")
    print(f"{'config':<12}{'default.qubit(s)':>18}{'lightning.qubit(s)':>20}{'lightning/default':>20}")
    backend_rows = []
    for name, cfg in CONFIGS.items():
        if cfg is None:
            continue
        dq_mean, dq_std = backend_forward_time("default.qubit", cfg["n_qubits"],
                                                cfg["n_qlayers"], N_REPS, seed=args.seed)
        lq_mean, lq_std = backend_forward_time("lightning.qubit", cfg["n_qubits"],
                                                cfg["n_qlayers"], 5, seed=args.seed)
        ratio = lq_mean / dq_mean if dq_mean > 0 else float("nan")
        print(f"{name:<12}{dq_mean:>18.4f}{lq_mean:>20.4f}{ratio:>19.1f}x")
        backend_rows.append(dict(config=name, n_qubits=cfg["n_qubits"],
                                  n_qlayers=cfg["n_qlayers"],
                                  default_qubit_mean=dq_mean, default_qubit_std=dq_std,
                                  lightning_qubit_mean=lq_mean, lightning_qubit_std=lq_std,
                                  lightning_over_default_ratio=ratio))

    out_csv2 = f"{out_dir}/timing_profile_backend_seed{args.seed}_{ts}.csv"
    with open(out_csv2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(backend_rows[0].keys()))
        w.writeheader()
        w.writerows(backend_rows)
    print(f"\nWrote {out_csv2}")

    if args.skip_sweeps:
        print("\n--skip-sweeps set: skipping qubit-count and batch-size sweeps.")
        return

    # ── Qubit-scaling curve: fit log2(time) vs n_qubits to get an actual
    # growth-rate-per-qubit instead of eyeballing 3 points. Fixed 2 layers,
    # default.qubit, forward-only (no grad), same batch as training. Each
    # n_qubits point is measured under SWEEP_SEEDS independent circuit-
    # param/input draws (not just N_REPS repeated timings of the same draw),
    # so the point estimate carries a seed-to-seed std, not only measurement
    # noise within one draw. ─────────────────────────────────────────────────
    QUBIT_SWEEP = [2, 4, 6, 8, 10, 12]
    print(f"\nQubit-scaling sweep (default.qubit, n_layers=2, forward-only, "
          f"batch={BATCH}, {len(SWEEP_SEEDS)} seeds):")
    print(f"{'n_qubits':<10}{'forward(s)':>14}{'std across seeds':>20}")
    sweep_seed_rows = []
    sweep_rows = []
    for nq in QUBIT_SWEEP:
        seed_means = []
        for seed in SWEEP_SEEDS:
            f_mean, f_std = backend_forward_time("default.qubit", nq, 2, N_REPS, seed=seed)
            seed_means.append(f_mean)
            sweep_seed_rows.append(dict(n_qubits=nq, seed=seed,
                                         forward_mean=f_mean, forward_std=f_std))
        mean_across = float(np.mean(seed_means))
        std_across = float(np.std(seed_means))
        print(f"{nq:<10}{mean_across:>14.4f}{std_across:>20.4f}")
        sweep_rows.append(dict(n_qubits=nq, forward_mean=mean_across,
                                forward_std_across_seeds=std_across,
                                n_seeds=len(SWEEP_SEEDS)))

    log2_t = np.log2([r["forward_mean"] for r in sweep_rows])
    nqs = np.array(QUBIT_SWEEP, dtype=float)
    slope, intercept = np.polyfit(nqs, log2_t, 1)
    print(f"\nFit: log2(time) = {slope:.3f} * n_qubits + {intercept:.3f}")
    print(f"  => each added qubit multiplies forward time by 2^{slope:.3f} "
          f"= {2**slope:.2f}x")
    print(f"  (pure statevector doubling would predict exactly 2x per qubit)")

    out_csv3 = f"{out_dir}/timing_profile_qubit_sweep_{ts}.csv"
    with open(out_csv3, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_qubits", "forward_mean",
                                           "forward_std_across_seeds", "n_seeds"])
        w.writeheader()
        w.writerows(sweep_rows)
    print(f"Wrote {out_csv3}")

    out_csv3b = f"{out_dir}/timing_profile_qubit_sweep_byseed_{ts}.csv"
    with open(out_csv3b, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_qubits", "seed", "forward_mean", "forward_std"])
        w.writeheader()
        w.writerows(sweep_seed_rows)
    print(f"Wrote {out_csv3b}")

    # ── Batch-size sweep: fixed baseline circuit (4 qubits, 2 layers), check
    # whether default.qubit's forward time stays flat as batch grows (true
    # vectorization) while lightning.qubit's grows ~linearly (per-sample
    # looping) -- the mechanism claimed above for why lightning was so slow.
    # Each batch point is measured under SWEEP_SEEDS independent circuit-
    # param/input draws, same rationale as the qubit sweep above. ───────────
    BATCH_SWEEP = [100, 500, 1000, 2540, 5000]
    print(f"\nBatch-size sweep (baseline: 4 qubits, 2 layers, {len(SWEEP_SEEDS)} seeds):")
    print(f"{'batch':<8}{'default.qubit(s)':>18}{'std':>10}"
          f"{'lightning.qubit(s)':>20}{'std':>10}{'lightning/default':>20}")
    batch_seed_rows = []
    batch_rows = []
    for b in BATCH_SWEEP:
        dq_seed_means, lq_seed_means = [], []
        for seed in SWEEP_SEEDS:
            dq_mean, dq_std = backend_forward_time("default.qubit", 4, 2, N_REPS,
                                                     batch=b, seed=seed)
            lq_mean, lq_std = backend_forward_time("lightning.qubit", 4, 2, 5,
                                                     batch=b, seed=seed)
            dq_seed_means.append(dq_mean)
            lq_seed_means.append(lq_mean)
            batch_seed_rows.append(dict(batch=b, seed=seed,
                                         default_qubit_mean=dq_mean, default_qubit_std=dq_std,
                                         lightning_qubit_mean=lq_mean, lightning_qubit_std=lq_std))
        dq_across_mean, dq_across_std = float(np.mean(dq_seed_means)), float(np.std(dq_seed_means))
        lq_across_mean, lq_across_std = float(np.mean(lq_seed_means)), float(np.std(lq_seed_means))
        ratio = lq_across_mean / dq_across_mean if dq_across_mean > 0 else float("nan")
        print(f"{b:<8}{dq_across_mean:>18.4f}{dq_across_std:>10.4f}"
              f"{lq_across_mean:>20.4f}{lq_across_std:>10.4f}{ratio:>19.1f}x")
        batch_rows.append(dict(batch=b,
                                default_qubit_mean=dq_across_mean,
                                default_qubit_std_across_seeds=dq_across_std,
                                lightning_qubit_mean=lq_across_mean,
                                lightning_qubit_std_across_seeds=lq_across_std,
                                lightning_over_default_ratio=ratio,
                                n_seeds=len(SWEEP_SEEDS)))

    dq_100, dq_5000 = batch_rows[0]["default_qubit_mean"], batch_rows[-1]["default_qubit_mean"]
    lq_100, lq_5000 = batch_rows[0]["lightning_qubit_mean"], batch_rows[-1]["lightning_qubit_mean"]
    print(f"\nAs batch grows 100 -> 5000 (50x):")
    print(f"  default.qubit time grows   {dq_5000/dq_100:.1f}x  (flat = vectorized)")
    print(f"  lightning.qubit time grows {lq_5000/lq_100:.1f}x  (linear = per-sample loop)")

    out_csv4 = f"{out_dir}/timing_profile_batch_sweep_{ts}.csv"
    with open(out_csv4, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(batch_rows[0].keys()))
        w.writeheader()
        w.writerows(batch_rows)
    print(f"Wrote {out_csv4}")

    out_csv4b = f"{out_dir}/timing_profile_batch_sweep_byseed_{ts}.csv"
    with open(out_csv4b, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(batch_seed_rows[0].keys()))
        w.writeheader()
        w.writerows(batch_seed_rows)
    print(f"Wrote {out_csv4b}")

    layer_sweep(out_dir, ts)


if __name__ == "__main__":
    main()
