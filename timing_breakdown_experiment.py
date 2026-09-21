"""
Fine-grained wall-clock timing breakdown for classical vs quantum PINN.

Runs, for a single chosen boundary condition (Dirichlet / Neumann / Robin),
all four scenarios (forward + inverse, classical + quantum), instrumented
with per-phase timers so the aggregate wall-clock ratio reported in the main
paper (Table 1) can be decomposed into:

    forward   — building the loss (includes the PDE-residual autograd, i.e.
                the double-backward used to get dT/dt, d2T/dz2)
    backward  — L.backward(), the gradient of the loss w.r.t. parameters
    opt_step  — optimizer.step() only (scheduler.step(), when present, is
                timed separately so it never gets folded into this number)
    circuit   — (quantum only) time spent purely inside the QNode call, a
                subset of `forward` that isolates quantum-circuit simulation
                from the classical pre/post-processor's own forward pass

Two experiment tracks are run for the chosen BC:

    Track 1 — "paper" budget: each branch trains with its own optimizer
              schedule exactly as in qcpinn_seawater.py (classical:
              Adam -> L-BFGS; quantum: Adam-only with cosine decay), so the
              *aggregate* numbers stay comparable to Table 1. Adam-phase and
              L-BFGS-phase timings are reported separately rather than
              blended, since L-BFGS's own parameter update can't be timed
              per-closure-call the way optimizer.step() can for Adam.
    Track 2 — "matched-step" microbenchmark: both branches run exactly
              N_FIXED_STEPS plain Adam steps (no L-BFGS, no LR schedule),
              removing the differing-total-budget confound and giving a
              clean, same-optimizer per-step time comparison.

Usage:
    python timing_breakdown_experiment.py [--bc dirichlet|neumann|robin]
                                           [--seeds N] [--fixed-steps N]

Reuses the physics/model definitions from qcpinn_seawater.py (analytical
solutions, ClassicalDNN, HybridQNN, the network-agnostic PINN class) rather
than redefining them, so the physics is guaranteed identical to the main
experiment pipeline.
"""
import argparse
import csv
import datetime
import os
import time

import numpy as np
import torch

from qcpinn_seawater import PINN, ClassicalDNN, HybridQNN

# ── Config ────────────────────────────────────────────────────────────────────
N_QUBITS, N_QLAYERS = 4, 2     # matches the paper's quantum baseline (n=4, L=2)

# Per-BC training budget (Track 1 only), matching qcpinn_seawater.py's train_cfg
TRAIN_CFG = {
    "dirichlet": dict(adam_steps=5000,  lbfgs_iter=10000, w_pde=1.0),
    "neumann":   dict(adam_steps=5000,  lbfgs_iter=10000, w_pde=1.0),
    "robin":     dict(adam_steps=20000, lbfgs_iter=20000, w_pde=10.0),
}


def _param_groups(model):
    """Same optimizer param-grouping logic as PINN.train() in qcpinn_seawater.py."""
    is_quantum = isinstance(model.net, HybridQNN)
    if is_quantum:
        quantum_params = list(model.net.quantum.parameters()) + [model.net.output_scale]
        classical_params = (list(model.net.preprocessor.parameters())
                             + list(model.net.postprocessor.parameters()))
        groups = [
            {"params": quantum_params, "lr": 5e-3},
            {"params": classical_params, "lr": 1e-3},
        ]
        if model.inverse:
            groups.append({"params": [model.log_kz], "lr": 1e-3})
    else:
        all_params = list(model.net.parameters())
        if model.inverse:
            all_params.append(model.log_kz)
        groups = [{"params": all_params, "lr": 1e-3}]
    return groups, is_quantum


def _wrap_circuit_timer(model, is_quantum):
    """
    Monkeypatch HybridQNN.quantum.forward to accumulate wall-clock time spent
    purely inside the QNode call — a subset of the 'forward' phase, isolating
    quantum-circuit simulation from the classical pre/post-processor's own
    forward pass. Returns (getter, reset) closures; no-ops for classical nets.
    """
    if not is_quantum:
        return (lambda: 0.0), (lambda: None)

    quantum_layer = model.net.quantum
    original_forward = quantum_layer.forward
    acc = {"t": 0.0}

    def timed_forward(x):
        t0 = time.perf_counter()
        out = original_forward(x)
        acc["t"] += time.perf_counter() - t0
        return out

    quantum_layer.forward = timed_forward
    return (lambda: acc["t"]), (lambda: acc.__setitem__("t", 0.0))


def _phase_stats(fwd, bwd, opt, circ):
    """Mean per-call time for one phase (Adam or L-BFGS); None if the phase never ran."""
    n = len(fwd)
    if n == 0:
        return None
    return dict(
        n=n,
        forward_mean=sum(fwd) / n,
        backward_mean=sum(bwd) / n,
        opt_step_mean=(sum(opt) / n) if opt else None,
        circuit_mean=sum(circ) / n,
    )


def train_instrumented(model, adam_steps, lbfgs_iter=0, eta_min=1e-5,
                        fixed_steps=None, log_path=None):
    """
    Fine-grained-timed re-implementation of PINN.train() (qcpinn_seawater.py).

    If `fixed_steps` is given: runs exactly that many plain Adam steps (no
    L-BFGS, no LR schedule) — the matched-step microbenchmark (Track 2).
    Otherwise: reproduces the branch's own paper schedule (classical:
    Adam(adam_steps) -> L-BFGS(lbfgs_iter); quantum: Adam(adam_steps+lbfgs_iter)
    with cosine decay to eta_min) — Track 1, comparable to Table 1.

    Returns:
        {
          "adam":  {n, forward_mean, backward_mean, opt_step_mean, circuit_mean} or None,
          "lbfgs": {n, forward_mean, backward_mean, opt_step_mean=None, circuit_mean} or None,
          "lbfgs_overhead_s": lump sum of L-BFGS's own (non-closure) bookkeeping cost,
          "wall_total": reconstructed total from every measured component,
        }
    Adam and L-BFGS phases are kept separate rather than averaged together:
    L-BFGS's parameter update isn't timeable per-closure-call (a single
    outer iteration can invoke the closure more than once during line
    search), so blending its all-zero opt_step entries into the Adam
    phase's real ones would silently understate the Adam-phase cost.

    If log_path is given, also writes a per-step CSV with columns:
        step, phase, forward_s, backward_s, opt_step_s, sched_s, circuit_s
    """
    param_groups, is_quantum = _param_groups(model)
    circuit_time, reset_circuit = _wrap_circuit_timer(model, is_quantum)

    rows = []
    adam_fwd, adam_bwd, adam_opt, adam_sched, adam_circ = [], [], [], [], []
    lbfgs_fwd, lbfgs_bwd, lbfgs_circ = [], [], []
    lbfgs_overhead = 0.0

    def run_adam(n_steps, optimizer, scheduler=None, phase_label="adam"):
        for step in range(n_steps):
            optimizer.zero_grad()
            reset_circuit()

            t0 = time.perf_counter()
            L = model.loss()
            t1 = time.perf_counter()
            L.backward()
            t2 = time.perf_counter()
            optimizer.step()
            t3 = time.perf_counter()
            if scheduler is not None:
                scheduler.step()
            t4 = time.perf_counter()

            c = circuit_time()
            adam_fwd.append(t1 - t0)
            adam_bwd.append(t2 - t1)
            adam_opt.append(t3 - t2)
            adam_sched.append(t4 - t3)
            adam_circ.append(c)
            rows.append([step, phase_label, t1 - t0, t2 - t1, t3 - t2, t4 - t3, c])

    if fixed_steps is not None:
        optimizer = torch.optim.Adam(param_groups)
        run_adam(fixed_steps, optimizer, phase_label="adam_fixed")

    elif is_quantum:
        total_steps = adam_steps + lbfgs_iter  # matches qcpinn_seawater.py's own budget
        optimizer = torch.optim.Adam(param_groups)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=eta_min)
        run_adam(total_steps, optimizer, scheduler, phase_label="adam")

    else:
        optimizer = torch.optim.Adam(param_groups)
        run_adam(adam_steps, optimizer, phase_label="adam")

        all_params_flat = list(model.net.parameters())
        if model.inverse:
            all_params_flat.append(model.log_kz)
        lbfgs = torch.optim.LBFGS(
            all_params_flat, lr=1.0, max_iter=lbfgs_iter,
            tolerance_grad=1e-9, tolerance_change=1e-11,
            history_size=100, line_search_fn="strong_wolfe",
        )
        step_counter = {"n": 0}

        def closure():
            reset_circuit()
            lbfgs.zero_grad()
            t0 = time.perf_counter()
            L = model.loss()
            t1 = time.perf_counter()
            L.backward()
            t2 = time.perf_counter()
            c = circuit_time()
            lbfgs_fwd.append(t1 - t0)
            lbfgs_bwd.append(t2 - t1)
            lbfgs_circ.append(c)
            rows.append([step_counter["n"], "lbfgs", t1 - t0, t2 - t1, 0.0, 0.0, c])
            step_counter["n"] += 1
            return L

        t_lbfgs0 = time.perf_counter()
        lbfgs.step(closure)
        lbfgs_wall = time.perf_counter() - t_lbfgs0
        lbfgs_fb_sum = sum(lbfgs_fwd) + sum(lbfgs_bwd)
        # Whatever L-BFGS spent internally (quasi-Newton update, line-search
        # control flow) beyond the forward+backward calls we could time
        # directly — reported as one lump sum rather than smeared per-step,
        # since the closure can be invoked more than once per outer iteration.
        lbfgs_overhead = max(0.0, lbfgs_wall - lbfgs_fb_sum)

    if log_path is not None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "phase", "forward_s", "backward_s",
                        "opt_step_s", "sched_s", "circuit_s"])
            w.writerows(rows)

    adam_summary = _phase_stats(adam_fwd, adam_bwd, adam_opt, adam_circ)
    lbfgs_summary = _phase_stats(lbfgs_fwd, lbfgs_bwd, [], lbfgs_circ)

    wall_total = (sum(adam_fwd) + sum(adam_bwd) + sum(adam_opt) + sum(adam_sched)
                  + sum(lbfgs_fwd) + sum(lbfgs_bwd) + lbfgs_overhead)

    return dict(adam=adam_summary, lbfgs=lbfgs_summary,
                lbfgs_overhead_s=lbfgs_overhead, wall_total=wall_total)


def run_one(bc_type, inverse, model_type, seed, cfg, fixed_steps, log_dir):
    np.random.seed(seed)
    torch.manual_seed(seed)

    net = ClassicalDNN() if model_type == "classical" else HybridQNN(
        n_qubits=N_QUBITS, n_qlayers=N_QLAYERS)
    model = PINN(bc_type=bc_type, inverse=inverse, network=net, w_pde=cfg["w_pde"])

    tag = f"{bc_type}_{'inv' if inverse else 'fwd'}_{model_type}_seed{seed}"
    track = "fixed" if fixed_steps is not None else "paper"
    log_path = f"{log_dir}/{track}_{tag}.csv"

    t_wall0 = time.time()
    result = train_instrumented(
        model, adam_steps=cfg["adam_steps"], lbfgs_iter=cfg["lbfgs_iter"],
        fixed_steps=fixed_steps, log_path=log_path,
    )
    # Training loop only — model construction and training-data sampling
    # (inside PINN.__init__, above) already happened before this timer starts.
    wall_clock = time.time() - t_wall0

    res = model.evaluate()
    a, l = result["adam"], result["lbfgs"]
    return dict(
        bc=bc_type, problem="inverse" if inverse else "forward",
        model=model_type, seed=seed, track=track,
        adam_n=a["n"], adam_forward_mean=a["forward_mean"],
        adam_backward_mean=a["backward_mean"], adam_opt_step_mean=a["opt_step_mean"],
        adam_circuit_mean=a["circuit_mean"],
        lbfgs_n=(l["n"] if l else 0),
        lbfgs_forward_mean=(l["forward_mean"] if l else ""),
        lbfgs_backward_mean=(l["backward_mean"] if l else ""),
        lbfgs_circuit_mean=(l["circuit_mean"] if l else ""),
        lbfgs_overhead_s=result["lbfgs_overhead_s"],
        wall_total=result["wall_total"], wall_clock_total=wall_clock,
        mae=res["mae"], l2=res["l2"], kz_err=res["kz_err"] if inverse else "",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bc", choices=["dirichlet", "neumann", "robin"], default="dirichlet")
    ap.add_argument("--seeds", type=int, default=1,
                     help="repeats per (problem, model) combo, default 1 (matches Table 1)")
    ap.add_argument("--fixed-steps", type=int, default=1000,
                     help="Track 2 equal-step microbenchmark size")
    ap.add_argument("--skip-track1", action="store_true",
                     help="skip the full paper-budget runs (Track 1) and only run Track 2")
    ap.add_argument("--skip-track2", action="store_true",
                     help="skip the matched-step microbenchmark (Track 2) and only run Track 1")
    args = ap.parse_args()

    cfg = TRAIN_CFG[args.bc]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"results/training_timing/{args.bc}_{timestamp}"
    log_dir = f"{out_dir}/per_step_logs"
    os.makedirs(log_dir, exist_ok=True)

    print(f"\nTiming breakdown experiment — bc={args.bc}  seeds={args.seeds}")
    print(f"Track 1 (paper budget): adam={cfg['adam_steps']}  lbfgs={cfg['lbfgs_iter']}"
          f" (classical) / adam={cfg['adam_steps'] + cfg['lbfgs_iter']} (quantum, cosine decay)")
    print(f"Track 2 (matched-step): {args.fixed_steps} plain Adam steps, both branches")
    print(f"Output: {out_dir}\n")

    all_rows = []
    combos = [(inv, mt) for inv in (False, True) for mt in ("classical", "quantum")]

    for track_name, fixed_steps in [("paper", None), ("fixed", args.fixed_steps)]:
        if track_name == "paper" and args.skip_track1:
            continue
        if track_name == "fixed" and args.skip_track2:
            continue

        for inverse, model_type in combos:
            label = f"{args.bc}-{'inverse' if inverse else 'forward'}-{model_type}-{track_name}"
            print(f"── {label} ──")
            for seed in range(args.seeds):
                row = run_one(args.bc, inverse, model_type, seed, cfg,
                               fixed_steps, log_dir)
                all_rows.append(row)

                adam_str = (f"adam[fwd={row['adam_forward_mean']*1e3:.2f}ms "
                            f"bwd={row['adam_backward_mean']*1e3:.2f}ms "
                            f"opt={row['adam_opt_step_mean']*1e3:.2f}ms "
                            f"circuit={row['adam_circuit_mean']*1e3:.2f}ms]")
                lbfgs_str = ""
                if row["lbfgs_n"]:
                    lbfgs_str = (f"  lbfgs[fwd={row['lbfgs_forward_mean']*1e3:.2f}ms "
                                 f"bwd={row['lbfgs_backward_mean']*1e3:.2f}ms "
                                 f"circuit={row['lbfgs_circuit_mean']*1e3:.2f}ms "
                                 f"+{row['lbfgs_overhead_s']:.2f}s overhead]")
                print(f"  seed={seed}  MAE={row['mae']:.3e}  {adam_str}{lbfgs_str}  "
                      f"wall={row['wall_clock_total']:.1f}s")

    # ── Save summary CSV ──────────────────────────────────────────────────────
    csv_path = f"{out_dir}/summary_{timestamp}.csv"
    fields = ["bc", "problem", "model", "track", "seed",
              "adam_n", "adam_forward_mean", "adam_backward_mean",
              "adam_opt_step_mean", "adam_circuit_mean",
              "lbfgs_n", "lbfgs_forward_mean", "lbfgs_backward_mean",
              "lbfgs_circuit_mean", "lbfgs_overhead_s",
              "wall_total", "wall_clock_total", "mae", "l2", "kz_err"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\nSummary saved: {csv_path}")
    print(f"Per-step logs saved under: {log_dir}/")

    # ── Print a compact per-phase breakdown table ─────────────────────────────
    print(f"\n{'='*110}")
    print(f"  PER-STEP TIME BREAKDOWN (mean over steps, averaged over {args.seeds} seed(s))")
    print(f"{'='*110}")
    header = (f"{'scenario':<26}{'track':<7}{'phase':<7}{'fwd ms':>9}{'bwd ms':>9}"
              f"{'opt ms':>9}{'circuit ms':>12}{'circuit % fwd':>15}")
    print(header)
    print("-" * len(header))
    for track_name in ["paper", "fixed"]:
        for inverse, model_type in combos:
            rows = [r for r in all_rows if r["track"] == track_name
                     and r["problem"] == ("inverse" if inverse else "forward")
                     and r["model"] == model_type]
            if not rows:
                continue
            scenario = f"{args.bc}-{'inv' if inverse else 'fwd'}-{model_type}"

            fwd = np.mean([r["adam_forward_mean"] for r in rows]) * 1e3
            bwd = np.mean([r["adam_backward_mean"] for r in rows]) * 1e3
            opt = np.mean([r["adam_opt_step_mean"] for r in rows]) * 1e3
            circ = np.mean([r["adam_circuit_mean"] for r in rows]) * 1e3
            circ_pct = (circ / fwd * 100) if fwd > 0 else 0.0
            print(f"{scenario:<26}{track_name:<7}{'adam':<7}{fwd:>9.3f}{bwd:>9.3f}"
                  f"{opt:>9.3f}{circ:>12.3f}{circ_pct:>14.1f}%")

            if rows[0]["lbfgs_n"]:
                lfwd = np.mean([r["lbfgs_forward_mean"] for r in rows]) * 1e3
                lbwd = np.mean([r["lbfgs_backward_mean"] for r in rows]) * 1e3
                lcirc = np.mean([r["lbfgs_circuit_mean"] for r in rows]) * 1e3
                loverhead = np.mean([r["lbfgs_overhead_s"] for r in rows])
                lcirc_pct = (lcirc / lfwd * 100) if lfwd > 0 else 0.0
                print(f"{'':<26}{track_name:<7}{'lbfgs':<7}{lfwd:>9.3f}{lbwd:>9.3f}"
                      f"{'—':>9}{lcirc:>12.3f}{lcirc_pct:>14.1f}%"
                      f"   (+{loverhead:.2f}s/run unseparable L-BFGS overhead)")


if __name__ == "__main__":
    main()
