"""
Phase-1 ablation for the QCPINN quantum branch.

Question: does the quantum branch underperform the classical baseline because
of (a) VQC parameter initialization scale, or (b) the cosine LR schedule
decaying to ~0 before the loss has actually flattened (see loss_history CSVs
in qcpinn_results/20260712_035925 — loss was still decreasing at the last
checkpoint in every scenario inspected)? This is a *diagnostic* sweep before
committing to the full n_runs=10 x 6-scenario experiment.

Scenarios: DE_Dir_FWD (fast, ~2.4k s/run) and DE_Rob_INV (slow/hard, ~12k s/run).
Configs (quantum branch only; classical is unchanged and included for reference):
    classical    - ClassicalDNN, unchanged, run for reference/sanity check only
    baseline     - current qcpinn_seawater.py defaults: xavier_normal_ init, eta_min=1e-5
    small_init   - VQC params ~ N(0, 0.1) instead of xavier_normal_
    slow_decay   - cosine anneal floor eta_min=1e-3 instead of 1e-5

3 seeds per (scenario, config). Results are appended to CSV as each run
finishes, so a partial run can still be analyzed if interrupted.

Usage:
    python qcpinn_ablation_phase1.py
"""
import collections
import csv
import datetime
import os
import time

import numpy as np
import torch

from qcpinn_seawater import ClassicalDNN, HybridQNN, PINN

N_SEEDS = 3
N_QUBITS, N_QLAYERS = 4, 2

# (bc_type, inverse, name, train_cfg) — same per-scenario budgets as qcpinn_seawater.py
SCENARIOS = [
    ("dirichlet", False, "DE_Dir_FWD",
     dict(adam_steps=5000, lbfgs_iter=10000, w_pde=1.0)),
    ("robin", True, "DE_Rob_INV",
     dict(adam_steps=20000, lbfgs_iter=20000, w_pde=10.0)),
]

# quantum_init_std=None -> xavier_normal_ (baseline). eta_min=1e-5 is the
# qcpinn_seawater.py default.
QUANTUM_CONFIGS = {
    "baseline":   dict(quantum_init_std=None, eta_min=1e-5),
    "small_init": dict(quantum_init_std=0.1,  eta_min=1e-5),
    "slow_decay": dict(quantum_init_std=None, eta_min=1e-3),
}


def run_one(bc, inv, cfg, model_type, seed, quantum_overrides=None):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if model_type == "classical":
        net = ClassicalDNN()
    else:
        net = HybridQNN(n_qubits=N_QUBITS, n_qlayers=N_QLAYERS,
                         quantum_init_std=quantum_overrides["quantum_init_std"])

    model = PINN(bc_type=bc, inverse=inv, network=net, w_pde=cfg["w_pde"])
    t0 = time.time()
    if model_type == "classical":
        model.train(adam_steps=cfg["adam_steps"], lbfgs_iter=cfg["lbfgs_iter"])
    else:
        model.train(adam_steps=cfg["adam_steps"], lbfgs_iter=cfg["lbfgs_iter"],
                     eta_min=quantum_overrides["eta_min"])
    elapsed = time.time() - t0

    res = model.evaluate()
    pde_r = model.mean_pde_residual()
    return dict(mae=res["mae"], l2=res["l2"], pde=pde_r,
                kz_err=(res["kz_err"] if inv else ""), time=elapsed)


def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"qcpinn_results/ablation_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    results_csv = f"{out_dir}/ablation_results.csv"
    fields = ["scenario", "config", "seed", "mae", "l2", "pde", "kz_err", "time"]

    print(f"\nQCPINN Phase-1 Ablation")
    print(f"n_qubits={N_QUBITS}, n_qlayers={N_QLAYERS}, n_seeds={N_SEEDS}")
    print(f"Scenarios: {[s[2] for s in SCENARIOS]}")
    print(f"Configs: classical (reference) + {list(QUANTUM_CONFIGS.keys())}")
    print(f"Output: {out_dir}\n")

    rows = []
    with open(results_csv, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    def log_row(row):
        rows.append(row)
        with open(results_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)

    for bc, inv, name, cfg in SCENARIOS:
        print(f"\n{'='*70}\n  {name}  (bc={bc}, inverse={inv})\n{'='*70}")

        # Reference classical run (unchanged model, for context only).
        for seed in range(N_SEEDS):
            r = run_one(bc, inv, cfg, "classical", seed)
            row = dict(scenario=name, config="classical", seed=seed, **r)
            log_row(row)
            kz_str = f" kz_err={r['kz_err']:.3e}" if inv else ""
            print(f"  [classical ][seed={seed}] MAE={r['mae']:.3e} "
                  f"L2={r['l2']:.3e} PDE={r['pde']:.3e} t={r['time']:.1f}s{kz_str}")

        # Quantum ablation configs.
        for cfg_name, overrides in QUANTUM_CONFIGS.items():
            for seed in range(N_SEEDS):
                r = run_one(bc, inv, cfg, "quantum", seed, overrides)
                row = dict(scenario=name, config=cfg_name, seed=seed, **r)
                log_row(row)
                kz_str = f" kz_err={r['kz_err']:.3e}" if inv else ""
                print(f"  [{cfg_name:10s}][seed={seed}] MAE={r['mae']:.3e} "
                      f"L2={r['l2']:.3e} PDE={r['pde']:.3e} t={r['time']:.1f}s{kz_str}")

    # ── Aggregate mean/std per (scenario, config) ──────────────────────────
    grouped = collections.defaultdict(
        lambda: dict(mae=[], l2=[], pde=[], time=[], kz_err=[]))
    for r in rows:
        g = grouped[(r["scenario"], r["config"])]
        g["mae"].append(r["mae"]); g["l2"].append(r["l2"])
        g["pde"].append(r["pde"]); g["time"].append(r["time"])
        if r["kz_err"] != "":
            g["kz_err"].append(r["kz_err"])

    summary_csv = f"{out_dir}/ablation_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "config", "n", "mae_mean", "mae_std",
                    "l2_mean", "l2_std", "pde_mean", "pde_std", "time_mean",
                    "kz_err_mean", "kz_err_std"])
        for (scenario, cfg_name), g in grouped.items():
            w.writerow([
                scenario, cfg_name, len(g["mae"]),
                np.mean(g["mae"]), np.std(g["mae"]),
                np.mean(g["l2"]),  np.std(g["l2"]),
                np.mean(g["pde"]), np.std(g["pde"]),
                np.mean(g["time"]),
                np.mean(g["kz_err"]) if g["kz_err"] else "",
                np.std(g["kz_err"]) if g["kz_err"] else "",
            ])

    print(f"\n\nPer-run results: {results_csv}")
    print(f"Summary (mean±std): {summary_csv}")

    print(f"\n{'='*90}")
    print("  SUMMARY (mean ± std over seeds)")
    print(f"{'='*90}")
    print(f"{'Scenario':<14} {'Config':<12} {'MAE':>22} {'L2':>22} {'Time(s)':>10}")
    for (scenario, cfg_name), g in grouped.items():
        print(f"{scenario:<14} {cfg_name:<12} "
              f"{np.mean(g['mae']):.3e}±{np.std(g['mae']):.3e}  "
              f"{np.mean(g['l2']):.3e}±{np.std(g['l2']):.3e}  "
              f"{np.mean(g['time']):>9.1f}")


if __name__ == "__main__":
    main()
