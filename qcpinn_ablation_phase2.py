"""
Phase-2 ablation for the QCPINN quantum branch.

Phase 1 (qcpinn_ablation_phase1.py) ruled out both of its candidate causes for
the quantum branch's underperformance: slow_decay (higher LR floor) was the
*worst* of the three configs in both scenarios, and small_init only bought a
marginal (3-16%) MAE improvement over baseline. Neither closes the 3-14x
accuracy gap or the 37-78x runtime gap to the classical net.

This phase asks a different question: is the circuit itself the bottleneck —
either not expressive enough (too few qubits/layers) or suffering a barren
plateau (gradients vanish as the circuit grows)? We sweep n_qubits/n_qlayers
independently around the current default (4 qubits, 2 layers) and log the
circuit-only gradient norm (self.circuit_grad_norm_history, added to
PINN.train() for this purpose) alongside the usual accuracy metrics:
    - if MAE improves monotonically with more qubits/layers -> expressivity
      was the bottleneck, keep scaling up
    - if MAE gets worse and circuit_grad_norm shrinks as qubits/layers grow ->
      barren plateau signature, scaling up makes things worse
    - if neither -> the bottleneck is elsewhere (data encoding, ansatz choice)

Init (xavier_normal_, i.e. quantum_init_std=None) and LR schedule
(eta_min=1e-5) are held at the phase-1 baseline throughout, so n_qubits/
n_qlayers is the only thing varying between configs.

3 seeds per (scenario, config). Results are appended to CSV as each run
finishes, so a partial run can still be analyzed if interrupted.

Usage:
    python qcpinn_ablation_phase2.py
"""
import collections
import csv
import datetime
import os
import time

import numpy as np
import torch

from qcpinn_seawater import ClassicalDNN, HybridQNN, PINN
from ablation_plots import plot_accuracy_metrics, plot_time, plot_kz_err, plot_qgrad

N_SEEDS = 3

# (bc_type, inverse, name, train_cfg) — same per-scenario budgets as phase 1.
SCENARIOS = [
    ("dirichlet", False, "DE_Dir_FWD",
     dict(adam_steps=5000, lbfgs_iter=10000, w_pde=1.0)),
    ("robin", True, "DE_Rob_INV",
     dict(adam_steps=20000, lbfgs_iter=20000, w_pde=10.0)),
]

# quantum_init_std=None -> xavier_normal_, eta_min=1e-5: both held at the
# phase-1 baseline so only circuit structure varies across configs.
CIRCUIT_CONFIGS = {
    "baseline":    dict(n_qubits=4, n_qlayers=2),   # current qcpinn_seawater.py default
    "narrow":      dict(n_qubits=2, n_qlayers=2),   # fewer qubits, same depth
    "wide":        dict(n_qubits=6, n_qlayers=2),   # more qubits, same depth
    "deep":        dict(n_qubits=4, n_qlayers=4),   # same qubits, more layers
}


def run_one(bc, inv, cfg, model_type, seed, circuit_overrides=None):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if model_type == "classical":
        net = ClassicalDNN()
    else:
        net = HybridQNN(n_qubits=circuit_overrides["n_qubits"],
                         n_qlayers=circuit_overrides["n_qlayers"],
                         quantum_init_std=None)

    model = PINN(bc_type=bc, inverse=inv, network=net, w_pde=cfg["w_pde"])
    t0 = time.time()
    if model_type == "classical":
        model.train(adam_steps=cfg["adam_steps"], lbfgs_iter=cfg["lbfgs_iter"])
    else:
        model.train(adam_steps=cfg["adam_steps"], lbfgs_iter=cfg["lbfgs_iter"],
                     eta_min=1e-5)
    elapsed = time.time() - t0

    res = model.evaluate()
    pde_r = model.mean_pde_residual()

    # Barren-plateau diagnostic: mean circuit-only gradient norm over the
    # first vs. last 10% of training. A shrinking final/initial ratio as
    # n_qubits/n_qlayers grows is the barren-plateau signature.
    cgn = getattr(model, "circuit_grad_norm_history", [])
    if cgn:
        k = max(1, len(cgn) // 10)
        qgrad_initial = float(np.mean(cgn[:k]))
        qgrad_final = float(np.mean(cgn[-k:]))
    else:
        qgrad_initial = qgrad_final = ""

    return dict(mae=res["mae"], l2=res["l2"], pde=pde_r,
                kz_err=(res["kz_err"] if inv else ""), time=elapsed,
                qgrad_initial=qgrad_initial, qgrad_final=qgrad_final)


def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"qcpinn_results/ablation2_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    results_csv = f"{out_dir}/ablation2_results.csv"
    fields = ["scenario", "config", "seed", "mae", "l2", "pde", "kz_err", "time",
              "qgrad_initial", "qgrad_final"]

    print(f"\nQCPINN Phase-2 Ablation (circuit structure)")
    print(f"n_seeds={N_SEEDS}")
    print(f"Scenarios: {[s[2] for s in SCENARIOS]}")
    print(f"Configs: classical (reference) + {list(CIRCUIT_CONFIGS.keys())}")
    print(f"Output: {out_dir}\n")

    with open(results_csv, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    rows = []

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
            print(f"  [classical      ][seed={seed}] MAE={r['mae']:.3e} "
                  f"L2={r['l2']:.3e} PDE={r['pde']:.3e} t={r['time']:.1f}s{kz_str}")

        # Circuit-structure ablation configs.
        for cfg_name, overrides in CIRCUIT_CONFIGS.items():
            for seed in range(N_SEEDS):
                r = run_one(bc, inv, cfg, "quantum", seed, overrides)
                row = dict(scenario=name, config=cfg_name, seed=seed, **r)
                log_row(row)
                kz_str = f" kz_err={r['kz_err']:.3e}" if inv else ""
                qg = f" qgrad(init->final)={r['qgrad_initial']:.2e}->{r['qgrad_final']:.2e}"
                print(f"  [{cfg_name:<15}][seed={seed}] MAE={r['mae']:.3e} "
                      f"L2={r['l2']:.3e} PDE={r['pde']:.3e} t={r['time']:.1f}s{kz_str}{qg}")

    summary_csv, data, all_configs = _write_summary(out_dir, rows)
    print(f"\nDone.\nPer-run results: {results_csv}\nSummary (mean+/-std): {summary_csv}")

    _make_plots(out_dir, SCENARIOS, all_configs, data)


def _write_summary(out_dir, rows):
    """Aggregate mean/std per (scenario, config) and write ablation2_summary.csv."""
    fields_to_agg = ["mae", "l2", "pde", "time", "kz_err", "qgrad_initial", "qgrad_final"]
    grouped = collections.defaultdict(lambda: {k: [] for k in fields_to_agg})
    for r in rows:
        g = grouped[(r["scenario"], r["config"])]
        for k in fields_to_agg:
            if r[k] != "":
                g[k].append(r[k])

    summary_csv = os.path.join(out_dir, "ablation2_summary.csv")
    header = ["scenario", "config", "n"]
    for k in fields_to_agg:
        header += [f"{k}_mean", f"{k}_std"]
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for (scenario, cfg_name), g in grouped.items():
            row = [scenario, cfg_name, len(g["mae"])]
            for k in fields_to_agg:
                row += [np.mean(g[k]) if g[k] else "", np.std(g[k]) if g[k] else ""]
            w.writerow(row)

    # Reshape into data[scenario][config] = {field_mean: float, field_std: float, ...}
    # with NaN for anything that wasn't logged (kz_err on non-inverse scenarios,
    # qgrad_* on the classical reference), matching plot_ablation_phase1.py's convention.
    data = collections.defaultdict(dict)
    for (scenario, cfg_name), g in grouped.items():
        entry = {}
        for k in fields_to_agg:
            entry[f"{k}_mean"] = float(np.mean(g[k])) if g[k] else np.nan
            entry[f"{k}_std"] = float(np.std(g[k])) if g[k] else np.nan
        data[scenario][cfg_name] = entry

    all_configs = ["classical"] + list(CIRCUIT_CONFIGS.keys())
    return summary_csv, data, all_configs


def _make_plots(out_dir, scenarios_cfg, all_configs, data):
    scenarios = [s[2] for s in scenarios_cfg]
    quantum_configs = list(CIRCUIT_CONFIGS.keys())

    out1 = os.path.join(out_dir, "ablation2_accuracy.png")
    plot_accuracy_metrics(out1, scenarios, all_configs, data,
                           "QCPINN Phase-2 ablation (circuit structure) — accuracy metrics "
                           "(log-y, error bars = std, n=3 seeds)")
    print(f"Wrote {out1}")

    out2 = os.path.join(out_dir, "ablation2_time.png")
    plot_time(out2, scenarios, all_configs, data,
              "QCPINN Phase-2 ablation (circuit structure) — wall-clock training time (log-y)")
    print(f"Wrote {out2}")

    out3 = os.path.join(out_dir, "ablation2_kz_err.png")
    if plot_kz_err(out3, scenarios, all_configs, data,
                   "QCPINN Phase-2 ablation (circuit structure) — inverse-problem parameter error (log-y)"):
        print(f"Wrote {out3}")

    out4 = os.path.join(out_dir, "ablation2_qgrad.png")
    plot_qgrad(out4, scenarios, quantum_configs, data,
               "QCPINN Phase-2 ablation — circuit-only gradient norm, first vs. last 10% "
               "of training (barren-plateau diagnostic)")
    print(f"Wrote {out4}")


if __name__ == "__main__":
    main()
