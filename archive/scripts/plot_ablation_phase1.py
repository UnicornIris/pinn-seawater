"""
Plot the Phase-1 quantum-branch ablation results (see qcpinn_ablation_phase1.py
for how ablation_results.csv / ablation_summary.csv are produced).

Produces, alongside the CSVs in the same ablation_<timestamp>/ directory:
    ablation_accuracy.png   - MAE / L2 / PDE residual, grouped bars per config,
                              one column per scenario, log-y with error bars
    ablation_time.png       - wall-clock training time, grouped bars, log-y
    ablation_kz_err.png     - inverse-problem parameter error (DE_Rob_INV only)

Usage:
    python plot_ablation_phase1.py [ablation_dir]
    (defaults to the most recent qcpinn_results/ablation_* directory)
"""
import csv
import glob
import os
import sys
from collections import defaultdict

import numpy as np

from ablation_plots import plot_accuracy_metrics, plot_time, plot_kz_err

CONFIGS = ["classical", "baseline", "small_init", "slow_decay"]


def load_summary(ablation_dir):
    path = os.path.join(ablation_dir, "ablation_summary.csv")
    rows = defaultdict(dict)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows[r["scenario"]][r["config"]] = {
                k: (float(v) if v not in ("", None) else np.nan)
                for k, v in r.items() if k not in ("scenario", "config", "n")
            }

    # ablation_summary.csv has no time_std column; derive it from the
    # per-seed rows in ablation_results.csv.
    times = defaultdict(list)
    with open(os.path.join(ablation_dir, "ablation_results.csv"), newline="") as f:
        for r in csv.DictReader(f):
            times[(r["scenario"], r["config"])].append(float(r["time"]))
    for (scenario, config), vals in times.items():
        rows[scenario][config]["time_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    return rows


def main():
    if len(sys.argv) > 1:
        ablation_dir = sys.argv[1]
    else:
        candidates = sorted(glob.glob("qcpinn_results/ablation_*"))
        if not candidates:
            raise SystemExit("No qcpinn_results/ablation_* directory found.")
        ablation_dir = candidates[-1]

    print(f"Reading {ablation_dir}/ablation_summary.csv")
    data = load_summary(ablation_dir)
    scenarios = list(data.keys())

    out1 = os.path.join(ablation_dir, "ablation_accuracy.png")
    plot_accuracy_metrics(out1, scenarios, CONFIGS, data,
                           "QCPINN Phase-1 ablation — accuracy metrics (log-y, error bars = std, n=3 seeds)")
    print(f"Wrote {out1}")

    out2 = os.path.join(ablation_dir, "ablation_time.png")
    plot_time(out2, scenarios, CONFIGS, data,
              "QCPINN Phase-1 ablation — wall-clock training time (log-y)")
    print(f"Wrote {out2}")

    out3 = os.path.join(ablation_dir, "ablation_kz_err.png")
    wrote = plot_kz_err(out3, scenarios, CONFIGS, data,
                         "QCPINN Phase-1 ablation — inverse-problem parameter error (log-y)")
    if wrote:
        print(f"Wrote {out3}")


if __name__ == "__main__":
    main()
