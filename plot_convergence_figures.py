"""
Loss-curve figures for the convergence section (Figure fig:loss-curves).

One panel per scenario: the classical PINN plus all four hybrid configurations,
read from the per-run raw/*_history.csv files written by qcpinn_convergence_benchmark.py.
The classical curve is taken from the baseline run; it is identical in every run
(same seed, same data, deterministic training). Curves are a rolling median (SMOOTH_WINDOW).

Usage:
    python plot_convergence_figures.py                 # writes paper/figures/loss_<bc>_<fwd|inv>.png
    python plot_convergence_figures.py --out-dir tmp/
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# Adam's loss oscillates by orders of magnitude from step to step, which buries five overlaid curves;
# a centred rolling median over this many evaluations shows the trend (stated in the figure caption).
SMOOTH_WINDOW = 201

RUNS = {
    "narrow":   "results/accuracy_convergence/narrow_seed0",
    "baseline": "results/accuracy_convergence/20260930_092855",
    "wide":     "results/accuracy_convergence/wide_seed0",
    "deep":     "results/accuracy_convergence/deep_seed0",
}
CLASSICAL_RUN = RUNS["baseline"]

SCENARIOS = {  # history-file prefix -> output figure name
    "DE_Dir_FWD":  "loss_dirichlet_fwd.png",
    "DE_Neum_FWD": "loss_neumann_fwd.png",
    "DE_Rob_FWD":  "loss_robin_fwd.png",
    "DE_Dir_INV":  "loss_dirichlet_inv.png",
    "DE_Neum_INV": "loss_neumann_inv.png",
    "DE_Rob_INV":  "loss_robin_inv.png",
}

# Categorical slots 1-4 of the reference palette, in fixed order; classical is the neutral reference.
# Line styles give a second encoding for print and colour-vision deficiency.
STYLE = {
    "classical": dict(color="#3d3d3a", ls="-",  label="Classical"),
    "narrow":    dict(color="#2a78d6", ls="-",  label="Narrow"),
    "baseline":  dict(color="#eb6834", ls="--", label="Baseline"),
    "wide":      dict(color="#1baf7a", ls="-.", label="Wide"),
    "deep":      dict(color="#eda100", ls=":",  label="Deep"),
}


def read_history(run_dir, scenario, model):
    with open(os.path.join(run_dir, "raw", f"{scenario}_{model}_history.csv")) as f:
        rows = list(csv.DictReader(f))
    loss = [float(r["loss"]) for r in rows]
    phases = [r["phase"] for r in rows]
    lbfgs_start = phases.index("lbfgs") if "lbfgs" in phases else None
    return loss, lbfgs_start


def rolling_median(x, window=SMOOTH_WINDOW):
    """Centred rolling median; the window shrinks at the ends so the curve keeps its full length."""
    x = np.asarray(x)
    half = window // 2
    padded = np.pad(x, half, mode="edge")
    return np.median(sliding_window_view(padded, window), axis=1)


def plot_scenario(scenario, out_path):
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    loss, lbfgs_start = read_history(CLASSICAL_RUN, scenario, "classical")
    # Smooth the Adam and L-BFGS phases separately so the window doesn't blur the drop at the switch.
    if lbfgs_start is not None:
        smoothed = np.concatenate([rolling_median(loss[:lbfgs_start]), rolling_median(loss[lbfgs_start:])])
    else:
        smoothed = rolling_median(loss)
    ax.semilogy(smoothed, lw=1.2, **STYLE["classical"])
    if lbfgs_start is not None:
        ax.axvline(lbfgs_start, color="#8a8a85", lw=0.7, ls=(0, (2, 2)))
        ax.text(lbfgs_start, 1.02, " L-BFGS", transform=ax.get_xaxis_transform(),
                fontsize=6, color="#6b6b66", va="bottom", ha="left")
    for cfg, run_dir in RUNS.items():
        loss, _ = read_history(run_dir, scenario, "hybrid")
        ax.semilogy(rolling_median(loss), lw=1.2, **STYLE[cfg])

    ax.set_xlabel("Loss evaluation", fontsize=8)
    ax.set_ylabel("Training loss", fontsize=8)
    ax.tick_params(labelsize=7, color="#8a8a85")
    ax.grid(True, which="major", color="#e4e3dc", lw=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#8a8a85")
    ax.legend(fontsize=6.5, frameon=False, loc="upper right", handlelength=2.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="paper/figures")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for scenario, fname in SCENARIOS.items():
        out_path = os.path.join(args.out_dir, fname)
        plot_scenario(scenario, out_path)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
