"""
Post-hoc plotting script for qcpinn_timing_profile.py's 3-seed results.
Mirrors the look-and-feel of ablation_plots.py (tab:* palette, dashed grid,
log y-axis, error bars from std) but combines all four timing-profile
sub-experiments (stage breakdown, backend comparison, qubit sweep, batch
sweep) into a single figure, same spirit as qcpinn_ablation_phase2.py's
combined ablation2_*.png figures.

Reads one results/timing_profile/<tag>/ run folder (same one-folder-per-run
layout qcpinn_timing_profile.py's --tag writes into) and globs CSVs by
filename prefix within it. Not every run folder holds all four sub-experiments
(e.g. a run that only refreshed the stage/backend numbers and the layer sweep),
so by default the newest folder (by name) that has all of them is used;
--dir overrides that.

Usage:
    python plot_timing_profile.py
    python plot_timing_profile.py --dir results/timing_profile/20260916
"""
import argparse
import csv
import glob
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUNS_ROOT = "results/timing_profile"
# Filename prefix of one CSV per panel; a run folder needs all of them to be plottable.
REQUIRED_PREFIXES = ["timing_profile_seed", "timing_profile_backend_seed",
                     "timing_profile_qubit_sweep_byseed", "timing_profile_batch_sweep_byseed"]
RESULTS_DIR = None  # set in main()


def _missing_prefixes(d):
    return [p for p in REQUIRED_PREFIXES if not glob.glob(f"{d}/{p}*.csv")]


def _latest_complete_run_dir():
    dirs = sorted(d for d in glob.glob(f"{RUNS_ROOT}/*") if os.path.isdir(d))
    if not dirs:
        raise FileNotFoundError(f"No run folder found under {RUNS_ROOT}/")
    for d in reversed(dirs):
        if not _missing_prefixes(d):
            return d
    raise FileNotFoundError(f"No folder under {RUNS_ROOT}/ has all of {REQUIRED_PREFIXES}")
COLORS = {"forward": "tab:blue", "first_grad": "tab:orange", "second_grad": "tab:green",
          "default": "tab:blue", "lightning": "tab:red"}


def _latest_group(prefix, n_seeds=3):
    """
    The n_seeds most recently written files matching prefix*.csv -- one CSV
    per --seed run (each has its own timestamp, since the seeds were run as
    separate process invocations), unlike the qubit/batch sweep files where
    all seeds already live inside one CSV.
    """
    files = glob.glob(f"{RESULTS_DIR}/{prefix}*.csv")
    if not files:
        raise FileNotFoundError(f"No files found for prefix {prefix!r}")
    files.sort(key=lambda f: f.rsplit("_", 1)[-1])
    return files[-n_seeds:]


def _latest_file(prefix):
    files = glob.glob(f"{RESULTS_DIR}/{prefix}*.csv")
    if not files:
        raise FileNotFoundError(f"No files found for prefix {prefix!r}")
    return sorted(files, key=lambda f: f.rsplit("_", 1)[-1])[-1]


def load_stage_breakdown():
    files = _latest_group("timing_profile_seed")
    data = defaultdict(lambda: defaultdict(list))
    for f in files:
        with open(f) as fh:
            for row in csv.DictReader(fh):
                cfg = row["config"]
                data[cfg]["forward"].append(float(row["forward_mean"]) * 1e3)
                data[cfg]["first_grad"].append(float(row["first_grad_mean"]) * 1e3)
                data[cfg]["second_grad"].append(float(row["second_grad_mean"]) * 1e3)
    return files, data


def load_backend_comparison():
    files = _latest_group("timing_profile_backend_seed")
    data = defaultdict(lambda: defaultdict(list))
    for f in files:
        with open(f) as fh:
            for row in csv.DictReader(fh):
                cfg = row["config"]
                data[cfg]["default"].append(float(row["default_qubit_mean"]) * 1e3)
                data[cfg]["lightning"].append(float(row["lightning_qubit_mean"]) * 1e3)
    return files, data


def load_qubit_sweep():
    f = _latest_file("timing_profile_qubit_sweep_byseed")
    data = defaultdict(list)
    with open(f) as fh:
        for row in csv.DictReader(fh):
            data[int(row["n_qubits"])].append(float(row["forward_mean"]) * 1e3)
    return f, data


def load_batch_sweep():
    f = _latest_file("timing_profile_batch_sweep_byseed")
    data = defaultdict(lambda: defaultdict(list))
    with open(f) as fh:
        for row in csv.DictReader(fh):
            b = int(row["batch"])
            data[b]["default"].append(float(row["default_qubit_mean"]) * 1e3)
            data[b]["lightning"].append(float(row["lightning_qubit_mean"]) * 1e3)
    return f, data


def panel_stage_breakdown(ax, data):
    order = ["classical", "narrow", "baseline", "wide", "deep"]
    stages = [("forward", "forward"), ("first_grad", "+1st-order grad"),
              ("second_grad", "+2nd-order grad")]
    x = np.arange(len(order))
    width = 0.25
    for i, (key, label) in enumerate(stages):
        means = [np.mean(data[c][key]) for c in order]
        stds = [np.std(data[c][key]) for c in order]
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=3,
               color=COLORS[key], alpha=0.85, label=label,
               error_kw=dict(lw=1, ecolor="dimgray"))
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("time (ms, log scale)")
    ax.set_title("(a) Forward / grad breakdown (mean ± std, 3 seeds)")
    ax.grid(True, which="both", axis="y", ls="--", alpha=0.3)
    ax.legend(fontsize=8)


def panel_backend_comparison(ax, data):
    order = ["narrow", "baseline", "wide", "deep"]
    x = np.arange(len(order))
    width = 0.35
    for i, key in enumerate(["default", "lightning"]):
        means = [np.mean(data[c][key]) for c in order]
        stds = [np.std(data[c][key]) for c in order]
        ax.bar(x + (i - 0.5) * width, means, width, yerr=stds, capsize=3,
               color=COLORS[key], alpha=0.85, label=f"{key}.qubit",
               error_kw=dict(lw=1, ecolor="dimgray"))
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("forward time (ms, log scale)")
    ax.set_title("(b) default.qubit vs. lightning.qubit (mean ± std, 3 seeds)")
    ax.grid(True, which="both", axis="y", ls="--", alpha=0.3)
    ax.legend(fontsize=8)


def panel_qubit_sweep(ax, data):
    nqs = sorted(data.keys())
    means = np.array([np.mean(data[n]) for n in nqs])
    stds = np.array([np.std(data[n]) for n in nqs])
    ax.errorbar(nqs, means, yerr=stds, marker="o", color="tab:blue",
                capsize=3, label="measured (default.qubit)")

    slope, intercept = np.polyfit(nqs, np.log2(means), 1)
    fit = 2 ** (slope * np.array(nqs) + intercept)
    ax.plot(nqs, fit, "--", color="tab:gray",
            label=f"fit: ×{2**slope:.2f} per qubit")

    ax.set_yscale("log")
    ax.set_xlabel("n_qubits")
    ax.set_ylabel("forward time (ms, log scale)")
    ax.set_title("(c) Forward time vs. qubit count (n_layers=2)")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(fontsize=8)


def panel_batch_sweep(ax, data):
    batches = sorted(data.keys())
    for key, label in [("default", "default.qubit"), ("lightning", "lightning.qubit")]:
        means = np.array([np.mean(data[b][key]) for b in batches])
        stds = np.array([np.std(data[b][key]) for b in batches])
        ax.errorbar(batches, means, yerr=stds, marker="o", capsize=3,
                    color=COLORS[key], label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("batch size (log scale)")
    ax.set_ylabel("forward time (ms, log scale)")
    ax.set_title("(d) Forward time vs. batch size (4 qubits, 2 layers)")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(fontsize=8)


def main():
    global RESULTS_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None,
                    help=f"run folder to plot (default: newest folder under {RUNS_ROOT}/ "
                         "that contains all four sub-experiments)")
    args = ap.parse_args()
    RESULTS_DIR = args.dir.rstrip("/") if args.dir else _latest_complete_run_dir()
    missing = _missing_prefixes(RESULTS_DIR)
    if missing:
        raise SystemExit(f"{RESULTS_DIR} is missing CSVs with prefix: {missing}")
    print(f"Reading {RESULTS_DIR}")

    stage_files, stage_data = load_stage_breakdown()
    backend_files, backend_data = load_backend_comparison()
    qubit_file, qubit_data = load_qubit_sweep()
    batch_file, batch_data = load_batch_sweep()

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    panel_stage_breakdown(axes[0, 0], stage_data)
    panel_backend_comparison(axes[0, 1], backend_data)
    panel_qubit_sweep(axes[1, 0], qubit_data)
    panel_batch_sweep(axes[1, 1], batch_data)
    fig.suptitle("QCPINN timing profile — 3-seed summary (Python 3.9 / PennyLane 0.38 / torch 2.8)")
    plt.tight_layout()

    out_path = f"{RESULTS_DIR}/timing_profile_summary.png"
    plt.savefig(out_path, dpi=120)
    plt.close()

    print("Source files:")
    for f in stage_files + backend_files + [qubit_file, batch_file]:
        print(f"  {f}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
