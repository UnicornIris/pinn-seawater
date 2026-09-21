"""
Shared plotting helpers for the QCPINN ablation studies (phase1, phase2, ...).
Kept separate from the ablation runner scripts so both a standalone
post-hoc script (plot_ablation_phase1.py) and a runner that plots itself
right after finishing (qcpinn_ablation_phase2.py) can use the same look.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Cycled in config order; first four match qcpinn_seawater.py's own
# plot_comparison() palette (classical=blue, quantum variants after it).
DEFAULT_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]


def colors_for(configs):
    return {c: DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i, c in enumerate(configs)}


def grouped_bar(ax, configs, colors, means, stds, ylabel, title, log=True):
    x = np.arange(len(configs))
    bars = ax.bar(x, means, yerr=stds, capsize=4,
                   color=[colors[c] for c in configs], alpha=0.85,
                   error_kw=dict(lw=1, ecolor="dimgray"))
    if log:
        ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", axis="y", ls="--", alpha=0.3)
    for b, m in zip(bars, means):
        ax.annotate(f"{m:.2e}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    textcoords="offset points", xytext=(0, 3),
                    ha="center", va="bottom", fontsize=7.5)
    return bars


def plot_accuracy_metrics(out_path, scenarios, configs, data, suptitle):
    """data[scenario][config] needs mae_mean/mae_std, l2_mean/l2_std, pde_mean/pde_std."""
    colors = colors_for(configs)
    fig, axes = plt.subplots(len(scenarios), 3, figsize=(13, 4.2 * len(scenarios)))
    if len(scenarios) == 1:
        axes = axes[None, :]
    metrics = [("mae_mean", "mae_std", "MAE"),
               ("l2_mean", "l2_std", "L2 relative error"),
               ("pde_mean", "pde_std", "PDE residual")]
    for row, scenario in enumerate(scenarios):
        for col, (mkey, skey, label) in enumerate(metrics):
            means = [data[scenario][c][mkey] for c in configs]
            stds = [data[scenario][c][skey] for c in configs]
            grouped_bar(axes[row, col], configs, colors, means, stds, label, scenario)
    fig.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_time(out_path, scenarios, configs, data, suptitle):
    """data[scenario][config] needs time_mean, time_std."""
    colors = colors_for(configs)
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6.5 * len(scenarios), 4.5))
    if len(scenarios) == 1:
        axes = [axes]
    for ax, scenario in zip(axes, scenarios):
        means = [data[scenario][c]["time_mean"] for c in configs]
        stds = [data[scenario][c].get("time_std", 0.0) for c in configs]
        grouped_bar(ax, configs, colors, means, stds, "Training time (s)", scenario)
    fig.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_kz_err(out_path, scenarios, configs, data, suptitle):
    """
    Only plots scenarios where data[scenario][configs[0]]['kz_err_mean'] is not
    NaN. Returns True if a file was written, False if there was nothing to plot.
    """
    inv_scenarios = [s for s in scenarios
                      if not np.isnan(data[s][configs[0]].get("kz_err_mean", np.nan))]
    if not inv_scenarios:
        return False
    colors = colors_for(configs)
    fig, axes = plt.subplots(1, len(inv_scenarios), figsize=(6.5 * len(inv_scenarios), 4.5))
    if len(inv_scenarios) == 1:
        axes = [axes]
    for ax, scenario in zip(axes, inv_scenarios):
        means = [data[scenario][c]["kz_err_mean"] for c in configs]
        stds = [data[scenario][c]["kz_err_std"] for c in configs]
        grouped_bar(ax, configs, colors, means, stds,
                    "kz_err (inverse param. abs. error)", scenario)
    fig.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    return True


def plot_qgrad(out_path, scenarios, configs, data, suptitle):
    """
    Barren-plateau diagnostic: circuit-only gradient norm, first vs. last 10%
    of training, side by side per config. data[scenario][config] needs
    qgrad_initial_mean/std, qgrad_final_mean/std. Configs without a circuit
    (e.g. "classical") should be excluded by the caller.
    """
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6.5 * len(scenarios), 4.5))
    if len(scenarios) == 1:
        axes = [axes]
    width = 0.35
    x = np.arange(len(configs))
    for ax, scenario in zip(axes, scenarios):
        init_means = [data[scenario][c]["qgrad_initial_mean"] for c in configs]
        init_stds  = [data[scenario][c]["qgrad_initial_std"] for c in configs]
        final_means = [data[scenario][c]["qgrad_final_mean"] for c in configs]
        final_stds  = [data[scenario][c]["qgrad_final_std"] for c in configs]
        ax.bar(x - width / 2, init_means, width, yerr=init_stds, capsize=3,
               color="tab:gray", alpha=0.85, label="first 10% of training",
               error_kw=dict(lw=1, ecolor="dimgray"))
        ax.bar(x + width / 2, final_means, width, yerr=final_stds, capsize=3,
               color="tab:red", alpha=0.85, label="last 10% of training",
               error_kw=dict(lw=1, ecolor="dimgray"))
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=20, ha="right")
        ax.set_ylabel("Circuit-only gradient norm")
        ax.set_title(scenario)
        ax.grid(True, which="both", axis="y", ls="--", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
