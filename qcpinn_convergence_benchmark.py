"""
Training-convergence benchmark: classical PINN vs. hybrid QCPINN loss curves.

Produces the six loss-curve figures used by paper/sections/results_convergence.tex
(Figure fig:loss-curves): one classical-vs-QCPINN training-loss plot per scenario
(3 boundary conditions x forward/inverse), for one hybrid configuration.

Unlike qcpinn_seawater.py's own run_comparison() -- which averages MAE/L2 over
several seeds and writes into a fresh results/accuracy/<timestamp>/ folder with no
record of which code produced it -- this script:
  * trains exactly one seed per scenario/model (a loss curve is a single trace, not
    an average; the MAE/L2 comparison is a separate, out-of-scope concern),
  * records the same provenance qcpinn_timing_benchmark.py does (git commit, dirty
    tree, package versions) in env.json, so a given set of figures can be tied back
    to the exact code that produced them,
  * writes the final PNGs both into results/accuracy_convergence/<tag>/ (the
    archival, provenance-carrying copy) and directly into paper/figures/ (what the
    paper actually includes), under the filenames results_convergence.tex expects:
    loss_<bc>_<fwd|inv>.png.

Classical and hybrid share the same seed for a given scenario, so they train on the
same collocation points and initial data (matching qcpinn_seawater.run_comparison's
convention) -- only the network differs.

Usage:
    python qcpinn_convergence_benchmark.py                     # baseline config, all 6 scenarios
    python qcpinn_convergence_benchmark.py --config wide
    python qcpinn_convergence_benchmark.py --quick              # smoke test, ~1 min
    python qcpinn_convergence_benchmark.py --scenarios dirichlet_fwd robin_inv
    python qcpinn_convergence_benchmark.py --seed 1 --tag reseed1
    python qcpinn_convergence_benchmark.py --classical-adam-only --tag classical_adam_seed0 --paper-figures-dir ""

Needs qcpinn_seawater.py next to this file (model and PINN definitions).
"""
import argparse
import csv
import datetime
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys

import numpy as np
import torch
import pennylane as qml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qcpinn_seawater import PINN, ClassicalDNN, HybridQNN

# Same four configurations as qcpinn_timing_benchmark.py / paper/sections/models.tex.
HYBRIDS = {
    "narrow":   (2, 2),
    "baseline": (4, 2),
    "wide":     (6, 2),
    "deep":     (4, 4),
}

# Same per-scenario training budgets as qcpinn_seawater.run_comparison's train_cfg;
# Robin needs more steps and a higher PDE weight to converge.
SCENARIOS = {
    "dirichlet_fwd": dict(bc="dirichlet", inverse=False, adam_steps=5000,  lbfgs_iter=10000, w_pde=1.0,  label="DE_Dir_FWD"),
    "neumann_fwd":   dict(bc="neumann",   inverse=False, adam_steps=5000,  lbfgs_iter=10000, w_pde=1.0,  label="DE_Neum_FWD"),
    "robin_fwd":     dict(bc="robin",     inverse=False, adam_steps=20000, lbfgs_iter=20000, w_pde=10.0, label="DE_Rob_FWD"),
    "dirichlet_inv": dict(bc="dirichlet", inverse=True,  adam_steps=5000,  lbfgs_iter=10000, w_pde=1.0,  label="DE_Dir_INV"),
    "neumann_inv":   dict(bc="neumann",   inverse=True,  adam_steps=5000,  lbfgs_iter=10000, w_pde=1.0,  label="DE_Neum_INV"),
    "robin_inv":     dict(bc="robin",     inverse=True,  adam_steps=20000, lbfgs_iter=20000, w_pde=10.0, label="DE_Rob_INV"),
}


# ── environment / provenance (same fields as qcpinn_timing_benchmark.py's env_info) ────────────
def _sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def env_info(args, n_qubits, n_qlayers):
    info = dict(
        timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
        platform=platform.platform(), python=sys.version.split()[0],
        torch=torch.__version__, pennylane=qml.__version__, numpy=np.__version__,
        hybrid_config=dict(name=args.config, n_qubits=n_qubits, n_qlayers=n_qlayers),
        seed=args.seed, scenarios=list(args.scenarios),
        train_cfg={k: SCENARIOS[k] for k in args.scenarios},
    )
    try:
        info["pennylane_lightning"] = importlib.metadata.version("pennylane-lightning")
    except Exception:
        info["pennylane_lightning"] = None
    here = os.path.dirname(os.path.abspath(__file__))
    info["git_commit"] = _sh(f'git -C "{here}" rev-parse HEAD')
    status = _sh(f'git -C "{here}" status --porcelain')
    info["git_dirty"] = None if status is None else bool(status)
    info["args"] = vars(args)
    return info


# ── training ─────────────────────────────────────────────────────────────────────────────────
def make_net(model_type, n_qubits, n_qlayers):
    return ClassicalDNN() if model_type == "classical" else HybridQNN(n_qubits=n_qubits, n_qlayers=n_qlayers)


def run_one(scenario_key, model_type, n_qubits, n_qlayers, seed, optimizer="auto"):
    """Seed, build the network + PINN (so training data and init are reproducible from `seed`
    alone), and train. Returns (model, eval_result, elapsed_seconds)."""
    cfg = SCENARIOS[scenario_key]
    np.random.seed(seed)
    torch.manual_seed(seed)
    net = make_net(model_type, n_qubits, n_qlayers)
    model = PINN(bc_type=cfg["bc"], inverse=cfg["inverse"], network=net, w_pde=cfg["w_pde"])
    elapsed = model.train(adam_steps=cfg["adam_steps"], lbfgs_iter=cfg["lbfgs_iter"], optimizer=optimizer)
    res = model.evaluate()
    return model, res, elapsed


# ── output ───────────────────────────────────────────────────────────────────────────────────
def save_history_csv(model, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "phase", "loss", "grad_norm", "circuit_grad_norm"])
        lbfgs_start = model.lbfgs_start
        for i, (loss, gn, cgn) in enumerate(zip(
                model.loss_history, model.grad_norm_history, model.circuit_grad_norm_history)):
            phase = "lbfgs" if (lbfgs_start is not None and i >= lbfgs_start) else "adam"
            w.writerow([i, phase, loss, gn, cgn])


def plot_comparison(classical_history, hybrid_history, title, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(classical_history, label="Classical PINN", alpha=0.8)
    ax.semilogy(hybrid_history, label="QCPINN", alpha=0.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def write_summary_csv(rows, path):
    fields = ["scenario", "model", "final_loss", "mae", "l2", "pde_residual", "kz_err", "time_s"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_report(out_dir, env, rows):
    lines = [
        "# Convergence benchmark report\n",
        f"Run folder: `{out_dir}`. Config: **{env['hybrid_config']['name']}** "
        f"(n_qubits={env['hybrid_config']['n_qubits']}, n_qlayers={env['hybrid_config']['n_qlayers']}), "
        f"seed={env['seed']}.\n",
        f"git commit: `{env['git_commit']}`" + (" (dirty)" if env["git_dirty"] else " (clean)") + "\n",
        "\n| scenario | model | final loss | MAE | L2 | PDE residual | time (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['scenario']} | {r['model']} | {r['final_loss']:.3e} | {r['mae']:.3e} | "
                      f"{r['l2']:.3e} | {r['pde_residual']:.3e} | {r['time_s']:.1f} |")
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ── main ─────────────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", choices=list(HYBRIDS), default="baseline",
                     help="hybrid configuration to compare against classical (default: baseline, "
                          "matching the current paper draft)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS),
                     help="subset of the 6 scenarios to run (default: all 6)")
    ap.add_argument("--tag", default=None, help="output folder name under --out-root (default: timestamp)")
    ap.add_argument("--out-root", default="results/accuracy_convergence")
    ap.add_argument("--paper-figures-dir", default="paper/figures",
                     help="also write the 6 PNGs here, at the filenames results_convergence.tex "
                          "expects (loss_<bc>_<fwd|inv>.png); pass '' to skip")
    ap.add_argument("--classical-adam-only", action="store_true",
                     help="optimiser control: train only the classical PINN, with the hybrids' Adam + cosine "
                          "schedule (adam_steps + lbfgs_iter steps, no L-BFGS); no hybrid, no figures")
    ap.add_argument("--quick", action="store_true",
                     help="tiny step counts for a smoke test (~1 min), not for the paper")
    args = ap.parse_args()

    if args.quick:
        for cfg in SCENARIOS.values():
            cfg["adam_steps"], cfg["lbfgs_iter"] = 20, 20

    n_qubits, n_qlayers = HYBRIDS[args.config]
    tag = args.tag or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_root, tag)
    raw_dir = os.path.join(out_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    if args.paper_figures_dir:
        os.makedirs(args.paper_figures_dir, exist_ok=True)

    env = env_info(args, n_qubits, n_qlayers)
    with open(os.path.join(out_dir, "env.json"), "w") as f:
        json.dump(env, f, indent=2)
    print(f"config={args.config} (n_qubits={n_qubits}, n_qlayers={n_qlayers})  seed={args.seed}")
    print(f"git commit {env['git_commit']}" + (" (DIRTY)" if env["git_dirty"] else " (clean)"))
    print(f"output: {out_dir}\n")

    # (row label, network, optimizer) per scenario
    if args.classical_adam_only:
        models = [("classical_adam", "classical", "adam_cosine")]
    else:
        models = [("classical", "classical", "auto"), ("hybrid", "hybrid", "auto")]

    rows = []
    for key in args.scenarios:
        cfg = SCENARIOS[key]
        print(f"=== {cfg['label']}  (bc={cfg['bc']}, inverse={cfg['inverse']}) ===")

        trained = {}
        for label, model_type, optimizer in models:
            print(f"  -- {label} --")
            model, res, elapsed = run_one(key, model_type, n_qubits, n_qlayers, args.seed, optimizer)
            print(f"     MAE={res['mae']:.3e}  L2={res['l2']:.3e}  t={elapsed:.1f}s")
            trained[label] = model
            save_history_csv(model, os.path.join(raw_dir, f"{cfg['label']}_{label}_history.csv"))
            rows.append(dict(scenario=cfg["label"], model=label,
                              final_loss=model.loss_history[-1], mae=res["mae"], l2=res["l2"],
                              pde_residual=model.mean_pde_residual(),
                              kz_err=res["kz_err"] if res["kz_err"] is not None else float("nan"),
                              time_s=elapsed))

        if args.classical_adam_only:
            print()
            continue
        classical_model, hybrid_model = trained["classical"], trained["hybrid"]
        bc, problem = key.rsplit("_", 1)  # "dirichlet_fwd" -> ("dirichlet", "fwd")
        fname = f"loss_{bc}_{problem}.png"
        title = f"Loss convergence: Classical vs QCPINN ({cfg['label']})"
        plot_comparison(classical_model.loss_history, hybrid_model.loss_history, title,
                         os.path.join(raw_dir, fname))
        if args.paper_figures_dir:
            shutil.copy(os.path.join(raw_dir, fname), os.path.join(args.paper_figures_dir, fname))
            print(f"  wrote {os.path.join(args.paper_figures_dir, fname)}")
        print()

    write_summary_csv(rows, os.path.join(out_dir, "summary.csv"))
    write_report(out_dir, env, rows)
    print(f"Done. Report: {os.path.join(out_dir, 'report.md')}")


if __name__ == "__main__":
    main()
