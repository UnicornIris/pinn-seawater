# Archive

Material from before the paper was re-scoped to timing + parameter count. Nothing here was deleted or edited,
only moved.

## scripts/
`pinn_seawater.py` (classical L-BFGS PINN), `qcpinn_ablation_phase1.py`, `qcpinn_ablation_phase2.py`,
`ablation_plots.py`, `plot_ablation_phase1.py`.
They import `qcpinn_seawater` and still write to the old `qcpinn_results/` / `pinn_results/` paths. To re-run one,
from the repo root: `PYTHONPATH=. python archive/scripts/<script>.py`.

## results/
| Folder | What it is |
|---|---|
| `pinn_classical_lbfgs/` | Output of `pinn_seawater.py` (2026-06-25) |
| `accuracy_20260701_113359`, `_20260702_143150`, `_20260706_110419` | Earlier accuracy runs (the newest one is in `results/accuracy/`) |
| `ablation_phase1_20260724_020639`, `ablation_phase2_20260729_225346` | Quantum-branch ablations (Appendix B candidates) |
| `timing_profile_20260915_superseded` | Single-seed timing runs, Python 3.14 env; re-measured 2026-09-16 |

## logs/
`run_20260706_accuracy.log` — stdout of the 2026-07-06 accuracy run (`accuracy_20260706_110419`).
