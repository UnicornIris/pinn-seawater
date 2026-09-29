# Timing and parameter count: classical PINN vs. hybrid QCPINN

Paper direction: see `paper/outline.md`. Two core metrics: hardware timing profile
(forward / first-order grad / second-order grad) and parameter count.

## Code (repo root)
| File | Role |
|---|---|
| `qcpinn_seawater.py` | Model definitions (`ClassicalDNN`, `HybridQNN`, `PINN`); imported by the others. Running it directly is the 6-scenario accuracy experiment. |
| `qcpinn_timing_benchmark.py` | **Final timing run** (all stages x configs/sweeps/backend + size-matched controls, one process, environment recorded) -> `results/timing_benchmark/<tag>/`. Also splits each hybrid stage's time into quantum-circuit vs. classical-MLP share (module hooks on `HybridQNN.quantum`; `<stage>_quantum`/`<stage>_classical` rows). Needs `qcpinn_seawater.py` and `requirements_benchmark.txt`. |
| `qcpinn_timing_profile.py` | Stage timing, backend comparison, qubit / batch / layer sweeps -> `results/timing_profile/<tag>/` |
| `plot_timing_profile.py` | 2x2 summary figure from the newest `results/timing_profile/<tag>/` |
| `qcpinn_parameter_count.py` | Parameter breakdown -> `results/parameter_count/<date>/` |
| `timing_breakdown_experiment.py` | Per-training-step timing logs -> `results/training_timing/` |

## Data (`results/`)
| Folder | Contents | Paper section |
|---|---|---|
| `timing_benchmark/<tag>/` | Final benchmark output: `report.md`, `summary.csv`, `raw/`, `env.json`. Five independent full runs so far (`win_py314`, `win_py314_run2`, `win_py314_run3`, `win_py314_run4`, `win_py314_run5_idle`); growth-rate fits agree within a few percent across all five, but the drift check has exceeded the 10% threshold in every run (worst so far: `win_py314_run4`, classical `first_grad` at 1.34x), so none should yet be treated as final (see `paper/sections/setup.tex`). Runs 1-4 were taken with no idle preparation. `win_py314_run5_idle` was taken on the High performance power plan, on AC power, with Discord, Overwolf, the proxy client and the browser closed (other background services were not audited); it still failed the drift check (classical `second_grad` at 1.19x), and its hybrid cases ran 5-9% *faster* at the end than in `main`, which points at timing noise on the sub-millisecond classical cases and CPU warm-up rather than background load. | Results II |
| `timing_profile/20260916/` | 3-seed stage / backend / qubit / batch sweeps (Py3.9, PennyLane 0.38); source of `summary_tables.md` | Results II |
| `timing_profile/20260920/` | 3-seed stage / backend + layer sweep (with param-grad stage) | Results II |
| `parameter_count/20260916/` | `parameter_count.csv` | Results I |
| `training_timing/` | Dirichlet per-step timing logs | Setup / Discussion |
| `accuracy/20260712_035925/` | Latest full 6-scenario MAE run | Appendix A |

## Other
- `paper/` — outline, drafted sections, figures, PDF.
- `archive/` — superseded / out-of-scope material; see `archive/README.md`.
