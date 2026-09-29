# Timing benchmark report

Run folder: `results/timing_benchmark\win_py314_run3`. All times in ms; `mean ± std` = mean over seeds of the per-seed mean, and sample std (ddof=1) across seeds. Ratios use the means.

## Environment

| item | value |
|---|---|
| timestamp | 2026-09-28T03:50:42 |
| platform | Windows-11-10.0.26200-SP0 |
| cpu | Intel64 Family 6 Model 191 Stepping 2, GenuineIntel |
| cpu_count_logical | 16 |
| python | 3.14.3 |
| torch | 2.11.0+cpu |
| pennylane | 0.44.1 |
| pennylane_lightning | 0.44.0 |
| torch_threads | 10 |
| git_dirty | False |

## Parameter counts

| config | total | quantum |
|---|---|---|
| classical | 4353 | 0 |
| narrow | 2421 | 16 |
| baseline | 2569 | 32 |
| wide | 2717 | 48 |
| deep | 2601 | 64 |
| ctrl_n2 | 2405 | 0 |
| ctrl_n4 | 2537 | 0 |
| ctrl_n6 | 2669 | 0 |

## Main table (stages, batch = default)

| config | params | forward | first_grad | second_grad | param_grad | second/forward | param/forward |
|---|---|---|---|---|---|---|---|
| classical | 4353 | 0.55 ± 0.01 | 0.83 ± 0.02 | 1.69 ± 0.02 | 4.46 ± 0.12 | 3.07 | 8.08 |
| narrow | 2421 | 7.98 ± 0.22 | 12.01 ± 0.15 | 23.33 ± 0.49 | 45.19 ± 0.50 | 2.93 | 5.67 |
| baseline | 2569 | 16.62 ± 0.33 | 32.15 ± 0.47 | 66.63 ± 0.65 | 134.2 ± 0.8 | 4.01 | 8.07 |
| wide | 2717 | 31.91 ± 0.25 | 86.32 ± 0.64 | 161.8 ± 1.2 | 422.0 ± 7.5 | 5.07 | 13.23 |
| deep | 2601 | 27.32 ± 0.33 | 54.07 ± 0.58 | 102.7 ± 0.3 | 216.7 ± 0.1 | 3.76 | 7.93 |
| ctrl_n2 | 2405 | 0.73 ± 0.01 | 1.13 ± 0.00 | 2.07 ± 0.07 | 5.11 ± 0.17 | 2.82 | 6.96 |
| ctrl_n4 | 2537 | 0.72 ± 0.02 | 1.12 ± 0.02 | 2.13 ± 0.02 | 5.26 ± 0.12 | 2.94 | 7.25 |
| ctrl_n6 | 2669 | 0.74 ± 0.00 | 1.15 ± 0.02 | 2.15 ± 0.04 | 5.31 ± 0.12 | 2.90 | 7.17 |

### Slowdown

| config | forward vs classical | first_grad vs classical | second_grad vs classical | param_grad vs classical | forward vs matched control | first_grad vs matched control | second_grad vs matched control | param_grad vs matched control |
|---|---|---|---|---|---|---|---|---|
| narrow | 14.4x | 14.5x | 13.8x | 10.1x | 10.9x | 10.6x | 11.3x | 8.8x |
| baseline | 30.1x | 38.7x | 39.3x | 30.1x | 22.9x | 28.8x | 31.3x | 25.5x |
| wide | 57.8x | 104.0x | 95.5x | 94.6x | 43.1x | 74.9x | 75.3x | 79.5x |
| deep | 49.5x | 65.2x | 60.7x | 48.6x | 37.7x | 48.4x | 48.3x | 41.2x |

Matched control = same classical layers, VQC replaced by tanh (narrow -> ctrl_n2, baseline and deep -> ctrl_n4, wide -> ctrl_n6).

### Quantum vs. classical component time

Time inside the QuantumLayer submodule (forward + backward, via module hooks) vs. the rest of the hybrid net (pre/postprocessor MLP + autograd bookkeeping), for the same calls as the main table above.

| config | forward quantum (% of total) | first_grad quantum (% of total) | second_grad quantum (% of total) | param_grad quantum (% of total) |
|---|---|---|---|---|
| narrow | 7.20 ± 0.21 (90%) | 10.29 ± 0.14 (86%) | 14.44 ± 0.32 (62%) | 19.15 ± 0.30 (42%) |
| baseline | 15.84 ± 0.31 (95%) | 29.51 ± 0.63 (92%) | 43.66 ± 0.18 (66%) | 59.29 ± 0.30 (44%) |
| wide | 31.08 ± 0.25 (97%) | 82.02 ± 0.61 (95%) | 108.0 ± 0.6 (67%) | 170.9 ± 2.6 (41%) |
| deep | 26.52 ± 0.33 (97%) | 50.71 ± 0.50 (94%) | 70.38 ± 0.46 (69%) | 97.80 ± 0.33 (45%) |

### Depth sweep: quantum share of total time

| n_layers | forward | first_grad | second_grad | param_grad |
|---|---|---|---|---|
| 1 | 93% | 90% | 65% | 42% |
| 2 | 95% | 86% | 66% | 44% |
| 4 | 97% | 93% | 66% | 47% |
| 8 | 98% | 95% | 68% | 47% |

### Batch sweep: quantum share of total time

| batch | forward | first_grad | second_grad | param_grad |
|---|---|---|---|---|
| 100 | 88% | 90% | 70% | 51% |
| 500 | 90% | 87% | 69% | 48% |
| 1000 | 95% | 91% | 68% | 46% |
| 2540 | 95% | 91% | 65% | 45% |
| 5000 | 94% | 93% | 64% | 42% |

### Qubit sweep: quantum share of total time

| n_qubits | forward | first_grad | second_grad | param_grad |
|---|---|---|---|---|
| 2 | 91% | 86% | 62% | 44% |
| 4 | 95% | 88% | 65% | 45% |
| 6 | 97% | 95% | 68% | 41% |
| 8 | 99% | 88% | 65% | 37% |
| 10 | 100% | - | - | - |
| 12 | 100% | - | - | - |

## Qubit sweep (2 layers)

| n_qubits | forward | first_grad | second_grad | param_grad |
|---|---|---|---|---|
| 2 | 8.60 ± 0.17 | 12.69 ± 0.10 | 23.12 ± 0.61 | 45.66 ± 0.37 |
| 4 | 17.47 ± 0.34 | 35.98 ± 0.29 | 70.08 ± 0.32 | 137.4 ± 1.1 |
| 6 | 33.06 ± 0.67 | 90.47 ± 3.19 | 159.4 ± 0.9 | 418.7 ± 5.5 |
| 8 | 152.4 ± 6.8 | 433.7 ± 20.7 | 931.1 ± 18.3 | 2731.7 ± 32.9 |
| 10 | 760.3 ± 3.8 | - | - | - |
| 12 | 3502.1 ± 51.1 | - | - | - |

Fit per stage (forward, first_grad, second_grad, param_grad): 1.85x per added qubit (log2-linear fit), 1.78x per added qubit (log2-linear fit), 1.81x per added qubit (log2-linear fit), 1.95x per added qubit (log2-linear fit)

## Depth sweep (4 qubits)

| n_layers | forward | first_grad | second_grad | param_grad |
|---|---|---|---|---|
| 1 | 11.38 ± 0.25 | 21.10 ± 0.19 | 44.39 ± 0.71 | 93.68 ± 0.31 |
| 2 | 16.69 ± 0.20 | 33.34 ± 0.55 | 67.79 ± 0.37 | 135.9 ± 0.6 |
| 4 | 27.48 ± 0.19 | 55.29 ± 0.75 | 105.6 ± 1.6 | 223.3 ± 1.4 |
| 8 | 48.37 ± 0.60 | 103.8 ± 2.0 | 183.0 ± 0.8 | 390.4 ± 5.2 |

Fit per stage (forward, first_grad, second_grad, param_grad): 5.29 ms per added layer (linear fit), 11.78 ms per added layer (linear fit), 19.61 ms per added layer (linear fit), 42.41 ms per added layer (linear fit)

## Batch sweep (baseline)

| batch | forward | first_grad | second_grad | param_grad |
|---|---|---|---|---|
| 100 | 12.39 ± 0.47 | 17.23 ± 0.06 | 27.73 ± 0.24 | 53.46 ± 0.53 |
| 500 | 14.22 ± 0.25 | 23.26 ± 0.76 | 38.85 ± 0.83 | 74.91 ± 0.70 |
| 1000 | 14.95 ± 0.38 | 25.99 ± 0.13 | 48.24 ± 0.41 | 98.13 ± 0.55 |
| 2540 | 16.63 ± 0.15 | 34.94 ± 1.18 | 68.08 ± 1.15 | 136.1 ± 0.7 |
| 5000 | 20.67 ± 0.22 | 46.95 ± 0.47 | 87.94 ± 1.01 | 199.2 ± 1.7 |

Fit per stage (forward, first_grad, second_grad, param_grad): 0.12 = exponent of time ~ batch^k (log-log fit), 0.25 = exponent of time ~ batch^k (log-log fit), 0.30 = exponent of time ~ batch^k (log-log fit), 0.33 = exponent of time ~ batch^k (log-log fit)

## Backend comparison (circuit-only forward)

| case | default.qubit | lightning.qubit | lightning / default |
|---|---|---|---|
| narrow | 6.85 ± 0.18 | 806.8 ± 10.4 | 117.8x |
| baseline | 14.78 ± 0.30 | 1580.9 ± 9.4 | 107.0x |
| wide | 29.73 ± 0.08 | 2309.1 ± 17.6 | 77.7x |
| deep | 24.79 ± 0.74 | 2819.8 ± 3.3 | 113.8x |
| batch100 | 8.55 ± 0.11 | 93.60 ± 0.27 | 10.9x |
| batch500 | 10.68 ± 0.22 | 322.5 ± 6.3 | 30.2x |
| batch1000 | 12.79 ± 0.07 | 612.2 ± 6.3 | 47.9x |
| batch2540 | 14.69 ± 0.08 | 1595.5 ± 14.1 | 108.6x |
| batch5000 | 17.91 ± 0.37 | 3048.5 ± 6.7 | 170.3x |

- batch exponent on default.qubit: 0.19 (1.0 = linear in batch, 0 = flat)
- batch exponent on lightning.qubit: 0.90 (1.0 = linear in batch, 0 = flat)

## Drift check (end of run vs. `main`)

| config | forward (end / main) | first_grad (end / main) | second_grad (end / main) | param_grad (end / main) |
|---|---|---|---|---|
| classical | 1.002 / 0.998 / 1.013 (fast) | 1.107 / 1.109 / 1.082 (fast) | 1.002 / 1.004 / 1.004 (fast) | 1.030 / 1.026 / 1.077 (fast) |
| baseline | 1.215 / 1.187 / 1.057 | 1.118 / 1.102 / 1.054 | 1.008 / 1.019 / 1.023 | 1.036 / 1.052 / 0.995 |
| wide | 1.089 / 1.084 / 1.032 | 1.039 / 1.055 / 1.086 | 0.990 / 0.982 / 0.976 | 1.004 / 0.993 / 1.014 |

Each cell: median / mean / min ratio. Ratios near 1.000 mean the machine state was stable. The verdict uses the median ratio of cells taking at least 5 ms in `main`; cells marked (fast) are faster than that, dominated by timing jitter, and shown for information only.

**Largest deviation is above 10% (baseline `forward` 1.215): the run drifted; repeat it on an idle machine.**

Largest deviation among (fast) cells (not used for the verdict): 10.7% (classical `first_grad` 1.107).
