# Timing benchmark report

Run folder: `results/timing_benchmark\win_py314_run5_idle`. All times in ms; `mean ± std` = mean over seeds of the per-seed mean, and sample std (ddof=1) across seeds. Ratios use the means.

## Environment

| item | value |
|---|---|
| timestamp | 2026-09-29T03:41:15 |
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
| classical | 4353 | 0.56 ± 0.01 | 0.80 ± 0.04 | 1.67 ± 0.05 | 4.47 ± 0.31 | 3.00 | 8.03 |
| narrow | 2421 | 7.96 ± 0.34 | 11.78 ± 0.05 | 22.48 ± 0.57 | 43.96 ± 0.42 | 2.82 | 5.52 |
| baseline | 2569 | 16.66 ± 0.50 | 32.40 ± 0.45 | 65.65 ± 0.57 | 131.9 ± 1.2 | 3.94 | 7.92 |
| wide | 2717 | 31.99 ± 0.53 | 87.45 ± 1.10 | 163.7 ± 2.0 | 418.0 ± 3.0 | 5.12 | 13.06 |
| deep | 2601 | 26.81 ± 0.59 | 54.51 ± 0.90 | 100.6 ± 1.0 | 213.1 ± 0.8 | 3.75 | 7.95 |
| ctrl_n2 | 2405 | 0.73 ± 0.03 | 1.07 ± 0.01 | 2.00 ± 0.02 | 4.78 ± 0.01 | 2.75 | 6.57 |
| ctrl_n4 | 2537 | 0.73 ± 0.02 | 1.07 ± 0.02 | 1.99 ± 0.06 | 4.82 ± 0.01 | 2.72 | 6.62 |
| ctrl_n6 | 2669 | 0.74 ± 0.02 | 1.11 ± 0.04 | 2.07 ± 0.04 | 4.97 ± 0.05 | 2.80 | 6.73 |

### Slowdown

| config | forward vs classical | first_grad vs classical | second_grad vs classical | param_grad vs classical | forward vs matched control | first_grad vs matched control | second_grad vs matched control | param_grad vs matched control |
|---|---|---|---|---|---|---|---|---|
| narrow | 14.3x | 14.7x | 13.4x | 9.8x | 10.9x | 11.0x | 11.2x | 9.2x |
| baseline | 29.9x | 40.5x | 39.2x | 29.5x | 22.9x | 30.2x | 33.1x | 27.3x |
| wide | 57.4x | 109.3x | 97.9x | 93.4x | 43.3x | 78.8x | 79.1x | 84.1x |
| deep | 48.1x | 68.1x | 60.1x | 47.6x | 36.8x | 50.8x | 50.7x | 44.2x |

Matched control = same classical layers, VQC replaced by tanh (narrow -> ctrl_n2, baseline and deep -> ctrl_n4, wide -> ctrl_n6).

### Quantum vs. classical component time

Time inside the QuantumLayer submodule (forward + backward, via module hooks) vs. the rest of the hybrid net (pre/postprocessor MLP + autograd bookkeeping), for the same calls as the main table above.

| config | forward quantum (% of total) | first_grad quantum (% of total) | second_grad quantum (% of total) | param_grad quantum (% of total) |
|---|---|---|---|---|
| narrow | 7.18 ± 0.32 (90%) | 10.15 ± 0.04 (86%) | 13.99 ± 0.42 (62%) | 18.68 ± 0.18 (42%) |
| baseline | 15.86 ± 0.48 (95%) | 29.72 ± 0.52 (92%) | 42.98 ± 0.38 (65%) | 59.16 ± 0.96 (45%) |
| wide | 31.17 ± 0.51 (97%) | 83.48 ± 1.11 (95%) | 109.2 ± 1.6 (67%) | 173.7 ± 1.1 (42%) |
| deep | 25.99 ± 0.57 (97%) | 51.31 ± 0.86 (94%) | 67.96 ± 1.00 (68%) | 96.55 ± 0.39 (45%) |

### Depth sweep: quantum share of total time

| n_layers | forward | first_grad | second_grad | param_grad |
|---|---|---|---|---|
| 1 | 93% | 87% | 67% | 43% |
| 2 | 95% | 90% | 66% | 46% |
| 4 | 97% | 94% | 68% | 47% |
| 8 | 98% | 95% | 67% | 49% |

### Batch sweep: quantum share of total time

| batch | forward | first_grad | second_grad | param_grad |
|---|---|---|---|---|
| 100 | 89% | 90% | 71% | 51% |
| 500 | 89% | 85% | 69% | 49% |
| 1000 | 95% | 92% | 68% | 46% |
| 2540 | 94% | 92% | 65% | 45% |
| 5000 | 95% | 93% | 65% | 42% |

### Qubit sweep: quantum share of total time

| n_qubits | forward | first_grad | second_grad | param_grad |
|---|---|---|---|---|
| 2 | 90% | 86% | 64% | 44% |
| 4 | 95% | 92% | 66% | 45% |
| 6 | 98% | 94% | 67% | 41% |
| 8 | 99% | 91% | 65% | 37% |
| 10 | 100% | - | - | - |
| 12 | 100% | - | - | - |

## Qubit sweep (2 layers)

| n_qubits | forward | first_grad | second_grad | param_grad |
|---|---|---|---|---|
| 2 | 7.48 ± 0.17 | 11.79 ± 0.20 | 19.71 ± 0.41 | 40.56 ± 0.11 |
| 4 | 15.89 ± 0.11 | 30.06 ± 0.31 | 59.79 ± 0.29 | 123.9 ± 0.6 |
| 6 | 30.73 ± 0.24 | 76.89 ± 0.75 | 149.0 ± 1.0 | 384.0 ± 2.2 |
| 8 | 121.8 ± 2.4 | 334.4 ± 4.4 | 794.5 ± 5.7 | 2314.2 ± 21.3 |
| 10 | 686.6 ± 3.2 | - | - | - |
| 12 | 3222.2 ± 2.3 | - | - | - |

Fit per stage (forward, first_grad, second_grad, param_grad): 1.85x per added qubit (log2-linear fit), 1.73x per added qubit (log2-linear fit), 1.82x per added qubit (log2-linear fit), 1.94x per added qubit (log2-linear fit)

## Depth sweep (4 qubits)

| n_layers | forward | first_grad | second_grad | param_grad |
|---|---|---|---|---|
| 1 | 10.92 ± 0.18 | 21.19 ± 0.32 | 41.24 ± 0.29 | 87.08 ± 1.32 |
| 2 | 15.96 ± 0.27 | 30.64 ± 0.39 | 63.88 ± 1.55 | 130.0 ± 3.4 |
| 4 | 26.00 ± 0.66 | 53.82 ± 1.12 | 95.76 ± 2.54 | 213.6 ± 1.5 |
| 8 | 47.19 ± 0.48 | 99.74 ± 0.47 | 170.6 ± 1.3 | 364.0 ± 1.5 |

Fit per stage (forward, first_grad, second_grad, param_grad): 5.19 ms per added layer (linear fit), 11.32 ms per added layer (linear fit), 18.21 ms per added layer (linear fit), 39.42 ms per added layer (linear fit)

## Batch sweep (baseline)

| batch | forward | first_grad | second_grad | param_grad |
|---|---|---|---|---|
| 100 | 11.84 ± 0.26 | 15.88 ± 0.63 | 25.38 ± 0.53 | 50.73 ± 0.83 |
| 500 | 14.01 ± 0.41 | 22.94 ± 0.54 | 36.69 ± 0.43 | 71.29 ± 1.50 |
| 1000 | 14.54 ± 0.35 | 25.08 ± 0.55 | 45.68 ± 0.42 | 91.18 ± 0.56 |
| 2540 | 16.65 ± 0.35 | 32.05 ± 0.18 | 64.69 ± 0.48 | 131.4 ± 3.0 |
| 5000 | 19.97 ± 0.31 | 45.17 ± 1.08 | 84.37 ± 0.97 | 184.7 ± 1.9 |

Fit per stage (forward, first_grad, second_grad, param_grad): 0.13 = exponent of time ~ batch^k (log-log fit), 0.25 = exponent of time ~ batch^k (log-log fit), 0.31 = exponent of time ~ batch^k (log-log fit), 0.33 = exponent of time ~ batch^k (log-log fit)

## Backend comparison (circuit-only forward)

| case | default.qubit | lightning.qubit | lightning / default |
|---|---|---|---|
| narrow | 6.40 ± 0.14 | 725.5 ± 6.4 | 113.3x |
| baseline | 13.80 ± 0.14 | 1438.6 ± 5.4 | 104.2x |
| wide | 27.45 ± 0.13 | 2104.1 ± 2.7 | 76.6x |
| deep | 22.50 ± 0.52 | 2544.5 ± 5.5 | 113.1x |
| batch100 | 8.02 ± 0.21 | 84.82 ± 1.65 | 10.6x |
| batch500 | 9.86 ± 0.20 | 299.0 ± 1.9 | 30.3x |
| batch1000 | 11.76 ± 0.22 | 558.1 ± 4.1 | 47.5x |
| batch2540 | 13.86 ± 0.18 | 1439.5 ± 7.3 | 103.8x |
| batch5000 | 16.54 ± 0.21 | 2766.3 ± 5.4 | 167.2x |

- batch exponent on default.qubit: 0.18 (1.0 = linear in batch, 0 = flat)
- batch exponent on lightning.qubit: 0.89 (1.0 = linear in batch, 0 = flat)

## Drift check (end of run vs. `main`)

| config | forward (end / main) | first_grad (end / main) | second_grad (end / main) | param_grad (end / main) |
|---|---|---|---|---|
| classical | 0.958 / 0.957 / 1.011 (fast) | 1.028 / 1.055 / 1.025 (fast) | 1.173 / 1.189 / 1.006 (fast) | 0.961 / 0.934 / 0.981 (fast) |
| baseline | 0.996 / 1.012 / 0.999 | 0.979 / 0.981 / 0.930 | 0.960 / 0.956 / 0.976 | 0.956 / 0.955 / 0.952 |
| wide | 1.003 / 0.981 / 1.092 | 0.952 / 0.927 / 0.925 | 0.912 / 0.907 / 0.927 | 0.914 / 0.920 / 0.999 |

Each cell: median / mean / min ratio. Ratios near 1.000 mean the machine state was stable. The verdict uses the median ratio of cells taking at least 5 ms in `main`; cells marked (fast) are faster than that, dominated by timing jitter, and shown for information only.

Largest deviation: 8.8% (wide `second_grad` 0.912), within 10%.

Largest deviation among (fast) cells (not used for the verdict): 17.3% (classical `second_grad` 1.173).
