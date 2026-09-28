# Timing benchmark report

Run folder: `results/timing_benchmark\win_py314_run4`. All times in ms; `mean ± std` = mean over seeds of the per-seed mean, and sample std (ddof=1) across seeds. Ratios use the means.

## Environment

| item | value |
|---|---|
| timestamp | 2026-09-28T04:50:37 |
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
| classical | 4353 | 0.54 ± 0.01 | 0.84 ± 0.00 | 1.73 ± 0.01 | 4.67 ± 0.07 | 3.19 | 8.61 |
| narrow | 2421 | 7.89 ± 0.09 | 12.11 ± 0.27 | 21.63 ± 0.40 | 44.70 ± 0.24 | 2.74 | 5.67 |
| baseline | 2569 | 16.60 ± 0.23 | 32.46 ± 0.40 | 63.63 ± 0.68 | 132.6 ± 1.0 | 3.83 | 7.99 |
| wide | 2717 | 32.08 ± 0.22 | 85.02 ± 0.49 | 160.7 ± 0.6 | 415.4 ± 3.8 | 5.01 | 12.95 |
| deep | 2601 | 27.44 ± 0.11 | 55.32 ± 0.51 | 102.2 ± 1.4 | 214.6 ± 2.1 | 3.73 | 7.82 |
| ctrl_n2 | 2405 | 0.72 ± 0.01 | 1.12 ± 0.04 | 2.05 ± 0.01 | 5.05 ± 0.05 | 2.84 | 7.01 |
| ctrl_n4 | 2537 | 0.75 ± 0.03 | 1.11 ± 0.03 | 2.06 ± 0.03 | 4.97 ± 0.14 | 2.74 | 6.61 |
| ctrl_n6 | 2669 | 0.75 ± 0.00 | 1.15 ± 0.01 | 2.09 ± 0.10 | 5.23 ± 0.15 | 2.80 | 6.99 |

### Slowdown

| config | forward vs classical | first_grad vs classical | second_grad vs classical | param_grad vs classical | forward vs matched control | first_grad vs matched control | second_grad vs matched control | param_grad vs matched control |
|---|---|---|---|---|---|---|---|---|
| narrow | 14.5x | 14.3x | 12.5x | 9.6x | 11.0x | 10.8x | 10.6x | 8.9x |
| baseline | 30.6x | 38.4x | 36.8x | 28.4x | 22.1x | 29.2x | 30.9x | 26.7x |
| wide | 59.2x | 100.6x | 92.8x | 89.0x | 42.9x | 74.1x | 76.7x | 79.4x |
| deep | 50.6x | 65.5x | 59.1x | 46.0x | 36.5x | 49.7x | 49.7x | 43.2x |

Matched control = same classical layers, VQC replaced by tanh (narrow -> ctrl_n2, baseline and deep -> ctrl_n4, wide -> ctrl_n6).

### Quantum vs. classical component time

Time inside the QuantumLayer submodule (forward + backward, via module hooks) vs. the rest of the hybrid net (pre/postprocessor MLP + autograd bookkeeping), for the same calls as the main table above.

| config | forward quantum (% of total) | first_grad quantum (% of total) | second_grad quantum (% of total) | param_grad quantum (% of total) |
|---|---|---|---|---|
| narrow | 7.11 ± 0.09 (90%) | 10.40 ± 0.28 (86%) | 13.69 ± 0.07 (63%) | 19.05 ± 0.03 (43%) |
| baseline | 15.81 ± 0.21 (95%) | 29.87 ± 0.33 (92%) | 41.91 ± 0.61 (66%) | 59.21 ± 0.31 (45%) |
| wide | 31.29 ± 0.22 (98%) | 80.87 ± 0.58 (95%) | 107.4 ± 0.9 (67%) | 169.5 ± 1.4 (41%) |
| deep | 26.66 ± 0.10 (97%) | 51.87 ± 0.54 (94%) | 69.96 ± 1.04 (68%) | 97.02 ± 1.07 (45%) |

### Depth sweep: quantum share of total time

| n_layers | forward | first_grad | second_grad | param_grad |
|---|---|---|---|---|
| 1 | 94% | 89% | 64% | 42% |
| 2 | 96% | 88% | 67% | 44% |
| 4 | 97% | 93% | 68% | 46% |
| 8 | 98% | 95% | 68% | 48% |

### Batch sweep: quantum share of total time

| batch | forward | first_grad | second_grad | param_grad |
|---|---|---|---|---|
| 100 | 88% | 90% | 69% | 51% |
| 500 | 90% | 89% | 68% | 49% |
| 1000 | 95% | 91% | 65% | 46% |
| 2540 | 95% | 90% | 65% | 44% |
| 5000 | 94% | 92% | 65% | 42% |

### Qubit sweep: quantum share of total time

| n_qubits | forward | first_grad | second_grad | param_grad |
|---|---|---|---|---|
| 2 | 90% | 85% | 61% | 42% |
| 4 | 95% | 89% | 66% | 44% |
| 6 | 98% | 95% | 68% | 42% |
| 8 | 99% | 91% | 65% | 38% |
| 10 | 100% | - | - | - |
| 12 | 100% | - | - | - |

## Qubit sweep (2 layers)

| n_qubits | forward | first_grad | second_grad | param_grad |
|---|---|---|---|---|
| 2 | 8.09 ± 0.16 | 13.20 ± 0.29 | 22.99 ± 0.44 | 46.67 ± 0.26 |
| 4 | 17.22 ± 0.20 | 34.56 ± 0.73 | 67.09 ± 0.54 | 133.3 ± 1.3 |
| 6 | 33.21 ± 0.42 | 85.39 ± 1.65 | 156.9 ± 2.5 | 417.2 ± 3.2 |
| 8 | 145.3 ± 7.7 | 395.2 ± 16.0 | 898.0 ± 18.0 | 2592.3 ± 30.2 |
| 10 | 744.2 ± 3.4 | - | - | - |
| 12 | 3441.8 ± 47.7 | - | - | - |

Fit per stage (forward, first_grad, second_grad, param_grad): 1.85x per added qubit (log2-linear fit), 1.74x per added qubit (log2-linear fit), 1.81x per added qubit (log2-linear fit), 1.93x per added qubit (log2-linear fit)

## Depth sweep (4 qubits)

| n_layers | forward | first_grad | second_grad | param_grad |
|---|---|---|---|---|
| 1 | 12.91 ± 0.08 | 22.73 ± 0.55 | 44.88 ± 0.27 | 92.27 ± 0.59 |
| 2 | 18.27 ± 0.08 | 34.36 ± 0.43 | 66.00 ± 0.58 | 132.3 ± 0.9 |
| 4 | 28.54 ± 0.19 | 55.60 ± 0.85 | 101.0 ± 0.4 | 219.0 ± 1.3 |
| 8 | 50.28 ± 0.22 | 102.8 ± 0.7 | 176.1 ± 0.5 | 378.3 ± 2.2 |

Fit per stage (forward, first_grad, second_grad, param_grad): 5.33 ms per added layer (linear fit), 11.41 ms per added layer (linear fit), 18.60 ms per added layer (linear fit), 40.95 ms per added layer (linear fit)

## Batch sweep (baseline)

| batch | forward | first_grad | second_grad | param_grad |
|---|---|---|---|---|
| 100 | 11.97 ± 0.53 | 16.92 ± 0.11 | 27.44 ± 0.21 | 52.52 ± 0.70 |
| 500 | 14.05 ± 0.62 | 22.04 ± 0.21 | 38.94 ± 0.35 | 73.25 ± 1.44 |
| 1000 | 14.90 ± 0.10 | 25.45 ± 0.23 | 47.77 ± 1.10 | 95.16 ± 0.18 |
| 2540 | 16.94 ± 0.20 | 33.70 ± 0.85 | 67.65 ± 0.64 | 133.1 ± 0.9 |
| 5000 | 20.31 ± 0.37 | 46.27 ± 1.07 | 86.46 ± 0.33 | 195.1 ± 1.1 |

Fit per stage (forward, first_grad, second_grad, param_grad): 0.13 = exponent of time ~ batch^k (log-log fit), 0.25 = exponent of time ~ batch^k (log-log fit), 0.29 = exponent of time ~ batch^k (log-log fit), 0.33 = exponent of time ~ batch^k (log-log fit)

## Backend comparison (circuit-only forward)

| case | default.qubit | lightning.qubit | lightning / default |
|---|---|---|---|
| narrow | 7.22 ± 0.33 | 799.9 ± 9.4 | 110.7x |
| baseline | 14.72 ± 0.15 | 1584.3 ± 14.9 | 107.6x |
| wide | 29.86 ± 0.32 | 2302.3 ± 7.0 | 77.1x |
| deep | 23.91 ± 0.75 | 2788.6 ± 10.4 | 116.6x |
| batch100 | 8.44 ± 0.09 | 92.26 ± 1.16 | 10.9x |
| batch500 | 10.74 ± 0.16 | 321.3 ± 4.9 | 29.9x |
| batch1000 | 13.08 ± 0.26 | 605.9 ± 9.6 | 46.3x |
| batch2540 | 14.81 ± 0.50 | 1578.2 ± 20.6 | 106.6x |
| batch5000 | 18.21 ± 0.39 | 3039.8 ± 2.2 | 166.9x |

- batch exponent on default.qubit: 0.19 (1.0 = linear in batch, 0 = flat)
- batch exponent on lightning.qubit: 0.90 (1.0 = linear in batch, 0 = flat)

## Drift check (end of run vs. `main`)

| config | forward (end / main) | first_grad (end / main) | second_grad (end / main) | param_grad (end / main) |
|---|---|---|---|---|
| classical | 1.003 | 1.344 | 1.248 | 1.018 |
| baseline | 1.063 | 1.033 | 1.056 | 1.018 |
| wide | 1.043 | 1.023 | 0.993 | 0.991 |

Ratios near 1.000 mean the machine state was stable. **Largest deviation is above 10%: the run drifted; repeat it on an idle machine.**
