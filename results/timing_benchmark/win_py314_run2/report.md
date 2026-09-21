# Timing benchmark report

Run folder: `results/timing_benchmark\win_py314_run2`. All times in ms; `mean ± std` = mean over seeds of the per-seed mean, and sample std (ddof=1) across seeds. Ratios use the means.

## Environment

| item | value |
|---|---|
| timestamp | 2026-09-21T08:50:30 |
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
| classical | 4353 | 0.54 ± 0.01 | 0.84 ± 0.02 | 1.67 ± 0.05 | 4.25 ± 0.06 | 3.11 | 7.92 |
| narrow | 2421 | 7.87 ± 0.03 | 11.68 ± 0.10 | 21.48 ± 0.09 | 43.60 ± 0.42 | 2.73 | 5.54 |
| baseline | 2569 | 16.48 ± 0.06 | 30.92 ± 0.36 | 63.87 ± 0.75 | 129.0 ± 0.9 | 3.88 | 7.83 |
| wide | 2717 | 31.49 ± 0.39 | 82.43 ± 0.75 | 156.3 ± 0.3 | 402.9 ± 1.5 | 4.97 | 12.80 |
| deep | 2601 | 27.03 ± 0.15 | 52.37 ± 0.40 | 100.1 ± 1.3 | 211.2 ± 0.7 | 3.70 | 7.82 |
| ctrl_n2 | 2405 | 0.72 ± 0.02 | 1.08 ± 0.01 | 1.98 ± 0.05 | 4.68 ± 0.07 | 2.75 | 6.50 |
| ctrl_n4 | 2537 | 0.73 ± 0.03 | 1.11 ± 0.02 | 2.06 ± 0.03 | 5.14 ± 0.66 | 2.82 | 7.04 |
| ctrl_n6 | 2669 | 0.73 ± 0.02 | 1.09 ± 0.02 | 2.07 ± 0.02 | 4.97 ± 0.08 | 2.84 | 6.82 |

### Slowdown

| config | forward vs classical | first_grad vs classical | second_grad vs classical | param_grad vs classical | forward vs matched control | first_grad vs matched control | second_grad vs matched control | param_grad vs matched control |
|---|---|---|---|---|---|---|---|---|
| narrow | 14.7x | 13.9x | 12.9x | 10.3x | 10.9x | 10.8x | 10.9x | 9.3x |
| baseline | 30.7x | 36.8x | 38.3x | 30.3x | 22.5x | 28.0x | 31.0x | 25.1x |
| wide | 58.6x | 98.0x | 93.7x | 94.7x | 43.2x | 75.3x | 75.5x | 81.1x |
| deep | 50.3x | 62.3x | 60.0x | 49.7x | 37.0x | 47.3x | 48.6x | 41.1x |

Matched control = same classical layers, VQC replaced by tanh (narrow -> ctrl_n2, baseline and deep -> ctrl_n4, wide -> ctrl_n6).

## Qubit sweep (2 layers)

| n_qubits | forward | first_grad | second_grad | param_grad |
|---|---|---|---|---|
| 2 | 7.81 ± 0.03 | 12.07 ± 0.20 | 21.89 ± 0.75 | 43.72 ± 0.29 |
| 4 | 16.46 ± 0.18 | 31.24 ± 0.49 | 63.38 ± 0.79 | 129.2 ± 0.1 |
| 6 | 31.77 ± 0.38 | 80.20 ± 0.69 | 153.9 ± 0.9 | 400.0 ± 1.6 |
| 8 | 133.9 ± 1.5 | 348.2 ± 3.6 | 825.5 ± 10.9 | 2390.4 ± 8.8 |
| 10 | 722.3 ± 3.5 | - | - | - |
| 12 | 3295.5 ± 8.8 | - | - | - |

Fit per stage (forward, first_grad, second_grad, param_grad): 1.85x per added qubit (log2-linear fit), 1.74x per added qubit (log2-linear fit), 1.80x per added qubit (log2-linear fit), 1.93x per added qubit (log2-linear fit)

## Depth sweep (4 qubits)

| n_layers | forward | first_grad | second_grad | param_grad |
|---|---|---|---|---|
| 1 | 11.19 ± 0.28 | 19.89 ± 0.22 | 41.27 ± 0.74 | 89.34 ± 0.75 |
| 2 | 16.49 ± 0.26 | 30.88 ± 0.31 | 62.52 ± 0.35 | 130.0 ± 1.1 |
| 4 | 27.10 ± 0.39 | 51.37 ± 0.73 | 96.04 ± 0.77 | 212.1 ± 0.6 |
| 8 | 48.24 ± 0.40 | 94.87 ± 0.62 | 168.0 ± 0.4 | 364.6 ± 1.2 |

Fit per stage (forward, first_grad, second_grad, param_grad): 5.29 ms per added layer (linear fit), 10.69 ms per added layer (linear fit), 17.92 ms per added layer (linear fit), 39.28 ms per added layer (linear fit)

## Batch sweep (baseline)

| batch | forward | first_grad | second_grad | param_grad |
|---|---|---|---|---|
| 100 | 12.07 ± 0.46 | 16.61 ± 0.31 | 26.62 ± 0.20 | 50.35 ± 0.39 |
| 500 | 13.80 ± 0.45 | 21.88 ± 0.30 | 36.87 ± 0.47 | 71.73 ± 0.57 |
| 1000 | 14.76 ± 0.22 | 24.90 ± 0.27 | 46.97 ± 0.52 | 92.41 ± 0.77 |
| 2540 | 16.66 ± 0.52 | 33.48 ± 1.08 | 64.69 ± 1.20 | 130.6 ± 1.1 |
| 5000 | 19.82 ± 0.17 | 44.15 ± 0.80 | 84.96 ± 0.14 | 190.4 ± 0.6 |

Fit per stage (forward, first_grad, second_grad, param_grad): 0.12 = exponent of time ~ batch^k (log-log fit), 0.24 = exponent of time ~ batch^k (log-log fit), 0.30 = exponent of time ~ batch^k (log-log fit), 0.33 = exponent of time ~ batch^k (log-log fit)

## Backend comparison (circuit-only forward)

| case | default.qubit | lightning.qubit | lightning / default |
|---|---|---|---|
| narrow | 6.38 ± 0.05 | 762.0 ± 0.9 | 119.4x |
| baseline | 14.35 ± 0.35 | 1497.6 ± 11.4 | 104.4x |
| wide | 28.40 ± 0.32 | 2174.9 ± 9.2 | 76.6x |
| deep | 23.14 ± 0.17 | 2608.1 ± 8.5 | 112.7x |
| batch100 | 8.38 ± 0.05 | 88.76 ± 1.34 | 10.6x |
| batch500 | 10.27 ± 0.12 | 309.2 ± 5.0 | 30.1x |
| batch1000 | 12.30 ± 0.18 | 580.1 ± 4.5 | 47.2x |
| batch2540 | 14.26 ± 0.17 | 1489.9 ± 15.6 | 104.4x |
| batch5000 | 17.18 ± 0.45 | 2840.4 ± 8.4 | 165.4x |

- batch exponent on default.qubit: 0.18 (1.0 = linear in batch, 0 = flat)
- batch exponent on lightning.qubit: 0.89 (1.0 = linear in batch, 0 = flat)

## Drift check (end of run vs. `main`)

| config | forward (end / main) | first_grad (end / main) | second_grad (end / main) | param_grad (end / main) |
|---|---|---|---|---|
| classical | 1.013 | 1.065 | 1.124 | 1.031 |
| baseline | 0.997 | 1.071 | 0.978 | 0.997 |
| wide | 0.972 | 0.986 | 0.991 | 0.978 |

Ratios near 1.000 mean the machine state was stable. **Largest deviation is above 10%: the run drifted; repeat it on an idle machine.**
