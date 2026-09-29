# Timing benchmark report

Run folder: `results/timing_benchmark\win_py314`. All times in ms; `mean ± std` = mean over seeds of the per-seed mean, and sample std (ddof=1) across seeds. Ratios use the means.

## Environment

| item | value |
|---|---|
| timestamp | 2026-09-21T08:26:22 |
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
| classical | 4353 | 0.55 ± 0.01 | 0.83 ± 0.02 | 1.70 ± 0.00 | 4.51 ± 0.03 | 3.12 | 8.26 |
| narrow | 2421 | 7.76 ± 0.02 | 11.78 ± 0.24 | 21.54 ± 0.63 | 42.89 ± 1.13 | 2.78 | 5.53 |
| baseline | 2569 | 16.03 ± 0.02 | 30.77 ± 0.12 | 64.17 ± 0.56 | 128.2 ± 0.3 | 4.00 | 8.00 |
| wide | 2717 | 31.67 ± 0.40 | 83.82 ± 1.76 | 159.9 ± 2.6 | 413.1 ± 5.0 | 5.05 | 13.04 |
| deep | 2601 | 26.96 ± 0.23 | 52.38 ± 0.91 | 99.04 ± 1.58 | 211.4 ± 1.0 | 3.67 | 7.84 |
| ctrl_n2 | 2405 | 0.74 ± 0.03 | 1.17 ± 0.08 | 2.15 ± 0.11 | 5.06 ± 0.24 | 2.90 | 6.84 |
| ctrl_n4 | 2537 | 0.72 ± 0.01 | 1.15 ± 0.12 | 2.13 ± 0.12 | 5.25 ± 0.74 | 2.94 | 7.26 |
| ctrl_n6 | 2669 | 0.70 ± 0.00 | 1.10 ± 0.02 | 2.11 ± 0.08 | 4.99 ± 0.09 | 3.00 | 7.09 |

### Slowdown

| config | forward vs classical | first_grad vs classical | second_grad vs classical | param_grad vs classical | forward vs matched control | first_grad vs matched control | second_grad vs matched control | param_grad vs matched control |
|---|---|---|---|---|---|---|---|---|
| narrow | 14.2x | 14.3x | 12.6x | 9.5x | 10.5x | 10.0x | 10.0x | 8.5x |
| baseline | 29.3x | 37.3x | 37.6x | 28.4x | 22.1x | 26.8x | 30.1x | 24.4x |
| wide | 57.9x | 101.5x | 93.8x | 91.5x | 45.0x | 76.1x | 75.8x | 82.8x |
| deep | 49.3x | 63.5x | 58.1x | 46.8x | 37.2x | 45.6x | 46.5x | 40.3x |

Matched control = same classical layers, VQC replaced by tanh (narrow -> ctrl_n2, baseline and deep -> ctrl_n4, wide -> ctrl_n6).

## Qubit sweep (2 layers)

| n_qubits | forward | first_grad | second_grad | param_grad |
|---|---|---|---|---|
| 2 | 7.87 ± 0.19 | 12.65 ± 0.72 | 22.49 ± 0.40 | 44.10 ± 0.73 |
| 4 | 16.88 ± 0.37 | 34.55 ± 0.91 | 65.16 ± 1.08 | 130.6 ± 2.8 |
| 6 | 32.87 ± 0.32 | 85.70 ± 1.73 | 154.3 ± 2.1 | 411.1 ± 4.6 |
| 8 | 139.7 ± 3.9 | 374.4 ± 6.3 | 865.3 ± 17.3 | 2533.8 ± 12.5 |
| 10 | 735.9 ± 4.4 | - | - | - |
| 12 | 3461.6 ± 23.3 | - | - | - |

Fit per stage (forward, first_grad, second_grad, param_grad): 1.85x per added qubit (log2-linear fit), 1.74x per added qubit (log2-linear fit), 1.81x per added qubit (log2-linear fit), 1.94x per added qubit (log2-linear fit)

## Depth sweep (4 qubits)

| n_layers | forward | first_grad | second_grad | param_grad |
|---|---|---|---|---|
| 1 | 11.09 ± 0.09 | 20.30 ± 0.07 | 40.31 ± 0.61 | 90.52 ± 0.42 |
| 2 | 16.26 ± 0.21 | 30.37 ± 0.31 | 59.86 ± 0.61 | 130.1 ± 0.6 |
| 4 | 26.74 ± 0.36 | 50.68 ± 0.11 | 95.40 ± 0.56 | 209.1 ± 1.9 |
| 8 | 48.04 ± 0.27 | 93.73 ± 0.58 | 174.8 ± 1.4 | 371.4 ± 3.1 |

Fit per stage (forward, first_grad, second_grad, param_grad): 5.28 ms per added layer (linear fit), 10.51 ms per added layer (linear fit), 19.17 ms per added layer (linear fit), 40.14 ms per added layer (linear fit)

## Batch sweep (baseline)

| batch | forward | first_grad | second_grad | param_grad |
|---|---|---|---|---|
| 100 | 12.26 ± 0.46 | 17.49 ± 0.15 | 27.81 ± 0.08 | 51.74 ± 0.71 |
| 500 | 13.79 ± 0.10 | 21.94 ± 0.29 | 37.87 ± 0.59 | 72.48 ± 1.24 |
| 1000 | 15.01 ± 0.24 | 25.93 ± 0.32 | 47.35 ± 0.89 | 96.30 ± 1.01 |
| 2540 | 17.24 ± 0.43 | 33.85 ± 0.22 | 66.27 ± 1.24 | 138.1 ± 1.8 |
| 5000 | 21.30 ± 0.09 | 47.48 ± 0.95 | 88.69 ± 0.51 | 198.2 ± 1.9 |

Fit per stage (forward, first_grad, second_grad, param_grad): 0.13 = exponent of time ~ batch^k (log-log fit), 0.25 = exponent of time ~ batch^k (log-log fit), 0.30 = exponent of time ~ batch^k (log-log fit), 0.34 = exponent of time ~ batch^k (log-log fit)

## Backend comparison (circuit-only forward)

| case | default.qubit | lightning.qubit | lightning / default |
|---|---|---|---|
| narrow | 6.81 ± 0.17 | 790.9 ± 4.9 | 116.2x |
| baseline | 14.89 ± 0.27 | 1552.5 ± 17.2 | 104.3x |
| wide | 29.46 ± 0.11 | 2274.1 ± 22.0 | 77.2x |
| deep | 24.47 ± 0.33 | 2742.2 ± 12.7 | 112.0x |
| batch100 | 8.56 ± 0.02 | 91.29 ± 0.78 | 10.7x |
| batch500 | 10.50 ± 0.19 | 322.1 ± 1.7 | 30.7x |
| batch1000 | 12.58 ± 0.20 | 603.2 ± 5.2 | 48.0x |
| batch2540 | 14.62 ± 0.12 | 1557.8 ± 16.9 | 106.6x |
| batch5000 | 17.64 ± 0.18 | 2972.7 ± 7.1 | 168.5x |

- batch exponent on default.qubit: 0.18 (1.0 = linear in batch, 0 = flat)
- batch exponent on lightning.qubit: 0.90 (1.0 = linear in batch, 0 = flat)

## Drift check (end of run vs. `main`)

| config | forward (end / main) | first_grad (end / main) | second_grad (end / main) | param_grad (end / main) |
|---|---|---|---|---|
| classical | 1.142 / 1.150 / 1.074 (fast) | 1.252 / 1.238 / 1.099 (fast) | 1.327 / 1.355 / 1.101 (fast) | 1.222 / 1.251 / 1.213 (fast) |
| baseline | 1.148 / 1.148 / 0.999 | 1.205 / 1.224 / 1.115 | 1.112 / 1.119 / 1.185 | 1.119 / 1.130 / 1.023 |
| wide | 1.118 / 1.113 / 1.036 | 1.083 / 1.100 / 1.100 | 1.135 / 1.122 / 1.083 | 1.145 / 1.129 / 1.028 |

Each cell: median / mean / min ratio. Ratios near 1.000 mean the machine state was stable. The verdict uses the median ratio of cells taking at least 5 ms in `main`; cells marked (fast) are faster than that, dominated by timing jitter, and shown for information only.

**Largest deviation is above 10% (baseline `first_grad` 1.205): the run drifted; repeat it on an idle machine.**

Largest deviation among (fast) cells (not used for the verdict): 32.7% (classical `second_grad` 1.327).
