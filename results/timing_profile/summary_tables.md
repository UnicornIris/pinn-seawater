# QCPINN Timing Profile — Summary Tables

Source: `qcpinn_timing_profile.py`, raw CSVs in `results/timing_profile/<tag>/`.
Batch size = 2540 (matches training batch) unless noted otherwise.
All four tables below are now from the **same environment**: system Python
3.9, PennyLane 0.38, torch 2.8 (the only environment on this machine with
PennyLane installed — the original Python 3.14 environment used for an
earlier, since-superseded version of Tables 1–2 is no longer available, so
those numbers were re-measured here rather than reused). Every table is
averaged over 3 independent seeds (seed controls net weights / batch data /
circuit params — not just repeated timing noise on one draw), so all four
report a mean and a seed-to-seed spread.

## Table 1 — Forward / 1st-order grad / 2nd-order grad breakdown (mean ± std across 3 seeds)

| Config | n_qubits | n_layers | Forward (ms) | std (ms) | +1st-order grad (ms) | +2nd-order grad (ms) | 2nd-grad / Forward | Forward vs. classical |
|---|---|---|---|---|---|---|---|---|
| classical | – | – | 0.691 | 0.008 | 1.093 | 2.029 | 2.94x | 1.0x |
| narrow | 2 | 2 | 4.485 | 0.045 | 7.745 | 13.567 | 3.02x | 6.5x |
| baseline | 4 | 2 | 12.368 | 0.068 | 29.759 | 54.836 | 4.43x | 17.9x |
| wide | 6 | 2 | 42.098 | 0.719 | 103.788 | 226.085 | 5.37x | 61.0x |
| deep | 4 | 4 | 21.260 | 0.138 | 47.558 | 91.467 | 4.30x | 30.8x |

## Table 2 — Backend comparison: default.qubit vs. lightning.qubit (mean ± std across 3 seeds, forward only)

| Config | n_qubits | n_layers | default.qubit (ms) | std (ms) | lightning.qubit (ms) | std (ms) | lightning / default |
|---|---|---|---|---|---|---|---|
| narrow | 2 | 2 | 3.736 | 0.043 | 287.1 | 11.87 | 76.9x |
| baseline | 4 | 2 | 11.657 | 0.036 | 517.2 | 2.55 | 44.4x |
| wide | 6 | 2 | 38.269 | 0.392 | 785.4 | 2.28 | 20.5x |
| deep | 4 | 4 | 20.454 | 0.055 | 815.9 | 7.07 | 39.9x |

## Table 3 — Forward time vs. qubit count (baseline layers=2, default.qubit, mean ± std across 3 seeds)

| n_qubits | Forward (ms) | Std across seeds (ms) | CV |
|---|---|---|---|
| 2 | 3.688 | 0.016 | 0.4% |
| 4 | 11.537 | 0.061 | 0.5% |
| 6 | 38.447 | 0.143 | 0.4% |
| 8 | 227.368 | 2.956 | 1.3% |
| 10 | 1629.625 | 2.819 | 0.2% |
| 12 | 7666.645 | 2.058 | 0.03% |

Log-linear fit across all 6 points (least squares on log2(time) vs. n_qubits):
**each added qubit multiplies forward time by 2.19x** (pure statevector
doubling predicts exactly 2x). Seed-to-seed variance is small everywhere
(CV ≤ 1.3%), so the point estimates and the fitted growth rate are stable —
the earlier concern about a single-run sweep being noise-driven does not
hold up now that it's been checked.

## Table 4 — Forward time vs. batch size (baseline: 4 qubits, 2 layers, mean ± std across 3 seeds)

| Batch size | default.qubit (ms) | std (ms) | lightning.qubit (ms) | std (ms) | lightning / default |
|---|---|---|---|---|---|
| 100 | 5.386 | 0.057 | 19.877 | 0.076 | 3.7x |
| 500 | 6.829 | 0.092 | 104.705 | 0.215 | 15.3x |
| 1000 | 7.970 | 0.061 | 214.075 | 5.875 | 26.9x |
| 2540 | 11.729 | 0.013 | 539.671 | 20.794 | 46.0x |
| 5000 | 16.924 | 0.154 | 1037.327 | 8.932 | 61.3x |

As batch grows 100→5000 (50x): default.qubit grows only 3.1x (flat =
vectorized), lightning.qubit grows 52.2x (essentially linear — almost
exactly tracking the 50x batch increase, a cleaner confirmation of
per-sample looping than the earlier single-run estimate). Seed-to-seed std
is small relative to the mean in all cases (largest CV ~3.9%, at
batch=2540/5000 for lightning), so the vectorized-vs-looped conclusion is
robust. Note the absolute "lightning is Nx slower" multiplier did shift
noticeably (46–181x) between an earlier Python 3.14 measurement and this
Python 3.9 one, so treat specific multipliers as tied to this measurement
environment rather than universal constants if the run environment changes
again in the future.
