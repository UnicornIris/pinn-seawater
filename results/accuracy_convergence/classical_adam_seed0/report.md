# Convergence benchmark report

Run folder: `results/accuracy_convergence/classical_adam_seed0`. Config: **baseline** (n_qubits=4, n_qlayers=2), seed=0.

git commit: `c148ecff5400e1ad045b934c92593f9dc0afa7b6` (clean)


| scenario | model | final loss | MAE | L2 | PDE residual | time (s) |
|---|---|---|---|---|---|---|
| DE_Dir_FWD | classical_adam | 1.734e-05 | 5.430e-04 | 1.156e-03 | 2.534e-03 | 92.7 |
| DE_Neum_FWD | classical_adam | 9.832e-06 | 6.929e-04 | 2.272e-03 | 1.762e-03 | 95.8 |
| DE_Rob_FWD | classical_adam | 3.200e-05 | 4.058e-04 | 3.490e-04 | 1.054e-03 | 257.3 |
| DE_Dir_INV | classical_adam | 3.287e-05 | 6.951e-04 | 1.421e-03 | 3.562e-03 | 97.0 |
| DE_Neum_INV | classical_adam | 9.747e-06 | 6.655e-04 | 1.542e-03 | 1.913e-03 | 102.9 |
| DE_Rob_INV | classical_adam | 3.078e-05 | 3.417e-04 | 3.267e-04 | 1.037e-03 | 277.4 |
