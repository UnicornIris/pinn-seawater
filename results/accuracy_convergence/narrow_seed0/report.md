# Convergence benchmark report

Run folder: `results/accuracy_convergence/narrow_seed0`. Config: **narrow** (n_qubits=2, n_qlayers=2), seed=0.

git commit: `9614deb9add7b9de3cfe5c7fbce3c886dfd5d393` (clean)


| scenario | model | final loss | MAE | L2 | PDE residual | time (s) |
|---|---|---|---|---|---|---|
| DE_Dir_FWD | classical | 2.087e-06 | 2.310e-04 | 4.643e-04 | 1.058e-03 | 37.8 |
| DE_Dir_FWD | hybrid | 1.077e-05 | 8.058e-04 | 1.648e-03 | 2.219e-03 | 936.7 |
| DE_Neum_FWD | classical | 8.552e-07 | 1.665e-04 | 3.526e-04 | 6.467e-04 | 40.4 |
| DE_Neum_FWD | hybrid | 4.854e-05 | 1.850e-03 | 3.674e-03 | 4.873e-03 | 990.2 |
| DE_Rob_FWD | classical | 1.749e-06 | 8.964e-05 | 7.383e-05 | 2.871e-04 | 137.0 |
| DE_Rob_FWD | hybrid | 4.658e-04 | 2.025e-03 | 2.005e-03 | 4.832e-03 | 2653.8 |
| DE_Dir_INV | classical | 3.153e-06 | 2.871e-04 | 5.724e-04 | 1.197e-03 | 41.6 |
| DE_Dir_INV | hybrid | 1.127e-05 | 7.130e-04 | 1.458e-03 | 2.130e-03 | 1067.6 |
| DE_Neum_INV | classical | 1.434e-06 | 2.488e-04 | 5.022e-04 | 8.271e-04 | 37.6 |
| DE_Neum_INV | hybrid | 6.669e-05 | 2.795e-03 | 6.470e-03 | 5.244e-03 | 1140.0 |
| DE_Rob_INV | classical | 1.923e-06 | 8.661e-05 | 7.596e-05 | 2.587e-04 | 142.8 |
| DE_Rob_INV | hybrid | 7.558e-04 | 2.340e-03 | 1.862e-03 | 5.991e-03 | 3037.2 |
