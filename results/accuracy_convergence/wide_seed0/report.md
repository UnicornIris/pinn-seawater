# Convergence benchmark report

Run folder: `results/accuracy_convergence/wide_seed0`. Config: **wide** (n_qubits=6, n_qlayers=2), seed=0.

git commit: `9614deb9add7b9de3cfe5c7fbce3c886dfd5d393` (clean)


| scenario | model | final loss | MAE | L2 | PDE residual | time (s) |
|---|---|---|---|---|---|---|
| DE_Dir_FWD | classical | 2.087e-06 | 2.310e-04 | 4.643e-04 | 1.058e-03 | 34.6 |
| DE_Dir_FWD | hybrid | 3.422e-06 | 4.263e-04 | 8.402e-04 | 1.190e-03 | 7466.4 |
| DE_Neum_FWD | classical | 8.552e-07 | 1.665e-04 | 3.526e-04 | 6.467e-04 | 40.2 |
| DE_Neum_FWD | hybrid | 1.633e-05 | 1.310e-03 | 2.778e-03 | 2.479e-03 | 7586.8 |
| DE_Rob_FWD | classical | 1.749e-06 | 8.964e-05 | 7.383e-05 | 2.871e-04 | 134.7 |
| DE_Rob_FWD | hybrid | 4.482e-05 | 4.879e-04 | 4.082e-04 | 1.417e-03 | 20290.3 |
| DE_Dir_INV | classical | 3.153e-06 | 2.871e-04 | 5.724e-04 | 1.197e-03 | 41.1 |
| DE_Dir_INV | hybrid | 5.198e-06 | 6.760e-04 | 1.327e-03 | 1.484e-03 | 14708.9 |
| DE_Neum_INV | classical | 1.434e-06 | 2.488e-04 | 5.022e-04 | 8.271e-04 | 571.3 |
| DE_Neum_INV | hybrid | 1.169e-05 | 1.047e-03 | 2.085e-03 | 2.349e-03 | 20759.3 |
| DE_Rob_INV | classical | 1.923e-06 | 8.661e-05 | 7.596e-05 | 2.587e-04 | 2149.9 |
| DE_Rob_INV | hybrid | 3.636e-05 | 2.720e-04 | 2.396e-04 | 1.271e-03 | 51607.6 |
