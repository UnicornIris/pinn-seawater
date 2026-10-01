# Convergence benchmark report

Run folder: `results/accuracy_convergence\20260930_092855`. Config: **baseline** (n_qubits=4, n_qlayers=2), seed=0.

git commit: `a183fa84c84111815a1b63b26d9be44f5fcd0514` (clean)


| scenario | model | final loss | MAE | L2 | PDE residual | time (s) |
|---|---|---|---|---|---|---|
| DE_Dir_FWD | classical | 2.087e-06 | 2.310e-04 | 4.643e-04 | 1.058e-03 | 39.6 |
| DE_Dir_FWD | hybrid | 3.395e-06 | 3.401e-04 | 6.896e-04 | 1.156e-03 | 2605.9 |
| DE_Neum_FWD | classical | 8.552e-07 | 1.665e-04 | 3.526e-04 | 6.467e-04 | 41.2 |
| DE_Neum_FWD | hybrid | 4.374e-05 | 2.389e-03 | 4.666e-03 | 4.815e-03 | 6504.1 |
| DE_Rob_FWD | classical | 1.749e-06 | 8.964e-05 | 7.383e-05 | 2.871e-04 | 335.2 |
| DE_Rob_FWD | hybrid | 3.467e-05 | 3.838e-04 | 3.156e-04 | 1.157e-03 | 9118.3 |
| DE_Dir_INV | classical | 3.153e-06 | 2.871e-04 | 5.724e-04 | 1.197e-03 | 46.2 |
| DE_Dir_INV | hybrid | 6.465e-06 | 4.999e-04 | 1.001e-03 | 1.580e-03 | 3061.9 |
| DE_Neum_INV | classical | 1.434e-06 | 2.488e-04 | 5.022e-04 | 8.271e-04 | 41.9 |
| DE_Neum_INV | hybrid | 2.178e-05 | 1.177e-03 | 2.483e-03 | 3.074e-03 | 3190.3 |
| DE_Rob_INV | classical | 1.923e-06 | 8.661e-05 | 7.596e-05 | 2.587e-04 | 165.8 |
| DE_Rob_INV | hybrid | 2.678e-05 | 2.896e-04 | 2.460e-04 | 1.051e-03 | 8520.0 |
