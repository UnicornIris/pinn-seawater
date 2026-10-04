# Convergence benchmark report

Run folder: `results/accuracy_convergence/deep_seed0`. Config: **deep** (n_qubits=4, n_qlayers=4), seed=0.

git commit: `9614deb9add7b9de3cfe5c7fbce3c886dfd5d393` (clean)


| scenario | model | final loss | MAE | L2 | PDE residual | time (s) |
|---|---|---|---|---|---|---|
| DE_Dir_FWD | classical | 2.087e-06 | 2.310e-04 | 4.643e-04 | 1.058e-03 | 35.9 |
| DE_Dir_FWD | hybrid | 1.359e-05 | 9.860e-04 | 1.985e-03 | 2.640e-03 | 5917.4 |
| DE_Neum_FWD | classical | 8.552e-07 | 1.665e-04 | 3.526e-04 | 6.467e-04 | 39.2 |
| DE_Neum_FWD | hybrid | 1.295e-05 | 1.092e-03 | 2.194e-03 | 2.448e-03 | 4460.0 |
| DE_Rob_FWD | classical | 1.749e-06 | 8.964e-05 | 7.383e-05 | 2.871e-04 | 131.9 |
| DE_Rob_FWD | hybrid | 3.333e-05 | 4.229e-04 | 3.639e-04 | 1.193e-03 | 11873.1 |
| DE_Dir_INV | classical | 3.153e-06 | 2.871e-04 | 5.724e-04 | 1.197e-03 | 40.0 |
| DE_Dir_INV | hybrid | 1.486e-05 | 1.163e-03 | 2.438e-03 | 2.753e-03 | 4663.7 |
| DE_Neum_INV | classical | 1.434e-06 | 2.488e-04 | 5.022e-04 | 8.271e-04 | 36.2 |
| DE_Neum_INV | hybrid | 2.658e-05 | 1.702e-03 | 3.531e-03 | 3.576e-03 | 4881.7 |
| DE_Rob_INV | classical | 1.923e-06 | 8.661e-05 | 7.596e-05 | 2.587e-04 | 138.8 |
| DE_Rob_INV | hybrid | 4.440e-05 | 6.468e-04 | 5.406e-04 | 1.309e-03 | 13019.9 |
