# Revised Paper Outline: Timing Profiles and Parameter Counts of Classical vs. Hybrid Quantum PINNs

## Scope

The paper is restricted to two core metrics comparing the classical PINN with hybrid quantum-classical PINNs (QCPINN):

1. **Hardware timing profile** (forward pass, first-order gradient, second-order gradient)
2. **Parameter count** (quantum vs. classical share)

Accuracy comparisons (MAE across the six scenarios), gradient-norm ablations, and barren-plateau discussion are secondary and move to the appendix (or are removed).

## Title and Abstract

- **Title direction:** measure-focused rather than accuracy-focused, e.g. *Cost of Hybrid Quantum PINNs: Parameter Count vs. Wall-Clock Time*.
- **Abstract** covers three points only:
  - Parameter count: the quantum branch adds very few trainable parameters, and total parameters are about 40% lower than the classical baseline (2421-2717 vs. 4353).
  - Timing: forward, first-order-gradient, and second-order-gradient time increase substantially and grow with qubit count, circuit depth, and batch size.
  - Takeaway: parameter count alone overstates the efficiency of hybrid models.
- Accuracy gets at most one sentence stating it is not the subject of this paper.

## 1. Introduction

- Background: PINNs and QCPINNs; parameter count is the usual efficiency metric in the literature.
- Gap: wall-clock time is rarely reported.
- **Research questions** (replacing the old RQ1/RQ2):
  - RQ1: How large is the parameter-count advantage of hybrid models, and where does it come from?
  - RQ2: What timing cost accompanies it, and how does that cost scale with qubit count, depth, batch size, and simulator backend?
- Contributions: (i) a parameter-count breakdown under a common accounting; (ii) stage-wise timing of forward, first-order, and second-order gradient computation; (iii) sweeps over qubits, batch size, and backend.

## 2. Background and Key Concepts (new)

- 2.1 PINN loss structure: PDE residual term, boundary term, data term.
- 2.2 **First-order gradient:** derivative of the network output with respect to its *input*, (dT/dz, dT/dt); needed for the time term of the PDE residual and for Neumann/Robin boundary residuals. Distinguish it from the *parameter* gradient dL/d(theta) used by the optimizer, which is not part of the timed stages.
- 2.3 **Second-order gradient:** d^2T/dz^2, a second backward pass through the graph of the first-order pass; needed for the diffusion term. The three timed stages (forward, first_grad, second_grad) are cumulative, and `second_over_forward_ratio` is second_grad / forward. Drafted text: `paper/gradient_definitions.md`.
- 2.4 VQC basics: qubit, layer, ansatz, trainable quantum parameters; simulator vs. real hardware.
- 2.5 How gradients are obtained: backpropagation (simulator) vs. parameter-shift rule (hardware) and their cost differences.
- Include a notation/terminology table.

## 3. Model Architectures (new circuit diagrams)

- 3.1 Classical baseline PINN (4353 parameters).
- 3.2 Hybrid overview figure: classical layers -> quantum layer -> classical layers.
- 3.3 Four hybrid configurations, each with an **exemplar circuit diagram** (side by side):

| Config | Qubits | Layers | Quantum params | Total params | Ratio to classical |
|---|---|---|---|---|---|
| classical | - | - | 0 | 4353 | 1.00 |
| narrow | 2 | 2 | 16 | 2421 | 0.56 |
| baseline | 4 | 2 | 32 | 2569 | 0.59 |
| wide | 6 | 2 | 48 | 2717 | 0.62 |
| deep | 4 | 4 | 64 | 2601 | 0.60 |

- 3.4 State explicitly how wide (more qubits) and deep (more layers) differ from baseline.

## 4. Experimental Setup

- Timing metrics: forward, first_grad, second_grad; mean +/- std over 3 seeds.
- Hardware and software environment: CPU/GPU, PennyLane version, backends (`default.qubit`, `lightning.qubit`). **To be filled in; required for a hardware-timing paper.**
- Timing protocol: warm-up runs, repetitions, batch size (default 2540).

## 5. Results I: Parameter Count

- Table: parameter breakdown (quantum vs. classical share) per configuration.
- Observation: the quantum branch contributes only 16-64 parameters, so nearly all of the total reduction comes from the smaller classical part. Worth its own paragraph.
- Note: `quantum_share` in `parameter_count.csv` appears to be a percentage (0.66 = 0.66%), not a fraction; label accordingly.

## 6. Results II: Hardware Timing

- 6.1 **Per-configuration timing** (three stages). Ratio to classical (from CSV means; re-verify before submission):

| Config | forward | second_grad |
|---|---|---|
| narrow | 6.3x | 7.4x |
| baseline | 18.3x | 28.8x |
| wide | 57.6x | 114.8x |
| deep | 29.7x | 47.6x |

- 6.2 **Second-order / forward ratio:** classical 2.7, narrow 3.2, baseline 4.3, wide 5.4, deep 4.4. The extra cost of second-order gradients grows with model size.
- 6.3 **Qubit sweep (2 to 10):** forward time rises from about 3.7 ms to about 1.63 s, with a sharp increase beyond 8 qubits.
- 6.4 **Batch-size sweep (100 to 5000).**
- 6.5 **Backend comparison:** `lightning.qubit` is 21-72x slower than `default.qubit`, opposite to the usual expectation; the text needs an explanation (e.g. per-sample circuit calls at this batch size).
- Figures: split `timing_profile_summary.png` into 3-4 subplots.

## 7. Discussion

- Parameter count and timing together: time cost per parameter saved.
- Scope of the timing results: classical simulation, not real quantum hardware.
- Limitations: single PDE benchmark, small scale, single ansatz.

## 8. Conclusion

## Appendix (secondary material, moved down)

- A. MAE comparison across the six scenarios (one table, no extended discussion).
- B. Gradient-norm and barren-plateau ablation (short, or removed).
- C. Per-seed raw timing tables (the `byseed` CSVs).

## Related Work

Cut to about half a page, keeping only work relevant to how efficiency is reported (e.g. Farea et al. reporting 15-40x per-iteration cost, Leong et al. reporting 17-82x). Accuracy conclusions of prior work are not discussed at length.

## Open items

1. ~~Locate the LaTeX source~~ — not found; rebuilt from scratch as `main.tex` + `sections/*.tex` + `refs.bib` (build: `tectonic main.tex`). The Sep 2 PDF is in `../archive/paper_old/`.
2. Confirm units in the timing CSVs (presumed seconds).
3. Verify all ratios above against the CSVs before they go into the text.
4. Confirm the `deep` configuration (4 qubits, 4 layers) matches what the paper should call "deep".
5. Fill in the author list and affiliation placeholders.
