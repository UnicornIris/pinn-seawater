"""
Parameter-count comparison for the QCPINN classical vs. quantum networks.

Prompted by an advisor-meeting note: "compare classical and quantum
parameters" -- a different axis from wall-clock time (qcpinn_timing_profile.py)
or accuracy (the main results table). A network with far fewer trainable
parameters taking 20-90x longer per forward pass is a much sharper way to
state the cost/expressivity tradeoff than time alone.

For each HybridQNN config, parameters are split into:
    quantum       - the VQC's own rotation angles (QuantumLayer.params)
    classical     - preprocessor + output_scale + postprocessor
This matters for the "which part of the chain" framing from the same
meeting: the classical wrapper around the circuit is a small, fixed cost
(same for every quantum config, roughly matching ClassicalDNN's own size),
while the "quantum" column is the only thing that scales with circuit size
-- so a raw total-parameter comparison would understate how little of a
larger quantum config's forward cost is coming from its own extra weights.

Usage:
    python qcpinn_parameter_count.py
"""
import csv
import datetime
import os

from qcpinn_seawater import ClassicalDNN, HybridQNN

CONFIGS = {
    "classical": None,
    "narrow":    dict(n_qubits=2, n_qlayers=2),
    "baseline":  dict(n_qubits=4, n_qlayers=2),
    "wide":      dict(n_qubits=6, n_qlayers=2),
    "deep":      dict(n_qubits=4, n_qlayers=4),
}


def count_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def main():
    rows = []
    print(f"\n{'config':<12}{'n_qubits':>10}{'n_layers':>10}{'quantum':>10}"
          f"{'classical':>12}{'total':>10}{'quantum share':>16}")
    print("-" * 80)

    classical_net = ClassicalDNN()
    classical_total = count_params(classical_net)
    rows.append(dict(config="classical", n_qubits="", n_layers="",
                      quantum_params=0, classical_params=classical_total,
                      total_params=classical_total, quantum_share=0.0))
    print(f"{'classical':<12}{'':>10}{'':>10}{0:>10}{classical_total:>12}"
          f"{classical_total:>10}{'0.0%':>16}")

    for name, cfg in CONFIGS.items():
        if cfg is None:
            continue
        net = HybridQNN(n_qubits=cfg["n_qubits"], n_qlayers=cfg["n_qlayers"])
        quantum_params = count_params(net.quantum)
        classical_params = (count_params(net.preprocessor)
                             + count_params(net.postprocessor)
                             + net.output_scale.numel())
        total = quantum_params + classical_params
        share = quantum_params / total * 100

        rows.append(dict(config=name, n_qubits=cfg["n_qubits"], n_layers=cfg["n_qlayers"],
                          quantum_params=quantum_params, classical_params=classical_params,
                          total_params=total, quantum_share=share))
        print(f"{name:<12}{cfg['n_qubits']:>10}{cfg['n_qlayers']:>10}{quantum_params:>10}"
              f"{classical_params:>12}{total:>10}{share:>15.1f}%")

    print(f"\nTotal parameters vs. classical ({classical_total}):")
    for r in rows:
        print(f"  {r['config']:<12} {r['total_params']:>6} params  "
              f"({r['total_params']/classical_total:.2f}x classical)")

    out_dir = f"results/parameter_count/{datetime.datetime.now().strftime('%Y%m%d')}"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = f"{out_dir}/parameter_count.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
