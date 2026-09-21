"""Draw exemplar VQC diagrams for the four hybrid configurations, plus the
hybrid-network overview figure.

The circuit body mirrors QuantumLayer._circuit in qcpinn_seawater.py
(AngleEmbedding -> [RZ RX | ring CNOT | RX RZ] x n_layers -> <Z_i>).
It is re-declared here so the script does not need torch; keep the two in sync.

Run with an interpreter that has pennylane + matplotlib, from anywhere:
    /usr/bin/python3 paper/figures/draw_circuits.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pennylane as qml

OUT = Path(__file__).resolve().parent

# Must match CONFIGS in qcpinn_parameter_count.py
CONFIGS = {
    "narrow":   dict(n_qubits=2, n_qlayers=2),
    "baseline": dict(n_qubits=4, n_qlayers=2),
    "wide":     dict(n_qubits=6, n_qlayers=2),
    "deep":     dict(n_qubits=4, n_qlayers=4),
}


def build_vqc_figure(n_qubits, n_layers_drawn):
    """qml.draw_mpl figure of AngleEmbedding + n_layers_drawn ansatz layers."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x, params):
        qml.templates.AngleEmbedding(x, wires=range(n_qubits), rotation="X")
        for layer in range(n_layers_drawn):
            p = params[layer]
            idx = 0
            for q in range(n_qubits):
                qml.RZ(p[idx], wires=q); idx += 1
                qml.RX(p[idx], wires=q); idx += 1
            for q in range(n_qubits):
                qml.CNOT(wires=[q, (q + 1) % n_qubits])
            for q in range(n_qubits):
                qml.RX(p[idx], wires=q); idx += 1
                qml.RZ(p[idx], wires=q); idx += 1
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    fig, _ = qml.draw_mpl(circuit, style="black_white", decimals=None)(
        np.zeros(n_qubits), np.zeros((n_layers_drawn, n_qubits * 4)))
    return fig


def draw_compact_grid():
    """2x2 panel: one ansatz layer per configuration, annotated with the layer count."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
    for ax, (name, cfg) in zip(axes.ravel(), CONFIGS.items()):
        nq, nl = cfg["n_qubits"], cfg["n_qlayers"]
        sub = build_vqc_figure(nq, 1)
        sub.canvas.draw()
        img = np.asarray(sub.canvas.buffer_rgba())
        plt.close(sub)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(
            f"{name}: n={nq} qubits, L={nl} layers ({nl * nq * 4} quantum parameters)\n"
            f"one variational layer drawn; it is applied {nl}\u00d7 between "
            f"AngleEmbedding and measurement", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"circuits_compact.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def draw_vqc(name, n_qubits, n_qlayers):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x, params):
        qml.templates.AngleEmbedding(x, wires=range(n_qubits), rotation="X")
        for layer in range(n_qlayers):
            p = params[layer]
            idx = 0
            for q in range(n_qubits):
                qml.RZ(p[idx], wires=q); idx += 1
                qml.RX(p[idx], wires=q); idx += 1
            for q in range(n_qubits):
                qml.CNOT(wires=[q, (q + 1) % n_qubits])
            for q in range(n_qubits):
                qml.RX(p[idx], wires=q); idx += 1
                qml.RZ(p[idx], wires=q); idx += 1
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    x = np.zeros(n_qubits)
    params = np.zeros((n_qlayers, n_qubits * 4))
    fig, _ = qml.draw_mpl(circuit, style="black_white", decimals=None)(x, params)
    n_params = n_qlayers * n_qubits * 4
    fig.suptitle(
        f"{name}: {n_qubits} qubits, {n_qlayers} layers, {n_params} quantum parameters",
        fontsize=11, y=1.02,
    )
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"circuit_{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def draw_overview():
    """Classical layers -> VQC -> classical layers, for the hybrid network."""
    fig, ax = plt.subplots(figsize=(14.5, 2.8))
    ax.set_xlim(0, 15.4)
    ax.set_ylim(0, 2.8)
    ax.axis("off")

    # (label, width, fill colour, group)
    blocks = [
        ("Input\n(z, t)", 1.2, "#f2f2f2", None),
        ("Normalize\nto [-1, 1]", 1.5, "#f2f2f2", None),
        ("Preprocessor MLP\n2 → 32 → 32 → n", 2.2, "#dbe9f6", "classical"),
        ("VQC\nAngleEmbedding +\nL variational layers", 2.4, "#fbe3c8", "quantum"),
        ("Output scale\n(learnable, n)", 1.6, "#f2f2f2", None),
        ("Postprocessor MLP\nn → 32 → 32 → 1", 2.2, "#dbe9f6", "classical"),
    ]
    gap, y0, h = 0.45, 0.9, 1.2
    x = 0.1
    centres = []
    for label, w, color, group in blocks:
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.03",
                                    fc=color, ec="black", lw=1))
        ax.text(x + w / 2, y0 + h / 2, label, ha="center", va="center", fontsize=9)
        if group:
            colour = "#2b5d8a" if group == "classical" else "#b5651d"
            ax.text(x + w / 2, 0.5, group, ha="center", fontsize=9, color=colour)
        ax.annotate("", xy=(x + w + gap, y0 + h / 2), xytext=(x + w + 0.03, y0 + h / 2),
                    arrowprops=dict(arrowstyle="->", lw=1))
        centres.append(x + w / 2)
        x += w + gap
    ax.text(x - 0.05, y0 + h / 2, r"$\hat{T}(z,t)$", ha="left", va="center", fontsize=10)
    ax.text(0.1, 2.5, "n = number of qubits, L = number of variational layers; "
            "tensors between blocks have shape [batch, n]", ha="left", fontsize=9,
            style="italic")
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"hybrid_overview.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    for name, cfg in CONFIGS.items():
        draw_vqc(name, **cfg)
    draw_compact_grid()
    draw_overview()
    print("wrote:", *sorted(p.name for p in OUT.glob("*.pdf")), sep="\n  ")
