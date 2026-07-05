"""
QCPINN for Seawater Temperature Diffusion Equation
Extends: Han et al. (2026), Journal of Oceanology and Limnology
Architecture: Classical pre-processor → Variational Quantum Circuit → Classical post-processor
Reference quantum circuit design: Afrah et al. (2025), arXiv:2503.16678

PDE:  dT/dt = kz * d²T/dz²
Domain: z ∈ [0, 2], t ∈ [0, 5]
"""

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import os
import csv
import datetime

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cpu")

Z_MIN, Z_MAX = 0.0, 2.0
T_MIN, T_MAX = 0.0, 5.0
KZ_TRUE = 0.05

# Normalize (z,t) → [-1, 1] for AngleEmbedding (avoids RX angle aliasing)
def normalize_input(zt: torch.Tensor) -> torch.Tensor:
    z_norm = 2.0 * (zt[:, 0:1] - Z_MIN) / (Z_MAX - Z_MIN) - 1.0
    t_norm = 2.0 * (zt[:, 1:2] - T_MIN) / (T_MAX - T_MIN) - 1.0
    return torch.cat([z_norm, t_norm], dim=1)


# ── Analytical solutions (same as classical PINN) ─────────────────────────────
def T_dirichlet(z, t):
    return (np.exp(-KZ_TRUE * (np.pi / 2) ** 2 * t) * np.sin(np.pi * z / 2)
            + np.exp(-KZ_TRUE * np.pi ** 2 * t) * np.sin(np.pi * z))

def T_neumann(z, t):
    return (np.exp(-KZ_TRUE * (np.pi / 2) ** 2 * t) * np.cos(np.pi * z / 2)
            + np.exp(-KZ_TRUE * np.pi ** 2 * t) * np.cos(np.pi * z))

def T_robin(z, t):
    return (np.exp(KZ_TRUE * t - z)
            + np.exp(-KZ_TRUE * (np.pi / 2) ** 2 * t)
              * (np.pi / 2 * np.cos(np.pi * z / 2) - np.sin(np.pi * z / 2))
            + np.exp(-KZ_TRUE * np.pi ** 2 * t)
              * (np.pi * np.cos(np.pi * z) - np.sin(np.pi * z)))


# ── Quantum Layer ──────────────────────────────────────────────────────────────
class QuantumLayer(nn.Module):
    """
    Variational Quantum Circuit using the 'layered' ansatz from Afrah et al.

    Circuit structure per layer:
        AngleEmbedding(x)          — encode classical features as RX rotations
        RZ(θ) RX(θ) on each qubit — single-qubit rotations (pre-entanglement)
        CNOT(i → i+1 mod n)       — ring entanglement
        RX(θ) RZ(θ) on each qubit — single-qubit rotations (post-entanglement)
    Output: PauliZ expectation value <Z_i> ∈ [-1, 1] for each qubit
    """
    def __init__(self, n_qubits: int = 6, n_layers: int = 3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Each layer: n_qubits × 4 parameters (RZ, RX, RX, RZ per qubit)
        self.params = nn.Parameter(
            torch.empty(n_layers, n_qubits * 4, dtype=torch.float32)
        )
        nn.init.xavier_normal_(self.params)

        dev = qml.device("default.qubit", wires=n_qubits)
        self._qnode = qml.QNode(
            self._circuit, dev, interface="torch", diff_method="backprop"
        )

    def _circuit(self, x):
        # x shape: [batch, n_qubits] — AngleEmbedding handles batched input
        qml.templates.AngleEmbedding(x, wires=range(self.n_qubits), rotation="X")
        for layer_idx in range(self.n_layers):
            p = self.params[layer_idx]
            idx = 0
            for q in range(self.n_qubits):
                qml.RZ(p[idx], wires=q); idx += 1
                qml.RX(p[idx], wires=q); idx += 1
            for q in range(self.n_qubits):
                qml.CNOT(wires=[q, (q + 1) % self.n_qubits])
            for q in range(self.n_qubits):
                qml.RX(p[idx], wires=q); idx += 1
                qml.RZ(p[idx], wires=q); idx += 1
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, n_qubits]
        # _qnode returns list of n_qubits tensors, each shape [batch]
        out = self._qnode(x)                          # list of n_qubits tensors
        return torch.stack(out, dim=1).float()        # [batch, n_qubits]


# ── Hybrid Quantum-Classical Network ──────────────────────────────────────────
class HybridQNN(nn.Module):
    """
    Architecture (mirrors DVPDESolver from Afrah et al., with improvements):

        Input (z, t)   [batch, 2]
            ↓ normalize_input → [-1,1]
            ↓ preprocessor: Linear(2→hidden) + Tanh + Linear(hidden→hidden) + Tanh + Linear(hidden→n_qubits)
        [batch, n_qubits]
            ↓ QuantumLayer (VQC)
        [batch, n_qubits]  — PauliZ expectation values ∈ [-1,1]
            ↓ output_scale  — learnable per-qubit scale (breaks [-1,1] saturation)
            ↓ postprocessor: Linear(n_qubits→hidden) + Tanh + Linear(hidden→hidden) + Tanh + Linear(hidden→1)
        T̂  [batch, 1]
    """
    def __init__(self, n_qubits: int = 6, n_qlayers: int = 3, hidden: int = 64):
        super().__init__()
        self.preprocessor = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_qubits),
        )
        self.quantum = QuantumLayer(n_qubits=n_qubits, n_layers=n_qlayers)
        # Learnable scale: breaks PauliZ ∈ [-1,1] saturation
        self.output_scale = nn.Parameter(torch.ones(n_qubits))
        self.postprocessor = nn.Sequential(
            nn.Linear(n_qubits, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        for m in list(self.preprocessor) + list(self.postprocessor):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = normalize_input(x)            # map to [-1,1] before angle embedding
        h = self.preprocessor(x_norm)          # [batch, n_qubits]
        h = self.quantum(h)                    # [batch, n_qubits], values in [-1,1]
        h = h * self.output_scale              # learnable rescale
        return self.postprocessor(h)           # [batch, 1]


# ── Classical baseline (same as pinn_seawater.py) ─────────────────────────────
class ClassicalDNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        in_dim = 2
        for _ in range(5):
            layers += [nn.Linear(in_dim, 32), nn.Tanh()]
            in_dim = 32
        layers.append(nn.Linear(32, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── PINN (network-agnostic) ───────────────────────────────────────────────────
class PINN:
    """
    Identical physics logic to pinn_seawater.py.
    Accepts any nn.Module as `network` — classical DNN or HybridQNN.
    """
    def __init__(self, bc_type="dirichlet", inverse=False,
                 network=None, w_pde=1.0, w_bc=1.0, w_ic=1.0, w_data=1.0):
        self.bc_type  = bc_type
        self.inverse  = inverse
        self.w_pde, self.w_bc, self.w_ic, self.w_data = w_pde, w_bc, w_ic, w_data

        self.net = network if network is not None else ClassicalDNN()
        self.net = self.net.to(DEVICE)

        if inverse:
            self.log_kz = nn.Parameter(torch.tensor([np.log(0.1)]))
        else:
            self.log_kz = None

        self._build_training_data()

    @property
    def kz(self):
        if self.inverse:
            return torch.exp(self.log_kz)
        return torch.tensor(KZ_TRUE, dtype=torch.float32)

    def _T_exact(self, z, t):
        if self.bc_type == "dirichlet": return T_dirichlet(z, t)
        elif self.bc_type == "neumann":  return T_neumann(z, t)
        else:                            return T_robin(z, t)

    def _tensor(self, z, t, requires_grad=False):
        zt = np.stack([z, t], axis=1).astype(np.float32)
        return torch.tensor(zt, requires_grad=requires_grad)

    def _build_training_data(self):
        z_f = np.random.uniform(Z_MIN, Z_MAX, 2540)
        t_f = np.random.uniform(T_MIN, T_MAX, 2540)
        self.zt_f = self._tensor(z_f, t_f, requires_grad=True)

        t_bc0 = np.random.uniform(T_MIN, T_MAX, 40)
        t_bcL = np.random.uniform(T_MIN, T_MAX, 40)
        z_bc  = np.concatenate([np.zeros(40), np.full(40, Z_MAX)])
        t_bc  = np.concatenate([t_bc0, t_bcL])
        T_bc  = self._T_exact(z_bc, t_bc).astype(np.float32)
        self.zt_bc = self._tensor(z_bc, t_bc,
                                   requires_grad=(self.bc_type != "dirichlet"))
        self.T_bc  = torch.tensor(T_bc).unsqueeze(1)

        z_ic = np.random.uniform(Z_MIN, Z_MAX, 160)
        t_ic = np.zeros(160)
        T_ic = self._T_exact(z_ic, t_ic).astype(np.float32)
        self.zt_ic = self._tensor(z_ic, t_ic)
        self.T_ic  = torch.tensor(T_ic).unsqueeze(1)

        if self.inverse:
            z_d = np.linspace(Z_MIN, Z_MAX, 4)
            t_d = np.linspace(T_MIN, T_MAX, 3)
            zz, tt = np.meshgrid(z_d, t_d)
            T_data = self._T_exact(zz.ravel(), tt.ravel()).astype(np.float32)
            self.zt_data = self._tensor(zz.ravel(), tt.ravel())
            self.T_data  = torch.tensor(T_data).unsqueeze(1)

    def _pde_residual(self):
        zt = self.zt_f
        zt.requires_grad_(True)
        T_pred = self.net(zt)
        dT     = torch.autograd.grad(T_pred, zt, torch.ones_like(T_pred),
                                     create_graph=True)[0]
        dT_dz, dT_dt = dT[:, 0:1], dT[:, 1:2]
        d2T_dz2 = torch.autograd.grad(dT_dz, zt, torch.ones_like(dT_dz),
                                      create_graph=True)[0][:, 0:1]
        return dT_dt - self.kz * d2T_dz2

    def _bc_residual(self):
        if self.bc_type == "dirichlet":
            return self.net(self.zt_bc) - self.T_bc
        zt = self.zt_bc.clone().requires_grad_(True)
        T_pred = self.net(zt)
        dT_dz  = torch.autograd.grad(T_pred, zt, torch.ones_like(T_pred),
                                     create_graph=True)[0][:, 0:1]
        if self.bc_type == "neumann":
            return dT_dz
        return dT_dz + T_pred   # robin: dT/dz + T = 0

    def loss(self):
        L = (self.w_pde * torch.mean(self._pde_residual() ** 2)
           + self.w_bc  * torch.mean(self._bc_residual()  ** 2)
           + self.w_ic  * torch.mean((self.net(self.zt_ic) - self.T_ic) ** 2))
        if self.inverse:
            L += self.w_data * torch.mean((self.net(self.zt_data) - self.T_data) ** 2)
        return L

    def train(self, adam_steps=5000, lbfgs_iter=10000):
        is_quantum = isinstance(self.net, HybridQNN)

        if is_quantum:
            # Separate LR: quantum params need higher LR; classical pre/post lower
            quantum_params  = list(self.net.quantum.parameters()) + [self.net.output_scale]
            classical_params = (list(self.net.preprocessor.parameters())
                               + list(self.net.postprocessor.parameters()))
            param_groups = [
                {"params": quantum_params,  "lr": 5e-3},
                {"params": classical_params, "lr": 1e-3},
            ]
            if self.inverse:
                param_groups.append({"params": [self.log_kz], "lr": 1e-3})
        else:
            all_params = list(self.net.parameters())
            if self.inverse:
                all_params.append(self.log_kz)
            param_groups = [{"params": all_params, "lr": 1e-3}]

        self.loss_history = []
        t0 = time.time()

        # For quantum: skip L-BFGS (too expensive per QNode eval), run more Adam steps
        # with cosine LR decay; for classical: keep original Adam + L-BFGS
        if is_quantum:
            total_steps = adam_steps + lbfgs_iter  # equivalent total budget
            adam = torch.optim.Adam(param_groups)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                adam, T_max=total_steps, eta_min=1e-5
            )
            for _ in range(total_steps):
                adam.zero_grad()
                L = self.loss()
                L.backward()
                adam.step()
                scheduler.step()
                self.loss_history.append(L.item())
        else:
            adam = torch.optim.Adam(param_groups)
            for _ in range(adam_steps):
                adam.zero_grad()
                L = self.loss()
                L.backward()
                adam.step()
                self.loss_history.append(L.item())

            all_params_flat = list(self.net.parameters())
            if self.inverse:
                all_params_flat.append(self.log_kz)
            lbfgs = torch.optim.LBFGS(
                all_params_flat, lr=1.0, max_iter=lbfgs_iter,
                tolerance_grad=1e-9, tolerance_change=1e-11,
                history_size=100, line_search_fn="strong_wolfe",
            )
            def closure():
                lbfgs.zero_grad()
                L = self.loss()
                L.backward()
                self.loss_history.append(L.item())
                return L
            lbfgs.step(closure)

        return time.time() - t0

    def evaluate(self, n_test=200):
        z_t = np.linspace(Z_MIN, Z_MAX, n_test)
        t_t = np.linspace(T_MIN, T_MAX, n_test)
        zz, tt = np.meshgrid(z_t, t_t)
        zt_test = torch.tensor(
            np.stack([zz.ravel(), tt.ravel()], axis=1), dtype=torch.float32)
        with torch.no_grad():
            T_pred = self.net(zt_test).numpy().reshape(n_test, n_test)
        T_true = self._T_exact(zz, tt)
        mae = np.mean(np.abs(T_pred - T_true))
        l2  = np.linalg.norm(T_pred - T_true) / (np.linalg.norm(T_true) + 1e-12)
        kz_err = None
        if self.inverse:
            kz_err = abs(float(torch.exp(self.log_kz).detach()) - KZ_TRUE)
        return dict(mae=mae, l2=l2, T_pred=T_pred, T_true=T_true,
                    z=z_t, t=t_t, kz_err=kz_err)

    def mean_pde_residual(self):
        with torch.enable_grad():
            zt = self.zt_f.detach().clone().requires_grad_(True)
            T  = self.net(zt)
            dT = torch.autograd.grad(T, zt, torch.ones_like(T),
                                     create_graph=True)[0]
            dT_dz = dT[:, 0:1]
            dT_dt = dT[:, 1:2]
            d2T_dz2 = torch.autograd.grad(dT_dz, zt, torch.ones_like(dT_dz),
                                           create_graph=False)[0][:, 0:1]
            r = (dT_dt - self.kz * d2T_dz2).detach().numpy()
        return float(np.mean(np.abs(r)))


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_result(result, title, out_path):
    z, t = result["z"], result["t"]
    T_pred, T_true = result["T_pred"], result["T_true"]
    error = np.abs(T_pred - T_true)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, data, label in zip(axes, [T_true, T_pred, error],
                                ["True T", "PINN T", "|Error|"]):
        im = ax.contourf(z, t, data, levels=50, cmap="RdBu_r")
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("z"); ax.set_ylabel("t"); ax.set_title(label)
    fig.suptitle(f"{title}  MAE={result['mae']:.3e}  L2={result['l2']:.3e}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_comparison(classical_history, quantum_history, out_path):
    """Loss curve comparison: classical vs quantum."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(classical_history, label="Classical PINN", alpha=0.8)
    ax.semilogy(quantum_history,   label="QCPINN",         alpha=0.8)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Loss")
    ax.set_title("Loss convergence: Classical vs QCPINN")
    ax.legend(); ax.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# ── Main comparison runner ────────────────────────────────────────────────────
def run_comparison(n_runs=5, n_qubits=6, n_qlayers=3):
    """
    Run all 6 scenarios with both classical and quantum networks.
    For each scenario and seed, trains both models and records metrics.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"qcpinn_results/{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nQCPINN Seawater Experiment")
    print(f"n_qubits={n_qubits}, n_qlayers={n_qlayers}, n_runs={n_runs}")
    print(f"Output: {out_dir}\n")

    train_cfg = {
        "DE_Dir_FWD":  dict(adam_steps=5000,  lbfgs_iter=10000, w_pde=1.0),
        "DE_Neum_FWD": dict(adam_steps=5000,  lbfgs_iter=10000, w_pde=1.0),
        "DE_Rob_FWD":  dict(adam_steps=20000, lbfgs_iter=20000, w_pde=10.0),
        "DE_Dir_INV":  dict(adam_steps=5000,  lbfgs_iter=10000, w_pde=1.0),
        "DE_Neum_INV": dict(adam_steps=5000,  lbfgs_iter=10000, w_pde=1.0),
        "DE_Rob_INV":  dict(adam_steps=20000, lbfgs_iter=20000, w_pde=10.0),
    }

    scenarios = [
        ("dirichlet", False, "DE_Dir_FWD"),
        ("neumann",   False, "DE_Neum_FWD"),
        ("robin",     False, "DE_Rob_FWD"),
        ("dirichlet", True,  "DE_Dir_INV"),
        ("neumann",   True,  "DE_Neum_INV"),
        ("robin",     True,  "DE_Rob_INV"),
    ]

    all_rows = []

    for bc, inv, name in scenarios:
        print(f"\n{'='*65}")
        print(f"  {name}  (bc={bc}, inverse={inv})  [{n_runs} runs]")
        print(f"{'='*65}")

        cfg = train_cfg[name]
        metrics = {"classical": [], "quantum": []}
        best = {"classical": None, "quantum": None}
        histories = {"classical": None, "quantum": None}

        for model_type in ["classical", "quantum"]:
            print(f"\n  -- {model_type.upper()} --")
            maes, l2s, pdes, kz_errs, times = [], [], [], [], []

            for seed in range(n_runs):
                np.random.seed(seed)
                torch.manual_seed(seed)

                if model_type == "classical":
                    net = ClassicalDNN()
                else:
                    net = HybridQNN(n_qubits=n_qubits, n_qlayers=n_qlayers)

                model = PINN(bc_type=bc, inverse=inv, network=net,
                             w_pde=cfg["w_pde"])
                elapsed = model.train(adam_steps=cfg["adam_steps"],
                                      lbfgs_iter=cfg["lbfgs_iter"])
                res     = model.evaluate()
                pde_r   = model.mean_pde_residual()

                maes.append(res["mae"])
                l2s.append(res["l2"])
                pdes.append(pde_r)
                times.append(elapsed)
                if inv and res["kz_err"] is not None:
                    kz_errs.append(res["kz_err"])

                kz_str = f"  kz_err={res['kz_err']:.3e}" if inv else ""
                print(f"  seed={seed}  MAE={res['mae']:.3e}  L2={res['l2']:.3e}"
                      f"  PDE={pde_r:.3e}  t={elapsed:.1f}s{kz_str}")

                if best[model_type] is None or res["l2"] < min(l2s[:-1], default=1e9):
                    best[model_type] = res
                    histories[model_type] = model.loss_history

            metrics[model_type] = dict(
                mae_mean=np.mean(maes), mae_std=np.std(maes),
                l2_mean=np.mean(l2s),   l2_std=np.std(l2s),
                pde_mean=np.mean(pdes), pde_std=np.std(pdes),
                time_mean=np.mean(times),
                kz_mean=np.mean(kz_errs) if kz_errs else None,
                kz_std=np.std(kz_errs)  if kz_errs else None,
            )

        # ── Print comparison ──────────────────────────────────────────────────
        print(f"\n  ── {name}: classical vs quantum ──")
        for mt in ["classical", "quantum"]:
            m = metrics[mt]
            print(f"  {mt:10s}  MAE={m['mae_mean']:.3e}±{m['mae_std']:.3e}"
                  f"  L2={m['l2_mean']:.3e}±{m['l2_std']:.3e}"
                  f"  PDE={m['pde_mean']:.3e}  t={m['time_mean']:.1f}s")

        # ── Save plots ────────────────────────────────────────────────────────
        for mt in ["classical", "quantum"]:
            plot_result(best[mt], f"{name} ({mt})",
                        f"{out_dir}/{name}_{mt}.png")

        if histories["classical"] and histories["quantum"]:
            plot_comparison(histories["classical"], histories["quantum"],
                            f"{out_dir}/{name}_loss_curve.png")

        # ── Collect CSV row ───────────────────────────────────────────────────
        for mt in ["classical", "quantum"]:
            m = metrics[mt]
            row = dict(scenario=name, model=mt,
                       mae_mean=m["mae_mean"], mae_std=m["mae_std"],
                       l2_mean=m["l2_mean"],   l2_std=m["l2_std"],
                       pde_mean=m["pde_mean"], pde_std=m["pde_std"],
                       time_mean=m["time_mean"],
                       kz_mean=m["kz_mean"] or "", kz_std=m["kz_std"] or "")
            all_rows.append(row)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = f"{out_dir}/results_{timestamp}.csv"
    fields = ["scenario", "model", "mae_mean", "mae_std",
              "l2_mean", "l2_std", "pde_mean", "pde_std",
              "time_mean", "kz_mean", "kz_std"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    print(f"\n\nResults saved to {csv_path}")

    # ── Final summary table ───────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  FINAL COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Scenario':<16} {'Model':<12} {'MAE':>12} {'L2':>12} {'Time(s)':>10}")
    print("-" * 64)
    for r in all_rows:
        print(f"{r['scenario']:<16} {r['model']:<12} "
              f"{r['mae_mean']:.3e}  {r['l2_mean']:.3e}  {r['time_mean']:>9.1f}")

    return all_rows


if __name__ == "__main__":
    # Quick smoke test: 1 run, small quantum circuit
    # For full comparison matching the paper, use n_runs=10
    run_comparison(n_runs=1, n_qubits=6, n_qlayers=3)
