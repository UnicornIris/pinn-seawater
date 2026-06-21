"""
PINN for Seawater Temperature Diffusion Equation
Replication of: Han et al. (2026), Journal of Oceanology and Limnology

PDE:  dT/dt = kz * d²T/dz²
Domain: z ∈ [0, 2], t ∈ [0, 5]
Network: 2-input → 5 hidden layers (32 neurons, tanh) → 1-output
Optimizer: L-BFGS, lr=0.001
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cpu")

# ── Domain ────────────────────────────────────────────────────────────────────
Z_MIN, Z_MAX = 0.0, 2.0
T_MIN, T_MAX = 0.0, 5.0
KZ_TRUE = 0.05          # true diffusion coefficient
DT_STEP = 0.05          # time step (same as paper)


# ── Analytical solutions ──────────────────────────────────────────────────────
def T_dirichlet(z, t):
    """Eq.14: T = exp(-0.05*(π/2)²t)sin(πz/2) + exp(-0.05π²t)sin(πz)"""
    return (np.exp(-KZ_TRUE * (np.pi / 2) ** 2 * t) * np.sin(np.pi * z / 2)
            + np.exp(-KZ_TRUE * np.pi ** 2 * t) * np.sin(np.pi * z))


def T_neumann(z, t):
    """Neumann general solution (cos-series)"""
    return (np.exp(-KZ_TRUE * (np.pi / 2) ** 2 * t) * np.cos(np.pi * z / 2)
            + np.exp(-KZ_TRUE * np.pi ** 2 * t) * np.cos(np.pi * z))


def T_robin(z, t):
    """Eq.18"""
    return (np.exp(KZ_TRUE * t - z)
            + np.exp(-KZ_TRUE * (np.pi / 2) ** 2 * t)
              * (np.cos(np.pi * z / 2) - np.sin(np.pi * z / 2))
            + np.exp(-KZ_TRUE * np.pi ** 2 * t)
              * (np.pi * np.cos(np.pi * z) - np.sin(np.pi * z)))


# ── Neural Network ────────────────────────────────────────────────────────────
class DNN(nn.Module):
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


# ── PINN ─────────────────────────────────────────────────────────────────────
class PINN:
    """
    Solves forward (kz known) or inverse (kz unknown, estimated from data) problem.
    bc_type: 'dirichlet' | 'neumann' | 'robin'
    inverse:  if True, kz becomes a learnable parameter
    """

    def __init__(self, bc_type="dirichlet", inverse=False,
                 w_pde=1.0, w_bc=1.0, w_ic=1.0, w_data=1.0):
        self.bc_type = bc_type
        self.inverse = inverse
        self.w_pde = w_pde
        self.w_bc = w_bc
        self.w_ic = w_ic
        self.w_data = w_data

        self.net = DNN().to(DEVICE)
        if inverse:
            # kz is a learnable log-parameter for positivity
            self.log_kz = nn.Parameter(torch.tensor([np.log(0.1)]))
        else:
            self.log_kz = None

        self._build_training_data()

    # ── helpers ──────────────────────────────────────────────────────────────
    @property
    def kz(self):
        if self.inverse:
            return torch.exp(self.log_kz)
        return torch.tensor(KZ_TRUE, dtype=torch.float32)

    def _T_exact(self, z_np, t_np):
        if self.bc_type == "dirichlet":
            return T_dirichlet(z_np, t_np)
        elif self.bc_type == "neumann":
            return T_neumann(z_np, t_np)
        else:
            return T_robin(z_np, t_np)

    def _tensor(self, z, t, requires_grad=False):
        zt = np.stack([z, t], axis=1).astype(np.float32)
        return torch.tensor(zt, requires_grad=requires_grad)

    def _build_training_data(self):
        # Collocation points (PDE residual) — 2 540 points
        z_f = np.random.uniform(Z_MIN, Z_MAX, 2540)
        t_f = np.random.uniform(T_MIN, T_MAX, 2540)
        self.zt_f = self._tensor(z_f, t_f, requires_grad=True)

        # Boundary points — 80 total (40 per side)
        t_bc = np.random.uniform(T_MIN, T_MAX, 40)
        z_bc0 = np.zeros(40)
        z_bcL = np.full(40, Z_MAX)
        z_bc = np.concatenate([z_bc0, z_bcL])
        t_bc = np.concatenate([t_bc, np.random.uniform(T_MIN, T_MAX, 40)])
        T_bc = self._T_exact(z_bc, t_bc).astype(np.float32)
        self.zt_bc = self._tensor(z_bc, t_bc, requires_grad=(self.bc_type != "dirichlet"))
        self.T_bc = torch.tensor(T_bc).unsqueeze(1)

        # Initial condition points — 160 points
        z_ic = np.random.uniform(Z_MIN, Z_MAX, 160)
        t_ic = np.zeros(160)
        T_ic = self._T_exact(z_ic, t_ic).astype(np.float32)
        self.zt_ic = self._tensor(z_ic, t_ic)
        self.T_ic = torch.tensor(T_ic).unsqueeze(1)

        # Observation data for inverse problem — 12 uniformly distributed points
        if self.inverse:
            z_d = np.linspace(Z_MIN, Z_MAX, 4)
            t_d = np.linspace(T_MIN, T_MAX, 3)
            zz, tt = np.meshgrid(z_d, t_d)
            z_data = zz.ravel()
            t_data = tt.ravel()
            T_data = self._T_exact(z_data, t_data).astype(np.float32)
            self.zt_data = self._tensor(z_data, t_data)
            self.T_data = torch.tensor(T_data).unsqueeze(1)

    # ── PDE residual ─────────────────────────────────────────────────────────
    def _pde_residual(self):
        zt = self.zt_f
        zt.requires_grad_(True)
        T_pred = self.net(zt)

        dT = torch.autograd.grad(T_pred, zt, torch.ones_like(T_pred),
                                 create_graph=True)[0]
        dT_dz = dT[:, 0:1]
        dT_dt = dT[:, 1:2]

        d2T = torch.autograd.grad(dT_dz, zt, torch.ones_like(dT_dz),
                                  create_graph=True)[0]
        d2T_dz2 = d2T[:, 0:1]

        return dT_dt - self.kz * d2T_dz2

    # ── Boundary residual ─────────────────────────────────────────────────────
    def _bc_residual(self):
        if self.bc_type == "dirichlet":
            T_pred = self.net(self.zt_bc)
            return T_pred - self.T_bc

        elif self.bc_type == "neumann":
            zt = self.zt_bc.clone().requires_grad_(True)
            T_pred = self.net(zt)
            dT = torch.autograd.grad(T_pred, zt, torch.ones_like(T_pred),
                                     create_graph=True)[0]
            dT_dz = dT[:, 0:1]
            return dT_dz   # gradient = 0 (adiabatic)

        else:  # robin
            zt = self.zt_bc.clone().requires_grad_(True)
            T_pred = self.net(zt)
            dT = torch.autograd.grad(T_pred, zt, torch.ones_like(T_pred),
                                     create_graph=True)[0]
            dT_dz = dT[:, 0:1]
            # dT/dz + T = 0 at z=0;  dT/dz - T = 0 at z=L
            z_vals = zt[:, 0:1]
            sign = torch.where(z_vals < 1e-6,
                               torch.ones_like(z_vals),
                               -torch.ones_like(z_vals))
            return dT_dz + sign * T_pred

    # ── Total loss ────────────────────────────────────────────────────────────
    def loss(self):
        r_pde = self._pde_residual()
        r_bc = self._bc_residual()
        r_ic = self.net(self.zt_ic) - self.T_ic

        L_pde = self.w_pde * torch.mean(r_pde ** 2)
        L_bc  = self.w_bc  * torch.mean(r_bc  ** 2)
        L_ic  = self.w_ic  * torch.mean(r_ic  ** 2)
        L = L_pde + L_bc + L_ic

        if self.inverse:
            r_data = self.net(self.zt_data) - self.T_data
            L += self.w_data * torch.mean(r_data ** 2)

        return L

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self, adam_steps=5000, lbfgs_iter=10000):
        params = list(self.net.parameters())
        if self.inverse:
            params.append(self.log_kz)

        self.loss_history = []
        t0 = time.time()

        # Phase 1: Adam pre-training
        adam = torch.optim.Adam(params, lr=1e-3)
        for _ in range(adam_steps):
            adam.zero_grad()
            L = self.loss()
            L.backward()
            adam.step()
            self.loss_history.append(L.item())

        # Phase 2: L-BFGS fine-tuning
        lbfgs = torch.optim.LBFGS(
            params, lr=1.0,
            max_iter=lbfgs_iter,
            tolerance_grad=1e-9,
            tolerance_change=1e-11,
            history_size=100,
            line_search_fn="strong_wolfe",
        )

        def closure():
            lbfgs.zero_grad()
            L = self.loss()
            L.backward()
            self.loss_history.append(L.item())
            return L

        lbfgs.step(closure)
        return time.time() - t0

    # ── Evaluation ────────────────────────────────────────────────────────────
    def evaluate(self, n_test=200):
        z_t = np.linspace(Z_MIN, Z_MAX, n_test)
        t_t = np.linspace(T_MIN, T_MAX, n_test)
        zz, tt = np.meshgrid(z_t, t_t)

        zt_test = torch.tensor(
            np.stack([zz.ravel(), tt.ravel()], axis=1), dtype=torch.float32
        )
        with torch.no_grad():
            T_pred = self.net(zt_test).numpy().reshape(n_test, n_test)

        T_true = self._T_exact(zz, tt)
        mae  = np.mean(np.abs(T_pred - T_true))
        l2   = np.linalg.norm(T_pred - T_true) / (np.linalg.norm(T_true) + 1e-12)
        kz_err = None
        if self.inverse:
            kz_est = float(torch.exp(self.log_kz).detach())
            kz_err = abs(kz_est - KZ_TRUE)
        return dict(mae=mae, l2=l2, T_pred=T_pred, T_true=T_true,
                    z=z_t, t=t_t, kz_err=kz_err)


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_result(result, title, out_path):
    z, t = result["z"], result["t"]
    T_pred, T_true = result["T_pred"], result["T_true"]
    error = np.abs(T_pred - T_true)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, data, label in zip(axes,
                                [T_true, T_pred, error],
                                ["True T", "PINN T", "|Error|"]):
        im = ax.contourf(z, t, data, levels=50, cmap="RdBu_r")
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("z")
        ax.set_ylabel("t")
        ax.set_title(label)
    fig.suptitle(f"{title}  MAE={result['mae']:.3e}  L2={result['l2']:.3e}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Saved: {out_path}")


# ── Run all 6 scenarios ───────────────────────────────────────────────────────
def run_all(n_runs=10):
    import os
    out_dir = "pinn_results"
    os.makedirs(out_dir, exist_ok=True)

    scenarios = [
        ("dirichlet", False, "DE_Dir_FWD"),
        ("neumann",   False, "DE_Neum_FWD"),
        ("robin",     False, "DE_Rob_FWD"),
        ("dirichlet", True,  "DE_Dir_INV"),
        ("neumann",   True,  "DE_Neum_INV"),
        ("robin",     True,  "DE_Rob_INV"),
    ]

    # Paper Table 1 & 3 reference values (mean ± std, ×10⁻⁵ for PDE/L2, ×10⁻⁶ for MAE)
    paper_ref = {
        "DE_Dir_FWD":  dict(pde=3.6585e-5, pde_s=0.7259e-5, l2=1.9540e-5, l2_s=0.2034e-5, mae=5.5912e-6, mae_s=1.5139e-6),
        "DE_Neum_FWD": dict(pde=3.1366e-5, pde_s=0.9654e-5, l2=1.7306e-5, l2_s=0.5565e-5, mae=5.3412e-6, mae_s=2.8130e-6),
        "DE_Rob_FWD":  dict(pde=17.135e-5, pde_s=6.0176e-5, l2=3.4923e-5, l2_s=1.5377e-5, mae=25.697e-6, mae_s=19.063e-6),
        "DE_Dir_INV":  dict(pde=None, l2=None, mae=10.163e-6, mae_s=None, kz_err=None),
        "DE_Neum_INV": dict(pde=None, l2=None, mae=None,      mae_s=None, kz_err=None),
        "DE_Rob_INV":  dict(pde=None, l2=None, mae=20.207e-6, mae_s=None, kz_err=None),
    }

    all_summary = []
    for bc, inv, name in scenarios:
        print(f"\n{'='*60}")
        print(f"  {name}  (bc={bc}, inverse={inv})  [{n_runs} runs]")
        print(f"{'='*60}")

        maes, l2s, pde_res, kz_errs = [], [], [], []
        best_res = None

        for seed in range(n_runs):
            np.random.seed(seed)
            torch.manual_seed(seed)
            model = PINN(bc_type=bc, inverse=inv)
            elapsed = model.train(adam_steps=5000, lbfgs_iter=10000)
            res = model.evaluate()

            # compute mean PDE residual on collocation points
            with torch.enable_grad():
                zt_eval = model.zt_f.detach().clone().requires_grad_(True)
                T_eval = model.net(zt_eval)
                dT = torch.autograd.grad(T_eval, zt_eval, torch.ones_like(T_eval),
                                         create_graph=True)[0]
                dT_dz = dT[:, 0:1]
                dT_dt = dT[:, 1:2]
                d2T = torch.autograd.grad(dT_dz, zt_eval, torch.ones_like(dT_dz),
                                          create_graph=False)[0]
                d2T_dz2 = d2T[:, 0:1]
                r_pde = (dT_dt - model.kz * d2T_dz2).detach().numpy()
            mean_pde = float(np.mean(np.abs(r_pde)))

            maes.append(res["mae"])
            l2s.append(res["l2"])
            pde_res.append(mean_pde)
            if inv:
                kz_errs.append(res["kz_err"])

            kz_info = (f"  kz_err={res['kz_err']:.3e}" if inv else "")
            print(f"  seed={seed}  MAE={res['mae']:.3e}  L2={res['l2']:.3e}"
                  f"  PDE={mean_pde:.3e}  t={elapsed:.1f}s{kz_info}")

            if best_res is None or res["l2"] < l2s[np.argmin(l2s)]:
                best_res = res

        mae_mean, mae_std = np.mean(maes),   np.std(maes)
        l2_mean,  l2_std  = np.mean(l2s),    np.std(l2s)
        pde_mean, pde_std = np.mean(pde_res), np.std(pde_res)

        print(f"\n  ── {name} statistics over {n_runs} runs ──")
        print(f"  Mean PDE residual : {pde_mean:.4e} ± {pde_std:.4e}")
        print(f"  L2 relative error : {l2_mean:.4e}  ± {l2_std:.4e}")
        print(f"  MAE               : {mae_mean:.4e} ± {mae_std:.4e}")
        if inv and kz_errs:
            kz_mean, kz_std = np.mean(kz_errs), np.std(kz_errs)
            print(f"  kz abs error      : {kz_mean:.4e} ± {kz_std:.4e}")

        ref = paper_ref.get(name, {})
        if ref.get("pde") is not None:
            print(f"\n  ── vs Paper Table 1 ──")
            print(f"  PDE  ours: {pde_mean:.4e}±{pde_std:.4e}  paper: {ref['pde']:.4e}±{ref['pde_s']:.4e}")
            print(f"  L2   ours: {l2_mean:.4e}±{l2_std:.4e}  paper: {ref['l2']:.4e}±{ref['l2_s']:.4e}")
            print(f"  MAE  ours: {mae_mean:.4e}±{mae_std:.4e}  paper: {ref['mae']:.4e}±{ref['mae_s']:.4e}")

        # save plot of the best run
        plot_result(best_res, name, f"{out_dir}/{name}.png")

        row = dict(name=name,
                   pde_mean=pde_mean, pde_std=pde_std,
                   l2_mean=l2_mean,   l2_std=l2_std,
                   mae_mean=mae_mean, mae_std=mae_std)
        if inv and kz_errs:
            row["kz_mean"] = np.mean(kz_errs)
            row["kz_std"]  = np.std(kz_errs)
        all_summary.append(row)

    # ── Final summary table ──
    sep = "=" * 80
    print(f"\n\n{sep}")
    print(f"  FINAL SUMMARY  (mean ± std over {n_runs} runs)")
    print(sep)
    print(f"{'Instance':<18} {'PDE res (mean±std)':>26} {'L2 rel err (mean±std)':>26} {'MAE (mean±std)':>26}")
    print("-" * 98)
    for r in all_summary:
        pde_s = f"{r['pde_mean']:.3e}±{r['pde_std']:.3e}"
        l2_s  = f"{r['l2_mean']:.3e}±{r['l2_std']:.3e}"
        mae_s = f"{r['mae_mean']:.3e}±{r['mae_std']:.3e}"
        print(f"{r['name']:<18} {pde_s:>26} {l2_s:>26} {mae_s:>26}")
        if "kz_mean" in r:
            print(f"  {'kz abs err':>16}: {r['kz_mean']:.3e} ± {r['kz_std']:.3e}")


if __name__ == "__main__":
    run_all()
