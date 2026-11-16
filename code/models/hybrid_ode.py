"""
Physics-Informed Hybrid ODE & Universal Differential Equation (UDE) Model
for Microalgae-Bacteria POME Remediation under Data Scarcity.

Implements the complete methodology using verified bioprocess physics (`bioprocess_core.py`):
1. Benson & Krause (1984) Dissolved Oxygen Saturation + UNESCO PSS-78 salinity & pressure corrections
2. Emerson et al. (1975) NH3/NH4+ chemical equilibrium mapping sensor free NH3 <-> internal state TAN
3. HybridODE (Mechanistic / UDE / UDE + Ammonification):
   - Photometric Lux -> PAR PPFD conversion (Thimijan & Heins, 1983)
   - Beer-Lambert self-shading light attenuation across 35 cm column
   - Rosso et al. (1993) Cardinal Temperature Model with Inflection (CTMI)
   - Monod nutrient uptake + Azov & Goldman (1982) free-NH3 inhibition
   - Lumped POME bacterial COD degradation & ASCE/Metcalf-Eddy kLa Arrhenius scaling (theta=1.024)
   - Bounded parameter learning (+/-1.5 decade via tanh-log10 mapping)
   - Zero-initialized small neural network (4 -> 16 -> 16 -> 1) with bounded exp(tanh(.)) residual
   - RK4 integration (dt = 1h) with exact pump dosing dilution mass balance (V_ML=2500 mL = 2.5 L)
4. 72h Sliding Window extraction aligned to 24h lab points & 24h Lab Masking for sparse OD680 measurements.
"""

import os
import sys
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    HYBRID_ODE_STATES as STATES,
    HYBRID_ODE_SHORT as SHORT,
    Q_REF,
    V_ML,
    C_MOL,
    X_INO,
    DEPTH_CM,
    HYBRID_ODE_PARAMS as PARAMS,
    HYBRID_ODE_SPAN as SPAN,
    LOCO_WINDOW_HOURS,
    LOCO_STRIDE_HOURS,
    LAB_MEASUREMENT_INTERVAL_HOURS,
)
from models.bioprocess_core import (
    salinity_from_tds,
    do_saturation_mgL,
    free_nh3_fraction,
    free_nh3_fraction_torch,
    lux_to_ppfd_torch,
    f_light_steele_torch,
    ctmi_cardinal_torch,
    kla_temp_factor_torch,
)


def free_fraction(T: Union[torch.Tensor, np.ndarray, float], pH: Union[torch.Tensor, np.ndarray, float]):
    """Unified unionized free ammonia (NH3) fraction relative to TAN (Emerson et al., 1975)."""
    if isinstance(T, torch.Tensor) or isinstance(pH, torch.Tensor):
        t_t = T if isinstance(T, torch.Tensor) else torch.as_tensor(T, dtype=torch.float32)
        p_t = pH if isinstance(pH, torch.Tensor) else torch.as_tensor(pH, dtype=torch.float32)
        return free_nh3_fraction_torch(t_t, p_t)
    return free_nh3_fraction(T, pH)


def to_state(obs: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    Convert sensor observation [X, COD, free NH3, DO] -> internal ODE state [X, COD, TAN, DO].
    """
    phi = free_fraction(u[..., 1], u[..., 2])
    return torch.stack([obs[..., 0], obs[..., 1], obs[..., 2] / phi, obs[..., 3]], -1)


def observe(state: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    Convert internal ODE state [X, COD, TAN, DO] -> sensor observation [X, COD, free NH3, DO].
    Inverse of `to_state`.
    """
    phi = free_fraction(u[..., 1], u[..., 2])
    return torch.stack([state[..., 0], state[..., 1], state[..., 2] * phi, state[..., 3]], -1)


# ==============================================================================
# 3. Hybrid ODE Module (Mechanistic / UDE / UDE + Ammonification)
# ==============================================================================
class HybridODE(nn.Module):
    """
    Physics-informed Hybrid ODE / Universal Differential Equation (UDE) for
    microalgae-bacterial POME cultivation.

    Modes:
      - mode="mech", amm=False: Pure mechanistic ODE with bounded learnable parameters
      - mode="ude",  amm=False: Mechanistic ODE + zero-initialized NN multiplicative growth correction
      - mode="ude",  amm=True : UDE + bacterial ammonification term (+ k_amm * r_S in dN/dt)
    """

    def __init__(self, mode: str = "mech", amm: bool = False):
        super().__init__()
        self.mode = mode
        self.amm = amm
        self.names = list(PARAMS)
        self.register_buffer("init", torch.tensor([PARAMS[k][0] for k in self.names], dtype=torch.float32))
        self.register_buffer("learn", torch.tensor([float(PARAMS[k][1]) for k in self.names], dtype=torch.float32))
        self.raw = nn.Parameter(torch.zeros(len(self.names), dtype=torch.float32))
        self.nn = None
        if mode == "ude":
            self.nn = nn.Sequential(
                nn.Linear(4, 16),
                nn.Tanh(),
                nn.Linear(16, 16),
                nn.Tanh(),
                nn.Linear(16, 1),
            )
            # Zero-initialize output layer so UDE starts identical to pure mechanistic model
            nn.init.zeros_(self.nn[-1].weight)
            nn.init.zeros_(self.nn[-1].bias)

    def params(self) -> Dict[str, torch.Tensor]:
        """Return bounded physical parameters within ±SPAN decades of initial values."""
        v = self.init * (10.0 ** (SPAN * torch.tanh(self.raw) * self.learn))
        return dict(zip(self.names, v))

    def rhs(self, y: torch.Tensor, u: torch.Tensor, p: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Right-hand side time derivatives d[X, S, N, O]/dt (per hour).
        y: [X (AU), S (COD mg/L), N (TAN mg/L), O (DO mg/L)]
        u: [I (Lux), T (°C), pH, Q (L/s), Osat (mg/L)]
        """
        X, S, N, O = y.clamp(min=0.0).unbind(-1)
        I, T, pH, Q, Osat = u.unbind(-1)

        # Photometric lux -> PAR PPFD (Thimijan & Heins, 1983) + Beer-Lambert self-shading
        i_ppfd = lux_to_ppfd_torch(I)
        a = (p["k_par"] * np.log(10.0) * X * DEPTH_CM).clamp(min=1e-6)
        I_avg = i_ppfd * -torch.expm1(-a) / a  # Depth-averaged PPFD (umol m^-2 s^-1)
        f_I = f_light_steele_torch(I_avg, p["I_opt"])

        # Cardinal Temperature Model with Inflection (Rosso et al., 1993)
        f_T = ctmi_cardinal_torch(T, p["T_min"], p["T_opt"], p["T_max"])

        # Emerson free NH3 fraction & Monod-Haldane specific growth rate
        phi = free_nh3_fraction_torch(T, pH)
        mu = (
            p["mu_max"]
            * f_I
            * f_T
            * (N / (p["K_N"] + N))
            * (p["K_iA"] / (p["K_iA"] + phi * N))
        )

        # Universal Differential Equation (UDE) bounded multiplicative correction
        if self.nn is not None:
            feat = torch.stack([I_avg / 15.0, (T - 30.0) / 5.0, N / 50.0, pH - 9.0], -1)
            mu = mu * torch.exp(torch.tanh(self.nn(feat).squeeze(-1)))

        # Lumped bacterial COD degradation rate r_S & Arrhenius oxygen mass transfer kLa
        r_S = p["k_S"] * (S / (p["K_S"] + S)) * (O / (p["K_O"] + O))
        kLa = (p["kLa0"] + p["kLa_a"] * (Q / Q_REF)) * kla_temp_factor_torch(T)

        dX = (mu - p["k_d"]) * X
        dS = -r_S
        dN = -p["Y_N"] * mu * X - p["k_strip"] * phi * N
        if self.amm:
            dN = dN + p["k_amm"] * r_S
        dO = p["Y_O"] * mu * X - p["k_r"] * X - p["Y_OS"] * r_S + kLa * (Osat - O)
        return torch.stack([dX, dS, dN, dO], -1)

    def forward(self, y0: torch.Tensor, u: torch.Tensor, imp: torch.Tensor) -> torch.Tensor:
        """
        Integrate state trajectory using 4th-order Runge-Kutta (RK4, dt = 1 hour)
        with discrete pump dosing dilution at each hourly step.
        y0: (B, 4) initial internal state [X, COD, TAN, DO]
        u:  (B, L, 5) exogenous inputs [I, T, pH, Q, Osat]
        imp: (B, L, 3) pump dosing impulses [v_bic, v_mol, v_ino] in mL
        Returns: (B, L, 4) state trajectory
        """
        p = self.params()
        y, out = y0, [y0]
        for h in range(u.shape[1] - 1):
            # Dosing during hour h takes effect at hour h+1; stock DO assumed equal to reactor DO
            v_bic, v_mol, v_ino = imp[:, h].unbind(-1)
            tot = V_ML + v_bic + v_mol + v_ino
            keep = V_ML / tot
            y = torch.stack(
                [
                    y[:, 0] * keep + X_INO * v_ino / tot,
                    y[:, 1] * keep + C_MOL * v_mol / tot,
                    y[:, 2] * keep,
                    y[:, 3],
                ],
                -1,
            )
            uh = u[:, h]
            k1 = self.rhs(y, uh, p)
            k2 = self.rhs(y + 0.5 * k1, uh, p)
            k3 = self.rhs(y + 0.5 * k2, uh, p)
            k4 = self.rhs(y + k3, uh, p)
            y = (y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0).clamp(min=0.0)
            out.append(y)
        return torch.stack(out, 1)


# ==============================================================================
# 4. Data Preparation, 72h Sliding Windows, 24h Lab Masking & Training
# ==============================================================================
def culture_arrays(df: pd.DataFrame, k: int) -> List[torch.Tensor]:
    """
    Extract (y, u, imp) tensors for cultivation k from hourly telemetry DataFrame.
    y:   (L, 4) sensor observations [OD680, COD, NH3, DO]
    u:   (L, 5) environmental drivers [Light, Temp_water, pH, Aerator_RU, DO_sat]
    imp: (L, 3) dosing volumes [Pump1_NaHCO3, Pump2_Molase, Pump3_Microalgae] in mL
    """
    g = df[df["kultivasi"] == k].sort_values("timestamp")
    T = g["1_temp_air(C)"].to_numpy(float)
    osat = do_saturation_mgL(
        T,
        g["2_pressure(hPa)"].to_numpy(float),
        salinity_from_tds(g["1_tds(ppm)"].to_numpy(float)),
    )
    u = np.column_stack(
        [g["2_light(Lux)"], T, g["1_ph"], g["Aerator_RU (L/s)"], osat]
    )
    imp = g[
        ["Pump1_NaHCO3 (mL)", "Pump2_Molase (mL)", "Pump3_Microalgae (mL)"]
    ].to_numpy(float)
    y = g[STATES].to_numpy(float)
    return [torch.tensor(a, dtype=torch.float32) for a in (y, u, imp)]


def windows(
    arrs: List[List[torch.Tensor]],
    W: int = LOCO_WINDOW_HOURS,
    stride: int = LOCO_STRIDE_HOURS,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Slice trajectories into sliding windows of length W (default 72 hours),
    starting every `stride` hours (default 24 hours) so relative hour 0 aligns with daily lab points.
    """
    ys, us, ims = [], [], []
    for y, u, imp in arrs:
        for s in range(0, len(y) - W + 1, stride):
            ys.append(y[s : s + W])
            us.append(u[s : s + W])
            ims.append(imp[s : s + W])
    return torch.stack(ys), torch.stack(us), torch.stack(ims)


def lab_mask(L: int, start: int = 0, interval: int = LAB_MEASUREMENT_INTERVAL_HOURS) -> torch.Tensor:
    """
    Boolean mask marking true laboratory measurement points (every 24 hours from H0).
    Prevents interpolated hourly OD680 values from corrupting biomass loss/evaluation.
    """
    return (torch.arange(start, start + L) % interval) == 0


def loss_fn(pred: torch.Tensor, obs: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Normalized MSE loss across the 4 states:
    - State 0 (OD680 / X) is evaluated strictly at 24-hour laboratory measurement timestamps (`lab_mask`)
    - States 1, 2, 3 (COD, NH3, DO) are evaluated across all hourly sensor timestamps.
    """
    err = ((pred - obs) / scale.to(pred.device)) ** 2  # (B, L, 4)
    m = lab_mask(obs.shape[1]).to(pred.device)
    return torch.stack([err[:, m, 0].mean()] + [err[..., j].mean() for j in (1, 2, 3)]).mean()


def train_hybrid_ode(
    model: HybridODE,
    arrs: List[List[torch.Tensor]],
    scale: torch.Tensor,
    epochs: int = 200,
    W: int = LOCO_WINDOW_HOURS,
    lr: float = 1e-2,
    log_every: int = 25,
    device: Optional[torch.device] = None,
) -> HybridODE:
    """
    Train HybridODE on 72-hour sliding windows using Adam and lab-masked NRMSE loss.
    """
    dev = device if device is not None else next(model.parameters()).device
    model = model.to(dev)
    y, u, imp = [t.to(dev) for t in windows(arrs, W)]
    scale_dev = scale.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        opt.zero_grad()
        pred_obs = observe(model(to_state(y[:, 0], u[:, 0]), u, imp), u)
        loss = loss_fn(pred_obs, y, scale_dev)
        if not torch.isfinite(loss):
            raise RuntimeError(f"loss tidak hingga di epoch {ep}; turunkan --lr")
        loss.backward()
        opt.step()
        if log_every > 0 and (ep % log_every == 0 or ep == epochs - 1):
            print(f"epoch {ep:4d}  loss {loss.item():.4f}")
    return model


def selfcheck(arrs: List[List[torch.Tensor]]) -> bool:
    """
    Run physics and numerical consistency self-checks (from hybrid_ode_loco.ipynb):
    1. `to_state` and `observe` are exact inverses
    2. Zero-initialized UDE is numerically identical to pure mechanistic ODE
    3. Trajectories are finite and non-negative
    4. Pure dosing dilution mass balance check (300 mL molasses into V_ML=2500 mL at 0 COD -> C_MOL*300/2800)
    """
    y, u, imp = windows(arrs[:2], 48)
    torch.manual_seed(0)
    mech, ude = HybridODE("mech"), HybridODE("ude")
    y0 = to_state(y[:, 0], u[:, 0])
    assert torch.allclose(observe(y0, u[:, 0]), y[:, 0], atol=1e-5), "to_state/observe harus saling balik"
    a, b = mech(y0, u, imp), ude(y0, u, imp)
    assert torch.allclose(a, b, atol=1e-5), "UDE dengan NN nol harus identik dengan mekanistik"
    assert torch.isfinite(a).all() and (a >= 0).all(), "lintasan harus hingga & non-negatif"

    # Pure dosing dilution check (without kinetics)
    m = HybridODE("mech")
    z = torch.zeros(1, 2, 5)
    imp1 = torch.tensor([[[0.0, 300.0, 0.0], [0.0, 0.0, 0.0]]])
    s = m(torch.tensor([[0.0, 0.0, 0.0, 0.0]]), z, imp1)[0, 1, 1].item()
    assert abs(s - C_MOL * 300.0 / (V_ML + 300.0)) < 1e-2, f"Dilution mismatch: {s}"
    print("self-check OK")
    return True


@torch.no_grad()
def evaluate_one(
    model: HybridODE,
    arrs: List[torch.Tensor],
    scale: torch.Tensor,
) -> List[Dict[str, Union[str, float]]]:
    """
    Full trajectory rollout from hour 0 on a held-out cultivation.
    NRMSE = RMSE / scale. OD680 (X) is evaluated strictly at 24h lab timestamps.
    """
    dev = next(model.parameters()).device
    y, u, imp = [t.to(dev) for t in arrs]
    pred = observe(model(to_state(y[None, 0], u[None, 0]), u[None], imp[None]), u[None])[0].cpu()
    y_cpu = y.cpu()
    scale_cpu = scale.cpu()

    rows = []
    for j, name in enumerate(SHORT):
        m = lab_mask(len(y_cpu)) if j == 0 else torch.ones(len(y_cpu), dtype=torch.bool)
        rmse = torch.sqrt(((pred[m, j] - y_cpu[m, j]) ** 2).mean()).item()
        rows.append({"state": name, "rmse": rmse, "nrmse": rmse / scale_cpu[j].item()})
    return rows

