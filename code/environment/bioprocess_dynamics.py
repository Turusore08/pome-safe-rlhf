"""
Bioprocess Dynamics & Reactor Kinetics for Gymnasium Closed-Loop Simulation.
Simulates a 2.5 L Chlorella vulgaris - bacteria POME photobioreactor using the unified
mechanistic bioprocess core (`models/bioprocess_core.py`).

NOTE: Offline RL training ingests real experimental telemetry & photos via `data/real_data_loader.py`
and calibrated corpora via `data/synthetic_generator.py`. This module executes closed-loop
evaluation & online safe RLHF fine-tuning rollouts in `POMEReactorEnv`.
"""

import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    SENSOR_COLUMNS,
    RAW_POME_INIT,
    ACTION_COLUMNS,
    DECISION_HORIZON_HOURS,
    REACTOR_VOLUME_LITERS,
    C_MOL,
    X_INO,
    DEPTH_CM,
    AERATOR_FLOW_LS,
)
from models.bioprocess_core import (
    MW_C,
    MW_N,
    MW_NAHCO3,
    DIFFUSIVITY_RATIO_CO2_O2_SQRT,
    lux_to_ppfd,
    beer_lambert_i_avg,
    salinity_from_tds,
    ionic_strength_from_tds,
    do_saturation_mgL,
    kla_temp_factor,
    free_nh3_fraction,
    carbonate_constants,
    carbonate_fractions,
    compute_conservative_charge_z,
    solve_ph,
    f_light_haldane_bernard,
    ctmi_cardinal_np,
)


class POMEBioprocessSimulator:
    """
    Mechanistic closed-loop POME microalgae-bacteria simulator (V_work = 2.5 L).
    Integrates hourly sub-steps within each RL decision horizon (`dt_hours = 6.0`).
    """

    def __init__(self, seed: int = 42, decision_horizon_hours: float = DECISION_HORIZON_HOURS):
        self.rng = np.random.RandomState(seed)
        self.decision_horizon_hours = float(decision_horizon_hours)
        self.last_photo_gain_gl: float = 0.0
        self._internal: Optional[Dict[str, float]] = None

    def reset_internal(self, state: Dict[str, float], biomass_gl: float) -> float:
        """Initialize geochemical & two-fraction COD reservoirs matching initial sensor state."""
        cod = float(state.get("cod", RAW_POME_INIT["cod"]))
        bod = float(state.get("bod", 0.50 * cod))
        # Split COD into fast/soluble (BOD-rich) and slow/recalcitrant fractions smoothly
        fast_frac = float(np.clip((bod / max(cod, 1e-3) - 0.25) / 0.50, 0.35, 0.72))
        cod_fast = cod * fast_frac
        cod_slow = max(0.0, cod - cod_fast)

        tds = float(state.get("total_dissolved_solids", RAW_POME_INIT["total_dissolved_solids"]))
        ph = float(state.get("ph", RAW_POME_INIT["ph"]))
        tan_mg = float(state.get("dissolved_ammonia", RAW_POME_INIT["dissolved_ammonia"]))
        temp_c = float(state.get("water_temperature", RAW_POME_INIT["water_temperature"]))
        tss = float(state.get("total_suspended_solids", RAW_POME_INIT["total_suspended_solids"]))

        ion = float(ionic_strength_from_tds(tds))
        tan_mol = (tan_mg / 1000.0) / MW_N
        dic_mol = float(np.clip(0.012 + 0.0008 * (tan_mg / 14.0), 0.012, 0.022))
        z_net = compute_conservative_charge_z(ph, dic_mol, tan_mol, temp_c, ion)

        detritus_mg = max(50.0, tss - biomass_gl * 1000.0 * 0.20)
        self._internal = {
            "cod_fast": cod_fast,
            "cod_slow": cod_slow,
            "dic_mol": dic_mol,
            "z_net": z_net,
            "detritus_mg": detritus_mg,
        }
        self.last_photo_gain_gl = 0.0
        return z_net * 1000.0

    def _sample_action(
        self,
        run_id: str,
        strategy: str,
        step: int,
        day: float,
        state: Dict[str, float],
        collapsed: bool,
    ) -> Dict[str, float]:
        """Sample heuristic reference action matching canonical ACTION_COLUMNS."""
        act = {col: 0.0 for col in ACTION_COLUMNS}
        if collapsed:
            return act
        if strategy == "hcr_k8":
            if state["ph"] < 7.4:
                act["sodium_bicarbonate"] = float(self.rng.uniform(0.5, 1.5))
            act["aeration"] = float(self.rng.uniform(3.5, 5.5))
            act["light_intensity"] = float(self.rng.uniform(1800.0, 3200.0))
            if step == 0:
                act["microalgae_inoculum"] = 40.0
        else:
            if state["ph"] < 7.2:
                act["sodium_bicarbonate"] = float(self.rng.uniform(0.5, 2.0))
            act["aeration"] = float(self.rng.uniform(3.0, 5.5))
            act["light_intensity"] = float(self.rng.uniform(1500.0, 3500.0))
        return act

    def _observe_state(
        self,
        state: Dict[str, float],
        step: int,
        noise_level: float,
        dt_hours: Optional[float] = None,
    ) -> Dict[str, float]:
        """Add realistic sensor noise and metadata."""
        horizon = dt_hours if dt_hours is not None else self.decision_horizon_hours
        obs = {"step": step, "hour": step * horizon}
        for col in SENSOR_COLUMNS:
            val = state[col]
            if noise_level > 0:
                val += self.rng.normal(0, noise_level * (abs(val) + 1.0))
            obs[col] = float(val)
        return obs

    def _step_kinetics(
        self,
        state: Dict[str, float],
        action: Dict[str, float],
        biomass: float,
        alkalinity: float,
        collapsed: bool,
        step: int,
        dt_hours: Optional[float] = None,
    ) -> Tuple[Dict[str, float], float, float]:
        """
        Advance bioprocess kinetics over `dt_hours` (integrated via 1h Runge-Kutta/semi-implicit sub-steps).
        `biomass` is in g/L dry microalgae; `self.last_photo_gain_gl` records photoautotrophic gain.
        """
        horizon = dt_hours if dt_hours is not None else self.decision_horizon_hours
        self.last_photo_gain_gl = 0.0
        new_state = dict(state)

        if self._internal is None or step <= 1 and abs(state["cod"] - (self._internal["cod_fast"] + self._internal["cod_slow"])) > 150.0:
            self.reset_internal(state, biomass)

        if collapsed:
            new_state["cod"] = min(RAW_POME_INIT["cod"], state["cod"] * 1.005)
            new_state["dissolved_oxygen"] = max(0.05, state["dissolved_oxygen"] * 0.85)
            biomass = max(0.01, biomass * 0.96)
            return new_state, biomass, alkalinity

        aeration_hours = float(np.clip(action.get("aeration", action.get("aeration_rate", 0.0)), 0.0, horizon))
        nahco3_g = float(max(0.0, action.get("sodium_bicarbonate", action.get("nahco3_dosage", 0.0))))
        inoculum_ml = float(max(0.0, action.get("microalgae_inoculum", 0.0)))
        molasses_ml = float(max(0.0, action.get("molasses", action.get("molasses_supplement", 0.0))))
        light_lux = float(max(0.0, action.get("light_intensity", 0.0)))

        # Reservoir state
        v_ml = REACTOR_VOLUME_LITERS * 1000.0
        cod_fast = self._internal["cod_fast"]
        cod_slow = self._internal["cod_slow"]
        dic_mol = self._internal["dic_mol"]
        z_net = self._internal["z_net"]
        detritus_mg = self._internal["detritus_mg"]

        tan_mg = float(max(0.1, state.get("dissolved_ammonia", 50.0)))
        do_val = float(max(0.05, state.get("dissolved_oxygen", 1.2)))
        tds = float(max(200.0, state.get("total_dissolved_solids", 2100.0)))
        ph_val = float(state.get("ph", 7.0))
        press_hpa = float(state.get("atmospheric_pressure", 1012.0))

        # 1. Instantaneous dosing & volume conservation (V_work = 2.5 L = 2500 mL)
        liq_added_ml = inoculum_ml + molasses_ml
        if liq_added_ml > 0.0:
            tot_ml = v_ml + liq_added_ml
            dil = v_ml / tot_ml
            cod_fast *= dil
            cod_slow *= dil
            biomass *= dil
            detritus_mg *= dil
            tan_mg *= dil
            dic_mol *= dil
            z_net *= dil
            tds *= dil
            # Inoculum concentration: X_INO * 1.266 g/L; Molasses COD: C_MOL mg/L
            biomass += (X_INO * 1.266) * (inoculum_ml / tot_ml)
            cod_fast += C_MOL * (molasses_ml / tot_ml)

        if nahco3_g > 0.0:
            added_mol_l = (nahco3_g / MW_NAHCO3) / REACTOR_VOLUME_LITERS
            dic_mol += added_mol_l
            z_net += added_mol_l
            tds += (nahco3_g * 1000.0) / REACTOR_VOLUME_LITERS

        # 2. Integrate kinetic ODEs across 1-hour sub-steps
        n_sub = max(1, int(round(horizon)))
        sub_h = horizon / n_sub
        sub_d = sub_h / 24.0
        q_ratio = aeration_hours / max(horizon, 1e-6)  # Fraction of window under full aerator Q_REF

        hour_base = (step - 1) * horizon
        photo_gain_total = 0.0
        temp_c = float(state.get("water_temperature", 30.0))

        for k in range(n_sub):
            curr_h = hour_base + (k + 0.5) * sub_h
            hod = curr_h % 24.0
            amb_temp = 29.5 + 2.8 * np.sin((hod - 10.0) * np.pi / 12.0)
            target_w_temp = amb_temp + (light_lux / 5000.0) * 1.5 - 0.5 * q_ratio
            temp_c += 0.25 * sub_h * (target_w_temp - temp_c)

            ion = float(ionic_strength_from_tds(tds))
            sal_ppt = float(salinity_from_tds(tds))
            do_sat = float(do_saturation_mgL(temp_c, press_hpa, sal_ppt))
            phi_nh3 = float(free_nh3_fraction(temp_c, ph_val, ion))
            free_nh3_mg = tan_mg * phi_nh3 * (17.031 / MW_N)

            # Beer-Lambert depth-averaged PPFD (lux -> umol m^-2 s^-1, Thimijan & Heins 1983)
            i_ppfd = float(lux_to_ppfd(light_lux))
            k_bg = 0.012 + 4.0e-6 * (cod_fast + cod_slow)
            i_avg = float(beer_lambert_i_avg(i_ppfd, biomass, k_par=0.055, depth_cm=DEPTH_CM, k_bg=k_bg))

            f_light = float(f_light_haldane_bernard(i_avg, i_opt_ppfd=12.0, beta_i=0.5))
            f_temp = float(ctmi_cardinal_np(temp_c, 10.0, 30.0, 40.0))
            f_ph = float(ctmi_cardinal_np(ph_val, 4.8, 8.2, 10.8))
            nut_lim = min(tan_mg / (tan_mg + 5.0), dic_mol / (dic_mol + 5e-5))
            nh3_inhib = 20.4 / (20.4 + free_nh3_mg)

            # Photoautotrophic specific growth rate (1/day) - strictly 0 in darkness (f_light == 0)
            mu_photo = 0.85 * f_light * f_temp * f_ph * nut_lim * nh3_inhib * max(0.0, 1.0 - biomass / 3.0)
            # Mixotrophic organic assimilation when readily-biodegradable COD is abundant
            mu_mixo = 0.25 * (cod_fast / (cod_fast + 500.0)) * (do_val / (do_val + 0.5)) * f_temp * f_ph * max(0.0, 1.0 - biomass / 3.0)

            d_bio_photo = mu_photo * biomass * sub_d
            d_bio_mixo = mu_mixo * biomass * sub_d
            d_bio_decay = 0.045 * biomass * sub_d

            photo_gain_total += d_bio_photo
            biomass = max(0.02, biomass + d_bio_photo + d_bio_mixo - d_bio_decay)

            # Aerobic heterotrophic + mixotrophic COD oxidation (no artificial DO floor)
            do_monod = do_val / (do_val + 0.35)
            het_temp = 1.035 ** (temp_c - 25.0)
            deg_fast = min(cod_fast, (0.48 * cod_fast * do_monod * het_temp * sub_d) + (d_bio_mixo / 0.45) * 1000.0)
            deg_slow = min(cod_slow, 0.22 * cod_slow * do_monod * het_temp * sub_d)
            oxidised = deg_fast + deg_slow

            cod_fast = max(0.0, cod_fast - deg_fast)
            cod_slow = max(0.0, cod_slow - deg_slow + 0.08 * 1420.0 * d_bio_decay)

            # Semi-implicit DO mass balance
            kla_h = (0.20 + 2.4 * q_ratio) * float(kla_temp_factor(temp_c))
            o2_photo = 1.60 * d_bio_photo * 1000.0  # mg O2 / L
            o2_resp = 0.045 * biomass * 1000.0 * f_temp * sub_d + 0.36 * oxidised
            do_val = float(np.clip((do_val + sub_h * kla_h * do_sat + o2_photo - o2_resp) / (1.0 + sub_h * kla_h), 0.05, 16.0))

            # Nitrogen cycling (assimilation, ammonification, nitrification, stripping)
            n_upt = 68.0 * (d_bio_photo + d_bio_mixo)  # mg N/L (~6.8% N content)
            n_amm = 0.012 * oxidised
            nitrif = min(tan_mg * (1.0 - phi_nh3), 0.14 * tan_mg * (1.0 - phi_nh3) * do_monod * het_temp * sub_d)
            strip = min(tan_mg * phi_nh3, 0.12 * (0.15 + 0.85 * q_ratio) * float(kla_temp_factor(temp_c)) * tan_mg * phi_nh3 * sub_d)
            tan_mg = max(0.5, tan_mg + n_amm - n_upt - nitrif - strip)

            # Conservative charge balance + CO2 exchange -> mechanistic pH
            z_net = z_net - (nitrif / 1000.0) / MW_N + (0.0018 * deg_fast) / 1000.0
            pk1_p, pk2_p, _, pkh_0, _, _ = carbonate_constants(temp_c, ion)
            a0_co2, _, _ = carbonate_fractions(ph_val, pk1_p, pk2_p)
            co2_sat = (10.0 ** (-pkh_0)) * 4.2e-4
            kla_co2_h = DIFFUSIVITY_RATIO_CO2_O2_SQRT * kla_h

            c_fixed = (0.50 * d_bio_photo) / MW_C  # g/L / g/mol = mol/L
            c_resp = ((0.35 * oxidised / 32.0) + (0.30 * (0.045 * biomass * 1000.0 * f_temp * sub_d) / 32.0)) / 1000.0
            dic_num = dic_mol + sub_h * kla_co2_h * co2_sat + c_resp - c_fixed
            dic_mol = float(np.clip(dic_num / max(1.0 + sub_h * kla_co2_h * a0_co2, 1.0), 5e-4, 0.09))

            ph_val = solve_ph(dic_mol, z_net, (tan_mg / 1000.0) / MW_N, temp_c, ion)

            # Bio-flocculation / detritus clarification (supernatant settling)
            detritus_mg = max(25.0, detritus_mg * np.exp(-0.22 * sub_d) + 80.0 * d_bio_decay)

        self._internal.update({
            "cod_fast": cod_fast,
            "cod_slow": cod_slow,
            "dic_mol": dic_mol,
            "z_net": z_net,
            "detritus_mg": detritus_mg,
        })
        self.last_photo_gain_gl = float(photo_gain_total)

        hod_end = (step * horizon) % 24.0
        new_state["cod"] = float(max(25.0, cod_fast + cod_slow))
        new_state["bod"] = float(max(10.0, 0.72 * cod_fast + 0.30 * cod_slow + self.rng.normal(0.0, 2.5)))
        new_state["total_organic_carbon"] = float(max(8.0, 0.316 * new_state["cod"] + self.rng.normal(0.0, 1.5)))
        new_state["dissolved_ammonia"] = float(tan_mg)
        new_state["ph"] = float(ph_val)
        new_state["dissolved_oxygen"] = float(do_val)
        # Effluent suspended solids: clarified supernatant fraction + microalgal flocs
        new_state["total_suspended_solids"] = float(max(25.0, detritus_mg + biomass * 1000.0 * 0.14))
        new_state["turbidity"] = float(max(10.0, 0.665 * new_state["total_suspended_solids"] + self.rng.normal(0.0, 4.0)))
        new_state["total_dissolved_solids"] = float(np.clip(tds, 200.0, 10000.0))
        new_state["water_temperature"] = float(temp_c)
        new_state["light_intensity"] = float(light_lux)
        new_state["ambient_temperature"] = float(29.5 + 2.8 * np.sin((hod_end - 10.0) * np.pi / 12.0))
        new_state["relative_humidity"] = float(75.0 - 7.5 * np.sin((hod_end - 9.0) * np.pi / 12.0))
        new_state["atmospheric_pressure"] = float(1012.0 + self.rng.normal(0.0, 1.0))
        new_state["air_oxygen"] = float(20.9 + self.rng.normal(0.0, 0.05))
        phi_end = float(free_nh3_fraction(temp_c, ph_val, float(ionic_strength_from_tds(tds))))
        new_state["air_ammonia"] = float(max(0.2, 0.49 * tan_mg * (0.35 + 0.65 * phi_end)))

        return new_state, float(biomass), float(z_net * 1000.0)


class POMEVisionGenerator:
    """
    Synthesizes bioreactor photography corresponding to physical state and optionally
    embeds it with the trained `VisionEncoder` (eliminating synthetic `sin(step)` leakage).
    """

    def __init__(self, visual_dim: int = 32, seed: int = 42, vision_encoder: Optional[torch.nn.Module] = None):
        self.visual_dim = visual_dim
        self.rng = np.random.RandomState(seed)
        self.vision_encoder = vision_encoder

    def generate_visual_state(
        self,
        cod: float,
        ph: float,
        turbidity: float,
        step: int,
        is_violated: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generates 64x64 RGB state image and 32-dim latent vector h_z."""
        cod_norm = float(np.clip(cod / 4500.0, 0.0, 1.0))
        turb_norm = float(np.clip(turbidity / 1000.0, 0.0, 1.0))
        treatment_progress = float(np.clip(1.0 - cod_norm, 0.0, 1.0))

        if is_violated or ph < 5.5 or ph > 10.5:
            r_base, g_base, b_base, clarity = 0.55, 0.40, 0.18, 0.15
        else:
            r_base = 0.30 * cod_norm + 0.12 * (1.0 - cod_norm)
            g_base = 0.18 * cod_norm + 0.62 * treatment_progress
            b_base = 0.10 * cod_norm + 0.22 * treatment_progress
            clarity = float(np.clip(treatment_progress * (1.0 - turb_norm * 0.45), 0.10, 0.95))

        img = np.zeros((64, 64, 3), dtype=np.uint8)
        noise = self.rng.normal(0.0, 6.0, (64, 64, 3))
        img[:, :, 0] = np.clip(r_base * 255.0 + noise[:, :, 0], 0, 255).astype(np.uint8)
        img[:, :, 1] = np.clip(g_base * 255.0 + noise[:, :, 1], 0, 255).astype(np.uint8)
        img[:, :, 2] = np.clip(b_base * 255.0 + noise[:, :, 2], 0, 255).astype(np.uint8)

        if self.vision_encoder is not None:
            dev = next(self.vision_encoder.parameters()).device
            img_t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(dev) / 255.0
            with torch.no_grad():
                hz_t = self.vision_encoder(img_t)
            return img, hz_t.squeeze(0).cpu().numpy().astype(np.float32)

        # Deterministic physical image descriptors + clean Gaussian measurement noise (zero sin(step) leakage)
        hz = np.zeros(self.visual_dim, dtype=np.float32)
        mean_rgb = img.mean(axis=(0, 1)) / 255.0
        std_rgb = img.std(axis=(0, 1)) / 255.0
        hz[0] = treatment_progress
        hz[1] = float(mean_rgb[1] - mean_rgb[0])
        hz[2] = clarity
        hz[3] = turb_norm
        hz[4] = cod_norm
        hz[5:8] = mean_rgb.astype(np.float32)
        hz[8:11] = std_rgb.astype(np.float32)
        if self.visual_dim > 11:
            hz[11:] = self.rng.normal(0.0, 0.02, self.visual_dim - 11).astype(np.float32)

        return img, hz
