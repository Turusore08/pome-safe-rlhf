"""
State Reward & Bioprocess Economics Framework.
Formulates the multi-objective State Reward matching the research architecture:

    State Reward = Keuntungan (+) - Biaya (-)
                 = (E + B + Q) - (CT + CO)

Underlying Standards & Literature:
- PerMenLH No. 5/2014 Lampiran III (COD <= 350, BOD <= 100, TSS <= 250, Total N <= 50, pH 6.0-9.0)
- Mirda et al. (2026) (Photobioreactor 2.5 L working volume & treatment benchmarks)
- Chisti (2007) photosynthetic stoichiometry (Y_CO2/X = 1.83 g CO2 / g photoautotrophic dry biomass)
- IDXCarbon 2024-2025 carbon credit valuation & Permen ESDM No. 7/2024 electricity tariff
- Ng, Harada, & Russell (1999) potential-based reward shaping (gamma * Phi(s') - Phi(s))
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    EFFLUENT_COMPLIANCE_LIMITS as REGULATORY_LIMITS,
    REACTOR_VOLUME_LITERS,
    ACTION_COLUMNS,
    ACTION_UNIT_COSTS_IDR,
    BIOMASS_PRICE_PER_GRAM_IDR,
    CO2_CREDIT_PER_GRAM_IDR,
    OPERATIONAL_OVERHEAD_STEP_IDR,
    REFERENCE_STEP_COST_IDR,
)
from models.bioprocess_core import CO2_TO_BIOMASS_STOICH

E_BAR_STEP_COST: float = REFERENCE_STEP_COST_IDR
BIOMASS_PRICE_PER_GRAM: float = BIOMASS_PRICE_PER_GRAM_IDR
CO2_CREDIT_PER_GRAM: float = CO2_CREDIT_PER_GRAM_IDR
PBR_VOLUME_LITERS: float = REACTOR_VOLUME_LITERS
OPERATIONAL_OVERHEAD_STEP: float = OPERATIONAL_OVERHEAD_STEP_IDR


def compute_compliance_gap(state_dict: Dict[str, float]) -> Tuple[float, bool]:
    """
    Computes distance to regulatory compliance e(s) under PerMenLHK No. 5/2014:
    e(s) = sum_{p in {COD, BOD, TSS, NH3}} [(y_p - L_p) / L_p]_+ + [6 - pH]_+ + [pH - 9]_+
    ok(s) = I[e(s) == 0]
    """
    cod = state_dict.get("cod", 4000.0)
    bod = state_dict.get("bod", 1500.0)
    tss = state_dict.get("total_suspended_solids", 1000.0)
    nh3 = state_dict.get("dissolved_ammonia", 100.0)
    ph = state_dict.get("ph", 7.0)

    # Positive excesses over statutory thresholds
    excess_cod = max(0.0, (cod - REGULATORY_LIMITS["cod"]) / REGULATORY_LIMITS["cod"])
    excess_bod = max(0.0, (bod - REGULATORY_LIMITS["bod"]) / REGULATORY_LIMITS["bod"])
    excess_tss = max(0.0, (tss - REGULATORY_LIMITS["total_suspended_solids"]) / REGULATORY_LIMITS["total_suspended_solids"])
    excess_nh3 = max(0.0, (nh3 - REGULATORY_LIMITS["dissolved_ammonia"]) / REGULATORY_LIMITS["dissolved_ammonia"])

    # pH boundary excess
    excess_ph_low = max(0.0, REGULATORY_LIMITS["ph_min"] - ph)
    excess_ph_high = max(0.0, ph - REGULATORY_LIMITS["ph_max"])

    e_s = excess_cod + excess_bod + excess_tss + excess_nh3 + excess_ph_low + excess_ph_high
    is_ok = bool(e_s < 1e-4)

    return float(e_s), is_ok


# ==============================================================================
# 1. BIAYA (-): CT (Biaya Material) & CO (Biaya Lain-lain)
# ==============================================================================

def compute_material_cost(action_dict_or_array) -> float:
    """
    Calculates chemical material cost CT (Biaya Material):
    - T4: Sodium Bicarbonate (NaHCO3, grams)
    - T3: Molasses (mL)
    - T5: Microalgae Inoculum (mL)
    """
    if isinstance(action_dict_or_array, dict):
        nahco3 = float(action_dict_or_array.get("sodium_bicarbonate", 0.0))
        molasses = float(action_dict_or_array.get("molasses", 0.0))
        inoculum = float(action_dict_or_array.get("microalgae_inoculum", 0.0))
    else:
        acts = np.asarray(action_dict_or_array, dtype=float)
        nahco3 = acts[1] if len(acts) > 1 else 0.0
        inoculum = acts[2] if len(acts) > 2 else 0.0
        molasses = acts[3] if len(acts) > 3 else 0.0

    ct = (
        nahco3 * ACTION_UNIT_COSTS_IDR["sodium_bicarbonate"]
        + molasses * ACTION_UNIT_COSTS_IDR["molasses"]
        + inoculum * ACTION_UNIT_COSTS_IDR["microalgae_inoculum"]
    )
    return float(ct)


def compute_operational_cost(action_dict_or_array, dt_hours: float = 6.0, include_overhead: bool = True) -> float:
    """
    Calculates non-material operational & utility cost CO (Biaya Lain-lain):
    - T1: Aeration pumping electrical cost (hours * tariff)
    - T2: Photobioreactor illumination electrical cost ((lux / 1000) * hours * tariff)
    - System monitoring & sensor network overhead per decision interval
    """
    if isinstance(action_dict_or_array, dict):
        aeration = float(action_dict_or_array.get("aeration", 0.0))
        light = float(action_dict_or_array.get("light_intensity", 0.0))
    else:
        acts = np.asarray(action_dict_or_array, dtype=float)
        aeration = acts[0] if len(acts) > 0 else 0.0
        light = acts[4] if len(acts) > 4 else 0.0

    aeration_cost = aeration * ACTION_UNIT_COSTS_IDR["aeration"]
    # UNIT_COSTS["light_intensity"] is already in IDR per (lux * 6h decision step) per Permen ESDM 7/2024
    light_cost = light * ACTION_UNIT_COSTS_IDR["light_intensity"] * (dt_hours / 6.0)
    overhead = OPERATIONAL_OVERHEAD_STEP * (dt_hours / 6.0) if include_overhead else 0.0

    co = aeration_cost + light_cost + overhead
    return float(co)


def compute_action_cost(action_dict_or_array) -> float:
    """
    Calculates total actuation cost (CT + CO) for backward-compatibility.
    E_t = sum_{k=1}^5 kappa_k * a^(k)_t (in IDR).
    """
    ct = compute_material_cost(action_dict_or_array)
    co = compute_operational_cost(action_dict_or_array, include_overhead=False)
    return float(ct + co)


# ==============================================================================
# 2. KEUNTUNGAN (+): E (Effluent Quality), B (Biomassa), Q (Penyerapan CO2)
# ==============================================================================

def compute_effluent_quality(
    s_curr_dict: Dict[str, float],
    s_next_dict: Dict[str, float],
) -> Dict[str, float]:
    """
    Calculates E (Effluent Quality) based on PerMenLHK No. 5/2014 compliance:
    - Distance reduction in regulatory compliance gap: Delta e = e(s_t) - e(s_{t+1})
    - COD pollutant degradation step: Delta COD (mg/L)
    - Compliance status indicator: ok(s_{t+1})
    Returns monetary equivalent and normalized score.
    """
    e_curr, ok_curr = compute_compliance_gap(s_curr_dict)
    e_next, ok_next = compute_compliance_gap(s_next_dict)

    delta_e = max(0.0, e_curr - e_next)
    cod_curr = float(s_curr_dict.get("cod", 4000.0))
    cod_next = float(s_next_dict.get("cod", 4000.0))
    delta_cod = max(0.0, cod_curr - cod_next)

    # Monetary value of wastewater treatment progress (penalty reduction + milestone compliance)
    progress_val = delta_e * 250.0 + (delta_cod / 50.0) * 15.0
    compliance_bonus = 350.0 if ok_next else 0.0
    e_idr = progress_val + compliance_bonus

    # Normalized score (0 to 5 scale per step)
    e_norm = (delta_e * 2.0) + (1.5 if ok_next else 0.0)

    return {
        "E_idr": float(e_idr),
        "E_norm": float(e_norm),
        "delta_e": float(delta_e),
        "delta_cod": float(delta_cod),
        "ok_next": bool(ok_next),
        "e_next": float(e_next),
    }


def compute_biomass_valorization(
    biomass_curr: float,
    biomass_next: float,
    reactor_volume_l: float = PBR_VOLUME_LITERS,
    price_per_gram: float = BIOMASS_PRICE_PER_GRAM,
) -> Dict[str, float]:
    """
    Calculates B (Valorisasi Biomassa):
    - Biomass growth delta: Delta X = max(0, X_{t+1} - X_t) (g/L)
    - Net dry biomass produced: Delta m_X = Delta X * Volume (grams)
    - Commercial value of harvestable Chlorella vulgaris biomass (feedstock / bio-fertilizer)
    """
    delta_biomass_gl = max(0.0, biomass_next - biomass_curr)
    delta_biomass_g = delta_biomass_gl * reactor_volume_l
    b_idr = delta_biomass_g * price_per_gram

    # Normalized score (0 to 3 scale)
    b_norm = delta_biomass_g * 1.5

    return {
        "B_idr": float(b_idr),
        "B_norm": float(b_norm),
        "delta_biomass_gl": float(delta_biomass_gl),
        "delta_biomass_g": float(delta_biomass_g),
        "biomass_next": float(biomass_next),
    }


def compute_co2_sequestration(
    biomass_curr: float,
    biomass_next: float,
    s_curr_dict: Optional[Dict[str, float]] = None,
    s_next_dict: Optional[Dict[str, float]] = None,
    reactor_volume_l: float = PBR_VOLUME_LITERS,
    credit_per_gram: float = CO2_CREDIT_PER_GRAM,
    photo_biomass_gain_g: Optional[float] = None,
) -> Dict[str, float]:
    """
    Calculates Q (Penyerapan CO2 / Carbon Sequestration):
    - Stoichiometric microalgal carbon fixation: 1.83 g CO2 per 1.0 g photoautotrophic biomass (Chisti, 2007)
    - Rejects dark heterotrophic growth (light_intensity <= 50 lux or non-photoautotrophic gain)
    - Carbon credit valuation grounded in IDXCarbon market pricing
    """
    if photo_biomass_gain_g is not None:
        photo_gain_g = max(0.0, float(photo_biomass_gain_g))
    else:
        delta_biomass_gl = max(0.0, biomass_next - biomass_curr)
        # Gate offline carbon fixation attribution by active illumination (> 50 lux)
        lux_active = 1.0
        if s_next_dict is not None and "light_intensity" in s_next_dict:
            lux_active = 1.0 if float(s_next_dict["light_intensity"]) > 50.0 else 0.0
        elif s_curr_dict is not None and "light_intensity" in s_curr_dict:
            lux_active = 1.0 if float(s_curr_dict["light_intensity"]) > 50.0 else 0.0
        photo_gain_g = delta_biomass_gl * reactor_volume_l * lux_active

    co2_fixed_g = photo_gain_g * CO2_TO_BIOMASS_STOICH
    q_idr = co2_fixed_g * credit_per_gram

    # Normalized multi-objective score for RL training (0 to ~2 scale per step)
    q_norm = co2_fixed_g * 1.0

    return {
        "Q_idr": float(q_idr),
        "Q_norm": float(q_norm),
        "co2_fixed_g": float(co2_fixed_g),
    }


# ==============================================================================
# 3. STATE REWARD COMPOSITE: Keuntungan (+) - Biaya (-)
# ==============================================================================

def compute_state_reward(
    s_curr_dict: Dict[str, float],
    a_t,
    s_next_dict: Dict[str, float],
    biomass_curr: Optional[float] = None,
    biomass_next: Optional[float] = None,
    dt_hours: float = 6.0,
    w_e: float = 1.0,
    w_b: float = 1.0,
    w_q: float = 1.0,
    w_ct: float = 1.0,
    w_co: float = 1.0,
    gamma: float = 0.99,
    photo_biomass_gain_g: Optional[float] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Computes complete State Reward matching the system architecture:

        State Reward = Keuntungan (+) - Biaya (-)
                     = (w_E * E + w_B * B + w_Q * Q) - (w_CT * CT + w_CO * CO)
    """
    b_curr = biomass_curr if biomass_curr is not None else float(s_curr_dict.get("biomass", 0.5))
    b_next = biomass_next if biomass_next is not None else float(s_next_dict.get("biomass", 0.5))

    # 1. Compute Keuntungan (+) terms
    res_e = compute_effluent_quality(s_curr_dict, s_next_dict)
    res_b = compute_biomass_valorization(b_curr, b_next)
    res_q = compute_co2_sequestration(
        b_curr,
        b_next,
        s_curr_dict,
        s_next_dict,
        photo_biomass_gain_g=photo_biomass_gain_g,
    )

    e_idr = res_e["E_idr"]
    b_idr = res_b["B_idr"]
    q_idr = res_q["Q_idr"]
    keuntungan_idr = e_idr + b_idr + q_idr

    # 2. Compute Biaya (-) terms
    ct_idr = compute_material_cost(a_t)
    co_idr = compute_operational_cost(a_t, dt_hours=dt_hours, include_overhead=True)
    biaya_idr = ct_idr + co_idr

    # 3. Monetary Net Benefit
    net_benefit_idr = keuntungan_idr - biaya_idr

    # 4. Normalized State Reward for RL convergence
    # Normalize costs relative to benchmark step cost E_bar (IDR ~308.5)
    ct_norm = ct_idr / E_BAR_STEP_COST
    co_norm = co_idr / E_BAR_STEP_COST
    biaya_norm = (w_ct * ct_norm) + (w_co * co_norm)

    # Potential-based shaping for monotonic progress toward compliance (Ng et al., 1999)
    phi_curr = -compute_compliance_gap(s_curr_dict)[0]
    phi_next = -res_e["e_next"]
    shaping_term = gamma * phi_next - phi_curr

    keuntungan_norm = (
        (w_e * res_e["E_norm"])
        + (w_b * res_b["B_norm"])
        + (w_q * res_q["Q_norm"])
        + (1.0 * shaping_term)
    )

    # Base state reward
    r_state = keuntungan_norm - biaya_norm

    # If already compliant, add sustaining bonus
    if res_e["ok_next"]:
        r_state += 1.0

    components = {
        # Core Architecture Terms (Target Slide)
        "state_reward": float(r_state),
        "r_rig": float(r_state),  # Backward compatibility alias
        "keuntungan_total_idr": float(keuntungan_idr),
        "biaya_total_idr": float(biaya_idr),
        "net_benefit_idr": float(net_benefit_idr),
        # Keuntungan (+) Breakdown
        "E_effluent_quality_idr": float(e_idr),
        "E_norm": float(res_e["E_norm"]),
        "B_biomass_valorization_idr": float(b_idr),
        "B_norm": float(res_b["B_norm"]),
        "Q_co2_sequestration_idr": float(q_idr),
        "Q_norm": float(res_q["Q_norm"]),
        # Biaya (-) Breakdown
        "CT_material_cost_idr": float(ct_idr),
        "CT_norm": float(ct_norm),
        "CO_operational_cost_idr": float(co_idr),
        "CO_norm": float(co_norm),
        # Physical Bioprocess State Metrics
        "delta_biomass_g": float(res_b["delta_biomass_g"]),
        "co2_fixed_g": float(res_q["co2_fixed_g"]),
        "e_next": float(res_e["e_next"]),
        "ok_next": float(1.0 if res_e["ok_next"] else 0.0),
        "delta_cod": float(res_e["delta_cod"]),
        "step_cost_idr": float(ct_idr + co_idr),
    }

    return float(r_state), components


def compute_rig_reward(
    s_curr_dict: Dict[str, float],
    a_t,
    s_next_dict: Dict[str, float],
    gamma: float = 0.99,
    w_t: float = 1.0,
    w_e: float = 1.0,
    w_phi: float = 1.0,
    biomass_curr: Optional[float] = None,
    biomass_next: Optional[float] = None,
    dt_hours: float = 6.0,
    photo_biomass_gain_g: Optional[float] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Wrapper for compute_state_reward preserving legacy signature.
    Seamlessly passes all 5 components of State Reward:
    Keuntungan (E, B, Q) - Biaya (CT, CO).
    """
    return compute_state_reward(
        s_curr_dict=s_curr_dict,
        a_t=a_t,
        s_next_dict=s_next_dict,
        biomass_curr=biomass_curr,
        biomass_next=biomass_next,
        dt_hours=dt_hours,
        w_e=w_t,
        w_b=1.0,
        w_q=1.0,
        w_ct=w_e,
        w_co=w_e,
        gamma=gamma,
        photo_biomass_gain_g=photo_biomass_gain_g,
    )
