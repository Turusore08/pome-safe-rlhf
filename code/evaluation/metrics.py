"""
Evaluation Metrics for Biochemical Efficacy, Actuation Efficiency, Safety,
and LAPSE Standardized Scientific Benchmarks.

Calculates:
1. Biochemical Efficacy: COD, BOD, and TSS pollutant removal percentage (R_p)
2. Actuation Efficiency: Chemical reagent and electricity expenditure in IDR (E, eta_act)
3. Safety Violation Rate (VR) and Intervention Rate (IR)
4. PerMenLHK No. 5/2014 statutory compliance verification (compliance_day)
5. LAPSE Standardized Metrics (Gerold, Brandner, & Lucia, ESCAPE 36 / LAPSE 2026):
   - G_mean, G_std: Mean and standard deviation of undiscounted cumulative return
   - n_cv (CNCV): Total count of constraint violations across episodes
   - d_cv_mean: Mean magnitude of constraint violations
6. 5-Treatment Action Breakdown (T1 Aeration, T2 Light, T3 Molasses, T4 NaHCO3, T5 Inoculum)
7. Step-by-Step Reward & Punishment Audit Ledger:
   State Reward = (E + B + Q) - (CT + CO) - Penalty
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    EFFLUENT_COMPLIANCE_LIMITS as REGULATORY_LIMITS,
    BIOLOGICAL_SAFETY_LIMITS as OPERATIONAL_CONSTRAINTS,
    ACTION_COLUMNS,
    ACTION_UNIT_COSTS_IDR,
    K8_BASELINE_COST_IDR,
    DECISION_HORIZON_HOURS,
)
from environment.reward_rig import (
    compute_action_cost,
    compute_material_cost,
    compute_operational_cost,
    compute_state_reward,
    compute_compliance_gap,
)


def compute_biochemical_efficacy(
    initial_pollutants: Dict[str, float],
    final_pollutants: Dict[str, float],
) -> Dict[str, float]:
    """
    Computes pollutant removal percentages:
    R_p = (C_0 - C_T) / C_0 * 100% for COD, BOD, and TSS.
    """
    results = {}
    for param in ["cod", "bod", "total_suspended_solids"]:
        c0 = initial_pollutants.get(param, 1.0)
        cT = final_pollutants.get(param, 0.0)
        removal_rate = ((c0 - cT) / max(1e-3, c0)) * 100.0
        results[f"{param}_removal_pct"] = float(np.clip(removal_rate, 0.0, 100.0))

    results["cod_removal"] = results["cod_removal_pct"]
    results["bod_reduction"] = results["bod_removal_pct"]
    results["bod_reduction_pct"] = results["bod_removal_pct"]
    results["tss_removal"] = results["total_suspended_solids_removal_pct"]
    results["tss_removal_pct"] = results["total_suspended_solids_removal_pct"]
    return results


def compute_actuation_efficiency(
    actions_sequence: np.ndarray,
    baseline_cost: float = K8_BASELINE_COST_IDR,
) -> Dict[str, float]:
    """
    Calculates total actuation expenditure E = sum_t sum_k kappa_k * a^(k)_t
    and relative efficiency saving eta_act = (E_base - E) / E_base * 100%.
    """
    acts = np.asarray(actions_sequence, dtype=float)
    if acts.ndim == 1:
        acts = acts[None, :]
    total_cost_idr = float(sum(compute_action_cost(a) for a in acts))
    savings_pct = ((baseline_cost - total_cost_idr) / max(1e-3, baseline_cost)) * 100.0

    return {
        "total_cost_idr": total_cost_idr,
        "reagent_cost": float(sum(compute_material_cost(a) for a in acts)),
        "energy_cost": float(np.sum(acts[:, 0] * ACTION_UNIT_COSTS_IDR["aeration"])),
        "actuation_efficiency_pct": savings_pct,
    }


def compute_safety_violation_rate(
    episode_violations: List[bool],
    intervention_count: int,
    total_steps: int,
) -> Dict[str, float]:
    """
    Calculates Safety Violation Rate (VR) and Execution-Time Intervention Rate (IR).
    VR = (1 / K) * sum I_k
    IR = (1 / sum T) * sum I[a^exec != argmax Q_r]
    """
    k = len(episode_violations)
    if k == 0:
        return {"violation_rate": 0.0, "intervention_rate": 0.0}

    violation_rate = float(np.mean([1.0 if v else 0.0 for v in episode_violations]))
    ir_rate = float(intervention_count / max(1, total_steps))

    return {
        "violation_rate": violation_rate,
        "safety_violation_count": int(np.sum(episode_violations)),
        "intervention_rate": ir_rate,
        "intervention_count": intervention_count,
    }


def verify_permenlhk_compliance(
    trajectory_states: List[Dict[str, float]],
    step_hours: float = DECISION_HORIZON_HOURS,
) -> Dict[str, Any]:
    """
    Verifies statutory discharge compliance under PerMenLHK No. 5/2014.
    Checks sustained compliance threshold over 24 consecutive hours
    (4 windows at 6h/step, or 24 steps at 1h/step).
    Includes COD, BOD, TSS, NH3, and pH [6.0, 9.0].
    """
    hold_steps = max(1, int(round(24.0 / max(0.25, step_hours))))
    consecutive_ok = 0
    compliance_step = None

    limit_cod = REGULATORY_LIMITS["cod"]
    limit_bod = REGULATORY_LIMITS["bod"]
    limit_tss = REGULATORY_LIMITS["total_suspended_solids"]
    limit_nh3 = REGULATORY_LIMITS.get("dissolved_ammonia", 50.0)

    for t, s in enumerate(trajectory_states):
        tss_val = s.get("total_suspended_solids", s.get("tss", 9999.0))
        nh3_val = s.get("dissolved_ammonia", s.get("nh3", 0.0))
        is_ok = (
            s.get("cod", 9999.0) <= limit_cod
            and s.get("bod", 9999.0) <= limit_bod
            and tss_val <= limit_tss
            and nh3_val <= limit_nh3
            and 6.0 <= s.get("ph", 0.0) <= 9.0
        )
        if is_ok:
            consecutive_ok += 1
            if consecutive_ok >= hold_steps and compliance_step is None:
                compliance_step = t - (hold_steps - 1)
        else:
            consecutive_ok = 0

    if compliance_step is not None:
        compliance_day = float(compliance_step * step_hours / 24.0)
        is_compliant = True
    else:
        compliance_day = 16.0
        is_compliant = False

    return {
        "compliance_check": is_compliant,
        "compliance_day": compliance_day,
        "limit_cod": limit_cod,
        "limit_bod": limit_bod,
        "limit_tss": limit_tss,
        "limit_nh3": limit_nh3,
    }


# ==============================================================================
# 5. LAPSE Standardized Metrics (Gerold, Brandner, & Lucia, ESCAPE 36 / 2026)
# ==============================================================================
def compute_step_violation_magnitude(state_dict: Dict[str, float]) -> Tuple[int, float]:
    """
    Computes binary biological safety constraint violation indicator c_t in {0, 1}
    and normalized violation magnitude d_cv >= 0 for a single state against
    `BIOLOGICAL_SAFETY_LIMITS` (distinct from Tier-1 discharge limits):
      - pH in [5.5, 10.5]
      - Dissolved Oxygen (DO) >= 0.5 mg/L (hypoxic crash boundary)
      - Water Temperature in [18.0, 38.0] C
    """
    ph = float(state_dict.get("ph", 7.5))
    do = float(state_dict.get("dissolved_oxygen", state_dict.get("do", 5.0)))
    temp = float(state_dict.get("water_temperature", state_dict.get("water_temp", 30.0)))

    ph_min = OPERATIONAL_CONSTRAINTS["ph"]["min"]
    ph_max = OPERATIONAL_CONSTRAINTS["ph"]["max"]
    do_min = OPERATIONAL_CONSTRAINTS["dissolved_oxygen"]["min"]
    temp_min = OPERATIONAL_CONSTRAINTS["water_temperature"]["min"]
    temp_max = OPERATIONAL_CONSTRAINTS["water_temperature"]["max"]

    d_ph = max(0.0, ph_min - ph) + max(0.0, ph - ph_max)
    d_do = max(0.0, (do_min - do) / max(do_min, 1e-6))
    d_temp = max(0.0, (temp - temp_max) / 5.0) + max(0.0, (temp_min - temp) / 5.0)

    d_cv = float(d_ph + d_do + d_temp)
    is_violated = 1 if (ph < ph_min or ph > ph_max or do < do_min or temp > temp_max or temp < temp_min) else 0
    return is_violated, d_cv


def compute_lapse_metrics(
    trajectories: List[Dict[str, Any]],
    step_hours: float = DECISION_HORIZON_HOURS,
) -> Dict[str, float]:
    """
    Computes standardized evaluation metrics following Tables 1-3 of LAPSE
    (Gerold et al., 2026):
      - G_mean: Mean undiscounted cumulative return across episodes
      - G_std: Standard deviation of undiscounted cumulative return
      - n_cv (CNCV): Total count of constraint violations across all episodes & steps
      - d_cv_mean: Mean violation magnitude across violating steps (0.0 if n_cv == 0)
      - compliance_day_mean: Mean day of PerMenLHK compliance (T_comp)
      - compliance_rate_pct: Percentage of episodes achieving sustained compliance
      - cj_cost_idr_mean: Mean cumulative actuation cost (CJ) in IDR

    Each item in `trajectories` is a dict containing:
      - 'rewards': list/array of step rewards r_t
      - 'states': list of state dicts (optional, for computing violations & compliance)
      - 'actions': array of shape (T, 5) (optional, for computing CJ cost)
      - 'violations': optional list of binary flags
    """
    if not trajectories:
        return {
            "G_mean": 0.0,
            "G_std": 0.0,
            "n_cv": 0,
            "cncv_per_episode": 0.0,
            "d_cv_mean": 0.0,
            "compliance_day_mean": 16.0,
            "compliance_rate_pct": 0.0,
            "cj_cost_idr_mean": 0.0,
        }

    returns = []
    total_n_cv = 0
    violation_magnitudes = []
    compliance_days = []
    compliant_flags = []
    episode_costs = []

    for traj in trajectories:
        rews = np.asarray(traj.get("rewards", []), dtype=float)
        returns.append(float(np.sum(rews)) if len(rews) > 0 else 0.0)

        states = traj.get("states", [])
        if states:
            comp_res = verify_permenlhk_compliance(states, step_hours=step_hours)
            compliance_days.append(float(comp_res["compliance_day"]))
            compliant_flags.append(1.0 if comp_res["compliance_check"] else 0.0)

            for s in states:
                v_flag, d_mag = compute_step_violation_magnitude(s)
                if v_flag > 0:
                    total_n_cv += 1
                    violation_magnitudes.append(d_mag)
        elif "violations" in traj:
            v_arr = np.asarray(traj["violations"], dtype=float)
            n_v = int(np.sum(v_arr > 0.5))
            total_n_cv += n_v
            if n_v > 0:
                violation_magnitudes.extend(v_arr[v_arr > 0.5].tolist())

        acts = traj.get("actions", None)
        if acts is not None and len(acts) > 0:
            eff = compute_actuation_efficiency(np.asarray(acts, dtype=float))
            episode_costs.append(eff["total_cost_idr"])

    g_arr = np.asarray(returns, dtype=float)
    n_ep = len(trajectories)

    return {
        "G_mean": float(np.mean(g_arr)),
        "G_std": float(np.std(g_arr)),
        "n_cv": int(total_n_cv),
        "cncv_per_episode": float(total_n_cv / max(1, n_ep)),
        "d_cv_mean": float(np.mean(violation_magnitudes)) if violation_magnitudes else 0.0,
        "compliance_day_mean": float(np.mean(compliance_days)) if compliance_days else 16.0,
        "compliance_rate_pct": float(np.mean(compliant_flags) * 100.0) if compliant_flags else 0.0,
        "cj_cost_idr_mean": float(np.mean(episode_costs)) if episode_costs else 0.0,
    }


# ==============================================================================
# 6. 5-Treatment Action Breakdown (T1-T5) & Step-by-Step Reward Ledger
# ==============================================================================
def compute_action_treatment_breakdown(
    action_history: np.ndarray,
    step_hours: float = DECISION_HORIZON_HOURS,
) -> Dict[str, Any]:
    """
    Computes detailed daily and total breakdown of the 5 treatments (T1-T5)
    answering Meet 1 questions on agent interventions:
      - T1: Aeration (hours or L/s)
      - T2: Light intensity (lux)
      - T3: Molasses carbon dose (mL)
      - T4: Sodium Bicarbonate NaHCO3 dose (grams / mL)
      - T5: Microalgae Inoculum dose (mL)

    Action column order in `ACTION_COLUMNS`:
      [0] aeration (T1)
      [1] sodium_bicarbonate (T4)
      [2] microalgae_inoculum (T5)
      [3] molasses (T3)
      [4] light_intensity (T2)
    """
    acts = np.asarray(action_history, dtype=float)
    if acts.ndim == 1:
        acts = acts[None, :]

    n_steps = len(acts)
    steps_per_day = max(1, int(round(24.0 / max(0.25, step_hours))))
    n_days = int(np.ceil(n_steps / steps_per_day))

    t1_aer = acts[:, 0]
    t4_bic = acts[:, 1]
    t5_ino = acts[:, 2]
    t3_mol = acts[:, 3]
    t2_light = acts[:, 4]

    daily_profile = []
    for d in range(n_days):
        s_idx = d * steps_per_day
        e_idx = min(n_steps, (d + 1) * steps_per_day)
        daily_profile.append({
            "day": d + 1,
            "T1_aeration_total_h": float(np.sum(t1_aer[s_idx:e_idx])),
            "T2_light_mean_lux": float(np.mean(t2_light[s_idx:e_idx])),
            "T3_molasses_total_ml": float(np.sum(t3_mol[s_idx:e_idx])),
            "T4_nahco3_total_g": float(np.sum(t4_bic[s_idx:e_idx])),
            "T5_inoculum_total_ml": float(np.sum(t5_ino[s_idx:e_idx])),
        })

    return {
        "total_steps": int(n_steps),
        "total_days": int(n_days),
        "T1_aeration_total_h": float(np.sum(t1_aer)),
        "T1_aeration_mean_h_per_step": float(np.mean(t1_aer)),
        "T1_active_pct": float(np.mean(t1_aer > 1e-2) * 100.0),
        "T2_light_mean_lux": float(np.mean(t2_light)),
        "T2_active_pct": float(np.mean(t2_light > 10.0) * 100.0),
        "T3_molasses_total_ml": float(np.sum(t3_mol)),
        "T3_active_pct": float(np.mean(t3_mol > 1e-2) * 100.0),
        "T4_nahco3_total_g": float(np.sum(t4_bic)),
        "T4_active_pct": float(np.mean(t4_bic > 1e-2) * 100.0),
        "T5_inoculum_total_ml": float(np.sum(t5_ino)),
        "T5_active_pct": float(np.mean(t5_ino > 1e-2) * 100.0),
        "daily_profile": daily_profile,
    }


def build_step_reward_punishment_ledger(
    states_sequence: List[Dict[str, float]],
    actions_sequence: np.ndarray,
    step_hours: float = DECISION_HORIZON_HOURS,
    safety_penalty_weight: float = 50.0,
) -> List[Dict[str, Any]]:
    """
    Constructs a transparent step-by-step audit ledger of State Reward & Punishment:
        State Reward = (E + B + Q) - (CT + CO) - Constraint_Penalty
    """
    acts = np.asarray(actions_sequence, dtype=float)
    if acts.ndim == 1:
        acts = acts[None, :]

    n_steps = min(len(acts), max(0, len(states_sequence) - 1))
    ledger = []
    cumulative_g = 0.0

    for t in range(n_steps):
        s_curr = states_sequence[t]
        s_next = states_sequence[t + 1]
        a_t = acts[t]

        b_curr = float(s_curr.get("biomass", s_curr.get("biomassa_estimasi(g/L)", 0.5)))
        b_next = float(s_next.get("biomass", s_next.get("biomassa_estimasi(g/L)", 0.5)))

        r_base, comp = compute_state_reward(
            s_curr_dict=s_curr,
            a_t=a_t,
            s_next_dict=s_next,
            biomass_curr=b_curr,
            biomass_next=b_next,
            dt_hours=step_hours,
        )
        v_flag, d_mag = compute_step_violation_magnitude(s_next)
        penalty_val = float(safety_penalty_weight * v_flag)
        net_reward = float(r_base - penalty_val)
        cumulative_g += net_reward

        ledger.append({
            "step": int(t),
            "day": float((t + 1) * step_hours / 24.0),
            "cod_mgL": float(s_next.get("cod", 0.0)),
            "bod_mgL": float(s_next.get("bod", 0.0)),
            "tss_mgL": float(s_next.get("total_suspended_solids", s_next.get("tss", 0.0))),
            "nh3_mgL": float(s_next.get("dissolved_ammonia", s_next.get("nh3", 0.0))),
            "ph": float(s_next.get("ph", 7.0)),
            "do_mgL": float(s_next.get("dissolved_oxygen", s_next.get("do", 5.0))),
            "biomass_gL": float(b_next),
            "T1_aeration_h": float(a_t[0]),
            "T2_light_lux": float(a_t[4]),
            "T3_molasses_ml": float(a_t[3]),
            "T4_nahco3_g": float(a_t[1]),
            "T5_inoculum_ml": float(a_t[2]),
            "E_effluent_idr": float(comp["E_effluent_quality_idr"]),
            "B_biomass_idr": float(comp["B_biomass_valorization_idr"]),
            "Q_co2_idr": float(comp["Q_co2_sequestration_idr"]),
            "CT_material_idr": float(comp["CT_material_cost_idr"]),
            "CO_operational_idr": float(comp["CO_operational_cost_idr"]),
            "violation_flag": int(v_flag),
            "violation_magnitude": float(d_mag),
            "constraint_penalty": float(penalty_val),
            "net_step_reward": float(net_reward),
            "cumulative_G": float(cumulative_g),
        })

    return ledger


# Published historical cultivation baselines (Mirda et al. 2026)
BENCHMARK_RUNS: Dict[str, Dict[str, Any]] = {
    "K8_HCR": {
        "controller": "hurdle_chained_regression",
        "compliance_day": 5.0,
        "cod_removal_pct": 95.70,
        "biomass_gl": 2.49,
        "total_cost_idr": 6170.0,
        "violations": 0,
    },
    "K5_BestHuman": {
        "controller": "human_expert",
        "compliance_day": 8.0,
        "cod_removal_pct": 98.56,
        "biomass_gl": 2.10,
        "total_cost_idr": 9872.0,
        "violations": 0,
    },
    "K7_Direct": {
        "controller": "direct_regression",
        "compliance_day": 16.0,
        "cod_removal_pct": 92.05,
        "biomass_gl": 1.68,
        "total_cost_idr": 19748.0,
        "violations": 0,
    },
}


def compare_with_hcr_baseline(rlhf_results: Dict[str, Any]):
    """
    Constructs comparative benchmark DataFrame between RLHF agent and historical runs.
    Consolidated from baseline_comparison.py without fallback hallucination defaults.
    """
    import pandas as pd

    required = ["compliance_day", "cod_removal_pct", "biomass_gl", "total_cost_idr", "violations"]
    missing = [k for k in required if k not in rlhf_results]
    if missing:
        raise KeyError(f"[Strict Metric Error] Missing measured RLHF result keys: {missing}")

    rows = []
    for run_name, stats in BENCHMARK_RUNS.items():
        rows.append({
            "Run": run_name,
            "Controller": stats["controller"],
            "Days to Compliance": f"{stats['compliance_day']:.1f}",
            "COD Removal (%)": f"{stats['cod_removal_pct']:.2f}%",
            "Biomass (g/L)": f"{stats['biomass_gl']:.2f}",
            "Operating Cost (IDR)": f"Rp{stats['total_cost_idr']:,.0f}",
            "Safety Violations": str(stats["violations"]),
        })

    rows.append({
        "Run": "RLHF_Agent",
        "Controller": "Safe Offline-to-Online RLHF (Ours)",
        "Days to Compliance": f"{float(rlhf_results['compliance_day']):.1f}",
        "COD Removal (%)": f"{float(rlhf_results['cod_removal_pct']):.2f}%",
        "Biomass (g/L)": f"{float(rlhf_results['biomass_gl']):.2f}",
        "Operating Cost (IDR)": f"Rp{float(rlhf_results['total_cost_idr']):,.0f}",
        "Safety Violations": f"{int(rlhf_results['violations'])}",
    })

    return pd.DataFrame(rows)
