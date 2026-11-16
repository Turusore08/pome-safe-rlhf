"""
Five-Tier Preference Labeling Rubric, Expert Elicitation, & Preference Dataset Generator (LAPSE Adaptation).

Consolidated module (merged legacy labeling_tool.py) applying the two-tier regulatory/safety rubric:
- Rule 1 (Biological Safety First): Tier-2 biological survival bounds (pH in [5.5, 10.5], DO >= 0.5 mg/L,
  Temp in [18.0, 38.0] C) checked across the segment; unsafe segments lose automatically.
- Rule 2 (Tier-1 PerMenLH No. 5/2014 Effluent Compliance): Terminal state reaching full discharge
  standards (BOD5 <= 100, COD <= 350, TSS <= 250, NH3 <= 50, pH in [6.0, 9.0]) wins cleanly.
- Rule 3 (Biochemical Pareto Dominance): Higher COD/BOD reduction AND lower operating cost wins.
- Rule 4 (Photosynthetic Biomass Slope): On horizon L_s >= 8, positive microalgal growth Delta X > 0 wins.
- Rule 5 (Net Techno-Economic Return): Tie-break using rigorous physics reward sum r_rig = (E+B+Q)-(CT+CO).

(Gerold, Brandner, & Lucia, ESCAPE 36 / LAPSE 2026 Eq. 4).
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from config import BIOLOGICAL_SAFETY_LIMITS, EFFLUENT_COMPLIANCE_LIMITS
from preference.segment_sampler import (
    TrajectorySegment,
    sample_segment_pairs,
    load_runs_from_transitions_csv,
    PREF_STATE_FEATURES,
    PREF_ACTION_FEATURES,
)
from environment.reward_rig import compute_state_reward, compute_action_cost


def _state_row_to_dict(state_row: np.ndarray, action_row: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Convert a 10-dim PREF_STATE_FEATURES vector (+ optional action) into canonical dict keys."""
    d = {
        "cod": float(state_row[0]),
        "bod": float(state_row[1]),
        "total_organic_carbon": float(state_row[2]),
        "total_suspended_solids": float(state_row[3]),
        "ph": float(state_row[4]),
        "dissolved_oxygen": float(state_row[5]),
        "total_dissolved_solids": float(state_row[6]),
        "dissolved_ammonia": float(state_row[7]),
        "water_temperature": float(state_row[8]),
        "biomass_gL": float(state_row[9]),
    }
    if action_row is not None and len(action_row) >= 5:
        d["light_intensity"] = float(action_row[4])
    return d


def _is_full_permenlh_compliant(last_state_row: np.ndarray) -> bool:
    """Verify full Tier-1 PerMenLH No. 5/2014 compliance on terminal segment state."""
    cod = float(last_state_row[0])
    bod = float(last_state_row[1])
    tss = float(last_state_row[3])
    ph = float(last_state_row[4])
    nh3 = float(last_state_row[7])
    return (
        cod <= EFFLUENT_COMPLIANCE_LIMITS["cod"]
        and bod <= EFFLUENT_COMPLIANCE_LIMITS["bod"]
        and tss <= EFFLUENT_COMPLIANCE_LIMITS["total_suspended_solids"]
        and nh3 <= EFFLUENT_COMPLIANCE_LIMITS["dissolved_ammonia"]
        and EFFLUENT_COMPLIANCE_LIMITS["ph_min"] <= ph <= EFFLUENT_COMPLIANCE_LIMITS["ph_max"]
    )


def _compute_segment_safety_stats(seg: TrajectorySegment) -> Tuple[int, float]:
    """Check Tier-2 biological safety bounds across segment states and optional metadata."""
    m_viol = seg.metadata.get("violations", None)
    m_dcv = seg.metadata.get("d_cv", None)

    ph_min = BIOLOGICAL_SAFETY_LIMITS["ph"]["min"]
    ph_max = BIOLOGICAL_SAFETY_LIMITS["ph"]["max"]
    do_min = BIOLOGICAL_SAFETY_LIMITS["dissolved_oxygen"]["min"]
    t_min = BIOLOGICAL_SAFETY_LIMITS["water_temperature"]["min"]
    t_max = BIOLOGICAL_SAFETY_LIMITS["water_temperature"]["max"]

    ph_arr = seg.states[:, 4]
    do_arr = seg.states[:, 5]
    t_arr = seg.states[:, 8]

    v_mask = (ph_arr < ph_min) | (ph_arr > ph_max) | (do_arr < do_min) | (t_arr < t_min) | (t_arr > t_max)
    d_cv_arr = (
        np.maximum(0.0, ph_min - ph_arr)
        + np.maximum(0.0, ph_arr - ph_max)
        + np.maximum(0.0, do_min - do_arr)
        + np.maximum(0.0, t_min - t_arr)
        + np.maximum(0.0, t_arr - t_max)
    )
    meta_count = int(np.sum(m_viol)) if m_viol is not None else 0
    meta_dcv = float(np.sum(m_dcv)) if m_dcv is not None else float(meta_count)
    return max(meta_count, int(np.sum(v_mask))), max(meta_dcv, float(np.sum(d_cv_arr)))


def compute_segment_rig_rewards(seg: TrajectorySegment, dt_hours: float = 6.0) -> np.ndarray:
    """Return exact per-step rigorous physics rewards r_rig(s_t, a_t) for segment steps."""
    m_rig = seg.metadata.get("rig_rewards", None)
    if m_rig is not None and len(m_rig) == len(seg):
        rig_arr = np.asarray(m_rig, dtype=np.float32)
        if np.any(rig_arr != 0.0):
            return rig_arr

    rewards = np.zeros(len(seg), dtype=np.float32)
    for t in range(len(seg)):
        s_curr = _state_row_to_dict(seg.states[t], seg.actions[t])
        s_next = _state_row_to_dict(seg.states[min(t + 1, len(seg) - 1)], seg.actions[t])
        r_t, _ = compute_state_reward(
            s_curr_dict=s_curr,
            a_t=seg.actions[t],
            s_next_dict=s_next,
            biomass_curr=float(seg.states[t, 9]),
            biomass_next=float(seg.states[min(t + 1, len(seg) - 1), 9]),
            dt_hours=dt_hours,
        )
        rewards[t] = float(r_t)
    return rewards


def evaluate_rubric(
    tau_1: TrajectorySegment,
    tau_2: TrajectorySegment,
) -> int:
    """
    Apply the five-tier rubric to determine which segment is preferred:
        0 if tau_1 is preferred (tau_1 > tau_2)
        1 if tau_2 is preferred (tau_2 > tau_1)
    """
    m1 = tau_1.metadata
    m2 = tau_2.metadata

    # 1. Rule 1: Biological Safety First (Tier-2 survival limits)
    v1_count, d1 = _compute_segment_safety_stats(tau_1)
    v2_count, d2 = _compute_segment_safety_stats(tau_2)

    if v1_count > 0 and v2_count == 0:
        return 1
    elif v2_count > 0 and v1_count == 0:
        return 0
    elif v1_count > 0 and v2_count > 0:
        if abs(d1 - d2) > 0.05 * max(d1, d2, 1e-6):
            return 0 if d1 < d2 else 1

    # 2. Rule 2: Full Statutory Compliance Milestone (Tier-1 PerMenLH No. 5/2014)
    ok1 = _is_full_permenlh_compliant(tau_1.states[-1])
    ok2 = _is_full_permenlh_compliant(tau_2.states[-1])
    if ok1 and not ok2:
        return 0
    elif ok2 and not ok1:
        return 1

    # 3. Rule 3: Biochemical Pareto Dominance
    red1 = float(tau_1.states[0, 0] - tau_1.states[-1, 0])
    red2 = float(tau_2.states[0, 0] - tau_2.states[-1, 0])
    cost1 = float(m1.get("total_cost", sum(compute_action_cost(a) for a in tau_1.actions)))
    cost2 = float(m2.get("total_cost", sum(compute_action_cost(a) for a in tau_2.actions)))

    if red1 > red2 + 20.0 and cost1 < cost2 - 5.0:
        return 0
    elif red2 > red1 + 20.0 and cost2 < cost1 - 5.0:
        return 1

    # 4. Rule 4: Microalgal Biomass Growth Slope (L_s >= 8)
    biomass_growth1 = float(tau_1.states[-1, 9] - tau_1.states[0, 9])
    biomass_growth2 = float(tau_2.states[-1, 9] - tau_2.states[0, 9])
    if len(tau_1) >= 8:
        if biomass_growth1 > 0.05 and biomass_growth2 < -0.05 and red1 >= red2 - 50.0:
            return 0
        elif biomass_growth2 > 0.05 and biomass_growth1 < -0.05 and red2 >= red1 - 50.0:
            return 1

    # 5. Rule 5: Net Techno-Economic Score via rigorous physics reward sum r_rig
    rig1 = float(np.sum(compute_segment_rig_rewards(tau_1)))
    rig2 = float(np.sum(compute_segment_rig_rewards(tau_2)))
    return 0 if rig1 >= rig2 else 1


def which_is_better(comparison_info: Dict[str, Any]) -> int:
    """Structured expert preference selector for pre-summarized segment comparisons."""
    info_a = comparison_info["segment_a"]
    info_b = comparison_info["segment_b"]

    if info_a["violations"] > 0 and info_b["violations"] == 0:
        return 1
    if info_b["violations"] > 0 and info_a["violations"] == 0:
        return 0

    score_a = info_a.get("rig_reward_sum", info_a["cod_reduction_mg_l"] / 100.0 - 0.01 * info_a["total_reagent_cost_idr"])
    score_b = info_b.get("rig_reward_sum", info_b["cod_reduction_mg_l"] / 100.0 - 0.01 * info_b["total_reagent_cost_idr"])
    return 0 if score_a >= score_b else 1


def elicit_preference(
    seg_a: TrajectorySegment,
    seg_b: TrajectorySegment,
    expert_mode: str = "auto",
) -> Tuple[int, Dict[str, Any]]:
    """Consolidated expert elicitation helper (replaces deprecated labeling_tool.py)."""
    cost_a = sum(compute_action_cost(a) for a in seg_a.actions)
    cost_b = sum(compute_action_cost(a) for a in seg_b.actions)
    rig_a = float(np.sum(compute_segment_rig_rewards(seg_a)))
    rig_b = float(np.sum(compute_segment_rig_rewards(seg_b)))
    va, _ = _compute_segment_safety_stats(seg_a)
    vb, _ = _compute_segment_safety_stats(seg_b)

    comparison_info = {
        "segment_a": {
            "run_id": seg_a.metadata.get("run_id", "Run A"),
            "steps": len(seg_a),
            "cod_reduction_mg_l": float(seg_a.states[0, 0] - seg_a.states[-1, 0]),
            "total_reagent_cost_idr": float(cost_a),
            "rig_reward_sum": rig_a,
            "violations": va,
        },
        "segment_b": {
            "run_id": seg_b.metadata.get("run_id", "Run B"),
            "steps": len(seg_b),
            "cod_reduction_mg_l": float(seg_b.states[0, 0] - seg_b.states[-1, 0]),
            "total_reagent_cost_idr": float(cost_b),
            "rig_reward_sum": rig_b,
            "violations": vb,
        },
    }

    winner = evaluate_rubric(seg_a, seg_b) if expert_mode == "auto" else which_is_better(comparison_info)
    return winner, comparison_info


class PreferenceDataset(Dataset):
    """PyTorch Dataset of pairwise trajectory comparisons (tau_1, tau_2, r_rig, preferred_idx)."""

    def __init__(self, pairs_data: List[Dict[str, torch.Tensor]], segment_length: Union[int, str] = "comb"):
        self.pairs = pairs_data
        self.segment_length = segment_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.pairs[idx]


def build_preference_dataset(
    trajectories: Union[List[Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]], str],
    num_pairs: int = 1200,
    segment_length: Union[int, str] = "comb",
    save_path: Optional[str] = None,
    seed: int = 42,
) -> PreferenceDataset:
    """
    Build labeled preference dataset from sampled trajectory pairs for a specific L_s,
    including rigorous physics step rewards tau_k_rig_rewards for residual BT learning.
    """
    if isinstance(trajectories, str):
        trajectories = load_runs_from_transitions_csv(trajectories)

    pairs = sample_segment_pairs(
        trajectories,
        num_pairs=num_pairs,
        segment_length=segment_length,
        seed=seed,
    )
    labeled_data = []

    for tau_1, tau_2 in pairs:
        preferred_idx = evaluate_rubric(tau_1, tau_2)
        rig_1 = compute_segment_rig_rewards(tau_1)
        rig_2 = compute_segment_rig_rewards(tau_2)
        item: Dict[str, torch.Tensor] = {
            "tau_1_states": torch.tensor(tau_1.states, dtype=torch.float32),
            "tau_1_actions": torch.tensor(tau_1.actions, dtype=torch.float32),
            "tau_1_rig_rewards": torch.tensor(rig_1, dtype=torch.float32),
            "tau_2_states": torch.tensor(tau_2.states, dtype=torch.float32),
            "tau_2_actions": torch.tensor(tau_2.actions, dtype=torch.float32),
            "tau_2_rig_rewards": torch.tensor(rig_2, dtype=torch.float32),
            "preferred_idx": torch.tensor(preferred_idx, dtype=torch.long),
            "seg_len": torch.tensor(len(tau_1), dtype=torch.long),
        }
        if "obs_224" in tau_1.metadata and "obs_224" in tau_2.metadata:
            item["tau_1_obs_224"] = torch.tensor(tau_1.metadata["obs_224"], dtype=torch.float32)
            item["tau_2_obs_224"] = torch.tensor(tau_2.metadata["obs_224"], dtype=torch.float32)
        labeled_data.append(item)

    pref_dataset = PreferenceDataset(labeled_data, segment_length=segment_length)
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save(pref_dataset, save_path)
        print(f"[Preference Dataset] Created {len(pref_dataset)} pairs (L_s={segment_length}) saved to: {save_path}")

    return pref_dataset
