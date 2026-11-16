"""
Trajectory Segment Pair Sampler for Preference Learning (LAPSE Adaptation).

Extracts trajectory pairs (tau_1, tau_2) with controlled segment lengths (L_s):
- L_s = 4  (24 hours / 1 day): localizes immediate safety boundary crossings (pH, DO)
- L_s = 8  (48 hours / 2 days): captures transient response and daily diurnal cycles
- L_s = 20 (120 hours / 5 days): evaluates sustained microalgal growth slope and PerMenLHK compliance
- L_s = 'comb': balanced 1/3 mixture across the three horizons (Figure 3, Gerold et al., 2026).
"""

import os
import sys
import csv
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import torch

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

# Standard physical state and action features for preference modeling and heatmaps
PREF_STATE_FEATURES: List[str] = [
    "cod", "bod", "toc", "tss", "ph", "do", "tds", "nh3", "water_temp", "biomass_gL"
]
PREF_ACTION_FEATURES: List[str] = [
    "T1_aeration", "T4_nahco3_g", "T5_inoculum_ml", "T3_molasses_ml", "T2_light_lux"
]


class TrajectorySegment:
    """Represents a sequence of state-action transitions over a cultivation horizon."""

    def __init__(self, states: np.ndarray, actions: np.ndarray, metadata: Dict):
        self.states = np.asarray(states, dtype=np.float32)    # (L, state_dim)
        self.actions = np.asarray(actions, dtype=np.float32)  # (L, action_dim)
        self.metadata = metadata  # run_id, start_step, seg_len, violations, cod_reduction, etc.

    def __len__(self) -> int:
        return len(self.states)


def load_runs_from_transitions_csv(
    csv_path: Optional[str] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Loads trajectories grouped by run_id from `synthetic_pome_transitions.csv`
    (or any compatible transition CSV with physical state & action columns).
    """
    if csv_path is None:
        csv_path = os.path.join(CODE_DIR, "data", "synthetic_pome_transitions.csv")

    runs: Dict[str, Dict[str, List]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            r_id = row.get("run_id", "synth_00")
            if r_id not in runs:
                runs[r_id] = {
                    "states": [],
                    "actions": [],
                    "violations": [],
                    "d_cv": [],
                    "rewards": [],
                    "costs": [],
                }
            s_vec = [float(row[c]) for c in PREF_STATE_FEATURES]
            # Robust action mapping
            a_vec = []
            for c in PREF_ACTION_FEATURES:
                if c in row:
                    a_vec.append(float(row[c]))
                elif c.replace("_g", "").replace("_ml", "") in row:
                    a_vec.append(float(row[c.replace("_g", "").replace("_ml", "")]))
                else:
                    a_vec.append(0.0)
            runs[r_id]["states"].append(s_vec)
            runs[r_id]["actions"].append(a_vec)
            runs[r_id]["violations"].append(float(row.get("violation", 0)))
            runs[r_id]["d_cv"].append(float(row.get("d_cv", 0.0)))
            runs[r_id]["rewards"].append(float(row.get("reward", 0.0)))
            runs[r_id]["costs"].append(float(row.get("CT_idr", 0.0)) + float(row.get("CO_idr", 0.0)))

    # Convert to NumPy arrays
    out_runs: Dict[str, Dict[str, np.ndarray]] = {}
    for r_id, d in runs.items():
        out_runs[r_id] = {
            "run_id": r_id,
            "states": np.array(d["states"], dtype=np.float32),
            "actions": np.array(d["actions"], dtype=np.float32),
            "violations": np.array(d["violations"], dtype=np.float32),
            "d_cv": np.array(d["d_cv"], dtype=np.float32),
            "rewards": np.array(d["rewards"], dtype=np.float32),
            "costs": np.array(d["costs"], dtype=np.float32),
        }
    return out_runs


def sample_segment_pairs(
    trajectories: Union[List[Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]]],
    num_pairs: int = 1000,
    segment_length: Union[int, str] = "comb",
    seed: int = 42,
) -> List[Tuple[TrajectorySegment, TrajectorySegment]]:
    """
    Sample pairs of trajectory snippets (tau_1, tau_2) with controlled length L_s:
      - segment_length = 4: strictly 4 steps
      - segment_length = 8: strictly 8 steps
      - segment_length = 20: strictly 20 steps
      - segment_length = 'comb' (or None): equal 1/3 mixture across [4, 8, 20]
    """
    if isinstance(trajectories, dict):
        traj_list = list(trajectories.values())
    else:
        traj_list = trajectories

    rng = np.random.RandomState(seed)
    pairs = []

    for idx in range(num_pairs):
        if segment_length in (4, 8, 20):
            seg_len = int(segment_length)
        elif segment_length == "comb" or segment_length is None:
            # Equal 1/3 mixture as in LAPSE
            seg_len = int(rng.choice([4, 8, 20]))
        else:
            seg_len = int(segment_length)

        # Pick two runs
        t1 = traj_list[rng.randint(0, len(traj_list))]
        t2 = traj_list[rng.randint(0, len(traj_list))]

        max_start1 = max(1, len(t1["states"]) - seg_len)
        max_start2 = max(1, len(t2["states"]) - seg_len)

        start1 = rng.randint(0, max_start1)
        start2 = rng.randint(0, max_start2)

        s1_slice = t1["states"][start1 : start1 + seg_len]
        a1_slice = t1["actions"][start1 : start1 + seg_len]
        s2_slice = t2["states"][start2 : start2 + seg_len]
        a2_slice = t2["actions"][start2 : start2 + seg_len]

        v1_slice = t1["violations"][start1 : start1 + seg_len] if "violations" in t1 else np.zeros(seg_len)
        v2_slice = t2["violations"][start2 : start2 + seg_len] if "violations" in t2 else np.zeros(seg_len)

        d1_slice = t1["d_cv"][start1 : start1 + seg_len] if "d_cv" in t1 else np.zeros(seg_len)
        d2_slice = t2["d_cv"][start2 : start2 + seg_len] if "d_cv" in t2 else np.zeros(seg_len)

        c1_slice = t1["costs"][start1 : start1 + seg_len] if "costs" in t1 else np.zeros(seg_len)
        c2_slice = t2["costs"][start2 : start2 + seg_len] if "costs" in t2 else np.zeros(seg_len)

        meta1 = {
            "run_id": t1.get("run_id", "Run1"),
            "start": start1,
            "seg_len": seg_len,
            "violations": v1_slice,
            "d_cv": d1_slice,
            "rig_rewards": t1["rewards"][start1 : start1 + seg_len] if "rewards" in t1 else np.zeros(seg_len, dtype=np.float32),
            "total_cost": float(np.sum(c1_slice)),
            "cod_start": float(s1_slice[0, 0]),
            "cod_end": float(s1_slice[-1, 0]),
            "cod_reduction": float(s1_slice[0, 0] - s1_slice[-1, 0]),
            "biomass_start": float(s1_slice[0, -1]),
            "biomass_end": float(s1_slice[-1, -1]),
            "biomass_growth": float(s1_slice[-1, -1] - s1_slice[0, -1]),
            "ph_min": float(np.min(s1_slice[:, 4])),
            "ph_max": float(np.max(s1_slice[:, 4])),
            "do_min": float(np.min(s1_slice[:, 5])),
        }
        meta2 = {
            "run_id": t2.get("run_id", "Run2"),
            "start": start2,
            "seg_len": seg_len,
            "violations": v2_slice,
            "d_cv": d2_slice,
            "rig_rewards": t2["rewards"][start2 : start2 + seg_len] if "rewards" in t2 else np.zeros(seg_len, dtype=np.float32),
            "total_cost": float(np.sum(c2_slice)),
            "cod_start": float(s2_slice[0, 0]),
            "cod_end": float(s2_slice[-1, 0]),
            "cod_reduction": float(s2_slice[0, 0] - s2_slice[-1, 0]),
            "biomass_start": float(s2_slice[0, -1]),
            "biomass_end": float(s2_slice[-1, -1]),
            "biomass_growth": float(s2_slice[-1, -1] - s2_slice[0, -1]),
            "ph_min": float(np.min(s2_slice[:, 4])),
            "ph_max": float(np.max(s2_slice[:, 4])),
            "do_min": float(np.min(s2_slice[:, 5])),
        }

        if "obs_224" in t1 and "obs_224" in t2:
            meta1["obs_224"] = t1["obs_224"][start1 : start1 + seg_len]
            meta2["obs_224"] = t2["obs_224"][start2 : start2 + seg_len]

        tau_1 = TrajectorySegment(s1_slice, a1_slice, meta1)
        tau_2 = TrajectorySegment(s2_slice, a2_slice, meta2)
        pairs.append((tau_1, tau_2))

    return pairs
