"""
Bradley-Terry (T-REX) Reward Model Training & Tri-Reward Architecture (LAPSE Adaptation).

Implements the complete methodology from Section 4 of LAPSE (Gerold et al., 2026):
1. Bradley-Terry Loss Training for an ensemble of M=3 neural networks on preference pairs:
   P_psi(tau_1 > tau_2) = exp(sum r(tau_1)) / [exp(sum r(tau_1)) + exp(sum r(tau_2))]
2. Segment Length Ablation (L_s in {4, 8, 20, comb}) training 4 separate reward ensembles.
3. Ranking accuracy validation on held-out test split (target >= 85%).
4. Tri-Reward Formulation Suite:
   - r_rig   : Rigorous techno-economic reward from physics and PerMenLHK standards
   - r_pref  : Preference-learned reward ensemble mean and epistemic uncertainty
   - r_hybrid: r_rig + alpha * r_pref
"""

import os
import sys
import copy
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from models.reward_net import RewardNet, RewardEnsemble
from preference.labeling_rubric import (
    PreferenceDataset,
    build_preference_dataset,
    PREF_STATE_FEATURES,
    PREF_ACTION_FEATURES,
)
from environment.reward_rig import compute_state_reward
from evaluation.metrics import compute_step_violation_magnitude


def bradley_terry_loss(
    r_tau1: torch.Tensor,
    r_tau2: torch.Tensor,
    preferred_idx: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the Bradley-Terry ranking loss using logsigmoid.
    If preferred_idx == 0: target is tau_1 > tau_2, loss = -logsigmoid(r1 - r2)
    If preferred_idx == 1: target is tau_2 > tau_1, loss = -logsigmoid(r2 - r1)
    """
    diff = r_tau1 - r_tau2
    # preferred_idx == 0 -> sign = +1.0; preferred_idx == 1 -> sign = -1.0
    label_sign = torch.where(preferred_idx == 0, 1.0, -1.0)
    bt_loss = -F.logsigmoid(label_sign * diff).mean()
    return bt_loss


def _pack_preference_batch(
    batch: List[Dict[str, torch.Tensor]],
    dev: torch.device,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    List[int],
    List[int],
    int,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Concatenates variable-length preference pairs into single forward-pass tensors."""
    len1 = [int(it["tau_1_states"].shape[0]) for it in batch]
    len2 = [int(it["tau_2_states"].shape[0]) for it in batch]
    total_n1 = sum(len1)

    s_cat = torch.cat(
        [it["tau_1_states"] for it in batch] + [it["tau_2_states"] for it in batch],
        dim=0,
    ).to(dev)
    a_cat = torch.cat(
        [it["tau_1_actions"] for it in batch] + [it["tau_2_actions"] for it in batch],
        dim=0,
    ).to(dev)

    r1_rig_sum = torch.stack(
        [
            it["tau_1_rig_rewards"].sum()
            if "tau_1_rig_rewards" in it
            else torch.tensor(0.0, dtype=torch.float32)
            for it in batch
        ]
    ).to(dev)
    r2_rig_sum = torch.stack(
        [
            it["tau_2_rig_rewards"].sum()
            if "tau_2_rig_rewards" in it
            else torch.tensor(0.0, dtype=torch.float32)
            for it in batch
        ]
    ).to(dev)
    pref_vec = torch.stack([it["preferred_idx"] for it in batch]).to(dev)
    return s_cat, a_cat, len1, len2, total_n1, r1_rig_sum, r2_rig_sum, pref_vec


def train_single_reward_ensemble(
    dataset: PreferenceDataset,
    segment_length_name: str = "comb",
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 32,
    state_dim: int = 10,
    action_dim: int = 5,
    ensemble_size: int = 3,
    save_dir: str = "checkpoints",
    val_split: float = 0.20,
    seed: int = 42,
    alpha_res: float = 0.5,
    lambda_res: float = 1e-3,
    save_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[RewardEnsemble, Dict[str, float]]:
    """
    Trains an ensemble of M RewardNet residual models r_res(s, a) jointly inside
    the residual hybrid Bradley-Terry objective (Gerold et al., ESCAPE 36 / LAPSE 2026 Eq. 4):
        r_hybrid(tau) = sum_t [ r_rig(s_t, a_t) + alpha_res * r_res(s_t, a_t) ]
    """
    torch.manual_seed(seed)
    dev = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_total = len(dataset)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

    ensemble = RewardEnsemble(
        state_dim=state_dim, action_dim=action_dim, ensemble_size=ensemble_size
    ).to(dev)

    optimizers = [
        torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
        for m in ensemble.models
    ]

    for epoch in range(epochs):
        for m in ensemble.models:
            m.train()

        train_losses = [0.0] * ensemble_size
        n_batches = 0

        for batch in train_loader:
            s_cat, a_cat, len1, len2, total_n1, r1_rig_sum, r2_rig_sum, pref_vec = _pack_preference_batch(
                batch, dev
            )
            for m_idx, (model, opt) in enumerate(zip(ensemble.models, optimizers)):
                res_all = model(s_cat, a_cat)
                split1 = torch.split(res_all[:total_n1], len1)
                split2 = torch.split(res_all[total_n1:], len2)

                r1_res_sum = torch.stack([seg.sum() for seg in split1])
                r2_res_sum = torch.stack([seg.sum() for seg in split2])
                r1_hyb = r1_rig_sum + alpha_res * r1_res_sum
                r2_hyb = r2_rig_sum + alpha_res * r2_res_sum

                bt_l = bradley_terry_loss(r1_hyb, r2_hyb, pref_vec)
                reg_l = lambda_res * (
                    torch.stack([seg.pow(2).mean() for seg in split1]).mean()
                    + torch.stack([seg.pow(2).mean() for seg in split2]).mean()
                )
                batch_loss = bt_l + reg_l

                opt.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                train_losses[m_idx] += float(batch_loss.item())
            n_batches += 1

    # Evaluate validation ranking accuracy and loss under residual hybrid objective
    for m in ensemble.models:
        m.eval()

    val_correct = 0
    val_total = 0
    val_loss_sum = 0.0

    with torch.no_grad():
        for batch in val_loader:
            s_cat, a_cat, len1, len2, total_n1, r1_rig_sum, r2_rig_sum, pref_vec = _pack_preference_batch(
                batch, dev
            )
            r1_preds, r2_preds = [], []
            for m in ensemble.models:
                res_all = m(s_cat, a_cat)
                split1 = torch.split(res_all[:total_n1], len1)
                split2 = torch.split(res_all[total_n1:], len2)
                r1_preds.append(r1_rig_sum + alpha_res * torch.stack([seg.sum() for seg in split1]))
                r2_preds.append(r2_rig_sum + alpha_res * torch.stack([seg.sum() for seg in split2]))

            r1_mean = torch.stack(r1_preds, dim=0).mean(dim=0)
            r2_mean = torch.stack(r2_preds, dim=0).mean(dim=0)

            pred_choice = torch.where(r1_mean >= r2_mean, 0, 1)
            val_correct += int((pred_choice == pref_vec).sum().item())
            val_total += len(batch)

            label_sign = torch.where(pref_vec == 0, 1.0, -1.0)
            val_loss_sum += float(F.softplus(-label_sign * (r1_mean - r2_mean)).sum().item())

    val_acc = (val_correct / max(1, val_total)) * 100.0
    val_loss = val_loss_sum / max(1, val_total)

    # Save model checkpoint
    resolved_save_path = save_path if save_path is not None else os.path.join(save_dir, f"reward_ensemble_Ls_{segment_length_name}.pt")
    os.makedirs(os.path.dirname(os.path.abspath(resolved_save_path)), exist_ok=True)
    torch.save(
        {
            "ensemble_state_dict": ensemble.state_dict(),
            "segment_length": segment_length_name,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "ensemble_size": ensemble_size,
            "alpha_res": alpha_res,
            "val_accuracy": val_acc,
            "val_loss": val_loss,
            "features_state": PREF_STATE_FEATURES,
            "features_action": PREF_ACTION_FEATURES,
        },
        resolved_save_path,
    )
    print(
        f"[Reward Training: L_s={segment_length_name:4s}] "
        f"Validation Accuracy: {val_acc:5.1f}% | Val BT-Loss: {val_loss:.4f} | Saved: {resolved_save_path}"
    )

    metrics = {
        "segment_length": segment_length_name,
        "val_accuracy": val_acc,
        "val_loss": val_loss,
        "save_path": resolved_save_path,
    }
    return ensemble, metrics


def train_reward_model(
    dataset: PreferenceDataset,
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 32,
    state_dim: Optional[int] = None,
    action_dim: int = 5,
    ensemble_size: int = 3,
    save_path: str = "checkpoints/reward_ensemble.pt",
    alpha_res: float = 0.5,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> RewardEnsemble:
    """Wrapper training residual RewardEnsemble compatible with run_pipeline.py."""
    if state_dim is None and len(dataset) > 0:
        state_dim = int(dataset[0]["tau_1_states"].shape[-1])
    inferred_state_dim = state_dim or 10
    ensemble, _ = train_single_reward_ensemble(
        dataset=dataset,
        segment_length_name="comb",
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        state_dim=inferred_state_dim,
        action_dim=action_dim,
        ensemble_size=ensemble_size,
        alpha_res=alpha_res,
        save_path=save_path,
        seed=seed,
        device=device,
    )
    return ensemble


def train_preference_ablation_suite(
    transitions_csv: Optional[str] = None,
    save_dir: str = "checkpoints",
    epochs: int = 18,
    num_pairs_per_horizon: int = 1200,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Executes the full segment length ablation (L_s in {4, 8, 20, comb})
    mirroring Section 4 of LAPSE.
    """
    print("=" * 80)
    print("PREFERENCE LEARNING SUITE: SEGMENT LENGTH ABLATION (L_s in {4, 8, 20, comb})")
    print("=" * 80)

    horizons = [4, 8, 20, "comb"]
    suite_results = {}

    for L_s in horizons:
        name = str(L_s)
        dataset_path = os.path.join(save_dir, f"preference_dataset_Ls_{name}.pt")
        print(f"\n[1/2] Sampling and labeling preference dataset for L_s={name} ({num_pairs_per_horizon} pairs)...")
        dataset = build_preference_dataset(
            trajectories=transitions_csv or os.path.join(CODE_DIR, "data", "synthetic_pome_transitions.csv"),
            num_pairs=num_pairs_per_horizon,
            segment_length=L_s,
            save_path=dataset_path,
            seed=seed,
        )
        print(f"[2/2] Training 3-model reward ensemble for L_s={name}...")
        _, metrics = train_single_reward_ensemble(
            dataset=dataset,
            segment_length_name=name,
            epochs=epochs,
            save_dir=save_dir,
            seed=seed,
        )
        suite_results[name] = metrics

    print("\n" + "=" * 80)
    print("PREFERENCE LEARNING ABLATION SUMMARY (LAPSE Benchmark)")
    print("=" * 80)
    for name, m in suite_results.items():
        print(f"Horizon L_s = {name:4s} -> Validation Ranking Accuracy: {m['val_accuracy']:.2f}% | Loss: {m['val_loss']:.4f}")
    return suite_results


class TriRewardEvaluator:
    """
    Unified Tri-Reward Interface providing:
      1. r_rig    : Rigorous techno-economic state reward (E + B + Q) - (CT + CO) - penalty
      2. r_pref   : Preference model ensemble prediction r_hat_pref(s, a)
      3. r_hybrid : r_rig + alpha * r_pref
    """

    def __init__(
        self,
        ensemble_checkpoint_path: Optional[str] = None,
        alpha: Optional[float] = None,
        device: Optional[torch.device] = None,
    ):
        self.alpha = 0.5 if alpha is None else alpha
        self.dev = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ensemble = None

        if ensemble_checkpoint_path and os.path.exists(ensemble_checkpoint_path):
            ckpt = torch.load(ensemble_checkpoint_path, map_location=self.dev, weights_only=False)
            self.state_dim = ckpt.get("state_dim", 10)
            self.action_dim = ckpt.get("action_dim", 5)
            self.ensemble_size = ckpt.get("ensemble_size", 3)
            if alpha is None and "alpha_res" in ckpt:
                self.alpha = float(ckpt["alpha_res"])
            self.ensemble = RewardEnsemble(
                state_dim=self.state_dim, action_dim=self.action_dim, ensemble_size=self.ensemble_size
            ).to(self.dev)
            self.ensemble.load_state_dict(ckpt["ensemble_state_dict"])
            self.ensemble.eval()
        else:
            self.state_dim = 10
            self.action_dim = 5

    def r_rig(
        self,
        state_dict: Dict[str, float],
        action_vec: np.ndarray,
        next_state_dict: Optional[Dict[str, float]] = None,
        dt_hours: float = 6.0,
    ) -> float:
        curr_s = dict(state_dict)
        if "light_intensity" not in curr_s and len(action_vec) >= 5:
            curr_s["light_intensity"] = float(action_vec[4])
        ns = dict(next_state_dict) if next_state_dict is not None else dict(curr_s)
        b_curr = float(curr_s.get("biomass", curr_s.get("biomass_gL", 0.5)))
        b_next = float(ns.get("biomass", ns.get("biomass_gL", 0.5)))
        r_base, _ = compute_state_reward(
            s_curr_dict=curr_s,
            a_t=action_vec,
            s_next_dict=ns,
            biomass_curr=b_curr,
            biomass_next=b_next,
            dt_hours=dt_hours,
        )
        v_flag, _ = compute_step_violation_magnitude(ns)
        return float(r_base - 50.0 * v_flag)

    def r_pref(
        self,
        state_vec: np.ndarray,
        action_vec: np.ndarray,
    ) -> Tuple[float, float]:
        """Returns (mean_reward, epistemic_std)."""
        if self.ensemble is None:
            # Fallback prior if no model loaded
            return 0.0, 0.0

        s_t = torch.tensor(state_vec, dtype=torch.float32, device=self.dev).unsqueeze(0)
        a_t = torch.tensor(action_vec, dtype=torch.float32, device=self.dev).unsqueeze(0)
        with torch.no_grad():
            mu, std = self.ensemble(s_t, a_t)
        return float(mu.item()), float(std.item())

    def r_hybrid(
        self,
        state_dict: Dict[str, float],
        state_vec: np.ndarray,
        action_vec: np.ndarray,
        next_state_dict: Optional[Dict[str, float]] = None,
        alpha: Optional[float] = None,
    ) -> float:
        eff_alpha = alpha if alpha is not None else self.alpha
        r_r = self.r_rig(state_dict, action_vec, next_state_dict)
        r_p, _ = self.r_pref(state_vec, action_vec)
        return float(r_r + eff_alpha * r_p)

    def evaluate(
        self,
        state_dict: Dict[str, float],
        action_vec: np.ndarray,
        next_state_dict: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, float, float]:
        """Convenience method returning (r_rig, r_pref, r_hybrid)."""
        ns = next_state_dict or state_dict
        r_r = self.r_rig(state_dict, action_vec, ns)

        s_vec = np.array([
            float(state_dict.get("cod", 4000.0)),
            float(state_dict.get("bod", 2000.0)),
            float(state_dict.get("toc", state_dict.get("total_organic_carbon", 1200.0))),
            float(state_dict.get("tss", state_dict.get("total_suspended_solids", 1500.0))),
            float(state_dict.get("ph", 7.5)),
            float(state_dict.get("do", state_dict.get("dissolved_oxygen", 4.0))),
            float(state_dict.get("tds", state_dict.get("total_dissolved_solids", 6000.0))),
            float(state_dict.get("dissolved_ammonia", state_dict.get("nh3", state_dict.get("ammonia", 50.0)))),
            float(state_dict.get("water_temp", state_dict.get("water_temperature", 30.0))),
            float(state_dict.get("biomass", state_dict.get("biomass_gL", 0.5))),
        ], dtype=np.float32)

        r_p, _ = self.r_pref(s_vec, action_vec)
        r_h = float(r_r + self.alpha * r_p)
        return float(r_r), float(r_p), float(r_h)


if __name__ == "__main__":
    train_preference_ablation_suite()
