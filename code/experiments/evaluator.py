"""
Unified Experiment Evaluator & Authentic Multi-Algorithm Trainers.

Consolidates per-algorithm authentic RL / Safe-RLHF / Offline-RL trainers,
closed-loop bioreactor evaluation in POMEReactorEnv, and full evaluation reporting
(previously split between evaluator.py and evaluation/run_evaluation.py).
"""

import os
import sys
import copy
import time
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from config import ACTION_BOUNDS, ACTION_COLUMNS

ACTION_LOW_NP = np.array([ACTION_BOUNDS[c][0] for c in ACTION_COLUMNS], dtype=np.float32)
ACTION_HIGH_NP = np.array([ACTION_BOUNDS[c][1] for c in ACTION_COLUMNS], dtype=np.float32)
from environment.pome_reactor_env import POMEReactorEnv
from models.policy_sft import HurdlePolicy, encode_state_symlog
from models.critics import RewardCritic, FailureCriticEnsemble
from rl.cal_ql import pretrain_calibrated_critic
from rl.sac_kl import SACKLAgent, warmstart_failure_ensemble
from rl.dppo import DPPOAgent
from rl.replay_buffer import MixedReplayBuffer
from preference.labeling_rubric import elicit_preference
from safety.risk_aware_selection import RiskAwareSelector
from safety.primal_dual import PrimalDualSafetyController
from training.train_sft import train_sft
from evaluation.metrics import (
    compute_biochemical_efficacy,
    compute_actuation_efficiency,
    compute_safety_violation_rate,
    verify_permenlhk_compliance,
    compute_step_violation_magnitude,
    compare_with_hcr_baseline,
)


class _SlicedOfflineDataset(Dataset):
    """Lightweight view of OfflineDataset sliced to `state_dim` without copying disk artifacts."""

    def __init__(self, dataset, state_dim: int, cost_weight: float = 1.0):
        self.states = dataset.states[:, :state_dim].float().clone()
        self.next_states = dataset.next_states[:, :state_dim].float().clone()
        self.actions = dataset.actions.float().clone()
        self.rewards = dataset.rewards.float().clone() * float(cost_weight)
        self.dones = dataset.dones.float().clone()
        self.violations = dataset.violations.float().clone()
        self.state_dim = state_dim

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "state": self.states[idx],
            "action": self.actions[idx],
            "reward": self.rewards[idx],
            "next_state": self.next_states[idx],
            "done": self.dones[idx],
            "violation": self.violations[idx],
        }


class _ValueNetwork(nn.Module):
    """State value function V(s) for PPO (GAE baseline) and IQL (expectile regression)."""

    def __init__(self, state_dim: int = 224, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(encode_state_symlog(state)).squeeze(-1)


class _PreferencePairDataset(Dataset):
    """Segment pair dataset matching POMEReactorEnv's exact state dimension for DPPO."""

    def __init__(self, sliced_ds: _SlicedOfflineDataset, num_pairs: int = 240, seg_len: int = 8, seed: int = 42):
        rng = np.random.default_rng(seed)
        s_np = sliced_ds.states.numpy()
        a_np = sliced_ds.actions.numpy()
        r_np = sliced_ds.rewards.numpy()
        v_np = sliced_ds.violations.numpy()
        d_np = sliced_ds.dones.numpy()

        # Split along real episode done boundaries
        episodes = []
        start = 0
        for i, done_flag in enumerate(d_np):
            if done_flag > 0.5 or i == len(d_np) - 1:
                if i - start + 1 >= seg_len:
                    episodes.append((start, i + 1))
                start = i + 1

        if not episodes:
            episodes = [(0, len(s_np))]

        segments = []
        target_pool = max(40, min(num_pairs, len(s_np) // max(1, seg_len // 2)))
        for _ in range(target_pool * 2):
            ep_s, ep_e = episodes[int(rng.integers(0, len(episodes)))]
            if ep_e - ep_s < seg_len:
                continue
            t0 = int(rng.integers(ep_s, ep_e - seg_len + 1))
            t1 = t0 + seg_len
            segments.append((s_np[t0:t1], a_np[t0:t1], float(np.sum(r_np[t0:t1])), int(np.sum(v_np[t0:t1] > 0.5))))

        if len(segments) < 2:
            t1 = min(seg_len, len(s_np))
            segments = [
                (s_np[:t1], a_np[:t1], float(np.sum(r_np[:t1])), int(np.sum(v_np[:t1] > 0.5))),
                (s_np[-t1:], a_np[-t1:], float(np.sum(r_np[-t1:])), int(np.sum(v_np[-t1:] > 0.5))),
            ]

        self.items = []
        for _ in range(num_pairs):
            i, j = int(rng.integers(0, len(segments))), int(rng.integers(0, len(segments)))
            if i == j:
                j = (i + 1) % len(segments)
            seg1, seg2 = segments[i], segments[j]
            # Lexicographic preference: strictly fewer biological violations, then higher cumulative reward
            if seg1[3] != seg2[3]:
                pref = 0 if seg1[3] < seg2[3] else 1
            else:
                pref = 0 if seg1[2] >= seg2[2] else 1

            self.items.append({
                "tau_1_states": torch.tensor(seg1[0], dtype=torch.float32),
                "tau_1_actions": torch.tensor(seg1[1], dtype=torch.float32),
                "tau_2_states": torch.tensor(seg2[0], dtype=torch.float32),
                "tau_2_actions": torch.tensor(seg2[1], dtype=torch.float32),
                "preferred_idx": torch.tensor(pref, dtype=torch.long),
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.items[idx]


def _warmstart_failure_ensemble(
    failure_ensemble: FailureCriticEnsemble,
    sliced_ds: _SlicedOfflineDataset,
    epochs: int = 4,
    batch_size: int = 256,
    lr: float = 1e-3,
    tau: float = 0.01,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Offline warmup of FailureCriticEnsemble with undiscounted Bellman target c_t + (1-c_t)(1-d_t)C_bar(s',a')."""
    if len(sliced_ds) == 0:
        return 0.0
    target_ens = copy.deepcopy(failure_ensemble).to(device).eval()
    opt = optim.AdamW(failure_ensemble.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(sliced_ds, batch_size=batch_size, shuffle=True)
    losses = []

    failure_ensemble.train()
    for _ in range(epochs):
        for batch in loader:
            s = batch["state"].to(device)
            a = batch["action"].to(device)
            ns = batch["next_state"].to(device)
            d = batch["done"].to(device)
            c = batch["violation"].to(device)

            with torch.no_grad():
                # Conservatively evaluate next action near empirical replay next-action or small perturbation
                _, all_next = target_ens(ns, a)
                next_max = all_next.max(dim=-1).values
                target_c = torch.clamp(c + (1.0 - c) * (1.0 - d) * next_max, 0.0, 1.0)

            _, all_pred = failure_ensemble(s, a)
            loss = sum(F.binary_cross_entropy(all_pred[:, m], target_c) for m in range(all_pred.shape[1])) / all_pred.shape[1]

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(failure_ensemble.parameters(), 2.0)
            opt.step()
            losses.append(float(loss.item()))

            with torch.no_grad():
                for tp, sp in zip(target_ens.parameters(), failure_ensemble.parameters()):
                    tp.data.mul_(1.0 - tau).add_(sp.data * tau)

    return float(np.mean(losses)) if losses else 0.0


def evaluate_policy_on_bioreactor(
    policy: HurdlePolicy,
    reward_critic: RewardCritic,
    failure_ensemble: FailureCriticEnsemble,
    num_candidates: int = 16,
    lambda_dual: float = 2.0,
    safety_budget: float = 0.15,
    gate_threshold: float = 0.50,
    device: torch.device = torch.device("cpu"),
    max_steps: int = 64,
    decision_horizon_hours: float = 6.0,
    algorithm: str = "safe_rlhf",
    use_vision: bool = True,
    seed: Optional[int] = None,
    initial_state: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Executes a closed-loop episode rollout in POMEReactorEnv under authentic algorithm semantics.
    Returns full biochemical/regulatory metrics, step-level state history, physical actions, and rewards.
    """
    state_dim = 224 if use_vision else 192
    eff_max_steps = max(1, int(round(384.0 / decision_horizon_hours))) if max_steps == 64 else int(max_steps)

    env = POMEReactorEnv(
        seed=seed if seed is not None else 42,
        max_steps=eff_max_steps,
        decision_horizon_hours=decision_horizon_hours,
    )
    reset_opts = {"initial_state": initial_state} if initial_state is not None else None
    obs, _ = env.reset(seed=seed, options=reset_opts)

    policy.eval()
    reward_critic.eval()
    failure_ensemble.eval()

    use_ras = algorithm in ("safe_rlhf", "lagrangian_sac") and num_candidates > 1
    ras_selector = RiskAwareSelector(
        reward_critic=reward_critic,
        failure_critic_ensemble=failure_ensemble,
        num_candidates=max(1, num_candidates),
        safety_budget=safety_budget,
        default_lambda=lambda_dual,
        gate_threshold=gate_threshold,
    ) if use_ras else None

    s0 = dict(env.current_state)
    s0["biomass"] = float(env.biomass)
    state_history: List[Dict[str, float]] = [s0]
    action_history: List[np.ndarray] = []
    step_rig_rewards: List[float] = []
    step_violations: List[int] = []
    latencies_ms: List[float] = []
    intervention_count = 0

    for _ in range(eff_max_steps):
        s_vec = obs[:state_dim]
        s_t = torch.nan_to_num(torch.tensor(s_vec, dtype=torch.float32, device=device).unsqueeze(0))

        t0 = time.perf_counter()
        with torch.no_grad():
            if ras_selector is not None:
                act_t, ras_info = ras_selector.select_action(s_t, policy, deterministic=True)
                if ras_info.get("was_vetoed", False):
                    intervention_count += 1
            else:
                act_t, _ = policy.sample(s_t, deterministic=True, gate_threshold=gate_threshold)
                act_t = act_t.squeeze(0)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        act_np = np.clip(act_t.detach().cpu().numpy().reshape(-1), ACTION_LOW_NP, ACTION_HIGH_NP).astype(np.float32)

        obs, reward, terminated, truncated, step_info = env.step(act_np)
        s_next = dict(env.current_state)
        s_next["biomass"] = float(env.biomass)

        v_flag, _ = compute_step_violation_magnitude(s_next)
        state_history.append(s_next)
        action_history.append(act_np.copy())
        step_rig_rewards.append(float(reward))
        step_violations.append(int(v_flag))

        if terminated or truncated:
            break

    init_state = state_history[0]
    final_state = state_history[-1]
    acts_arr = np.asarray(action_history, dtype=float) if action_history else np.zeros((1, 5), dtype=float)

    biochem = compute_biochemical_efficacy(init_state, final_state)
    actuation = compute_actuation_efficiency(acts_arr)
    compliance = verify_permenlhk_compliance(state_history, step_hours=decision_horizon_hours)

    cod_rem = float(biochem["cod_removal_pct"])
    total_cost = float(actuation["total_cost_idr"])
    viol_rate_pct = float(np.mean(step_violations) * 100.0) if step_violations else 0.0
    had_violation = bool(any(v > 0 for v in step_violations))

    # Regulatory treatment gate (untreated dumping cannot score high economic/safety scores)
    treatment_gate = float(np.clip(cod_rem / 50.0, 0.0, 1.0))
    cost_score = max(0.0, 100.0 - (total_cost / 200.0)) * treatment_gate
    safety_score = max(0.0, 100.0 - (viol_rate_pct * 3.0)) * treatment_gate
    pareto_score = float((0.40 * cod_rem) + (0.35 * safety_score) + (0.25 * cost_score))

    return {
        "cod_removal_pct": cod_rem,
        "bod_reduction_pct": float(biochem["bod_reduction_pct"]),
        "tss_removal_pct": float(biochem["total_suspended_solids_removal_pct"]),
        "initial_cod_mg_l": float(init_state["cod"]),
        "final_cod_mg_l": float(final_state["cod"]),
        "final_bod_mg_l": float(final_state["bod"]),
        "final_tss_mg_l": float(final_state["total_suspended_solids"]),
        "final_nh3_mg_l": float(final_state["dissolved_ammonia"]),
        "final_ph": float(final_state["ph"]),
        "final_biomass_gl": float(final_state["biomass"]),
        "compliance_day": float(compliance["compliance_day"]),
        "permenlh_compliant": bool(compliance["compliance_check"]),
        "total_cost_idr": total_cost,
        "violation_rate_pct": viol_rate_pct,
        "had_violation": had_violation,
        "intervention_count": int(intervention_count),
        "steps": len(action_history),
        "mean_latency_ms": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
        "pareto_score": pareto_score,
        "cod_trajectory": [float(s["cod"]) for s in state_history],
        "ph_trajectory": [float(s["ph"]) for s in state_history],
        "action_history": [a.tolist() for a in action_history],
        "state_history": state_history,
        "step_rig_rewards": step_rig_rewards,
        "mean_step_reward": float(np.mean(step_rig_rewards)) if step_rig_rewards else 0.0,
    }


def train_algorithm_agent(
    algorithm: str,
    dataset,
    params: Optional[Dict[str, Any]] = None,
    state_dim: int = 224,
    device: torch.device = torch.device("cpu"),
    quick_mode: bool = False,
    seed: int = 42,
    tri_reward=None,
    reward_mode: str = "rig",
) -> Tuple[HurdlePolicy, RewardCritic, FailureCriticEnsemble, Dict[str, List[float]]]:
    """
    Trains one of 9 genuinely distinct algorithms on `dataset` (+ online POMEReactorEnv where applicable):
      - hcr: Supervised Hurdle Chained Regression behavioral cloning baseline
      - dppo: Direct Preference Policy Optimization (paired segments, no BC overwrite)
      - cql: Pure offline Conservative Q-Learning (Cal-QL logsumexp + Q policy extraction)
      - iql: Pure offline Implicit Q-Learning (expectile V + AWR, zero OOD queries)
      - ppo: On-policy Proximal Policy Optimization (Clipped Surrogate + GAE + V(s))
      - ddpg / td3: Twin Delayed Deterministic Policy Gradient on physical action bounds
      - sac: Unconstrained Maximum-Entropy SAC (kl_beta=0, lambda_dual=0, no safety veto)
      - lagrangian_sac: Constrained SAC with single failure critic M=1 + Primal-Dual RAS
      - safe_rlhf: Full Safe Offline-to-Online RLHF (SFT + Cal-QL + SAC-KL + FailureEnsemble M + Primal-Dual + RAS)
    Returns (policy, reward_critic, failure_ensemble, training_history).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    p = params or {}

    algo = algorithm.lower().strip()
    cost_weight = float(p.get("cost_weight", 1.0))
    horizon_h = float(p.get("decision_horizon_hours", 6.0))
    safety_budget = float(p.get("safety_budget", 0.10))
    lambda_init = float(p.get("lambda_dual_init", 2.0))
    eta_dual = float(p.get("eta_dual", 0.05))
    kl_beta = float(p.get("kl_beta", 1.0))
    target_kl = float(p.get("target_kl", 0.05))
    adaptive_kl = bool(p.get("adaptive_kl", True))
    alpha_cql = float(p.get("alpha_cql", 1.0))
    gate_threshold = float(p.get("gate_threshold", 0.50))
    num_candidates = int(p.get("num_candidates_n", 16))
    ensemble_m = 1 if algo == "lagrangian_sac" else int(p.get("ensemble_size_m", 4))

    sliced_ds = _SlicedOfflineDataset(dataset, state_dim=state_dim, cost_weight=cost_weight)
    history: Dict[str, List[float]] = {
        "iterations": [],
        "return_mean": [],
        "critic_loss": [],
        "violations": [],
        "lambda_dual": [],
    }

    reward_critic = RewardCritic(state_dim=state_dim, action_dim=5).to(device)
    failure_ensemble = FailureCriticEnsemble(state_dim=state_dim, action_dim=5, ensemble_size=max(1, ensemble_m)).to(device)

    def _log_eval_point(it: int, pol: HurdlePolicy, rc: RewardCritic, fe: FailureCriticEnsemble, c_loss: float, lam: float, use_ras: bool):
        eval_res = evaluate_policy_on_bioreactor(
            policy=pol,
            reward_critic=rc,
            failure_ensemble=fe,
            num_candidates=num_candidates if use_ras else 1,
            lambda_dual=lam,
            safety_budget=safety_budget,
            gate_threshold=gate_threshold,
            device=device,
            max_steps=32 if quick_mode else 64,
            decision_horizon_hours=horizon_h,
            algorithm=algo,
            use_vision=(state_dim == 224),
            seed=seed + 100 + it,
        )
        history["iterations"].append(int(it))
        history["return_mean"].append(float(np.sum(eval_res["step_rig_rewards"])))
        history["critic_loss"].append(float(c_loss))
        history["violations"].append(float(eval_res["violation_rate_pct"]))
        history["lambda_dual"].append(float(lam))

    # ------------------------------------------------------------------
    # 1. HCR Baseline: Pure Supervised Behavioral Cloning
    # ------------------------------------------------------------------
    if algo == "hcr":
        epochs_sft = 8 if quick_mode else min(20, int(p.get("epochs_sft", 20)))
        policy = train_sft(dataset=sliced_ds, epochs=epochs_sft, state_dim=state_dim, save_path=None, device=device)
        for ep in range(1, 5):
            _log_eval_point(ep, policy, reward_critic, failure_ensemble, c_loss=0.0, lam=0.0, use_ras=False)
        return policy, reward_critic, failure_ensemble, history

    # ------------------------------------------------------------------
    # 2. DPPO: Warm-start SFT + Paired Preference Gradient (No BC Overwrite)
    # ------------------------------------------------------------------
    if algo == "dppo":
        pi_init = train_sft(dataset=sliced_ds, epochs=5 if quick_mode else 10, state_dim=state_dim, save_path=None, device=device)
        dppo_agent = DPPOAgent(state_dim=state_dim, action_dim=5, lambda_reg=0.1, lr=3e-4, init_policy=pi_init, device=device)
        pair_ds = _PreferencePairDataset(sliced_ds, num_pairs=160 if quick_mode else 400, seg_len=8, seed=seed)
        loader = DataLoader(pair_ds, batch_size=32, shuffle=True)
        n_epochs = 6 if quick_mode else 12
        for ep in range(1, n_epochs + 1):
            m = dppo_agent.train_epoch(loader)
            if ep == 1 or ep % 2 == 0 or ep == n_epochs:
                _log_eval_point(ep, dppo_agent.policy, reward_critic, failure_ensemble, c_loss=m.get("loss", 0.0), lam=0.0, use_ras=False)
        return dppo_agent.policy, reward_critic, failure_ensemble, history

    # ------------------------------------------------------------------
    # 3. CQL (Offline Conservative Q-Learning + Policy Extraction)
    # ------------------------------------------------------------------
    if algo == "cql":
        policy = HurdlePolicy(state_dim=state_dim, action_dim=5).to(device)
        cql_epochs = 6 if quick_mode else 12
        reward_critic = pretrain_calibrated_critic(
            q_critic=reward_critic,
            dataset=sliced_ds,
            policy=None,
            epochs=cql_epochs,
            batch_size=128,
            lr=3e-4,
            alpha_cql=alpha_cql,
            device=device,
        )
        p_opt = optim.AdamW(policy.parameters(), lr=3e-4, weight_decay=1e-4)
        loader = DataLoader(sliced_ds, batch_size=128, shuffle=True)
        ext_epochs = 6 if quick_mode else 12
        for ep in range(1, ext_epochs + 1):
            policy.train()
            p_losses = []
            for batch in loader:
                s = batch["state"].to(device)
                a_pi, lp = policy.sample(s, deterministic=False)
                q_val = reward_critic(s, a_pi)
                loss = (-q_val + 0.05 * lp).mean()
                p_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 2.0)
                p_opt.step()
                p_losses.append(float(loss.item()))
            if ep == 1 or ep % 3 == 0 or ep == ext_epochs:
                _log_eval_point(ep, policy, reward_critic, failure_ensemble, c_loss=float(np.mean(p_losses)), lam=0.0, use_ras=False)
        return policy, reward_critic, failure_ensemble, history

    # ------------------------------------------------------------------
    # 4. IQL (Implicit Q-Learning: Expectile V + AWR, Pure Offline)
    # ------------------------------------------------------------------
    if algo == "iql":
        policy = HurdlePolicy(state_dim=state_dim, action_dim=5).to(device)
        v_net = _ValueNetwork(state_dim=state_dim).to(device)
        q_target = copy.deepcopy(reward_critic).to(device).eval()

        v_opt = optim.AdamW(v_net.parameters(), lr=3e-4)
        q_opt = optim.AdamW(reward_critic.parameters(), lr=3e-4)
        p_opt = optim.AdamW(policy.parameters(), lr=3e-4)

        loader = DataLoader(sliced_ds, batch_size=128, shuffle=True)
        n_epochs = 6 if quick_mode else 14
        tau_exp = 0.7
        beta_awr = 3.0
        gamma = 0.99

        for ep in range(1, n_epochs + 1):
            q_losses = []
            for batch in loader:
                s = batch["state"].to(device)
                a = batch["action"].to(device)
                r = batch["reward"].to(device)
                ns = batch["next_state"].to(device)
                d = batch["done"].to(device)

                # Expectile regression on V(s)
                with torch.no_grad():
                    qt_sa = q_target(s, a)
                v_s = v_net(s)
                diff = qt_sa - v_s
                weight_exp = torch.where(diff > 0, tau_exp, 1.0 - tau_exp)
                v_loss = (weight_exp * (diff ** 2)).mean()
                v_opt.zero_grad()
                v_loss.backward()
                v_opt.step()

                # Q backup using V(s') — zero OOD action querying
                with torch.no_grad():
                    q_backup = r + gamma * (1.0 - d) * v_net(ns)
                q_loss = F.mse_loss(reward_critic(s, a), q_backup)
                q_opt.zero_grad()
                q_loss.backward()
                q_opt.step()
                q_losses.append(float(q_loss.item()))

                # Advantage-Weighted Regression for Policy
                with torch.no_grad():
                    adv = q_target(s, a) - v_net(s)
                    w_awr = torch.exp(torch.clamp(beta_awr * adv, -10.0, 4.6))  # <= 100
                lp_data = policy.log_prob(s, a)
                p_loss = (w_awr * (-lp_data)).mean()
                p_opt.zero_grad()
                p_loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 2.0)
                p_opt.step()

                with torch.no_grad():
                    for tp, sp in zip(q_target.parameters(), reward_critic.parameters()):
                        tp.data.mul_(0.995).add_(sp.data * 0.005)

            if ep == 1 or ep % 3 == 0 or ep == n_epochs:
                _log_eval_point(ep, policy, reward_critic, failure_ensemble, c_loss=float(np.mean(q_losses)), lam=0.0, use_ras=False)
        return policy, reward_critic, failure_ensemble, history

    # ------------------------------------------------------------------
    # 5. PPO (On-Policy Clipped Surrogate + GAE, Unconstrained)
    # ------------------------------------------------------------------
    if algo == "ppo":
        policy = HurdlePolicy(state_dim=state_dim, action_dim=5).to(device)
        v_net = _ValueNetwork(state_dim=state_dim).to(device)
        p_opt = optim.AdamW(policy.parameters(), lr=3e-4)
        v_opt = optim.AdamW(v_net.parameters(), lr=5e-4)

        n_iters = 5 if quick_mode else 10
        ep_steps = 32 if quick_mode else 64
        env = POMEReactorEnv(seed=seed, max_steps=ep_steps, decision_horizon_hours=horizon_h)
        gamma, gae_lam, clip_eps = 0.99, 0.95, 0.2

        for it in range(1, n_iters + 1):
            obs_buf, act_buf, lp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
            obs, _ = env.reset(seed=seed + it)
            policy.eval()
            for _ in range(ep_steps):
                s_np = obs[:state_dim]
                s_t = torch.tensor(s_np, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    a_t, lp_t = policy.sample(s_t, deterministic=False)
                    v_t = v_net(s_t)
                a_np = np.clip(a_t.squeeze(0).cpu().numpy(), ACTION_LOW_NP, ACTION_HIGH_NP)
                obs_next, r, term, trunc, _ = env.step(a_np)
                done = term or trunc

                obs_buf.append(s_np)
                act_buf.append(a_np)
                lp_buf.append(float(lp_t.item()))
                rew_buf.append(float(r))
                val_buf.append(float(v_t.item()))
                done_buf.append(float(done))

                obs = obs_next
                if done:
                    break

            # Compute GAE
            with torch.no_grad():
                last_v = 0.0 if done_buf[-1] > 0.5 else float(v_net(torch.tensor(obs[:state_dim], dtype=torch.float32, device=device).unsqueeze(0)).item())
            advs = np.zeros(len(rew_buf), dtype=np.float32)
            gae = 0.0
            for t in reversed(range(len(rew_buf))):
                nv = last_v if t == len(rew_buf) - 1 else val_buf[t + 1]
                nd = 1.0 - done_buf[t]
                delta = rew_buf[t] + gamma * nv * nd - val_buf[t]
                gae = delta + gamma * gae_lam * nd * gae
                advs[t] = gae
            rets = advs + np.asarray(val_buf, dtype=np.float32)
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)

            s_all = torch.tensor(np.asarray(obs_buf), dtype=torch.float32, device=device)
            a_all = torch.tensor(np.asarray(act_buf), dtype=torch.float32, device=device)
            old_lp_all = torch.tensor(np.asarray(lp_buf), dtype=torch.float32, device=device)
            adv_all = torch.tensor(advs, dtype=torch.float32, device=device)
            ret_all = torch.tensor(rets, dtype=torch.float32, device=device)

            policy.train()
            v_losses = []
            for _ in range(4):
                new_lp = policy.log_prob(s_all, a_all)
                ratio = torch.exp(torch.clamp(new_lp - old_lp_all, -10.0, 10.0))
                surr1 = ratio * adv_all
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_all
                p_loss = -torch.min(surr1, surr2).mean() + 0.01 * new_lp.mean()

                p_opt.zero_grad()
                p_loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                p_opt.step()

                v_loss = F.smooth_l1_loss(v_net(s_all), ret_all)
                v_opt.zero_grad()
                v_loss.backward()
                v_opt.step()
                v_losses.append(float(v_loss.item()))

            if it == 1 or it % 2 == 0 or it == n_iters:
                _log_eval_point(it, policy, reward_critic, failure_ensemble, c_loss=float(np.mean(v_losses)), lam=0.0, use_ras=False)
        return policy, reward_critic, failure_ensemble, history

    # ------------------------------------------------------------------
    # 6. DDPG / TD3 (Twin Delayed Deterministic PG on Physical Action Bounds)
    # ------------------------------------------------------------------
    if algo in ("ddpg", "td3"):
        policy = HurdlePolicy(state_dim=state_dim, action_dim=5).to(device)
        q1 = reward_critic
        q2 = RewardCritic(state_dim=state_dim, action_dim=5).to(device)
        q1_t = copy.deepcopy(q1).to(device).eval()
        q2_t = copy.deepcopy(q2).to(device).eval()
        pi_t = copy.deepcopy(policy).to(device).eval()

        q_opt = optim.AdamW(list(q1.parameters()) + list(q2.parameters()), lr=1e-3)
        p_opt = optim.AdamW(policy.parameters(), lr=3e-4)

        buf = MixedReplayBuffer(state_dim=state_dim, action_dim=5, offline_ratio=0.25)
        buf.load_offline_data(sliced_ds)

        n_eps = 5 if quick_mode else 10
        ep_steps = 32 if quick_mode else 64
        env = POMEReactorEnv(seed=seed, max_steps=ep_steps, decision_horizon_hours=horizon_h)
        act_high_t = torch.tensor(ACTION_HIGH_NP, dtype=torch.float32, device=device)

        grad_step = 0
        c_losses = []
        for ep in range(1, n_eps + 1):
            obs, _ = env.reset(seed=seed + ep)
            for _ in range(ep_steps):
                s_np = obs[:state_dim]
                s_t = torch.tensor(s_np, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    a_det, _ = policy.sample(s_t, deterministic=True)
                noise = np.random.normal(0.0, 0.10, size=5) * ACTION_HIGH_NP
                a_np = np.clip(a_det.squeeze(0).cpu().numpy() + noise, ACTION_LOW_NP, ACTION_HIGH_NP).astype(np.float32)

                obs_next, r, term, trunc, step_info = env.step(a_np)
                done = term or trunc
                buf.add_online(s_np, a_np, float(r) * cost_weight, obs_next[:state_dim], float(done), float(step_info.get("violation", 0)))
                obs = obs_next

                batch = buf.sample(batch_size=128)
                bs = batch["states"].to(device)
                ba = batch["actions"].to(device)
                br = batch["rewards"].to(device)
                bns = batch["next_states"].to(device)
                bd = batch["dones"].to(device)

                with torch.no_grad():
                    na, _ = pi_t.sample(bns, deterministic=True)
                    t_noise = torch.clamp(torch.randn_like(na) * 0.1 * act_high_t, -0.25 * act_high_t, 0.25 * act_high_t)
                    na = torch.clamp(na + t_noise, torch.zeros_like(act_high_t), act_high_t)
                    tq = br + 0.99 * (1.0 - bd) * torch.min(q1_t(bns, na), q2_t(bns, na))

                q_loss = F.mse_loss(q1(bs, ba), tq) + F.mse_loss(q2(bs, ba), tq)
                q_opt.zero_grad()
                q_loss.backward()
                q_opt.step()
                c_losses.append(float(q_loss.item()))

                grad_step += 1
                if grad_step % 2 == 0:
                    a_curr, _ = policy.sample(bs, deterministic=True)
                    p_loss = -q1(bs, a_curr).mean()
                    p_opt.zero_grad()
                    p_loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 2.0)
                    p_opt.step()

                    with torch.no_grad():
                        for t_m, s_m in [(q1_t, q1), (q2_t, q2), (pi_t, policy)]:
                            for tp, sp in zip(t_m.parameters(), s_m.parameters()):
                                tp.data.mul_(0.995).add_(sp.data * 0.005)
                if done:
                    break
            if ep == 1 or ep % 2 == 0 or ep == n_eps:
                _log_eval_point(ep, policy, q1, failure_ensemble, c_loss=float(np.mean(c_losses[-50:])), lam=0.0, use_ras=False)
        return policy, q1, failure_ensemble, history

    # ------------------------------------------------------------------
    # 7. Unconstrained SAC vs 8. Lagrangian-SAC (M=1) vs 9. Full Safe RLHF
    # ------------------------------------------------------------------
    if algo == "sac":
        pi_sft = None
        pretrained_q = None
        agent = SACKLAgent(
            state_dim=state_dim,
            action_dim=5,
            pi_sft=None,
            pretrained_q_critic=None,
            failure_ensemble=failure_ensemble,
            num_q=2,
            ensemble_size_m=1,
            kl_beta_init=0.0,
            adaptive_kl=False,
        )
        eff_offline_ratio = 0.20
        safety_ctrl = None
        ras_sel = None
    elif algo == "lagrangian_sac":
        epochs_sft = 6 if quick_mode else min(12, int(p.get("epochs_sft", 12)))
        pi_sft = train_sft(dataset=sliced_ds, epochs=epochs_sft, state_dim=state_dim, save_path=None, device=device)
        agent = SACKLAgent(
            state_dim=state_dim,
            action_dim=5,
            pi_sft=pi_sft,
            pretrained_q_critic=None,
            failure_ensemble=failure_ensemble,
            num_q=2,
            ensemble_size_m=1,
            kl_beta_init=kl_beta,
            target_kl=target_kl,
            adaptive_kl=adaptive_kl,
        )
        warmstart_failure_ensemble(agent.failure_ensemble, sliced_ds, policy=pi_sft, epochs=3 if quick_mode else 5, device=device)
        agent.target_failure_ensemble.load_state_dict(agent.failure_ensemble.state_dict())
        eff_offline_ratio = 0.50
        safety_ctrl = PrimalDualSafetyController(constraint_level=safety_budget, dual_step_size=eta_dual, lambda_init=lambda_init)
        ras_sel = RiskAwareSelector(
            reward_critic=agent.primary_reward_critic,
            failure_critic_ensemble=agent.failure_ensemble,
            num_candidates=num_candidates,
            safety_budget=safety_budget,
            default_lambda=lambda_init,
            gate_threshold=gate_threshold,
        )
    else:
        # Full Safe Offline-to-Online RLHF
        epochs_sft = 8 if quick_mode else min(15, int(p.get("epochs_sft", 15)))
        pi_sft = train_sft(dataset=sliced_ds, epochs=epochs_sft, state_dim=state_dim, save_path=None, device=device)
        cal_epochs = 6 if quick_mode else 12
        pretrained_q = pretrain_calibrated_critic(
            q_critic=reward_critic,
            dataset=sliced_ds,
            policy=pi_sft,
            epochs=cal_epochs,
            batch_size=128,
            lr=3e-4,
            alpha_cql=alpha_cql,
            device=device,
        )
        agent = SACKLAgent(
            state_dim=state_dim,
            action_dim=5,
            pi_sft=pi_sft,
            pretrained_q_critic=pretrained_q,
            failure_ensemble=failure_ensemble,
            num_q=4,
            ensemble_size_m=max(1, ensemble_m),
            kl_beta_init=kl_beta,
            target_kl=target_kl,
            adaptive_kl=adaptive_kl,
        )
        warmstart_failure_ensemble(agent.failure_ensemble, sliced_ds, policy=pi_sft, epochs=4 if quick_mode else 6, device=device)
        agent.target_failure_ensemble.load_state_dict(agent.failure_ensemble.state_dict())
        eff_offline_ratio = 0.30
        safety_ctrl = PrimalDualSafetyController(constraint_level=safety_budget, dual_step_size=eta_dual, lambda_init=lambda_init)
        ras_sel = RiskAwareSelector(
            reward_critic=agent.primary_reward_critic,
            failure_critic_ensemble=agent.failure_ensemble,
            num_candidates=num_candidates,
            safety_budget=safety_budget,
            default_lambda=lambda_init,
            gate_threshold=gate_threshold,
        )

    agent.to(device)
    buf = MixedReplayBuffer(state_dim=state_dim, action_dim=5, offline_ratio=eff_offline_ratio)
    buf.load_offline_data(sliced_ds)

    steps_target = int(p.get("steps_online", 3000))
    n_eps = 5 if quick_mode else max(6, min(24, max(6, steps_target // 160)))
    ep_steps = 32 if quick_mode else 64
    env = POMEReactorEnv(seed=seed, max_steps=ep_steps, decision_horizon_hours=horizon_h)

    recent_c_losses = []
    pre_crash_indices = []
    for ep in range(1, n_eps + 1):
        obs, _ = env.reset(seed=seed + ep)
        ep_viol = False
        ep_start_ptr = buf.on_ptr
        ep_len = 0
        if ras_sel is not None and safety_ctrl is not None:
            ras_sel.lambda_dual = safety_ctrl.lambda_dual

        for _ in range(ep_steps):
            s_np = obs[:state_dim]
            s_t = torch.tensor(s_np, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                if ras_sel is not None:
                    a_t, _ = ras_sel.select_action(s_t, agent.actor, deterministic=False)
                else:
                    a_t, _ = agent.actor.sample(s_t, deterministic=False)
                    a_t = a_t.squeeze(0)
            a_np = np.clip(a_t.cpu().numpy(), ACTION_LOW_NP, ACTION_HIGH_NP).astype(np.float32)
            if ras_sel is not None and ep <= 2 and np.random.rand() < 0.25:
                dim = int(np.random.choice([0, 1, 3, 4]))
                a_np[dim] = float(np.random.uniform(ACTION_LOW_NP[dim], ACTION_HIGH_NP[dim]))

            s_curr_dict = dict(env.current_state)
            s_curr_dict["biomass"] = float(env.biomass)
            obs_next, r, term, trunc, step_info = env.step(a_np)
            s_next_dict = dict(env.current_state)
            s_next_dict["biomass"] = float(env.biomass)
            done = term or trunc
            v_flag = float(step_info.get("violation", 0))
            if v_flag > 0.5:
                ep_viol = True

            eff_r = float(r)
            if tri_reward is not None and reward_mode in ("pref", "hybrid"):
                _, r_p, r_h = tri_reward.evaluate(s_curr_dict, a_np, s_next_dict)
                eff_r = float(r_p - 50.0 * v_flag) if reward_mode == "pref" else float(r_h)

            buf.add_online(s_np, a_np, eff_r * cost_weight, obs_next[:state_dim], float(done), v_flag)
            ep_len += 1
            obs = obs_next

            if buf.on_size >= 64:
                batch = buf.sample_with_priority(
                    batch_size=128,
                    priority_indices=pre_crash_indices if safety_ctrl is not None else None,
                    num_priority=16,
                )
                bs = batch["states"].to(device)
                ba = batch["actions"].to(device)
                br = batch["rewards"].to(device)
                bns = batch["next_states"].to(device)
                bd = batch["dones"].to(device)
                bc = batch["violations"].to(device)

                c_loss = agent.update_critics(bs, ba, br, bns, bd)
                recent_c_losses.append(float(c_loss))
                if safety_ctrl is not None:
                    agent.update_failure_critics(bs, ba, bns, bd, bc)
                    lam_curr = safety_ctrl.lambda_dual
                else:
                    lam_curr = 0.0
                agent.update_actor(bs[:128], lambda_dual=lam_curr)
                agent.soft_update()

            if done:
                break

        if ep_viol and ep_len > 0:
            for k in range(min(5, ep_len)):
                pre_crash_indices.append((ep_start_ptr + ep_len - 1 - k) % buf.max_size)

        if safety_ctrl is not None:
            safety_ctrl.record_episode_result(ep_viol)
            safety_ctrl.dual_update()

        if ep == 1 or ep % 2 == 0 or ep == n_eps:
            cur_lam = float(safety_ctrl.lambda_dual) if safety_ctrl is not None else 0.0
            _log_eval_point(
                ep,
                agent.actor,
                agent.primary_reward_critic,
                agent.failure_ensemble,
                c_loss=float(np.mean(recent_c_losses[-64:])) if recent_c_losses else 0.0,
                lam=cur_lam,
                use_ras=(ras_sel is not None),
            )

    final_q = agent.primary_reward_critic
    final_f = agent.failure_ensemble
    return agent.actor, final_q, final_f, history


def run_experiment_evaluation(
    exp_config: Dict[str, Any],
    dataset,
    device: torch.device,
    quick_mode: bool = False,
) -> Dict[str, Any]:
    """Executes one experiment config with authentic per-algorithm training + multi-seed rollout evaluation."""
    t_start = time.time()
    exp_id = exp_config["id"]
    p = exp_config["params"]
    use_vision = bool(p.get("use_vision", True))
    state_dim = 224 if use_vision else 192
    algo = p.get("algorithm", "safe_rlhf")

    policy, q_critic, fail_ens, _ = train_algorithm_agent(
        algorithm=algo,
        dataset=dataset,
        params=p,
        state_dim=state_dim,
        device=device,
        quick_mode=quick_mode,
        seed=42,
    )
    train_time = time.time() - t_start

    eval_seeds = [2026] if quick_mode else [2026, 2027, 2028]
    rollouts = []
    for sd in eval_seeds:
        res = evaluate_policy_on_bioreactor(
            policy=policy,
            reward_critic=q_critic,
            failure_ensemble=fail_ens,
            num_candidates=int(p.get("num_candidates_n", 16)),
            lambda_dual=float(p.get("lambda_dual_init", 2.0)),
            safety_budget=float(p.get("safety_budget", 0.10)),
            gate_threshold=float(p.get("gate_threshold", 0.50)),
            device=device,
            decision_horizon_hours=float(p.get("decision_horizon_hours", 6.0)),
            algorithm=algo,
            use_vision=use_vision,
            seed=sd,
        )
        rollouts.append(res)

    return {
        "exp_id": exp_id,
        "name": exp_config["name"],
        "axis": exp_config["axis"],
        "description": exp_config["description"],
        "params": p,
        "cod_removal_pct": float(np.mean([r["cod_removal_pct"] for r in rollouts])),
        "compliance_day": float(np.mean([r["compliance_day"] for r in rollouts])),
        "total_cost_idr": float(np.mean([r["total_cost_idr"] for r in rollouts])),
        "violation_rate_pct": float(np.mean([r["violation_rate_pct"] for r in rollouts])),
        "mean_latency_ms": float(np.mean([r["mean_latency_ms"] for r in rollouts])),
        "pareto_score": float(np.mean([r["pareto_score"] for r in rollouts])),
        "train_time_seconds": float(train_time),
        "initial_cod_mg_l": float(rollouts[0]["initial_cod_mg_l"]),
        "final_cod_mg_l": float(np.mean([r["final_cod_mg_l"] for r in rollouts])),
        "cod_trajectory": rollouts[0]["cod_trajectory"],
        "ph_trajectory": rollouts[0]["ph_trajectory"],
    }


def load_checkpoint_strict(
    model: nn.Module,
    ckpt_path: str,
    keys_to_try: Optional[List[str]] = None,
):
    """Strictly loads a trained checkpoint without silent random-weight fallback."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"[Strict Evaluation Error] Checkpoint '{ckpt_path}' tidak ditemukan. "
            "Evaluasi dengan bobot acak dilarang. Latih model terlebih dahulu."
        )
    keys = keys_to_try or [
        "model_state_dict",
        "policy_state_dict",
        "q_state_dict",
        "critic_state_dict",
        "failure_ensemble_state_dict",
        "ensemble_state_dict",
    ]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        for k in keys:
            if k in ckpt:
                model.load_state_dict(ckpt[k])
                return
    model.load_state_dict(ckpt)


def _load_checkpoint_strict(model: nn.Module, ckpt_path: str, keys_to_try: Optional[List[str]] = None):
    return load_checkpoint_strict(model, ckpt_path, keys_to_try)


def run_full_evaluation(
    policy_path: str = os.path.join(CODE_DIR, "checkpoints/safe_rlhf_policy.pt"),
    reward_critic_path: str = os.path.join(CODE_DIR, "checkpoints/safe_rlhf_reward_critic.pt"),
    failure_ensemble_path: str = os.path.join(CODE_DIR, "checkpoints/safe_rlhf_failure_ensemble.pt"),
    num_eval_episodes: int = 5,
    output_dir: str = os.path.join(CODE_DIR, "evaluation"),
) -> Dict[str, Any]:
    """
    Comprehensive full-controller evaluation consolidated from evaluation/run_evaluation.py.
    Strictly loads trained policy + Q-critic + failure critic checkpoints and compares against HCR K8.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")

    policy = HurdlePolicy(state_dim=224, action_dim=5).to(device)
    _load_checkpoint_strict(policy, policy_path, ["model_state_dict", "policy_state_dict"])

    q_critic = RewardCritic(state_dim=224, action_dim=5).to(device)
    _load_checkpoint_strict(q_critic, reward_critic_path, ["model_state_dict", "q_state_dict", "critic_state_dict"])

    failure_ensemble = FailureCriticEnsemble(state_dim=224, action_dim=5, ensemble_size=4).to(device)
    _load_checkpoint_strict(failure_ensemble, failure_ensemble_path, ["model_state_dict", "failure_ensemble_state_dict"])

    episode_records = []
    all_violations = []
    total_interventions = 0
    total_steps = 0

    for ep in range(num_eval_episodes):
        res = evaluate_policy_on_bioreactor(
            policy=policy,
            reward_critic=q_critic,
            failure_ensemble=failure_ensemble,
            num_candidates=16,
            lambda_dual=2.0,
            safety_budget=0.10,
            gate_threshold=0.50,
            device=device,
            max_steps=64,
            decision_horizon_hours=6.0,
            algorithm="safe_rlhf",
            use_vision=True,
            seed=2026 + ep,
        )
        all_violations.append(bool(res["had_violation"]))
        total_interventions += int(res["intervention_count"])
        total_steps += int(res["steps"])
        episode_records.append({
            "episode": ep + 1,
            "steps": res["steps"],
            "days": res["steps"] * 6.0 / 24.0,
            "compliance_day": res["compliance_day"],
            "permenlh_compliant": res["permenlh_compliant"],
            "cod_removal_pct": res["cod_removal_pct"],
            "bod_reduction_pct": res["bod_reduction_pct"],
            "tss_removal_pct": res["tss_removal_pct"],
            "final_cod_mg_l": res["final_cod_mg_l"],
            "final_ph": res["final_ph"],
            "biomass_gl": res["final_biomass_gl"],
            "total_cost_idr": res["total_cost_idr"],
            "violation": res["had_violation"],
        })

    df_results = pd.DataFrame(episode_records)
    csv_path = os.path.join(output_dir, "results.csv")
    df_results.to_csv(csv_path, index=False)

    mean_cod_removal = float(df_results["cod_removal_pct"].mean())
    mean_cost = float(df_results["total_cost_idr"].mean())
    mean_compliance_day = float(df_results["compliance_day"].mean())
    mean_biomass = float(df_results["biomass_gl"].mean())
    safety_stats = compute_safety_violation_rate(all_violations, total_interventions, total_steps)

    summary_stats = {
        "compliance_day": mean_compliance_day,
        "cod_removal_pct": mean_cod_removal,
        "bod_reduction_pct": float(df_results["bod_reduction_pct"].mean()),
        "tss_removal_pct": float(df_results["tss_removal_pct"].mean()),
        "biomass_gl": mean_biomass,
        "total_cost_idr": mean_cost,
        "violations": safety_stats["safety_violation_count"],
        "violation_rate": safety_stats["violation_rate"],
        "intervention_rate": safety_stats["intervention_rate"],
    }

    benchmark_table = compare_with_hcr_baseline(summary_stats)
    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Safe Offline-to-Online RLHF Evaluation Report\n\n")
        f.write("## 1. Executive Summary & Baseline Comparison\n\n")
        f.write("Evaluation results comparing the Safe RLHF controller against published predecessor runs on the 2.5 L POME photobioreactor:\n\n")
        f.write(benchmark_table.to_markdown(index=False))
        f.write("\n\n")
        f.write("## 2. Biochemical Efficacy (Pollutant Removal)\n\n")
        f.write(f"- **COD Removal Percentage:** {mean_cod_removal:.2f}%\n")
        f.write(f"- **BOD Removal Percentage:** {summary_stats['bod_reduction_pct']:.2f}%\n")
        f.write(f"- **TSS Removal Percentage:** {summary_stats['tss_removal_pct']:.2f}%\n")
        f.write(f"- **Final Algal Biomass:** {mean_biomass:.2f} g/L (Chlorella vulgaris)\n\n")
        f.write("## 3. Actuation Efficiency & Chemical Costs\n\n")
        f.write(f"- **Average Operating Cost:** IDR {mean_cost:,.0f} per batch\n")
        f.write(f"- **Cost vs Hurdle Baseline (K8, IDR 6,170):** {((6170.0 - mean_cost) / 6170.0) * 100:+.2f}%\n")
        f.write(f"- **Cost vs Best Human (K5, IDR 9,872):** {((9872.0 - mean_cost) / 9872.0) * 100:+.2f}%\n\n")
        f.write("## 4. Safety & Regulatory Compliance (PerMenLHK No. 5/2014)\n\n")
        f.write(f"- **Days to Full Compliance:** {mean_compliance_day:.1f} days (Target: <= 5 days)\n")
        f.write(f"- **Safety Violation Rate (VR):** {safety_stats['violation_rate'] * 100:.1f}%\n")
        f.write(f"- **Risk-Aware Selection Intervention Rate (IR):** {safety_stats['intervention_rate'] * 100:.1f}%\n")
        f.write(f"- **Total Dangerous Actions Blocked / Rejected:** {total_interventions}\n")

    print(f"Generated evaluation report: {report_path}")
    print(f"Generated results CSV: {csv_path}")
    return summary_stats
