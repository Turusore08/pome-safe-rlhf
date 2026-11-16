"""
End-to-End Orchestrator for the Safe Offline-to-Online RLHF POME Control System.

Executes the full validated pipeline:
1. Real Cultivation Dataset Verification (ITS-POME K1–K16)
2. Telemetry Cleaning & ConvNeXt Vision-Sensor Corpus D_hist (224-dim)
3. (Optional / Automatic) Physics-Informed Hybrid ODE & 14-Fold LOCO Validation
4. SFT Behavioral Cloning Reference Policy Pre-training (pi_sft)
5. Residual Bradley-Terry Preference Reward Ensemble Suite (r_hybrid = r_rig + alpha * r_res, L_s ablations)
6. Direct Preference Policy Optimization (DPPO on 224-dim POMEReactorEnv space)
7. Cal-QL Critic Pre-training -> SAC-KL + FailureCriticEnsemble + Primal-Dual + RAS Online RLPD Loop
8. Edge Optimization: ONNX Export, INT8 Quantization & Latency Benchmark
9. Consolidated Multi-Episode Evaluation & Lineage Registration
"""

import os
import sys
import time
import shutil
import argparse
import numpy as np
import torch

CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from config import ACTION_BOUNDS, ACTION_COLUMNS, validate_real_dataset_paths
from data.offline_dataset import assemble_offline_corpus
from training.train_sft import train_sft
from training.train_hybrid_ode_loco import run_hybrid_ode_loco
from preference.train_reward import train_preference_ablation_suite
from rl.train_dppo import main as train_dppo_main
from rl.cal_ql import pretrain_calibrated_critic
from rl.sac_kl import SACKLAgent, warmstart_failure_ensemble
from rl.replay_buffer import MixedReplayBuffer
from models.critics import RewardCritic, FailureCriticEnsemble
from environment.pome_reactor_env import POMEReactorEnv
from safety.primal_dual import PrimalDualSafetyController
from safety.risk_aware_selection import RiskAwareSelector
from deployment.export_onnx import export_policy
from deployment.quantize import apply_int8_quantization, compare_quantization_performance
from deployment.computational_analysis import benchmark_end_to_end_latency
from experiments.evaluator import run_full_evaluation
from lineage.manifest import ArtifactLineageTracker


def run_pipeline(
    skip_loco: bool = False,
    force_loco: bool = False,
    force_data: bool = False,
    sft_epochs: int = 20,
    pref_epochs: int = 15,
    dppo_epochs: int = 12,
    calql_epochs: int = 12,
    online_episodes: int = 24,
    seed: int = 42,
):
    print("=" * 72)
    print("SAFE OFFLINE-TO-ONLINE RLHF CONTROL PIPELINE (POME 2.5L Bioreactor)")
    print("=" * 72)

    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tracker = ArtifactLineageTracker()

    ckpt_dir = os.path.join(CODE_ROOT, "checkpoints")
    data_dir = os.path.join(CODE_ROOT, "data")
    deploy_dir = os.path.join(CODE_ROOT, "deployment", "models")
    eval_dir = os.path.join(CODE_ROOT, "evaluation")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(deploy_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # Stage 1: Verify Real Cultivation Dataset (K1–K16)
    print("\n[Stage 1/9] Verifying Real Cultivation Dataset (ITS-POME K1–K16)...")
    validate_real_dataset_paths()
    print("✓ Real cultivation dataset found.")

    # Stage 2: Load or Assemble D_hist (224-dim multimodal offline corpus)
    d_hist_path = os.path.join(data_dir, "d_hist.pt")
    print(f"\n[Stage 2/9] Loading/Assembling 224-dim Corpus D_hist ({d_hist_path})...")
    d_hist = assemble_offline_corpus(save_path=d_hist_path, force_rebuild=force_data) if force_data else assemble_offline_corpus(save_path=d_hist_path)

    # Stage 3: Hybrid ODE & 14-Fold LOCO (run if requested or CSV missing)
    loco_csv = os.path.join(eval_dir, "hybrid_ode_loco.csv")
    if not skip_loco and (force_loco or not os.path.exists(loco_csv)):
        print("\n[Stage 3/9] Physics-Informed Hybrid ODE & 14-Fold LOCO Validation...")
        run_hybrid_ode_loco(epochs=60, force_rerun=force_loco)
    else:
        print(f"\n[Stage 3/9] Skipping HybridODE LOCO (skip_loco={skip_loco}, csv_exists={os.path.exists(loco_csv)}).")

    # Stage 4: Pre-train Hurdle SFT Reference Policy (pi_sft)
    sft_path = os.path.join(ckpt_dir, "sft_policy.pt")
    print(f"\n[Stage 4/9] Pre-training Reference HurdlePolicy pi_sft ({sft_epochs} epochs)...")
    pi_sft = train_sft(dataset=d_hist, epochs=sft_epochs, state_dim=224, action_dim=5, save_path=sft_path, device=device)

    # Stage 5: Preference-Based Residual Reward Modeling (r_hybrid = r_rig + alpha * r_res)
    print(f"\n[Stage 5/9] Training Residual Bradley-Terry Reward Ensemble Suite (L_s in {{4, 8, 20, comb}})...")
    pref_results = train_preference_ablation_suite(
        transitions_csv=os.path.join(data_dir, "synthetic_pome_transitions.csv"),
        save_dir=ckpt_dir,
        epochs=pref_epochs,
        num_pairs_per_horizon=800,
        seed=seed,
    )
    comb_ckpt = os.path.join(ckpt_dir, "reward_ensemble_Ls_comb.pt")
    reward_ckpt = os.path.join(ckpt_dir, "reward_ensemble.pt")
    shutil.copy2(comb_ckpt, reward_ckpt)
    shutil.copy2(
        os.path.join(ckpt_dir, "preference_dataset_Ls_comb.pt"),
        os.path.join(data_dir, "preference_dataset.pt"),
    )
    print(f"✓ Synced primary preference reward ensemble to {reward_ckpt}")

    # Stage 6: Direct Preference Policy Optimization (DPPO in 224-dim env space)
    print(f"\n[Stage 6/9] Training Direct Preference Policy Optimization (DPPO, {dppo_epochs} epochs)...")
    dppo_path = train_dppo_main(epochs=dppo_epochs, num_pairs=600, segment_length=8, seed=seed)

    # Stage 7: Cal-QL Offline Pre-training -> SAC-KL + FailureEnsemble + Primal-Dual Online RLPD Loop
    print(f"\n[Stage 7/9] Cal-QL Critic ({calql_epochs} epochs) & SAC-KL Safe RLPD ({online_episodes} episodes)...")
    cal_q = RewardCritic(state_dim=224, action_dim=5).to(device)
    cal_q = pretrain_calibrated_critic(
        q_critic=cal_q,
        dataset=d_hist,
        policy=pi_sft,
        epochs=calql_epochs,
        batch_size=128,
        lr=3e-4,
        alpha_cql=1.0,
        device=device,
    )

    fail_ens = FailureCriticEnsemble(state_dim=224, action_dim=5, ensemble_size=4).to(device)
    agent = SACKLAgent(
        state_dim=224,
        action_dim=5,
        pi_sft=pi_sft,
        pretrained_q_critic=cal_q,
        failure_ensemble=fail_ens,
        num_q=4,
        ensemble_size_m=4,
        kl_beta_init=0.04,
        target_kl=1.20,
        adaptive_kl=True,
    ).to(device)

    warmstart_loss = warmstart_failure_ensemble(
        agent.failure_ensemble,
        d_hist,
        policy=pi_sft,
        epochs=6,
        device=device,
    )
    agent.target_failure_ensemble.load_state_dict(agent.failure_ensemble.state_dict())
    print(f"FailureCriticEnsemble offline warmup BCE loss: {warmstart_loss:.4f}")

    replay_buf = MixedReplayBuffer(state_dim=224, action_dim=5, offline_ratio=0.30)
    replay_buf.load_offline_data(d_hist)

    safety_ctrl = PrimalDualSafetyController(
        constraint_level=0.05,
        dual_step_size=0.50,
        lambda_init=2.0,
        lambda_max=25.0,
    )
    ras_selector = RiskAwareSelector(
        reward_critic=agent.primary_reward_critic,
        failure_critic_ensemble=agent.failure_ensemble,
        num_candidates=16,
        safety_budget=0.10,
        default_lambda=safety_ctrl.lambda_dual,
        gate_threshold=0.50,
    )

    action_low = np.array([ACTION_BOUNDS[c][0] for c in ACTION_COLUMNS], dtype=np.float32)
    action_high = np.array([ACTION_BOUNDS[c][1] for c in ACTION_COLUMNS], dtype=np.float32)
    env = POMEReactorEnv(seed=seed, max_steps=64, decision_horizon_hours=6.0)
    pre_crash_indices = []

    for ep in range(1, online_episodes + 1):
        obs, _ = env.reset(seed=seed + ep)
        ep_viol = False
        ep_ret = 0.0
        ep_start_ptr = replay_buf.on_ptr
        ep_len = 0
        ras_selector.lambda_dual = safety_ctrl.lambda_dual
        ep_c_losses = []

        for _ in range(64):
            s_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                a_t, _ = ras_selector.select_action(s_t, agent.actor, deterministic=False)
            a_np = np.clip(a_t.cpu().numpy(), action_low, action_high).astype(np.float32)
            if ep <= 2 and np.random.rand() < 0.25:
                dim = int(np.random.choice([0, 1, 3, 4]))
                a_np[dim] = float(np.random.uniform(action_low[dim], action_high[dim]))

            obs_next, rew, term, trunc, info = env.step(a_np)
            done = term or trunc
            v_flag = float(info.get("violation", 0))
            if v_flag > 0.5:
                ep_viol = True
            ep_ret += float(rew)

            replay_buf.add_online(obs, a_np, float(rew), obs_next, float(done), v_flag)
            ep_len += 1
            obs = obs_next

            if replay_buf.on_size >= 64:
                batch = replay_buf.sample_with_priority(
                    batch_size=128,
                    priority_indices=pre_crash_indices,
                    num_priority=16,
                )
                bs = batch["states"].to(device)
                ba = batch["actions"].to(device)
                br = batch["rewards"].to(device)
                bns = batch["next_states"].to(device)
                bd = batch["dones"].to(device)
                bc = batch["violations"].to(device)

                c_loss = agent.update_critics(bs, ba, br, bns, bd)
                agent.update_failure_critics(bs, ba, bns, bd, bc)
                agent.update_actor(bs[:128], lambda_dual=safety_ctrl.lambda_dual)
                agent.soft_update()
                ep_c_losses.append(float(c_loss))

            if done:
                break

        if ep_viol and ep_len > 0:
            for k in range(min(5, ep_len)):
                pre_crash_indices.append((ep_start_ptr + ep_len - 1 - k) % replay_buf.max_size)

        safety_ctrl.record_episode_result(ep_viol)
        lam_new = safety_ctrl.dual_update()
        if ep == 1 or ep % 3 == 0 or ep == online_episodes:
            mean_cl = float(np.mean(ep_c_losses)) if ep_c_losses else 0.0
            print(
                f"  Episode {ep:02d}/{online_episodes} | Return: {ep_ret:+.1f} | "
                f"CriticLoss: {mean_cl:.3f} | Viol: {ep_viol} | lambda: {lam_new:.3f}"
            )

    safe_policy_path = os.path.join(ckpt_dir, "safe_rlhf_policy.pt")
    safe_q_path = os.path.join(ckpt_dir, "safe_rlhf_reward_critic.pt")
    safe_f_path = os.path.join(ckpt_dir, "safe_rlhf_failure_ensemble.pt")

    torch.save({"model_state_dict": agent.actor.state_dict(), "lambda_dual": safety_ctrl.lambda_dual}, safe_policy_path)
    torch.save({"model_state_dict": agent.primary_reward_critic.state_dict()}, safe_q_path)
    torch.save({"model_state_dict": agent.failure_ensemble.state_dict()}, safe_f_path)
    print(f"✓ Saved Safe RLHF artifacts: {safe_policy_path}, {safe_q_path}, {safe_f_path}")

    # Stage 8: Edge Deployment Optimization (ONNX, INT8 & Latency)
    print("\n[Stage 8/9] Exporting ONNX Runtime Graph, INT8 Quantization & Edge Benchmark...")
    onnx_path = export_policy(agent.actor, output_path=os.path.join(deploy_dir, "policy_edge.onnx"))
    q_policy = apply_int8_quantization(agent.actor, save_path=os.path.join(deploy_dir, "policy_quantized_int8.pt"))
    compare_quantization_performance(agent.actor, q_policy, num_samples=50)
    latency_stats = benchmark_end_to_end_latency(num_cycles=200)

    # Stage 9: Consolidated Multi-Episode Evaluation
    print("\n[Stage 9/9] Evaluating Full Safe RLHF Controller Against HCR K8 Baseline...")
    eval_stats = run_full_evaluation(
        policy_path=safe_policy_path,
        reward_critic_path=safe_q_path,
        failure_ensemble_path=safe_f_path,
        num_eval_episodes=5,
        output_dir=eval_dir,
    )

    tracker.register_run(
        run_id=f"run_pome_{int(time.time())}",
        stage="complete_pipeline",
        parameters={"state_dim": 224, "action_dim": 5, "candidates": 16, "reactor_volume_l": 2.5},
        artifacts={
            "sft_policy": sft_path,
            "reward_ensemble": reward_ckpt,
            "dppo_policy": dppo_path,
            "safe_rlhf_policy": safe_policy_path,
            "safe_rlhf_reward_critic": safe_q_path,
            "safe_rlhf_failure_ensemble": safe_f_path,
            "onnx_model": onnx_path,
            "evaluation_report": os.path.join(eval_dir, "report.md"),
        },
        metrics={**eval_stats, "mean_latency_ms": float(latency_stats["mean_latency_ms"])},
    )

    elapsed = time.time() - t0
    print("\n" + "=" * 72)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.1f} SECONDS!")
    print(f"COD Removal: {eval_stats['cod_removal_pct']:.2f}% | Cost: IDR {eval_stats['total_cost_idr']:,.0f} | VR: {eval_stats['violation_rate']*100:.1f}%")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Safe Offline-to-Online RLHF POME Pipeline")
    parser.add_argument("--skip-loco", action="store_true", help="Skip 14-fold HybridODE LOCO training")
    parser.add_argument("--force-loco", action="store_true", help="Force re-running HybridODE LOCO")
    parser.add_argument("--force-data", action="store_true", help="Force rebuilding D_hist corpus")
    args = parser.parse_args()
    run_pipeline(skip_loco=args.skip_loco, force_loco=args.force_loco, force_data=args.force_data)
