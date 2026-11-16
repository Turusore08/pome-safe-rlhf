# Safe Offline-to-Online RLHF Bioprocess Control System

Implementation of autonomous data-driven control for microalgae-based Palm Oil Mill Effluent (POME) bioremediation in a 2.5 L working liquid volume photobioreactor (inside a 3.0 L glass vessel, *Chlorella vulgaris*).

Based on the formal CMDP & bioprocess specification in [`code/docs/research_formulation.md`](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/docs/research_formulation.md) and benchmarked against Hurdle Chained Regression (HCR run K8, *Mirda et al., 2026*).

---

## 1. Directory Structure

```
code/
├── config.py                       # Sensor schema (16 vars), action bounds (5 levers), costs, 2-tier limits
├── run_pipeline.py                 # Master script executing training, Safe-RLHF, DPPO, & ONNX export
├── run_experiments.py              # Full multi-axis experimental sweep & evaluation runner
├── requirements.txt                # Python dependencies
├── data/
│   ├── real_data_loader.py         # Real cultivation data ingestion (K1–K16, 5,696 hourly rows + photos)
│   ├── synthetic_generator.py      # Calibrated bioprocess transitions & synthetic trajectory generator
│   ├── cleaning.py                 # IQR outlier filter, linear interpolation, MICE imputation
│   ├── windowing.py                # Multi-resolution rolling windowing & 12 statistical descriptors (192 dims)
│   └── offline_dataset.py          # Assembly of historical offline corpus D_hist (224-dim state vectors)
├── models/
│   ├── bioprocess_core.py          # Unified physics/chemistry library (Benson & Krause 1984 DO, Emerson NH3, Plummer & Busenberg CO3, CTMI, Beer-Lambert)
│   ├── sensor_encoder.py           # 1D-Convolutional sensor encoder for h_x
│   ├── vision_encoder.py           # ConvNeXt-based vision encoder for h_z
│   ├── fusion_engine.py            # Zero-Order Hold (ZOH) & Multimodal State Fusion s_t = [h_x, h_z]
│   ├── policy_sft.py               # Hurdle (zero-inflated) policy network for reference policy pi_sft
│   ├── reward_net.py               # Bradley-Terry Residual Reward Model ensemble r_res
│   ├── critics.py                  # Reward critic Q_r & Failure Critic ensemble with LayerNorm
│   ├── pets_world_model.py         # 5-MLP Probabilistic Ensemble world model (PETS)
│   └── hybrid_ode.py               # Physics-Informed Hybrid ODE / UDE (mech, ude, ude+amm) + Benson & Krause DO & Emerson NH3
├── environment/
│   ├── pome_reactor_env.py         # Gymnasium bioreactor environment with biological absorbing failure boundaries
│   ├── bioprocess_dynamics.py      # 2.5 L microalgae-bacteria kinetics & carbonate charge-balance pH solver
│   └── reward_rig.py               # Explicit rig reward r_rig, PerMenLH compliance gap e(s), potential shaping
├── preference/
│   ├── segment_sampler.py          # Trajectory segment pair sampling (4, 8, 20 steps)
│   ├── labeling_rubric.py          # Tiered preference labeling rubric & dataset generator
│   └── train_reward.py             # Residual Bradley-Terry (r_rig + alpha*r_res) loss training loop
├── safety/
│   ├── risk_aware_selection.py     # Execution-time Risk-Aware Selection (RAS) with candidate pool
│   └── primal_dual.py              # Primal-dual multiplier update for lambda at episode boundaries
├── rl/
│   ├── replay_buffer.py            # Mixed offline-online replay buffer with symmetric sampling
│   ├── cal_ql.py                   # Calibrated offline Q-learning (Cal-QL logsumexp + MC lower bound)
│   ├── sac_kl.py                   # Offline-to-online SAC with KL divergence penalty & failure critic ensemble
│   ├── dppo.py                     # Direct Preference Policy Optimization (DPPO) with Advantage Reward Models
│   └── train_dppo.py               # Standalone 224-dim offline-compatible DPPO trainer
├── training/
│   ├── train_sft.py                # Pre-training execution entrypoint for reference policy pi_sft
│   └── train_hybrid_ode_loco.py    # 14-fold Leave-One-Cultivation-Out (LOCO) validation for Hybrid ODE / UDE
├── deployment/
│   ├── export_onnx.py              # Portable ONNX graph export
│   ├── quantize.py                 # INT8 dynamic quantization
│   ├── computational_analysis.py   # Latency, throughput, model size & FP32/INT8/ONNX parity benchmark
│   └── serve_edge.py               # Local edge inference service (FastAPI + ONNXRuntime)
├── evaluation/
│   ├── metrics.py                  # Biochemical efficacy, HCR baseline comparisons, & safety violation rates
│   ├── evaluate_generalist_100.py  # 100-batch randomized generalist robustness evaluation
│   └── generate_reward_ledger.py   # Step-by-step reward/punishment ledger & T1-T5 breakdown from trained .pt
├── experiments/
│   ├── experiment_configs.py       # Experimental matrix across reward, safety, horizon & algorithm axes
│   ├── evaluator.py                # Authentic standalone algorithm trainers/evaluators & Pareto scoring
│   └── multi_algorithm_benchmark.py# Head-to-head RL/Offline-RL benchmark & actual training curve logger
└── lineage/
    └── manifest.py                 # MLOps artifact tracking, checkpoint lineage, and run manifest
```

---

## 2. Quickstart

### Prerequisites
- Python 3.10+
- PyTorch 2.0+
- Gymnasium, ONNX, ONNXRuntime, FastAPI, Scikit-learn, Pandas, SciPy

Install dependencies:
```bash
pip install -r requirements.txt
```

### Run End-to-End Pipeline
To generate calibrated data and execute the entire RLHF training and ONNX export pipeline:
```bash
python run_pipeline.py
```

### Run Individual Components
- **Ingest & Preprocess Real Dataset (K1–K16):**
  ```bash
  python data/offline_dataset.py
  ```
- **Pre-train Reference Policy ($\pi_{\text{SFT}}$):**
  ```bash
  python training/train_sft.py
  ```
- **Benchmark Edge Latency & INT8/ONNX Parity:**
  ```bash
  python deployment/computational_analysis.py
  ```
- **Run Multi-Algorithm Benchmark & Experimental Matrix:**
  ```bash
  python experiments/multi_algorithm_benchmark.py
  python run_experiments.py
  ```

---

## 3. Methodological Highlights

1. **Multimodal State Fusion ($s_t = [h_x, h_z]$):**
   1D-Convolution encodes 16 IoT sensor telemetry streams into $h_x$. ConvNeXt encodes broth surface photography into $h_z$. A Zero-Order Hold (ZOH) buffers visual features across faster sensor control ticks.

2. **Hurdle Policy ($\pi_{\text{SFT}}$):**
   Handles the severe 1:15 zero-inflation of dosing interventions by decoupling the discrete gating decision (whether to dose) from the continuous magnitude calculation.

3. **Execution-Time Risk-Aware Selection (RAS):**
   Guarantees bioprocess safety without an analytical physical model. Proposes $N$ candidate actions, computes conservative risk across a failure critic ensemble, and executes the action maximizing Lagrangian utility:
   $$a^{\text{exec}} = \arg\max_{a \in \mathcal{C}(s)} \left[ Q_r(s,a) - \lambda \hat{C}(s,a) \right]$$

4. **Offline-to-Online SAC with KL Leash:**
   Maintains a tight Kullback-Leibler leash to $\pi_{\text{SFT}}$ to prevent reward hacking and uncalibrated exploration.
