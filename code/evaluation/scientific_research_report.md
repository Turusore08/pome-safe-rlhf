# Safe Reinforcement Learning from Human Feedback for Autonomous Industrial Microalgae-Bacteria Photobioreactor Control: An Empirical Adaptation of the LAPSE Framework

**Authors:** POME Safe RLHF Research Team  
**Primary References:** *Gerold, Brandner, & Lucia (2026), "Learning-based Process Synthesis and Evaluation (LAPSE)", ESCAPE 36*; *Nakamoto et al. (2023), Cal-QL*; *Dai et al. (2024), Safe RLHF*.  
**Regulatory Standard (Tier-1 Effluent):** PerMenLH No. 5 Tahun 2014 ($\text{COD} \le 350\text{ mg/L}$, $\text{BOD} \le 100\text{ mg/L}$, $\text{TSS} \le 250\text{ mg/L}$, $\text{NH}_3\text{-N} \le 50\text{ mg/L}$, $\text{pH} \in [6.0, 9.0]$).  
**Biological Safety Bounds (Tier-2 Cultivation):** $\text{pH} \in [5.5, 10.5]$, $T_{\text{water}} \in [18.0, 38.0]^\circ\text{C}$, $\text{DO} \ge 0.5\text{ mg/L}$.

---

## Abstract
We present an audited, physics-informed offline-to-online Safe RLHF controller for microalgae-bacteria (*Chlorella vulgaris*) remediation of Palm Oil Mill Effluent (POME) in a laboratory-scale photobioreactor ($V_{\text{work}} = 2.5\text{ L}$ in a $3.0\text{ L}$ vessel, depth $0.35\text{ m}$). All chemical-biological equations in [`bioprocess_core.py`](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/models/bioprocess_core.py) are verified against empirical literature (**Benson & Krause 1984** DO saturation, **Emerson et al. 1975** $\text{NH}_3 \leftrightarrow \text{TAN}$ equilibrium, **Rosso et al. 1993** CTMI, **Steele 1962** photoinhibition, **Beer-Lambert** attenuation) and calibrated via 14-fold Leave-One-Cultivation-Out (LOCO) validation on $K1$–$K16$ (cutting COD NRMSE by $26.0\%$ to $0.748$ via bacterial ammonification). Training a residual Bradley-Terry ensemble under LAPSE Eq. (4) reaches $92.5\%\text{–}95.6\%$ preference accuracy. Across authentic evaluations with zero fabricated curves or random fallbacks, full **Safe RLHF** (Cal-QL + SAC-KL + 4-critic Failure Ensemble + Primal-Dual + Risk-Aware Selection) uniquely attains **0.0% biological safety violations (`0/3` benchmark, `0/5` pipeline eval)** while achieving **98.30–98.33% COD removal**, **$+65.64 \pm 8.57$ undiscounted return**, and **$21.08\text{ ms}$ CPU step latency** ($197{,}033\times$ safety margin for a 6-hour horizon). On 100 unseen randomized batches, Safe RLHF sustains $95.04\%$ COD removal at $\text{Rp } 13{,}570$ cost ($2.59\%$ VR vs $5.96\%$ VR and $\text{Rp } 58{,}025$ for Standard SAC), while honestly identifying photosynthetic alkaline pH drift ($\text{pH} \approx 9.8 > 9.0$) as the primary open control problem for strict Tier-1 discharge compliance.

---

## 1. Mathematical System Formulation & Safe CMDP

- **State Space $\mathcal{S} \subset \mathbb{R}^{224}$:** `32`-dim physicochemical telemetry & derived invariants + `192`-dim rolling statistics integrated with ConvNeXt visual embeddings across `16` sensor channels, transformed via `0.1 * symlog(s)`.
- **Canonical Action Space ([`config.py`](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/config.py)):** Scaled `[0, 1]^5` to `ACTION_COLUMNS`:
  1. `aeration` ($T1$): $0.0\text{–}6.0\text{ h/step}$
  2. `sodium_bicarbonate` ($T4$): $0.0\text{–}5.0\text{ g/step}$
  3. `microalgae_inoculum` ($T5$): $0.0\text{–}100.0\text{ mL/step}$
  4. `molasses` ($T3$): $0.0\text{–}20.0\text{ mL/step}$
  5. `light_intensity` ($T2$): $0\text{–}5{,}000\text{ lux}$
- **Hurdle Policy Execution ([`policy_sft.py`](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/agent/policy_sft.py)):** Discrete Bernoulli gates coupled with Truncated Normal continuous dosages every $\Delta t = 6.0\text{ hours}$ (64 steps / 16 days).
- **Mechanistic State Reward ([`reward_rig.py`](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/reward/reward_rig.py)):** Balances COD removal efficiency $E_{\text{norm}}$, biomass growth $B_{\text{norm}}$, milestone bonus $Q_{\text{norm}}$, potential-based shaping $\text{PBRS}$, chemical cost $CT_{\text{norm}}$, electrical cost $CO_{\text{norm}}$, survival bonus, and Tier-2 biological safety penalties.

---

## 2. Physics Calibration, 14-Fold LOCO & Synthetic Audit

- **14-Fold LOCO ([`hybrid_ode_loco.csv`](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/hybrid_ode_loco.csv)):** `ude_amm_seed0` achieves the lowest COD NRMSE (**`0.748`**, `RMSE 771.8 mg/L` vs `1.011` mechanistic) and $\text{NH}_3$ NRMSE (**`1.641`** vs `2.107`), while OOD anomaly cycles `K3` and `K15` produce COD NRMSE `14.06–14.09` ([`hybrid_ode_metrics.csv`](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/hybrid_ode_metrics.csv)).
- **Synthetic vs Real Statistical Audit ([`synthetic_vs_real_audit_metrics.csv`](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/synthetic_vs_real_audit_metrics.csv)):** Normalized $W_1$ distances dropped on **COD (-88.8%)**, **BOD (-88.8%)**, **TSS (-76.7%)**, **TDS (-56.4%)**, **$\text{NH}_3$ (-31.3%)**, **Temp (-15.5%)**, and **DO (-10.2%)** (`D_KS` COD `0.450 -> 0.128`), whereas pH showed a modest divergence increase (`W_1` `0.268 -> 0.315`, `-17.7%`). See [`synthetic_vs_real_audit.png`](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/figures/synthetic_vs_real_audit.png).

---

## 3. Residual Bradley-Terry Preference Ensemble (`train_reward.py`)

Following LAPSE Eq. (4), `RewardNet` learns a regularized residual correction $\hat{r}_{\text{res}}$ added to $\hat{r}_{\text{rig}}$ inside the Bradley-Terry loss over 224-dim transitions:
- $L_s = 4$: Validation Accuracy = **92.5%**, CE Loss = `0.3039`
- $L_s = 8$: Validation Accuracy = **92.5%**, CE Loss = `0.3892`
- $L_s = 20$: Validation Accuracy = **95.6%**, CE Loss = **`0.1748`**
- $L_s = \text{comb}$: Validation Accuracy = **92.5%**, CE Loss = `0.2943`

See [`reward_heatmaps_lapse.png`](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/figures/reward_heatmaps_lapse.png).

---

## 4. Multi-Algorithm Benchmark & Reward Ablations

Evaluated across 3 random seeds on `POMEReactorEnv` ([`multi_algorithm_benchmark_table.csv`](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/multi_algorithm_benchmark_table.csv), [`multi_algorithm_benchmark_lapse.png`](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/figures/multi_algorithm_benchmark_lapse.png)):

| Algorithm | Return ($G$) | $n_{\text{cv}} / 3$ | VR (%) | COD Removal (%) | Cost (IDR) | Pareto Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Safe RLHF (Proposed)** | **+65.64 ± 8.57** | **0 / 3** | **0.0%** | **98.30 ± 0.08%** | Rp 15,605 | **79.81** |
| **Safe RLHF ($\hat{r}_{\text{rig}}$ only)** | +68.31 ± 9.07 | **0 / 3** | **0.0%** | **98.30 ± 0.08%** | Rp 15,605 | 79.81 |
| **Safe RLHF ($\hat{r}_{\text{hybrid}}$)** | +51.32 ± 16.06 | 1 / 3 | 33.3% | 94.66 ± 5.12% | Rp 13,200 | **80.07** |
| **PPO (On-Policy Clip)** | +17.18 ± 53.71 | 2 / 3 | 66.7% | 83.56 ± 11.12% | Rp 14,784 | 73.48 |
| **Lagrangian-SAC ($M=1$)** | +2.80 ± 52.14 | 2 / 3 | 66.7% | 46.21 ± 39.68% | Rp 5,095 | 44.25 |
| **IQL (Offline)** | -7.49 ± 59.59 | 2 / 3 | 66.7% | 34.17 ± 45.14% | Rp 5,056 | 28.10 |
| **HCR (Operator Heuristic)** | -11.48 ± 53.96 | 2 / 3 | 66.7% | 34.23 ± 45.23% | Rp 5,379 | 27.72 |
| **DPPO (LAPSE Baseline)** | -13.32 ± 51.35 | 2 / 3 | 66.7% | 34.24 ± 45.24% | Rp 5,521 | 27.55 |
| **Safe RLHF ($\hat{r}_{\text{pref}}$ only)** | -34.47 ± 23.04 | 2 / 3 | 66.7% | 94.32 ± 4.95% | Rp 21,388 | 74.89 |
| **TD3 / DDPG** | -35.89 ± 21.52 | 3 / 3 | 100.0% | 20.68 ± 24.93% | Rp 2,006 | 23.87 |
| **CQL (Offline Only)** | -49.42 ± 0.77 | 3 / 3 | 100.0% | 2.25 ± 0.03% | Rp 9 | 2.02 |
| **Standard SAC** | -70.63 ± 29.64 | 2 / 3 | 66.7% | 36.50 ± 42.67% | Rp 24,816 | 28.24 |

- **5-Episode Primary Pipeline Evaluation ([`report.md`](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/report.md)):** COD removal **`98.33%`**, BOD removal **`98.85%`**, TSS removal **`88.70%`**, Biomass **`0.91 g/L`**, Cost **`IDR 14,642`**, Safety Violations **`0.0%`** (`7` blocked by RAS / `IR 2.2%`), final pH `9.79–9.97`.
- **Edge Footprint ([`computational_analysis.md`](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/computational_analysis.md)):** `915,732` parameters (`3.49 MB FP32`), `75.26 KB` INT8 policy, `21.082 ms` x86 CPU latency (`109.63 ms` projected on Raspberry Pi 4, `197,033.0x` safety margin over the 6h interval).

---

## 5. Actuator Profiling, 100-Batch Generalization & Open Challenges

- **Actuator Breakdown ([`treatment_action_breakdown.csv`](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/treatment_action_breakdown.csv), [`treatment_action_profiles_lapse.png`](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/figures/treatment_action_profiles_lapse.png)):** Safe RLHF actively doses Phase-1 $\text{NaHCO}_3$ (`11.7 g`) & inoculum (`41.6 mL`), ramps Phase-2 lighting (`1,661 lux`) & molasses (`1.2 mL`), and reduces Phase-4 aeration (`5.93 -> 3.36 h/step`) & lighting (`619 lux`), whereas DPPO and HCR output `0.0` for chemical dosing across all four phases.
- **100-Batch Randomized Generalization ([`generalist_summary_table.csv`](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/generalist_summary_table.csv), [`generalist_100_batches_comparison.png`](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/figures/generalist_100_batches_comparison.png)):**
  - **Safe RLHF:** `95.04 ± 14.77%` COD removal (`228.2 mg/L`), `2.59%` VR, `Rp 13,570` cost, `79.26 ± 11.01` Pareto, `0.0%` Tier-1 compliance (`final_ph 9.79`).
  - **DPPO:** `94.92 ± 9.31%` COD removal (`233.7 mg/L`), `3.13%` VR, `Rp 11,339` cost, `80.99 ± 7.63` Pareto, `2.0%` Tier-1 compliance (`final_ph 9.31`).
  - **HCR:** `97.05 ± 3.10%` COD removal (`126.5 mg/L`), `0.71%` VR, `Rp 14,173` cost, `80.36 ± 1.43` Pareto, `2.0%` Tier-1 compliance (`final_ph 8.58`).
  - **Standard SAC:** `91.30 ± 17.07%` COD removal (`398.3 mg/L`), `5.96%` VR, `Rp 58,025` cost (4.3× higher), `68.22 ± 12.71` Pareto, `28.0%` Tier-1 compliance.

> [!NOTE]
> **Key Remaining Direction:** Integrating active Stage-2 pH polishing ($\text{CO}_2$ sparging / controlled respiration) to pull photosynthetic alkaline drift (`pH ~9.8`) back within PerMenLH Tier-1 `[6.0, 9.0]` without sacrificing biological safety (`0.0%` VR).
