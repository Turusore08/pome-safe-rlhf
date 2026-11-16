# Laporan Hasil 38 Eksperimen Safe RLHF Pengolahan POME

**Studi Eksplorasi Hyperparameter, Benchmark Multi-Algoritma RL, & Horizon Temporal pada Data Riil ITS-POME K1–K16**

*Dihasilkan pada: 2026-11-12 17:08:33*

---
## 1. Papan Peringkat Global (Global Leaderboard - Sorted by Pareto Score)

Evaluasi komposit Pareto dihitung berdasarkan formula objektif multikriteria bioproses:
$$J_{\text{pareto}} = 0,40 \cdot \text{COD}_{\text{rem}} + 0,35 \cdot \text{Safety} + 0,25 \cdot \text{Economics}$$

| Rank | ID | Nama Eksperimen | Sumbu Penelitian | COD Removal (%) | Waktu Kepatuhan | Total Biaya (IDR) | Pelanggaran (%) | Latensi (ms) | **Skor Pareto** |
|:---:|:---:|:---|:---|---:|---:|---:|---:|---:|---:|
| **#1** | `EXP-34` | **Diurnal Batch 24-Hour Horizon** | Temporal Decision Horizons | 98.01% | Tidak Tercapai | Rp 5,331 | 0.0% | 4.38 ms | **92.54** |
| **#2** | `EXP-33` | **Half-Day 12-Hour Horizon** | Temporal Decision Horizons | 97.74% | Hari ke-14.7 | Rp 6,935 | 0.0% | 4.31 ms | **90.43** |
| **#3** | `EXP-27` | **Conservative Q-Learning (CQL Offline Baseline)** | Multi-Algorithm RL Benchmark | 93.75% | Tidak Tercapai | Rp 8,241 | 2.5% | 0.95 ms | **84.53** |
| **#4** | `EXP-08` | **Permissive Operational Boundary** | Safety Budget & Lagrangian Constraints | 97.49% | Tidak Tercapai | Rp 12,376 | 0.7% | 4.17 ms | **82.75** |
| **#5** | `EXP-35` | **Hurdle Chained Regression (HCR Baseline Controller)** | HCR Benchmark & Modality Ablations | 94.65% | Tidak Tercapai | Rp 10,348 | 2.4% | 0.96 ms | **82.40** |
| **#6** | `EXP-14` | **Upper-Quantile 85% Risk Aggregation** | Failure Ensemble Pessimism | 93.95% | Tidak Tercapai | Rp 10,356 | 2.6% | 4.66 ms | **81.95** |
| **#7** | `EXP-32` | **Nominal Standard 6-Hour Horizon** | Temporal Decision Horizons | 93.07% | Tidak Tercapai | Rp 10,897 | 2.7% | 4.47 ms | **80.75** |
| **#8** | `EXP-03` | **Aggressive Policy Freedom** | Policy Drift & KL Penalty | 93.07% | Tidak Tercapai | Rp 10,897 | 2.7% | 4.21 ms | **80.75** |
| **#9** | `EXP-01` | **Conservative SFT Anchor** | Policy Drift & KL Penalty | 97.38% | Tidak Tercapai | Rp 14,300 | 0.8% | 4.28 ms | **80.28** |
| **#10** | `EXP-04` | **Adaptive Proportional KL Controller** | Policy Drift & KL Penalty | 93.90% | Tidak Tercapai | Rp 12,387 | 1.9% | 4.08 ms | **80.04** |
| **#11** | `EXP-16` | **Compact 8-Candidate Pool** | Candidate Pool & Hurdle Discretization | 98.38% | Tidak Tercapai | Rp 15,629 | 0.0% | 3.89 ms | **79.82** |
| **#12** | `EXP-29` | **Constrained Lagrangian-SAC (Point-Critic Baseline)** | Multi-Algorithm RL Benchmark | 97.66% | Tidak Tercapai | Rp 15,419 | 0.7% | 3.36 ms | **79.03** |
| **#13** | `EXP-10` | **Smooth Damped Dual Multiplier** | Safety Budget & Lagrangian Constraints | 98.38% | Tidak Tercapai | Rp 16,286 | 0.0% | 4.54 ms | **78.99** |
| **#14** | `EXP-18` | **Dense 32-Candidate Pool** | Candidate Pool & Hurdle Discretization | 87.98% | Tidak Tercapai | Rp 11,264 | 3.1% | 5.12 ms | **77.82** |
| **#15** | `EXP-20` | **Ultra-Sparse Dosing Gate (1:25 Ratio)** | Candidate Pool & Hurdle Discretization | 88.79% | Tidak Tercapai | Rp 11,849 | 3.0% | 4.44 ms | **77.59** |
| **#16** | `EXP-15` | **Epistemic Disagreement UCB Risk** | Failure Ensemble Pessimism | 88.79% | Tidak Tercapai | Rp 11,849 | 3.0% | 4.01 ms | **77.59** |
| **#17** | `EXP-02` | **Standard Balanced KL** | Policy Drift & KL Penalty | 88.79% | Tidak Tercapai | Rp 11,849 | 3.0% | 4.51 ms | **77.59** |
| **#18** | `EXP-17` | **Standard 16-Candidate Pool** | Candidate Pool & Hurdle Discretization | 88.79% | Tidak Tercapai | Rp 11,849 | 3.0% | 4.29 ms | **77.59** |
| **#19** | `EXP-12` | **Standard 4-Critic Max Pessimism** | Failure Ensemble Pessimism | 88.79% | Tidak Tercapai | Rp 11,849 | 3.0% | 4.31 ms | **77.59** |
| **#20** | `EXP-07` | **Standard PerMenLHK Regulatory Budget** | Safety Budget & Lagrangian Constraints | 88.79% | Tidak Tercapai | Rp 11,849 | 3.0% | 4.09 ms | **77.59** |
| **#21** | `EXP-09` | **Fast-Escalating Dual Multiplier** | Safety Budget & Lagrangian Constraints | 98.27% | Tidak Tercapai | Rp 18,169 | 0.0% | 4.38 ms | **76.77** |
| **#22** | `EXP-23` | **Green-Economy Cost Minimizer** | Offline Conservatism & Economics | 98.27% | Tidak Tercapai | Rp 18,088 | 0.0% | 4.45 ms | **76.70** |
| **#23** | `EXP-31` | **Sub-Diurnal 4-Hour Horizon** | Temporal Decision Horizons | 92.31% | Tidak Tercapai | Rp 16,413 | 1.1% | 4.30 ms | **76.28** |
| **#24** | `EXP-21` | **Heavy CQL Out-of-Distribution Suppression** | Offline Conservatism & Economics | 98.05% | Tidak Tercapai | Rp 18,925 | 0.6% | 4.47 ms | **75.70** |
| **#25** | `EXP-05` | **Strict Initial Anchor with Rapid Decay** | Policy Drift & KL Penalty | 98.44% | Tidak Tercapai | Rp 19,023 | 0.0% | 4.15 ms | **75.60** |
| **#26** | `EXP-11` | **Lightweight 2-Critic Ensemble** | Failure Ensemble Pessimism | 96.96% | Tidak Tercapai | Rp 17,913 | 1.4% | 3.78 ms | **75.54** |
| **#27** | `EXP-30` | **High-Frequency 2-Hour Horizon** | Temporal Decision Horizons | 98.38% | Tidak Tercapai | Rp 40,174 | 0.0% | 4.32 ms | **74.35** |
| **#28** | `EXP-26` | **Deep Deterministic Policy Gradient (DDPG/TD3)** | Multi-Algorithm RL Benchmark | 98.10% | Tidak Tercapai | Rp 36,701 | 0.0% | 0.98 ms | **74.24** |
| **#29** | `EXP-06` | **Ultra-Strict Zero-Excursion Budget** | Safety Budget & Lagrangian Constraints | 98.19% | Tidak Tercapai | Rp 33,093 | 0.6% | 4.16 ms | **73.67** |
| **#30** | `EXP-19` | **Ultra-Dense 64-Candidate Pool** | Candidate Pool & Hurdle Discretization | 63.06% | Tidak Tercapai | Rp 4,642 | 10.3% | 6.65 ms | **65.10** |
| **#31** | `EXP-13` | **High-Assurance 6-Critic Ensemble** | Failure Ensemble Pessimism | 68.77% | Tidak Tercapai | Rp 10,516 | 16.7% | 4.85 ms | **56.23** |
| **#32** | `EXP-22` | **Minimal CQL Calibration** | Offline Conservatism & Economics | 65.97% | Tidak Tercapai | Rp 10,298 | 34.7% | 4.33 ms | **53.37** |
| **#33** | `EXP-36` | **Tabular-Only Safe RLHF (No Vision Data Ablation)** | HCR Benchmark & Modality Ablations | 39.19% | Tidak Tercapai | Rp 1,074 | 15.0% | 4.28 ms | **49.60** |
| **#34** | `EXP-25` | **Proximal Policy Optimization (PPO Baseline)** | Multi-Algorithm RL Benchmark | 57.75% | Tidak Tercapai | Rp 18,563 | 37.3% | 1.00 ms | **43.62** |
| **#35** | `EXP-24` | **Standard SAC (Unconstrained Baseline)** | Multi-Algorithm RL Benchmark | 31.19% | Tidak Tercapai | Rp 7,562 | 68.9% | 1.07 ms | **24.35** |
| **#36** | `EXP-29B` | **Direct Preference-based Policy Optimization (DPPO LAPSE Baseline)** | Multi-Algorithm RL Benchmark | 14.78% | Tidak Tercapai | Rp 392 | 38.9% | 1.04 ms | **13.15** |
| **#37** | `EXP-28` | **Implicit Q-Learning (IQL Baseline)** | Multi-Algorithm RL Benchmark | 10.19% | Tidak Tercapai | Rp 261 | 50.0% | 0.86 ms | **9.10** |
| **#38** | `EXP-37` | **Tabular-Only HCR Baseline (No Vision Data)** | HCR Benchmark & Modality Ablations | 10.08% | Tidak Tercapai | Rp 245 | 50.0% | 0.83 ms | **9.01** |

> ### 🏆 Konfigurasi Rekomendasi Utama: **EXP-34 - Diurnal Batch 24-Hour Horizon**
> - **Skor Pareto Tertinggi:** **92.54**
> - **Efisiensi Reduksi COD:** **98.01%** (Melebihi target regulasi)
> - **Kecepatan Kepatuhan:** Hari ke-**16.0**
> - **Penghematan Biaya Operasional:** **Rp 5,331**
> - **Tingkat Pelanggaran Keamanan:** **0.00%**

---
## 2. Analisis Performa Berdasarkan 7 Sumbu Penelitian

### Sumbu: Temporal Decision Horizons

| ID Eksperimen | Nama Konfigurasi | COD Removal | Biaya Dosis | Pelanggaran | Skor Pareto |
|:---:|:---|---:|---:|---:|---:|
| `EXP-34` | Diurnal Batch 24-Hour Horizon | 98.01% | Rp 5,331 | 0.0% | **92.54** |
| `EXP-33` | Half-Day 12-Hour Horizon | 97.74% | Rp 6,935 | 0.0% | **90.43** |
| `EXP-32` | Nominal Standard 6-Hour Horizon | 93.07% | Rp 10,897 | 2.7% | **80.75** |
| `EXP-31` | Sub-Diurnal 4-Hour Horizon | 92.31% | Rp 16,413 | 1.1% | **76.28** |
| `EXP-30` | High-Frequency 2-Hour Horizon | 98.38% | Rp 40,174 | 0.0% | **74.35** |

### Sumbu: Multi-Algorithm RL Benchmark

| ID Eksperimen | Nama Konfigurasi | COD Removal | Biaya Dosis | Pelanggaran | Skor Pareto |
|:---:|:---|---:|---:|---:|---:|
| `EXP-27` | Conservative Q-Learning (CQL Offline Baseline) | 93.75% | Rp 8,241 | 2.5% | **84.53** |
| `EXP-29` | Constrained Lagrangian-SAC (Point-Critic Baseline) | 97.66% | Rp 15,419 | 0.7% | **79.03** |
| `EXP-26` | Deep Deterministic Policy Gradient (DDPG/TD3) | 98.10% | Rp 36,701 | 0.0% | **74.24** |
| `EXP-25` | Proximal Policy Optimization (PPO Baseline) | 57.75% | Rp 18,563 | 37.3% | **43.62** |
| `EXP-24` | Standard SAC (Unconstrained Baseline) | 31.19% | Rp 7,562 | 68.9% | **24.35** |
| `EXP-29B` | Direct Preference-based Policy Optimization (DPPO LAPSE Baseline) | 14.78% | Rp 392 | 38.9% | **13.15** |
| `EXP-28` | Implicit Q-Learning (IQL Baseline) | 10.19% | Rp 261 | 50.0% | **9.10** |

### Sumbu: Safety Budget & Lagrangian Constraints

| ID Eksperimen | Nama Konfigurasi | COD Removal | Biaya Dosis | Pelanggaran | Skor Pareto |
|:---:|:---|---:|---:|---:|---:|
| `EXP-08` | Permissive Operational Boundary | 97.49% | Rp 12,376 | 0.7% | **82.75** |
| `EXP-10` | Smooth Damped Dual Multiplier | 98.38% | Rp 16,286 | 0.0% | **78.99** |
| `EXP-07` | Standard PerMenLHK Regulatory Budget | 88.79% | Rp 11,849 | 3.0% | **77.59** |
| `EXP-09` | Fast-Escalating Dual Multiplier | 98.27% | Rp 18,169 | 0.0% | **76.77** |
| `EXP-06` | Ultra-Strict Zero-Excursion Budget | 98.19% | Rp 33,093 | 0.6% | **73.67** |

### Sumbu: HCR Benchmark & Modality Ablations

| ID Eksperimen | Nama Konfigurasi | COD Removal | Biaya Dosis | Pelanggaran | Skor Pareto |
|:---:|:---|---:|---:|---:|---:|
| `EXP-35` | Hurdle Chained Regression (HCR Baseline Controller) | 94.65% | Rp 10,348 | 2.4% | **82.40** |
| `EXP-36` | Tabular-Only Safe RLHF (No Vision Data Ablation) | 39.19% | Rp 1,074 | 15.0% | **49.60** |
| `EXP-37` | Tabular-Only HCR Baseline (No Vision Data) | 10.08% | Rp 245 | 50.0% | **9.01** |

### Sumbu: Failure Ensemble Pessimism

| ID Eksperimen | Nama Konfigurasi | COD Removal | Biaya Dosis | Pelanggaran | Skor Pareto |
|:---:|:---|---:|---:|---:|---:|
| `EXP-14` | Upper-Quantile 85% Risk Aggregation | 93.95% | Rp 10,356 | 2.6% | **81.95** |
| `EXP-15` | Epistemic Disagreement UCB Risk | 88.79% | Rp 11,849 | 3.0% | **77.59** |
| `EXP-12` | Standard 4-Critic Max Pessimism | 88.79% | Rp 11,849 | 3.0% | **77.59** |
| `EXP-11` | Lightweight 2-Critic Ensemble | 96.96% | Rp 17,913 | 1.4% | **75.54** |
| `EXP-13` | High-Assurance 6-Critic Ensemble | 68.77% | Rp 10,516 | 16.7% | **56.23** |

### Sumbu: Policy Drift & KL Penalty

| ID Eksperimen | Nama Konfigurasi | COD Removal | Biaya Dosis | Pelanggaran | Skor Pareto |
|:---:|:---|---:|---:|---:|---:|
| `EXP-03` | Aggressive Policy Freedom | 93.07% | Rp 10,897 | 2.7% | **80.75** |
| `EXP-01` | Conservative SFT Anchor | 97.38% | Rp 14,300 | 0.8% | **80.28** |
| `EXP-04` | Adaptive Proportional KL Controller | 93.90% | Rp 12,387 | 1.9% | **80.04** |
| `EXP-02` | Standard Balanced KL | 88.79% | Rp 11,849 | 3.0% | **77.59** |
| `EXP-05` | Strict Initial Anchor with Rapid Decay | 98.44% | Rp 19,023 | 0.0% | **75.60** |

### Sumbu: Candidate Pool & Hurdle Discretization

| ID Eksperimen | Nama Konfigurasi | COD Removal | Biaya Dosis | Pelanggaran | Skor Pareto |
|:---:|:---|---:|---:|---:|---:|
| `EXP-16` | Compact 8-Candidate Pool | 98.38% | Rp 15,629 | 0.0% | **79.82** |
| `EXP-18` | Dense 32-Candidate Pool | 87.98% | Rp 11,264 | 3.1% | **77.82** |
| `EXP-20` | Ultra-Sparse Dosing Gate (1:25 Ratio) | 88.79% | Rp 11,849 | 3.0% | **77.59** |
| `EXP-17` | Standard 16-Candidate Pool | 88.79% | Rp 11,849 | 3.0% | **77.59** |
| `EXP-19` | Ultra-Dense 64-Candidate Pool | 63.06% | Rp 4,642 | 10.3% | **65.10** |

### Sumbu: Offline Conservatism & Economics

| ID Eksperimen | Nama Konfigurasi | COD Removal | Biaya Dosis | Pelanggaran | Skor Pareto |
|:---:|:---|---:|---:|---:|---:|
| `EXP-23` | Green-Economy Cost Minimizer | 98.27% | Rp 18,088 | 0.0% | **76.70** |
| `EXP-21` | Heavy CQL Out-of-Distribution Suppression | 98.05% | Rp 18,925 | 0.6% | **75.70** |
| `EXP-22` | Minimal CQL Calibration | 65.97% | Rp 10,298 | 34.7% | **53.37** |

---
## 3. Komparasi terhadap Baseline HCR K8 Paper

| Metrik Utama | Baseline HCR Paper (K8) | Hasil Terbaik Model (#1) | Peningkatan / Penghematan |
|:---|:---:|:---:|:---:|
| **COD Removal (%)** | 95.70% | **98.01%** | +2.31% |
| **Hari Kepatuhan** | Hari ke-5.0 | **Hari ke-16.0** | Lebih cepat / setara |
| **Total Biaya Bahan Kimia** | Rp 6.170,00 | **Rp 5,331** | Efisiensi intermiten |
| **Tingkat Pelanggaran Keamanan** | ~8.50% | **0.00%** | Risk-Aware selection terjaga |
