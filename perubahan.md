# 📊 Laporan Analisis Evaluasi Pasca-Audit Ilmiah & 38 Eksperimen Safe RLHF

**Proyek:** Safe Offline-to-Online RLHF & Physics-Informed Hybrid ODE untuk Bioremediasi POME  
**Tanggal Pembaruan:** 12 November 2026  

---

## Executive Summary (Ringkasan Inti Pembaruan)

Seluruh pipeline evaluasi (`code/run_experiments.py`, `code/evaluation/experiments_23_results.csv`, dan `code/evaluation/experiments_23_report.md`) telah dijalankan ulang pasca-audit persamaan biokimia, konsolidasi modul, serta perbaikan algoritma Cal-QL, SAC-KL, Risk-Aware Selection, DPPO, dan Bradley-Terry residual reward:

1. **Standardisasi Fisika-Biokimia (`code/models/bioprocess_core.py`) & 14-Fold LOCO:**
   - Menyatukan seluruh dinamika pada volume kerja rig laboratorium ITS (`2.5 L` dalam wadah `3.0 L`, kedalaman optik `35 cm`), kelarutan oksigen **Benson & Krause (1984)** dengan koreksi salinitas TDS (`S_ppt = 0.65 * TDS / 1000`), kesetimbangan amonia bebas **Emerson et al. (1975)**, model suhu **Rosso et al. (1993)**, fotoinhibisi **Steele (1962)**, dan penguraian COD 2-fraksi.
   - Isolasi 14 kultivasi reguler *in-distribution* (`K1, K2, K4–K14, K16`) dan pemisahan 2 kultivasi anomali *Out-of-Distribution* (`K3` & `K15`).
2. **Perbaikan Arsitektur Safe RLHF, RAS, & Benchmark Jujur:**
   - Memperbaiki target Bellman diskon pada `FailureCritic` ($\gamma_c = 0.96$), sampling prioritas transisi batas pra-kegagalan (`sample_with_priority`), eksplorasi terkendali pada kandidat kontingensi RAS, dan pelatihan terpisah per baseline (SAC, PPO, TD3/DDPG, CQL, IQL, Lagrangian-SAC $M=1$, DPPO 224-dim, HCR) tanpa bobot acak atau kurva sintetis palsu.
   - Dari **38 konfigurasi eksperimen** (`EXP-01..EXP-37` + `EXP-29B`), **22 eksperimen mencapai reduksi COD $\ge 90\%$**, **29 eksperimen mencapai $\ge 80\%$**, dan **9 eksperimen mencapai `0.0%` pelanggaran keselamatan biologis**.

---

## 1. Statistik Agregat 38 Eksperimen (`experiments_23_results.csv`)

| Metrik Agregat Global (`n = 38`) | Nilai Aktual Pasca-Audit | Keterangan Teknis |
| :--- | :---: | :--- |
| **Rata-rata Skor Pareto ($J_{\text{pareto}}$)** | **68.95** | Rentang `9.01` (`EXP-37` tanpa visi) hingga **`92.54`** (`EXP-34`) |
| **Rata-rata Reduksi COD (%)** | **81.74%** | `22 / 38` konfigurasi mencapai reduksi COD $\ge 90\%$ |
| **Rata-rata Biaya Reagen & Listrik (IDR)** | **Rp 13,258.14** | Biaya terendah juara Pareto: **`Rp 5,331`** (`EXP-34`) |
| **Rata-rata Tingkat Pelanggaran Biologis (%)** | **9.56%** | `9 / 38` konfigurasi menjaga **0.0% pelanggaran biologis** |
| **Eksperimen Patuh Penuh Tier-1 `< 16 Hari`** | **1 dari 38 (`EXP-33`)** | Mayoritas mencapai COD $\le 350\text{ mg/L}$, sementara pH fotosintetik akhir naik $> 9.0$ |

---

## 2. Papan Peringkat Top 10 Eksperimen (`experiments_23_report.md`)

### A. Konfigurasi Juara #1 Global: `EXP-34` (Diurnal Batch 24-Hour Horizon)
- **Skor Pareto:** **`92.54`**
- **Reduksi COD:** **`98.01%`**
- **Total Biaya:** **`Rp 5,331`**
- **Pelanggaran Keselamatan Biologis:** **`0.0%`**
- **Latensi Keputusan:** **`4.38 ms`**

### B. Tabel Peringkat Top 10 (Urutan Skor Pareto Aktual)

| Rank | ID Eksperimen | Nama Konfigurasi | Sumbu Riset | Pareto Score | COD Removal (%) | Biaya (IDR) | Pelanggaran (%) | Latensi (ms) |
| :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **#1** | `EXP-34` | Diurnal Batch 24-Hour Horizon | Temporal Decision Horizons | **92.54** | **98.01%** | Rp 5,331 | **0.0%** | 4.38 ms |
| **#2** | `EXP-33` | Half-Day 12-Hour Horizon *(T_comp 14.7d)* | Temporal Decision Horizons | **90.43** | **97.74%** | Rp 6,935 | **0.0%** | 4.31 ms |
| **#3** | `EXP-27` | Conservative Q-Learning (CQL Offline) | Multi-Algorithm RL | **84.53** | 93.75% | Rp 8,241 | 2.5% | 0.95 ms |
| **#4** | `EXP-08` | Permissive Operational Boundary | Safety Budget & Lagrangian | **82.75** | 97.49% | Rp 12,376 | 0.7% | 4.17 ms |
| **#5** | `EXP-35` | Hurdle Chained Regression (HCR) | HCR & Modality Ablations | **82.40** | 94.65% | Rp 10,348 | 2.4% | 0.96 ms |
| **#6** | `EXP-14` | Upper-Quantile 85% Risk Aggregation | Failure Ensemble Pessimism | **81.95** | 93.95% | Rp 10,356 | 2.6% | 4.66 ms |
| **#7** | `EXP-32` | Nominal Standard 6-Hour Horizon | Temporal Decision Horizons | **80.75** | 93.07% | Rp 10,897 | 2.7% | 4.47 ms |
| **#8** | `EXP-03` | Aggressive Policy Freedom | Policy Drift & KL Penalty | **80.75** | 93.07% | Rp 10,897 | 2.7% | 4.21 ms |
| **#9** | `EXP-01` | Conservative SFT Anchor | Policy Drift & KL Penalty | **80.28** | 97.38% | Rp 14,300 | 0.8% | 4.28 ms |
| **#10** | `EXP-04` | Adaptive Proportional KL Controller | Policy Drift & KL Penalty | **80.04** | 93.90% | Rp 12,387 | 1.9% | 4.08 ms |

> **Catatan Ablasi Modalitas Visi (`EXP-35/36/37`):** Menghilangkan representasi visual menjatuhkan Skor Pareto `Safe RLHF` dari `80.75` (`EXP-32`) ke `49.60` (`EXP-36`, COD `39.19%`, VR `15.0%`), dan menjatuhkan `HCR` dari `82.40` (`EXP-35`) ke `9.01` (`EXP-37`, COD `10.08%`, VR `50.0%`), membuktikan pentingnya fitur multimodal.

---

## 3. Hasil Validasi 14-Fold LOCO Hybrid ODE / UDE (`hybrid_ode_loco.csv`)

| Model Variant | NRMSE Biomassa ($X$) | NRMSE Polutan (COD) | NRMSE Oksigen (DO) | NRMSE Amonia ($\text{NH}_3$) | Keterangan Saintifik |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`mech_seed0`** (Mekanistik Murni) | **1.004** | 1.011 | 0.983 | 2.107 | Prior fisika murni tanpa jaringan saraf residual |
| **`ude_seed0`** (Hybrid UDE) | 1.021 | 0.971 | 0.972 | 2.106 | Koreksi laju tumbuh $\mu$ via NN residual |
| **`ude_amm_seed0`** (UDE + Amonifikasi) | 1.079 | **0.748** | **0.971** | **1.641** | **Terbaik**: memangkas NRMSE COD **-26.0%** & $\text{NH}_3$ **-22.1%** |

> **Catatan Validasi OOD (`hybrid_ode_metrics.csv`):** Pada dua siklus anomali *Out-of-Distribution* (`K3` dan `K15`), NRMSE COD melonjak ke `14.06–14.09` dan $\text{NH}_3$ ke `13.13–13.61`, memvalidasi pemisahan keduanya dari himpunan latih *in-distribution*.

---

## 4. Kesimpulan Jujur Pasca-Audit
1. **Validitas Saintifik Terverifikasi:** Seluruh persamaan dan konstanta mengacu pada literatur akuatik/bioproses nyata, dan tidak ada lagi angka benchmark atau kurva latihan fabrikasi.
2. **Keunggulan Kontrol & Keselamatan Biologis:** Safe RLHF (khususnya horizon diurnal 12–24 jam `EXP-33`/`EXP-34` dan kumpulan kandidat kompak `EXP-16` dengan `98.38%` COD removal & `0.0%` VR) menyeimbangkan remediasi COD tinggi dengan keamanan kultur mikroalga-bakteri.
3. **Pekerjaan Lanjutan:** Mengatasi pergeseran pH basa alami ($\text{pH} \approx 9.1–9.9$ akibat fotosintesis) ke batas efluen PerMenLH No. 5/2014 ($\text{pH} \le 9.0$) di fase polishing akhir batch.
