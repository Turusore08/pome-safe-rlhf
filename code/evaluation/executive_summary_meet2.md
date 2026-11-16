# Ringkasan Eksekutif Penelitian Biokontrol POME Berbasis Safe RLHF & Metodologi LAPSE (Meet 2)

**Tanggal Pembaruan:** 12 November 2026  
**Topik:** Audit Ilmiah & Transformasi Kendali Fotobioreaktor Mikroalga-Bakteri POME Menjadi Riset Tervalidasi Literatur (*LAPSE 2026 / Cal-QL / Safe RLHF*)  
**Dokumen & Artefak Pendukung Aktual:**
- Formulasi Matematis CMDP: [`research_formulation.md`](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/docs/research_formulation.md)
- Evaluasi Pipeline Utama: [`report.md`](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/report.md)
- Audit Data Sintetis Fisika: [`synthetic_vs_real_audit_metrics.csv`](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/synthetic_vs_real_audit_metrics.csv) & [`synthetic_vs_real_audit.png`](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/figures/synthetic_vs_real_audit.png)
- Heatmap Tri-Reward LAPSE: [`reward_heatmaps_lapse.png`](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/figures/reward_heatmaps_lapse.png)
- Benchmark Multi-Algoritma: [`multi_algorithm_benchmark_table.csv`](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/multi_algorithm_benchmark_table.csv) & [`multi_algorithm_benchmark_lapse.png`](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/figures/multi_algorithm_benchmark_lapse.png)
- Profil Aksi Aktuator ($T1$–$T5$) & Ledger 64 Step: [`treatment_action_breakdown.csv`](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/treatment_action_breakdown.csv) & [`reward_punishment_ledger_sample.csv`](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/reward_punishment_ledger_sample.csv)
- Uji Generalis 100 Batch: [`generalist_summary_table.csv`](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/generalist_summary_table.csv) & [`generalist_100_batches_comparison.png`](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/figures/generalist_100_batches_comparison.png)

---

## 1. Menjawab Pertanyaan Kunci Mentor dari Meet 1

### Pertanyaan 1: "Action apa saja yang dilakukan agent ($T1$–$T5$) dan bagaimana profil dosisnya?"
Agent beroperasi pada ruang aksi 5-dimensi (`[0, 1]^5` yang diskalakan ke `ACTION_COLUMNS` `[0..4]` dalam [`config.py`](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/config.py)) melalui **Hurdle Policy (Bernoulli Gate + Truncated Normal)** setiap $\Delta t = 6.0\text{ jam}$ (64 step / 16 hari):

1. **$T1$ Aerasi (`aeration`, $0\text{–}6.0\text{ jam/step}$):** Safe RLHF menjaga aerasi `4.89–5.93 h/step` pada Fase 1–3 untuk biodegradasi aerobik dan kestabilan DO, kemudian menurunkannya ke **`3.36 h/step`** pada Fase 4 demi efisiensi listrik.
2. **$T2$ Pencahayaan LED (`light_intensity`, $0\text{–}5,000\text{ lux}$):** Ditingkatkan dari **`844 lux`** (Fase 1, menghindari fotoinhibisi awal) ke **`1,661 lux`** (Fase 2 eksponensial), lalu diturunkan ke **`619 lux`** (Fase 4).
3. **$T3$ Molase (`molasses`, $0\text{–}20.0\text{ mL/step}$):** Ditambahkan secara terukur (`1.2 mL` di Fase 2, `0.2 mL` di Fase 3, `0.7 mL` di Fase 4), menghindari over-dosing COD sekunder.
4. **$T4$ Buffer Alkali $\text{NaHCO}_3$ (`sodium_bicarbonate`, $0\text{–}5.0\text{ g/step}$):** Terkonsentrasi **`11.7 g`** di Fase 1 aklimatisasi untuk menangkal *acid shock* influen POME.
5. **$T5$ Inokulum *Chlorella vulgaris* (`microalgae_inoculum`, $0\text{–}100.0\text{ mL/step}$):** Starter **`41.6 mL`** di Fase 1 diikuti penambahan adaptif (`5.7 mL` di Fase 3, `9.8 mL` di Fase 4).

---

### Pertanyaan 2: "Hasilnya bagaimana dan perbandingan multi-algoritma?"
Evaluasi melintasi 3 seed acak pada lingkungan terkalibrasi fisika (`POMEReactorEnv`, volume kerja $2.5\text{ L}$):

| Algoritma Bioproses | Kategori | Undiscounted Return ($G$) | Pelanggaran ($n_{\text{cv}}/3$) | Violation Rate | Penyisihan COD (%) | Biaya Operasional (IDR) | Pareto Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Safe RLHF (Proposed)** | **Offline-to-Online Safe RLHF** | **+65.64 ± 8.57** | **0 / 3** | **0.0%** | **98.30 ± 0.08%** | **Rp 15,605** | **79.81** |
| **PPO (On-Policy Clip)** | On-Policy RL | +17.18 ± 53.71 | 2 / 3 | 66.7% | 83.56 ± 11.12% | Rp 14,784 | 73.48 |
| **Lagrangian-SAC ($M=1$)** | Constrained RL | +2.80 ± 52.14 | 2 / 3 | 66.7% | 46.21 ± 39.68% | Rp 5,095 | 44.25 |
| **IQL Offline** | Offline RL | -7.49 ± 59.59 | 2 / 3 | 66.7% | 34.17 ± 45.14% | Rp 5,056 | 28.10 |
| **HCR (Operator Baseline)** | Heuristic Supervised | -11.48 ± 53.96 | 2 / 3 | 66.7% | 34.23 ± 45.23% | Rp 5,379 | 27.72 |
| **DPPO (LAPSE Reference)** | Direct Preference Opt | -13.32 ± 51.35 | 2 / 3 | 66.7% | 34.24 ± 45.24% | Rp 5,521 | 27.55 |
| **TD3 / DDPG** | Deterministic RL | -35.89 ± 21.52 | 3 / 3 | 100.0% | 20.68 ± 24.93% | Rp 2,006 | 23.87 |
| **CQL Offline** | Offline RL | -49.42 ± 0.77 | 3 / 3 | 100.0% | 2.25 ± 0.03% | Rp 9 | 2.02 |
| **Standard SAC** | Unconstrained RL | -70.63 ± 29.64 | 2 / 3 | 66.7% | 36.50 ± 42.67% | Rp 24,816 | 28.24 |

> [!IMPORTANT]
> **Temuan Inti & Kejujuran Ilmiah:**
> 1. **Safe RLHF** adalah satu-satunya model yang menjaga **0.0% Violation Rate** sekaligus mencapai **COD Removal `98.30–98.33%`**, **BOD Removal `98.85%`**, dan **TSS Removal `88.70%`** (`report.md`).
> 2. **Gap Kepatuhan pH Tier-1:** Pada data POME mikroalga, serapan $\text{CO}_2$ fotosintetik menaikkan pH efluen akhir ke kisaran alkali `9.1–9.97` ($> 9.0$ PerMenLH Tier-1), sehingga kepatuhan penuh PerMenLH (`compliance_rate_pct`) pada horizon 16 hari tetap `0.0%` untuk Safe RLHF sebelum ada modul *acid polishing* terdedikasi.

---

### Pertanyaan 3: "Untuk generalist kita perlu apa saja dan bagaimana performa pada 100 batch acak?"
Fondasi generalist meliputi representasi `224`-dimensi (`0.1 * symlog(s)`), **Pessimistic Failure Critic Ensemble ($M=4$)**, dan **Risk-Aware Selection** (`risk_aware_selection.py`):
- **Safe RLHF (Proposed):** COD Removal **`95.04 ± 14.77%`** (`COD akhir 228.2 mg/L <= 350 mg/L`), Biaya **`Rp 13,570`**, Violation Rate biologis **`2.59%`**, Pareto **`79.26 ± 11.01`**.
- **DPPO (LAPSE Reference):** COD Removal `94.92 ± 9.31%` (`233.7 mg/L`), Biaya `Rp 11,339`, VR `3.13%`, Pareto `80.99 ± 7.63`.
- **Standard SAC (Unconstrained):** Meski mencatat kepatuhan pH Tier-1 `28.0%`, biaya melonjak **4.3× lipat ke `Rp 58,025`** dengan Violation Rate biologis tertinggi (**`5.96%`**) dan Pareto terendah (`68.22 ± 12.71`).

---

### Pertanyaan 4: "Bagaimana kita menyelesaikan masalah data sintesis?"
Generator fisika-biokimia pada [`bioprocess_core.py`](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/models/bioprocess_core.py) memperbaiki cacat legacy:
1. Volume kerja distandardisasi ke **`2.5 L`** dalam bejana **`3.0 L`** ($z = 0.35\text{ m}$).
2. Kelarutan oksigen mengikuti **Benson & Krause (1984)**, kesetimbangan amonia **Emerson et al. (1975)**, suhu **Rosso et al. (1993)**, cahaya **Steele (1962)**, dan kinetika COD 2-fraksi.
3. **Hasil Audit ([`synthetic_vs_real_audit_metrics.csv`](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/synthetic_vs_real_audit_metrics.csv)):** Normalized $W_1$-distance terpangkas pada **COD (`-88.8%`)**, **BOD (`-88.8%`)**, **TSS (`-76.7%`)**, **TDS (`-56.4%`)**, **$\text{NH}_3$ (`-31.3%`)**, dan **DO (`-10.2%`)** (`D_KS` COD turun `0.450 -> 0.128`); sementara pada pH terjadi peningkatan minor `W_1` (`-17.7%`).

---

### Pertanyaan 5: "Bagaimana keselarasan dengan paper LAPSE (2026)?"
1. **Residual Bradley-Terry (`train_reward.py`):** Akurasi validasi mencapai **`95.6%`** ($L_s=20$, CE loss `0.1748`) dan **`92.5%`** ($L_s=4, 8, \text{comb}$).
2. **DPPO (`dppo.py`):** Dilatih langsung pada trajektori `224`-dim berskala `ACTION_COLUMNS` tanpa overwrite BC loss.
3. **Keunggulan Safe RLHF:** Penambahan Cal-QL + Failure Ensemble ($M=4$) + Primal-Dual RAS menaikkan Return dari `-13.32` (DPPO) menjadi **`+65.64`** dengan **0.0% VR**.
