# Analisis Daya Komputasi & Profiling Edge Deployment

**Sistem Kontrol Pengolahan POME Berbasis Safe Offline-to-Online RLHF**

*Dihasilkan pada: 2026-11-12 16:03:24*

---
## 1. Kompleksitas Parameter & Jejak Memori (Memory Footprint)

Arsitektur jaringan saraf dirancang secara modular dan sangat ringkas agar dapat dijalankan pada perangkat komputasi tepi (*edge computing*) seperti Raspberry Pi 4 tanpa memerlukan akselerator GPU eksternal.

| Komponen Model | Fungsi Teknis | Jumlah Parameter | Ukuran FP32 (KB) |
|:---|:---|---:|---:|
| `SensorEncoder (1D-Conv)` | Ekstraksi fitur / Kebijakan / Critic | 113,792 | 444.5 KB |
| `VisionEncoder (ConvNeXt)` | Ekstraksi fitur / Kebijakan / Critic | 109,184 | 426.5 KB |
| `HurdlePolicy (pi_theta / pi_sft)` | Ekstraksi fitur / Kebijakan / Critic | 62,991 | 246.06 KB |
| `RewardCritic (Q_r)` | Ekstraksi fitur / Kebijakan / Critic | 125,953 | 492.0 KB |
| `FailureCriticEnsemble (M=4)` | Ekstraksi fitur / Kebijakan / Critic | 503,812 | 1968.02 KB |
| **Total Pipeline Inferensi** | **Seluruh Graf Kontrol** | **915,732** | **3.49 MB** |

### Jejak File pada Media Penyimpanan (Disk Footprint):
| Format Berkas | Keterangan | Ukuran (KB) | Kompresi |
|:---|:---|---:|:---:|
| `sft_policy.pt` | Model checkpoint | 252.8 KB | PyTorch Weights |
| `safe_rlhf_policy.pt` | Model checkpoint | 252.82 KB | PyTorch Weights |
| `reward_ensemble.pt` | Model checkpoint | 285.65 KB | PyTorch Weights |
| `policy_edge.onnx` | Model checkpoint | 259.12 KB | ONNX Runtime |
| `policy_quantized_int8.pt` | Model checkpoint | 75.26 KB | INT8 Quantized |

> **Efisiensi:** Kuantisasi INT8 memampatkan model kebijakan menjadi 75.26 KB, memungkinkan pemuatan model secara instan ke dalam memori L2/L3 cache prosesor.

---
## 2. Dekomposisi Latensi Waktu Eksekusi (Latency Breakdown)

Berdasarkan formulasi kendala latensi pada **penjelasan.md §6** dan **Formula F-08.1**:
$$T_{\text{step}} = T_{\text{enc}} + T_{\pi}(N) + T_{Q_r}(N) + M \cdot T_{C}(N) \le \Delta t_{\text{ctrl}}$$

Diuji secara empiris pada CPU dengan $N = 16$ kandidat aksi dan ensemble critic $M = 4$:

| Tahapan Komputasi | Simbol Rumus | Rata-rata (ms) | Median p50 (ms) | Persentil p95 (ms) | Kontribusi (%) |
|:---|:---:|---:|---:|---:|---:|
| Sensor Encoding (1D-Conv) | $T_enc_sensor$ | 0.151 ms | 0.147 ms | 0.170 ms | 0.7% |
| Vision Encoding (ConvNeXt) | $T_enc_vision$ | 2.289 ms | 2.284 ms | 2.539 ms | 10.9% |
| Sampling N Kandidat Aksi | $T_pi(N)$ | 15.987 ms | 15.763 ms | 16.737 ms | 75.8% |
| Scoring Reward Critic | $T_Qr(N)$ | 0.390 ms | 0.382 ms | 0.445 ms | 1.8% |
| Scoring Failure Critic Ensemble (M=4) | $M * T_C(N)$ | 1.483 ms | 1.477 ms | 1.653 ms | 7.0% |
| Lagrangian Selection & Veto | $T_select$ | 0.765 ms | 0.657 ms | 1.220 ms | 3.6% |
| **Total Satu Siklus Keputusan** | $\mathbf{T_{\text{step}}}$ | **21.082 ms** | **20.813 ms** | **22.856 ms** | **100.0%** |

- **Throughput Kontrol:** **47.43 keputusan / detik**.

---
## 3. Analisis Sensitivitas & Skalabilitas

### A. Pengaruh Jumlah Kandidat Aksi ($N$):
Semakin banyak kandidat aksi ($N$) yang diusulkan oleh aktor, semakin kecil probabilitas agen terjebak pada aksi berbahaya, namun beban komputasi meningkat secara linier:

| Jumlah Kandidat ($N$) | Latensi Total (ms) | Porsi Sampling Aktor (ms) | Porsi Evaluasi Critic (ms) | Throughput (Hz) |
|:---:|---:|---:|---:|---:|
| **N=4** | 8.184 ms | 4.079 ms | 1.404 ms | 122.2 Hz |
| **N=8** | 12.526 ms | 8.04 ms | 1.591 ms | 79.8 Hz |
| **N=16** | 21.141 ms | 15.905 ms | 1.941 ms | 47.3 Hz |
| **N=32** | 41.461 ms | 34.14 ms | 2.656 ms | 24.1 Hz |
| **N=64** | 76.433 ms | 65.959 ms | 3.805 ms | 13.1 Hz |
| **N=128** | 150.801 ms | 132.83 ms | 6.579 ms | 6.6 Hz |

### B. Pengaruh Ukuran Ensemble Failure Critic ($M$):
| Jumlah Critic ($M$) | Latensi Evaluasi (ms) | Latensi p95 (ms) |
|:---:|---:|---:|
| **M=2** | 0.749 ms | 0.843 ms |
| **M=4** | 1.354 ms | 1.492 ms |
| **M=6** | 1.999 ms | 2.094 ms |
| **M=8** | 2.669 ms | 2.832 ms |
| **M=10** | 3.32 ms | 3.372 ms |

> **Rekomendasi Konfigurasi:** $N = 16$ kandidat aksi dan $M = 4$ failure critic memberikan *sweet spot* optimal antara diversitas keamanan dan efisiensi waktu eksekusi (21.08 ms pada CPU lokal).

---
## 4. Proyeksi Kinerja pada Perangkat Edge (Raspberry Pi 4)

Fotobioreaktor fisik menggunakan mikrokontroler ESP32 sebagai sensor node dan **Raspberry Pi 4 (Quad-Core Cortex-A72 @ 1.5 GHz)** sebagai edge gateway controller:

- **Latensi Komputasi pada Raspberry Pi 4:** **109.63 ms** (hanya ~0,1 detik).
- **Batas Waktu Kontrol Bioreaktor (Decision Horizon):** **6 jam** = 21.600 detik = 21.600.000 ms.
- **Rasio Margin Keamanan:** **197,033.0x lebih cepat** daripada batas waktu siklus.
- **Duty Cycle Prosesor:** **0.000508%** (prosesor berada dalam status *sleep/idle* 99,999% dari waktu operasional).

### Konsumsi Energi & Daya Operasional:
- **Energi per Siklus Keputusan:** **0.6029 Joule**.
- **Konsumsi Energi Harian Komputasi:** **0.6699 mWh/hari**.
- **Daya Aktuator Fisik Bioreaktor (Pompa, Aerator, Lampu):** **59.4 Watt**.
- **Overhead Komputasi terhadap Aktuator:** Komputasi agen hanya menyumbang fraksi energi tak terukur dibandingkan konsumsi listrik pompa peristaltik dan lampu LED kultivasi.

---
## 5. Perbandingan terhadap Kontroler Klasik (Model Predictive Control / MPC)

Salah satu kontribusi utama penelitian ini adalah mengeliminasi ketergantungan pada *Model Predictive Control* (MPC) analitik murni:

| Aspek Komparasi | Model Predictive Control (MPC) Analitik | Safe Data-Driven RLHF (Penelitian Ini) |
|:---|:---|:---|
| **Paradigma** | Online Non-Linear Programming / Quadratic Programming (NLP/QP) optimization | Forward neural feedforward pass + Risk-Aware Lagrangian selection |
| **Metode Solver** | Interior Point / Sequential Quadratic Programming (IPOPT / OSQP) | Deterministic neural evaluation across N candidates |
| **Kompleksitas Komputasi** | $O(H * (n_x^3 + n_u^3) * N_iter) where H=prediction horizon$ | $O(N * W_layers) = strictly bounded linear complexity$ |
| **Karakteristik Waktu** | Iteratif, tidak deterministik | Deterministik (forward pass konstan) |
| **Latensi Tipikal Edge** | 1,500 ms – 45,000 ms (often times out on non-convex kinetics) | **~109.6 ms projected on Raspberry Pi 4** |
| **Resiko Kegagalan** | Solver *non-convergence* / matrix singular | Fallback ke kandidat risiko minimum |
| **Kebutuhan Memori** | Large sparse Hessian / Jacobian matrices (15–80 MB RAM) | **Compact neural weights (3.49 MB FP32 / 0.87 MB INT8)** |

---
## 6. Kesimpulan Analisis Daya Komputasi

1. **Sangat Ringan & Efisien:** Total graf inferensi kontroler hanya membutuhkan **3.49 MB RAM (FP32) / 0.87 MB (INT8)** dan **915,732 parameter**, sangat kompatibel untuk mikrokomputer kelas rendah.
2. **Latensi Sangat Cepat:** Satu siklus pemilihan aksi aman hanya memakan waktu **21.08 ms pada workstation CPU** dan terproyeksi **109.63 ms pada Raspberry Pi 4**, mengeliminasi bottleneck komputasi.
3. **Keunggulan atas MPC:** Menggantikan solver numerik optimasi kuadratik berulang dengan evaluasi jaringan saraf deterministik menghasilkan eksekusi jauh di bawah jendela kontrol 6 jam tanpa risiko solver macet (*divergence*).