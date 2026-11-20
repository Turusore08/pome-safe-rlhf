# Analisis Fitur & Komparasi 3 Dataset Gambar POME (`Baseline` vs `FOTO_crop_K1-K16` vs `FOTO1_crop_K1-K16`)

Laporan otomatis dan terverifikasi dari [`analyze_cropped_photo_datasets.py`](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/analyze_cropped_photo_datasets.py).

## 1. File Hasil & Visualisasi Terkait
- [fig_crop_1_dataset_roi_and_mapping.png](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/figures/fig_crop_1_dataset_roi_and_mapping.png)
- [fig_crop_2_feature_importance_mapping.png](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/figures/fig_crop_2_feature_importance_mapping.png)
- [fig_crop_3_correlation_and_selection_heatmap.png](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/figures/fig_crop_3_correlation_and_selection_heatmap.png)
- [fig_crop_4_3dataset_distribution_and_benchmark.png](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/figures/fig_crop_4_3dataset_distribution_and_benchmark.png)
- [photo_datasets_extracted_features_summary.csv](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/photo_datasets_extracted_features_summary.csv)
- [photo_datasets_feature_importance.csv](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/photo_datasets_feature_importance.csv)
- [photo_datasets_feature_selection.csv](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/photo_datasets_feature_selection.csv)
- [photo_datasets_3way_benchmark.csv](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/photo_datasets_3way_benchmark.csv)

## 2. Ringkasan Statistik & Cakupan Dataset
- **`Baseline` (`foto/` uncropped `FOTO` + `FOTO1`):** `93.134` foto *in-period* di `16/16` kultivasi (`5.362 / 5.696` jam cocok = `94,1%`; `5.356` valid, `6` korup).
- **`FOTO_crop_K1-K16` (`128×128` crop `FOTO`):** `61.879` foto di `16/16` kultivasi (`5.328 / 5.696` jam cocok = `93,5%`; `0` korup).
- **`FOTO1_crop_K1-K16` (`168×168` crop `FOTO1`):** `27.703` foto di `11/16` kultivasi (`2.613 / 5.696` jam cocok = `45,9%`; `0` korup), setelah membuang `6.735` foto di `daftar_foto_tidak_masuk.csv` (`3.406` di luar periode, `2.776` frame hitam lampu mati `PWM 0%`, `549` gelap/terhalang/geser, `4` tidak bisa dicrop).
- **Irisan Timestamp Sama:** `2.582` jam di `11` kultivasi.

## 3. Feature Importance & Selection
- Share Kepentingan Kategori (`Algal Pigment Indices` / `HSV & CIE L*a*b*` / `RGB Reflectance` / `Optical & Spatial Texture`):
  - `Baseline`: `14,37%` / `25,66%` / `32,82%` / **`27,14%`** (`Gradient_Edge_Energy` = `81,365` Rank #1; `Center_Border_Contrast` = `55,536` Rank #2)
  - `FOTO_crop`: `16,33%` / `28,47%` / `36,67%` / `18,54%` (`B_mean` = `70,889` Rank #1; `HSV_V_mean` = `61,102`; `R_mean` = `59,668`)
  - `FOTO1_crop`: **`19,01%`** / **`33,32%`** / `32,97%` / **`14,71%`** (`B_mean` = `65,400` Rank #1; `G_mean` = `62,492`; `LAB_b_mean` = `60,274`; `Optical_Darkness` = `60,167`)
- Global Top-10 Non-Redundan (`|r| <= 0,92`): `B_mean`, `Gradient_Edge_Energy`, `R_mean`, `R_std`, `LAB_b_mean`, `NGRDI`, `B_std`, `LAB_a_mean`, `Center_Border_Contrast`, `HSV_S_mean`.

## 4. Benchmark 3 Dataset + Dual-Crop (Irisan Sama $n = 2.582$)
- Mean $R^2$ 5-Fold CV (`RF 20 Optical Features`): `FOTO1_crop` **`0,8583`** > `Baseline` `0,8424` > `FOTO_crop` `0,7380`.
- Mean $R^2$ 5-Fold CV (`RF Top-10 Selected`): `Dual_Crop` **`0,8672`** > `FOTO1_crop` `0,8412` > `Baseline` `0,8286` > `FOTO_crop` `0,7412`.
- ConvNeXt `vision_encoder.pt` eksisting (dilatih pada gambar *Baseline* uncropped): `Baseline` `0,5636` vs `FOTO_crop` `0,3269` vs `FOTO1_crop` `0,2420` (menunjukkan `VisionEncoder` perlu dilatih ulang jika pindah ke dataset crop).
- ConvNeXt Fine-Tuned LOCO Held-Out (`K03, K06, K12`): `Baseline` `-0,2832` vs `FOTO_crop` `-0,0733` vs **`FOTO1_crop` `+0,0107`** (`log1p(COD) = +0,0450`, `log1p(TSS) = +0,0680`).

## 5. Audit Grad-CAM CNN (`VisionEncoder stage2`)
Skrip [`generate_gradcam_analysis.py`](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/generate_gradcam_analysis.py) menghasilkan:
- [fig_gradcam_1_baseline_shortcut_vs_cropped_roi.png](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/figures/fig_gradcam_1_baseline_shortcut_vs_cropped_roi.png)
- [fig_gradcam_2_multitarget_bioprocess_attribution.png](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/figures/fig_gradcam_2_multitarget_bioprocess_attribution.png)
- [fig_gradcam_3_quantitative_attention_audit.png](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/figures/fig_gradcam_3_quantitative_attention_audit.png)
- [gradcam_quantitative_metrics.csv](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/gradcam_quantitative_metrics.csv)

Ringkasan metrik rata-rata (`n = 300 × 4 target`):
- **`Baseline` (`foto/` uncropped):** CNN $R^2 = 0,7128$, tetapi **`0,85%`** energi di kotak ROI cairan (**`99,15%` di luar ROI**; **`88,99%` bocor ke latar ruangan** di luar `2×` jar box) dengan area aktif hanya `4,16%` (*shortcut learning* pada tutup botol, selang, & layar alat ukur).
- **`FOTO_crop` (`128×128`):** CNN $R^2 = 0,3430$, **`100,0%` di ROI cairan** (`0,0%` leak), `core_energy_pct = 50,10%`, area aktif `15,33%`.
- **`FOTO1_crop` (`168×168`):** CNN $R^2 = 0,0945$, **`100,0%` di ROI cairan** (`0,0%` leak), `core_energy_pct = 61,45%` (`TSS 78,77%`, `pH 65,71%`), area aktif `18,49%`.

## 6. Eksperimen End-to-End Safe-RLHF & Multi-Algoritma (`90` Run: `Baseline` vs `Crop`)
Skrip [`run_crop_rl_experiments_comparison.py`](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/run_crop_rl_experiments_comparison.py) melatih `OpticalConvNeXtEncoder` (fusi 16 fitur optik Beer-Lambert + 16 laten ConvNeXt 35 *epoch*, mendongkrak `Val R²` `FOTO_crop` ke `0,4589` & `FOTO1_crop` ke `0,4386`), menyusun `d_hist_*.pt`, lalu melatih dan mengevaluasi `6` algoritma × `3` seed × `5` varian dataset (`POMEReactorEnv` Standard Loop & Real-Photo-Grounded Loop):
- [fig_crop_5_rl_pipeline_comparison.png](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/figures/fig_crop_5_rl_pipeline_comparison.png)
- [fig_crop_6_closedloop_trajectories_by_dataset.png](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/figures/fig_crop_6_closedloop_trajectories_by_dataset.png)
- [crop_rl_summary_by_dataset.csv](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/crop_rl_summary_by_dataset.csv), [crop_rl_experiments_comparison.csv](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/crop_rl_experiments_comparison.csv), [crop_rl_experiments_all_runs.csv](file:///home/turusore08/Desktop/KP/pome-safe-rlhf/code/evaluation/crop_rl_experiments_all_runs.csv)

Ringkasan `Safe RLHF` (`Cal-QL+SAC-KL+RAS`) & rata-rata `6` algoritma (*Real-Photo Loop* | *Std Loop*):
- **`Baseline_foto` (`16/16`, `5.680` trans):** Real-Photo `G = 44,84`, `COD Rem = 97,86%`, `Cost = IDR 16.949,4`, `VR = 33,3%`, `Pareto = 77,32` (Std `G = 34,78`, `VR = 33,3%`; 6-Algo Mean `G = -0,47`).
- **`FOTO_crop_K1-K16` (`16/16`, `5.680` trans):** Real-Photo `G = 60,14` (`+34,1%`), `COD Rem = 96,87%`, **`Cost = IDR 12.538,1`** (terhemat -26%), `VR = 33,3%`, **`Pareto = 82,33`** (#1) (Std `G = 26,43`, `VR = 33,3%`; 6-Algo Mean `G = 0,18`).
- **`FOTO1_crop_16runs` (`11/16` + 5 nol, `5.680` trans):** Real-Photo `G = 53,29`, `COD Rem = 97,56%`, `Cost = IDR 15.415,3`, `VR = 33,3%`, `Pareto = 79,12` (Std `G = 28,08`, `VR = 100%` akibat vektor nol).
- **`FOTO1_crop_11runs` (`11/11` aktif murni, `4.180` trans):** **Real-Photo #1: `G = 76,09` (`+69,7%`), `COD Rem = 98,15%`, `BOD = 98,75%`, `TSS = 90,03%`, `Cost = IDR 13.947,9`, `VR = 0,0%` (`0/3` crash!), `Pareto = 81,83`** (Std `G = 64,94`, `VR = 33,3%`; **6-Algo Mean #1: `G = +19,60`, `Pareto = 67,53`**).
- **`Dual_Crop_Fusion` (`16/16` hybrid, `5.680` trans):** **Std Loop #1: `G = 78,00`, `COD Rem = 98,35%`, `VR = 0,0%` (`0/3` crash!), `Pareto = 83,23`**; Real-Photo `G = 56,73`, `COD Rem = 98,08%`, `Cost = IDR 15.219,9`, `VR = 33,3%`, `Pareto = 79,94` (6-Algo Mean `G = +13,46`, `Pareto = 67,12`).
