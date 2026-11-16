# Safe Offline-to-Online RLHF Evaluation Report

## 1. Executive Summary & Baseline Comparison

Evaluation results comparing the Safe RLHF controller against published predecessor runs on the 2.5 L POME photobioreactor:

| Run          | Controller                         |   Days to Compliance | COD Removal (%)   |   Biomass (g/L) | Operating Cost (IDR)   |   Safety Violations |
|:-------------|:-----------------------------------|---------------------:|:------------------|----------------:|:-----------------------|--------------------:|
| K8_HCR       | hurdle_chained_regression          |                    5 | 95.70%            |            2.49 | Rp6,170                |                   0 |
| K5_BestHuman | human_expert                       |                    8 | 98.56%            |            2.1  | Rp9,872                |                   0 |
| K7_Direct    | direct_regression                  |                   16 | 92.05%            |            1.68 | Rp19,748               |                   0 |
| RLHF_Agent   | Safe Offline-to-Online RLHF (Ours) |                   16 | 98.33%            |            0.91 | Rp14,642               |                   0 |

## 2. Biochemical Efficacy (Pollutant Removal)

- **COD Removal Percentage:** 98.33%
- **BOD Removal Percentage:** 98.85%
- **TSS Removal Percentage:** 88.70%
- **Final Algal Biomass:** 0.91 g/L (Chlorella vulgaris)

## 3. Actuation Efficiency & Chemical Costs

- **Average Operating Cost:** IDR 14,642 per batch
- **Cost vs Hurdle Baseline (K8, IDR 6,170):** -137.31%
- **Cost vs Best Human (K5, IDR 9,872):** -48.32%

## 4. Safety & Regulatory Compliance (PerMenLHK No. 5/2014)

- **Days to Full Compliance:** 16.0 days (Target: <= 5 days)
- **Safety Violation Rate (VR):** 0.0%
- **Risk-Aware Selection Intervention Rate (IR):** 2.2%
- **Total Dangerous Actions Blocked / Rejected:** 7
