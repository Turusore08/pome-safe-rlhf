# Formal Mathematical Problem Formulation: Constrained MDP & Physics-Informed Safe RLHF for POME Photobioreactors

**Primary References:**
1. Gerold, H., Brandner, D., & Lucia, S. (2026). *Enhancing Control in Chemical Processes using Reinforcement from Human Feedback*. ESCAPE 36 / LAPSE-2026.0510.
2. Notulensi Pertemuan 1 (`Diskusi KP_TA Drone, IoT, dan AI.pdf` & `sistem reward.png` — *"Penerapan Sistem AI Untuk Mencapai Tujuan"*).
3. Physics-Informed Hybrid ODE & LOCO Calibration (`hybrid_ode_loco.ipynb` & `DataSintesis/daftar_hukum_data_sintetis_TA.xlsx`).
4. PerMenLHK No. 5/2014 Lampiran III (Baku Mutu Air Limbah Industri Minyak Sawit).

---

## 1. Constrained Markov Decision Process (CMDP) Formulation

We formulate the closed-loop control of the *Chlorella vulgaris*–bacterial consortium photobioreactor (PBR) treating Palm Oil Mill Effluent (POME) as a finite-horizon **Constrained Markov Decision Process (CMDP)**:
$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, r, c, \gamma, \rho_0, H)$$
where:
- **Decision Horizon ($\Delta t$, $H$)**: Operational control interval $\Delta t = 6\text{ hours}$ (or $1\text{ hour}$ in high-frequency digital twin evaluation) over a 15–16 day batch cultivation horizon ($H = 64$ steps at 6h intervals, or $360$ steps at 1h intervals).
- **Initial State Distribution ($\rho_0$)**: Empirical multivariate copula fitted to initial raw POME loads across the 14 in-distribution historical cultivations ($\mathcal{K}_{\text{in}} = \{1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16\}$), holding out $\mathcal{K}_{\text{OOD}} = \{3, 15\}$ (molasses overdose and early termination anomalies) strictly for Out-of-Distribution stress testing.

---

## 2. 16-Variable IoT Telemetry Schema & State Space $\mathcal{S}$

Following the K1–K16 rig telemetry specification (`config.SENSOR_COLUMNS`), real-time observations are acquired across **16 monitored physical, water-quality, and ambient variables** grouped into four domains (working volume $V_{\text{work}} = 2.5\text{ L}$ inside a $3.0\text{ L}$ cylindrical glass vessel of column depth $d = 35\text{ cm}$):

### A. Liquid-Phase Water Quality Channels
- **Organic Pollutant Load**: Chemical Oxygen Demand ($\text{COD}$, mg/L), Biochemical Oxygen Demand ($\text{BOD}_5$, mg/L), Total Organic Carbon ($\text{TOC}$, mg/L), and Turbidity (NTU). Empirical K1–K16 audit ratios: $\text{BOD}/\text{COD} \approx 0.50$, $\text{TOC}/\text{COD} \approx 0.316$.
- **Dissolved Oxygen & Thermal State**: Dissolved Oxygen ($\text{DO}$, mg/L) and Liquid Broth Temperature ($T_{\text{water}}, ^\circ\text{C}$, recorded as `1_temp_air(C)` where `air` denotes aqueous phase). Saturation $\text{DO}_{\text{sat}}(T, P, S)$ uses **Benson & Krause (1984)** with UNESCO vapor pressure and barometric corrections.
- **Suspended Solids**: Total Suspended Solids ($\text{TSS}$, mg/L), alongside daily manual optical density $\text{OD}_{680}$ ($\text{Biomass} \approx 1.266 \cdot \text{OD}_{680}\text{ g/L}$).
- **Ionic Strength / Salinity**: Total Dissolved Solids ($\text{TDS}$, ppm/mg/L), mapped via the conservative inorganic fraction ($\gamma_{\text{inorg}} = 0.65$) to practical salinity $S_{\text{ppt}} = 0.65 \cdot \text{TDS} / 1000$ and Langelier (1936) ionic strength $I = 2.5 \times 10^{-5} \cdot 0.65 \cdot \text{TDS}$.
- **Acidity / Alkalinity**: Broth $\text{pH}$, governed by the carbonate ($\text{DIC} / \text{Alk}$) and ammonium ($\text{NH}_4^+ / \text{NH}_3$) conservative charge-balance solver $f(\text{pH}) = 0$ (Plummer & Busenberg, 1982; Casagli et al., 2021) with Davies (1962) activity corrections.
- **Dissolved Ammonia**: Unionized free ammonia ($\text{NH}_3\text{-N}$, mg/L) and Total Ammonia Nitrogen ($\text{TAN}$), governed by **Emerson et al. (1975)** temperature-, pH-, and ionic-strength-dependent speciation:
  $$\phi_{\text{NH}_3}(T, \text{pH}, I) = \frac{1}{1 + 10^{\text{pK}_a'(T, I) - \text{pH}}}, \quad \text{pK}_{a,0}(T) = 0.09018 + \frac{2729.92}{T + 273.15}$$

### B. Optical Radiation Channel
- **Photosynthetic Irradiance**: Illuminance (`2_light(Lux)`, $[0, 5000]\text{ lux}$), converted to PAR PPFD via $I_0 = 0.0145 \cdot \text{Lux}$ ($\mu\text{mol photons m}^{-2}\text{ s}^{-1}$, Thimijan & Heins, 1983) and attenuated across the $d = 35\text{ cm}$ column under Beer-Lambert self-shading:
  $$I_{\text{avg}} = I_0 \frac{1 - \exp(-\tau)}{\tau}, \quad \tau = (k_{\text{par}} \ln(10) X + k_{\text{bg}}) \cdot d$$

### C. Ambient & Headspace Gas Channels
- Ambient room temperature (`2_temp(C)`, $^\circ\text{C}$), relative humidity (`2_humidity(%)`, %), barometric pressure (`2_pressure(hPa)`, hPa), headspace/ambient oxygen (`2_o2(ppm)`, uncalibrated MQ channel — excluded from mechanistic ODEs), and gas-phase ammonia (`2_nh3_udara(ppm)`, ppm).

### D. Multimodal State Vector Representation
The agent observes the fused state vector $s_t = [h_x(t) \;\|\; h_z(t)] \in \mathbb{R}^{224}$, combining $192$ multi-resolution rolling statistical descriptors ($h_x$) over the 16 primary telemetry streams with $32$ visual latent features ($h_z$) extracted from reactor photography via a ConvNeXt encoder.

---

## 3. 5-Treatment Action Space $\mathcal{A}$ ($T1$–$T5$, Canonical Indices `[0..4]`)

At each decision step $t$, the agent outputs a 5-dimensional continuous intervention vector matching `config.ACTION_COLUMNS`:
$a_t = (\text{aeration}, \text{sodium\_bicarbonate}, \text{microalgae\_inoculum}, \text{molasses}, \text{light\_intensity}) \in \mathcal{A} \subset \mathbb{R}^5$:

| Canonical Index | Treatment ID | Actuator Description | Operational Bounds | Primary Bioprocess Role |
| :---: | :---: | :--- | :---: | :--- |
| `[0]` | **$T1$** | **Aeration Duration** | $[0.0, 6.0]\text{ h/step}$ ($Q_{\text{ref}} = 0.04\text{ L/s}$) | Supplies $\text{O}_2$ via $k_La(Q)$ for heterotrophic COD oxidation, strips excess $\text{NH}_3$, and exchanges $\text{CO}_2$. |
| `[1]` | **$T4$** | **$\text{NaHCO}_3$ Buffer** | $[0.0, 5.0]\text{ g/step}$ (stock $8.33\text{ g/L}$) | Replenishes carbonate alkalinity ($\text{Alk}$) and DIC, preventing acidic pH crashes. |
| `[2]` | **$T5$** | **Microalgae Inoculum** | $[0.0, 100.0]\text{ mL/step}$ ($X_{\text{ino}} = 1.2\text{ AU}$) | Boosts active *Chlorella vulgaris* cell density during initial lag or post-shock recovery. |
| `[3]` | **$T3$** | **Molasses Supplement** | $[0.0, 20.0]\text{ mL/step}$ ($C_{\text{mol}} = 5000\text{ mg/L}$) | Supplements readily biodegradable organic carbon to balance C:N stoichiometry during bacterial lag. |
| `[4]` | **$T2$** | **Light Intensity** | $[0, 5000]\text{ lux}$ | Drives microalgal photosynthesis ($\mu(I_{\text{avg}})$) for $\text{O}_2$ generation, nutrient uptake, and $\text{CO}_2$ fixation. |

Because chemical dosing in real operations is sparse (zero-inflated with a 1:10 to 1:20 non-zero ratio), the policy $\pi_\theta(a_t \mid s_t)$ uses a **Hurdle Architecture**: a Bernoulli gating head $g_\theta(s_t) \in [0, 1]^5$ coupled with a bounded continuous magnitude head $m_\theta(s_t)$.

---

## 4. State Reward Architecture, Two-Tier Limits & Tri-Reward Variants

### A. Rigorous Techno-Economic State Reward ($\hat{r}_{\text{rig}}$)
Directly implementing the system architecture diagram (`sistem reward.png`), the step reward decomposes into **Benefits (+)** minus **Costs (-)** and **Safety Penalties**:
$$\hat{r}_{\text{rig}}(s_t, a_t, s_{t+1}) = \underbrace{\big(w_E E_t + w_B B_t + w_Q Q_t + \gamma\Phi(s_{t+1}) - \Phi(s_t)\big)}_{\text{Keuntungan (+)}} - \underbrace{\big(w_{CT} CT_t + w_{CO} CO_t\big)}_{\text{Biaya (-)}}$$

1. **Tier-1 Effluent Quality Gain ($E_t$ — PerMenLHK No. 5/2014 Lampiran III)**:
   Let the normalized statutory compliance gap be:
   $$e(s) = \sum_{p \in \{\text{COD, BOD, TSS, NH}_3\}} \left[\frac{y_p(s) - L_p}{L_p}\right]_+ + [6.0 - \text{pH}(s)]_+ + [\text{pH}(s) - 9.0]_+$$
   where $L_{\text{COD}} = 350\text{ mg/L}$, $L_{\text{BOD}} = 100\text{ mg/L}$, $L_{\text{TSS}} = 250\text{ mg/L}$, and $L_{\text{NH}_3} = 50\text{ mg/L}$. Then $E_t$ rewards monotonic gap reduction $\Delta e_t = \max(0, e(s_t) - e(s_{t+1}))$, potential-based shaping $\gamma \Phi(s_{t+1}) - \Phi(s_t)$ with $\Phi(s) = -e(s)$ (Ng et al., 1999), and a sustained compliance bonus $+1.0 \cdot \mathbb{I}[e(s_{t+1}) = 0]$.
2. **Biomass Valorization ($B_t$)**:
   Commercial value of net microalgal biomass growth $\Delta m_{X,t} = \max(0, X_{t+1} - X_t) \cdot V_{\text{work}}$ ($V_{\text{work}} = 2.5\text{ L}$).
3. **$\text{CO}_2$ Sequestration Credit ($Q_t$)**:
   Stoichiometric photosynthetic carbon fixation ($1.83\text{ g CO}_2\text{ fixed per g photoautotrophic dry biomass}$, Chisti 2007).
4. **Material Expenditure ($CT_t$)**:
   Chemical reagent cost for $T4$ ($\text{NaHCO}_3$, Rp 25/g), $T5$ (inoculum, Rp 15/mL), and $T3$ (molasses, Rp 35/mL).
5. **Operational & Overhead Expenditure ($CO_t$)**:
   Electrical utility cost (Permen ESDM No. 7/2024 tariff Rp 1.444,70/kWh) for $T1$ (25 W aerator blower, Rp 36.12/h), $T2$ (18 W LED array, Rp 0.03121/(lux·6h)), and IoT gateway overhead (Rp 15/step).
6. **Tier-2 Biological Safety Constraint Cost ($c_t$)**:
   Absorbing safety violations follow the physiological survival envelope of *C. vulgaris* (`config.BIOLOGICAL_SAFETY_LIMITS`), distinct from Tier-1 discharge limits so photosynthetic alkalization up to pH ~10.3 remains viable during mid-batch remediation:
   $$c(s_t, a_t, s_{t+1}) = \mathbb{I}\Big[\text{pH}_{t+1} \notin [5.5, 10.5] \;\lor\; \text{DO}_{t+1} < 0.5\text{ mg/L} \;\lor\; T_{\text{water}, t+1} \notin [18.0, 38.0]^\circ\text{C} \;\lor\; \frac{\text{COD}_{t+1}}{\text{COD}_0} > 1.25\Big]$$

### B. Preference-Learned Residual Reward ($\hat{r}_{\text{res}}$) & Hybrid Reward ($\hat{r}_{\text{hybrid}}$) — LAPSE Adaptation
Following Gerold et al. (2026), given trajectory segments $\sigma^1, \sigma^2$ of length $L_s \in \{4, 8, 20, \text{comb}\}$, an ensemble of $M_{\text{rew}} = 3$ neural residual networks $\hat{r}_{\psi_m}(s_t, a_t)$ is trained jointly with the explicit rig prior inside the Bradley-Terry preference model:
$$\hat{r}_{\text{hybrid}, m}(s_t, a_t, s_{t+1}) = \hat{r}_{\text{rig}}(s_t, a_t, s_{t+1}) + \alpha_{\text{res}} \hat{r}_{\psi_m}(s_t, a_t)$$
$$P_\psi(\sigma^1 \succ \sigma^2) = \frac{\exp\left(\sum_{t=1}^{L_s} \hat{r}_{\text{hybrid}}(s_t^1, a_t^1, s_{t+1}^1)\right)}{\exp\left(\sum_{t=1}^{L_s} \hat{r}_{\text{hybrid}}(s_t^1, a_t^1, s_{t+1}^1)\right) + \exp\left(\sum_{t=1}^{L_s} \hat{r}_{\text{hybrid}}(s_t^2, a_t^2, s_{t+1}^2)\right)}$$

---

## 5. Standardized Scientific Evaluation Metrics (LAPSE Benchmark Suite)

To align with Table 1, Table 2, and Table 3 of Gerold et al. (2026) and satisfy all Meet 1 evaluation requirements:

1. **Undiscounted Cumulative Return ($\bar{G} \pm \sigma_G$)**:
   $$G(\tau) = \sum_{t=0}^{H-1} r_{\text{eval}}(s_t, a_t, s_{t+1}), \quad \bar{G} = \frac{1}{N_{\text{ep}}} \sum_{i=1}^{N_{\text{ep}}} G(\tau_i)$$
2. **Total Count of Constraint Violations ($n_{\text{CV}}$ / CNCV)**:
   $$n_{\text{CV}} = \sum_{i=1}^{N_{\text{ep}}} \sum_{t=0}^{H-1} c(s_{i,t}, a_{i,t}, s_{i,t+1})$$
3. **Mean Constraint Violation Magnitude ($\bar{d}_{\text{CV}}$)**:
   Normalized L1 excursion outside the Tier-2 biological safety envelope (`BIOLOGICAL_SAFETY_LIMITS`: $\text{pH} \in [5.5, 10.5]$, $\text{DO} \ge 0.5\text{ mg/L}$, $T_{\text{water}} \in [18.0, 38.0]^\circ\text{C}$, COD rebound $\le 1.25$) across all violating steps.
4. **Statutory Compliance Day ($T_{\text{comp}}$)**:
   First cultivation day (in days) at which all 5 Tier-1 PerMenLHK No. 5/2014 effluent parameters ($\text{COD} \le 350$, $\text{BOD} \le 100$, $\text{TSS} \le 250$, $\text{NH}_3 \le 50$, $\text{pH} \in [6.0, 9.0]$) are simultaneously satisfied and held for $\ge 24\text{ hours}$.
5. **5-Treatment Action Breakdown ($T1$–$T5$) & Step-by-Step Reward/Punishment Ledger**:
   Daily and cumulative expenditure of each actuator $T1$–$T5$ alongside a per-step ledger of $(E_t, B_t, Q_t, CT_t, CO_t, c_t, r_t, G_t)$.
