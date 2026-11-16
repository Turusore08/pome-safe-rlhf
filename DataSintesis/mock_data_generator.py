"""
Calibrated Physics-Informed Mechanistic & Copula Synthetic Data Generator
for the POME Microalgae-Bacteria Photobioreactor (DataSintesis v3 - Literature Unified).

Uses `code/models/bioprocess_core.py` and `code/config.py` as single authoritative sources:
1. Reactor Geometry: V_work = 2.5 L (2500 mL inside 3.0 L glass shell), optical depth z = 0.35 m.
2. Benson & Krause (1984) DO solubility + UNESCO PSS-78 salinity & pressure corrections.
3. Plummer & Busenberg (1982) + Davies (1962) carbonate/ammonia speciation with conservative
   charge-balance root solving `solve_ph(dic, z_net, tan, T, I)` (no ad-hoc pH pull).
4. Thimijan & Heins (1983) Lux -> PPFD conversion + Beer-Lambert self-shading + Bernard & Remond (2012)
   Haldane photoinhibition (`f_I(0) == 0.0`, zero dark growth floor).
5. Rosso et al. (1993) CTMI cardinal temperature (`f_T == 0` outside [T_min, T_max], zero floor) +
   Casagli et al. (2021 ALBA) pH CPM + Azov & Goldman (1982) free-NH3 inhibition.
6. Two-fraction heterotrophic COD oxidation + VFA alkalinity release + nitrification alkalinity sink +
   organic-N ammonification + semi-implicit mass-balance DO integration.
"""

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "code"))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

try:
    from config import TIMESTAMP_COL
except ImportError:
    TIMESTAMP_COL = "timestamp"

from config import (  # noqa: E402
    REACTOR_VOLUME_LITERS,
    DEPTH_CM,
    C_MOL,
    X_INO,
    AERATOR_FLOW_LS,
    NAHCO3_STOCK_G_PER_ML,
)
from models.bioprocess_core import (  # noqa: E402
    LUX_TO_PPFD_LED,
    MW_C,
    MW_N,
    MW_NAHCO3,
    DIFFUSIVITY_RATIO_CO2_O2_SQRT,
    salinity_from_tds,
    ionic_strength_from_tds,
    do_saturation_mgL,
    kla_temp_factor,
    free_nh3_fraction,
    carbonate_constants,
    carbonate_fractions,
    compute_conservative_charge_z,
    solve_ph,
    f_light_haldane_bernard,
    ctmi_cardinal_np,
)

STEPS_PER_15MIN_SECONDS = 900.0

# Empirical initial condition prototypes extracted from the 14 in-distribution
# real cultivations (K1, K2, K4, K5, K6, K7, K8, K9, K10, K11, K12, K13, K14, K16)
REAL_INDIST_PROTOTYPES = [
    {"cod": 1000.4, "tss": 72.0,   "do": 1.24, "ph": 8.02, "tds": 1773.9, "nh3": 38.0,  "od680": 0.81},
    {"cod": 3418.4, "tss": 2277.4, "do": 1.00, "ph": 8.64, "tds": 2195.4, "nh3": 91.3,  "od680": 1.07},
    {"cod": 4704.8, "tss": 2339.9, "do": 3.09, "ph": 8.36, "tds": 7317.3, "nh3": 125.0, "od680": 1.28},
    {"cod": 4893.5, "tss": 1674.8, "do": 3.08, "ph": 8.66, "tds": 6342.4, "nh3": 104.4, "od680": 1.43},
    {"cod": 4118.5, "tss": 1088.4, "do": 0.38, "ph": 9.33, "tds": 5702.9, "nh3": 79.5,  "od680": 1.14},
    {"cod": 4168.4, "tss": 927.1,  "do": 2.11, "ph": 9.34, "tds": 6510.7, "nh3": 72.8,  "od680": 1.53},
    {"cod": 4286.5, "tss": 3196.8, "do": 3.20, "ph": 8.41, "tds": 6324.2, "nh3": 138.6, "od680": 1.98},
    {"cod": 3715.0, "tss": 2429.4, "do": 1.58, "ph": 8.85, "tds": 5338.1, "nh3": 197.8, "od680": 1.37},
    {"cod": 5170.6, "tss": 2047.8, "do": 0.25, "ph": 9.55, "tds": 5692.4, "nh3": 60.6,  "od680": 1.77},
    {"cod": 4466.3, "tss": 1481.3, "do": 0.21, "ph": 8.97, "tds": 5779.8, "nh3": 71.6,  "od680": 1.19},
    {"cod": 4180.1, "tss": 1262.8, "do": 0.40, "ph": 8.22, "tds": 5568.5, "nh3": 67.4,  "od680": 0.97},
    {"cod": 5379.0, "tss": 1377.2, "do": 9.79, "ph": 8.17, "tds": 8054.9, "nh3": 44.8,  "od680": 1.10},
    {"cod": 4566.4, "tss": 1134.3, "do": 5.08, "ph": 8.32, "tds": 7082.3, "nh3": 52.6,  "od680": 0.86},
    {"cod": 4764.0, "tss": 1386.4, "do": 2.78, "ph": 8.00, "tds": 3943.9, "nh3": 48.2,  "od680": 1.09},
]


@dataclass
class GeneratorParams:
    # --- Physical Reactor Geometry (K1-K16, Mirda et al. 2026) ---
    reactor_volume_l: float = REACTOR_VOLUME_LITERS       # 2.5 L liquid volume
    light_path_m: float = DEPTH_CM / 100.0                # 0.35 m water column depth
    evaporation_frac_per_day: float = 0.0022              # [data] matches net evaporative TDS drift

    # --- Dosing Stock Concentrations (linked to config.py) ---
    nahco3_g_per_ml: float = NAHCO3_STOCK_G_PER_ML        # 0.00833 g/mL (~0.0992 mmol/mL)
    molasses_cod_mg_per_ml: float = C_MOL / 1000.0        # 5.0 mg COD / mL stock (5000 mg/L)
    algae_stock_mg_per_ml: float = X_INO * 1.266          # ~1.519 mg dry biomass / mL at OD680=1.2

    # --- Photobioreactor Light & Kinetics (Bernard & Remond 2012, Rosso 1993, Casagli 2021) ---
    lux_to_umol: float = LUX_TO_PPFD_LED                  # 0.0145 umol/m^2/s per lux (Thimijan & Heins 1983)
    k_algae_m2_per_g: float = 0.055                       # broadband PAR self-shading cross-section
    k_background_per_m: float = 1.6                       # POME humic/tannin color attenuation
    mu_max_per_day: float = 0.85                          # [literatur] C. vulgaris max specific growth (~0.035/h)
    i_opt_umol: float = 4.5                               # col-averaged PPFD inside turbid 35cm POME
    light_beta_i: float = 0.5                             # Bernard & Remond (2012) curvature
    t_min_c: float = 10.0                                 # CTMI cardinal minimum
    t_opt_c: float = 30.0                                 # CTMI cardinal optimum
    t_max_c: float = 40.0                                 # CTMI cardinal maximum
    ph_min: float = 4.8                                   # Casagli et al. (2021) CPM lower bound
    ph_opt: float = 8.4                                   # alkalitolerant C. vulgaris consortium optimum
    ph_max: float = 10.8                                  # CPM upper bound
    k_n_mg_per_l: float = 5.0                             # TAN half-saturation
    k_dic_mol_per_l: float = 5e-5                         # inorganic C half-saturation (mol/L)
    ki_free_nh3_mg_per_l: float = 20.4                    # Azov & Goldman (1982) unionized NH3 inhibition
    algae_decay_per_day: float = 0.06                     # endogenous decay / grazing
    algae_respiration_per_day: float = 0.035              # basal metabolic O2 respiration
    lysis_cod_fraction: float = 0.08                      # soluble/slow COD returned upon decay

    # --- Mass-Balance Stoichiometry (Henze 2000 ASM / Chisti 2007) ---
    o2_per_biomass: float = 1.60                          # g O2 produced / g photoautotrophic biomass
    n_per_biomass: float = 0.068                          # g N / g biomass (C5H7NO2 / Chlorella ~6.8% N)
    c_per_biomass: float = 0.50                           # g C / g biomass (CO2_TO_BIOMASS=1.833)
    cod_per_biomass: float = 1.42                         # g COD / g volatile biomass
    biomass_mg_per_od680: float = 780.0                   # empirical OD680 optical calibration against TSS/algae

    # --- Heterotrophic COD Degradation & Nitrogen Cycling (ASM1/ALBA style) ---
    fast_cod_initial_share: float = 0.62                  # biodegradable VFA/soluble share
    k_fast_cod_per_day: float = 0.38                      # rapid soluble COD consumption (Days 0-5)
    k_slow_cod_per_day: float = 0.14                      # slow hydrolysis / particulate COD rate
    k_do_half_mg_per_l: float = 0.35                      # aerobic Monod DO half-saturation
    o2_per_cod_oxidized: float = 0.42                     # net heterotrophic O2 demand per mg COD oxidized (1 - Y_H)
    theta_het: float = 1.035                              # heterotrophic Arrhenius temperature coefficient
    n_per_cod: float = 0.016                              # organic-N ammonification per mg COD oxidized (~60:1 POME C:N)
    k_nitrif_per_day: float = 0.018                       # nitrification NH4+ -> NO3- (algal-bacterial PBR)
    detritus_settling_per_day: float = 0.095              # particulate POME floc sedimentation
    vfa_alk_per_cod_eq: float = 0.0045                    # VFA oxidation mineral alkalinity release (meq/mg COD)

    # --- Gas-Liquid Mass Transfer & Volatilization ---
    kla_surface_per_h: float = 0.25                       # passive headspace transfer at Q=0
    kla_aerator_per_h: float = 1.85                       # active aeration transfer at Q_REF=0.04 L/s
    theta_kla: float = 1.024                              # ASCE / Metcalf & Eddy Arrhenius theta
    co2_kla_ratio: float = DIFFUSIVITY_RATIO_CO2_O2_SQRT  # 0.90 (Danckwerts surface renewal)
    nh3_strip_per_day: float = 0.085                      # interfacial NH3 stripping (function of phi_NH3 & Q)

    # --- Thermal Balance ---
    heat_exchange_per_h: float = 0.28

    # --- Empirical Run-to-Run Distributions (K1-K16 In-Distribution) ---
    air_temp_c: tuple = (33.55, 1.55)
    water_cooling_c: tuple = (3.45, 1.40)
    co2_ppm: tuple = (595.0, 0.25)
    pressure_hpa: tuple = (1007.1, 1.6)

    # --- Calibrated Telemetry Noise Model ---
    fast_tau_min: float = 5.0
    slow_tau_h: float = 12.0
    spike_decay_min: float = 20.0
    cod_noise_intercept: float = 2.5
    cod_noise_slope: float = 0.010
    bod_cod_ratio: float = 0.500
    toc_cod_ratio: float = 0.316
    cod_floor_mg_per_l: float = 38.0
    do_gain_log_sd: float = 0.14
    tss_gain_log_sd: float = 0.12
    ph_offset_sd: float = 0.22
    jump_rate_per_day: float = 0.35


SENSOR_NOISE: Dict[str, tuple] = {
    "tss": (14.0, 22.0),
    "ph": (0.008, 0.006),
    "do": (0.035, 0.028),
    "tds": (20.0, 12.0),
    "nh3": (0.10, 0.32),
    "water_temp": (0.11, 0.038),
    "air_temp": (0.095, 0.035),
    "humidity": (0.19, 0.085),
    "solar_radiation": (0.0035, 0.14),
}
JUMP_SD: Dict[str, float] = {"tss": 85.0, "nh3": 3.5}
FAST_NOISE_15MIN_FACTOR = 0.67


def compute_do_saturation(
    temp_c: float,
    pressure_hpa: float = 1007.1,
    tds_ppm: float = 5600.0,
) -> float:
    """Dissolved oxygen saturation (mg/L) via Benson & Krause (1984) + UNESCO salinity."""
    sal = float(salinity_from_tds(tds_ppm))
    return float(max(2.0, do_saturation_mgL(temp_c, pressure_hpa, sal)))


def free_ammonia_fraction(ph: float, temp_c: float, ionic_strength: float = 0.04) -> float:
    """Emerson et al. (1975) unionized NH3 fraction."""
    return float(free_nh3_fraction(temp_c, ph, ionic_strength))


def light_response(i_umol: float, p: GeneratorParams) -> float:
    """Specific growth rate (1/day) under light (Bernard & Remond, 2012): strictly 0 in dark."""
    return float(p.mu_max_per_day * f_light_haldane_bernard(i_umol, p.i_opt_umol, p.light_beta_i))


def temperature_response(temp_c: float, p: GeneratorParams) -> float:
    """Cardinal Temperature Model with Inflection (Rosso et al., 1993)."""
    return float(ctmi_cardinal_np(temp_c, p.t_min_c, p.t_opt_c, p.t_max_c))


def ph_response(ph: float, p: GeneratorParams) -> float:
    """Casagli et al. (2021) / Rosso (1995) cardinal pH multiplier."""
    return float(ctmi_cardinal_np(ph, p.ph_min, p.ph_opt, p.ph_max))


def sample_conditions(rng: np.random.Generator, p: GeneratorParams, lamp_max_lux: Optional[float]) -> dict:
    """Bootstrap initial cultivation states from the 14 real in-distribution runs."""
    proto = REAL_INDIST_PROTOTYPES[int(rng.integers(0, len(REAL_INDIST_PROTOTYPES)))]

    cod_start = float(np.clip(proto["cod"] * np.exp(rng.normal(0.0, 0.08)), 1000.0, 5400.0))
    tss_start = float(np.clip(proto["tss"] * np.exp(rng.normal(0.0, 0.12)), 450.0, 2400.0))
    do_start = float(np.clip(proto["do"] * np.exp(rng.normal(0.0, 0.18)), 0.35, 5.20))
    ph_start = float(np.clip(proto["ph"] + rng.normal(0.0, 0.28), 7.60, 9.75))
    tds_start = float(np.clip(proto["tds"] + rng.normal(0.0, 320.0), 1800.0, 8100.0))
    tan_start = float(np.clip(proto["nh3"] * np.exp(rng.normal(0.0, 0.14)), 25.0, 145.0))
    od680_start = float(np.clip(proto["od680"] * np.exp(rng.normal(0.0, 0.10)), 0.55, 1.85))

    biomass_start = float(np.clip(od680_start * p.biomass_mg_per_od680, 220.0, 950.0))
    detritus_start = float(max(80.0, tss_start - biomass_start))

    return {
        "lamp_max_lux": lamp_max_lux if lamp_max_lux else float(rng.choice([1050.0, 1350.0, 1800.0])),
        "air_temp": float(np.clip(rng.normal(*p.air_temp_c), 30.2, 37.2)),
        "water_cooling": float(np.clip(rng.normal(*p.water_cooling_c), 1.2, 6.2)),
        "co2_ppm": float(np.clip(p.co2_ppm[0] * np.exp(rng.normal(0.0, p.co2_ppm[1])), 260.0, 2200.0)),
        "pressure_hpa": float(np.clip(rng.normal(*p.pressure_hpa), 1003.5, 1011.0)),
        "init_ph": ph_start,
        "cod_start": cod_start,
        "biomass_start": biomass_start,
        "detritus_start": detritus_start,
        "od680_start": od680_start,
        "do_start": do_start,
        "tan_start": tan_start,
        "tds_start": tds_start,
        "fast_cod_rate": float(p.k_fast_cod_per_day * np.exp(rng.normal(0.0, 0.16))),
        "slow_cod_rate": float(p.k_slow_cod_per_day * np.exp(rng.normal(0.0, 0.16))),
        "mu_max_run": float(p.mu_max_per_day * np.exp(rng.normal(0.0, 0.20))),
        "dic_init_scale": float(np.exp(rng.normal(0.0, 0.25))),
        "do_gain": float(np.exp(rng.normal(0.0, p.do_gain_log_sd))),
        "tss_gain": float(np.exp(rng.normal(0.0, p.tss_gain_log_sd))),
        "ph_offset": float(rng.normal(0.0, p.ph_offset_sd)),
        "irradiance_base": float(np.clip(rng.normal(3.45, 0.85), 1.2, 6.2)),
    }


def make_schedule(rng: np.random.Generator, n: int, hours: np.ndarray, dt_s: float, treated: bool) -> dict:
    """Hurdle-calibrated operator actuator schedule matching K1-K16 dosing frequencies."""
    schedule = {name: np.zeros(n) for name in ["aeration", "nahco3", "algae", "molasses", "pwm"]}
    if not treated:
        return schedule

    def daily_on(start, duration):
        return ((hours - start) % 24.0) < duration

    aer_dur = float(rng.choice([24.0, 22.0, 18.0], p=[0.78, 0.12, 0.10]))
    schedule["aeration"] = daily_on(4.0, aer_dur) * float(rng.choice([0.04, 0.0397, 0.036], p=[0.65, 0.20, 0.15]))

    lamp_dur = float(rng.choice([24.0, 22.0, 18.0], p=[0.75, 0.15, 0.10]))
    schedule["pwm"] = daily_on(5.0, lamp_dur) * float(rng.choice([100.0, 98.3, 90.0, 75.0], p=[0.50, 0.20, 0.22, 0.08]))

    steps_per_hour = max(1, int(round(3600.0 / dt_s)))
    n_hours = n // steps_per_hour

    dosing_specs = [
        ("nahco3", 0.112, np.log(7.2), 0.72, 0.5, 65.0),
        ("molasses", 0.096, np.log(5.8), 0.75, 0.5, 55.0),
        ("algae", 0.052, np.log(10.5), 0.68, 0.5, 60.0),
    ]
    for name, prob_hr, mu_log, sd_log, vmin, vmax in dosing_specs:
        for h_idx in range(n_hours):
            day_frac = h_idx / max(1.0, float(n_hours))
            eff_prob = prob_hr * (1.25 if day_frac < 0.65 else 0.65)
            if rng.random() < eff_prob:
                vol_hr = float(np.clip(rng.lognormal(mu_log, sd_log), vmin, vmax))
                dur_steps = min(steps_per_hour, max(1, int(rng.integers(1, max(2, steps_per_hour // 6)))))
                start_offset = int(rng.integers(0, max(1, steps_per_hour - dur_steps + 1)))
                s_step = h_idx * steps_per_hour + start_offset
                schedule[name][s_step : s_step + dur_steps] += vol_hr / dur_steps

    return schedule


def simulate_process(
    n: int, dt_s: float, cond: dict, sched: dict, env: dict, p: GeneratorParams
) -> Dict[str, np.ndarray]:
    """
    Integrates verified mass- and charge-conservative bioprocess dynamics over n steps (V_work = 2.5 L).
    """
    dt_h = dt_s / 3600.0
    dt_d = dt_s / 86400.0
    names = [
        "cod", "cod_fast", "cod_slow", "algae", "detritus",
        "tan", "ph", "do", "tds", "water_temp",
    ]
    out = {name: np.zeros(n) for name in names}

    water_temp = cond["air_temp"] - cond["water_cooling"]
    tds = cond["tds_start"]
    ion = float(ionic_strength_from_tds(tds))
    tan_mg_l = cond["tan_start"]
    tan_mol_l = (tan_mg_l / 1000.0) / MW_N
    ph = cond["init_ph"]

    # Initialize buffered POME DIC (~12-36 mmol/L across batches) & invert electroneutrality
    dic_mol_l = float(np.clip((0.014 + 0.0012 * (tan_mg_l / 14.0)) * cond.get("dic_init_scale", 1.0), 0.010, 0.042))
    z_net = compute_conservative_charge_z(ph, dic_mol_l, tan_mol_l, water_temp, ion)

    cod_total = cond["cod_start"]
    cod_fast = cod_total * p.fast_cod_initial_share
    cod_slow = cod_total * (1.0 - p.fast_cod_initial_share)
    algae = cond["biomass_start"]
    detritus = cond["detritus_start"]
    do = cond["do_start"]
    v_ml = p.reactor_volume_l * 1000.0

    for i in range(n):
        t_days = (i * dt_s) / 86400.0

        # 1. Conservative volume-overflow mass balance on 2.5 L working liquid volume
        nahco3_ml = float(sched["nahco3"][i])
        molasses_ml = float(sched["molasses"][i])
        inoculum_ml = float(sched["algae"][i])
        added_ml = nahco3_ml + molasses_ml + inoculum_ml
        if added_ml > 0.0:
            tot_ml = v_ml + added_ml
            dilution = v_ml / tot_ml
            cod_fast *= dilution
            cod_slow *= dilution
            algae *= dilution
            detritus *= dilution
            tan_mol_l *= dilution
            dic_mol_l *= dilution
            z_net *= dilution
            tds *= dilution

            # Exact stock additions in concentration units after mixing
            cod_fast += (molasses_ml * p.molasses_cod_mg_per_ml) * 1000.0 / tot_ml
            algae += (inoculum_ml * p.algae_stock_mg_per_ml) * 1000.0 / tot_ml
            added_bic_mol_l = (nahco3_ml * p.nahco3_g_per_ml / MW_NAHCO3) * 1000.0 / tot_ml
            dic_mol_l += added_bic_mol_l
            z_net += added_bic_mol_l  # Na+ (+1 eq) balances HCO3- addition
            tds += added_bic_mol_l * MW_NAHCO3 * 1000.0

        # 2. Gentle evaporation concentrates conservative solutes
        evaporated = min(0.05, p.evaporation_frac_per_day * dt_d)
        concentrate = 1.0 / (1.0 - evaporated)
        tds = min(9250.0, tds * concentrate)
        z_net *= concentrate
        dic_mol_l *= concentrate
        tan_mol_l *= concentrate

        aeration = float(sched["aeration"][i])
        pwm = float(sched["pwm"][i])

        # 3. Thermal balance
        diurnal_wave = 0.65 * np.sin(2.0 * np.pi * (t_days - 0.25))
        target_water_temp = env["air_temp"][i] - cond["water_cooling"] + diurnal_wave
        water_temp += dt_h * p.heat_exchange_per_h * (target_water_temp - water_temp)
        water_temp = float(np.clip(water_temp, 20.0, 39.5))

        # 4. Ion speciation & Davies-corrected free NH3
        ion = float(ionic_strength_from_tds(tds))
        tan_mg_l = max(0.0, tan_mol_l * MW_N * 1000.0)
        free_frac = float(free_nh3_fraction(water_temp, ph, ion))
        free_nh3_mg_l = tan_mg_l * free_frac * (17.031 / MW_N)

        # 5. Photoautotrophic Algal Growth (Bernard & Remond 2012, CTMI, CPM, Liebig)
        i_surface_ppfd = (pwm / 100.0) * cond["lamp_max_lux"] * p.lux_to_umol
        optical_depth = max(0.05, (p.k_algae_m2_per_g * (algae / 1000.0) + p.k_background_per_m) * p.light_path_m)
        i_avg_ppfd = i_surface_ppfd * (-np.expm1(-optical_depth)) / optical_depth

        f_t = temperature_response(water_temp, p)
        f_ph = ph_response(ph, p)
        mu_l = light_response(i_avg_ppfd, p)
        # Liebig minimum of TAN and DIC limitation + Azov & Goldman free-NH3 inhibition
        nutrient_lim = min(tan_mg_l / (tan_mg_l + p.k_n_mg_per_l), dic_mol_l / (dic_mol_l + p.k_dic_mol_per_l))
        nh3_inhib = p.ki_free_nh3_mg_per_l / (p.ki_free_nh3_mg_per_l + free_nh3_mg_l)
        mu_run_scale = cond.get("mu_max_run", p.mu_max_per_day) / max(p.mu_max_per_day, 1e-6)
        mu = mu_l * mu_run_scale * f_t * f_ph * nutrient_lim * nh3_inhib

        growth = mu * max(0.0, 1.0 - algae / 1200.0) * algae * dt_d
        decay = p.algae_decay_per_day * algae * dt_d

        # 6. Two-Fraction Heterotrophic COD Oxidation (no artificial DO floor!)
        het_temp = p.theta_het ** (water_temp - 25.0)
        do_monod = max(0.0, do) / (max(0.0, do) + p.k_do_half_mg_per_l)
        deg_fast = min(cod_fast, cond["fast_cod_rate"] * cod_fast * do_monod * het_temp * dt_d)
        deg_slow = min(cod_slow, cond["slow_cod_rate"] * cod_slow * do_monod * het_temp * dt_d)
        oxidised = deg_fast + deg_slow

        cod_fast = max(0.0, cod_fast - deg_fast)
        cod_slow = max(0.0, cod_slow - deg_slow + p.lysis_cod_fraction * p.cod_per_biomass * decay)
        cod_total = cod_fast + cod_slow

        detritus_loss = p.detritus_settling_per_day * detritus * dt_d
        detritus = max(25.0, detritus - detritus_loss + 0.12 * decay)
        algae = max(50.0, algae + growth - decay)

        # 7. Mechanistic Semi-Implicit Dissolved Oxygen Balance (Benson & Krause + ASCE kLa)
        do_sat = compute_do_saturation(water_temp, cond["pressure_hpa"], tds)
        q_ratio = np.clip(aeration / AERATOR_FLOW_LS, 0.0, 1.5)
        kla_h = (p.kla_surface_per_h + p.kla_aerator_per_h * q_ratio) * float(kla_temp_factor(water_temp))

        o2_photo = p.o2_per_biomass * growth
        o2_algae_resp = p.algae_respiration_per_day * algae * f_t * dt_d
        o2_cod_demand = p.o2_per_cod_oxidized * oxidised
        net_o2_bio = o2_photo - o2_algae_resp - o2_cod_demand

        do = (do + dt_h * kla_h * do_sat + net_o2_bio) / (1.0 + dt_h * kla_h)
        do = float(np.clip(do, 0.05, max(do_sat * 1.8, 16.0)))

        # 8. Nitrogen Balance (Ammonification, Assimilation, Nitrification, Stripping)
        n_upt_mg = p.n_per_biomass * growth
        n_amm_mg = p.n_per_cod * oxidised
        nitrif_mg = min(
            tan_mg_l * (1.0 - free_frac),
            p.k_nitrif_per_day * tan_mg_l * (1.0 - free_frac) * do_monod * het_temp * dt_d,
        )
        strip_rate_d = p.nh3_strip_per_day * (0.15 + 0.85 * q_ratio) * float(kla_temp_factor(water_temp))
        stripped_mg = min(tan_mg_l * free_frac, strip_rate_d * tan_mg_l * free_frac * dt_d)

        tan_mg_l = max(0.5, tan_mg_l + n_amm_mg - n_upt_mg - nitrif_mg - stripped_mg)
        tan_mol_l = (tan_mg_l / 1000.0) / MW_N

        # 9. Carbonate + Conservative Charge Balance -> True Mechanistic pH via solve_ph
        # Nitrification consumes 1 eq strong charge per mol N (H+ + NO3-); VFA oxidation restores alkalinity
        z_net = z_net - (nitrif_mg / 1000.0) / MW_N + (p.vfa_alk_per_cod_eq * deg_fast) / 1000.0

        pk1_p, pk2_p, _, pkh_0, _, _ = carbonate_constants(water_temp, ion)
        a0_co2, _, _ = carbonate_fractions(ph, pk1_p, pk2_p)
        co2_sat_mol_l = (10.0 ** (-pkh_0)) * (cond["co2_ppm"] * 1e-6)
        kla_co2_h = p.co2_kla_ratio * kla_h

        c_fixed_mol_l = (p.c_per_biomass * growth / 1000.0) / MW_C
        c_resp_mol_l = ((0.35 * oxidised / 32.0) + (0.28 * o2_algae_resp / 32.0)) / 1000.0

        # Semi-implicit CO2 gas-transfer step on DIC
        dic_num = dic_mol_l + dt_h * kla_co2_h * co2_sat_mol_l + c_resp_mol_l - c_fixed_mol_l
        dic_den = 1.0 + dt_h * kla_co2_h * a0_co2
        dic_mol_l = float(np.clip(dic_num / max(dic_den, 1.0), 5e-4, 0.08))

        ph = solve_ph(dic_mol_l, z_net, tan_mol_l, water_temp, ion)

        out["cod"][i] = cod_total
        out["cod_fast"][i] = cod_fast
        out["cod_slow"][i] = cod_slow
        out["algae"][i] = algae
        out["detritus"][i] = detritus
        out["tan"][i] = tan_mg_l
        out["ph"][i] = ph
        out["do"][i] = do
        out["tds"][i] = tds
        out["water_temp"][i] = water_temp

    return out


def ornstein_uhlenbeck(rng: np.random.Generator, n: int, dt_s: float, tau_s: float, sd: float) -> np.ndarray:
    """Stationary OU noise with standard deviation sd and time constant tau_s."""
    phi = np.exp(-dt_s / tau_s)
    innovations = rng.normal(0.0, sd * np.sqrt(1.0 - phi**2), n)
    series = np.zeros(n)
    series[0] = rng.normal(0.0, sd)
    for i in range(1, n):
        series[i] = phi * series[i - 1] + innovations[i]
    return series


def sensor_noise(rng: np.random.Generator, n: int, dt_s: float, name: str, p: GeneratorParams) -> np.ndarray:
    """Two-scale stationary OU sensor noise + mean-reverting transient spikes."""
    fast_15, slow_step_15 = SENSOR_NOISE[name]
    fast = ornstein_uhlenbeck(rng, n, dt_s, p.fast_tau_min * 60.0, fast_15 / FAST_NOISE_15MIN_FACTOR)
    tau_s = p.slow_tau_h * 3600.0
    slow_sd = slow_step_15 / np.sqrt(1.0 - np.exp(-2.0 * STEPS_PER_15MIN_SECONDS / tau_s))
    slow = ornstein_uhlenbeck(rng, n, dt_s, tau_s, slow_sd)
    noise = fast + slow
    if name in JUMP_SD:
        probability = p.jump_rate_per_day * dt_s / 86400.0
        raw_impulses = (rng.random(n) < probability) * np.clip(rng.standard_t(4, n), -4.5, 4.5) * JUMP_SD[name]
        spike_phi = np.exp(-dt_s / (p.spike_decay_min * 60.0))
        spikes = np.zeros(n)
        for i in range(1, n):
            spikes[i] = spike_phi * spikes[i - 1] + raw_impulses[i]
        noise = noise + spikes
    return noise


def apply_sensors(rng, n: int, dt_s: float, true_state: dict, env: dict, cond: dict, p: GeneratorParams) -> dict:
    """Turns true bioprocess state into multi-channel sensor telemetry with unified OD680 scaling."""
    scale = (p.cod_noise_intercept + p.cod_noise_slope * true_state["cod"]) / 18.8
    fast = ornstein_uhlenbeck(rng, n, dt_s, p.fast_tau_min * 60.0, 9.5 / FAST_NOISE_15MIN_FACTOR)
    tau_s = p.slow_tau_h * 3600.0
    slow_sd = 8.5 / np.sqrt(1.0 - np.exp(-2.0 * STEPS_PER_15MIN_SECONDS / tau_s))
    slow = ornstein_uhlenbeck(rng, n, dt_s, tau_s, slow_sd)
    cod = np.clip(true_state["cod"] + scale * (fast + slow), p.cod_floor_mg_per_l, 5500.0)

    bod = np.clip(p.bod_cod_ratio * cod + ornstein_uhlenbeck(rng, n, dt_s, 3600.0, 3.5), 20.0, 2800.0)
    toc = np.clip(p.toc_cod_ratio * cod + ornstein_uhlenbeck(rng, n, dt_s, 3600.0, 2.2), 12.0, 1850.0)

    solids = true_state["algae"] + true_state["detritus"]
    tss = np.clip(cond["tss_gain"] * solids + sensor_noise(rng, n, dt_s, "tss", p), 85.0, 3950.0)
    ph = np.clip(true_state["ph"] + cond["ph_offset"] + sensor_noise(rng, n, dt_s, "ph", p), 5.5, 10.45)
    do = np.clip(cond["do_gain"] * true_state["do"] + sensor_noise(rng, n, dt_s, "do", p), 0.15, 16.0)
    tds = np.clip(true_state["tds"] + sensor_noise(rng, n, dt_s, "tds", p), 1600.0, 9300.0)
    nh3 = np.clip(true_state["tan"] + sensor_noise(rng, n, dt_s, "nh3", p), 1.5, 210.0)
    water_temp = np.clip(true_state["water_temp"] + sensor_noise(rng, n, dt_s, "water_temp", p), 22.0, 38.0)

    turbidity = np.clip(0.665 * tss + ornstein_uhlenbeck(rng, n, dt_s, 3600.0, 28.0), 45.0, 1650.0)
    # Unified optical density scale: active algae + minor detritus absorbance
    od680 = np.clip(
        (true_state["algae"] + 0.15 * true_state["detritus"]) / p.biomass_mg_per_od680
        + ornstein_uhlenbeck(rng, n, dt_s, 7200.0, 0.04),
        0.22,
        2.40,
    )
    biomass_gl = np.clip(od680 * 1.266, 0.28, 3.10)
    bga_cells = np.clip(od680 * 3.85e5 + ornstein_uhlenbeck(rng, n, dt_s, 3600.0, 2.5e4), 4.5e4, 1.4e6)
    free_frac = free_nh3_fraction(water_temp, ph)
    nh3_udara = np.clip(0.49 * nh3 * (0.35 + 0.65 * free_frac) + ornstein_uhlenbeck(rng, n, dt_s, 3600.0, 3.2), 1.2, 95.0)
    co2_udara = np.clip(cond["co2_ppm"] + ornstein_uhlenbeck(rng, n, dt_s, 7200.0, 65.0), 240.0, 2900.0)
    # Sensor column 2_o2(ppm) reported as uncalibrated channel ~272k ADC/ppm for schema parity only
    o2_udara = np.clip(272000.0 + ornstein_uhlenbeck(rng, n, dt_s, 7200.0, 8500.0), 255000.0, 420000.0)
    pressure_hpa = np.clip(cond["pressure_hpa"] + ornstein_uhlenbeck(rng, n, dt_s, 14400.0, 0.65), 1002.5, 1011.5)

    return {
        "cod": cod,
        "bod": bod,
        "toc": toc,
        "tss": tss,
        "ph": ph,
        "do": do,
        "tds": tds,
        "nh3": nh3,
        "water_temp": water_temp,
        "turbidity": turbidity,
        "bga_cells": bga_cells,
        "od680": od680,
        "biomass_gl": biomass_gl,
        "nh3_udara": nh3_udara,
        "co2_udara": co2_udara,
        "o2_udara": o2_udara,
        "pressure_hpa": pressure_hpa,
    }


def make_environment(rng, n: int, dt_s: float, cond: dict, sched: dict, p: GeneratorParams) -> dict:
    """Indoor air, humidity, and optical irradiance sensors calibrated to K1-K16."""
    t_days = (np.arange(n) * dt_s) / 86400.0
    diurnal = 0.55 * np.sin(2.0 * np.pi * (t_days - 0.25))
    air = cond["air_temp"] + diurnal + ornstein_uhlenbeck(rng, n, dt_s, 43200.0, 0.75)
    humidity = 49.05 - 1.15 * (air - p.air_temp_c[0]) + ornstein_uhlenbeck(rng, n, dt_s, 43200.0, 2.4)
    irradiance = cond["irradiance_base"] + 1.85 * (sched["pwm"] / 100.0)
    return {
        "air_temp": air,
        "air_temp_sensor": np.clip(air + sensor_noise(rng, n, dt_s, "air_temp", p), 29.2, 38.8),
        "humidity": np.clip(humidity + sensor_noise(rng, n, dt_s, "humidity", p), 28.0, 88.0),
        "solar_radiation": np.clip(irradiance + sensor_noise(rng, n, dt_s, "solar_radiation", p), 0.0, 9.8),
    }


def generate_bioprocess_run(
    num_days: int = 15,
    sampling_interval_seconds: int = 60,
    is_treatment: bool = True,
    seed: int = 42,
    params: Optional[GeneratorParams] = None,
    lamp_max_lux: Optional[float] = None,
    include_extended_columns: bool = False,
) -> pd.DataFrame:
    """Simulates one 2.5 L cultivation run and returns telemetry DataFrame."""
    if sampling_interval_seconds > 3600:
        raise ValueError("sampling_interval_seconds must be 3600 or less")
    p = params or GeneratorParams()
    rng = np.random.default_rng(seed)
    dt_s = float(sampling_interval_seconds)
    n = int(num_days * 86400 / dt_s)

    start_time = datetime(2026, 3, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(seconds=i * dt_s) for i in range(n)]
    hours = (np.arange(n) * dt_s / 3600.0) % 24.0

    cond = sample_conditions(rng, p, lamp_max_lux)
    sched = make_schedule(rng, n, hours, dt_s, is_treatment)
    env = make_environment(rng, n, dt_s, cond, sched, p)
    true_state = simulate_process(n, dt_s, cond, sched, env, p)
    readings = apply_sensors(rng, n, dt_s, true_state, env, cond, p)

    base_dict = {
        TIMESTAMP_COL: [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps],
        "cod": readings["cod"],
        "bod": readings["bod"],
        "toc": readings["toc"],
        "tss": readings["tss"],
        "ph": readings["ph"],
        "do": readings["do"],
        "tds": readings["tds"],
        "nh3": readings["nh3"],
        "water_temp": readings["water_temp"],
        "air_temp": env["air_temp_sensor"],
        "humidity": env["humidity"],
        "solar_radiation": env["solar_radiation"],
        "aeration_level": sched["aeration"],
        "nahco3_dose": sched["nahco3"],
        "algae_dose": sched["algae"],
        "molasses_dose": sched["molasses"],
        "light_pwm": sched["pwm"],
    }

    if include_extended_columns:
        lamp_lux = (sched["pwm"] / 100.0) * cond["lamp_max_lux"]
        base_dict.update({
            "1_turbidity(NTU)": readings["turbidity"],
            "1_bga(cells/mL)": readings["bga_cells"],
            "1_illuminance(Lux)": env["solar_radiation"] * 119.5,
            "2_light(Lux)": lamp_lux * 0.72,
            "2_o2(ppm)": readings["o2_udara"],
            "2_pressure(hPa)": readings["pressure_hpa"],
            "2_altitude(m)": np.full(n, 51.0),
            "2_nh3_udara(ppm)": readings["nh3_udara"],
            "2_co2(ppm)": readings["co2_udara"],
            "Lampu (lux)": lamp_lux,
            "od680_estimasi(AU)": readings["od680"],
            "biomassa_estimasi(g/L)": readings["biomass_gl"],
            "berat_kering_estimasi(g)": readings["biomass_gl"] * (p.reactor_volume_l * 0.04),
            "berat_basah_estimasi(g)": readings["biomass_gl"] * (p.reactor_volume_l * 0.044),
        })

    return pd.DataFrame(base_dict)


def create_mock_dataset_if_absent(target_dir: str, overwrite: bool = False):
    """Creates synthetic CSV runs in target_dir if absent."""
    os.makedirs(target_dir, exist_ok=True)
    runs = [("no_treatment", False, 100)] + [(f"kultivasi_{i}", True, i) for i in range(1, 9)]
    for name, treated, seed in runs:
        path = os.path.join(target_dir, f"{name}.csv")
        if os.path.exists(path) and not overwrite:
            continue
        frame = generate_bioprocess_run(is_treatment=treated, seed=seed)
        frame.to_csv(path, index=False)


if __name__ == "__main__":
    print("Testing unified mechanistic generator v3 (V=2.5 L, solve_ph, Benson-Krause)...")
    df_test = generate_bioprocess_run(num_days=15, sampling_interval_seconds=3600, seed=1000)
    print(f"Generated {len(df_test):,} hourly steps.")
    print(df_test[["cod", "bod", "tss", "do", "ph", "nh3"]].describe().loc[["min", "50%", "mean", "max"]])
