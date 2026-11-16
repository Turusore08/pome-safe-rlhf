"""
Global configuration, physical constants, sensor schemas, and baseline targets
for the Safe Offline-to-Online RLHF POME Control System.
"""

from typing import List, Dict

# ==============================================================================
# 1. Physical Bioreactor Constants & Decision Horizon
# ==============================================================================
REACTOR_VOLUME_LITERS: float = 2.5
REACTOR_TOTAL_CAPACITY_LITERS: float = 3.0  # 3.0 L glass vessel containing 2.5 L liquid + 0.5 L headspace
MICROALGAE_SPECIES: str = "Chlorella vulgaris"
DECISION_HORIZON_HOURS: float = 6.0   # Operational decision window
SAMPLING_INTERVAL_SECONDS: int = 5    # Raw sensor sampling interval
MAX_EPISODE_STEPS: int = 64           # 64 steps * 6h = 16 days (matches K7 max run)
COMPLIANCE_HOLD_STEPS: int = 4        # Must hold compliance for 4 steps (24h)

# ==============================================================================
# 2. 16-Variable IoT Sensor Schema (penjelasan.md §2, Mirda et al. 2026)
# ==============================================================================
# The rig monitors 16 water-quality and environmental variables sampled by ESP32
# and collected on a Raspberry Pi 4 gateway.
SENSOR_COLUMNS: List[str] = [
    "cod",                       # 1. Chemical Oxygen Demand (mg/L)
    "bod",                       # 2. Biochemical Oxygen Demand (mg/L)
    "total_organic_carbon",      # 3. Total Organic Carbon (TOC, mg/L)
    "total_suspended_solids",    # 4. Total Suspended Solids (TSS, mg/L)
    "turbidity",                 # 5. Turbidity (NTU)
    "ph",                        # 6. Water pH (-)
    "dissolved_oxygen",          # 7. Dissolved Oxygen (DO, mg/L)
    "total_dissolved_solids",    # 8. Total Dissolved Solids (TDS, mg/L)
    "dissolved_ammonia",         # 9. Dissolved Ammonia (NH3-N, mg/L)
    "water_temperature",         # 10. Water Temperature (°C; CSV '1_temp_air(C)', 'air' = water)
    "light_intensity",           # 11. Photobioreactor Light Intensity (lux)
    "ambient_temperature",       # 12. Ambient Room Temperature (°C)
    "relative_humidity",         # 13. Ambient Relative Humidity (%)
    "atmospheric_pressure",      # 14. Atmospheric Pressure (hPa)
    "air_oxygen",                # 15. Ambient Air Oxygen (sensor ppm/ADC; uncalibrated in K1-16)
    "air_ammonia",               # 16. Ambient Air Ammonia (ppm)
]

SENSOR_VARIABLES: List[str] = SENSOR_COLUMNS

# Sensor provenance notes (empirical dataset audit K1-K16)
SENSOR_METADATA: Dict[str, str] = {
    "water_temperature": "Derived from CSV '1_temp_air(C)' ('air' = Indonesian for liquid water; 23.9-36.0 C).",
    "air_oxygen": "Derived from CSV '2_o2(ppm)' uncalibrated MQ/gas-sensor channel; excluded from mechanistic ODEs.",
    "biodata_ratios": "Empirical K1-K16 lab calibration: BOD/COD ~ 0.50, TOC/COD ~ 0.316, Biomass(g/L) ~ 1.266 * OD680.",
}

# Typical raw POME initial concentrations before treatment
RAW_POME_INIT: Dict[str, float] = {
    "cod": 4279.0,               # mg/L (HCR paper baseline K8 initial)
    "bod": 1820.0,               # mg/L
    "total_organic_carbon": 1450.0, # mg/L
    "total_suspended_solids": 1650.0, # mg/L
    "turbidity": 850.0,          # NTU
    "ph": 6.8,                   # pH
    "dissolved_oxygen": 1.2,     # mg/L
    "total_dissolved_solids": 2100.0, # mg/L
    "dissolved_ammonia": 185.0,  # mg/L
    "water_temperature": 32.35,  # °C (empirical rig average: 32.35 ± 2.66 °C)
    "light_intensity": 1800.0,   # lux (K1-K16 indoor PBR range)
    "ambient_temperature": 29.5, # °C
    "relative_humidity": 75.0,   # %
    "atmospheric_pressure": 1012.0, # hPa
    "air_oxygen": 20.9,          # %
    "air_ammonia": 2.5,          # ppm
}

# ==============================================================================
# 3. Two-Tier Regulatory & Biological Safety Standards
# ==============================================================================
# Tier 1: Effluent Discharge Compliance (PerMenLH No. 5 Tahun 2014 Lampiran III - Industri Minyak Sawit)
REGULATORY_LIMITS: Dict[str, float] = {
    "cod": 350.0,                # Max 350 mg/L
    "bod": 100.0,                # Max 100 mg/L (PerMenLH No. 5/2014 statutory BOD5 limit)
    "total_suspended_solids": 250.0, # Max 250 mg/L
    "dissolved_ammonia": 50.0,   # Max 50 mg/L (Total Nitrogen / TN)
    "ph_min": 6.0,               # Min pH 6.0
    "ph_max": 9.0,               # Max pH 9.0
    "max_debit_m3_per_ton_cpo": 2.5,  # Statutory maximum wastewater volume
}
EFFLUENT_COMPLIANCE_LIMITS: Dict[str, float] = REGULATORY_LIMITS

# Tier 2: Online Biological Safety Envelope (absorbing failure boundary c_t = 1)
# Wider than discharge compliance because C. vulgaris photosynthesizes cleanly up to pH ~10.3 on K1-K16
BIOLOGICAL_SAFETY_LIMITS: Dict[str, Dict[str, float]] = {
    "ph": {"min": 5.5, "max": 10.5},                  # Microalgal physiological survival envelope
    "water_temperature": {"min": 18.0, "max": 38.0},  # Culture thermal bounds (CTMI T_max=40 C)
    "dissolved_oxygen": {"min": 0.5, "max": 20.0},    # Anoxic lower limit (photosynthetic DO hits ~15.8 mg/L)
    "cod_rebound_ratio": {"max": 1.25},               # COD surge > 25% indicates bacterial lysis / spill
}
OPERATIONAL_CONSTRAINTS: Dict[str, Dict[str, float]] = BIOLOGICAL_SAFETY_LIMITS

# ==============================================================================
# 4. 5-Dimensional Action Space & Literature/Statutory Economics
# ==============================================================================
# Canonical order [0..4] shared across OfflineDataset, POMEReactorEnv, HurdlePolicy, & Plotting:
ACTION_DIM: int = 5
ACTION_COLUMNS: List[str] = [
    "aeration",                  # [0] T1: Aeration duration at Q_REF=0.04 L/s (hours/window, 0 to 6)
    "sodium_bicarbonate",        # [1] T4: NaHCO3 buffer dosing (grams/window, 0 to 5)
    "microalgae_inoculum",       # [2] T5: C. vulgaris inoculum addition (mL/window, 0 to 100)
    "molasses",                  # [3] T3: Molasses carbon supplementation (mL/window, 0 to 20)
    "light_intensity",           # [4] T2: Illumination level (lux, 0 to 5000)
]

ACTION_BOUNDS: Dict[str, tuple] = {
    "aeration": (0.0, 6.0),
    "sodium_bicarbonate": (0.0, 5.0),
    "microalgae_inoculum": (0.0, 100.0),
    "molasses": (0.0, 20.0),
    "light_intensity": (0.0, 5000.0),
}

# Rig telemetry conversion factors (K1-K16 pump logs -> policy action space)
AERATOR_FLOW_LS: float = 0.04
NAHCO3_STOCK_G_PER_ML: float = 5.0 / 600.0   # 8.33 g/L stock (~0.1 M NaHCO3): 600 mL = 5.0 g
MOLASSES_RAW_TO_WINDOW_SCALE: float = 20.0 / 400.0

# Statutory & empirical market pricing (Permen ESDM No. 7/2024 & IDXCarbon 2024-2025)
ELECTRICITY_TARIFF_IDR_PER_KWH: float = 1444.70   # PLN R-1/B-1 tariff (Permen ESDM 7/2024)
AERATOR_POWER_KW: float = 0.025                   # 25 W diaphragm air pump
LED_ARRAY_MAX_POWER_KW: float = 0.018             # 18 W full-spectrum LED at 5000 lux

BIOMASS_PRICE_PER_GRAM_IDR: float = 35.0          # IDR 35,000 / kg dry C. vulgaris feedstock
CO2_CREDIT_PER_GRAM_IDR: float = 0.07             # IDR 70,000 / tCO2e (IDXCarbon 2024-2025 market rate)
OPERATIONAL_OVERHEAD_STEP_IDR: float = 15.0       # ESP32/RPi4 telemetry + peristaltic wear per 6h step

# Unit costs kappa_k per 1 unit of action vector a_t during a 6h decision step (IDR)
ACTION_UNIT_COSTS_IDR: Dict[str, float] = {
    "aeration": round(AERATOR_POWER_KW * ELECTRICITY_TARIFF_IDR_PER_KWH, 2),                      # 36.12 IDR/h
    "sodium_bicarbonate": 25.0,                                                                  # 25.0 IDR/g tech-grade
    "microalgae_inoculum": 15.0,                                                                 # 15.0 IDR/mL seed culture
    "molasses": 35.0,                                                                            # 35.0 IDR/mL molasses stock
    "light_intensity": round((LED_ARRAY_MAX_POWER_KW * DECISION_HORIZON_HOURS * ELECTRICITY_TARIFF_IDR_PER_KWH) / 5000.0, 5),  # 0.03121 IDR/(lux*6h)
}
REFERENCE_STEP_COST_IDR: float = 308.5  # Normalization scale matching typical 6h step expenditure (~6170 / 20)

# Literature experimental benchmark outcomes (Mirda et al., 2026)
K8_BASELINE_COST_IDR: float = 6170.0
K8_COMPLIANCE_DAY: int = 5
K8_COD_REMOVAL_PCT: float = 95.70
K8_FINAL_BIOMASS_GL: float = 2.49

K5_HUMAN_COST_IDR: float = 9872.0
K5_COMPLIANCE_DAY: int = 8
K5_COD_REMOVAL_PCT: float = 98.56

K7_DIRECT_COST_IDR: float = 19748.0
K7_COMPLIANCE_DAY: int = 16
K7_COD_FINAL: float = 340.0

# ==============================================================================
# 5. Real Data Integration: ITS-POME Kultivasi K1–K16 Mapping
# ==============================================================================
import os

def resolve_real_data_dir() -> str:
    """
    Dynamically resolves REAL_DATA_DIR, automatically handling nested extraction folders
    such as 'Data Kultivasi Mikroalga K1-16/Data Kultivasi Mikroalga K1-16/'.
    """
    default_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env_dir = os.getenv("REAL_DATA_DIR", None)

    candidates = []
    if env_dir:
        c_env = os.path.abspath(env_dir)
        candidates.extend([
            c_env,
            os.path.join(c_env, "Data Kultivasi Mikroalga K1-16"),
        ])

    candidates.extend([
        os.path.join(default_root, "Data Kultivasi Mikroalga K1-16"),
        os.path.join(default_root, "Data Kultivasi Mikroalga K1-16", "Data Kultivasi Mikroalga K1-16"),
        os.path.join(default_root, "..", "Data Kultivasi Mikroalga K1-16"),
        os.path.join(default_root, "..", "Data Kultivasi Mikroalga K1-16", "Data Kultivasi Mikroalga K1-16"),
        default_root,
    ])

    for cand in candidates:
        if os.path.exists(os.path.join(cand, "gabungan", "GABUNGAN_sensor_jam.csv")):
            return cand

    # Scan one level deep inside default_root/Data Kultivasi Mikroalga K1-16
    parent_folder = os.path.join(default_root, "Data Kultivasi Mikroalga K1-16")
    if os.path.exists(parent_folder) and os.path.isdir(parent_folder):
        try:
            for item in os.listdir(parent_folder):
                sub_path = os.path.join(parent_folder, item)
                if os.path.isdir(sub_path) and os.path.exists(os.path.join(sub_path, "gabungan", "GABUNGAN_sensor_jam.csv")):
                    return sub_path
        except Exception:
            pass

    return os.path.join(default_root, "Data Kultivasi Mikroalga K1-16")


REAL_DATA_DIR: str = resolve_real_data_dir()
REAL_GABUNGAN_SENSOR_JAM: str = os.path.join(REAL_DATA_DIR, "gabungan", "GABUNGAN_sensor_jam.csv")
REAL_GABUNGAN_SENSOR_MENIT: str = os.path.join(REAL_DATA_DIR, "gabungan", "GABUNGAN_sensor_menit.csv")
REAL_FOTO_DIR: str = os.path.join(REAL_DATA_DIR, "foto")
REAL_PER_KULTIVASI_DIR: str = os.path.join(REAL_DATA_DIR, "per_kultivasi")


def validate_real_dataset_paths() -> None:
    """
    Validates that the real cultivation dataset files are present.
    Dynamically refreshes paths and raises FileNotFoundError with clear instructions if missing.
    """
    global REAL_DATA_DIR, REAL_GABUNGAN_SENSOR_JAM, REAL_GABUNGAN_SENSOR_MENIT, REAL_FOTO_DIR, REAL_PER_KULTIVASI_DIR
    REAL_DATA_DIR = resolve_real_data_dir()
    REAL_GABUNGAN_SENSOR_JAM = os.path.join(REAL_DATA_DIR, "gabungan", "GABUNGAN_sensor_jam.csv")
    REAL_GABUNGAN_SENSOR_MENIT = os.path.join(REAL_DATA_DIR, "gabungan", "GABUNGAN_sensor_menit.csv")
    REAL_FOTO_DIR = os.path.join(REAL_DATA_DIR, "foto")
    REAL_PER_KULTIVASI_DIR = os.path.join(REAL_DATA_DIR, "per_kultivasi")

    if not os.path.exists(REAL_GABUNGAN_SENSOR_JAM):
        raise FileNotFoundError(
            f"\n[ERROR] Dataset riil telemetri sensor tidak ditemukan:\n"
            f"Path dicari : {REAL_GABUNGAN_SENSOR_JAM}\n"
            f"Penyebab    : Folder ter-ekstrak ganda (nested) atau belum berisi file gabungan/GABUNGAN_sensor_jam.csv.\n"
            f"Solusi      : Buka folder 'Data Kultivasi Mikroalga K1-16'. Jika di dalamnya ada subfolder lagi bernama\n"
            f"              'Data Kultivasi Mikroalga K1-16', pindahkan isinya (folder gabungan, foto, per_kultivasi)\n"
            f"              keluar ke folder utama."
        )
    if not os.path.exists(REAL_FOTO_DIR):
        print(f"[Warning] Direktori foto bioproses tidak ditemukan di: {REAL_FOTO_DIR}")



REAL_SENSOR_MAP: Dict[str, str] = {
    "1_cod(mg/L)": "cod",
    "1_bod(mg/L)": "bod",
    "1_toc(mg/L)": "total_organic_carbon",
    "1_tss(mg/L)": "total_suspended_solids",
    "1_turbidity(NTU)": "turbidity",
    "1_ph": "ph",
    "1_do(mg/L)": "dissolved_oxygen",
    "1_tds(ppm)": "total_dissolved_solids",
    "1_nh3(mg/L)": "dissolved_ammonia",
    "1_temp_air(C)": "water_temperature",
    "2_light(Lux)": "light_intensity",
    "2_temp(C)": "ambient_temperature",
    "2_humidity(%)": "relative_humidity",
    "2_pressure(hPa)": "atmospheric_pressure",
    "2_o2(ppm)": "air_oxygen",
    "2_nh3_udara(ppm)": "air_ammonia",
}

REAL_ACTION_MAP: Dict[str, str] = {
    "Aerator_RU (L/s)": "aeration",
    "Pump1_NaHCO3 (mL)": "sodium_bicarbonate",
    "Pump3_Microalgae (mL)": "microalgae_inoculum",
    "Pump2_Molase (mL)": "molasses",
    "Lampu (lux)": "light_intensity",
}

# ==============================================================================
# 6. NVIDIA GeForce RTX 5090 Acceleration Profile & Architecture Constants
# ==============================================================================
RTX_5090_SPECS: Dict[str, any] = {
    "gpu_name": "NVIDIA GeForce RTX 5090",
    "architecture": "Blackwell (GB202)",
    "sm_count": 170,
    "cuda_cores": 21760,
    "tensor_cores": 680,
    "tensor_core_generation": "5th Gen (FP8/BF16/TF32)",
    "vram_gb": 32.0,
    "memory_type": "GDDR7",
    "memory_bus_width_bits": 512,
    "memory_bandwidth_gb_s": 1792.0,
    "fp32_tflops": 104.0,
    "tensor_bf16_tflops": 1600.0,
    "tdp_watts": 600.0,
    "typical_ai_inference_watts": 280.0,
    "recommended_batch_size": 512,
    "dataloader_pin_memory": True,
    "dataloader_num_workers": 4,
    "enable_torch_compile": True,
    "enable_mixed_precision": True,
    "mixed_precision_dtype": "bfloat16",
}

# ==============================================================================
# 7. Hybrid ODE (Mech / UDE / UDE + Ammonification) & LOCO Validation Constants
# ==============================================================================
# Standardized reactor geometry & dosing stock parameters (Mirda et al., 2026)
HYBRID_ODE_STATES: List[str] = [
    "od680_estimasi(AU)",
    "1_cod(mg/L)",
    "1_nh3(mg/L)",
    "1_do(mg/L)",
]
HYBRID_ODE_SHORT: List[str] = ["X", "COD", "NH3", "DO"]

Q_REF: float = AERATOR_FLOW_LS                     # 0.04 L/s maximum aerator flow rate in K1-16
V_ML: float = REACTOR_VOLUME_LITERS * 1000.0       # 2500.0 mL working liquid volume (inside 3.0 L glass vessel)
C_MOL: float = 5000.0                              # mg/L working molasses stock COD (empirically calibrated to K1-16 pump pulses)
X_INO: float = 1.2                                 # AU OD680 of starter inoculum suspension (empirically calibrated)
DEPTH_CM: float = 35.0                             # cm liquid depth in 3.0 L column PBR
HYBRID_ODE_SPAN: float = 1.5                       # +/-1.5 decade log10 bound around nominal initial parameters

# Parameter dictionary: name -> (nominal_init, is_learnable)
# Literature constraints: CTMI temperatures frozen (Bernard & Remond, 2012);
# K_iA initialized at 20.4 mg NH3-N/L (Azov & Goldman, 1982); remaining params calibrated via LOCO on K1-K16.
HYBRID_ODE_PARAMS: Dict[str, tuple] = {
    # Microalgae: Chlorella vulgaris
    "mu_max": (0.05, True),     # /hour (Bernard & Remond, 2012 order ~0.04-0.08 /h)
    "I_opt": (5.0, True),       # umol photons m^-2 s^-1 (column-averaged PPFD inside turbid POME)
    "k_par": (0.3, True),       # broadband PAR attenuation factor relative to OD680 (1 cm cuvette)
    "T_min": (10.0, False),     # °C, CTMI cardinal minimum (Bernard & Remond, 2012)
    "T_opt": (30.0, False),     # °C, CTMI cardinal optimum (tropical C. vulgaris)
    "T_max": (40.0, False),     # °C, CTMI cardinal maximum
    "K_N": (5.0, True),         # mg TAN/L half-saturation constant (Casagli et al., 2021)
    "K_iA": (20.4, True),       # mg free-NH3-N/L inhibition constant (Azov & Goldman, 1982)
    "k_d": (0.005, True),       # /hour basal mortality/decay rate
    "Y_N": (10.0, True),        # mg TAN consumed per AU OD680 biomass grown
    "Y_O": (5.0, True),         # mg O2 produced per AU OD680 photoautotrophic growth
    "k_r": (0.2, True),         # mg O2/L/hour per AU basal respiration
    # Lumped POME bacteria (ASM1-style aerobic heterotrophic degradation)
    "k_S": (20.0, True),        # mg COD/L/hour max heterotrophic COD degradation rate
    "K_S": (300.0, True),       # mg COD/L half-saturation constant
    "K_O": (0.5, True),         # mg DO/L oxygen half-saturation constant (Henze et al., 2000)
    "Y_OS": (0.25, True),       # mg O2 consumed per mg COD oxidized (ASM1 yield-consistent)
    "k_amm": (0.02, True),      # mg N ammonified per mg COD degraded (when amm=True)
    # Physical mass transfer & ammonia stripping
    "kLa0": (0.05, True),       # /hour passive surface oxygen transfer at 20 C
    "kLa_a": (0.5, True),       # /hour forced aeration O2 mass transfer at Q = Q_REF, 20 C
    "k_strip": (0.01, True),    # /hour free-NH3 volatilization coefficient
}

# Leave-One-Cultivation-Out (LOCO) splits & sliding window configuration
LOCO_TRAIN_CULTIVATIONS: List[int] = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16]
OOD_CULTIVATIONS: List[int] = [3, 15]
LOCO_WINDOW_HOURS: int = 72
LOCO_STRIDE_HOURS: int = 24
LAB_MEASUREMENT_INTERVAL_HOURS: int = 24
LOCO_MODES: List[tuple] = [("mech", False), ("ude", False), ("ude", True)]


