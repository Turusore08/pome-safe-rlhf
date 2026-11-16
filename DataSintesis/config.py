"""
Configuration bridge for DataSintesis & PETS World Model.
Explicitly loads authoritative physical constants and two-tier limits from `code/config.py`
via `importlib.util` (avoiding name collision between the two `config.py` files)
while preserving the 18-column schema expected by `buat_data_sintetis.py` and `pets_world_model.py`.
"""

import importlib.util
import os
from dataclasses import dataclass, field
from typing import List

_CODE_CFG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code", "config.py"))
_spec = importlib.util.spec_from_file_location("_pome_code_config", _CODE_CFG_PATH)
_code_cfg = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(_code_cfg)

# Re-export every public symbol from authoritative code/config.py so sys.path order never breaks imports
globals().update({k: v for k, v in vars(_code_cfg).items() if not k.startswith("__")})

# ==============================================================================
# FEATURE DEFINITIONS
# ==============================================================================

# Sensor state variables: Water quality continuous measurements
WATER_QUALITY_FEATURES: List[str] = [
    "cod",          # Chemical Oxygen Demand (mg/L)
    "bod",          # Biochemical Oxygen Demand (mg/L)
    "toc",          # Total Organic Carbon (mg/L)
    "tss",          # Total Suspended Solids (mg/L)
    "ph",           # pH level
    "do",           # Dissolved Oxygen (mg/L)
    "tds",          # Total Dissolved Solids (ppm)
    "nh3",          # Ammonia / Ammonium (mg/L)
    "water_temp"    # Water Temperature (°C)
]

# Environmental context continuous variables
ENVIRONMENT_FEATURES: List[str] = [
    "air_temp",         # Ambient Air Temperature (°C)
    "humidity",         # Relative Humidity (%)
    "solar_radiation"   # Solar Irradiance / PAR (W/m2 or umol/m2/s)
]

# Complete state vector s_t
STATE_FEATURES: List[str] = WATER_QUALITY_FEATURES + ENVIRONMENT_FEATURES

# Action variables a_t
# - Continuous actions: 'aeration_level', 'light_pwm' (airflow rate and artificial lighting dimmer)
# - Discrete/Dosing actions: amount added per 15-minute window
ACTION_FEATURES: List[str] = [
    "aeration_level",   # Continuous: 0.0 - 0.04 L/s (Mean aggregated)
    "nahco3_dose",      # Dosing: Buffer volume in mL (Sum aggregated)
    "algae_dose",       # Inoculum addition in mL (Sum aggregated)
    "molasses_dose",    # Carbon substrate addition in mL (Sum aggregated)
    "light_pwm"         # Continuous: LED Dimmer 0.0 - 100.0% (Mean aggregated)
]

# Dosing action variables that MUST be summed during resampling
DOSING_ACTIONS: List[str] = [
    "nahco3_dose",
    "algae_dose",
    "molasses_dose"
]

# Continuous state & action variables that MUST be averaged during resampling
CONTINUOUS_VARIABLES: List[str] = STATE_FEATURES + ["aeration_level", "light_pwm"]

# Timestamp column identifier
TIMESTAMP_COL: str = "timestamp"


# ==============================================================================
# HYPERPARAMETERS & CONFIGURATION CLASS
# ==============================================================================

def _default_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@dataclass
class Config:
    # Data pipeline settings
    resample_interval: str = "15min"      # 15-minute aggregation window
    chunk_size: int = 100_000             # Rows per chunk for reading large CSVs
    max_missing_interpolate_limit: int = 4 # Interpolate up to 4 missing steps (1 hour)

    # State & Action dimensions
    state_dim: int = len(STATE_FEATURES)
    action_dim: int = len(ACTION_FEATURES)

    # PETS World Model Architecture
    num_ensemble_models: int = 5
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 256])
    activation: str = "silu"              # SiLU (Swish) smooth non-linearity
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    epochs: int = 80
    early_stopping_patience: int = 15

    # Variance clamping bounds (Log variance stability in PETS)
    min_logvar: float = -10.0
    max_logvar: float = 0.5

    # Device
    device: str = field(default_factory=_default_device)
