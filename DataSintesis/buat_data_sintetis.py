"""
Membuat 14 run data sintetis terkalibrasi (DataSintesis v2) tanpa ketergantungan pada PyTorch,
atau dengan integrasi 35 kolom kompatibel `GABUNGAN_sensor_jam.csv`.

Hasil di data_sintetis/:
- menit/sintetis_XX.csv : keluaran generator per menit (atau interval sampling yang dipilih)
- jam/sintetis_XX.csv   : hasil agregasi per jam (rata-rata, dosis dijumlahkan)

Jalankan:
    python3 buat_data_sintetis.py
"""

import csv
import os
import sys
import types
from typing import Dict, List

import numpy as np

from config import CONTINUOUS_VARIABLES, DOSING_ACTIONS, TIMESTAMP_COL

# Fallback ringan apabila pandas tidak tersedia di environment sandbox
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd_mod = types.ModuleType("pandas")
    pd_mod.DataFrame = lambda d: d
    sys.modules["pandas"] = pd_mod

from mock_data_generator import generate_bioprocess_run

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_sintetis")
NUM_RUNS = 14
FIRST_SEED = 1000
LAMPS = [1050.0, 1800.0]


def to_hourly_dict(frame_dict: Dict[str, List], steps_per_hour: int = 60) -> Dict[str, List]:
    """Agregasi kamus data per menit menjadi data per jam (mean untuk sensor, sum untuk dosis)."""
    n_total = len(frame_dict[TIMESTAMP_COL])
    n_hours = n_total // steps_per_hour
    hourly: Dict[str, List] = {
        TIMESTAMP_COL: [frame_dict[TIMESTAMP_COL][h * steps_per_hour] for h in range(n_hours)]
    }
    for col in CONTINUOUS_VARIABLES:
        arr = np.asarray(frame_dict[col][: n_hours * steps_per_hour], dtype=float).reshape(n_hours, steps_per_hour)
        hourly[col] = np.mean(arr, axis=1).tolist()
    for col in DOSING_ACTIONS:
        arr = np.asarray(frame_dict[col][: n_hours * steps_per_hour], dtype=float).reshape(n_hours, steps_per_hour)
        hourly[col] = np.sum(arr, axis=1).tolist()
    return hourly


def save_dict_to_csv(data: Dict[str, List], path: str) -> None:
    cols = list(data.keys())
    n_rows = len(data[cols[0]])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for i in range(n_rows):
            writer.writerow([data[c][i] for c in cols])


def main(sampling_interval_seconds: int = 60):
    os.makedirs(f"{OUTPUT_DIR}/menit", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/jam", exist_ok=True)
    steps_per_hour = max(1, 3600 // sampling_interval_seconds)

    for i in range(NUM_RUNS):
        frame = generate_bioprocess_run(
            seed=FIRST_SEED + i,
            lamp_max_lux=LAMPS[i % 2],
            sampling_interval_seconds=sampling_interval_seconds,
        )
        if PANDAS_AVAILABLE and hasattr(frame, "to_csv"):
            idx = pd.to_datetime(frame[TIMESTAMP_COL])
            df_indexed = frame.set_index(idx)
            continuous = df_indexed[CONTINUOUS_VARIABLES].resample("1h").mean()
            doses = df_indexed[DOSING_ACTIONS].resample("1h").sum()
            hourly = pd.concat([continuous, doses], axis=1).reset_index()
            hourly["run_id"] = f"sintetis_{i}"
            hourly["is_treatment"] = 1
            frame.to_csv(f"{OUTPUT_DIR}/menit/sintetis_{i:02d}.csv", index=False)
            hourly.to_csv(f"{OUTPUT_DIR}/jam/sintetis_{i:02d}.csv", index=False)
        else:
            hourly = to_hourly_dict(frame, steps_per_hour=steps_per_hour)
            n_hr = len(hourly[TIMESTAMP_COL])
            hourly["run_id"] = [f"sintetis_{i}"] * n_hr
            hourly["is_treatment"] = [1] * n_hr
            save_dict_to_csv(frame, f"{OUTPUT_DIR}/menit/sintetis_{i:02d}.csv")
            save_dict_to_csv(hourly, f"{OUTPUT_DIR}/jam/sintetis_{i:02d}.csv")
        print(f"tersimpan {i + 1}/{NUM_RUNS}", flush=True)


if __name__ == "__main__":
    main()
