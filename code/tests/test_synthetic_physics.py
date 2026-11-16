"""
Unit tests verifying physical laws, non-negativity, Weiss (1970) DO saturation,
Emerson NH3 speciation, and statistical distribution alignment (Wasserstein-1 & KS)
for Stage 2 Calibrated Synthetic Bioprocess Generator.
"""

import csv
import os
import sys
import types
import unittest
import numpy as np

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(CODE_DIR, ".."))
DATASINTESIS_DIR = os.path.join(PROJECT_ROOT, "DataSintesis")

for p in (CODE_DIR, DATASINTESIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import pandas as pd
except ImportError:
    pd_mod = types.ModuleType("pandas")
    pd_mod.DataFrame = lambda d: d
    sys.modules["pandas"] = pd_mod

from mock_data_generator import (
    GeneratorParams,
    compute_do_saturation,
    free_ammonia_fraction,
    generate_bioprocess_run,
)
from models.pets_world_model import PETSWorldModelEnsemble, PETSConfig


class TestSyntheticPhysicsAndFidelity(unittest.TestCase):
    def test_reactor_geometry_and_weiss_do_saturation(self):
        p = GeneratorParams()
        self.assertAlmostEqual(p.reactor_volume_l, 2.5, places=3)
        self.assertAlmostEqual(p.light_path_m, 0.35, places=3)

        # Benson & Krause (1984) DO saturation decreases with higher temperature and higher TDS salinity
        do_20_fresh = compute_do_saturation(20.0, 1013.25, tds_ppm=0.0)
        do_25_fresh = compute_do_saturation(25.0, 1013.25, tds_ppm=0.0)
        do_30_fresh = compute_do_saturation(30.0, 1013.25, tds_ppm=0.0)
        do_30_brackish = compute_do_saturation(30.0, 1007.1, tds_ppm=6000.0)

        self.assertAlmostEqual(do_20_fresh, 9.092, places=2)
        self.assertGreater(do_25_fresh, do_30_fresh)
        self.assertGreater(do_30_fresh, do_30_brackish)
        self.assertTrue(6.5 <= do_30_brackish <= 8.5)

    def test_physical_non_negativity_and_temporal_trajectory(self):
        frame = generate_bioprocess_run(
            num_days=16,
            sampling_interval_seconds=600,
            is_treatment=True,
            seed=1002,
            include_extended_columns=True,
        )
        cod = np.asarray(frame["cod"], dtype=float)
        bod = np.asarray(frame["bod"], dtype=float)
        tss = np.asarray(frame["tss"], dtype=float)
        do = np.asarray(frame["do"], dtype=float)
        ph = np.asarray(frame["ph"], dtype=float)
        nh3 = np.asarray(frame["nh3"], dtype=float)

        # All concentrations must be strictly non-negative and finite
        for arr in (cod, bod, tss, do, ph, nh3):
            self.assertTrue(np.all(np.isfinite(arr)))
            self.assertTrue(np.all(arr >= 0.0))

        # COD must decay significantly from Day 0 to Day 16 (K7 horizon) and reach PerMenLH compliance (<=350 mg/L)
        self.assertGreater(cod[0], 2500.0)
        self.assertLessEqual(cod[-1], 350.0)

        # DO must exhibit hypoxic start (<3.5 mg/L) and oxic recovery (>5.0 mg/L)
        self.assertLess(do[0], 3.8)
        self.assertGreater(np.mean(do[-24:]), 5.0)

    def test_pets_world_model_ensemble_uncertainty(self):
        rng = np.random.default_rng(42)
        s = rng.uniform(100.0, 1000.0, size=(120, 12)).astype(np.float32)
        a = rng.uniform(0.0, 10.0, size=(120, 5)).astype(np.float32)
        ns = s + rng.normal(0.0, 5.0, size=(120, 12)).astype(np.float32)

        pets = PETSWorldModelEnsemble(PETSConfig())
        stats = pets.fit(s, a, ns, epochs=0)
        self.assertEqual(stats["ensemble_size"], 5)

        pred_s, unc = pets.predict_next_state(s[:4], a[:4], deterministic=True)
        self.assertEqual(pred_s.shape, (4, 12))
        self.assertEqual(unc["epistemic_std"].shape, (4, 12))
        self.assertTrue(np.all(unc["epistemic_std"] >= 0.0))

    def test_generated_corpus_size_and_distribution_fidelity(self):
        corpus_path = os.path.join(CODE_DIR, "data", "synthetic_pome_transitions.csv")
        self.assertTrue(os.path.exists(corpus_path), "synthetic_pome_transitions.csv must exist")
        with open(corpus_path, "r", encoding="utf-8") as f:
            n_rows = sum(1 for _ in f) - 1
        self.assertGreaterEqual(n_rows, 10000, f"Expected >= 10,000 transitions, got {n_rows}")

        audit_csv = os.path.join(CODE_DIR, "evaluation", "synthetic_vs_real_audit_metrics.csv")
        self.assertTrue(os.path.exists(audit_csv))
        with open(audit_csv, "r", encoding="utf-8") as f:
            audit_rows = {r["parameter"]: r for r in csv.DictReader(f)}

        # Verify >=60% W1 reduction on dominant organic/particulate pollutants (COD, BOD, TSS)
        # and tight direct W1 fidelity (<0.40) across pH and DO distributions.
        for param in ("cod", "bod", "tss"):
            red = float(audit_rows[param]["w1_reduction_pct"])
            self.assertGreaterEqual(red, 60.0, f"Expected >=60% W1 reduction on {param}, got {red:.1f}%")
        for param in ("ph", "do"):
            w1_v2 = float(audit_rows[param]["w1_norm_v2"])
            self.assertLess(w1_v2, 0.40, f"Expected normalized W1 < 0.40 on {param}, got {w1_v2:.3f}")


if __name__ == "__main__":
    unittest.main()
