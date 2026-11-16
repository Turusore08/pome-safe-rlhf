"""
Single-Instruction Master Entrypoint: Run Full RLHF Experiments Suite.
Allows the user to launch and benchmark all experiment configurations with one command:
    python run_experiments.py

Supports:
    python run_experiments.py --quick       # Fast smoke-test mode
    python run_experiments.py --exp EXP-01  # Run single experiment
"""

import os
import sys
import argparse

# Reconfigure stdout/stderr to UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure code root is in python path
code_root = os.path.dirname(os.path.abspath(__file__))
if code_root not in sys.path:
    sys.path.insert(0, code_root)

from experiments.parallel_runner import run_all_23_experiments
from training.train_hybrid_ode_loco import run_hybrid_ode_loco


def main():
    parser = argparse.ArgumentParser(
        description="Master Runner: Safe Offline-to-Online RLHF Experiments & Hybrid ODE LOCO Validation"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast mode with reduced epochs (ideal for quick validation)",
    )
    parser.add_argument(
        "--exp",
        type=str,
        default=None,
        help="Run a specific experiment by ID (e.g., --exp EXP-01)",
    )
    parser.add_argument(
        "--skip-loco",
        action="store_true",
        help="Skip Hybrid ODE 14-fold LOCO validation stage",
    )
    parser.add_argument(
        "--loco-only",
        action="store_true",
        help="Run only the Hybrid ODE 14-fold LOCO validation (mech / UDE / UDE+amm)",
    )
    args = parser.parse_args()

    if not args.skip_loco and args.exp is None:
        print("\n" + "=" * 75)
        print(" 🧬 STAGE 1: HYBRID ODE / UDE 14-FOLD LOCO VALIDATION")
        print("=" * 75)
        run_hybrid_ode_loco(quick_test=args.quick)
        if args.loco_only:
            return

    print("\n" + "=" * 75)
    print(" 🚀 LAUNCHING SAFE RLHF EXPERIMENTS SUITE ON NVIDIA HARDWARE")
    print("=" * 75)

    results = run_all_23_experiments(quick_mode=args.quick, target_exp_id=args.exp)

    print("\n" + "=" * 75)
    print(" ✅ ALL EXPERIMENTS COMPLETED!")
    print(" 📊 Leaderboard Report : evaluation/experiments_23_report.md")
    print(" 📈 CSV Metric Data    : evaluation/experiments_23_results.csv")
    print(" 📑 JSON Artifact      : evaluation/experiments_23_leaderboard.json")
    if not args.skip_loco and args.exp is None:
        print(" 🧪 Hybrid ODE LOCO    : evaluation/hybrid_ode_loco.csv")
        print(" 📉 Hybrid ODE Metrics : evaluation/hybrid_ode_metrics.csv")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
