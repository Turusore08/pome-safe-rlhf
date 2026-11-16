"""
Master Orchestrator for Publication Visualizations (Individual & Combined Modes).
Executes and generates high-resolution publication figures (300 DPI PNG & vector PDF):
- Mode 'individual': Generates EACH graph as an independent, standalone image file
- Mode 'combined': Generates full multi-panel composite figures
- Figure 1: 10 individual graphs (RMSE Dosing & R² Effluent across 2h–24h)
- Figure 2: 11 individual graphs (Confusion Matrix + 10 Scatters)
- Figure 3: 8 individual graphs (Stage 1 & 2 Bar Charts with deltas)
- Figure 4: 9 individual graphs (K1–K8 Performance & MCDM Evaluations)

Usage:
  # Generate all graphs as individual separate files:
  python generate_all_figures.py --mode individual

  # Generate individual graphs for Figure 1 only:
  python generate_all_figures.py --fig 1 --mode individual

  # Generate individual graphs for Figure 4 only:
  python generate_all_figures.py --fig 4 --mode individual
"""

import os
import sys
import argparse
import shutil

# Add code root to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(CURRENT_DIR)
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from visualization.plot_fig1_temporal_resolution import plot_all_individual_graphs
from visualization.plot_fig2_hcr_scatter_matrices import plot_all_individual_fig2
from visualization.plot_fig3_hcr_vs_direct import plot_all_individual_fig3
from visualization.plot_fig4_cultivation_mcdm import plot_all_individual_fig4
from visualization.plot_rl_experiments import (
    plot_fig5_algorithm_benchmark,
    plot_fig6_temporal_horizons,
    plot_fig7_global_pareto_frontier,
    plot_fig8_hcr_comparison_and_ablation,
)


def main():
    parser = argparse.ArgumentParser(description="Generate POME Bioprocess Visualizations (Individual Separate Graphs)")
    parser.add_argument(
        "--fig",
        type=str,
        default="all",
        choices=["all", "1", "2", "3", "4", "5", "6", "7", "8", "rl"],
        help="Specify figure category (1, 2, 3, 4, 5, 6, 7, rl, or all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(CODE_ROOT, "evaluation", "figures"),
        help="Base directory to save generated individual figures",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display interactive matplotlib GUI window for each generated figure"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 75)
    print("🎨 GENERATING INDIVIDUAL STANDALONE FIGURES (300 DPI PNG & VECTOR PDF)")
    print(f"Target Base Directory: {args.output_dir}")
    print("=" * 75)

    if args.fig == "all":
        figs_to_run = [1, 2, 3, 4, 5, 6, 7, 8]
    elif args.fig == "rl":
        figs_to_run = [5, 6, 7, 8]
    else:
        figs_to_run = [int(args.fig)]

    for f_num in figs_to_run:
        if f_num == 1:
            out_f1 = os.path.join(args.output_dir, "fig1_individual")
            plot_all_individual_graphs(output_dir=out_f1, show_each=args.show)
        elif f_num == 2:
            out_f2 = os.path.join(args.output_dir, "fig2_individual")
            plot_all_individual_fig2(output_dir=out_f2, show_each=args.show)
        elif f_num == 3:
            out_f3 = os.path.join(args.output_dir, "fig3_individual")
            plot_all_individual_fig3(output_dir=out_f3, show_each=args.show)
        elif f_num == 4:
            out_f4 = os.path.join(args.output_dir, "fig4_individual")
            plot_all_individual_fig4(output_dir=out_f4, show_each=args.show)
        elif f_num == 5:
            out_rl = os.path.join(args.output_dir, "rl_figures")
            plot_fig5_algorithm_benchmark(output_dir=out_rl, show=args.show)
        elif f_num == 6:
            out_rl = os.path.join(args.output_dir, "rl_figures")
            plot_fig6_temporal_horizons(output_dir=out_rl, show=args.show)
        elif f_num == 7:
            out_rl = os.path.join(args.output_dir, "rl_figures")
            plot_fig7_global_pareto_frontier(output_dir=out_rl, show=args.show)
        elif f_num == 8:
            out_rl = os.path.join(args.output_dir, "rl_figures")
            try:
                plot_fig8_hcr_comparison_and_ablation(output_dir=out_rl, show=args.show)
            except RuntimeError as e:
                if args.fig in ("all", "rl"):
                    print(f"   [Fig 8 Strictly Skipped] {e}")
                else:
                    raise

    # Auto-sync to web/public/figures if directory exists
    web_figures_dir = os.path.join(os.path.dirname(CODE_ROOT), "web", "public", "figures")
    if os.path.exists(web_figures_dir):
        out_rl = os.path.join(args.output_dir, "rl_figures")
        if os.path.exists(out_rl):
            rl_web_dir = os.path.join(web_figures_dir, "rl_figures")
            os.makedirs(rl_web_dir, exist_ok=True)
            for fname in os.listdir(out_rl):
                shutil.copy2(os.path.join(out_rl, fname), os.path.join(rl_web_dir, fname))
            print(f"   [Web Sync] Berhasil sinkronisasi figur RL ke: {rl_web_dir}")

    print("\n" + "=" * 75)
    print("🎉 ALL INDIVIDUAL STANDALONE FIGURES GENERATED SUCCESSFULLY!")
    print(f"Setiap grafik kini tersimpan sebagai 1 gambar terpisah di:")
    print(f"   {args.output_dir}")
    print("=" * 75)


if __name__ == "__main__":
    main()
