"""Script to create Matplotlib volcano plots using risk ratios on the x-axis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from main.create_volcano_plot import (
    augment_volcano_dataframe,
    ensure_output_directory,
    ensure_required_columns,
    filter_methods_with_arm_cis,
    load_semaglutide_estimates,
    print_dataset_overview,
    rename_prevalence_columns,
    summarise_per_run_effects,
)
from trace.io import (
    PrevalenceStats,
    load_atc_dictionary,
    load_prevalence_statistics,
)
from trace.plotting.volcano import prepare_volcano_data, volcano_plot_per_method
from trace.plotting.volcano_plotly import build_plotly_volcano, save_plotly_figure
from trace.statistics import (
    combine_rr_random_effects_HKSJ,
    compute_rr_from_arm_estimates,
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
ESTIMATES_PATH = Path("data/semaglutide/combined_estimatest.txt")
STATS_PATH = Path("data/semaglutide/combined_stats.txt")
FIGURES_DIR = Path("figures")
METHODS_WITH_ARMS = ("IPW", "TMLE")
MATPLOTLIB_ALPHA = 0.05


def summarise_rr_results(df_rr):
    if df_rr.empty:
        print("No risk ratio results to summarise.")
        return

    rr_min = df_rr["RR"].min()
    rr_max = df_rr["RR"].max()
    rr_median = df_rr["RR"].median()
    print("\nRisk ratio summary (pooled by method/outcome):")
    print(f"  RR range: [{rr_min:.3f}, {rr_max:.3f}] (median {rr_median:.3f})")

    if "log_RR" in df_rr.columns:
        print(
            f"  log(RR) mean ± sd: {df_rr['log_RR'].mean():.3f} ± "
            f"{df_rr['log_RR'].std():.3f}"
        )


def main() -> None:
    df_raw = load_semaglutide_estimates(ESTIMATES_PATH)
    print_dataset_overview(df_raw)

    df_with_arms = filter_methods_with_arm_cis(df_raw, METHODS_WITH_ARMS)
    print(f"\nFiltered to risk-ratio compatible methods: {len(df_with_arms)} rows")

    required_cols = [
        "effect_1",
        "effect_0",
        "effect_1_CI95_lower",
        "effect_1_CI95_upper",
        "effect_0_CI95_lower",
        "effect_0_CI95_upper",
    ]
    ensure_required_columns(df_with_arms, required_cols)

    print("\nComputing per-run risk ratios...")
    df_rr_per_run = compute_rr_from_arm_estimates(
        df_with_arms, group_cols=None, verbose=False
    )

    print("Pooling risk ratios with DL + HKSJ adjustment...")
    df_rr_pooled = combine_rr_random_effects_HKSJ(
        df_rr_per_run, group_cols=("method", "outcome")
    )
    print(f"Computed {len(df_rr_pooled)} method-outcome combinations")

    summarise_rr_results(df_rr_pooled)

    df_volcano_rr = prepare_volcano_data(
        df_rr_pooled,
        rd_col="RR",
        p_col="p_value",
        method_col="method",
        outcome_col="outcome",
        adjust="bh",
        adjust_per="by_method",
        effect_alias="RR",
    )

    atc_mapping = load_atc_dictionary()
    prevalence_stats: PrevalenceStats = load_prevalence_statistics(STATS_PATH)
    prevalence_summary = rename_prevalence_columns(prevalence_stats.summary)
    per_run_summary = summarise_per_run_effects(
        df_rr_per_run, effect_col="RR", effect_alias="RR"
    )

    df_volcano_enriched = augment_volcano_dataframe(
        df_volcano_rr,
        df_rr_pooled,
        prevalence_summary,
        per_run_summary,
        atc_mapping,
        effect_col="RR",
    )

    print("\nSummary statistics:")
    for method in df_volcano_enriched["method"].unique():
        d = df_volcano_enriched[df_volcano_enriched["method"] == method]
        n_sig = (d["q_value"] < MATPLOTLIB_ALPHA).sum()
        print(
            f"  {method}: {len(d)} outcomes, {n_sig} significant (q < {MATPLOTLIB_ALPHA})"
        )

    ensure_output_directory(FIGURES_DIR)

    print("\nCreating risk-ratio volcano plot...")
    fig, axes = volcano_plot_per_method(
        df_volcano_enriched,
        alpha=MATPLOTLIB_ALPHA,
        method_col="method",
        outcome_col="outcome",
        effect_col="RR",
        effect_label="Risk ratio (RR)",
        null_value=1.0,
        xscale="log",
        max_labels_per_panel=10,
        figsize_per_panel=(7, 5),
        point_size=20,
        sig_color="#1f77b4",
        ns_color="#7f7f7f",
    )

    output_path_png = FIGURES_DIR / "volcano_plot_rr.png"
    fig.savefig(output_path_png, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {output_path_png}")

    output_path_pdf = FIGURES_DIR / "volcano_plot_rr.pdf"
    fig.savefig(output_path_pdf, bbox_inches="tight")
    print(f"Saved plot to: {output_path_pdf}")

    plt.show()

    print("\nCreating interactive risk-ratio volcano plot...")
    plotly_fig = build_plotly_volcano(
        df_volcano_enriched,
        alpha=MATPLOTLIB_ALPHA,
        method_col="method",
        outcome_col="outcome",
        effect_col="RR",
        effect_label="Risk ratio (RR)",
        null_value=1.0,
        xscale="log",
    )

    plotly_html = FIGURES_DIR / "volcano_plot_rr_interactive.html"
    plotly_png = FIGURES_DIR / "volcano_plot_rr_interactive.png"
    save_plotly_figure(plotly_fig, html_path=plotly_html, png_path=plotly_png)
    print(f"Saved interactive plot to: {plotly_html}")
    print(f"Saved interactive snapshot to: {plotly_png}")

    print("\nDone!")


if __name__ == "__main__":
    main()
