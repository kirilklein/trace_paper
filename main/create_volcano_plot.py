"""Script to create Matplotlib and Plotly volcano plots for semaglutide data."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import pandas as pd

from trace.io import (
    PrevalenceStats,
    load_atc_dictionary,
    load_prevalence_statistics,
)
from trace.plotting.volcano import (
    prepare_volcano_data,
    volcano_plot_per_method,
    volcano_overlay_methods,
)
from trace.plotting.volcano_plotly import (
    build_plotly_volcano,
    build_plotly_overlay_methods,
    save_plotly_figure,
)
from trace.statistics import compute_rd_pvalues, combine_random_effects_HKSJ


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
ESTIMATES_PATH = Path("data/semaglutide/combined_estimatest.txt")
STATS_PATH = Path("data/semaglutide/combined_stats.txt")
FIGURES_DIR = Path("figures")
MATPLOTLIB_ALPHA = 0.05
PLOTLY_ALPHA = 0.05
METHODS_WITH_ARMS = ("IPW", "TMLE")
POOLING_METHOD = "random_effects_dl"


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def load_semaglutide_estimates(path: Path) -> pd.DataFrame:
    """Load the semaglutide arm-level estimates CSV and drop unnamed columns."""

    df = pd.read_csv(path)
    unnamed_cols = [col for col in df.columns if col.startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    return df


def filter_methods_with_arm_cis(
    df: pd.DataFrame, methods: Iterable[str]
) -> pd.DataFrame:
    """Restrict the dataframe to methods providing arm-level confidence intervals."""

    return df[df["method"].isin(list(methods))].copy()


def ensure_required_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """Print a diagnostic summary of missing values for required columns."""

    print("\nChecking for required columns...")
    for col in required:
        n_missing = df[col].isna().sum()
        print(f"  {col}: {n_missing} missing values")


def summarise_per_run_effects(
    df_per_run: pd.DataFrame,
    *,
    effect_col: str = "RD",
    effect_alias: str | None = None,
) -> pd.DataFrame:
    """Aggregate per-run effect sizes and arm probabilities for hover metadata."""

    if df_per_run.empty:
        return pd.DataFrame()

    agg_kwargs = dict(
        per_run_n_runs=("run_id", "nunique"),
        per_run_effect1_mean=("effect_1", "mean"),
        per_run_effect1_std=("effect_1", "std"),
        per_run_effect0_mean=("effect_0", "mean"),
        per_run_effect0_std=("effect_0", "std"),
    )

    if effect_col in df_per_run.columns:
        prefix = (effect_alias or effect_col).lower()
        agg_kwargs.update(
            {
                f"per_run_{prefix}_mean": (effect_col, "mean"),
                f"per_run_{prefix}_median": (effect_col, "median"),
                f"per_run_{prefix}_std": (effect_col, "std"),
                f"per_run_{prefix}_min": (effect_col, "min"),
                f"per_run_{prefix}_max": (effect_col, "max"),
            }
        )

    grouped = (
        df_per_run.groupby(["method", "outcome"], dropna=False)
        .agg(**agg_kwargs)
        .reset_index()
    )
    return grouped


def rename_prevalence_columns(summary: pd.DataFrame) -> pd.DataFrame:
    """Make prevalence summary column names easier to work with."""

    rename_map = {
        "prevalence_overall_total": "prevalence_total",
        "prevalence_overall_treated": "prevalence_treated",
        "prevalence_overall_untreated": "prevalence_untreated",
        "population_total": "population_total",
        "population_treated": "population_treated",
        "population_untreated": "population_untreated",
        "outcome_events_total": "outcome_events_total",
        "outcome_events_treated": "outcome_events_treated",
        "outcome_events_untreated": "outcome_events_untreated",
        "prevalence_mean_total": "prevalence_mean_total",
        "prevalence_mean_treated": "prevalence_mean_treated",
        "prevalence_mean_untreated": "prevalence_mean_untreated",
        "prevalence_std_total": "prevalence_std_total",
        "prevalence_std_treated": "prevalence_std_treated",
        "prevalence_std_untreated": "prevalence_std_untreated",
        "prevalence_median_total": "prevalence_median_total",
        "prevalence_median_treated": "prevalence_median_treated",
        "prevalence_median_untreated": "prevalence_median_untreated",
        "run_count_total": "prevalence_run_count_total",
        "run_count_treated": "prevalence_run_count_treated",
        "run_count_untreated": "prevalence_run_count_untreated",
    }
    return summary.rename(
        columns={k: v for k, v in rename_map.items() if k in summary.columns}
    )


def augment_volcano_dataframe(
    volcano_df: pd.DataFrame,
    pooled_df: pd.DataFrame,
    prevalence_summary: pd.DataFrame,
    per_run_summary: pd.DataFrame,
    atc_mapping: Dict[str, str],
    *,
    effect_col: str = "RD",
) -> pd.DataFrame:
    """Merge pooled results with metadata for plotting and hover details."""

    # Avoid duplicating RD/p-values that already exist in volcano_df
    exclude_cols = {effect_col, "p_value"}
    pooled_meta = pooled_df.drop(
        columns=[c for c in exclude_cols if c in pooled_df.columns]
    )

    merged = volcano_df.merge(pooled_meta, on=["method", "outcome"], how="left")
    if not per_run_summary.empty:
        merged = merged.merge(per_run_summary, on=["method", "outcome"], how="left")
    if not prevalence_summary.empty:
        merged = merged.merge(prevalence_summary, on="outcome", how="left")

    merged["atc_description"] = merged["outcome"].map(atc_mapping)
    merged["outcome_label"] = merged["outcome"].where(
        merged["atc_description"].isna(),
        merged["outcome"] + " · " + merged["atc_description"],
    )
    return merged


def ensure_output_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Diagnostics helpers (retain original detailed output)
# -----------------------------------------------------------------------------
def print_dataset_overview(df: pd.DataFrame) -> None:
    print(f"Loaded {len(df)} rows")
    print(f"Methods: {df['method'].unique()}")
    print(f"Outcomes: {df['outcome'].nunique()} unique")
    print(f"Runs: {df['run_id'].nunique()} unique")


def print_pooled_diagnostics(df_pooled: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("DIAGNOSTIC: P-value Distribution Analysis")
    print("=" * 70)

    print("\nP-value statistics:")
    print(f"  Min p-value: {df_pooled['p_value'].min():.2e}")
    print(f"  Max p-value: {df_pooled['p_value'].max():.2e}")
    print(f"  Median p-value: {df_pooled['p_value'].median():.2e}")
    print(f"  Mean p-value: {df_pooled['p_value'].mean():.2e}")

    n = max(len(df_pooled), 1)
    n_at_floor = (df_pooled["p_value"] <= 1e-300).sum()
    n_below_1e100 = (df_pooled["p_value"] < 1e-100).sum()
    n_below_1e50 = (df_pooled["p_value"] < 1e-50).sum()
    n_below_1e20 = (df_pooled["p_value"] < 1e-20).sum()
    n_below_1e10 = (df_pooled["p_value"] < 1e-10).sum()

    print("\nExtreme p-values:")
    print(f"  <= 1e-300: {n_at_floor} ({100 * n_at_floor / n:.1f}%)")
    print(f"  < 1e-100: {n_below_1e100} ({100 * n_below_1e100 / n:.1f}%)")
    print(f"  < 1e-50: {n_below_1e50} ({100 * n_below_1e50 / n:.1f}%)")
    print(f"  < 1e-20: {n_below_1e20} ({100 * n_below_1e20 / n:.1f}%)")
    print(f"  < 1e-10: {n_below_1e10} ({100 * n_below_1e10 / n:.1f}%)")

    print("\nZ-statistic statistics:")
    print(f"  Min z: {df_pooled['z'].min():.2f}")
    print(f"  Max z: {df_pooled['z'].max():.2f}")
    print(f"  Median |z|: {df_pooled['z'].abs().median():.2f}")
    print(f"  Mean |z|: {df_pooled['z'].abs().mean():.2f}")

    n_extreme_z = (df_pooled["z"].abs() > 30).sum()
    n_very_extreme_z = (df_pooled["z"].abs() > 50).sum()
    print("\nExtreme z-statistics:")
    print(f"  |z| > 30: {n_extreme_z} ({100 * n_extreme_z / n:.1f}%)")
    print(f"  |z| > 50: {n_very_extreme_z} ({100 * n_very_extreme_z / n:.1f}%)")

    print("\nSE_RD statistics:")
    print(f"  Min SE_RD: {df_pooled['SE_RD'].min():.2e}")
    print(f"  Max SE_RD: {df_pooled['SE_RD'].max():.2e}")
    print(f"  Median SE_RD: {df_pooled['SE_RD'].median():.2e}")
    print(f"  Mean SE_RD: {df_pooled['SE_RD'].mean():.2e}")

    n_tiny_se = (df_pooled["SE_RD"] < 1e-6).sum()
    n_small_se = (df_pooled["SE_RD"] < 1e-4).sum()
    print("\nSmall SE_RD values:")
    print(f"  < 1e-6: {n_tiny_se} ({100 * n_tiny_se / n:.1f}%)")
    print(f"  < 1e-4: {n_small_se} ({100 * n_small_se / n:.1f}%)")

    if {"p1_hat", "p0_hat"}.issubset(df_pooled.columns):
        print("\nProbability estimates (p1_hat, p0_hat):")
        n_extreme_p1 = (
            (df_pooled["p1_hat"] < 0.001) | (df_pooled["p1_hat"] > 0.999)
        ).sum()
        n_extreme_p0 = (
            (df_pooled["p0_hat"] < 0.001) | (df_pooled["p0_hat"] > 0.999)
        ).sum()
        print(
            f"  p1_hat < 0.001 or > 0.999: {n_extreme_p1} ({100 * n_extreme_p1 / n:.1f}%)"
        )
        print(
            f"  p0_hat < 0.001 or > 0.999: {n_extreme_p0} ({100 * n_extreme_p0 / n:.1f}%)"
        )
        print(
            f"  p1_hat range: [{df_pooled['p1_hat'].min():.4f}, {df_pooled['p1_hat'].max():.4f}]"
        )
        print(
            f"  p0_hat range: [{df_pooled['p0_hat'].min():.4f}, {df_pooled['p0_hat'].max():.4f}]"
        )
    else:
        print(
            "\nProbability estimates (p1_hat, p0_hat): Not available for this pooling method"
        )

    if {"eta1_pooled_se", "eta0_pooled_se"}.issubset(df_pooled.columns):
        print("\nArm-level SE statistics (after pooling):")
        print(
            f"  eta1_pooled_se: min={df_pooled['eta1_pooled_se'].min():.2e}, "
            f"median={df_pooled['eta1_pooled_se'].median():.2e}, "
            f"max={df_pooled['eta1_pooled_se'].max():.2e}"
        )
        print(
            f"  eta0_pooled_se: min={df_pooled['eta0_pooled_se'].min():.2e}, "
            f"median={df_pooled['eta0_pooled_se'].median():.2e}, "
            f"max={df_pooled['eta0_pooled_se'].max():.2e}"
        )


def print_extreme_cases(df_pooled: pd.DataFrame, df_with_arms: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("DIAGNOSTIC: Detailed inspection of most extreme cases")
    print("=" * 70)

    print("\nTop 5 smallest p-values:")
    display_cols = ["method", "outcome", "RD", "SE_RD", "z", "p_value"]
    if {"p1_hat", "p0_hat"}.issubset(df_pooled.columns):
        display_cols.extend(["p1_hat", "p0_hat"])
    extreme_cases = df_pooled.nsmallest(5, "p_value")[display_cols]
    for _, row in extreme_cases.iterrows():
        print(f"\n  {row['method']} - {row['outcome'][:50]}")
        print(f"    RD={row['RD']:.4f}, SE_RD={row['SE_RD']:.2e}")
        print(f"    z={row['z']:.2f}, p={row['p_value']:.2e}")
        if "p1_hat" in row:
            print(f"    p1_hat={row['p1_hat']:.4f}, p0_hat={row['p0_hat']:.4f}")

        mask = (df_with_arms["method"] == row["method"]) & (
            df_with_arms["outcome"] == row["outcome"]
        )
        orig = df_with_arms[mask]
        if len(orig) > 0:
            print(f"    Original data from {len(orig)} runs:")
            print(
                f"      effect_1: mean={orig['effect_1'].mean():.4f}, "
                f"std={orig['effect_1'].std():.4f}"
            )
            print(
                f"      effect_0: mean={orig['effect_0'].mean():.4f}, "
                f"std={orig['effect_0'].std():.4f}"
            )
            ci_width_1 = (
                orig["effect_1_CI95_upper"] - orig["effect_1_CI95_lower"]
            ).mean()
            ci_width_0 = (
                orig["effect_0_CI95_upper"] - orig["effect_0_CI95_lower"]
            ).mean()
            print(
                f"      Mean CI width: effect_1={ci_width_1:.4f}, effect_0={ci_width_0:.4f}"
            )


def deep_dive_extreme_case(df_pooled: pd.DataFrame, df_with_arms: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("DIAGNOSTIC: Deep dive into most extreme case")
    print("=" * 70)

    if df_pooled.empty:
        return

    extreme_idx = df_pooled["p_value"].idxmin()
    extreme_row = df_pooled.loc[extreme_idx]

    print("\nMost extreme case:")
    print(f"  Method: {extreme_row['method']}")
    print(f"  Outcome: {extreme_row['outcome'][:80]}")
    print(f"  Final RD: {extreme_row['RD']:.6f}")
    print(f"  Final SE_RD: {extreme_row['SE_RD']:.6e}")
    print(f"  Final z: {extreme_row['z']:.2f}")
    print(f"  Final p-value: {extreme_row['p_value']:.6e}")

    mask = (df_with_arms["method"] == extreme_row["method"]) & (
        df_with_arms["outcome"] == extreme_row["outcome"]
    )
    orig_data = df_with_arms[mask].copy()

    print(f"\n  Original data ({len(orig_data)} runs):")
    from trace.statistics import inv_logit, logit, se_from_prob_ci_on_logit
    import numpy as np
    from scipy.stats import norm

    eta1_vals: List[float] = []
    se_eta1_vals: List[float] = []
    eta0_vals: List[float] = []
    se_eta0_vals: List[float] = []

    for i, (idx, row) in enumerate(orig_data.iterrows()):
        print(f"\n    Run {i + 1} (run_id={row['run_id']}):")
        print(
            f"      effect_1: {row['effect_1']:.6f} "
            f"[{row['effect_1_CI95_lower']:.6f}, {row['effect_1_CI95_upper']:.6f}]"
        )
        print(
            f"      effect_0: {row['effect_0']:.6f} "
            f"[{row['effect_0_CI95_lower']:.6f}, {row['effect_0_CI95_upper']:.6f}]"
        )

        ci_width_1 = row["effect_1_CI95_upper"] - row["effect_1_CI95_lower"]
        ci_width_0 = row["effect_0_CI95_upper"] - row["effect_0_CI95_lower"]
        print(f"      CI widths: effect_1={ci_width_1:.6f}, effect_0={ci_width_0:.6f}")

        eta1 = logit(row["effect_1"])
        se_eta1 = se_from_prob_ci_on_logit(
            row["effect_1_CI95_lower"], row["effect_1_CI95_upper"]
        )
        eta0 = logit(row["effect_0"])
        se_eta0 = se_from_prob_ci_on_logit(
            row["effect_0_CI95_lower"], row["effect_0_CI95_upper"]
        )
        eta1_vals.append(eta1)
        se_eta1_vals.append(se_eta1)
        eta0_vals.append(eta0)
        se_eta0_vals.append(se_eta0)

    print("\n  Step-by-step calculation:")

    eta1_arr = np.array(eta1_vals)
    se_eta1_arr = np.array(se_eta1_vals)
    eta0_arr = np.array(eta0_vals)
    se_eta0_arr = np.array(se_eta0_vals)

    finite_mask_1 = np.isfinite(eta1_arr) & np.isfinite(se_eta1_arr)
    finite_mask_0 = np.isfinite(eta0_arr) & np.isfinite(se_eta0_arr)

    w1 = 1.0 / (se_eta1_arr[finite_mask_1] ** 2)
    eta1_pooled = np.sum(w1 * eta1_arr[finite_mask_1]) / np.sum(w1)
    se_eta1_pooled = np.sqrt(1.0 / np.sum(w1))

    w0 = 1.0 / (se_eta0_arr[finite_mask_0] ** 2)
    eta0_pooled = np.sum(w0 * eta0_arr[finite_mask_0]) / np.sum(w0)
    se_eta0_pooled = np.sqrt(1.0 / np.sum(w0))

    print("\n    After inverse-variance pooling:")
    print(f"      eta1_pooled={eta1_pooled:.4f} (se={se_eta1_pooled:.4e})")
    print(f"      eta0_pooled={eta0_pooled:.4f} (se={se_eta0_pooled:.4e})")
    print(f"      Weights (1/SE²): arm1={w1}, arm0={w0}")
    print(f"      Sum of weights: arm1={np.sum(w1):.2e}, arm0={np.sum(w0):.2e}")

    p1_pooled = inv_logit(eta1_pooled)
    p0_pooled = inv_logit(eta0_pooled)
    rd_pooled = p1_pooled - p0_pooled

    print("\n    After inverse logit:")
    print(f"      p1_pooled={p1_pooled:.6f}")
    print(f"      p0_pooled={p0_pooled:.6f}")
    print(f"      RD_pooled={rd_pooled:.6f}")

    var_term1 = (p1_pooled * (1 - p1_pooled)) ** 2 * (se_eta1_pooled**2)
    var_term0 = (p0_pooled * (1 - p0_pooled)) ** 2 * (se_eta0_pooled**2)
    var_rd = var_term1 + var_term0
    se_rd = np.sqrt(var_rd)

    print("\n    Delta method variance calculation:")
    print(f"      dp1/deta1 = p1*(1-p1) = {p1_pooled * (1 - p1_pooled):.6f}")
    print(f"      dp0/deta0 = p0*(1-p0) = {p0_pooled * (1 - p0_pooled):.6f}")
    print(f"      Var(p1) = (dp/deta)² * Var(eta1) = {var_term1:.6e}")
    print(f"      Var(p0) = (dp/deta)² * Var(eta0) = {var_term0:.6e}")
    print(f"      Var(RD) = Var(p1) + Var(p0) = {var_rd:.6e}")
    print(f"      SE(RD) = {se_rd:.6e}")

    z_stat = rd_pooled / se_rd
    p_val = 2 * (1 - norm.cdf(np.abs(z_stat)))

    print("\n    Final inference:")
    print(f"      z = RD / SE_RD = {rd_pooled:.6f} / {se_rd:.6e} = {z_stat:.2f}")
    print(f"      p-value = 2 * (1 - Phi(|z|)) = {p_val:.6e}")
    print(f"      -log10(p) = {-np.log10(max(p_val, 1e-300)):.2f}")

    print("\n    Verification against df_pooled:")
    print(f"      Match RD: {np.isclose(rd_pooled, extreme_row['RD'])}")
    print(f"      Match SE_RD: {np.isclose(se_rd, extreme_row['SE_RD'])}")
    print(f"      Match z: {np.isclose(z_stat, extreme_row['z'])}")
    matches_p_value = (
        np.isclose(p_val, extreme_row["p_value"]) if p_val > 0 else "both ~0"
    )
    print(f"      Match p_value: {matches_p_value}")


# -----------------------------------------------------------------------------
# Main execution flow
# -----------------------------------------------------------------------------
def main() -> None:
    df_raw = load_semaglutide_estimates(ESTIMATES_PATH)
    print_dataset_overview(df_raw)

    df_with_arms = filter_methods_with_arm_cis(df_raw, METHODS_WITH_ARMS)
    print(f"\nFiltered to IPW and TMLE: {len(df_with_arms)} rows")

    required_cols = [
        "effect_1",
        "effect_0",
        "effect_1_CI95_lower",
        "effect_1_CI95_upper",
        "effect_0_CI95_lower",
        "effect_0_CI95_upper",
    ]
    ensure_required_columns(df_with_arms, required_cols)

    print(
        f"\nComputing pooled risk differences and p-values using '{POOLING_METHOD}' method..."
    )

    df_per_run = compute_rd_pvalues(
        df_with_arms,
        group_cols=None,
        pooling_method="inverse_variance_arms",
        verbose=False,
    )

    df_pooled = combine_random_effects_HKSJ(
        df_per_run, group_cols=("method", "outcome")
    )
    print(f"Computed {len(df_pooled)} method-outcome combinations")

    if "tau2" in df_pooled.columns:
        print("\nHeterogeneity statistics (tau²):")
        print(f"  Mean tau²: {df_pooled['tau2'].mean():.4e}")
        print(f"  Median tau²: {df_pooled['tau2'].median():.4e}")
        print(f"  Max tau²: {df_pooled['tau2'].max():.4e}")
        n_heterogeneous = (df_pooled["tau2"] > 0.01).sum()
        print(
            f"  Cases with substantial heterogeneity (tau² > 0.01): {n_heterogeneous}"
        )

    print_pooled_diagnostics(df_pooled)
    print_extreme_cases(df_pooled, df_with_arms)
    deep_dive_extreme_case(df_pooled, df_with_arms)

    print("\n" + "=" * 70)

    print("\nPreparing volcano plot data...")
    df_volcano = prepare_volcano_data(
        df_pooled,
        rd_col="RD",
        p_col="p_value",
        method_col="method",
        outcome_col="outcome",
        adjust="bh",
        adjust_per="by_method",
    )

    atc_mapping = load_atc_dictionary()
    prevalence_stats: PrevalenceStats = load_prevalence_statistics(STATS_PATH)
    prevalence_summary = rename_prevalence_columns(prevalence_stats.summary)
    per_run_summary = (
        summarise_per_run_effects(df_per_run)
        if not df_per_run.empty
        else pd.DataFrame()
    )
    df_volcano_enriched = augment_volcano_dataframe(
        df_volcano,
        df_pooled,
        prevalence_summary,
        per_run_summary,
        atc_mapping,
    )

    print("\nSummary statistics:")
    for method in df_volcano_enriched["method"].unique():
        d = df_volcano_enriched[df_volcano_enriched["method"] == method]
        n_sig = (d["q_value"] < MATPLOTLIB_ALPHA).sum()
        print(
            f"  {method}: {len(d)} outcomes, {n_sig} significant (q < {MATPLOTLIB_ALPHA})"
        )

    ensure_output_directory(FIGURES_DIR)

    outcome_label_map = (
        df_volcano_enriched.dropna(subset=["outcome_label"])
        .drop_duplicates("outcome")
        .set_index("outcome")["outcome_label"]
        .to_dict()
    )

    print("\nCreating TMLE vs IPW overlay plot...")
    # Visual overlay helps inspect method-specific discrepancies for matched outcomes.
    try:
        fig_overlay, _ = volcano_overlay_methods(
            df_volcano_enriched,
            methods=("TMLE", "IPW"),
            method_col="method",
            outcome_col="outcome",
            label_map=outcome_label_map,
            annotate_top_n=10,
            point_size=45,
        )
    except ValueError as err:
        print(f"Skipping TMLE vs IPW overlay plot: {err}")
    else:
        overlay_png = FIGURES_DIR / "volcano_plot_tmle_ipw_overlay.png"
        fig_overlay.savefig(overlay_png, dpi=300, bbox_inches="tight")
        print(f"Saved overlay plot to: {overlay_png}")

        overlay_pdf = FIGURES_DIR / "volcano_plot_tmle_ipw_overlay.pdf"
        fig_overlay.savefig(overlay_pdf, bbox_inches="tight")
        print(f"Saved overlay plot to: {overlay_pdf}")

        plt.close(fig_overlay)

    print("\nCreating TMLE vs IPW overlay plot (interactive)...")
    try:
        plotly_overlay = build_plotly_overlay_methods(
            df_volcano_enriched,
            methods=("TMLE", "IPW"),
            method_col="method",
            outcome_col="outcome",
            label_map=outcome_label_map,
            marker_size=9,
        )
    except ValueError as err:
        print(f"Skipping Plotly TMLE vs IPW overlay: {err}")
    else:
        overlay_html = FIGURES_DIR / "volcano_plot_tmle_ipw_overlay_interactive.html"
        overlay_png = FIGURES_DIR / "volcano_plot_tmle_ipw_overlay_interactive.png"
        save_plotly_figure(
            plotly_overlay,
            html_path=overlay_html,
            png_path=overlay_png,
            width=900,
            height=600,
            scale=2.0,
        )
        print(f"Saved interactive overlay to: {overlay_html}")
        print(f"Saved interactive overlay snapshot to: {overlay_png}")

    print("\nCreating volcano plot...")
    fig, axes = volcano_plot_per_method(
        df_volcano_enriched,
        alpha=MATPLOTLIB_ALPHA,
        method_col="method",
        outcome_col="outcome",
        max_labels_per_panel=10,
        figsize_per_panel=(7, 5),
        point_size=20,
        sig_color="#d62728",
        ns_color="#7f7f7f",
    )

    output_path_png = FIGURES_DIR / "volcano_plot.png"
    fig.savefig(output_path_png, dpi=300, bbox_inches="tight")
    print(f"\nSaved plot to: {output_path_png}")

    output_path_pdf = FIGURES_DIR / "volcano_plot.pdf"
    fig.savefig(output_path_pdf, bbox_inches="tight")
    print(f"Saved plot to: {output_path_pdf}")

    plt.show()

    print("\n" + "=" * 70)
    print("Creating truncated volcano plot (excluding clipped p-values)")
    print("=" * 70)

    df_volcano_truncated = df_volcano_enriched[
        df_volcano_enriched["neglog10p"] < 299
    ].copy()
    n_excluded = len(df_volcano_enriched) - len(df_volcano_truncated)
    pct_excluded = 100 * n_excluded / max(len(df_volcano_enriched), 1)
    print(
        f"\nExcluded {n_excluded} outcomes with clipped p-values ({pct_excluded:.1f}%)"
    )
    print(f"Remaining outcomes: {len(df_volcano_truncated)}")

    for method in df_volcano_truncated["method"].unique():
        d = df_volcano_truncated[df_volcano_truncated["method"] == method]
        n_sig = (d["q_value"] < MATPLOTLIB_ALPHA).sum()
        print(
            f"  {method}: {len(d)} outcomes, {n_sig} significant (q < {MATPLOTLIB_ALPHA})"
        )

    print("\nCreating truncated volcano plot...")
    fig_trunc, axes_trunc = volcano_plot_per_method(
        df_volcano_truncated,
        alpha=MATPLOTLIB_ALPHA,
        method_col="method",
        outcome_col="outcome",
        max_labels_per_panel=10,
        figsize_per_panel=(7, 5),
        point_size=20,
        sig_color="#d62728",
        ns_color="#7f7f7f",
    )

    output_path_trunc_png = FIGURES_DIR / "volcano_plot_truncated.png"
    fig_trunc.savefig(output_path_trunc_png, dpi=300, bbox_inches="tight")
    print(f"Saved truncated plot to: {output_path_trunc_png}")

    output_path_trunc_pdf = FIGURES_DIR / "volcano_plot_truncated.pdf"
    fig_trunc.savefig(output_path_trunc_pdf, bbox_inches="tight")
    print(f"Saved truncated plot to: {output_path_trunc_pdf}")

    plt.show()

    print("\nCreating interactive Plotly volcano plot...")
    plotly_fig = build_plotly_volcano(
        df_volcano_enriched,
        alpha=PLOTLY_ALPHA,
        method_col="method",
        outcome_col="outcome",
    )

    plotly_html = FIGURES_DIR / "volcano_plot_interactive.html"
    plotly_png = FIGURES_DIR / "volcano_plot_interactive.png"
    save_plotly_figure(plotly_fig, html_path=plotly_html, png_path=plotly_png)
    print(f"Saved interactive plot to: {plotly_html}")
    print(f"Saved interactive snapshot to: {plotly_png}")

    print("\nDone!")


if __name__ == "__main__":
    main()
