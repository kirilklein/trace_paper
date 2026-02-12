"""Create simulation analysis plots for treatment effect estimators.

This script:
- pools arm-level estimates across reshuffle runs to obtain pooled estimates per simulation run
- computes bias, coverage, and other performance metrics
- aggregates these metrics across simulation runs
- produces plots comparing estimator performance across experiments and outcomes
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit as inv_logit
from scipy.special import logit

from main.helpers import ensure_output_directory
from trace.statistics import compute_logit_se_from_ci, pool_arm_logits


# -----------------------------------------------------------------------------
# Experiment Parsing
# -----------------------------------------------------------------------------


def parse_experiment_params(experiment_str: str) -> dict[str, float]:
    """Extract ce, cy, y, i parameters from experiment string.

    Example: 'ce0p2_cy0p2_y0p2_i0p2' -> {'ce': 0.2, 'cy': 0.2, 'y': 0.2, 'i': 0.2}
    """
    pattern = r"ce(\d+p\d+)_cy(\d+p\d+)_y(\d+p\d+)_i(\d+p\d+)"
    match = re.search(pattern, experiment_str)
    if not match:
        return {"ce": np.nan, "cy": np.nan, "y": np.nan, "i": np.nan}

    def parse_num(s: str) -> float:
        """Convert '0p2' to 0.2"""
        return float(s.replace("p", "."))

    return {
        "ce": parse_num(match.group(1)),
        "cy": parse_num(match.group(2)),
        "y": parse_num(match.group(3)),
        "i": parse_num(match.group(4)),
    }


# -----------------------------------------------------------------------------
# First Aggregation: Pool Across Reshuffle Runs
# -----------------------------------------------------------------------------


def add_logit_arm_metrics_simulation(df: pd.DataFrame) -> pd.DataFrame:
    """Augment the DataFrame with logit-transformed point estimates and SEs
    for both treatment (1) and control (0) arms, adapted for simulation data.
    """
    out = df.copy()

    # Arm 1 (Treatment)
    out["eta1"] = logit(out["effect_1"])
    out["effect_1_logit_CI95_lower"] = logit(out["effect_1_CI95_lower"])
    out["effect_1_logit_CI95_upper"] = logit(out["effect_1_CI95_upper"])
    out["se_eta1"] = compute_logit_se_from_ci(
        out["effect_1_CI95_lower"], out["effect_1_CI95_upper"]
    )

    # Arm 0 (Control)
    out["eta0"] = logit(out["effect_0"])
    out["effect_0_logit_CI95_lower"] = logit(out["effect_0_CI95_lower"])
    out["effect_0_logit_CI95_upper"] = logit(out["effect_0_CI95_upper"])
    out["se_eta0"] = compute_logit_se_from_ci(
        out["effect_0_CI95_lower"], out["effect_0_CI95_upper"]
    )

    return out


def pool_reshuffle_runs(
    df: pd.DataFrame,
    arm_pooling: Literal[
        "simple_mean", "rubins_rules", "random_effects_hksj", "inter_intra_variance"
    ] = "inter_intra_variance",
    verbose: bool = False,
) -> pd.DataFrame:
    """Pool across reshuffle_run for each (method, outcome, simulation_run, experiment).

    Returns one row per (method, outcome, simulation_run, experiment) with pooled
    effect estimates, standard errors, and confidence intervals.
    """
    if verbose:
        print("  Adding logit transformations...")

    df_logit = add_logit_arm_metrics_simulation(df)

    group_cols = ["method", "outcome", "simulation_run", "experiment"]

    if verbose:
        print(f"  Pooling treatment arm using {arm_pooling}...")

    # Pool Treatment Arm (effect_1)
    arm1 = pool_arm_logits(
        df_logit,
        group_cols=group_cols,
        eta_col="eta1",
        se_col="se_eta1",
        out_prefix="eta1_pooled",
        pooling=arm_pooling,
    )

    if verbose:
        print(f"  Pooling control arm using {arm_pooling}...")

    # Pool Control Arm (effect_0)
    arm0 = pool_arm_logits(
        df_logit,
        group_cols=group_cols,
        eta_col="eta0",
        se_col="se_eta0",
        out_prefix="eta0_pooled",
        pooling=arm_pooling,
    )

    # Merge arms
    pooled = pd.merge(
        arm1, arm0, on=group_cols, how="inner", suffixes=("_arm1", "_arm0")
    )

    # Transform back to probability scale
    pooled["effect_1_pooled"] = inv_logit(pooled["eta1_pooled"])
    pooled["effect_0_pooled"] = inv_logit(pooled["eta0_pooled"])

    # Compute pooled effect (Risk Difference)
    pooled["effect_pooled"] = pooled["effect_1_pooled"] - pooled["effect_0_pooled"]

    # Delta method for SE of RD
    deriv1 = pooled["effect_1_pooled"] * (1 - pooled["effect_1_pooled"])
    deriv0 = pooled["effect_0_pooled"] * (1 - pooled["effect_0_pooled"])
    var_rd = (deriv1**2 * pooled["eta1_pooled_se"] ** 2) + (
        deriv0**2 * pooled["eta0_pooled_se"] ** 2
    )
    pooled["effect_pooled_se"] = np.sqrt(var_rd)

    # Confidence intervals
    z_crit = 1.96
    pooled["effect_pooled_CI95_lower"] = (
        pooled["effect_pooled"] - z_crit * pooled["effect_pooled_se"]
    )
    pooled["effect_pooled_CI95_upper"] = (
        pooled["effect_pooled"] + z_crit * pooled["effect_pooled_se"]
    )

    # Merge in true_effect from original data (should be constant within group)
    true_effects = df.groupby(group_cols)["true_effect"].first().reset_index()
    pooled = pd.merge(pooled, true_effects, on=group_cols, how="left")

    if verbose:
        print(f"  Pooled {len(pooled)} groups")

    return pooled


# -----------------------------------------------------------------------------
# Compute Simulation Metrics
# -----------------------------------------------------------------------------


def compute_simulation_metrics(df_pooled: pd.DataFrame) -> pd.DataFrame:
    """Compute bias, coverage, and other metrics from pooled results.

    Expects df_pooled to have:
    - effect_pooled: pooled effect estimate
    - effect_pooled_se: standard error
    - effect_pooled_CI95_lower, effect_pooled_CI95_upper: confidence intervals
    - true_effect: true causal effect
    """
    out = df_pooled.copy()

    # Bias
    out["bias"] = out["effect_pooled"] - out["true_effect"]

    # Relative Bias (only for non-zero true effects)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["relative_bias"] = np.where(
            out["true_effect"] != 0,
            out["bias"] / out["true_effect"],
            np.nan,
        )

    # Z-score (Standardized Bias)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["z_score"] = np.where(
            out["effect_pooled_se"] > 0,
            out["bias"] / out["effect_pooled_se"],
            np.nan,
        )

    # Coverage
    out["covered"] = (
        (out["effect_pooled_CI95_lower"] <= out["true_effect"])
        & (out["true_effect"] <= out["effect_pooled_CI95_upper"])
    ).astype(int)

    return out


# -----------------------------------------------------------------------------
# Second Aggregation: Across Simulation Runs
# -----------------------------------------------------------------------------


def aggregate_across_simulation_runs(df_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics across simulation_run for each (method, outcome, experiment).

    Returns:
    - Coverage: mean coverage
    - Bias: mean and std
    - Relative Bias: mean and std
    - Z-score: mean and std
    - Empirical SD: std of effect estimates
    - SE Calibration: empirical SD / mean estimated SE
    """
    group_cols = ["method", "outcome", "experiment"]

    # Prepare aggregations
    aggregations = {
        # Coverage
        "coverage_mean": ("covered", "mean"),
        "coverage_n": ("covered", "count"),
        # Bias
        "bias_mean": ("bias", "mean"),
        "bias_std": ("bias", lambda x: x.std(ddof=1)),
        # Relative Bias
        "relative_bias_mean": ("relative_bias", lambda x: x.mean()),
        "relative_bias_std": ("relative_bias", lambda x: x.std(ddof=1)),
        # Z-score
        "z_score_mean": ("z_score", lambda x: x.mean()),
        "z_score_std": ("z_score", lambda x: x.std(ddof=1)),
        # Empirical SD
        "empirical_sd": ("effect_pooled", lambda x: x.std(ddof=1)),
        # Mean estimated SE (for calibration)
        "mean_estimated_se": ("effect_pooled_se", "mean"),
        # Keep true_effect (should be constant)
        "true_effect": ("true_effect", "first"),
    }

    agg_df = df_metrics.groupby(group_cols, as_index=False).agg(**aggregations)

    # SE Calibration Ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        agg_df["se_calibration"] = np.where(
            agg_df["mean_estimated_se"] > 0,
            agg_df["empirical_sd"] / agg_df["mean_estimated_se"],
            np.nan,
        )

    return agg_df


# -----------------------------------------------------------------------------
# Plotting Functions
# -----------------------------------------------------------------------------


def plot_simulation_metric(
    df: pd.DataFrame,
    metric_name: str,
    y_label: str,
    output_path: Path,
    reference_line: Optional[float] = None,
    has_std: bool = True,
    figsize: tuple[int, int] = (14, 6),
) -> None:
    """Create a plot for a single metric across experiments and outcomes.

    Args:
        df: Aggregated results dataframe
        metric_name: Column name of the metric to plot (e.g., 'coverage_mean', 'bias_mean')
        y_label: Label for y-axis
        output_path: Path to save the figure
        reference_line: Optional y-value for horizontal reference line
        has_std: Whether this metric has a corresponding _std column
        figsize: Figure size (width, height)
    """
    if df.empty:
        print(f"  No data to plot for {metric_name}")
        return

    # Create x-axis labels: experiment + outcome
    df = df.copy()
    df["x_label"] = df["experiment"] + "_" + df["outcome"]

    # Sort by experiment and outcome for consistent ordering
    df = df.sort_values(["experiment", "outcome"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=figsize)

    # Get unique methods
    methods = sorted(df["method"].unique())
    colors = {
        "IPW": "#E63946",
        "TMLE": "#2E86AB",
        "TMLE_TH": "#06A77D",
        "RD": "#F77F00",
    }
    markers = {"IPW": "o", "TMLE": "s", "TMLE_TH": "^", "RD": "D"}

    # Calculate x-positions with offset for multiple methods
    x_labels = df["x_label"].unique()
    x_positions = {label: i for i, label in enumerate(x_labels)}

    # Plot each method
    for idx, method in enumerate(methods):
        method_data = df[df["method"] == method].copy()
        if method_data.empty:
            continue

        # Calculate offset for this method
        n_methods = len(methods)
        if n_methods > 1:
            offset_range = 0.3
            offset = (idx - (n_methods - 1) / 2) * (offset_range / (n_methods - 1))
        else:
            offset = 0

        x = np.array([x_positions[label] + offset for label in method_data["x_label"]])
        y = method_data[metric_name].values

        color = colors.get(method, "#6C757D")
        marker = markers.get(method, "o")

        if has_std:
            std_col = metric_name.replace("_mean", "_std")
            if std_col in method_data.columns:
                yerr = method_data[std_col].values
                ax.errorbar(
                    x,
                    y,
                    yerr=yerr,
                    marker=marker,
                    linestyle="",
                    color=color,
                    capsize=4,
                    capthick=1.5,
                    markersize=8,
                    markeredgewidth=1.5,
                    markeredgecolor="white",
                    elinewidth=2,
                    alpha=0.85,
                    label=method,
                )
            else:
                ax.plot(
                    x,
                    y,
                    marker=marker,
                    linestyle="",
                    color=color,
                    markersize=8,
                    markeredgewidth=1.5,
                    markeredgecolor="white",
                    alpha=0.85,
                    label=method,
                )
        else:
            ax.plot(
                x,
                y,
                marker=marker,
                linestyle="",
                color=color,
                markersize=10,
                markeredgewidth=1.5,
                markeredgecolor="white",
                alpha=0.85,
                label=method,
            )

    # Add reference line if specified
    if reference_line is not None:
        linestyle = ":" if metric_name == "coverage_mean" else "--"
        linewidth = 2 if metric_name == "coverage_mean" else 1.5
        alpha = 0.6 if metric_name == "coverage_mean" else 0.4
        color_line = "#E74C3C" if metric_name == "coverage_mean" else "#2C3E50"
        ax.axhline(
            reference_line,
            color=color_line,
            linestyle=linestyle,
            alpha=alpha,
            linewidth=linewidth,
            zorder=1,
        )

    # Formatting
    ax.set_xlabel("Experiment + Outcome", fontsize=12, fontweight="medium")
    ax.set_ylabel(y_label, fontsize=12, fontweight="medium")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9)
    ax.legend(loc="best", frameon=True, framealpha=0.95, edgecolor="#CCCCCC")
    ax.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_edgecolor("#CCCCCC")
        spine.set_linewidth(1.2)

    ax.set_facecolor("white")

    plt.tight_layout()
    plt.savefig(
        output_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    plt.close(fig)

    print(f"  Saved {metric_name} plot to: {output_path}")


# -----------------------------------------------------------------------------
# CLI and Main Orchestration
# -----------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create simulation analysis plots for treatment effect estimators",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/simulation"),
        help="Directory containing simulation results file",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="exp1_finished.txt",
        help="Name of the simulation results file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Base directory for output figures",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="IPW",
        help="Estimation method to analyze (IPW, TMLE, TMLE_TH, RD)",
    )
    parser.add_argument(
        "--arm-pooling",
        choices=[
            "random_effects_hksj",
            "simple_mean",
            "rubins_rules",
            "inter_intra_variance",
        ],
        default="inter_intra_variance",
        help=(
            "Arm-level pooling method for reshuffle runs. "
            "See trace.statistics.pool_arm_logits for details."
        ),
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Verbose output",
    )
    parser.add_argument(
        "--fast",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fast mode: only produce main plots, skip detailed diagnostics",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    input_path = args.input_dir / args.input_file
    input_folder_name = args.input_dir.name
    output_dir = (
        args.output_dir / input_folder_name / "simulation_analysis" / args.arm_pooling
    )

    print("=" * 80)
    print("SIMULATION ANALYSIS")
    print("=" * 80)
    print(f"\nMethod: {args.method}")
    print(f"Input file: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Arm pooling: {args.arm_pooling}")
    if args.fast:
        print("Fast mode: enabled")
    print()

    # ------------------------------------------------------------------
    # Load and preprocess data
    # ------------------------------------------------------------------
    print("Loading simulation data...")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df_raw = pd.read_csv(input_path)
    print(f"Loaded {len(df_raw)} rows")
    print(f"Methods: {sorted(df_raw['method'].unique())}")
    print(f"Outcomes: {sorted(df_raw['outcome'].unique())}")
    print(f"Experiments: {sorted(df_raw['experiment'].unique())}")
    print(
        f"Simulation runs: {df_raw['simulation_run'].nunique()}, "
        f"Reshuffle runs per simulation: ~{df_raw.groupby('simulation_run')['reshuffle_run'].nunique().mean():.1f}"
    )

    # Filter to requested method
    df_method = df_raw[df_raw["method"] == args.method].copy()
    if df_method.empty:
        raise ValueError(
            f"No data found for method '{args.method}'. "
            f"Available methods: {sorted(df_raw['method'].unique())}"
        )

    print(f"\nFiltered to method '{args.method}': {len(df_method)} rows")

    # Validate required columns
    required_cols = [
        "effect_1",
        "effect_0",
        "effect_1_CI95_lower",
        "effect_1_CI95_upper",
        "effect_0_CI95_lower",
        "effect_0_CI95_upper",
        "true_effect",
        "outcome",
        "simulation_run",
        "reshuffle_run",
        "experiment",
    ]
    missing = [col for col in required_cols if col not in df_method.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Parse experiment parameters
    if args.verbose:
        print("\nParsing experiment parameters...")

    parsed_params = df_method["experiment"].apply(parse_experiment_params)
    for param in ["ce", "cy", "y", "i"]:
        df_method[param] = parsed_params.apply(lambda x: x[param])

    # ------------------------------------------------------------------
    # First Aggregation: Pool across reshuffle runs
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 1: Pooling across reshuffle runs")
    print("=" * 80)

    df_pooled = pool_reshuffle_runs(
        df_method, arm_pooling=args.arm_pooling, verbose=args.verbose
    )

    print(f"\nPooled to {len(df_pooled)} simulation runs")

    # ------------------------------------------------------------------
    # Compute simulation metrics
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 2: Computing simulation metrics")
    print("=" * 80)

    df_metrics = compute_simulation_metrics(df_pooled)

    print(f"\nComputed metrics for {len(df_metrics)} pooled estimates")
    print(
        f"  Bias range: [{df_metrics['bias'].min():.4f}, {df_metrics['bias'].max():.4f}]"
    )
    print(f"  Coverage mean: {df_metrics['covered'].mean():.3f}")

    # ------------------------------------------------------------------
    # Second Aggregation: Across simulation runs
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 3: Aggregating across simulation runs")
    print("=" * 80)

    df_agg = aggregate_across_simulation_runs(df_metrics)

    print(f"\nAggregated to {len(df_agg)} (method, outcome, experiment) combinations")
    print(
        f"  Coverage: mean={df_agg['coverage_mean'].mean():.3f}, "
        f"min={df_agg['coverage_mean'].min():.3f}, max={df_agg['coverage_mean'].max():.3f}"
    )
    print(
        f"  Bias: mean={df_agg['bias_mean'].mean():.4f}, "
        f"std={df_agg['bias_mean'].std():.4f}"
    )

    # ------------------------------------------------------------------
    # Save aggregated results
    # ------------------------------------------------------------------
    ensure_output_directory(output_dir)
    results_path = output_dir / "aggregated_results.csv"
    df_agg.to_csv(results_path, index=False)
    print(f"\nSaved aggregated results to: {results_path}")

    # ------------------------------------------------------------------
    # Create plots
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 4: Creating plots")
    print("=" * 80)
    print()

    # Coverage
    plot_simulation_metric(
        df_agg,
        metric_name="coverage_mean",
        y_label="95% CI Coverage",
        output_path=output_dir / "coverage_by_experiment.png",
        reference_line=0.95,
        has_std=False,
    )

    # Bias
    plot_simulation_metric(
        df_agg,
        metric_name="bias_mean",
        y_label="Bias",
        output_path=output_dir / "bias_by_experiment.png",
        reference_line=0.0,
        has_std=True,
    )

    # Relative Bias
    plot_simulation_metric(
        df_agg,
        metric_name="relative_bias_mean",
        y_label="Relative Bias",
        output_path=output_dir / "relative_bias_by_experiment.png",
        reference_line=0.0,
        has_std=True,
    )

    # Z-score
    plot_simulation_metric(
        df_agg,
        metric_name="z_score_mean",
        y_label="Standardized Bias (Z-Score)",
        output_path=output_dir / "z_score_by_experiment.png",
        reference_line=0.0,
        has_std=True,
    )

    # Empirical SD
    plot_simulation_metric(
        df_agg,
        metric_name="empirical_sd",
        y_label="Empirical Standard Deviation",
        output_path=output_dir / "empirical_sd_by_experiment.png",
        reference_line=None,
        has_std=False,
    )

    # SE Calibration
    plot_simulation_metric(
        df_agg,
        metric_name="se_calibration",
        y_label="SE Calibration Ratio (Empirical SE / Estimated SE)",
        output_path=output_dir / "se_calibration_by_experiment.png",
        reference_line=1.0,
        has_std=False,
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
