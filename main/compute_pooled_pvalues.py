"""Compute pooled estimates and p-values for treatment effect analysis.

Simplified script focused on statistical computation without visualization.
Performs arm-level pooling and p-value adjustment, then exports results to CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from trace.constants import METHODS_WITH_ARMS
from trace.io import filter_methods_with_arm_cis, load_estimates
from trace.plotting.volcano import prepare_volcano_data
from trace.statistics import compute_rd_pvalues


def main() -> None:
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Compute pooled estimates and p-values for treatment effects",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/semaglutide"),
        help="Directory containing input data files (combined_estimates.txt)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory for output CSV files",
    )
    parser.add_argument(
        "--effect-type",
        choices=["RD", "RR", "log-RR"],
        default="RD",
        help="Effect measure: Risk Difference (RD), Risk Ratio (RR), or log Risk Ratio",
    )
    parser.add_argument(
        "--adjust",
        choices=[
            "bh",
            "by",
            "tsbh",
            "tsbky",
            "bonferroni",
            "sidak",
            "holm",
            "holm-sidak",
            "hochberg",
            "hommel",
            "none",
        ],
        default="bh",
        help="Multiple testing adjustment method",
    )
    parser.add_argument(
        "--adjust-per",
        dest="adjust_per",
        choices=["by_method", "global"],
        default="by_method",
        help="Scope of multiple testing adjustment",
    )
    parser.add_argument(
        "--arm-pooling",
        choices=[
            "random_effects_hksj",
            "correlation_adjusted",
            "simple_mean",
            "rubins_rules",
            "inter_intra_variance",
        ],
        default="simple_mean",
        help=(
            "Arm-level pooling on the logit scale across runs: "
            "'random_effects_hksj' (DerSimonian–Laird with HKSJ SE), "
            "'correlation_adjusted' (uses weights and rho), "
            "'simple_mean' (unweighted mean of logits; SEM uses sample std with ddof=1), "
            "'rubins_rules' (Rubin's rules for multiple imputation), "
            "'inter_intra_variance' (combines within and between variance)"
        ),
    )

    args = parser.parse_args()

    # Construct output directory
    input_folder_name = args.input_dir.name
    output_dir = args.output_dir / input_folder_name / args.adjust / args.arm_pooling
    output_dir.mkdir(parents=True, exist_ok=True)

    # Construct input file path
    estimates_path = args.input_dir / "combined_estimates.txt"

    # Determine effect parameters
    effect_type = args.effect_type
    if effect_type in ["RR", "log-RR"]:
        effect_col = "RR"
        effect_alias = "RR"
    else:  # RD
        effect_col = "RD"
        effect_alias = "RD"

    # Create output suffix for files
    output_suffix = effect_type.lower().replace("-", "_")

    print("=" * 80)
    print("COMPUTE POOLED P-VALUES")
    print("=" * 80)
    print(f"\nEffect type: {effect_type}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Arm pooling method: {args.arm_pooling}")
    print(f"P-value adjustment: {args.adjust} ({args.adjust_per})")
    print()

    # Load data
    print("Loading data...")
    df_raw = load_estimates(estimates_path)
    print(f"Loaded {len(df_raw)} rows from {estimates_path}")

    # Filter to methods with arm-level CIs
    df_with_arms = filter_methods_with_arm_cis(df_raw, METHODS_WITH_ARMS)
    print(f"Filtered to IPW and TMLE: {len(df_with_arms)} rows")

    if df_with_arms.empty:
        print("No data available after filtering. Exiting.")
        return

    # Compute pooled estimates based on effect type
    if effect_type in ["RR", "log-RR"]:
        print("\nPooling arm logits and computing p-values...")
        df_pooled = compute_rd_pvalues(
            df_with_arms,
            group_cols=("method", "outcome"),
            arm_pooling=args.arm_pooling,
            verbose=False,
        )

        # Derive pooled RR from pooled arm probabilities
        if not df_pooled.empty:
            df_pooled = df_pooled.copy()
            df_pooled["RR"] = df_pooled["p1_hat"] / df_pooled["p0_hat"]

        print(f"Computed {len(df_pooled)} method-outcome combinations")

        if df_pooled.empty:
            print("No pooled results available. Exiting.")
            return

        print("\nRisk ratio summary:")
        print(f"  RR range: [{df_pooled['RR'].min():.3f}, {df_pooled['RR'].max():.3f}]")
        print(f"  RR median: {df_pooled['RR'].median():.3f}")

    else:  # RD
        print("\nPooling arm logits and computing risk differences...")
        df_pooled = compute_rd_pvalues(
            df_with_arms,
            group_cols=("method", "outcome"),
            arm_pooling=args.arm_pooling,
            verbose=False,
        )
        print(f"Computed {len(df_pooled)} method-outcome combinations")

        if df_pooled.empty:
            print("No pooled results available. Exiting.")
            return

        # Display heterogeneity statistics if available
        if "eta1_tau2" in df_pooled.columns:
            print("\nHeterogeneity statistics (tau²):")
            tau_cols = [c for c in ["eta1_tau2", "eta0_tau2"] if c in df_pooled.columns]
            for col in tau_cols:
                print(
                    f"  {col}: mean={df_pooled[col].mean():.4e}, "
                    f"max={df_pooled[col].max():.4e}"
                )

    # Apply p-value adjustment
    print("\nApplying p-value adjustment...")
    df_results = prepare_volcano_data(
        df_pooled,
        rd_col=effect_col,
        p_col="p_value",
        method_col="method",
        outcome_col="outcome",
        adjust=args.adjust,
        adjust_per=args.adjust_per,
        effect_alias=effect_alias,
    )

    # Save results to CSV
    output_path = output_dir / f"pooled_results_{output_suffix}.csv"
    df_results.to_csv(output_path, index=False)
    print(f"\nSaved pooled results to: {output_path}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total method-outcome combinations: {len(df_results)}")
    print(f"\nBreakdown by method:")
    for method in sorted(df_results["method"].unique()):
        n = (df_results["method"] == method).sum()
        print(f"  {method}: {n} outcomes")

    # Check for significant results
    if "q_value" in df_results.columns:
        alpha = 0.05
        print(f"\nSignificant results (q < {alpha}):")
        for method in sorted(df_results["method"].unique()):
            method_data = df_results[df_results["method"] == method]
            n_sig = (method_data["q_value"] < alpha).sum()
            pct = 100 * n_sig / len(method_data) if len(method_data) > 0 else 0
            print(f"  {method}: {n_sig}/{len(method_data)} ({pct:.1f}%)")

    print("\nDone!")


if __name__ == "__main__":
    main()
