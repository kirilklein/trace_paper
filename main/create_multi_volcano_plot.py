"""Unified script to create multi-dataset volcano plots.

Creates a faceted volcano plot (one subplot per input directory) for a single method.
Uses a Manhattan-style color scheme for significance bins and effect direction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from adjustText import adjust_text

from main.helpers import (
    ensure_output_directory,
    ensure_required_columns,
    summarise_per_run_effects,
)
from trace.constants import DEFAULT_ALPHA
from trace.io import (
    PrevalenceStats,
    load_atc_dictionary,
    load_estimates,
    load_prevalence_statistics,
    rename_prevalence_columns,
)
from trace.plotting.volcano import (
    prepare_volcano_data,
)
from trace.statistics import compute_rd_pvalues


def main() -> None:
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Create multi-dataset volcano plots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="List of directories containing input data files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory for output figures",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="IPW",
        help="Method to filter for (e.g., IPW, TMLE)",
    )
    parser.add_argument(
        "--effect-type",
        choices=["RD", "RR", "log-RR"],
        default="log-RR",
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
        "--min-prevalence",
        type=float,
        default=0.0,
        help="Minimum prevalence threshold (as proportion 0-1)",
    )
    parser.add_argument(
        "--exclude-outcomes",
        type=str,
        default="",
        help="Comma-separated list of outcome codes to exclude (e.g., 'A10BJ,A10BK')",
    )
    parser.add_argument(
        "--subplot-labels",
        type=str,
        default="",
        help="Comma-separated mapping of dataset names to display labels (e.g., 'plus50:Main,cvd:CVD').",
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
        default="inter_intra_variance",
        help="Arm-level pooling on the logit scale across runs",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Verbose output",
    )

    args = parser.parse_args()

    # Parse subplot labels mapping
    label_map = {}
    if args.subplot_labels:
        try:
            pairs = [item.strip() for item in args.subplot_labels.split(",") if item.strip()]
            for pair in pairs:
                if ":" in pair:
                    key, value = pair.split(":", 1)
                    label_map[key.strip()] = value.strip()
                else:
                    pass
        except Exception as e:
            print(f"Warning: Could not parse --subplot-labels: {e}")

    # Determine effect parameters
    effect_type = args.effect_type
    if effect_type == "log-RR":
        effect_col = "log_RR"  # We compute log_RR explicitly
        effect_label = "Log Risk Ratio"
        null_value = 0.0
        xscale = "linear"
        effect_alias = "log_RR"
    elif effect_type == "RR":
        effect_col = "RR"
        effect_label = "Risk ratio (RR)"
        null_value = 1.0
        xscale = "log" # usually better for RR
        effect_alias = "RR"
    else:  # RD
        effect_col = "RD"
        effect_label = "Risk difference (RD)"
        null_value = 0.0
        xscale = "linear"
        effect_alias = "RD"

    output_suffix = effect_type.lower().replace("-", "_")

    print(f"Effect type: {effect_type}")
    print(f"Method: {args.method}")
    print(f"Adjustment: {args.adjust}")
    if label_map:
        print(f"Subplot labels: {label_map}")

    # Prepare output directory
    output_dir = args.output_dir / "multi_dataset" / args.adjust / args.arm_pooling
    ensure_output_directory(output_dir)
    print(f"Output directory: {output_dir}\n")

    # Load ATC dictionary once
    atc_mapping = load_atc_dictionary()

    collected_data = []

    for input_dir in args.input_dirs:
        print(f"Processing: {input_dir}")
        estimates_path = input_dir / "combined_estimates.txt"
        stats_path = input_dir / "combined_stats.txt"

        if not estimates_path.exists():
            print(f"  Skipping (not found): {estimates_path}")
            continue

        # Load estimates
        df_raw = load_estimates(estimates_path)

        # Filter method
        df_method = df_raw[df_raw["method"] == args.method].copy()
        if df_method.empty:
            print(f"  No data for method {args.method}")
            continue

        required_cols = [
            "effect_1",
            "effect_0",
            "effect_1_CI95_lower",
            "effect_1_CI95_upper",
            "effect_0_CI95_lower",
            "effect_0_CI95_upper",
        ]
        ensure_required_columns(df_method, required_cols)

        # Compute per-run stats (for summary)
        df_per_run = compute_rd_pvalues(
            df_method,
            group_cols=None,
            arm_pooling=args.arm_pooling,
            verbose=False,
        )

        # Compute pooled estimates
        df_pooled = compute_rd_pvalues(
            df_method,
            group_cols=("method", "outcome"),
            arm_pooling=args.arm_pooling,
            verbose=False,
        )

        if df_pooled.empty:
            print("  No pooled results produced.")
            continue

        # Handle Effect Types (RR calculation)
        # Always compute RR and log_RR if possible
        with np.errstate(divide="ignore", invalid="ignore"):
            if not df_per_run.empty:
                df_per_run = df_per_run.copy()
                df_per_run["RR"] = df_per_run["p1_hat"] / df_per_run["p0_hat"]

            if not df_pooled.empty:
                df_pooled = df_pooled.copy()
                # If log_RR not already present from compute_rd_pvalues (it usually is for pooled)
                if "log_RR" not in df_pooled.columns:
                     # If p1_hat/p0_hat are available
                     if "p1_hat" in df_pooled.columns and "p0_hat" in df_pooled.columns:
                          df_pooled["RR"] = df_pooled["p1_hat"] / df_pooled["p0_hat"]
                          df_pooled["log_RR"] = np.log(df_pooled["RR"])
                else:
                     df_pooled["RR"] = np.exp(df_pooled["log_RR"])


        # Summarise per-run
        per_run_summary = (
            summarise_per_run_effects(
                df_per_run, effect_col=effect_col if effect_col in df_per_run.columns else "RD", effect_alias=effect_alias
            )
            if not df_per_run.empty
            else pd.DataFrame()
        )

        # Load prevalence stats
        prevalence_summary = pd.DataFrame()
        if stats_path.exists():
            try:
                prevalence_stats: PrevalenceStats = load_prevalence_statistics(
                    stats_path
                )
                prevalence_summary = rename_prevalence_columns(prevalence_stats.summary)
            except Exception as e:
                print(f"  Warning: Could not load prevalence stats: {e}")

        # Merge metadata
        # 1. Per run summary
        if not per_run_summary.empty:
            df_pooled = df_pooled.merge(
                per_run_summary, on=["method", "outcome"], how="left"
            )

        # 2. Prevalence
        if not prevalence_summary.empty:
            df_pooled = df_pooled.merge(prevalence_summary, on="outcome", how="left")

        # 3. ATC
        df_pooled["atc_description"] = df_pooled["outcome"].map(atc_mapping)
        df_pooled["outcome_label"] = df_pooled["outcome"].where(
            df_pooled["atc_description"].isna(),
            df_pooled["outcome"] + " · " + df_pooled["atc_description"],
        )

        # Add dataset tag
        dataset_name = input_dir.name
        df_pooled["dataset"] = dataset_name

        collected_data.append(df_pooled)
        print(f"  Added {len(df_pooled)} outcomes for dataset: {dataset_name}")

    if not collected_data:
        print("No data collected from any directory. Exiting.")
        return

    # Combine all datasets
    df_combined = pd.concat(collected_data, ignore_index=True)
    print(f"\nTotal combined rows: {len(df_combined)}")

    # Filter by min prevalence
    if args.min_prevalence > 0.0:
        n_before = len(df_combined)
        df_combined = df_combined[
            df_combined["prevalence_mean_total"] >= args.min_prevalence
        ].copy()
        print(
            f"Filtered by min prevalence {args.min_prevalence}: {n_before} -> {len(df_combined)}"
        )

    # Prepare Volcano Data (Compute q-values per dataset)
    # passing method_col="dataset" so that p-value adjustment is done per dataset group
    # We use RD col here as a placeholder for 'effect' but later we ensure we have the right col
    rd_col_arg = "RD"
    if effect_col in df_combined.columns:
        rd_col_arg = effect_col
    
    df_volcano = prepare_volcano_data(
        df_combined,
        rd_col=rd_col_arg,
        p_col="p_value",
        method_col="dataset",
        outcome_col="outcome",
        adjust=args.adjust,
        adjust_per="by_method",  # means by dataset here
        effect_alias=effect_alias,
    )

    # Merge back the metadata columns
    meta_cols = [
        c
        for c in df_combined.columns
        if c not in df_volcano.columns and c not in ["p_value", effect_alias]
    ]
    
    df_final = df_volcano.merge(
        df_combined[["dataset", "outcome"] + meta_cols],
        on=["dataset", "outcome"],
        how="left",
    )

    # Filter excluded outcomes (after q-value calculation, to match create_manhattan_plot logic)
    if args.exclude_outcomes:
        exclude_list = [
            o.strip() for o in args.exclude_outcomes.split(",") if o.strip()
        ]
        if exclude_list:
            n_before_exclude = len(df_final)
            df_final = df_final[~df_final["outcome"].isin(exclude_list)].copy()
            n_excluded = n_before_exclude - len(df_final)
            print(
                f"\nExcluded {n_excluded} outcomes based on filter list: {', '.join(exclude_list)}"
            )

    # If effect_col was somehow lost or not the main one, ensure it is present
    if effect_col not in df_final.columns and effect_col in df_combined.columns:
         # This happens if prepare_volcano_data renamed it or didn't include it because we passed a different col
         pass

    # Save combined results
    output_csv = output_dir / f"combined_results_multi_{output_suffix}.csv"
    df_final.to_csv(output_csv, index=False)
    print(f"Saved combined results to: {output_csv}")

    # Summary statistics
    print("\nSummary statistics:")
    
    # Determine plotting order based on input arguments (not alphabetical sorting)
    input_dataset_names = [d.name for d in args.input_dirs]
    # Filter to those actually in the dataframe (handles empty results etc)
    existing_datasets = set(df_final["dataset"].unique())
    datasets = [name for name in input_dataset_names if name in existing_datasets]

    for ds in datasets:
        d = df_final[df_final["dataset"] == ds]
        n_sig = (d["q_value"] < DEFAULT_ALPHA).sum()
        print(f"  {ds}: {len(d)} outcomes, {n_sig} significant (q < {DEFAULT_ALPHA})")

    # ------------------------------------------------------------------
    # Custom Multi-Dataset Volcano Plot with Manhattan Coloring
    # ------------------------------------------------------------------
    print("\nCreating multi-dataset volcano plot (Manhattan style coloring)...")
    
    n_panels = len(datasets)
    if n_panels == 0:
        return

    # Vertical layout: nrows=n_panels, ncols=1
    # Independent axes: sharex=False, sharey=False
    fig, axes = plt.subplots(
        n_panels, 
        1, 
        figsize=(7, 5 * n_panels), 
        sharey=False, 
        sharex=False
    )
    
    if n_panels == 1:
        axes = [axes]

    # Significance thresholds and colors
    thresholds = np.array([0.05, 0.01, 0.001])
    reds = np.array(["#ffb3b3", "#bf3a3a", "#9c0202"], dtype=object)
    blues = np.array(["#b3c6ff", "#4d79ff", "#0033cc"], dtype=object)
    neutral = "#7f7f7f" # Grey for ns

    for ax, ds in zip(axes, datasets):
        d = df_final[df_final["dataset"] == ds].copy()
        
        # Ensure we have data
        if d.empty:
            display_title = label_map.get(ds, ds)
            ax.set_title(display_title)
            continue

        # Determine colors based on q-value and effect direction
        # 1. Bin q-values
        bins = np.digitize(d["q_value"].fillna(1.0).to_numpy(), thresholds)
        
        colors = []
        for effect, q, b in zip(d[effect_col], d["q_value"], bins):
            if pd.isna(q) or q > 0.05:
                colors.append(neutral)
                continue
            
            # Significant
            idx = min(b - 1, len(reds) - 1)
            if effect > 0:
                colors.append(reds[idx])
            else:
                colors.append(blues[idx])
        
        d["color"] = colors
        
        # Scatter plot
        # Separate significant vs non-significant for z-order
        is_sig = d["q_value"] < 0.05
        
        # Plot NS first (background)
        d_ns = d[~is_sig]
        if not d_ns.empty:
            ax.scatter(
                d_ns[effect_col],
                d_ns["neglog10p"],
                c=d_ns["color"],
                s=20,
                alpha=0.5,
                label="NS"
            )
            
        # Plot Sig second (foreground)
        d_sig = d[is_sig]
        if not d_sig.empty:
            ax.scatter(
                d_sig[effect_col],
                d_sig["neglog10p"],
                c=d_sig["color"],
                s=25,
                alpha=0.9,
                label="Significant"
            )

        # Reference lines
        ax.axhline(-np.log10(DEFAULT_ALPHA), linestyle="--", linewidth=1, color="gray", alpha=0.5)
        ax.axvline(null_value, linestyle="--", linewidth=1, color="gray", alpha=0.5)

        # Use mapped label if available
        display_title = label_map.get(ds, ds)
        ax.set_title(display_title)
        ax.set_ylabel("-log10(p-value)")
        # Add x-label for each subplot since sharex=False
        ax.set_xlabel(effect_label)
        ax.grid(alpha=0.2, linestyle=":", linewidth=0.8)

        if xscale:
             ax.set_xscale(xscale)

        # Annotate top hits per panel
        max_labels = 10
        top_hits = d.sort_values("neglog10p", ascending=False).head(max_labels)
        texts = []
        for _, row in top_hits.iterrows():
            label = row["outcome"]
            # Simplified label if mapping exists
            if pd.notna(row["atc_description"]):
                 # Maybe use just outcome code? Or code + short desc?
                 # Let's use outcome code for compactness in volcano
                 label = row["outcome"]
            
            texts.append(
                ax.text(
                    row[effect_col],
                    row["neglog10p"],
                    label,
                    fontsize=8,
                    alpha=0.8
                )
            )
        
        if texts:
             adjust_text(
                texts,
                ax=ax,
                arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5, lw=0.5),
             )

    # Custom Legend (place on the first axes or figure level?)
    # Figure level is better for shared legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#9c0202', markersize=8, label='Effect > 0, q ≤ 0.001'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#bf3a3a', markersize=8, label='Effect > 0, q ≤ 0.01'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ffb3b3', markersize=8, label='Effect > 0, q ≤ 0.05'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=neutral, markersize=8, label='q > 0.05'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#0033cc', markersize=8, label='Effect < 0 (Blue scale)'),
    ]
    
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.05), # Bottom center
        ncol=2,
        title="Significance",
        fontsize=9
    )

    fig.suptitle(f"Volcano Plot ({args.method}) - {args.adjust} adjusted", fontsize=14, y=0.99)
    # Adjust layout to accommodate bottom legend
    fig.tight_layout(rect=[0, 0.08, 1, 0.98]) 

    output_png = output_dir / f"volcano_plot_multi_{output_suffix}.png"
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {output_png}")
    plt.close(fig)

    print("Done!")


if __name__ == "__main__":
    main()
