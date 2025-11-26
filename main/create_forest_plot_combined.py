"""Create combined forest plots for Semaglutide, Diabetes, and CVD cohorts.

Displays Log Risk Ratios (Log-RR) in 2 parts, each with 2 columns, ordered by effect size.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from trace.io import load_atc_dictionary


def load_and_filter_data(path: Path, method: str = "IPW") -> pd.DataFrame:
    """Load pooled results and filter for specific method and valid outcomes."""
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    df = pd.read_csv(path)

    # Filter by method
    if "method" in df.columns:
        df = df[df["method"] == method].copy()

    # Filter outcomes
    # Remove A10BJ
    df = df[df["outcome"] != "A10BJ"]

    # NOTE: The user explicitly requested to check why we filter V-outcomes and noted a discrepancy.
    # In many medical datasets, 'V' codes in ATC often refer to "Various" or non-drug products
    # (e.g. contrast media, diagnostic agents), which might be excluded from standard drug analysis.
    # However, if the user sees valid effects there, we can comment this out or adjust based on their feedback.
    # The previous code was: df = df[~df["outcome"].str.startswith("V")]
    # The user commented it out in their edit, so we keep it commented out here to align with their intent.
    # df = df[~df["outcome"].str.startswith("V")]

    return df


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create combined forest plots (Log-RR) in 2 parts (2 columns each)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Root of the repository",
    )
    parser.add_argument(
        "--semaglutide-file",
        type=Path,
        default=Path(
            "figures/plus50/bh/inter_intra_variance/pooled_results_log_rr.csv"
        ),
        help="Path to Semaglutide pooled results CSV",
    )
    # parser.add_argument(
    #     "--diabetes-file",
    #     type=Path,
    #     default=Path("data/diabetes_2/pooled_results_log_rr.csv"),
    #     help="Path to Diabetes pooled results CSV",
    # )
    parser.add_argument(
        "--cvd-file",
        type=Path,
        default=Path("figures/cvd/bh/inter_intra_variance/pooled_results_log_rr.csv"),
        help="Path to CVD pooled results CSV",
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
        help="Method to filter by (e.g., IPW, TMLE)",
    )

    args = parser.parse_args()

    # Load ATC dictionary
    try:
        atc_mapping = load_atc_dictionary()
    except Exception as e:
        print(f"Warning: Could not load ATC dictionary from default location: {e}")
        atc_mapping = {}

    # Load data
    print(f"Loading Semaglutide data from {args.semaglutide_file}...")
    df_sema = load_and_filter_data(args.semaglutide_file, args.method)

    print(f"Loading CVD data from {args.cvd_file}...")
    df_cvd = load_and_filter_data(args.cvd_file, args.method)

    # Merge dataframes
    # Semaglutide is the base
    # Outer join to keep all outcomes from both cohorts
    print("Merging dataframes...")

    df_sema = df_sema.rename(
        columns={
            "log_RR": "log_RR_sema",
            "log_RR_CI95_lower": "ci_low_sema",
            "log_RR_CI95_upper": "ci_high_sema",
        }
    )
    df_cvd = df_cvd.rename(
        columns={
            "log_RR": "log_RR_cvd",
            "log_RR_CI95_lower": "ci_low_cvd",
            "log_RR_CI95_upper": "ci_high_cvd",
        }
    )

    combined = df_sema.merge(
        df_cvd[["outcome", "log_RR_cvd", "ci_low_cvd", "ci_high_cvd"]],
        on="outcome",
        how="outer",
    )

    # Add ATC descriptions
    combined["atc_desc"] = combined["outcome"].map(atc_mapping).fillna("Unknown")
    combined["label"] = combined["outcome"]

    # Sort by Semaglutide effect size (descending)
    combined = combined.sort_values("log_RR_sema", ascending=False).reset_index(
        drop=True
    )

    print(f"Total outcomes after merging and filtering: {len(combined)}")

    # Prepare plot data
    # Split into 2 parts, each with 2 columns (total 4 columns logic, but split into 2 files)
    n_items = len(combined)

    cols_total = 4
    chunk_size = np.ceil(n_items / cols_total).astype(int)

    chunks = [combined.iloc[i : i + chunk_size] for i in range(0, n_items, chunk_size)]

    # Ensure we have chunks for 4 columns (pad if needed)
    while len(chunks) < 4:
        chunks.append(pd.DataFrame(columns=combined.columns))

    # Part 1: chunks 0 and 1
    # Part 2: chunks 2 and 3

    parts = [("part1", chunks[0:2]), ("part2", chunks[2:4])]

    for part_name, part_chunks in parts:
        # Setup figure
        # Adjust height to fit content without excessive whitespace
        # Base margin + per-item height
        fig_height = 2 + (chunk_size * 0.25)
        fig, axes = plt.subplots(1, 2, figsize=(12, fig_height), sharey=False)
        plt.subplots_adjust(wspace=0.4)

        # If only 1 chunk in this part (rare/edge case), axes might not be array
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        for i, ax in enumerate(axes):
            if i >= len(part_chunks):
                ax.axis("off")
                continue

            df_chunk = part_chunks[i].copy()
            if df_chunk.empty:
                ax.axis("off")
                continue

            # Reverse order for plotting (top-down)
            df_chunk = df_chunk.iloc[::-1]

            y_pos = np.arange(len(df_chunk))

            # Plot Semaglutide (Black)
            # Filter out missing values for this cohort
            mask_sema = df_chunk["log_RR_sema"].notna()
            if mask_sema.any():
                ax.errorbar(
                    df_chunk.loc[mask_sema, "log_RR_sema"],
                    y_pos[mask_sema],
                    xerr=[
                        df_chunk.loc[mask_sema, "log_RR_sema"]
                        - df_chunk.loc[mask_sema, "ci_low_sema"],
                        df_chunk.loc[mask_sema, "ci_high_sema"]
                        - df_chunk.loc[mask_sema, "log_RR_sema"],
                    ],
                    fmt="o",
                    color="black",
                    ecolor="gray",
                    capsize=2,
                    markersize=4,
                    label="Main" if i == 0 else "",
                )

            # Plot CVD (Gentler Green) - offset slightly
            mask_cvd = df_chunk["log_RR_cvd"].notna()
            if mask_cvd.any():
                offset = 0.25
                # Use a gentler green, e.g., 'mediumseagreen' or a hex code like '#3cb371' or '#66c2a5' (from Set2)
                cvd_color = "#66c2a5"
                ax.errorbar(
                    df_chunk.loc[mask_cvd, "log_RR_cvd"],
                    y_pos[mask_cvd] + offset,
                    xerr=[
                        df_chunk.loc[mask_cvd, "log_RR_cvd"]
                        - df_chunk.loc[mask_cvd, "ci_low_cvd"],
                        df_chunk.loc[mask_cvd, "ci_high_cvd"]
                        - df_chunk.loc[mask_cvd, "log_RR_cvd"],
                    ],
                    fmt="o",
                    color=cvd_color,
                    ecolor="#a3d9c9",
                    capsize=2,
                    markersize=4,
                    label="CVD" if i == 0 else "",
                )

            # Formatting
            ax.set_yticks(y_pos)
            ax.set_yticklabels(df_chunk["label"], fontsize=9)
            ax.axvline(
                x=0, color="red", linestyle="--", alpha=0.5
            )  # Log RR = 0 is RR = 1
            ax.set_xlabel("Log Risk Ratio (95% CI)", fontsize=10)

            # Grid: both x and y
            ax.grid(True, linestyle="--", alpha=0.3)

            # Set ylim to remove excessive whitespace
            # y_pos ranges from 0 to len(df_chunk)-1
            ax.set_ylim(-1, len(df_chunk))

        # Legend (Global)
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="black",
                label="Semaglutide",
                markersize=8,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#66c2a5",
                label="CVD",
                markersize=8,
            ),
        ]
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            ncol=2,
            bbox_to_anchor=(0.5, 0.005),
        )

        # Save
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / f"combined_forest_plot_log_rr_{part_name}.png"
        fig.tight_layout(rect=[0, 0.03, 1, 1])  # Make room for legend
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to: {output_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
