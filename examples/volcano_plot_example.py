"""
Volcano Plot Example

This script demonstrates how to create volcano plots to visualize
risk differences and their statistical significance across multiple outcomes.
"""

import sys
import os

# Add parent directory to path to import trace modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import matplotlib.pyplot as plt
from trace.plotting import adjust_pvalues, prepare_volcano_data, volcano_plot_per_method

# -----------------------------
# Create example data
# -----------------------------
# This example shows results from two methods (IPW and TMLE) applied to
# multiple outcomes with varying effect sizes and p-values.

example_data = pd.DataFrame(
    {
        "method": ["IPW", "IPW", "IPW", "TMLE", "TMLE", "TMLE"],
        "outcome": ["A01AA", "B02BB", "C03CC", "A01AA", "B02BB", "C03CC"],
        "RD": [0.012, -0.015, 0.004, 0.011, -0.022, 0.001],
        "p_value": [0.003, 0.12, 0.40, 2e-5, 0.049, 0.80],
    }
)

print("=" * 80)
print("EXAMPLE: Volcano Plot Creation")
print("=" * 80)
print("\nInput data:")
print(example_data)
print("\nData description:")
print("  - 2 methods: IPW and TMLE")
print("  - 3 outcomes per method: A01AA, B02BB, C03CC")
print("  - RD: Risk difference (positive = treatment increases risk)")
print("  - p_value: Unadjusted p-value from statistical test")

# -----------------------------
# Optional: Create nicer labels for outcomes
# -----------------------------
# You can provide a dictionary to map outcome codes to more readable names
label_map = {
    "A01AA": "Outcome A (nice name)",
    "B02BB": "Outcome B (nice name)",
    # Leave C03CC unmapped to show it uses the original name
}

# -----------------------------
# Prepare data for volcano plot
# -----------------------------
# This step:
# 1. Adjusts p-values for multiple testing (default: Benjamini-Hochberg)
# 2. Computes -log10(p) for visualization
# 3. Returns a standardized DataFrame

print("\n" + "=" * 80)
print("DATA PREPARATION")
print("=" * 80)

volcano_data = prepare_volcano_data(
    example_data,
    rd_col="RD",
    p_col="p_value",
    method_col="method",
    outcome_col="outcome",
    adjust="bh",  # Options: 'bh' (Benjamini-Hochberg), 'bonferroni', 'none'
    adjust_per="by_method",  # Options: 'by_method' (adjust within each method), 'global'
)

print("\nPrepared volcano data:")
print(volcano_data)
print("\nNew columns added:")
print("  - q_value: Adjusted p-value (controls False Discovery Rate)")
print("  - neglog10p: -log10(p_value) for y-axis of volcano plot")

# Show effect of adjustment
print("\nEffect of Benjamini-Hochberg adjustment:")
for _, row in volcano_data.iterrows():
    sig_status = "significant" if row["q_value"] < 0.05 else "not significant"
    print(
        f"  {row['method']:5s} {row['outcome']:5s}: "
        f"p={row['p_value']:.4f} → q={row['q_value']:.4f} ({sig_status})"
    )

# -----------------------------
# Create volcano plot
# -----------------------------
print("\n" + "=" * 80)
print("CREATING VOLCANO PLOT")
print("=" * 80)

fig, axes = volcano_plot_per_method(
    volcano_data,
    alpha=0.05,  # Significance threshold (q < 0.05)
    method_col="method",
    outcome_col="outcome",
    label_map=label_map,  # Use custom labels for some outcomes
    max_labels_per_panel=5,  # Annotate up to 5 top hits per panel
    figsize_per_panel=(6, 4),  # Size of each panel in inches
    point_size=18,  # Marker size
    sig_color=None,  # Use matplotlib default colors
    ns_color=None,
)

print("\nPlot created successfully!")
print("\nPlot interpretation:")
print("  - X-axis: Risk difference (RD)")
print("  - Y-axis: -log10(p-value) — higher means more significant")
print("  - Horizontal dashed line: Significance threshold (-log10(0.05) ≈ 1.3)")
print("  - Vertical dashed line: Null hypothesis (RD = 0)")
print("  - Red points: Significant after multiple testing correction (q < α)")
print("  - Blue points: Not significant after correction (q ≥ α)")
print("  - Top hits are labeled with outcome names")

# -----------------------------
# Customization examples
# -----------------------------
print("\n" + "=" * 80)
print("CUSTOMIZATION OPTIONS")
print("=" * 80)

print("\nYou can customize the plot by:")
print("  1. Changing adjustment method:")
print("     - adjust='bh': Benjamini-Hochberg (controls FDR, less conservative)")
print("     - adjust='bonferroni': Bonferroni (controls FWER, more conservative)")
print("     - adjust='none': No adjustment (not recommended)")
print()
print("  2. Changing adjustment scope:")
print("     - adjust_per='by_method': Adjust within each method separately")
print("     - adjust_per='global': Adjust across all tests globally")
print()
print("  3. Visual customization:")
print("     - alpha: Change significance threshold (e.g., 0.01 for stricter)")
print("     - sig_color, ns_color: Custom colors for points")
print("     - point_size: Adjust marker size")
print("     - max_labels_per_panel: Show more or fewer labels")
print("     - label_map: Provide readable names for outcomes")

# -----------------------------
# Example: More conservative adjustment
# -----------------------------
print("\n" + "=" * 80)
print("EXAMPLE: BONFERRONI ADJUSTMENT")
print("=" * 80)

volcano_data_bonf = prepare_volcano_data(
    example_data,
    adjust="bonferroni",
    adjust_per="global",
)

print("\nBonferroni-adjusted q-values (more conservative):")
for _, row in volcano_data_bonf.iterrows():
    sig_status = "significant" if row["q_value"] < 0.05 else "not significant"
    print(
        f"  {row['method']:5s} {row['outcome']:5s}: "
        f"p={row['p_value']:.4f} → q={row['q_value']:.4f} ({sig_status})"
    )

print("\nNote: Bonferroni is more conservative — fewer outcomes are significant")

# Show the plot
plt.show()

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(
    """
Volcano plots are useful for:
  1. Visualizing many outcomes simultaneously
  2. Identifying outcomes with large AND significant effects
  3. Comparing results across different methods
  4. Quality control (checking for outliers or patterns)

Best practices:
  - Always adjust for multiple testing
  - Use Benjamini-Hochberg for exploratory analysis (balances power and control)
  - Use Bonferroni when false positives are very costly
  - Label and investigate top hits
  - Consider biological/clinical significance, not just statistical
"""
)
