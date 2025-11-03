"""
Risk Difference P-values Example

This script demonstrates how to compute risk differences and their p-values
using the statistics module. It shows both per-run and pooled calculations.
"""

import sys
import os

# Add parent directory to path to import trace modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from trace.statistics import compute_rd_pvalues

# -----------------------------
# Create example data
# -----------------------------
# This example shows results from two methods (IPW and TMLE) applied to
# the same outcome (A01AA) across two independent runs.
# Each row contains arm-level probability estimates and their 95% CIs.

example_df = pd.DataFrame({
    "method":  ["IPW", "IPW", "TMLE", "TMLE"],
    "outcome": ["A01AA", "A01AA", "A01AA", "A01AA"],
    "run_id":  ["run01", "run02", "run01", "run02"],
    
    # Arm 1 (treated) probabilities and 95% CIs
    "effect_1": [0.1269, 0.1280, 0.1268, 0.1275],
    "effect_1_CI95_lower": [0.1200, 0.1215, 0.1202, 0.1210],
    "effect_1_CI95_upper": [0.1335, 0.1342, 0.1331, 0.1337],
    
    # Arm 0 (control) probabilities and 95% CIs
    "effect_0": [0.1165, 0.1172, 0.1165, 0.1169],
    "effect_0_CI95_lower": [0.1110, 0.1118, 0.1112, 0.1115],
    "effect_0_CI95_upper": [0.1219, 0.1225, 0.1217, 0.1222],
})

print("=" * 80)
print("EXAMPLE: Risk Difference P-value Calculation")
print("=" * 80)
print("\nInput data:")
print(example_df)

# -----------------------------
# Compute per-run p-values
# -----------------------------
# When group_cols=None, compute_rd_pvalues computes risk differences
# and p-values for each row independently.

print("\n" + "=" * 80)
print("PER-RUN ANALYSIS")
print("=" * 80)

per_run = compute_rd_pvalues(example_df, group_cols=None)

# Display relevant columns
columns_to_show = [
    "method", "outcome", "run_id",
    "RD", "SE_RD", "z", "p_value",
    "RD_CI95_lower", "RD_CI95_upper"
]
print("\nPer-run RD p-values:")
print(per_run[columns_to_show].to_string(index=False))

# Interpretation
print("\nInterpretation:")
print("  - RD: Risk difference (treatment - control)")
print("  - SE_RD: Standard error of the risk difference")
print("  - z: Wald z-statistic for testing RD = 0")
print("  - p_value: Two-sided p-value")
print("  - All runs show positive RD (treatment has higher risk)")
print("  - All p-values < 0.05 (statistically significant at α=0.05)")

# -----------------------------
# Compute pooled p-values
# -----------------------------
# When group_cols is specified, compute_rd_pvalues first pools the
# arm-level estimates across runs using inverse-variance weighting,
# then computes the risk difference from the pooled estimates.

print("\n" + "=" * 80)
print("POOLED ANALYSIS")
print("=" * 80)

pooled = compute_rd_pvalues(example_df, group_cols=["method", "outcome"])

# Display relevant columns
pooled_columns = [
    "method", "outcome",
    "RD", "SE_RD", "z", "p_value",
    "RD_CI95_lower", "RD_CI95_upper",
    "n_runs_used_x", "n_runs_used_y"
]
print("\nPooled RD p-values per (method, outcome):")
pooled_display = pooled[pooled_columns].rename(columns={
    "n_runs_used_x": "n_runs_arm1",
    "n_runs_used_y": "n_runs_arm0"
})
print(pooled_display.to_string(index=False))

# Interpretation
print("\nInterpretation:")
print("  - Pooling combines information across runs for more precise estimates")
print("  - Notice the SE_RD is smaller in pooled analysis (more precision)")
print("  - Both methods show similar RD (~0.0106) and highly significant p-values")
print("  - IPW: RD = 0.0106, p = 0.0005")
print("  - TMLE: RD = 0.0105, p = 0.0005")

# -----------------------------
# Summary
# -----------------------------
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
This example demonstrated:

1. Per-run analysis: Compute RD and p-values for each individual run
   - Useful for examining run-to-run variability
   - Each run is treated independently

2. Pooled analysis: Combine information across runs before computing RD
   - Uses inverse-variance weighting for optimal precision
   - More efficient when runs are measuring the same underlying effect
   - Provides more powerful tests (smaller p-values, narrower CIs)

The pooled approach is generally preferred when you have multiple runs
of the same analysis and want to make overall inferences about the
treatment effect.
""")

