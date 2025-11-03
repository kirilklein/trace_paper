"""
Script to create volcano plot from semaglutide data
"""
import pandas as pd
import matplotlib.pyplot as plt

from trace.statistics import compute_rd_pvalues
from trace.plotting.volcano import prepare_volcano_data, volcano_plot_per_method

# Load data
df = pd.read_csv('data/semaglutide/combined_estimatest.txt', index_col=0)

print(f"Loaded {len(df)} rows")
print(f"Methods: {df['method'].unique()}")
print(f"Outcomes: {df['outcome'].nunique()} unique")
print(f"Runs: {df['run_id'].nunique()} unique")

# Filter to methods with arm-level confidence intervals (IPW and TMLE)
# RD and RR methods don't have the individual arm CIs needed for our approach
df_with_arms = df[df['method'].isin(['IPW', 'TMLE'])].copy()
print(f"\nFiltered to IPW and TMLE: {len(df_with_arms)} rows")

# Check for required columns
required_cols = ['effect_1', 'effect_0', 'effect_1_CI95_lower', 'effect_1_CI95_upper',
                 'effect_0_CI95_lower', 'effect_0_CI95_upper']
print(f"\nChecking for required columns...")
for col in required_cols:
    n_missing = df_with_arms[col].isna().sum()
    print(f"  {col}: {n_missing} missing values")

# Compute pooled RD p-values across runs for each method-outcome combination
print("\nComputing pooled risk differences and p-values...")
df_pooled = compute_rd_pvalues(df_with_arms, group_cols=['method', 'outcome'])
print(f"Computed {len(df_pooled)} method-outcome combinations")

# Prepare volcano data with Benjamini-Hochberg adjustment
print("\nPreparing volcano plot data...")
df_volcano = prepare_volcano_data(
    df_pooled,
    rd_col="RD",
    p_col="p_value",
    method_col="method",
    outcome_col="outcome",
    adjust="bh",
    adjust_per="by_method"
)

print(f"\nSummary statistics:")
for method in df_volcano['method'].unique():
    d = df_volcano[df_volcano['method'] == method]
    n_sig = (d['q_value'] < 0.05).sum()
    print(f"  {method}: {len(d)} outcomes, {n_sig} significant (q < 0.05)")

# Create volcano plot
print("\nCreating volcano plot...")
fig, axes = volcano_plot_per_method(
    df_volcano,
    alpha=0.05,
    method_col="method",
    outcome_col="outcome",
    max_labels_per_panel=10,
    figsize_per_panel=(7, 5),
    point_size=20,
    sig_color='#d62728',  # red
    ns_color='#7f7f7f'    # gray
)

# Save figure
output_path = 'figures/volcano_plot.png'
fig.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nSaved plot to: {output_path}")

# Also save as PDF
output_path_pdf = 'figures/volcano_plot.pdf'
fig.savefig(output_path_pdf, bbox_inches='tight')
print(f"Saved plot to: {output_path_pdf}")

plt.show()

print("\nDone!")

