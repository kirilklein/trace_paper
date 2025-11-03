"""
Script to create volcano plot from semaglutide data
"""

import pandas as pd
import matplotlib.pyplot as plt

from trace.statistics import compute_rd_pvalues
from trace.plotting.volcano import prepare_volcano_data, volcano_plot_per_method

# Load data
df = pd.read_csv("data/semaglutide/combined_estimatest.txt", index_col=0)

print(f"Loaded {len(df)} rows")
print(f"Methods: {df['method'].unique()}")
print(f"Outcomes: {df['outcome'].nunique()} unique")
print(f"Runs: {df['run_id'].nunique()} unique")

# Filter to methods with arm-level confidence intervals (IPW and TMLE)
# RD and RR methods don't have the individual arm CIs needed for our approach
df_with_arms = df[df["method"].isin(["IPW", "TMLE"])].copy()
print(f"\nFiltered to IPW and TMLE: {len(df_with_arms)} rows")

# Check for required columns
required_cols = [
    "effect_1",
    "effect_0",
    "effect_1_CI95_lower",
    "effect_1_CI95_upper",
    "effect_0_CI95_lower",
    "effect_0_CI95_upper",
]
print(f"\nChecking for required columns...")
for col in required_cols:
    n_missing = df_with_arms[col].isna().sum()
    print(f"  {col}: {n_missing} missing values")

# Compute pooled RD p-values across runs for each method-outcome combination
# Choose pooling method: "inverse_variance_arms", "rubins_rules", or "random_effects_dl"
pooling_method = "random_effects_dl"  # Most conservative - accounts for heterogeneity
print(
    f"\nComputing pooled risk differences and p-values using '{pooling_method}' method..."
)
df_pooled = compute_rd_pvalues(
    df_with_arms,
    group_cols=["method", "outcome"],
    pooling_method=pooling_method,
    verbose=True,
)
print(f"Computed {len(df_pooled)} method-outcome combinations")

# Check for tau2 (heterogeneity) if using random effects
if "tau2" in df_pooled.columns:
    print(f"\nHeterogeneity statistics (tau²):")
    print(f"  Mean tau²: {df_pooled['tau2'].mean():.4e}")
    print(f"  Median tau²: {df_pooled['tau2'].median():.4e}")
    print(f"  Max tau²: {df_pooled['tau2'].max():.4e}")
    n_heterogeneous = (df_pooled["tau2"] > 0.01).sum()
    print(f"  Cases with substantial heterogeneity (tau² > 0.01): {n_heterogeneous}")

# ============================================================================
# DIAGNOSTICS: P-value and SE distributions
# ============================================================================
print("\n" + "=" * 70)
print("DIAGNOSTIC: P-value Distribution Analysis")
print("=" * 70)

print(f"\nP-value statistics:")
print(f"  Min p-value: {df_pooled['p_value'].min():.2e}")
print(f"  Max p-value: {df_pooled['p_value'].max():.2e}")
print(f"  Median p-value: {df_pooled['p_value'].median():.2e}")
print(f"  Mean p-value: {df_pooled['p_value'].mean():.2e}")

# Check how many are at or near the floor
n_at_floor = (df_pooled["p_value"] <= 1e-300).sum()
n_below_1e100 = (df_pooled["p_value"] < 1e-100).sum()
n_below_1e50 = (df_pooled["p_value"] < 1e-50).sum()
n_below_1e20 = (df_pooled["p_value"] < 1e-20).sum()
n_below_1e10 = (df_pooled["p_value"] < 1e-10).sum()

print(f"\nExtreme p-values:")
print(f"  <= 1e-300: {n_at_floor} ({100 * n_at_floor / len(df_pooled):.1f}%)")
print(f"  < 1e-100: {n_below_1e100} ({100 * n_below_1e100 / len(df_pooled):.1f}%)")
print(f"  < 1e-50: {n_below_1e50} ({100 * n_below_1e50 / len(df_pooled):.1f}%)")
print(f"  < 1e-20: {n_below_1e20} ({100 * n_below_1e20 / len(df_pooled):.1f}%)")
print(f"  < 1e-10: {n_below_1e10} ({100 * n_below_1e10 / len(df_pooled):.1f}%)")

print(f"\nZ-statistic statistics:")
print(f"  Min z: {df_pooled['z'].min():.2f}")
print(f"  Max z: {df_pooled['z'].max():.2f}")
print(f"  Median |z|: {df_pooled['z'].abs().median():.2f}")
print(f"  Mean |z|: {df_pooled['z'].abs().mean():.2f}")

n_extreme_z = (df_pooled["z"].abs() > 30).sum()
n_very_extreme_z = (df_pooled["z"].abs() > 50).sum()
print(f"\nExtreme z-statistics:")
print(f"  |z| > 30: {n_extreme_z} ({100 * n_extreme_z / len(df_pooled):.1f}%)")
print(
    f"  |z| > 50: {n_very_extreme_z} ({100 * n_very_extreme_z / len(df_pooled):.1f}%)"
)

print(f"\nSE_RD statistics:")
print(f"  Min SE_RD: {df_pooled['SE_RD'].min():.2e}")
print(f"  Max SE_RD: {df_pooled['SE_RD'].max():.2e}")
print(f"  Median SE_RD: {df_pooled['SE_RD'].median():.2e}")
print(f"  Mean SE_RD: {df_pooled['SE_RD'].mean():.2e}")

n_tiny_se = (df_pooled["SE_RD"] < 1e-6).sum()
n_small_se = (df_pooled["SE_RD"] < 1e-4).sum()
print(f"\nSmall SE_RD values:")
print(f"  < 1e-6: {n_tiny_se} ({100 * n_tiny_se / len(df_pooled):.1f}%)")
print(f"  < 1e-4: {n_small_se} ({100 * n_small_se / len(df_pooled):.1f}%)")

if "p1_hat" in df_pooled.columns and "p0_hat" in df_pooled.columns:
    print(f"\nProbability estimates (p1_hat, p0_hat):")
    n_extreme_p1 = ((df_pooled["p1_hat"] < 0.001) | (df_pooled["p1_hat"] > 0.999)).sum()
    n_extreme_p0 = ((df_pooled["p0_hat"] < 0.001) | (df_pooled["p0_hat"] > 0.999)).sum()
    print(
        f"  p1_hat < 0.001 or > 0.999: {n_extreme_p1} ({100 * n_extreme_p1 / len(df_pooled):.1f}%)"
    )
    print(
        f"  p0_hat < 0.001 or > 0.999: {n_extreme_p0} ({100 * n_extreme_p0 / len(df_pooled):.1f}%)"
    )
    print(
        f"  p1_hat range: [{df_pooled['p1_hat'].min():.4f}, {df_pooled['p1_hat'].max():.4f}]"
    )
    print(
        f"  p0_hat range: [{df_pooled['p0_hat'].min():.4f}, {df_pooled['p0_hat'].max():.4f}]"
    )
else:
    print(
        f"\nProbability estimates (p1_hat, p0_hat): Not available for this pooling method"
    )

# Check arm-level pooled SEs if available
if "eta1_pooled_se" in df_pooled.columns:
    print(f"\nArm-level SE statistics (after pooling):")
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

print("\n" + "=" * 70)
print("DIAGNOSTIC: Detailed inspection of most extreme cases")
print("=" * 70)

# Show top 5 cases with smallest p-values
print("\nTop 5 smallest p-values:")
display_cols = ["method", "outcome", "RD", "SE_RD", "z", "p_value"]
if "p1_hat" in df_pooled.columns:
    display_cols.extend(["p1_hat", "p0_hat"])
extreme_cases = df_pooled.nsmallest(5, "p_value")[display_cols]
for idx, row in extreme_cases.iterrows():
    print(f"\n  {row['method']} - {row['outcome'][:50]}")
    print(f"    RD={row['RD']:.4f}, SE_RD={row['SE_RD']:.2e}")
    print(f"    z={row['z']:.2f}, p={row['p_value']:.2e}")
    if "p1_hat" in row:
        print(f"    p1_hat={row['p1_hat']:.4f}, p0_hat={row['p0_hat']:.4f}")

    # Check original data for this case
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
        # Check CI widths
        ci_width_1 = (orig["effect_1_CI95_upper"] - orig["effect_1_CI95_lower"]).mean()
        ci_width_0 = (orig["effect_0_CI95_upper"] - orig["effect_0_CI95_lower"]).mean()
        print(
            f"      Mean CI width: effect_1={ci_width_1:.4f}, effect_0={ci_width_0:.4f}"
        )

print("\n" + "=" * 70)
print("DIAGNOSTIC: Deep dive into most extreme case")
print("=" * 70)

# Take the most extreme case and trace through the entire calculation
if len(df_pooled) > 0:
    extreme_idx = df_pooled["p_value"].idxmin()
    extreme_row = df_pooled.loc[extreme_idx]

    print(f"\nMost extreme case:")
    print(f"  Method: {extreme_row['method']}")
    print(f"  Outcome: {extreme_row['outcome'][:80]}")
    print(f"  Final RD: {extreme_row['RD']:.6f}")
    print(f"  Final SE_RD: {extreme_row['SE_RD']:.6e}")
    print(f"  Final z: {extreme_row['z']:.2f}")
    print(f"  Final p-value: {extreme_row['p_value']:.6e}")

    # Get original data for this case
    mask = (df_with_arms["method"] == extreme_row["method"]) & (
        df_with_arms["outcome"] == extreme_row["outcome"]
    )
    orig_data = df_with_arms[mask].copy()

    print(f"\n  Original data ({len(orig_data)} runs):")
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

        # Compute CI widths
        ci_width_1 = row["effect_1_CI95_upper"] - row["effect_1_CI95_lower"]
        ci_width_0 = row["effect_0_CI95_upper"] - row["effect_0_CI95_lower"]
        print(f"      CI widths: effect_1={ci_width_1:.6f}, effect_0={ci_width_0:.6f}")

    # Now manually trace through the calculation
    print(f"\n  Step-by-step calculation:")

    # Step 1: Logit transformation
    from trace.statistics import logit, se_from_prob_ci_on_logit, inv_logit
    import numpy as np

    eta1_vals = []
    se_eta1_vals = []
    eta0_vals = []
    se_eta0_vals = []

    for idx, row in orig_data.iterrows():
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

    print(f"\n    After logit transformation (per run):")
    for i in range(len(eta1_vals)):
        print(
            f"      Run {i + 1}: eta1={eta1_vals[i]:.4f} (se={se_eta1_vals[i]:.4e}), "
            f"eta0={eta0_vals[i]:.4f} (se={se_eta0_vals[i]:.4e})"
        )

    # Step 2: Inverse-variance pooling
    eta1_arr = np.array(eta1_vals)
    se_eta1_arr = np.array(se_eta1_vals)
    eta0_arr = np.array(eta0_vals)
    se_eta0_arr = np.array(se_eta0_vals)

    # Remove any non-finite values
    finite_mask_1 = np.isfinite(eta1_arr) & np.isfinite(se_eta1_arr)
    finite_mask_0 = np.isfinite(eta0_arr) & np.isfinite(se_eta0_arr)

    w1 = 1.0 / (se_eta1_arr[finite_mask_1] ** 2)
    eta1_pooled = np.sum(w1 * eta1_arr[finite_mask_1]) / np.sum(w1)
    se_eta1_pooled = np.sqrt(1.0 / np.sum(w1))

    w0 = 1.0 / (se_eta0_arr[finite_mask_0] ** 2)
    eta0_pooled = np.sum(w0 * eta0_arr[finite_mask_0]) / np.sum(w0)
    se_eta0_pooled = np.sqrt(1.0 / np.sum(w0))

    print(f"\n    After inverse-variance pooling:")
    print(f"      eta1_pooled={eta1_pooled:.4f} (se={se_eta1_pooled:.4e})")
    print(f"      eta0_pooled={eta0_pooled:.4f} (se={se_eta0_pooled:.4e})")
    print(f"      Weights (1/SE²): arm1={w1}, arm0={w0}")
    print(f"      Sum of weights: arm1={np.sum(w1):.2e}, arm0={np.sum(w0):.2e}")

    # Step 3: Transform back to probability scale and compute RD
    p1_pooled = inv_logit(eta1_pooled)
    p0_pooled = inv_logit(eta0_pooled)
    rd_pooled = p1_pooled - p0_pooled

    print(f"\n    After inverse logit:")
    print(f"      p1_pooled={p1_pooled:.6f}")
    print(f"      p0_pooled={p0_pooled:.6f}")
    print(f"      RD_pooled={rd_pooled:.6f}")

    # Step 4: Delta method variance
    var_term1 = (p1_pooled * (1 - p1_pooled)) ** 2 * (se_eta1_pooled**2)
    var_term0 = (p0_pooled * (1 - p0_pooled)) ** 2 * (se_eta0_pooled**2)
    var_rd = var_term1 + var_term0
    se_rd = np.sqrt(var_rd)

    print(f"\n    Delta method variance calculation:")
    print(f"      dp1/deta1 = p1*(1-p1) = {p1_pooled * (1 - p1_pooled):.6f}")
    print(f"      dp0/deta0 = p0*(1-p0) = {p0_pooled * (1 - p0_pooled):.6f}")
    print(f"      Var(p1) = (dp/deta)² * Var(eta1) = {var_term1:.6e}")
    print(f"      Var(p0) = (dp/deta)² * Var(eta0) = {var_term0:.6e}")
    print(f"      Var(RD) = Var(p1) + Var(p0) = {var_rd:.6e}")
    print(f"      SE(RD) = {se_rd:.6e}")

    # Step 5: z-statistic and p-value
    z_stat = rd_pooled / se_rd
    from scipy.stats import norm

    p_val = 2 * (1 - norm.cdf(np.abs(z_stat)))

    print(f"\n    Final inference:")
    print(f"      z = RD / SE_RD = {rd_pooled:.6f} / {se_rd:.6e} = {z_stat:.2f}")
    print(f"      p-value = 2 * (1 - Phi(|z|)) = {p_val:.6e}")
    print(f"      -log10(p) = {-np.log10(max(p_val, 1e-300)):.2f}")

    # Check if this matches the pooled result
    print(f"\n    Verification against df_pooled:")
    print(f"      Match RD: {np.isclose(rd_pooled, extreme_row['RD'])}")
    print(f"      Match SE_RD: {np.isclose(se_rd, extreme_row['SE_RD'])}")
    print(f"      Match z: {np.isclose(z_stat, extreme_row['z'])}")
    print(
        f"      Match p_value: {np.isclose(p_val, extreme_row['p_value']) if p_val > 0 else 'both ~0'}"
    )

print("\n" + "=" * 70)

# Prepare volcano data with Benjamini-Hochberg adjustment
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

print(f"\nSummary statistics:")
for method in df_volcano["method"].unique():
    d = df_volcano[df_volcano["method"] == method]
    n_sig = (d["q_value"] < 0.05).sum()
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
    sig_color="#d62728",  # red
    ns_color="#7f7f7f",  # gray
)

# Save figure
output_path = "figures/volcano_plot.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"\nSaved plot to: {output_path}")

# Also save as PDF
output_path_pdf = "figures/volcano_plot.pdf"
fig.savefig(output_path_pdf, bbox_inches="tight")
print(f"Saved plot to: {output_path_pdf}")

plt.show()

# ============================================================================
# Create truncated version excluding clipped p-values (neglog10p >= 300)
# ============================================================================
print("\n" + "=" * 70)
print("Creating truncated volcano plot (excluding clipped p-values)")
print("=" * 70)

# Filter out entries with neglog10p at the ceiling (300 = -log10(1e-300))
# Use 299 as threshold to catch floating point issues
df_volcano_truncated = df_volcano[df_volcano["neglog10p"] < 299].copy()

n_excluded = len(df_volcano) - len(df_volcano_truncated)
pct_excluded = 100 * n_excluded / len(df_volcano)
print(f"\nExcluded {n_excluded} outcomes with clipped p-values ({pct_excluded:.1f}%)")
print(f"Remaining outcomes: {len(df_volcano_truncated)}")

for method in df_volcano_truncated["method"].unique():
    d = df_volcano_truncated[df_volcano_truncated["method"] == method]
    n_sig = (d["q_value"] < 0.05).sum()
    print(f"  {method}: {len(d)} outcomes, {n_sig} significant (q < 0.05)")

# Create truncated volcano plot
print("\nCreating truncated volcano plot...")
fig_trunc, axes_trunc = volcano_plot_per_method(
    df_volcano_truncated,
    alpha=0.05,
    method_col="method",
    outcome_col="outcome",
    max_labels_per_panel=10,
    figsize_per_panel=(7, 5),
    point_size=20,
    sig_color="#d62728",  # red
    ns_color="#7f7f7f",  # gray
)

# Save truncated figure
output_path_trunc = "figures/volcano_plot_truncated.png"
fig_trunc.savefig(output_path_trunc, dpi=300, bbox_inches="tight")
print(f"Saved truncated plot to: {output_path_trunc}")

# Also save as PDF
output_path_trunc_pdf = "figures/volcano_plot_truncated.pdf"
fig_trunc.savefig(output_path_trunc_pdf, bbox_inches="tight")
print(f"Saved truncated plot to: {output_path_trunc_pdf}")

plt.show()

print("\nDone!")
