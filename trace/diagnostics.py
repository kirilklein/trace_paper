"""Diagnostic utilities for analyzing pooled effect estimates and p-values."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

from trace.statistics import inv_logit, logit, se_from_prob_ci_on_logit


def print_pooled_diagnostics(df_pooled: pd.DataFrame) -> None:
    """Print detailed diagnostic statistics for pooled estimates."""
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

    # Check for SE column (different names for RD vs RR)
    se_col = None
    if "SE_RD" in df_pooled.columns:
        se_col = "SE_RD"
    elif "SE_log_RR" in df_pooled.columns:
        se_col = "SE_log_RR"

    if se_col:
        print(f"\n{se_col} statistics:")
        print(f"  Min {se_col}: {df_pooled[se_col].min():.2e}")
        print(f"  Max {se_col}: {df_pooled[se_col].max():.2e}")
        print(f"  Median {se_col}: {df_pooled[se_col].median():.2e}")
        print(f"  Mean {se_col}: {df_pooled[se_col].mean():.2e}")

        n_tiny_se = (df_pooled[se_col] < 1e-6).sum()
        n_small_se = (df_pooled[se_col] < 1e-4).sum()
        print(f"\nSmall {se_col} values:")
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
    """Print detailed inspection of the most extreme cases."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC: Detailed inspection of most extreme cases")
    print("=" * 70)

    print("\nTop 5 smallest p-values:")
    display_cols = ["method", "outcome", "p_value"]

    # Add effect column (RD or RR)
    if "RD" in df_pooled.columns:
        display_cols.insert(2, "RD")
        if "SE_RD" in df_pooled.columns:
            display_cols.insert(3, "SE_RD")
    elif "RR" in df_pooled.columns:
        display_cols.insert(2, "RR")
        if "log_RR" in df_pooled.columns:
            display_cols.insert(3, "log_RR")
        if "SE_log_RR" in df_pooled.columns:
            display_cols.insert(4, "SE_log_RR")

    if "z" in df_pooled.columns:
        display_cols.insert(-1, "z")

    if {"p1_hat", "p0_hat"}.issubset(df_pooled.columns):
        display_cols.extend(["p1_hat", "p0_hat"])

    extreme_cases = df_pooled.nsmallest(5, "p_value")[
        [col for col in display_cols if col in df_pooled.columns]
    ]

    for _, row in extreme_cases.iterrows():
        print(f"\n  {row['method']} - {row['outcome'][:50]}")

        if "RD" in row:
            print(f"    RD={row['RD']:.4f}", end="")
            if "SE_RD" in row:
                print(f", SE_RD={row['SE_RD']:.2e}", end="")
            print()
        elif "RR" in row:
            print(f"    RR={row['RR']:.4f}", end="")
            if "log_RR" in row:
                print(f", log_RR={row['log_RR']:.4f}", end="")
            if "SE_log_RR" in row:
                print(f", SE_log_RR={row['SE_log_RR']:.2e}", end="")
            print()

        if "z" in row:
            print(f"    z={row['z']:.2f}, p={row['p_value']:.2e}")
        else:
            print(f"    p={row['p_value']:.2e}")

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
    """Deep dive into the most extreme case (only for RD analysis with arm-level data)."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC: Deep dive into most extreme case")
    print("=" * 70)

    if df_pooled.empty:
        return

    # Only do deep dive for RD (requires specific arm-level pooling)
    if "RD" not in df_pooled.columns or "SE_RD" not in df_pooled.columns:
        print("\nDeep dive only available for Risk Difference analysis.")
        return

    extreme_idx = df_pooled["p_value"].idxmin()
    extreme_row = df_pooled.loc[extreme_idx]

    print("\nMost extreme case:")
    print(f"  Method: {extreme_row['method']}")
    print(f"  Outcome: {extreme_row['outcome'][:80]}")
    print(f"  Final RD: {extreme_row['RD']:.6f}")
    print(f"  Final SE_RD: {extreme_row['SE_RD']:.6e}")
    if "z" in extreme_row:
        print(f"  Final z: {extreme_row['z']:.2f}")
    print(f"  Final p-value: {extreme_row['p_value']:.6e}")

    mask = (df_with_arms["method"] == extreme_row["method"]) & (
        df_with_arms["outcome"] == extreme_row["outcome"]
    )
    orig_data = df_with_arms[mask].copy()

    print(f"\n  Original data ({len(orig_data)} runs):")

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
    if "z" in extreme_row:
        print(f"      Match z: {np.isclose(z_stat, extreme_row['z'])}")
    matches_p_value = (
        np.isclose(p_val, extreme_row["p_value"]) if p_val > 0 else "both ~0"
    )
    print(f"      Match p_value: {matches_p_value}")


def plot_std_comparison(
    df_pooled: pd.DataFrame,
    df_with_arms: pd.DataFrame,
    group_cols: tuple[str, str] = ("method", "outcome"),
) -> "plt.Figure":
    """
    Compare per-run arm-level logit SEs (median across runs) against pooled arm-level SEs.

    Returns a Matplotlib Figure with two panels (arm1, arm0) when available.
    """
    required_cols = {
        "effect_1_CI95_lower",
        "effect_1_CI95_upper",
        "effect_0_CI95_lower",
        "effect_0_CI95_upper",
        group_cols[0],
        group_cols[1],
    }
    missing = required_cols - set(df_with_arms.columns)
    if missing:
        raise ValueError(
            "plot_std_comparison missing required columns in df_with_arms: "
            + ", ".join(sorted(missing))
        )

    d = df_with_arms.copy()
    d["se_eta1"] = se_from_prob_ci_on_logit(
        d["effect_1_CI95_lower"], d["effect_1_CI95_upper"]
    )
    d["se_eta0"] = se_from_prob_ci_on_logit(
        d["effect_0_CI95_lower"], d["effect_0_CI95_upper"]
    )

    group_list = list(group_cols)
    per_group = (
        d.groupby(group_list, dropna=False)
        .agg(
            se_eta1_median=("se_eta1", "median"),
            se_eta0_median=("se_eta0", "median"),
            n_runs=("se_eta1", "count"),
        )
        .reset_index()
    )

    merged = per_group.merge(df_pooled, on=group_list, how="inner")

    panels = []
    if {"se_eta1_median", "eta1_pooled_se"}.issubset(merged.columns):
        panels.append(("Arm 1 (treatment)", "se_eta1_median", "eta1_pooled_se"))
    if {"se_eta0_median", "eta0_pooled_se"}.issubset(merged.columns):
        panels.append(("Arm 0 (control)", "se_eta0_median", "eta0_pooled_se"))

    if not panels:
        raise ValueError(
            "plot_std_comparison could not find pooled arm SEs in df_pooled "
            "(expected: eta1_pooled_se and/or eta0_pooled_se)."
        )

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5), dpi=120)
    if len(panels) == 1:
        axes = [axes]

    methods = merged[group_cols[0]].dropna().unique().tolist()
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {m: palette[i % len(palette)] for i, m in enumerate(methods)}

    for ax, (title, xcol, ycol) in zip(axes, panels):
        data = (
            merged[[*group_cols, xcol, ycol]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=[xcol, ycol])
        )
        if data.empty:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            continue

        x = data[xcol].values
        y = data[ycol].values
        xy_min = float(np.nanmin([np.nanmin(x), np.nanmin(y)]))
        xy_max = float(np.nanmax([np.nanmax(x), np.nanmax(y)]))
        pad = 0.05 * (xy_max - xy_min if xy_max > xy_min else 1.0)
        lo, hi = max(0.0, xy_min - pad), xy_max + pad
        ax.plot([lo, hi], [lo, hi], ls="--", color="#888888", lw=1, label="y = x")

        for m in methods:
            dm = data[data[group_cols[0]] == m]
            if dm.empty:
                continue
            ax.scatter(
                dm[xcol], dm[ycol], s=28, alpha=0.75, color=color_map[m], label=str(m)
            )

        r2 = np.nan
        if len(x) >= 2:
            with np.errstate(invalid="ignore"):
                r = np.corrcoef(x, y)[0, 1]
            r2 = float(r * r) if np.isfinite(r) else np.nan

        ax.set_title(
            f"{title}  (R² = {r2:.3f})" if np.isfinite(r2) else f"{title}  (R² = NA)"
        )
        ax.set_xlabel("Median per-run SE (logit)")
        ax.set_ylabel("Pooled SE (logit)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(True, ls=":", alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=False
        )
        fig.subplots_adjust(top=0.85)

    fig.tight_layout()
    return fig


def run_diagnostics(
    df_pooled: pd.DataFrame,
    df_with_arms: pd.DataFrame,
    effect_type: str = "RD",
    out_dir: str | None = None,
) -> None:
    """Run all diagnostic analyses on pooled results.

    Parameters
    ----------
    df_pooled : pd.DataFrame
        Pooled effect estimates with p-values
    df_with_arms : pd.DataFrame
        Original per-run arm-level estimates
    effect_type : str
        Type of effect being analyzed ("RD", "RR", or "log-RR")
    """
    print_pooled_diagnostics(df_pooled)
    print_extreme_cases(df_pooled, df_with_arms)

    # Deep dive only available for RD
    if effect_type == "RD":
        deep_dive_extreme_case(df_pooled, df_with_arms)

    # Std comparison figure (arm-level, logit scale)
    if out_dir:
        try:
            fig_std = plot_std_comparison(
                df_pooled, df_with_arms, group_cols=("method", "outcome")
            )
            out_path = f"{out_dir}/std_comparison_{effect_type.lower()}.png"
            fig_std.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig_std)
            print(f"\nSaved std comparison figure to: {out_path}")
        except Exception as err:
            print(f"\nSkipping std comparison figure: {err}")

    print("\n" + "=" * 70)
