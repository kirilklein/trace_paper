"""Helper functions for the volcano plot script."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


def print_dataset_overview(df: pd.DataFrame) -> None:
    """Print basic dataset statistics."""
    print(f"Loaded {len(df)} rows")
    print(f"Methods: {df['method'].unique()}")
    print(f"Outcomes: {df['outcome'].nunique()} unique")
    print(f"Runs: {df['run_id'].nunique()} unique")


def ensure_required_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """Print a diagnostic summary of missing values for required columns."""
    print("\nChecking for required columns...")
    for col in required:
        n_missing = df[col].isna().sum()
        print(f"  {col}: {n_missing} missing values")


def print_significance_confusion_matrix(
    df: pd.DataFrame,
    *,
    methods: tuple[str, str],
    method_col: str = "method",
    outcome_col: str = "outcome",
    q_col: str = "q_value",
    alpha: float = 0.05,
    return_confusion: bool = False,
) -> tuple[pd.DataFrame, float, int] | None:
    """Print a confusion matrix comparing significance across two methods.

    When ``return_confusion`` is ``True``, the function additionally returns a
    tuple containing the formatted confusion matrix, the agreement proportion,
    and the number of overlapping outcomes used to build the matrix.
    """
    if len(methods) != 2:
        raise ValueError(
            "Exactly two methods are required to build the confusion matrix."
        )

    df_subset = (
        df[df[method_col].isin(methods)][[outcome_col, method_col, q_col]]
        .dropna(subset=[q_col])
        .copy()
    )

    if df_subset.empty:
        print("  No outcomes with valid q-values for the requested methods.")
        return None

    df_subset["is_significant"] = df_subset[q_col] < alpha

    pivot = df_subset.pivot_table(
        index=outcome_col,
        columns=method_col,
        values="is_significant",
        aggfunc="first",
    )

    missing = [method for method in methods if method not in pivot.columns]
    if missing:
        print(
            "  Missing methods in results: "
            + ", ".join(missing)
            + ". Not enough overlap to build the confusion matrix."
        )
        return None

    pivot = pivot.dropna(subset=list(methods))

    if pivot.empty:
        print("  No overlapping outcomes between the selected methods.")
        return None

    method_a, method_b = methods
    confusion = pd.crosstab(
        pivot[method_a],
        pivot[method_b],
        rownames=[f"{method_a} significant"],
        colnames=[f"{method_b} significant"],
        dropna=False,
    )

    confusion = confusion.reindex(
        index=[False, True], columns=[False, True], fill_value=0
    )

    formatted = confusion.rename(
        index={False: "No", True: "Yes"}, columns={False: "No", True: "Yes"}
    )
    print(formatted.to_string())

    agreement = (pivot[method_a] == pivot[method_b]).mean()
    print(f"  Agreement: {agreement * 100:.1f}% (n={len(pivot)})")

    if return_confusion:
        return formatted.copy(), agreement, len(pivot)

    return None


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
    # Avoid duplicating effect/p-values that already exist in volcano_df
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
    """Create output directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
