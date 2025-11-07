"""Utility functions for loading reference data files used across TRACE."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

_PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_ATC_DICT_PATH = _PACKAGE_DIR.parent / "data" / "atc_dict.txt"
DEFAULT_COMBINED_STATS_PATH = (
    _PACKAGE_DIR.parent / "data" / "semaglutide" / "combined_stats.txt"
)


def _iter_atc_lines(lines: Iterable[str]) -> Iterable[str]:
    """Yield non-empty, non-header lines from an iterable."""

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith("code") and " " not in line:
            # Lines like "Code" are not valid entries.
            continue
        if line.lower().startswith("code "):
            # Skip header line "Code Text".
            continue
        yield line


def load_atc_dictionary(
    path: str | Path | None = None,
    *,
    strip_prefix: str | None = "M",
    keep_empty_codes: bool = False,
) -> Dict[str, str]:
    """Load ATC reference data from ``atc_dict.txt`` into a dictionary.

    Parameters
    ----------
    path:
        Location of the ATC dictionary text file. If omitted, the default file
        within the repository's ``data`` directory is used.
    strip_prefix:
        Optional prefix to strip from the ATC codes (defaults to ``"M"``).
        Pass ``None`` or an empty string to disable stripping.
    keep_empty_codes:
        Whether to include entries that become empty after stripping the
        prefix. These represent root-level classification rows. Defaults to
        ``False`` (empty codes are skipped).

    Returns
    -------
    dict
        Mapping of ATC code (without the prefix) to its textual description.
    """

    source_path = Path(path) if path is not None else DEFAULT_ATC_DICT_PATH
    if not source_path.exists():
        raise FileNotFoundError(f"ATC dictionary file not found: {source_path}")

    atc_mapping: Dict[str, str] = {}
    with source_path.open(encoding="utf-8") as handle:
        for line in _iter_atc_lines(handle):
            try:
                code, description = line.split(maxsplit=1)
            except ValueError as error:
                raise ValueError(
                    f"Unable to parse line '{line}' in {source_path}."
                ) from error

            if strip_prefix:
                if code.startswith(strip_prefix):
                    code = code[len(strip_prefix) :]
                else:
                    code = code.lstrip(strip_prefix)

            if not code and not keep_empty_codes:
                continue

            atc_mapping[code] = description.strip()

    return atc_mapping


@dataclass
class PrevalenceStats:
    """Container with raw and aggregated outcome prevalence statistics."""

    raw: pd.DataFrame
    summary: pd.DataFrame


def load_prevalence_statistics(
    path: str | Path | None = None,
) -> PrevalenceStats:
    """Load and summarise prevalence counts from ``combined_stats.txt``.

    The returned :class:`PrevalenceStats` object contains both the raw table
    (with an added ``prevalence`` column per row) and a wide ``summary``
    dataframe with one row per outcome and status-specific aggregates.
    ``summary`` keeps metrics such as outcome counts, population totals, and
    prevalence measures (overall, mean, median, min, max, and standard
    deviation across runs).
    """

    source_path = Path(path) if path is not None else DEFAULT_COMBINED_STATS_PATH
    if not source_path.exists():
        raise FileNotFoundError(f"Prevalence statistics file not found: {source_path}")

    df = pd.read_csv(source_path)
    expected_columns = {"status", "No Outcome", "Outcome", "Total", "outcome", "run_id"}
    missing_cols = expected_columns.difference(df.columns)
    if missing_cols:
        raise ValueError(
            "Prevalence statistics missing required columns: "
            + ", ".join(sorted(missing_cols))
        )

    raw = df.copy()
    total_denominator = raw["Total"].replace(0, pd.NA)
    raw["prevalence"] = raw["Outcome"] / total_denominator

    grouped = (
        raw.groupby(["outcome", "status"], dropna=False)
        .agg(
            outcome_events=("Outcome", "sum"),
            population=("Total", "sum"),
            prevalence_mean=("prevalence", "mean"),
            prevalence_median=("prevalence", "median"),
            prevalence_min=("prevalence", "min"),
            prevalence_max=("prevalence", "max"),
            prevalence_std=("prevalence", "std"),
            run_count=("run_id", "nunique"),
        )
        .reset_index()
    )

    # Replace the lambda-derived column with deterministic calculations based on
    # the summed counts to avoid sensitivity to missing values.
    population = grouped["population"].replace(0, pd.NA)
    grouped["prevalence_overall"] = grouped["outcome_events"] / population

    summary = grouped.pivot(index="outcome", columns="status")
    summary.columns = [
        f"{metric}_{status}".lower()
        for metric, status in summary.columns.to_flat_index()
    ]
    summary = summary.reset_index().sort_values("outcome").reset_index(drop=True)

    return PrevalenceStats(raw=raw, summary=summary)


__all__ = [
    "load_atc_dictionary",
    "load_prevalence_statistics",
    "DEFAULT_ATC_DICT_PATH",
    "DEFAULT_COMBINED_STATS_PATH",
    "PrevalenceStats",
]
