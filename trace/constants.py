"""Constants and configuration mappings for the trace package."""

from __future__ import annotations

# Column rename mapping for prevalence statistics
PREVALENCE_COLUMN_RENAME_MAP = {
    "prevalence_overall_total": "prevalence_total",
    "prevalence_overall_treated": "prevalence_treated",
    "prevalence_overall_untreated": "prevalence_untreated",
    "population_total": "population_total",
    "population_treated": "population_treated",
    "population_untreated": "population_untreated",
    "outcome_events_total": "outcome_events_total",
    "outcome_events_treated": "outcome_events_treated",
    "outcome_events_untreated": "outcome_events_untreated",
    "prevalence_mean_total": "prevalence_mean_total",
    "prevalence_mean_treated": "prevalence_mean_treated",
    "prevalence_mean_untreated": "prevalence_mean_untreated",
    "prevalence_std_total": "prevalence_std_total",
    "prevalence_std_treated": "prevalence_std_treated",
    "prevalence_std_untreated": "prevalence_std_untreated",
    "prevalence_median_total": "prevalence_median_total",
    "prevalence_median_treated": "prevalence_median_treated",
    "prevalence_median_untreated": "prevalence_median_untreated",
    "run_count_total": "prevalence_run_count_total",
    "run_count_treated": "prevalence_run_count_treated",
    "run_count_untreated": "prevalence_run_count_untreated",
}

# Default alpha threshold for significance testing
DEFAULT_ALPHA = 0.05

# Methods that provide arm-level confidence intervals
METHODS_WITH_ARMS = ("IPW", "TMLE")
