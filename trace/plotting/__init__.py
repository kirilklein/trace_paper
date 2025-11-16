"""
Plotting Package

This package provides visualization functions for causal inference analysis,
with a focus on volcano plots for displaying risk differences and their
statistical significance, as well as circular plots for ATC group visualization.
"""

from .circle import plot_circle
from .volcano import (
    adjust_pvalues,
    prepare_volcano_data,
    volcano_plot_per_method,
)

__all__ = [
    "adjust_pvalues",
    "plot_circle",
    "prepare_volcano_data",
    "volcano_plot_per_method",
]
