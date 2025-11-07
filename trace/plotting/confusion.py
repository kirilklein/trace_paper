"""Confusion matrix plotting utilities."""

from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure


def plot_confusion_matrix(
    confusion_df: pd.DataFrame,
    agreement: float,
    n_overlap: int,
    *,
    method_a: str = "TMLE",
    method_b: str = "IPW",
    figsize: Tuple[float, float] = (6, 5),
    cmap: str = "Blues",
) -> Figure:
    """Create an sklearn-style confusion matrix heatmap.

    Parameters
    ----------
    confusion_df : pd.DataFrame
        2x2 confusion matrix with index/columns labeled "No"/"Yes"
    agreement : float
        Proportion of agreement between methods (0-1)
    n_overlap : int
        Number of overlapping outcomes
    method_a : str, optional
        Name of first method (y-axis), by default "TMLE"
    method_b : str, optional
        Name of second method (x-axis), by default "IPW"
    figsize : Tuple[float, float], optional
        Figure size, by default (6, 5)
    cmap : str, optional
        Colormap name, by default "Blues"

    Returns
    -------
    Figure
        Matplotlib figure with confusion matrix heatmap
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(confusion_df.values, cmap=cmap, aspect="auto")

    # Add colorbar
    plt.colorbar(im, ax=ax)

    # Set ticks and labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No", "Yes"])
    ax.set_yticklabels(["No", "Yes"])

    # Add text annotations
    for i in range(2):
        for j in range(2):
            text_color = (
                "white"
                if confusion_df.values[i, j] > confusion_df.values.max() / 2
                else "black"
            )
            ax.text(
                j,
                i,
                int(confusion_df.values[i, j]),
                ha="center",
                va="center",
                color=text_color,
                fontsize=16,
                fontweight="bold",
            )

    # Labels and title
    ax.set_xlabel(f"{method_b} Significant", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"{method_a} Significant", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Significance Agreement\n{agreement * 100:.1f}% (n={n_overlap})",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )

    plt.tight_layout()

    return fig
