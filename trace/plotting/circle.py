"""Circle plot visualization for treatment effects by ATC group.

This module provides functions to create circular (polar) plots showing
log risk ratios for multiple outcomes, grouped by ATC classification.
Bars are colored according to significance level and effect direction.
"""

from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_circle(
    df: pd.DataFrame,
    *,
    outcome_col: str = "outcome",
    log_rr_col: str = "log_RR",
    q_value_col: str = "q_value",
    group_col: str = "group",
    figsize: Tuple[float, float] = (9, 12.6),
    sig_thresholds: Tuple[float, float, float] = (0.05, 0.01, 0.001),
    reds: Tuple[str, str, str] = ("#ffb3b3", "#bf3a3a", "#9c0202"),
    blues: Tuple[str, str, str] = ("#b3c6ff", "#4d79ff", "#0033cc"),
    neutral_color: str = "lightgrey",
    group_label_fontsize: int = 12,
    group_label_fontweight: str = "bold",
    group_label_radius: float = -1.8,
    radial_limit: float | None = None,
    max_bar_radius: float | None = None,
    show_separator_circle: bool = False,
    separator_circle_margin: float = 0.05,
    separator_circle_color: str = "gray",
    separator_circle_linewidth: float = 1.5,
    separator_circle_linestyle: str = "-",
) -> plt.Figure:
    """Create a circular (polar) plot of log risk ratios grouped by ATC class.

    Produces a polar bar chart where each bar represents an outcome, with:
    - Bar height representing magnitude of log RR
    - Bar color representing direction (positive/negative) and significance level
    - Bars grouped by ATC classification (first letter of outcome code)
    - ATC group labels positioned at the center of each group
    - Vertical separator lines between groups
    - Legend showing significance levels for positive and negative effects

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with columns for outcome, log_RR, q_value, and group.
        Should be pre-sorted by outcome to maintain consistent ordering.
    outcome_col : str, default "outcome"
        Column name for outcome identifiers
    log_rr_col : str, default "log_RR"
        Column name for log risk ratio values
    q_value_col : str, default "q_value"
        Column name for adjusted p-values (q-values)
    group_col : str, default "group"
        Column name for ATC group (typically first letter of outcome code)
    figsize : tuple of float, default (9, 12.6)
        Figure size in inches (width, height)
    sig_thresholds : tuple of float, default (0.05, 0.01, 0.001)
        Three significance thresholds for color binning (descending order)
    reds : tuple of str, default ("#ffb3b3", "#bf3a3a", "#9c0202")
        Three colors for positive effects (light to dark for increasing significance)
    blues : tuple of str, default ("#b3c6ff", "#4d79ff", "#0033cc")
        Three colors for negative effects (light to dark for increasing significance)
    neutral_color : str, default "lightgrey"
        Color for non-significant effects (q >= first threshold)
    group_label_fontsize : int, default 12
        Font size for ATC group labels
    group_label_fontweight : str, default "bold"
        Font weight for ATC group labels
    group_label_radius : float, default -1.8
        Radial position for ATC group labels (negative values place labels
        outside the bars; more negative = further from center)
    radial_limit : float or None, default None
        Maximum radial extent (most negative ylim value). If None, uses fixed
        limit of -3 to accommodate labels at various positions.
    max_bar_radius : float or None, default None
        Radial position where the longest bar should end. If specified, all bars
        are scaled proportionally so the longest reaches this radius. If None,
        bars use original unscaled values.
    show_separator_circle : bool, default False
        Whether to draw a circular separator line between bars and labels
    separator_circle_margin : float, default 0.05
        Radial gap between longest bar and separator circle
    separator_circle_color : str, default "gray"
        Color of the separator circle line
    separator_circle_linewidth : float, default 1.5
        Width of the separator circle line
    separator_circle_linestyle : str, default "-"
        Line style of the separator circle

    Returns
    -------
    plt.Figure
        Matplotlib figure containing the circular plot with legend

    Notes
    -----
    The plot uses a polar projection with bars extending downward from radius=1.
    This creates a "ring" appearance where longer bars indicate larger effect
    magnitudes. Groups are separated by thin vertical lines, and group labels
    are positioned at the angular midpoint of each group.

    Examples
    --------
    >>> df_plot = df.sort_values("outcome").copy()
    >>> df_plot["group"] = df_plot["outcome"].str[0]
    >>> fig = plot_circle(df_plot)
    >>> fig.savefig("circle_plot.png", dpi=300)
    """
    # Validate input
    required_cols = {outcome_col, log_rr_col, q_value_col, group_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Dataframe missing required columns: {', '.join(sorted(missing_cols))}"
        )

    if df.empty:
        raise ValueError("Cannot create circle plot from empty dataframe")

    # Ensure df is sorted by outcome for consistent ordering
    df = df.sort_values(outcome_col).reset_index(drop=True)

    # Extract data
    log_rr_values = df[log_rr_col].to_numpy()
    q_values = df[q_value_col].to_numpy()
    groups = df[group_col].unique()

    # Create angular positions for each outcome
    n_outcomes = len(df)
    theta = np.linspace(0, 2 * np.pi, n_outcomes, endpoint=False)
    width = (2 * np.pi / n_outcomes) * 0.9  # slight gap between bars

    # Bin q-values into significance levels
    # np.digitize with thresholds [0.05, 0.01, 0.001] gives:
    #   bin=0 if q >= 0.05 (not significant)
    #   bin=1 if 0.01 <= q < 0.05 (low significance)
    #   bin=2 if 0.001 <= q < 0.01 (medium significance)
    #   bin=3 if q < 0.001 (high significance)
    bins = np.digitize(q_values, sig_thresholds)

    # Assign colors based on significance and direction
    colors_list = []
    reds_array = np.array(reds, dtype=object)
    blues_array = np.array(blues, dtype=object)

    for log_rr, q, b in zip(log_rr_values, q_values, bins):
        if np.isnan(q) or q >= sig_thresholds[0]:
            # Not significant
            colors_list.append(neutral_color)
        else:
            # Map bin to color index: bin 1,2,3 -> index 0,1,2
            idx = min(b - 1, len(reds) - 1)
            if log_rr > 0:
                colors_list.append(reds_array[idx])
            else:
                colors_list.append(blues_array[idx])

    # Create figure with polar projection
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})

    # Scale bars if max_bar_radius is specified
    if max_bar_radius is not None:
        # Scale so longest bar reaches max_bar_radius
        max_original_length = np.max(np.abs(log_rr_values))
        target_length = 1 - max_bar_radius  # How long the longest bar should be
        scale_factor = target_length / max_original_length
        bar_heights = -np.abs(log_rr_values) * scale_factor
        actual_max_bar_length = target_length
    else:
        # Use original values
        bar_heights = -np.abs(log_rr_values)
        actual_max_bar_length = np.max(np.abs(log_rr_values))

    # Plot bars extending downward from radius=1
    ax.bar(
        theta,
        bar_heights,
        bottom=1,
        width=width,
        color=colors_list,
    )

    # Calculate separator circle radius based on actual (possibly scaled) bar length
    circle_radius = 1 - actual_max_bar_length - separator_circle_margin

    # Set radial limits
    if radial_limit is None:
        # Use fixed generous limit that accommodates labels at various positions
        # This keeps bar scale consistent when you move labels
        ax.set_ylim(-3, 1)
    else:
        ax.set_ylim(radial_limit, 1)

    # Draw separator circle
    if show_separator_circle:
        circle_theta = np.linspace(0, 2 * np.pi, 200)
        circle_r = np.full(200, circle_radius)
        ax.plot(
            circle_theta,
            circle_r,
            color=separator_circle_color,
            linewidth=separator_circle_linewidth,
            linestyle=separator_circle_linestyle,
            clip_on=False,
            zorder=10,
        )

    # Add ATC group labels at mean theta position of each group
    for group in groups:
        group_mask = df[group_col] == group
        group_indices = df[group_mask].index.tolist()
        if not group_indices:
            continue

        # Calculate mean theta for this group
        idx = int(np.mean(group_indices))
        mid_theta = theta[idx]

        # Place label at specified radial distance
        ax.text(
            mid_theta,
            group_label_radius,
            str(group),
            ha="center",
            va="center",
            rotation_mode="anchor",
            fontsize=group_label_fontsize,
            fontweight=group_label_fontweight,
            clip_on=False,
        )

    # Add vertical separator lines between groups
    group_counts = df.groupby(group_col, sort=False).size()
    cumsum_indices = group_counts.cumsum().tolist()

    for idx in cumsum_indices[:-1]:  # skip last one (wraps around)
        # Draw line at the edge of the last bar in this group
        separator_theta = theta[idx - 1] + width / 2
        ax.axvline(
            separator_theta,
            color="gray",
            lw=0.5,
            ls="--",
            clip_on=False,
            ymax=1,
        )

    # Add separator at wrap-around point (between last and first group)
    # Only needed if first and last outcomes belong to different groups
    if df[group_col].iloc[0] != df[group_col].iloc[-1]:
        wraparound_theta = theta[-1] + width / 2
        ax.axvline(
            wraparound_theta,
            color="gray",
            lw=0.5,
            ls="--",
            clip_on=False,
            ymax=1,
        )

    # Remove grid and angular labels
    ax.set_rgrids([])
    ax.set_thetagrids([])
    ax.grid(False)

    # Create custom legend as inset
    _add_legend_inset(fig, reds, blues, neutral_color)

    return fig


def _add_legend_inset(
    fig: plt.Figure,
    reds: Tuple[str, str, str],
    blues: Tuple[str, str, str],
    neutral_color: str,
) -> None:
    """Add a custom legend inset showing significance levels.

    Creates a horizontal legend table with:
    - Top row: positive effect colors
    - Middle row: significance labels (None, Low, Medium, High)
    - Bottom row: negative effect colors

    Parameters
    ----------
    fig : plt.Figure
        Figure to add legend to
    reds : tuple of str
        Three colors for positive effects
    blues : tuple of str
        Three colors for negative effects
    neutral_color : str
        Color for non-significant effects
    """
    # Create inset axes for legend (positioned at bottom center)
    legend_ax = fig.add_axes([0.4, -0.05, 0.5, 0.2])
    legend_ax.axis("off")

    labels = ["None", "Low", "Medium", "High"]

    # Prepend neutral color to both palettes
    reds_with_neutral = [neutral_color] + list(reds)
    blues_with_neutral = [neutral_color] + list(blues)

    # Row headers
    legend_ax.text(0, 0.9, "Positive", ha="right", va="center", fontweight="bold")
    legend_ax.text(0, 0.71, "Significance", ha="right", va="bottom")
    legend_ax.text(0, 0.6, "Negative", ha="right", va="center", fontweight="bold")

    # Draw colored squares and significance labels
    x_positions = np.linspace(0.075, 0.5, len(labels))
    for x, lab, r, b in zip(x_positions, labels, reds_with_neutral, blues_with_neutral):
        # Top row (positive)
        legend_ax.add_patch(
            plt.Rectangle(
                (x - 0.05, 0.85),
                0.1,
                0.1,
                color=r,
                transform=legend_ax.transAxes,
                clip_on=False,
            )
        )
        # Bottom row (negative)
        legend_ax.add_patch(
            plt.Rectangle(
                (x - 0.05, 0.55),
                0.1,
                0.1,
                color=b,
                transform=legend_ax.transAxes,
                clip_on=False,
            )
        )
        # Shared significance label below each column
        legend_ax.text(x, 0.71, lab, ha="center", va="bottom")


__all__ = ["plot_circle"]
