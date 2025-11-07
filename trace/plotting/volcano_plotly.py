"""Interactive Plotly volcano plot utilities."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Mapping, Sequence, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


DEFAULT_COLORS: Mapping[str, str] = {
    "Significant": "#d62728",
    "Not significant": "#7f7f7f",
}


def _format_float(value: float | int | None, fmt: str) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, (float, int)) and not np.isfinite(value):
        return "NA"
    try:
        return fmt.format(value)
    except (ValueError, TypeError):
        return str(value)


def _format_count_pair(events: float | int | None, total: float | int | None) -> str:
    if events is None or total is None or pd.isna(events) or pd.isna(total):
        return "NA"
    if isinstance(total, (float, int)) and total == 0:
        return "NA"
    try:
        return f"{int(events):,}/{int(total):,}"
    except (TypeError, ValueError):
        return "NA"


def _build_hover_text(
    row: pd.Series,
    *,
    method_col: str,
    outcome_col: str,
    effect_col: str,
    effect_label: str,
) -> str:
    label = row.get("outcome_label")
    if pd.isna(label):
        label = row.get(outcome_col)

    lines = [f"<b>{label}</b>"]

    outcome_code = row.get(outcome_col)
    atc_description = row.get("atc_description")
    if pd.notna(outcome_code) and pd.notna(atc_description):
        lines.append(f"ATC: {outcome_code} — {atc_description}")
    elif pd.notna(outcome_code):
        lines.append(f"ATC: {outcome_code}")

    method_value = row.get(method_col)
    if pd.notna(method_value):
        lines.append(f"Method: {method_value}")

    effect_value = row.get(effect_col)
    lines.append(f"{effect_label}: {_format_float(effect_value, '{:.4f}')}")

    se_candidates = [f"SE_{effect_col}", "SE_RD", "SE_RR", "SE_log_RR"]
    for se_col in se_candidates:
        if se_col in row:
            lines.append(f"{se_col}: {_format_float(row.get(se_col), '{:.2e}')}")
            break
    lines.append(f"z: {_format_float(row.get('z'), '{:.2f}')}")
    lines.append(f"p-value: {_format_float(row.get('p_value'), '{:.2e}')}")
    lines.append(f"q-value: {_format_float(row.get('q_value'), '{:.2e}')}")

    ci_candidates = [
        (f"{effect_col}_CI95_lower", f"{effect_col}_CI95_upper"),
        ("RD_CI95_lower", "RD_CI95_upper"),
        ("RR_CI95_lower", "RR_CI95_upper"),
    ]
    for lo_col, hi_col in ci_candidates:
        if pd.notna(row.get(lo_col)) and pd.notna(row.get(hi_col)):
            ci_low = _format_float(row.get(lo_col), "{:.4f}")
            ci_high = _format_float(row.get(hi_col), "{:.4f}")
            lines.append(f"95% CI: [{ci_low}, {ci_high}]")
            break

    tau_candidates = (
        ("tau2", "tau² (RD)"),
        ("eta1_tau2", "tau² (arm1)"),
        ("eta0_tau2", "tau² (arm0)"),
    )
    for col, label in tau_candidates:
        value = row.get(col)
        if pd.notna(value):
            lines.append(f"{label}: {_format_float(value, '{:.2e}')}")

    if pd.notna(row.get("per_run_n_runs")):
        lines.append(f"Runs (per method): {int(row['per_run_n_runs'])}")
        prefix = effect_col.lower()
        range_cols = (
            f"per_run_{prefix}_min",
            f"per_run_{prefix}_max",
        )
        mean_cols = (
            f"per_run_{prefix}_mean",
            f"per_run_{prefix}_std",
        )
        if all(col in row for col in range_cols):
            lines.append(
                f"{effect_label} range (runs): "
                f"{_format_float(row.get(range_cols[0]), '{:.4f}')} – "
                f"{_format_float(row.get(range_cols[1]), '{:.4f}')}"
            )
        if all(col in row for col in mean_cols):
            lines.append(
                f"{effect_label} mean ± sd (runs): "
                f"{_format_float(row.get(mean_cols[0]), '{:.4f}')} ± "
                f"{_format_float(row.get(mean_cols[1]), '{:.4f}')}"
            )
        lines.append(
            f"Arm1 mean ± sd: {_format_float(row.get('per_run_effect1_mean'), '{:.4f}')} "
            f"± {_format_float(row.get('per_run_effect1_std'), '{:.4f}')}"
        )
        lines.append(
            f"Arm0 mean ± sd: {_format_float(row.get('per_run_effect0_mean'), '{:.4f}')} "
            f"± {_format_float(row.get('per_run_effect0_std'), '{:.4f}')}"
        )

    prevalence_total = row.get("prevalence_total")
    population_total = row.get("population_total")
    outcome_total = row.get("outcome_events_total")
    if pd.notna(prevalence_total):
        lines.append(
            "Prevalence (total): "
            f"{_format_float(prevalence_total * 100, '{:.2f}%')} "
            f"({_format_count_pair(outcome_total, population_total)})"
        )

    prevalence_treated = row.get("prevalence_treated")
    population_treated = row.get("population_treated")
    outcome_treated = row.get("outcome_events_treated")
    if pd.notna(prevalence_treated):
        lines.append(
            "Prevalence (treated): "
            f"{_format_float(prevalence_treated * 100, '{:.2f}%')} "
            f"({_format_count_pair(outcome_treated, population_treated)})"
        )

    prevalence_untreated = row.get("prevalence_untreated")
    population_untreated = row.get("population_untreated")
    outcome_untreated = row.get("outcome_events_untreated")
    if pd.notna(prevalence_untreated):
        lines.append(
            "Prevalence (untreated): "
            f"{_format_float(prevalence_untreated * 100, '{:.2f}%')} "
            f"({_format_count_pair(outcome_untreated, population_untreated)})"
        )

    return "<br>".join(lines)


def build_plotly_volcano(
    df: pd.DataFrame,
    *,
    alpha: float = 0.05,
    method_col: str = "method",
    outcome_col: str = "outcome",
    effect_col: str = "RD",
    effect_label: str = "Risk difference (RD)",
    neglog_col: str = "neglog10p",
    colors: Mapping[str, str] | None = None,
    point_size: int = 12,
    null_value: Optional[float] = 0.0,
    xscale: Optional[str] = None,
) -> go.Figure:
    """Build an interactive volcano plot with Plotly."""

    if df.empty:
        raise ValueError("Input dataframe is empty; cannot build volcano plot.")

    palette = dict(DEFAULT_COLORS)
    if colors:
        palette.update(colors)

    data = df.copy()
    data["significance"] = np.where(
        data["q_value"] < alpha, "Significant", "Not significant"
    )
    data["hover_text"] = data.apply(
        _build_hover_text,
        axis=1,
        method_col=method_col,
        outcome_col=outcome_col,
        effect_col=effect_col,
        effect_label=effect_label,
    )

    methods = list(dict.fromkeys(data[method_col]))
    n_methods = len(methods)

    fig = make_subplots(
        rows=1,
        cols=n_methods,
        shared_yaxes=True,
        horizontal_spacing=0.06,
        subplot_titles=methods,
    )

    for col_idx, method in enumerate(methods, start=1):
        subset = data[data[method_col] == method]
        if subset.empty:
            continue

        for significance_label in ["Significant", "Not significant"]:
            method_subset = subset[subset["significance"] == significance_label]
            if method_subset.empty:
                continue

            marker_color = palette.get(significance_label, "#333333")

            fig.add_trace(
                go.Scatter(
                    x=method_subset[effect_col],
                    y=method_subset[neglog_col],
                    mode="markers",
                    marker=dict(color=marker_color, size=point_size, opacity=0.9),
                    name=significance_label,
                    legendgroup=significance_label,
                    showlegend=col_idx == 1,
                    hovertext=method_subset["hover_text"],
                    hovertemplate="%{hovertext}<extra></extra>",
                ),
                row=1,
                col=col_idx,
            )

        fig.add_hline(
            y=-np.log10(alpha),
            line=dict(color="rgba(120,120,120,0.5)", dash="dash"),
            row=1,
            col=col_idx,
        )
        if null_value is not None:
            fig.add_vline(
                x=null_value,
                line=dict(color="rgba(120,120,120,0.5)", dash="dash"),
                row=1,
                col=col_idx,
            )

    for col_idx in range(1, n_methods + 1):
        fig.update_xaxes(title_text=effect_label, row=1, col=col_idx)
        if xscale == "log":
            fig.update_xaxes(type="log", row=1, col=col_idx)
    fig.update_yaxes(title_text="-log10(p-value)", row=1, col=1)

    fig.update_layout(
        legend_title_text="",
        hoverlabel=dict(bgcolor="white", font_color="#222"),
        margin=dict(l=70, r=40, t=70, b=60),
        template="plotly_white",
    )

    return fig


def build_plotly_overlay_methods(
    df: pd.DataFrame,
    *,
    methods: Sequence[str] = ("TMLE", "IPW"),
    method_col: str = "method",
    outcome_col: str = "outcome",
    effect_col: str = "RD",
    effect_label: str = "Risk difference (RD)",
    neglog_col: str = "neglog10p",
    alpha: float = 0.05,
    marker_size: int = 9,
    line_color: str = "rgba(108,117,125,0.6)",
    line_width: float = 1.2,
    label_map: Mapping[str, str] | None = None,
    null_value: Optional[float] = 0.0,
    xscale: Optional[str] = None,
) -> go.Figure:
    """Build an interactive overlay comparing two methods on one volcano plot."""

    if len(methods) != 2:
        raise ValueError("build_plotly_overlay_methods expects exactly two methods.")

    subset = df[df[method_col].isin(methods)].copy()
    if subset.empty:
        raise ValueError(
            "No rows found for the requested methods in the volcano dataframe."
        )

    if label_map:
        subset["outcome_label"] = subset[outcome_col].map(label_map)

    subset["hover_text"] = subset.apply(
        _build_hover_text,
        axis=1,
        method_col=method_col,
        outcome_col=outcome_col,
        effect_col=effect_col,
        effect_label=effect_label,
    )

    paired_outcomes = subset.groupby(outcome_col)[method_col].nunique().eq(len(methods))
    paired_outcomes = paired_outcomes[paired_outcomes].index

    paired = subset[subset[outcome_col].isin(paired_outcomes)].copy()
    if paired.empty:
        raise ValueError("No outcomes contain both methods; overlay not created.")

    line_x: list[float | None] = []
    line_y: list[float | None] = []
    for _, group in paired.groupby(outcome_col):
        group = group.set_index(method_col)
        if not all(m in group.index for m in methods):
            continue
        line_x.extend(
            [
                group.loc[methods[0], effect_col],
                group.loc[methods[1], effect_col],
                None,
            ]
        )
        line_y.extend(
            [
                group.loc[methods[0], neglog_col],
                group.loc[methods[1], neglog_col],
                None,
            ]
        )

    fig = go.Figure()

    if line_x and line_y:
        fig.add_trace(
            go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                line=dict(color=line_color, width=line_width),
                hoverinfo="skip",
                name="Method difference",
                showlegend=False,
            )
        )

    for method in methods:
        method_subset = subset[subset[method_col] == method]
        if method_subset.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=method_subset[effect_col],
                y=method_subset[neglog_col],
                mode="markers",
                marker=dict(size=marker_size),
                name=method,
                hovertext=method_subset["hover_text"],
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )

    fig.add_hline(
        y=-np.log10(alpha),
        line=dict(color="rgba(120,120,120,0.4)", dash="dash"),
    )
    if null_value is not None:
        fig.add_vline(
            x=null_value,
            line=dict(color="rgba(120,120,120,0.4)", dash="dash"),
        )

    fig.update_xaxes(title_text=effect_label)
    if xscale == "log":
        fig.update_xaxes(type="log")
    fig.update_yaxes(title_text="-log10(p-value)")
    fig.update_layout(
        template="plotly_white",
        legend_title_text="Method",
        hoverlabel=dict(bgcolor="white", font_color="#222"),
        margin=dict(l=70, r=40, t=60, b=60),
    )

    return fig


def save_plotly_figure(
    fig: go.Figure,
    *,
    html_path: str | Path,
    png_path: str | Path | None = None,
    width: int = 1100,
    height: int = 520,
    scale: float = 2.0,
) -> None:
    """Persist a Plotly figure to HTML (and optionally PNG)."""

    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(html_path), include_plotlyjs="cdn")

    if png_path is not None:
        png_path = Path(png_path)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.write_image(str(png_path), width=width, height=height, scale=scale)
        except (ValueError, ImportError, OSError) as exc:
            warnings.warn(
                f"Unable to save Plotly figure to PNG at {png_path}: {exc}",
                RuntimeWarning,
            )


__all__ = [
    "build_plotly_volcano",
    "build_plotly_overlay_methods",
    "save_plotly_figure",
]
