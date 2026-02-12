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


def _extract_code_prefixes(
    codes: pd.Series, max_length: int = 4
) -> tuple[pd.Series, list[str]]:
    """Extract ATC code prefixes and return unique prefixes for filtering.

    Returns:
        Series with code prefixes, and list of unique prefixes sorted by length.
    """
    # Extract prefixes of different lengths
    prefixes = []
    unique_prefixes = set()

    for length in range(1, max_length + 1):
        prefix_series = codes.astype(str).str[:length]
        prefixes.append(prefix_series)
        unique_prefixes.update(prefix_series.dropna().unique())

    # Use 2-character prefix by default (most common filtering level)
    code_prefixes = codes.astype(str).str[:2]

    # Sort unique prefixes: "All" first, then by length, then alphabetically
    sorted_prefixes = ["All"] + sorted(unique_prefixes, key=lambda x: (len(x), x))

    return code_prefixes, sorted_prefixes


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
    title: Optional[str] = None,
) -> go.Figure:
    """Build an interactive volcano plot with Plotly and ATC code filtering."""

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

    # Extract code prefixes for filtering
    # Use 5-character prefixes for full ATC code granularity (A01AA, A02AA, N03AG, etc.)
    # Format: letter, number, number, letter, letter (e.g., "A01AA", "N03AG")
    data["code_prefix"] = data[outcome_col].astype(str).str[:5]
    _, unique_prefixes = _extract_code_prefixes(data[outcome_col])
    # Group traces by 5-character prefix for precise filtering
    unique_5char_prefixes = sorted(data["code_prefix"].unique())
    unique_prefixes = ["All"] + unique_5char_prefixes

    methods = list(dict.fromkeys(data[method_col]))
    n_methods = len(methods)

    fig = make_subplots(
        rows=1,
        cols=n_methods,
        shared_yaxes=True,
        horizontal_spacing=0.06,
        subplot_titles=methods,
    )

    # Store trace info for filtering: (trace_idx, code_prefix)
    trace_info = []

    for col_idx, method in enumerate(methods, start=1):
        subset = data[data[method_col] == method]
        if subset.empty:
            continue

        for significance_label in ["Significant", "Not significant"]:
            method_subset = subset[subset["significance"] == significance_label]
            if method_subset.empty:
                continue

            marker_color = palette.get(significance_label, "#333333")

            # Create traces grouped by code prefix for filtering
            for prefix in unique_prefixes[1:]:  # Skip "All"
                prefix_subset = method_subset[method_subset["code_prefix"] == prefix]
                if prefix_subset.empty:
                    continue

                # Capture trace index before adding
                trace_idx_before = len(fig.data)
                fig.add_trace(
                    go.Scatter(
                        x=prefix_subset[effect_col],
                        y=prefix_subset[neglog_col],
                        mode="markers",
                        marker=dict(color=marker_color, size=point_size, opacity=0.9),
                        name=significance_label,
                        legendgroup=significance_label,
                        showlegend=col_idx == 1 and prefix == unique_prefixes[1],
                        hovertext=prefix_subset["hover_text"],
                        hovertemplate="%{hovertext}<extra></extra>",
                        visible=True,  # All visible by default
                    ),
                    row=1,
                    col=col_idx,
                )
                # Capture trace index after adding - should be the last trace
                trace_idx_after = len(fig.data) - 1
                # Verify the index is correct
                if trace_idx_after != trace_idx_before:
                    print(
                        f"WARNING: Trace index mismatch - expected {trace_idx_before}, got {trace_idx_after}"
                    )

                # Store both the prefix and the actual outcome codes for this trace
                # Convert to strings to ensure proper matching
                outcome_codes = prefix_subset[outcome_col].astype(str).unique().tolist()
                # Also store method and significance for debugging
                trace_info.append(
                    (trace_idx_after, prefix, outcome_codes, method, significance_label)
                )

    # Count scatter traces (before hlines/vlines are added)
    n_scatter_traces = len(fig.data)

    for col_idx, method in enumerate(methods, start=1):
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

    # Store trace info for JavaScript filtering
    # We'll embed this in the HTML via save_plotly_figure
    if len(unique_prefixes) > 1:
        # Store trace prefix mapping and outcome codes for filtering
        # Format: {trace_idx: {prefix: "...", codes: ["...", "..."], method: "...", significance: "..."}}
        trace_info_map = {}
        for idx, prefix, codes, method, significance in trace_info:
            trace_info_map[str(idx)] = {
                "prefix": prefix,
                "codes": codes,
                "method": method,
                "significance": significance,
            }
        # Store in figure's customdata or as a hidden annotation
        # We'll extract this in save_plotly_figure and inject JavaScript
        if not hasattr(fig, "_trace_filter_info"):
            fig._trace_filter_info = {}
        fig._trace_filter_info["trace_info_map"] = trace_info_map
        fig._trace_filter_info["n_scatter_traces"] = n_scatter_traces
        fig._trace_filter_info["n_total_traces"] = len(fig.data)

    layout_dict = {
        "legend_title_text": "",
        "hoverlabel": dict(bgcolor="white", font_color="#222"),
        "margin": dict(l=70, r=120, t=70, b=60),  # Increased right margin for input box
        "template": "plotly_white",
    }
    if title:
        layout_dict["title"] = {
            "text": title,
            "x": 0.5,
            "xanchor": "center",
            "font": dict(size=16),
        }
        # Increase top margin to accommodate title
        layout_dict["margin"]["t"] = 100

    fig.update_layout(**layout_dict)

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
    title: Optional[str] = None,
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

    # Extract code prefixes for filtering
    # Use 5-character prefixes for full ATC code granularity (A01AA, A02AA, N03AG, etc.)
    # Format: letter, number, number, letter, letter (e.g., "A01AA", "N03AG")
    subset["code_prefix"] = subset[outcome_col].astype(str).str[:5]
    _, unique_prefixes_temp = _extract_code_prefixes(subset[outcome_col])
    # Group traces by 5-character prefix for precise filtering
    unique_5char_prefixes = sorted(subset["code_prefix"].unique())
    unique_prefixes = ["All"] + unique_5char_prefixes

    paired_outcomes = subset.groupby(outcome_col)[method_col].nunique().eq(len(methods))
    paired_outcomes = paired_outcomes[paired_outcomes].index

    paired = subset[subset[outcome_col].isin(paired_outcomes)].copy()
    if paired.empty:
        raise ValueError("No outcomes contain both methods; overlay not created.")

    fig = go.Figure()

    # Store trace info for filtering: (trace_idx, code_prefix, trace_type)
    # trace_type: 'line', 'method_0', 'method_1'
    trace_info = []

    # Create connecting lines grouped by prefix
    for prefix in unique_prefixes[1:]:  # Skip "All"
        prefix_paired = paired[paired["code_prefix"] == prefix]
        if prefix_paired.empty:
            continue

        line_x: list[float | None] = []
        line_y: list[float | None] = []
        for _, group in prefix_paired.groupby(outcome_col):
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

        if line_x and line_y:
            # trace_idx_before = len(fig.data)
            fig.add_trace(
                go.Scatter(
                    x=line_x,
                    y=line_y,
                    mode="lines",
                    line=dict(color=line_color, width=line_width),
                    hoverinfo="skip",
                    name="Method difference",
                    showlegend=False,
                    visible=True,
                )
            )
            trace_idx_after = len(fig.data) - 1
            # Get outcome codes for this prefix
            outcome_codes = prefix_paired[outcome_col].astype(str).unique().tolist()
            trace_info.append((trace_idx_after, prefix, "line", outcome_codes))

    # Create method traces grouped by prefix
    for method_idx, method in enumerate(methods):
        method_subset = subset[subset[method_col] == method]
        if method_subset.empty:
            continue

        for prefix in unique_prefixes[1:]:  # Skip "All"
            prefix_subset = method_subset[method_subset["code_prefix"] == prefix]
            if prefix_subset.empty:
                continue

            # trace_idx_before = len(fig.data)
            fig.add_trace(
                go.Scatter(
                    x=prefix_subset[effect_col],
                    y=prefix_subset[neglog_col],
                    mode="markers",
                    marker=dict(size=marker_size),
                    name=method,
                    showlegend=prefix
                    == unique_prefixes[1],  # Show legend only for first prefix
                    hovertext=prefix_subset["hover_text"],
                    hovertemplate="%{hovertext}<extra></extra>",
                    visible=True,
                )
            )
            trace_idx_after = len(fig.data) - 1
            # Store both the prefix and the actual outcome codes for this trace
            # Convert to strings to ensure proper matching
            outcome_codes = prefix_subset[outcome_col].astype(str).unique().tolist()
            trace_info.append(
                (trace_idx_after, prefix, f"method_{method_idx}", outcome_codes, method)
            )

    # Count scatter traces (before hlines/vlines)
    n_scatter_traces = len(fig.data)

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

    # Store trace info for JavaScript filtering
    if len(unique_prefixes) > 1:
        # Store trace info mapping with outcome codes
        trace_info_map = {}
        for item in trace_info:
            if len(item) == 4:  # Line trace: (idx, prefix, "line", codes)
                idx, prefix, trace_type, codes = item
                trace_info_map[str(idx)] = {
                    "prefix": prefix,
                    "type": trace_type,
                    "codes": codes,
                    "method": None,
                }
            elif (
                len(item) == 5
            ):  # Method trace: (idx, prefix, "method_X", codes, method)
                idx, prefix, trace_type, codes, method = item
                trace_info_map[str(idx)] = {
                    "prefix": prefix,
                    "type": trace_type,
                    "codes": codes,
                    "method": method,
                }
        if not hasattr(fig, "_trace_filter_info"):
            fig._trace_filter_info = {}
        fig._trace_filter_info["trace_info_map"] = trace_info_map
        fig._trace_filter_info["n_scatter_traces"] = n_scatter_traces
        fig._trace_filter_info["n_total_traces"] = len(fig.data)

    layout_dict = {
        "template": "plotly_white",
        "legend_title_text": "Method",
        "hoverlabel": dict(bgcolor="white", font_color="#222"),
        "margin": dict(l=70, r=120, t=60, b=60),  # Increased right margin for input box
    }
    if title:
        layout_dict["title"] = {
            "text": title,
            "x": 0.5,
            "xanchor": "center",
            "font": dict(size=16),
        }
        # Increase top margin to accommodate title
        layout_dict["margin"]["t"] = 100

    fig.update_layout(**layout_dict)

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
    """Persist a Plotly figure to HTML (and optionally PNG) with code filtering input."""

    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if figure has trace filter info
    has_filter = hasattr(fig, "_trace_filter_info") and fig._trace_filter_info

    if has_filter:
        # Write to temporary file first
        temp_html = html_path.with_suffix(".tmp.html")
        fig.write_html(str(temp_html), include_plotlyjs="cdn")

        # Read the HTML and inject JavaScript for filtering
        with open(temp_html, "r", encoding="utf-8") as f:
            html_content = f.read()

        # Extract trace filter info
        trace_info_map = fig._trace_filter_info["trace_info_map"]
        n_scatter_traces = fig._trace_filter_info["n_scatter_traces"]
        n_total_traces = fig._trace_filter_info["n_total_traces"]

        # Create JavaScript code for filtering
        # Convert trace info map to JavaScript object
        import json

        trace_map_js = json.dumps(trace_info_map)

        filter_script = f"""
        <style>
            #code-filter-container {{
                position: fixed;
                top: 50px;
                right: 10px;
                z-index: 1000;
                background: white;
                padding: 8px;
                border: 1px solid #ccc;
                border-radius: 4px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            #code-filter-container input {{
                padding: 4px 8px;
                border: 1px solid #ccc;
                border-radius: 3px;
                margin-right: 4px;
                width: 120px;
            }}
            #code-filter-container button {{
                padding: 4px 12px;
                border: 1px solid #007bff;
                background: #007bff;
                color: white;
                border-radius: 3px;
                cursor: pointer;
            }}
            #code-filter-container button:hover {{
                background: #0056b3;
            }}
            #code-filter-debug {{
                position: fixed;
                top: 50px;
                left: 10px;
                z-index: 1000;
                background: #f9f9f9;
                padding: 8px;
                border: 1px solid #ccc;
                border-radius: 4px;
                font-size: 11px;
                max-width: 300px;
                max-height: 200px;
                overflow-y: auto;
                display: none;
            }}
            #code-filter-debug.show {{
                display: block;
            }}
        </style>
        <div id="code-filter-container">
            <label for="code-filter-input" style="display: block; margin-bottom: 4px; font-size: 12px; font-weight: bold;">Filter by code:</label>
            <div style="display: flex; align-items: center;">
                <input type="text" id="code-filter-input" placeholder="e.g., N03, A01AA..." style="width: 150px; padding: 4px; border: 1px solid #ccc; border-radius: 2px; font-size: 12px;">
                <button id="code-filter-clear" style="margin-left: 4px; padding: 4px 8px; border: 1px solid #ccc; border-radius: 2px; background: #f0f0f0; cursor: pointer; font-size: 12px;">Clear</button>
            </div>
            <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #ddd;">
                <label for="xaxis-min-input" style="display: block; margin-bottom: 4px; font-size: 12px; font-weight: bold;">X-axis range:</label>
                <div style="display: flex; align-items: center; gap: 4px;">
                    <input type="number" id="xaxis-min-input" placeholder="Min" step="any" style="width: 70px; padding: 4px; border: 1px solid #ccc; border-radius: 2px; font-size: 12px;">
                    <span style="font-size: 12px;">to</span>
                    <input type="number" id="xaxis-max-input" placeholder="Max" step="any" style="width: 70px; padding: 4px; border: 1px solid #ccc; border-radius: 2px; font-size: 12px;">
                    <button id="xaxis-reset-btn" style="padding: 4px 8px; border: 1px solid #007bff; border-radius: 2px; background: #007bff; color: white; cursor: pointer; font-size: 12px;">Set</button>
                    <button id="xaxis-auto-btn" style="padding: 4px 8px; border: 1px solid #6c757d; border-radius: 2px; background: #6c757d; color: white; cursor: pointer; font-size: 12px;">Auto</button>
                </div>
            </div>
            <button id="code-filter-debug-toggle" style="margin-top: 8px; padding: 4px 8px; border: 1px solid #ccc; border-radius: 2px; background: #e0e0e0; cursor: pointer; font-size: 11px;">Debug</button>
        </div>
        <div id="code-filter-debug">
            <strong>Debug Info:</strong><br>
            <div id="debug-content">Waiting for Plotly...</div>
        </div>
        
        <script>
        (function() {{
            const traceInfoMap = {trace_map_js};
            const nScatterTraces = {n_scatter_traces};
            const nTotalTraces = {n_total_traces};
            
            // Store the graph div reference once found
            let cachedGraphDiv = null;
            // Store the original axis ranges to preserve them during filtering
            let originalAxisRanges = null;
            
            function findPlotlyGraphDiv() {{
                if (cachedGraphDiv && cachedGraphDiv.data && Array.isArray(cachedGraphDiv.data)) {{
                    return cachedGraphDiv;
                }}
                
                // Try multiple methods to find the Plotly graph div
                let gd = null;
                
                // Method 1: Look for divs with IDs starting with "plotly"
                const plotlyDivsById = document.querySelectorAll('[id^="plotly"]');
                for (let div of plotlyDivsById) {{
                    // Check if this div has Plotly data attached
                    if (div.data && Array.isArray(div.data) && div.data.length > 0) {{
                        gd = div;
                        break;
                    }}
                    // Also check if Plotly has registered this element
                    if (div._fullLayout || div.layout) {{
                        gd = div;
                        break;
                    }}
                }}
                
                // Method 2: Look for divs with class "plotly" or "js-plotly-plot"
                if (!gd) {{
                    const plotlyDivsByClass = document.querySelectorAll('.plotly, .js-plotly-plot');
                    for (let div of plotlyDivsByClass) {{
                        if (div.data && Array.isArray(div.data) && div.data.length > 0) {{
                            gd = div;
                            break;
                        }}
                        if (div._fullLayout || div.layout) {{
                            gd = div;
                            break;
                        }}
                    }}
                }}
                
                // Method 3: Search all divs for one with Plotly data
                if (!gd) {{
                    const allDivs = document.querySelectorAll('div');
                    for (let div of allDivs) {{
                        if (div.data && Array.isArray(div.data) && div.data.length > 0) {{
                            gd = div;
                            break;
                        }}
                        if (div._fullLayout || (div.layout && div.data)) {{
                            gd = div;
                            break;
                        }}
                    }}
                }}
                
                // Method 4: Try to get from Plotly's internal registry
                if (!gd && window.Plotly) {{
                    const plotlyDivs = document.querySelectorAll('[id^="plotly"]');
                    for (let div of plotlyDivs) {{
                        try {{
                            const graphDiv = document.getElementById(div.id);
                            if (graphDiv && graphDiv.data) {{
                                gd = graphDiv;
                                break;
                            }}
                        }} catch(e) {{
                            // Continue searching
                        }}
                    }}
                }}
                
                if (gd) {{
                    cachedGraphDiv = gd;
                }}
                
                return gd;
            }}
            
            function filterTraces(filterValue) {{
                const gd = findPlotlyGraphDiv();
                
                if (!gd || !gd.data || !Array.isArray(gd.data)) {{
                    const debugContent = document.getElementById('debug-content');
                    if (debugContent) {{
                        const errorMsg = 'ERROR: Could not find Plotly graph div. Found divs: ' + 
                            document.querySelectorAll('[id^="plotly"]').length + ' with plotly ID, ' +
                            document.querySelectorAll('.plotly').length + ' with plotly class';
                        debugContent.innerHTML = '<span style="color: red;">' + errorMsg + '</span>';
                    }}
                    console.log('Could not find Plotly graph div');
                    return;
                }}
                
                // Debug: verify trace count matches
                if (gd.data.length !== nTotalTraces) {{
                    console.warn('Trace count mismatch! Expected ' + nTotalTraces + ' but found ' + gd.data.length);
                    const debugContent = document.getElementById('debug-content');
                    if (debugContent) {{
                        debugContent.innerHTML += '<br><span style="color: orange;">WARNING: Trace count mismatch (expected ' + nTotalTraces + ', found ' + gd.data.length + ')</span>';
                    }}
                }}
                
                const visible = [];
                const filterLower = filterValue.toLowerCase().trim();
                
                for (let i = 0; i < nTotalTraces; i++) {{
                    if (i < nScatterTraces) {{
                        // Scatter traces - check if any outcome code matches
                        const traceInfo = traceInfoMap[i.toString()];
                        if (!traceInfo) {{
                            // If no trace info, show it (shouldn't happen, but safe fallback)
                            // This might indicate a trace wasn't stored properly
                            console.warn('Trace ' + i + ' has no trace info - showing by default');
                            visible.push(true);
                        }} else if (filterLower === '') {{
                            // Empty filter - show all
                            visible.push(true);
                        }} else {{
                            // Check if any outcome code in this trace starts with filter value
                            const codes = traceInfo.codes || [];
                            let matches = false;
                            if (codes.length > 0 && filterLower.length > 0) {{
                                for (let code of codes) {{
                                    // Convert to string and normalize
                                    const codeStr = String(code).trim().toLowerCase();
                                    // Check if code starts with the filter (exact prefix match)
                                    // Must match from the beginning of the code
                                    if (codeStr && codeStr.startsWith(filterLower)) {{
                                        matches = true;
                                        break;
                                    }}
                                }}
                            }} else if (codes.length === 0) {{
                                // No codes stored - this shouldn't happen
                                console.warn('Trace ' + i + ' has no codes stored');
                                matches = false; // Hide if no codes (safer)
                            }}
                            visible.push(matches);
                        }}
                    }} else {{
                        // Non-scatter traces (hlines, vlines) - always visible
                        visible.push(true);
                    }}
                }}
                
                // Debug: log the trace mapping and visibility
                const debugContent = document.getElementById('debug-content');
                const visibleCount = visible.filter(v => v).length;
                
                if (debugContent) {{
                    const filterVal = filterValue || '';
                    const totalTraces = nTotalTraces;
                    const scatterTraces = nScatterTraces;
                    const mapKeys = Object.keys(traceInfoMap).length;
                    const firstPrefixes = Object.values(traceInfoMap).slice(0, 5).map(info => info.prefix).join(', ');
                    
                    // Show which traces match - check ALL traces, not just first 10
                    const matchingTraces = [];
                    const allMatchingIndices = [];
                    for (let i = 0; i < scatterTraces; i++) {{
                        const traceInfo = traceInfoMap[i.toString()];
                        if (traceInfo) {{
                            const codes = traceInfo.codes || [];
                            let matches = false;
                            if (filterVal === '') {{
                                matches = true;
                            }} else {{
                                for (let code of codes) {{
                                    const codeStr = String(code).trim().toLowerCase();
                                    if (codeStr && codeStr.startsWith(filterVal.toLowerCase())) {{
                                        matches = true;
                                        break;
                                    }}
                                }}
                            }}
                            if (matches) {{
                                allMatchingIndices.push(i);
                                // Only show first 10 in detail
                                if (matchingTraces.length < 10) {{
                                    matchingTraces.push(i + ':' + traceInfo.prefix + '[' + codes.slice(0, 3).join(',') + ']');
                                }}
                            }}
                        }}
                    }}
                    
                    // Show sample codes from first matching trace for debugging
                    let sampleCodes = '';
                    let missingTraces = [];
                    if (matchingTraces.length > 0) {{
                        const firstMatchIdx = matchingTraces[0].split(':')[0];
                        const firstTraceInfo = traceInfoMap[firstMatchIdx];
                        if (firstTraceInfo && firstTraceInfo.codes) {{
                            sampleCodes = firstTraceInfo.codes.slice(0, 5).join(', ');
                        }}
                    }}
                    
                    // Check for traces without info
                    for (let i = 0; i < Math.min(20, scatterTraces); i++) {{
                        if (!traceInfoMap[i.toString()]) {{
                            missingTraces.push(i);
                        }}
                    }}
                    
                    // Find ALL traces with N prefix for debugging
                    let nTraces = [];
                    let n03Traces = [];
                    let methodBreakdown = {{}};
                    for (let i = 0; i < scatterTraces; i++) {{
                        const traceInfo = traceInfoMap[i.toString()];
                        if (traceInfo) {{
                            const prefix = traceInfo.prefix || '';
                            const codes = traceInfo.codes || [];
                            const method = traceInfo.method || 'unknown';
                            const significance = traceInfo.significance || 'unknown';
                            
                            if (prefix.toLowerCase().startsWith('n')) {{
                                nTraces.push(i + ':' + prefix + '[' + codes.slice(0, 5).join(',') + '] (' + method + '/' + significance + ')');
                            }}
                            // Check if this trace has codes starting with the current filter value
                            // (for debugging - shows which traces should match the filter)
                            let hasFilterMatch = false;
                            if (filterVal && filterVal.length > 0) {{
                                const filterLower = filterVal.toLowerCase().trim();
                                for (let code of codes) {{
                                    const codeStr = String(code).trim().toLowerCase();
                                    // Exact prefix match - same logic as filtering (must use startsWith)
                                    if (codeStr && codeStr.startsWith(filterLower)) {{
                                        hasFilterMatch = true;
                                        break;
                                    }}
                                }}
                            }}
                            if (hasFilterMatch) {{
                                const key = method + '/' + significance;
                                if (!methodBreakdown[key]) {{
                                    methodBreakdown[key] = [];
                                }}
                                methodBreakdown[key].push(i);
                                n03Traces.push(i + ':' + prefix + '[' + codes.slice(0, 5).join(',') + '] (' + method + '/' + significance + ')');
                            }}
                        }}
                    }}
                    
                    debugContent.innerHTML = 
                        'Filter: "' + filterVal + '"<br>' +
                        'Total traces: ' + totalTraces + '<br>' +
                        'Scatter traces: ' + scatterTraces + '<br>' +
                        'Visible traces: ' + visibleCount + '<br>' +
                        'Matching trace count: ' + allMatchingIndices.length + '<br>' +
                        'Trace map keys: ' + mapKeys + '<br>' +
                        (missingTraces.length > 0 ? '<span style="color: orange;">Missing traces: ' + missingTraces.join(', ') + '</span><br>' : '') +
                        '<small>First 5 prefixes: ' + firstPrefixes + '</small><br>' +
                        '<small>Matching traces (first 10): ' + (matchingTraces.length > 0 ? matchingTraces.join(', ') : 'none') + '</small><br>' +
                        (nTraces.length > 0 ? '<small>All N-prefix traces (' + nTraces.length + '): ' + nTraces.slice(0, 3).join(', ') + (nTraces.length > 3 ? '...' : '') + '</small><br>' : '') +
                        (filterVal && filterVal.length > 0 ? (n03Traces.length > 0 ? '<small style="color: green;">Traces matching "' + filterVal + '" (' + n03Traces.length + '): ' + n03Traces.slice(0, 3).join(', ') + (n03Traces.length > 3 ? '...' : '') + '</small><br>' : '<small style="color: red;">No traces found matching "' + filterVal + '"</small><br>') : '') +
                        (Object.keys(methodBreakdown).length > 0 ? '<small>Match breakdown: ' + Object.keys(methodBreakdown).map(k => k + ':' + methodBreakdown[k].length).join(', ') + '</small><br>' : '') +
                        (sampleCodes ? '<br><small>Sample codes from first match: ' + sampleCodes + '</small>' : '');
                }}
                
                console.log('Filter value:', filterValue);
                console.log('Trace info map:', traceInfoMap);
                console.log('Total traces:', nTotalTraces, 'Scatter traces:', nScatterTraces);
                console.log('Visibility array:', visible);
                console.log('Number of visible traces:', visibleCount);
                
                // Apply the visibility update
                const update = {{visible: visible}};
                
                Plotly.restyle(gd, update).then(function() {{
                    // After restyle, ensure axes are locked if we have original ranges
                    if (originalAxisRanges) {{
                        const relayoutUpdate = {{}};
                        for (let axisKey in originalAxisRanges) {{
                            const range = originalAxisRanges[axisKey];
                            relayoutUpdate[axisKey + '.range'] = [range.min, range.max];
                            relayoutUpdate[axisKey + '.autorange'] = false;
                        }}
                        Plotly.relayout(gd, relayoutUpdate);
                    }}
                    console.log('Filter applied successfully');
                    if (debugContent) {{
                        debugContent.innerHTML += '<br><span style="color: green;">✓ Filter applied</span>';
                    }}
                }}).catch(function(err) {{
                    console.error('Error applying filter:', err);
                    if (debugContent) {{
                        debugContent.innerHTML += '<br><span style="color: red;">✗ Error: ' + err.message + '</span>';
                    }}
                }});
            }}
            
            // Function to set x-axis range
            function setXAxisRange(min, max) {{
                const gd = findPlotlyGraphDiv();
                if (!gd) {{
                    console.warn('Could not find Plotly graph div for x-axis update');
                    return;
                }}
                
                // Determine which x-axis to update (could be xaxis, xaxis2, etc. for subplots)
                const update = {{}};
                const layout = gd.layout || {{}};
                
                // Check if we have subplots (multiple xaxes)
                let xAxisKeys = [];
                if (layout.xaxis) {{
                    xAxisKeys.push('xaxis');
                }}
                // Check for additional xaxes (xaxis2, xaxis3, etc.)
                for (let i = 2; i <= 10; i++) {{
                    if (layout['xaxis' + i]) {{
                        xAxisKeys.push('xaxis' + i);
                    }}
                }}
                
                // If no xaxis found, default to 'xaxis'
                if (xAxisKeys.length === 0) {{
                    xAxisKeys = ['xaxis'];
                }}
                
                // Update all xaxes
                for (let key of xAxisKeys) {{
                    if (min !== null && min !== undefined && min !== '') {{
                        update[key + '.range[0]'] = parseFloat(min);
                    }}
                    if (max !== null && max !== undefined && max !== '') {{
                        update[key + '.range[1]'] = parseFloat(max);
                    }}
                    update[key + '.autorange'] = false;
                }}
                
                Plotly.relayout(gd, update).then(function() {{
                    console.log('X-axis range updated');
                }}).catch(function(err) {{
                    console.error('Error updating x-axis range:', err);
                }});
            }}
            
            // Function to reset x-axis to auto
            function resetXAxisAuto() {{
                const gd = findPlotlyGraphDiv();
                if (!gd) {{
                    console.warn('Could not find Plotly graph div for x-axis reset');
                    return;
                }}
                
                const layout = gd.layout || {{}};
                const update = {{}};
                
                // Find all xaxes
                let xAxisKeys = [];
                if (layout.xaxis) {{
                    xAxisKeys.push('xaxis');
                }}
                for (let i = 2; i <= 10; i++) {{
                    if (layout['xaxis' + i]) {{
                        xAxisKeys.push('xaxis' + i);
                    }}
                }}
                
                if (xAxisKeys.length === 0) {{
                    xAxisKeys = ['xaxis'];
                }}
                
                // Set autorange to true for all xaxes
                for (let key of xAxisKeys) {{
                    update[key + '.autorange'] = true;
                }}
                
                Plotly.relayout(gd, update).then(function() {{
                    console.log('X-axis reset to auto');
                }}).catch(function(err) {{
                    console.error('Error resetting x-axis:', err);
                }});
            }}
            
            // Wait for plotly to be ready
            function setupFilter() {{
                const input = document.getElementById('code-filter-input');
                const clearBtn = document.getElementById('code-filter-clear');
                const debugToggle = document.getElementById('code-filter-debug-toggle');
                const debugDiv = document.getElementById('code-filter-debug');
                const xAxisMinInput = document.getElementById('xaxis-min-input');
                const xAxisMaxInput = document.getElementById('xaxis-max-input');
                const xAxisResetBtn = document.getElementById('xaxis-reset-btn');
                const xAxisAutoBtn = document.getElementById('xaxis-auto-btn');
                
                // Real-time filtering on input with debouncing
                if (input) {{
                    let timeoutId = null;
                    input.addEventListener('input', function(e) {{
                        clearTimeout(timeoutId);
                        timeoutId = setTimeout(function() {{
                            filterTraces(e.target.value);
                        }}, 150);
                    }});
                }}
                
                if (clearBtn) {{
                    clearBtn.addEventListener('click', function() {{
                        if (input) {{
                            input.value = '';
                            filterTraces('');
                        }}
                    }});
                }}
                
                // X-axis range controls
                if (xAxisResetBtn) {{
                    xAxisResetBtn.addEventListener('click', function() {{
                        const min = xAxisMinInput ? xAxisMinInput.value : null;
                        const max = xAxisMaxInput ? xAxisMaxInput.value : null;
                        if (min || max) {{
                            setXAxisRange(min, max);
                        }}
                    }});
                }}
                
                if (xAxisAutoBtn) {{
                    xAxisAutoBtn.addEventListener('click', function() {{
                        resetXAxisAuto();
                        // Clear the input fields
                        if (xAxisMinInput) xAxisMinInput.value = '';
                        if (xAxisMaxInput) xAxisMaxInput.value = '';
                    }});
                }}
                
                // Allow Enter key to set x-axis range
                if (xAxisMinInput) {{
                    xAxisMinInput.addEventListener('keypress', function(e) {{
                        if (e.key === 'Enter') {{
                            e.preventDefault();
                            if (xAxisResetBtn) xAxisResetBtn.click();
                        }}
                    }});
                }}
                
                if (xAxisMaxInput) {{
                    xAxisMaxInput.addEventListener('keypress', function(e) {{
                        if (e.key === 'Enter') {{
                            e.preventDefault();
                            if (xAxisResetBtn) xAxisResetBtn.click();
                        }}
                    }});
                }}
                
                if (debugToggle && debugDiv) {{
                    debugToggle.addEventListener('click', function() {{
                        debugDiv.classList.toggle('show');
                    }});
                }}
                
                // Update debug info on load
                const debugContent = document.getElementById('debug-content');
                if (debugContent) {{
                    const totalTraces = nTotalTraces;
                    const scatterTraces = nScatterTraces;
                    const mapEntries = Object.keys(traceInfoMap).length;
                    debugContent.innerHTML = 
                        'Trace prefix map loaded<br>' +
                        'Total traces: ' + totalTraces + '<br>' +
                        'Scatter traces: ' + scatterTraces + '<br>' +
                        'Map entries: ' + mapEntries + '<br>' +
                        '<small>Ready for filtering</small>';
                }}
            }}
            
            // Wait for Plotly to be fully loaded
            function waitForPlotly() {{
                const checkPlotly = setInterval(function() {{
                    const gd = findPlotlyGraphDiv();
                    
                    if (gd && gd.data && Array.isArray(gd.data) && gd.data.length > 0) {{
                        clearInterval(checkPlotly);
                        
                        // Store the original axis ranges before any filtering
                        if (!originalAxisRanges && gd.layout) {{
                            const layout = gd.layout || gd._fullLayout;
                            
                            // Handle subplots (multiple x/y axes)
                            const axisRanges = {{}};
                            let hasRanges = false;
                            
                            // Check for subplot structure
                            if (layout.xaxis && layout.yaxis) {{
                                // Single plot
                                const xaxis = layout.xaxis;
                                const yaxis = layout.yaxis;
                                const xrange = xaxis.range || (xaxis._rl ? [xaxis._rl[0], xaxis._rl[1]] : null);
                                const yrange = yaxis.range || (yaxis._rl ? [yaxis._rl[0], yaxis._rl[1]] : null);
                                
                                if (xrange && yrange) {{
                                    axisRanges['xaxis'] = {{min: xrange[0], max: xrange[1]}};
                                    axisRanges['yaxis'] = {{min: yrange[0], max: yrange[1]}};
                                    hasRanges = true;
                                }}
                            }} else {{
                                // Multiple subplots - find all xaxis and yaxis
                                for (let key in layout) {{
                                    if (key.startsWith('xaxis') && layout[key]) {{
                                        const axis = layout[key];
                                        const range = axis.range || (axis._rl ? [axis._rl[0], axis._rl[1]] : null);
                                        if (range) {{
                                            axisRanges[key] = {{min: range[0], max: range[1]}};
                                            hasRanges = true;
                                        }}
                                    }}
                                    if (key.startsWith('yaxis') && layout[key]) {{
                                        const axis = layout[key];
                                        const range = axis.range || (axis._rl ? [axis._rl[0], axis._rl[1]] : null);
                                        if (range) {{
                                            axisRanges[key] = {{min: range[0], max: range[1]}};
                                            hasRanges = true;
                                        }}
                                    }}
                                }}
                            }}
                            
                            if (hasRanges) {{
                                originalAxisRanges = axisRanges;
                                
                                // Disable auto-ranging for all axes
                                const relayoutUpdate = {{}};
                                for (let key in axisRanges) {{
                                    relayoutUpdate[key + '.autorange'] = false;
                                }}
                                Plotly.relayout(gd, relayoutUpdate);
                            }}
                        }}
                        
                        // Update debug to show success
                        const debugContent = document.getElementById('debug-content');
                        if (debugContent) {{
                            debugContent.innerHTML = 
                                'Trace prefix map loaded<br>' +
                                'Total traces: ' + nTotalTraces + '<br>' +
                                'Scatter traces: ' + nScatterTraces + '<br>' +
                                'Map entries: ' + Object.keys(traceInfoMap).length + '<br>' +
                                '<span style="color: green;">✓ Plotly graph found (' + gd.data.length + ' traces)</span><br>' +
                                (originalAxisRanges ? '<small>Axis ranges locked</small><br>' : '') +
                                '<small>Ready for filtering</small>';
                        }}
                        setupFilter();
                    }}
                }}, 100);
                
                // Give up after 10 seconds
                setTimeout(function() {{
                    clearInterval(checkPlotly);
                    const gd = findPlotlyGraphDiv();
                    const debugContent = document.getElementById('debug-content');
                    if (debugContent) {{
                        if (gd && gd.data) {{
                            debugContent.innerHTML = 
                                'Trace prefix map loaded<br>' +
                                'Total traces: ' + nTotalTraces + '<br>' +
                                'Scatter traces: ' + nScatterTraces + '<br>' +
                                'Map entries: ' + Object.keys(traceInfoMap).length + '<br>' +
                                '<span style="color: orange;">⚠ Timeout - graph may not be ready</span><br>' +
                                '<small>Found ' + (gd.data ? gd.data.length : 0) + ' traces</small>';
                        }} else {{
                            debugContent.innerHTML = 
                                '<span style="color: red;">ERROR: Could not find Plotly graph after 10 seconds</span><br>' +
                                'Plotly divs found: ' + document.querySelectorAll('[id^="plotly"]').length;
                        }}
                    }}
                    setupFilter();
                }}, 10000);
            }}
            
            // Try to setup immediately, and also on load
            if (document.readyState === 'loading') {{
                window.addEventListener('load', waitForPlotly);
            }} else {{
                waitForPlotly();
            }}
        }})();
        </script>
        """

        # Inject the script before the closing body tag
        if "</body>" in html_content:
            html_content = html_content.replace("</body>", filter_script + "</body>")
        else:
            # If no body tag, append before closing html
            html_content = html_content.replace("</html>", filter_script + "</html>")

        # Write the modified HTML
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        # Remove temporary file
        temp_html.unlink()
    else:
        # No filtering, just save normally
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
