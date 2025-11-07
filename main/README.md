# Volcano Plot Script

## Overview

The `create_volcano_plot.py` script creates volcano plots for treatment effect analysis. It supports both Risk Difference (RD) and Risk Ratio (RR) analyses with configurable parameters via command-line interface.

## Usage

### Basic Usage

```bash
# Create volcano plot with Risk Difference (default)
python main/create_volcano_plot.py

# Create volcano plot with Risk Ratio
python main/create_volcano_plot.py --effect-type RR

# Create volcano plot with log Risk Ratio
python main/create_volcano_plot.py --effect-type log-RR
```

### Command-line Arguments

- `--input-dir PATH`: Directory containing input data files (default: `data/semaglutide`)
- `--output-dir PATH`: Directory for output figures (default: `figures`)
- `--effect-type {RD,RR,log-RR}`: Effect measure to use (default: `RD`)
  - `RD`: Risk Difference
  - `RR`: Risk Ratio (linear scale)
  - `log-RR`: Risk Ratio (logarithmic scale)
- `--diagnostics` / `--no-diagnostics`: Enable/disable diagnostic analyses (default: enabled)

### Examples

```bash
# Custom input/output directories
python main/create_volcano_plot.py --input-dir data/study1 --output-dir results/study1

# Risk ratio without diagnostics
python main/create_volcano_plot.py --effect-type RR --no-diagnostics

# Log risk ratio with custom output directory
python main/create_volcano_plot.py --effect-type log-RR --output-dir figures/log_rr_analysis
```

## Output Files

The script generates the following files (with suffix based on effect type):

### Matplotlib Figures
- `volcano_plot_{effect_type}.png` - Main volcano plot (PNG, 300 DPI)
- `volcano_plot_{effect_type}.pdf` - Main volcano plot (PDF, vector)
- `volcano_plot_tmle_ipw_overlay_{effect_type}.png` - Overlay comparison plot

### Interactive Plotly Figures
- `volcano_plot_{effect_type}_interactive.html` - Interactive volcano plot
- `volcano_plot_{effect_type}_interactive.png` - Snapshot of interactive plot
- `volcano_plot_tmle_ipw_overlay_{effect_type}_interactive.html` - Interactive overlay

### Diagnostic Files
- `tmle_ipw_significance_confusion_{effect_type}.csv` - Confusion matrix
- `tmle_ipw_significance_confusion_{effect_type}_summary.txt` - Summary statistics

### Additional Files (RD only)
- `volcano_plot_{effect_type}_truncated.png` - Truncated plot excluding extreme p-values

## Input Data Requirements

The script expects the following files in the input directory:

- `combined_estimatest.txt`: Arm-level effect estimates with confidence intervals
- `combined_stats.txt`: Prevalence statistics

Required columns in estimates file:
- `method`: Statistical method (must include "IPW" and "TMLE")
- `outcome`: Outcome identifier
- `run_id`: Run identifier for multiple runs
- `effect_1`, `effect_0`: Treatment and control arm probabilities
- `effect_1_CI95_lower`, `effect_1_CI95_upper`: Treatment arm 95% CI bounds
- `effect_0_CI95_lower`, `effect_0_CI95_upper`: Control arm 95% CI bounds

## Diagnostics

When diagnostics are enabled (default), the script outputs:
- P-value distribution analysis
- Z-statistic statistics
- Standard error diagnostics
- Extreme case inspection
- Deep dive analysis (RD only)

To disable diagnostics for faster execution:
```bash
python main/create_volcano_plot.py --no-diagnostics
```

## Notes

- Alpha threshold is fixed at 0.05 (not configurable via CLI)
- Only IPW and TMLE methods are analyzed (methods with arm-level estimates)
- RD uses inverse variance pooling on logit-transformed probabilities
- RR uses DerSimonian-Laird random effects with Hartung-Knapp-Sidik-Jonkman adjustment

