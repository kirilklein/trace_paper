# TRACE Paper Analysis Pipeline

Risk difference analysis and volcano plot visualization for the TRACE paper.

## Installation

Install the package in editable mode:

```bash
pip install -e .
```

Or with development dependencies:

```bash
pip install -e ".[dev]"
```

## Usage

### Computing Risk Differences

```python
from code.statistics import compute_rd_pvalues
import pandas as pd

# Load your data
df = pd.read_csv('data/semaglutide/combined_estimatest.txt', index_col=0)

# Compute pooled risk differences and p-values
df_pooled = compute_rd_pvalues(df, group_cols=['method', 'outcome'])
```

### Creating Volcano Plots

```python
from code.plotting.volcano import prepare_volcano_data, volcano_plot_per_method

# Prepare data for volcano plot
df_volcano = prepare_volcano_data(
    df_pooled,
    rd_col="RD",
    p_col="p_value",
    method_col="method",
    outcome_col="outcome",
    adjust="bh",
    adjust_per="by_method"
)

# Create volcano plot
fig, axes = volcano_plot_per_method(
    df_volcano,
    alpha=0.05,
    figsize_per_panel=(7, 5)
)
```

## Project Structure

- `code/`: Core analysis modules
  - `statistics.py`: Risk difference calculations using delta method
  - `plotting/volcano.py`: Volcano plot visualization
- `data/`: Input data files
- `examples/`: Example scripts
- `tests/`: Unit tests
- `figures/`: Output figures

