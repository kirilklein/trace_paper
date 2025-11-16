# Example Test Data

This directory contains minimal test data for CI/CD pipeline testing.

## Files

- `combined_estimates.txt` - Treatment effect estimates from IPW, TMLE, RD, and RR methods
- `combined_stats.txt` - Prevalence statistics (treated, untreated, total counts)

## Structure

- **10 outcomes** across 3 ATC groups:
  - A01 (Digestive): A01AA, A01AB, A01AD
  - A02 (Digestive): A02AA, A02BC
  - B01 (Blood): B01AA, B01AC
  - B02 (Blood): B02AB
- **2 runs** per outcome: `test_run_01`, `test_run_02`
- **4 methods** per outcome-run: IPW, TMLE, RD, RR
- **Total**: 80 rows in estimates, 50 rows in stats

## Data Characteristics

The data is synthetic but realistic:

- Sample sizes: ~50,000 untreated, ~5,000 treated per run
- Prevalence ranges from <1% to ~38%
- Risk differences range from -0.0015 to 0.154
- Risk ratios range from 0.93 to 1.72
- Includes both positive and negative effects
- IPW and TMLE have arm-level confidence intervals
- RD and RR have aggregate confidence intervals only

## Usage

This data is used by the GitHub Actions CI pipeline to test:

1. `compute_pooled_pvalues.py` - Pooling and p-value computation
2. `create_volcano_plot.py` - Volcano plot generation
3. `create_manhattan_plot.py` - Manhattan-style plot generation

The small size ensures fast CI runs while still testing all code paths.
