"""
Test script to verify all pooling methods work correctly
"""

import pandas as pd
import numpy as np
from trace.statistics import compute_rd_pvalues

# Create a simple test dataset
np.random.seed(42)
n_runs = 5
n_outcomes = 3

data = []
for method in ["IPW", "TMLE"]:
    for outcome_id in range(n_outcomes):
        for run_id in range(n_runs):
            # Simulate probabilities
            p1 = 0.2 + np.random.normal(0, 0.02)
            p0 = 0.15 + np.random.normal(0, 0.02)

            # Simulate CI widths (realistic for these probabilities)
            ci_width_1 = 0.05 + np.random.normal(0, 0.01)
            ci_width_0 = 0.04 + np.random.normal(0, 0.01)

            data.append(
                {
                    "method": method,
                    "outcome": f"outcome_{outcome_id}",
                    "run_id": run_id,
                    "effect_1": np.clip(p1, 0.01, 0.99),
                    "effect_0": np.clip(p0, 0.01, 0.99),
                    "effect_1_CI95_lower": np.clip(p1 - ci_width_1 / 2, 0.01, 0.99),
                    "effect_1_CI95_upper": np.clip(p1 + ci_width_1 / 2, 0.01, 0.99),
                    "effect_0_CI95_lower": np.clip(p0 - ci_width_0 / 2, 0.01, 0.99),
                    "effect_0_CI95_upper": np.clip(p0 + ci_width_0 / 2, 0.01, 0.99),
                }
            )

df_test = pd.DataFrame(data)

print("Test dataset created:")
print(f"  Methods: {df_test['method'].unique()}")
print(f"  Outcomes: {df_test['outcome'].nunique()}")
print(f"  Runs per method-outcome: {n_runs}")
print()

# Test all arm pooling methods
methods_to_test = [
    "random_effects_hksj",
    "rubins_rules",
    "inter_intra_variance",
    "simple_mean",
]

for pooling_method in methods_to_test:
    print("=" * 70)
    print(f"Testing arm pooling method: {pooling_method}")
    print("=" * 70)

    try:
        result = compute_rd_pvalues(
            df_test,
            group_cols=["method", "outcome"],
            arm_pooling=pooling_method,
            verbose=False,
        )

        print(f"✓ Success! Generated {len(result)} pooled estimates")
        print(f"  Columns: {list(result.columns)}")
        print(f"  RD range: [{result['RD'].min():.4f}, {result['RD'].max():.4f}]")
        print(
            f"  SE_RD range: [{result['SE_RD'].min():.4e}, {result['SE_RD'].max():.4e}]"
        )
        print(
            f"  p-value range: [{result['p_value'].min():.4e}, {result['p_value'].max():.4e}]"
        )

        # Check for required columns
        required_cols = [
            "method",
            "outcome",
            "RD",
            "SE_RD",
            "z",
            "p_value",
            "RD_CI95_lower",
            "RD_CI95_upper",
        ]
        missing = [c for c in required_cols if c not in result.columns]
        if missing:
            print(f"  ⚠ Warning: Missing columns: {missing}")
        else:
            print(f"  ✓ All required columns present")

        print()

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        print()

print("=" * 70)
print("Test complete!")
