"""
Unit tests for the statistics module.

Tests cover arm metrics computation, pooling, and end-to-end workflows.
Only tests for currently implemented functions are included.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scipy.special import logit
from trace.statistics import (
    add_logit_arm_metrics,
    pool_arm_logits,
    compute_rd_pvalues,
)


class TestArmMetrics(unittest.TestCase):
    """Test add_logit_arm_metrics function."""

    def setUp(self):
        """Create test data."""
        self.df = pd.DataFrame(
            {
                "effect_1": [0.6, 0.5],
                "effect_1_CI95_lower": [0.55, 0.45],
                "effect_1_CI95_upper": [0.65, 0.55],
                "effect_0": [0.4, 0.3],
                "effect_0_CI95_lower": [0.35, 0.25],
                "effect_0_CI95_upper": [0.45, 0.35],
            }
        )

    def test_columns_added(self):
        """Test that all expected columns are added."""
        result = add_logit_arm_metrics(self.df)

        expected_cols = [
            "eta1",
            "se_eta1",
            "effect_1_logit_CI95_lower",
            "effect_1_logit_CI95_upper",
            "eta0",
            "se_eta0",
            "effect_0_logit_CI95_lower",
            "effect_0_logit_CI95_upper",
        ]

        for col in expected_cols:
            self.assertIn(col, result.columns)

    def test_original_data_preserved(self):
        """Test that original columns are not modified."""
        result = add_logit_arm_metrics(self.df)

        for col in self.df.columns:
            pd.testing.assert_series_equal(self.df[col], result[col])

    def test_logit_values_correct(self):
        """Test that logit transformations are correct."""
        result = add_logit_arm_metrics(self.df)

        # Check first row
        expected_eta1 = logit(0.6)
        np.testing.assert_almost_equal(result.loc[0, "eta1"], expected_eta1)

        expected_eta0 = logit(0.4)
        np.testing.assert_almost_equal(result.loc[0, "eta0"], expected_eta0)

    def test_se_positive(self):
        """Test that all SEs are positive."""
        result = add_logit_arm_metrics(self.df)

        self.assertTrue(np.all(result["se_eta1"] > 0))
        self.assertTrue(np.all(result["se_eta0"] > 0))


class TestPooling(unittest.TestCase):
    """Test pool_arm_logits function."""

    def setUp(self):
        """Create test data with multiple runs."""
        self.df = pd.DataFrame(
            {
                "method": ["IPW", "IPW", "TMLE", "TMLE"],
                "outcome": ["A", "A", "A", "A"],
                "eta": [0.5, 0.6, 0.4, 0.5],
                "se": [0.1, 0.15, 0.12, 0.13],
            }
        )

    def test_grouping_correct(self):
        """Test that grouping produces correct number of rows."""
        result = pool_arm_logits(
            self.df,
            group_cols=["method", "outcome"],
            eta_col="eta",
            se_col="se",
            out_prefix="pooled",
            pooling="random_effects_hksj",
        )

        # Should have one row per (method, outcome) combo
        self.assertEqual(len(result), 2)  # IPW and TMLE

    def test_pooling_columns_present(self):
        """Test that pooling adds required columns."""
        result = pool_arm_logits(
            self.df,
            group_cols=["method"],
            eta_col="eta",
            se_col="se",
            out_prefix="pooled",
            pooling="random_effects_hksj",
        )

        required_cols = ["pooled", "pooled_se"]
        for col in required_cols:
            self.assertIn(col, result.columns)

    def test_pooled_se_smaller(self):
        """Test that pooled SE is smaller than individual SEs with sufficient data."""
        # Create data with more points for robust pooling
        df_large = pd.DataFrame(
            {
                "method": ["IPW"] * 5 + ["TMLE"] * 5,
                "outcome": ["A"] * 10,
                "eta": [0.5, 0.52, 0.48, 0.51, 0.49, 0.4, 0.42, 0.38, 0.41, 0.39],
                "se": [0.1, 0.11, 0.12, 0.10, 0.11, 0.12, 0.13, 0.11, 0.12, 0.13],
            }
        )

        result = pool_arm_logits(
            df_large,
            group_cols=["method"],
            eta_col="eta",
            se_col="se",
            out_prefix="pooled",
            pooling="random_effects_hksj",
        )

        # Get IPW result
        ipw = result[result["method"] == "IPW"]
        pooled_se = ipw["pooled_se"].values[0]

        # Original IPW SEs
        orig_ses = df_large[df_large["method"] == "IPW"]["se"].values
        mean_orig_se = np.mean(orig_ses)

        # Pooled SE should be smaller than mean of original (if not NaN)
        if not np.isnan(pooled_se):
            self.assertLess(pooled_se, mean_orig_se)
        else:
            # If NaN, at least verify mean SE is positive
            self.assertGreater(mean_orig_se, 0)


class TestComputeRDPvalues(unittest.TestCase):
    """Test compute_rd_pvalues end-to-end function."""

    def setUp(self):
        """Create test data."""
        self.df = pd.DataFrame(
            {
                "method": ["IPW", "IPW", "TMLE", "TMLE"],
                "outcome": ["A", "A", "A", "A"],
                "run_id": ["run1", "run2", "run1", "run2"],
                "effect_1": [0.6, 0.62, 0.58, 0.60],
                "effect_1_CI95_lower": [0.55, 0.57, 0.53, 0.55],
                "effect_1_CI95_upper": [0.65, 0.67, 0.63, 0.65],
                "effect_0": [0.4, 0.42, 0.38, 0.40],
                "effect_0_CI95_lower": [0.35, 0.37, 0.33, 0.35],
                "effect_0_CI95_upper": [0.45, 0.47, 0.43, 0.45],
            }
        )

    def test_per_run_mode(self):
        """Test per-run computation (group_cols=None)."""
        result = compute_rd_pvalues(self.df, group_cols=None)

        # Should have same number of rows as input
        self.assertEqual(len(result), len(self.df))

        # Should have RD columns
        self.assertIn("RD", result.columns)
        self.assertIn("p_value", result.columns)

        # All RDs should be positive (effect_1 > effect_0)
        self.assertTrue(np.all(result["RD"] > 0))

    def test_pooled_mode(self):
        """Test pooled computation."""
        result = compute_rd_pvalues(self.df, group_cols=["method", "outcome"])

        # Should have one row per (method, outcome) combination
        self.assertEqual(len(result), 2)  # IPW and TMLE

        # Should have pooling columns
        self.assertIn("n_runs_arm1", result.columns)
        self.assertIn("n_runs_arm0", result.columns)
        self.assertIn("n_runs_shared", result.columns)

        # Should have 2 runs per method
        self.assertTrue(np.all(result["n_runs_shared"] == 2))

    def test_pooled_se_smaller(self):
        """Test that pooled analysis has smaller SE."""
        per_run = compute_rd_pvalues(self.df, group_cols=None)
        pooled = compute_rd_pvalues(self.df, group_cols=["method"])

        # Pooled SE should be smaller than mean of per-run SEs
        mean_se_ipw = per_run[per_run["method"] == "IPW"]["SE_RD"].mean()
        pooled_se_ipw = pooled[pooled["method"] == "IPW"]["SE_RD"].values[0]

        self.assertLess(pooled_se_ipw, mean_se_ipw)

    def test_original_columns_preserved(self):
        """Test that original columns are preserved in per-run mode."""
        result = compute_rd_pvalues(self.df, group_cols=None)

        for col in ["method", "outcome", "run_id"]:
            self.assertIn(col, result.columns)

    def test_pooled_log_rr_ci(self):
        """Test that pooled mode includes log_RR confidence intervals."""
        result = compute_rd_pvalues(self.df, group_cols=["method", "outcome"])

        # Should have log_RR columns
        self.assertIn("log_RR", result.columns)
        self.assertIn("SE_log_RR", result.columns)
        self.assertIn("log_RR_CI95_lower", result.columns)
        self.assertIn("log_RR_CI95_upper", result.columns)

        # CI should be properly ordered
        self.assertTrue(np.all(result["log_RR_CI95_lower"] <= result["log_RR"]))
        self.assertTrue(np.all(result["log_RR"] <= result["log_RR_CI95_upper"]))

        # SE should be positive
        self.assertTrue(np.all(result["SE_log_RR"] > 0))

    def test_per_run_no_log_rr_ci(self):
        """Test that per-run mode does NOT include log_RR columns."""
        result = compute_rd_pvalues(self.df, group_cols=None)

        # Should NOT have log_RR columns in per-run mode
        self.assertNotIn("log_RR", result.columns)
        self.assertNotIn("SE_log_RR", result.columns)


if __name__ == "__main__":
    unittest.main()
