"""
Unit tests for the statistics module.

Tests cover utility functions, arm metrics computation, pooling,
risk difference inference, and end-to-end workflows.
"""

import unittest
import numpy as np
import pandas as pd
from scipy.stats import norm
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trace.statistics import (
    _clip_prob,
    logit,
    inv_logit,
    se_from_prob_ci_on_logit,
    add_logit_arm_metrics,
    pool_arm_logits,
    rd_inference_from_arm_logits,
    compute_rd_pvalues,
)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions for probability transformations."""
    
    def test_clip_prob_scalar(self):
        """Test clipping of scalar probabilities."""
        # Test values that should be clipped
        self.assertGreater(_clip_prob(0.0), 0.0)
        self.assertLess(_clip_prob(1.0), 1.0)
        
        # Test normal values (should be unchanged)
        np.testing.assert_almost_equal(_clip_prob(0.5), 0.5)
        np.testing.assert_almost_equal(_clip_prob(0.1), 0.1)
    
    def test_clip_prob_array(self):
        """Test clipping of probability arrays."""
        p = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
        clipped = _clip_prob(p)
        
        # Check no exact 0 or 1
        self.assertTrue(np.all(clipped > 0))
        self.assertTrue(np.all(clipped < 1))
        
        # Check middle values unchanged
        np.testing.assert_almost_equal(clipped[1:4], p[1:4])
    
    def test_logit_inv_logit_inverse(self):
        """Test that logit and inv_logit are inverses."""
        p = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        recovered = inv_logit(logit(p))
        np.testing.assert_array_almost_equal(p, recovered, decimal=6)
    
    def test_logit_known_values(self):
        """Test logit with known values."""
        # logit(0.5) = 0
        np.testing.assert_almost_equal(logit(0.5), 0.0)
        
        # logit(p) = -logit(1-p)
        p = 0.7
        np.testing.assert_almost_equal(logit(p), -logit(1-p))
    
    def test_inv_logit_known_values(self):
        """Test inv_logit with known values."""
        # inv_logit(0) = 0.5
        np.testing.assert_almost_equal(inv_logit(0), 0.5)
        
        # inv_logit(large positive) ≈ 1
        self.assertGreater(inv_logit(10), 0.99)
        
        # inv_logit(large negative) ≈ 0
        self.assertLess(inv_logit(-10), 0.01)
    
    def test_se_from_prob_ci_basic(self):
        """Test SE calculation from probability CI."""
        # For 95% CI: width = 2 * 1.96 * SE (on logit scale)
        # If we have symmetric CI around 0.5, we can verify approximately
        lo, hi = 0.4, 0.6
        se = se_from_prob_ci_on_logit(lo, hi)
        
        # SE should be positive
        self.assertGreater(se, 0)
        
        # Wider CI should give larger SE
        se_wide = se_from_prob_ci_on_logit(0.3, 0.7)
        self.assertGreater(se_wide, se)
    
    def test_se_from_prob_ci_array(self):
        """Test SE calculation with arrays."""
        lo = np.array([0.4, 0.3])
        hi = np.array([0.6, 0.7])
        se = se_from_prob_ci_on_logit(lo, hi)
        
        self.assertEqual(len(se), 2)
        self.assertTrue(np.all(se > 0))
        # Second CI is wider, should have larger SE
        self.assertGreater(se[1], se[0])


class TestArmMetrics(unittest.TestCase):
    """Test add_logit_arm_metrics function."""
    
    def setUp(self):
        """Create test data."""
        self.df = pd.DataFrame({
            "effect_1": [0.6, 0.5],
            "effect_1_CI95_lower": [0.55, 0.45],
            "effect_1_CI95_upper": [0.65, 0.55],
            "effect_0": [0.4, 0.3],
            "effect_0_CI95_lower": [0.35, 0.25],
            "effect_0_CI95_upper": [0.45, 0.35],
        })
    
    def test_columns_added(self):
        """Test that all expected columns are added."""
        result = add_logit_arm_metrics(self.df)
        
        expected_cols = [
            "eta1", "se_eta1",
            "effect_1_logit_CI95_lower", "effect_1_logit_CI95_upper",
            "eta0", "se_eta0",
            "effect_0_logit_CI95_lower", "effect_0_logit_CI95_upper",
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
        self.df = pd.DataFrame({
            "method": ["IPW", "IPW", "TMLE", "TMLE"],
            "outcome": ["A", "A", "A", "A"],
            "eta": [0.5, 0.6, 0.4, 0.5],
            "se": [0.1, 0.15, 0.12, 0.13],
        })
    
    def test_grouping_correct(self):
        """Test that grouping produces correct number of rows."""
        result = pool_arm_logits(
            self.df,
            group_cols=["method", "outcome"],
            eta_col="eta",
            se_col="se",
            out_prefix="pooled"
        )
        
        # Should have 2 groups (IPW and TMLE)
        self.assertEqual(len(result), 2)
    
    def test_inverse_variance_weighting(self):
        """Test that inverse-variance weighting is correct."""
        # Simple case: two estimates with known SEs
        df = pd.DataFrame({
            "group": ["A", "A"],
            "eta": [1.0, 2.0],
            "se": [1.0, 1.0],  # Equal weights
        })
        
        result = pool_arm_logits(
            df,
            group_cols="group",
            eta_col="eta",
            se_col="se",
            out_prefix="pooled"
        )
        
        # With equal SEs, pooled estimate should be simple average
        np.testing.assert_almost_equal(result.loc[0, "pooled"], 1.5)
    
    def test_pooled_se_smaller(self):
        """Test that pooled SE is smaller than individual SEs."""
        result = pool_arm_logits(
            self.df,
            group_cols="method",
            eta_col="eta",
            se_col="se",
            out_prefix="pooled"
        )
        
        # Pooled SE should be smaller than individual SEs (for same method)
        ipw_row = result[result["method"] == "IPW"].iloc[0]
        original_ses = self.df[self.df["method"] == "IPW"]["se"].values
        
        self.assertLess(ipw_row["pooled_se"], min(original_ses))
    
    def test_n_runs_counted(self):
        """Test that n_runs_used is correctly computed."""
        result = pool_arm_logits(
            self.df,
            group_cols="method",
            eta_col="eta",
            se_col="se",
            out_prefix="pooled"
        )
        
        # Each method has 2 runs
        self.assertTrue(np.all(result["n_runs_used"] == 2))
    
    def test_nan_handling(self):
        """Test handling of NaN values."""
        df = pd.DataFrame({
            "group": ["A", "A", "A"],
            "eta": [1.0, np.nan, 2.0],
            "se": [0.1, 0.1, 0.1],
        })
        
        result = pool_arm_logits(
            df,
            group_cols="group",
            eta_col="eta",
            se_col="se",
            out_prefix="pooled"
        )
        
        # Should use only 2 runs (excluding NaN)
        self.assertEqual(result.loc[0, "n_runs_used"], 2)
    
    def test_empty_group(self):
        """Test handling when all values in group are NaN."""
        df = pd.DataFrame({
            "group": ["A", "A"],
            "eta": [np.nan, np.nan],
            "se": [0.1, 0.1],
        })
        
        result = pool_arm_logits(
            df,
            group_cols="group",
            eta_col="eta",
            se_col="se",
            out_prefix="pooled"
        )
        
        # Should return NaN and n_runs_used = 0
        self.assertTrue(np.isnan(result.loc[0, "pooled"]))
        self.assertEqual(result.loc[0, "n_runs_used"], 0)


class TestRDInference(unittest.TestCase):
    """Test rd_inference_from_arm_logits function."""
    
    def test_basic_calculation(self):
        """Test basic RD calculation."""
        # Simple case: p1=0.6, p0=0.4, so RD = 0.2
        eta1 = logit(0.6)
        eta0 = logit(0.4)
        se1 = 0.1
        se0 = 0.1
        
        result = rd_inference_from_arm_logits(eta1, se1, eta0, se0)
        
        # Check RD is approximately 0.2
        np.testing.assert_almost_equal(result["RD"], 0.2, decimal=5)
        
        # Check that all expected keys are present
        expected_keys = ["RD", "SE_RD", "z", "p_value",
                         "RD_CI95_lower", "RD_CI95_upper",
                         "p1_hat", "p0_hat"]
        for key in expected_keys:
            self.assertIn(key, result)
    
    def test_array_input(self):
        """Test with array inputs."""
        eta1 = np.array([logit(0.6), logit(0.7)])
        eta0 = np.array([logit(0.4), logit(0.5)])
        se1 = np.array([0.1, 0.1])
        se0 = np.array([0.1, 0.1])
        
        result = rd_inference_from_arm_logits(eta1, se1, eta0, se0)
        
        # Check shapes
        self.assertEqual(len(result["RD"]), 2)
        self.assertEqual(len(result["p_value"]), 2)
    
    def test_null_hypothesis(self):
        """Test when p1 = p0 (null hypothesis)."""
        # Both arms have same probability
        eta1 = logit(0.5)
        eta0 = logit(0.5)
        se1 = 0.1
        se0 = 0.1
        
        result = rd_inference_from_arm_logits(eta1, se1, eta0, se0)
        
        # RD should be very close to 0
        np.testing.assert_almost_equal(result["RD"], 0.0, decimal=5)
        
        # p-value should be close to 1 (non-significant)
        self.assertGreater(result["p_value"], 0.1)
    
    def test_ci_contains_rd(self):
        """Test that confidence interval contains point estimate."""
        eta1 = logit(0.6)
        eta0 = logit(0.4)
        se1 = 0.1
        se0 = 0.1
        
        result = rd_inference_from_arm_logits(eta1, se1, eta0, se0)
        
        # CI should contain RD
        self.assertLess(result["RD_CI95_lower"], result["RD"])
        self.assertGreater(result["RD_CI95_upper"], result["RD"])
    
    def test_large_difference_significant(self):
        """Test that large differences are statistically significant."""
        # Large difference with small SEs
        eta1 = logit(0.7)
        eta0 = logit(0.3)
        se1 = 0.05
        se0 = 0.05
        
        result = rd_inference_from_arm_logits(eta1, se1, eta0, se0)
        
        # Should be highly significant
        self.assertLess(result["p_value"], 0.001)
        self.assertGreater(np.abs(result["z"]), 3)


class TestComputeRDPvalues(unittest.TestCase):
    """Test compute_rd_pvalues end-to-end function."""
    
    def setUp(self):
        """Create test data."""
        self.df = pd.DataFrame({
            "method": ["IPW", "IPW", "TMLE", "TMLE"],
            "outcome": ["A", "A", "A", "A"],
            "run_id": ["run1", "run2", "run1", "run2"],
            "effect_1": [0.6, 0.62, 0.58, 0.60],
            "effect_1_CI95_lower": [0.55, 0.57, 0.53, 0.55],
            "effect_1_CI95_upper": [0.65, 0.67, 0.63, 0.65],
            "effect_0": [0.4, 0.42, 0.38, 0.40],
            "effect_0_CI95_lower": [0.35, 0.37, 0.33, 0.35],
            "effect_0_CI95_upper": [0.45, 0.47, 0.43, 0.45],
        })
    
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
        self.assertIn("n_runs_used_x", result.columns)
        self.assertIn("n_runs_used_y", result.columns)
        
        # Should have 2 runs per method
        self.assertTrue(np.all(result["n_runs_used_x"] == 2))
    
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


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_single_row(self):
        """Test with single row input."""
        df = pd.DataFrame({
            "effect_1": [0.6],
            "effect_1_CI95_lower": [0.55],
            "effect_1_CI95_upper": [0.65],
            "effect_0": [0.4],
            "effect_0_CI95_lower": [0.35],
            "effect_0_CI95_upper": [0.45],
        })
        
        result = compute_rd_pvalues(df, group_cols=None)
        
        self.assertEqual(len(result), 1)
        self.assertIn("RD", result.columns)
    
    def test_extreme_probabilities(self):
        """Test with probabilities close to 0 or 1."""
        df = pd.DataFrame({
            "effect_1": [0.95],
            "effect_1_CI95_lower": [0.90],
            "effect_1_CI95_upper": [0.98],
            "effect_0": [0.05],
            "effect_0_CI95_lower": [0.02],
            "effect_0_CI95_upper": [0.10],
        })
        
        result = compute_rd_pvalues(df, group_cols=None)
        
        # Should not produce NaN or inf
        self.assertTrue(np.isfinite(result["RD"].values[0]))
        self.assertTrue(np.isfinite(result["p_value"].values[0]))


if __name__ == "__main__":
    unittest.main()

