"""Unit tests for compute_rd_pvalues_bootstrap."""

import unittest
import numpy as np
import pandas as pd

from trace.statistics import compute_rd_pvalues_bootstrap


class TestComputeRDPvaluesBootstrap(unittest.TestCase):
    """Test compute_rd_pvalues_bootstrap function."""

    def setUp(self):
        """Create test data: 2 methods x 2 bootstrap replicates.

        RDs per replicate:
          IPW:  0.60-0.40=0.20, 0.62-0.40=0.22
          TMLE: 0.58-0.38=0.20, 0.60-0.38=0.22
        """
        self.df = pd.DataFrame(
            {
                "method": ["IPW", "IPW", "TMLE", "TMLE"],
                "outcome": ["A", "A", "A", "A"],
                "effect_1": [0.60, 0.62, 0.58, 0.60],
                "effect_0": [0.40, 0.40, 0.38, 0.38],
            }
        )

    def test_output_columns(self):
        """Output has all required columns."""
        result = compute_rd_pvalues_bootstrap(self.df)
        required = [
            "method", "outcome", "RD", "SE_RD",
            "RD_CI95_lower", "RD_CI95_upper",
            "p1_hat", "p0_hat", "z", "p_value",
            "n_runs_shared", "df_logit",
        ]
        for col in required:
            self.assertIn(col, result.columns)

    def test_one_row_per_group(self):
        """Should produce one row per (method, outcome) group."""
        result = compute_rd_pvalues_bootstrap(self.df)
        self.assertEqual(len(result), 2)

    def test_rd_is_mean_of_differences(self):
        """RD should be mean(effect_1 - effect_0) per group."""
        result = compute_rd_pvalues_bootstrap(self.df)
        ipw = result[result["method"] == "IPW"].iloc[0]
        rd_b = np.array([0.60 - 0.40, 0.62 - 0.40])
        np.testing.assert_almost_equal(ipw["RD"], np.mean(rd_b))

    def test_se_is_std_not_divided_by_sqrt_b(self):
        """SE should be std(RD_b, ddof=1), NOT divided by sqrt(B)."""
        result = compute_rd_pvalues_bootstrap(self.df)
        ipw = result[result["method"] == "IPW"].iloc[0]
        rd_b = np.array([0.60 - 0.40, 0.62 - 0.40])
        expected_se = float(np.std(rd_b, ddof=1))
        np.testing.assert_almost_equal(ipw["SE_RD"], expected_se)

    def test_p1_hat_p0_hat(self):
        """p1_hat and p0_hat should be means of effect columns."""
        result = compute_rd_pvalues_bootstrap(self.df)
        ipw = result[result["method"] == "IPW"].iloc[0]
        np.testing.assert_almost_equal(ipw["p1_hat"], np.mean([0.60, 0.62]))
        np.testing.assert_almost_equal(ipw["p0_hat"], np.mean([0.40, 0.40]))

    def test_n_runs_shared(self):
        """n_runs_shared should equal number of replicates in group."""
        result = compute_rd_pvalues_bootstrap(self.df)
        self.assertTrue(np.all(result["n_runs_shared"] == 2))

    def test_df_logit_is_b_minus_1(self):
        """df_logit should be B - 1."""
        result = compute_rd_pvalues_bootstrap(self.df)
        self.assertTrue(np.all(result["df_logit"] == 1.0))

    def test_ci_contains_rd(self):
        """CI should contain the point estimate."""
        result = compute_rd_pvalues_bootstrap(self.df)
        for _, row in result.iterrows():
            self.assertLess(row["RD_CI95_lower"], row["RD"])
            self.assertGreater(row["RD_CI95_upper"], row["RD"])

    def test_single_replicate_returns_nan(self):
        """With only 1 replicate, SE and p_value should be NaN."""
        df = pd.DataFrame(
            {
                "method": ["IPW"],
                "outcome": ["A"],
                "effect_1": [0.60],
                "effect_0": [0.40],
            }
        )
        result = compute_rd_pvalues_bootstrap(df)
        self.assertEqual(len(result), 1)
        np.testing.assert_almost_equal(result.iloc[0]["RD"], 0.20)
        self.assertTrue(np.isnan(result.iloc[0]["SE_RD"]))
        self.assertTrue(np.isnan(result.iloc[0]["p_value"]))

    def test_string_group_cols(self):
        """Should work with a single string group_cols."""
        result = compute_rd_pvalues_bootstrap(self.df, group_cols="method")
        self.assertEqual(len(result), 2)
        self.assertIn("method", result.columns)


if __name__ == "__main__":
    unittest.main()
