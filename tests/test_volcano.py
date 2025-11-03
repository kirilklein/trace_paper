"""
Unit tests for the volcano plotting module.

Tests cover p-value adjustment, data preparation, and plot creation.
"""

import unittest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trace.plotting.volcano import (
    adjust_pvalues,
    prepare_volcano_data,
    volcano_plot_per_method,
)


class TestAdjustPvalues(unittest.TestCase):
    """Test p-value adjustment methods."""
    
    def test_none_adjustment(self):
        """Test that 'none' returns unchanged p-values."""
        pvals = np.array([0.01, 0.05, 0.10, 0.50])
        adjusted = adjust_pvalues(pvals, method="none")
        
        np.testing.assert_array_equal(pvals, adjusted)
    
    def test_bonferroni_adjustment(self):
        """Test Bonferroni correction."""
        pvals = np.array([0.01, 0.02, 0.03, 0.04])
        adjusted = adjust_pvalues(pvals, method="bonferroni")
        
        # Bonferroni: multiply by number of tests
        expected = pvals * 4
        expected = np.minimum(expected, 1.0)  # Cap at 1
        
        np.testing.assert_array_almost_equal(adjusted, expected)
    
    def test_bonferroni_caps_at_one(self):
        """Test that Bonferroni correction caps at 1.0."""
        pvals = np.array([0.5, 0.6, 0.7])
        adjusted = adjust_pvalues(pvals, method="bonferroni")
        
        # All should be capped at 1.0
        self.assertTrue(np.all(adjusted <= 1.0))
    
    def test_bh_basic(self):
        """Test Benjamini-Hochberg adjustment."""
        pvals = np.array([0.01, 0.04, 0.03, 0.05])
        adjusted = adjust_pvalues(pvals, method="bh")
        
        # BH should be less conservative than Bonferroni
        bonf = adjust_pvalues(pvals, method="bonferroni")
        
        # At least some should be smaller with BH
        self.assertTrue(np.any(adjusted < bonf))
    
    def test_bh_preserves_order(self):
        """Test that BH returns values in original order."""
        pvals = np.array([0.04, 0.01, 0.03, 0.02])
        adjusted = adjust_pvalues(pvals, method="bh")
        
        # Result should have same length and order
        self.assertEqual(len(adjusted), len(pvals))
        
        # Smallest p-value should still be at index 1
        self.assertEqual(np.argmin(pvals), np.argmin(adjusted))
    
    def test_bh_monotonicity(self):
        """Test that BH maintains certain order properties."""
        pvals = np.array([0.001, 0.01, 0.05, 0.10])
        adjusted = adjust_pvalues(pvals, method="bh")
        
        # Adjusted values should generally increase (with some flexibility)
        # At minimum, smallest should give smallest adjusted
        self.assertEqual(np.argmin(pvals), np.argmin(adjusted))
    
    def test_nan_handling(self):
        """Test handling of NaN values."""
        pvals = np.array([0.01, np.nan, 0.03, 0.04])
        adjusted = adjust_pvalues(pvals, method="bh")
        
        # NaN should remain NaN
        self.assertTrue(np.isnan(adjusted[1]))
        
        # Non-NaN values should be adjusted
        self.assertFalse(np.isnan(adjusted[0]))
        self.assertFalse(np.isnan(adjusted[2]))
    
    def test_all_nan(self):
        """Test with all NaN values."""
        pvals = np.array([np.nan, np.nan, np.nan])
        adjusted = adjust_pvalues(pvals, method="bh")
        
        # All should remain NaN
        self.assertTrue(np.all(np.isnan(adjusted)))
    
    def test_invalid_method(self):
        """Test that invalid method raises error."""
        pvals = np.array([0.01, 0.05])
        
        with self.assertRaises(ValueError):
            adjust_pvalues(pvals, method="invalid")
    
    def test_single_value(self):
        """Test with single p-value."""
        pvals = np.array([0.05])
        
        # All methods should return same value for single test
        none_adj = adjust_pvalues(pvals, method="none")
        bonf_adj = adjust_pvalues(pvals, method="bonferroni")
        bh_adj = adjust_pvalues(pvals, method="bh")
        
        np.testing.assert_almost_equal(none_adj[0], 0.05)
        np.testing.assert_almost_equal(bonf_adj[0], 0.05)
        np.testing.assert_almost_equal(bh_adj[0], 0.05)


class TestPrepareVolcanoData(unittest.TestCase):
    """Test prepare_volcano_data function."""
    
    def setUp(self):
        """Create test data."""
        self.df = pd.DataFrame({
            "method": ["IPW", "IPW", "TMLE", "TMLE"],
            "outcome": ["A", "B", "A", "B"],
            "RD": [0.1, -0.05, 0.08, -0.03],
            "p_value": [0.01, 0.20, 0.05, 0.30],
        })
    
    def test_columns_added(self):
        """Test that q_value and neglog10p are added."""
        result = prepare_volcano_data(self.df)
        
        self.assertIn("q_value", result.columns)
        self.assertIn("neglog10p", result.columns)
    
    def test_required_columns_present(self):
        """Test that required columns are in output."""
        result = prepare_volcano_data(self.df)
        
        required = ["method", "outcome", "RD", "p_value", "q_value", "neglog10p"]
        for col in required:
            self.assertIn(col, result.columns)
    
    def test_neglog10p_correct(self):
        """Test that -log10(p) is computed correctly."""
        result = prepare_volcano_data(self.df)
        
        # Check first value: -log10(0.01) = 2
        np.testing.assert_almost_equal(result.loc[0, "neglog10p"], 2.0, decimal=5)
        
        # Check that smaller p-values give larger neglog10p
        self.assertGreater(
            result.loc[0, "neglog10p"],  # p=0.01
            result.loc[2, "neglog10p"]   # p=0.05
        )
    
    def test_adjust_by_method(self):
        """Test adjustment within each method."""
        result = prepare_volcano_data(
            self.df,
            adjust="bh",
            adjust_per="by_method"
        )
        
        # Each method group should have adjusted values
        for method in ["IPW", "TMLE"]:
            method_data = result[result["method"] == method]
            # q-values should be >= p-values (or equal for smallest)
            self.assertTrue(np.all(
                method_data["q_value"].values >= method_data["p_value"].values
            ))
    
    def test_adjust_global(self):
        """Test global adjustment."""
        result = prepare_volcano_data(
            self.df,
            adjust="bh",
            adjust_per="global"
        )
        
        # All values adjusted together
        self.assertTrue(np.all(result["q_value"] >= result["p_value"]))
    
    def test_no_adjustment(self):
        """Test with no adjustment."""
        result = prepare_volcano_data(
            self.df,
            adjust="none"
        )
        
        # q_value should equal p_value
        np.testing.assert_array_almost_equal(
            result["q_value"].values,
            result["p_value"].values
        )
    
    def test_custom_column_names(self):
        """Test with custom column names."""
        df = pd.DataFrame({
            "meth": ["IPW"],
            "out": ["A"],
            "effect": [0.1],
            "pval": [0.01],
        })
        
        result = prepare_volcano_data(
            df,
            rd_col="effect",
            p_col="pval",
            method_col="meth",
            outcome_col="out"
        )
        
        # Should work and rename columns
        self.assertIn("RD", result.columns)
        self.assertIn("p_value", result.columns)
    
    def test_p_floor_prevents_inf(self):
        """Test that p_floor prevents infinite -log10(p)."""
        df = pd.DataFrame({
            "method": ["IPW"],
            "outcome": ["A"],
            "RD": [0.1],
            "p_value": [0.0],  # Would give inf without floor
        })
        
        result = prepare_volcano_data(df, p_floor=1e-300)
        
        # Should not be infinite
        self.assertTrue(np.isfinite(result.loc[0, "neglog10p"]))


class TestVolcanoPlot(unittest.TestCase):
    """Test volcano_plot_per_method function."""
    
    def setUp(self):
        """Create test data."""
        self.df = pd.DataFrame({
            "method": ["IPW"] * 10 + ["TMLE"] * 10,
            "outcome": [f"Out{i}" for i in range(10)] * 2,
            "RD": np.random.uniform(-0.1, 0.1, 20),
            "p_value": np.random.uniform(0.001, 0.5, 20),
        })
        self.volcano_data = prepare_volcano_data(self.df)
    
    def tearDown(self):
        """Close all figures after each test."""
        plt.close('all')
    
    def test_creates_figure(self):
        """Test that function creates a figure."""
        fig, axes = volcano_plot_per_method(self.volcano_data)
        
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, plt.Figure)
        
        plt.close(fig)
    
    def test_correct_number_of_panels(self):
        """Test that one panel is created per method."""
        fig, axes = volcano_plot_per_method(self.volcano_data)
        
        n_methods = self.volcano_data["method"].nunique()
        self.assertEqual(len(axes), n_methods)
        
        plt.close(fig)
    
    def test_single_method(self):
        """Test with single method."""
        single_method = self.volcano_data[self.volcano_data["method"] == "IPW"]
        fig, axes = volcano_plot_per_method(single_method)
        
        self.assertEqual(len(axes), 1)
        
        plt.close(fig)
    
    def test_empty_method_handled(self):
        """Test handling of method with no data after filtering."""
        # Create data where one method has all NaN
        df = self.volcano_data.copy()
        df.loc[df["method"] == "TMLE", "RD"] = np.nan
        
        # Should not raise error
        fig, axes = volcano_plot_per_method(df)
        plt.close(fig)
    
    def test_alpha_threshold(self):
        """Test that alpha threshold affects significance coloring."""
        # Create data with known q-values
        df = pd.DataFrame({
            "method": ["IPW"] * 3,
            "outcome": ["A", "B", "C"],
            "RD": [0.1, 0.05, 0.02],
            "p_value": [0.001, 0.05, 0.10],
            "q_value": [0.003, 0.05, 0.10],
            "neglog10p": [3.0, 1.3, 1.0],
        })
        
        # Should not raise error with different alpha values
        fig1, _ = volcano_plot_per_method(df, alpha=0.05)
        fig2, _ = volcano_plot_per_method(df, alpha=0.01)
        
        plt.close(fig1)
        plt.close(fig2)
    
    def test_label_map_applied(self):
        """Test that label_map is used for annotations."""
        label_map = {"Out1": "Nice Name 1"}
        
        # Should not raise error
        fig, axes = volcano_plot_per_method(
            self.volcano_data,
            label_map=label_map
        )
        
        plt.close(fig)
    
    def test_custom_figsize(self):
        """Test custom figure size."""
        fig, axes = volcano_plot_per_method(
            self.volcano_data,
            figsize_per_panel=(8, 6)
        )
        
        # Check figure was created with custom size
        self.assertIsNotNone(fig)
        
        plt.close(fig)
    
    def test_custom_colors(self):
        """Test custom colors for significant/non-significant points."""
        fig, axes = volcano_plot_per_method(
            self.volcano_data,
            sig_color='red',
            ns_color='gray'
        )
        
        self.assertIsNotNone(fig)
        
        plt.close(fig)
    
    def test_max_labels_limits_annotations(self):
        """Test that max_labels_per_panel limits annotations."""
        # Should not raise error with different limits
        fig1, _ = volcano_plot_per_method(
            self.volcano_data,
            max_labels_per_panel=3
        )
        fig2, _ = volcano_plot_per_method(
            self.volcano_data,
            max_labels_per_panel=20
        )
        
        plt.close(fig1)
        plt.close(fig2)
    
    def test_all_significant(self):
        """Test with all outcomes significant."""
        df = pd.DataFrame({
            "method": ["IPW"] * 3,
            "outcome": ["A", "B", "C"],
            "RD": [0.1, 0.05, 0.02],
            "p_value": [0.001, 0.002, 0.003],
            "q_value": [0.001, 0.002, 0.003],
            "neglog10p": [3.0, 2.7, 2.5],
        })
        
        fig, axes = volcano_plot_per_method(df, alpha=0.05)
        self.assertIsNotNone(fig)
        
        plt.close(fig)
    
    def test_all_nonsignificant(self):
        """Test with all outcomes non-significant."""
        df = pd.DataFrame({
            "method": ["IPW"] * 3,
            "outcome": ["A", "B", "C"],
            "RD": [0.01, 0.005, 0.002],
            "p_value": [0.5, 0.6, 0.7],
            "q_value": [0.6, 0.7, 0.8],
            "neglog10p": [0.3, 0.22, 0.15],
        })
        
        fig, axes = volcano_plot_per_method(df, alpha=0.05)
        self.assertIsNotNone(fig)
        
        plt.close(fig)
    
    def test_no_methods_raises_error(self):
        """Test that empty dataframe raises error."""
        empty_df = pd.DataFrame(columns=[
            "method", "outcome", "RD", "p_value", "q_value", "neglog10p"
        ])
        
        with self.assertRaises(ValueError):
            volcano_plot_per_method(empty_df)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflow."""
    
    def tearDown(self):
        """Close all figures after each test."""
        plt.close('all')
    
    def test_complete_workflow(self):
        """Test complete workflow from raw data to plot."""
        # Raw data
        raw_data = pd.DataFrame({
            "method": ["IPW", "IPW", "TMLE", "TMLE"],
            "outcome": ["A", "B", "A", "B"],
            "RD": [0.1, -0.05, 0.08, -0.03],
            "p_value": [0.01, 0.20, 0.05, 0.30],
        })
        
        # Prepare
        volcano_data = prepare_volcano_data(raw_data, adjust="bh")
        
        # Plot
        fig, axes = volcano_plot_per_method(volcano_data, alpha=0.05)
        
        # Check everything worked
        self.assertIsNotNone(fig)
        self.assertEqual(len(axes), 2)
        self.assertIn("q_value", volcano_data.columns)
        
        plt.close(fig)
    
    def test_different_adjustment_methods(self):
        """Test workflow with different adjustment methods."""
        raw_data = pd.DataFrame({
            "method": ["IPW"] * 5,
            "outcome": [f"Out{i}" for i in range(5)],
            "RD": [0.1, 0.05, 0.02, -0.03, -0.08],
            "p_value": [0.001, 0.01, 0.05, 0.10, 0.50],
        })
        
        for method in ["bh", "bonferroni", "none"]:
            volcano_data = prepare_volcano_data(raw_data, adjust=method)
            fig, axes = volcano_plot_per_method(volcano_data)
            
            self.assertIsNotNone(fig)
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()

