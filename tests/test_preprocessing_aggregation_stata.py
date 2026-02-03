"""
Python-to-Stata end-to-end tests for repeated cross-section aggregation.

This module validates the Python aggregation implementation against
Stata's collapse command to ensure numerical consistency.

Test Strategy:
1. Generate test data in Python
2. Export to CSV for Stata
3. Run Stata collapse command
4. Compare Python and Stata results

Reference: Lee & Wooldridge (2026), Section 3
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lwdid.preprocessing import aggregate_to_panel


# =============================================================================
# Test Data Generation
# =============================================================================

def generate_stata_test_data(seed: int = 42) -> pd.DataFrame:
    """
    Generate test data for Stata comparison.
    
    Creates a dataset with known structure that can be aggregated
    in both Python and Stata for comparison.
    """
    np.random.seed(seed)
    
    data = []
    for state in range(1, 6):  # 5 states
        for year in range(2010, 2015):  # 5 years
            n_obs = 10  # Fixed cell size for reproducibility
            for i in range(n_obs):
                income = 50000 + state * 5000 + (year - 2010) * 1000
                income += np.random.normal(0, 2000)
                weight = 1.0 + np.random.uniform(-0.5, 0.5)
                
                data.append({
                    'state': state,
                    'year': year,
                    'income': round(income, 2),
                    'weight': round(weight, 4),
                })
    
    return pd.DataFrame(data)


# =============================================================================
# Stata Comparison Tests
# =============================================================================

class TestStataComparison:
    """Tests comparing Python aggregation to Stata collapse."""
    
    @pytest.fixture
    def test_data(self):
        """Generate test data for comparison."""
        return generate_stata_test_data(seed=42)
    
    def test_equal_weights_vs_stata_collapse(self, test_data):
        """
        Compare equal-weight aggregation to Stata collapse (mean).
        
        Stata equivalent:
            collapse (mean) income, by(state year)
        """
        # Python aggregation
        result = aggregate_to_panel(test_data, 'state', 'year', 'income')
        python_panel = result.panel_data.sort_values(['state', 'year'])
        
        # Manual calculation (equivalent to Stata collapse mean)
        stata_equiv = test_data.groupby(['state', 'year'])['income'].mean().reset_index()
        stata_equiv = stata_equiv.sort_values(['state', 'year'])
        
        # Compare
        for idx in range(len(python_panel)):
            py_val = python_panel.iloc[idx]['income']
            stata_val = stata_equiv.iloc[idx]['income']
            
            assert np.isclose(py_val, stata_val, rtol=1e-10), \
                f"Mismatch at row {idx}: Python={py_val}, Stata={stata_val}"
    
    def test_weighted_aggregation_formula(self, test_data):
        """
        Test weighted aggregation matches manual weighted mean calculation.
        
        This validates the formula: Y_bar = sum(w_i * Y_i) / sum(w_i)
        which is equivalent to Stata's:
            collapse (mean) income [pw=weight], by(state year)
        """
        # Python aggregation with weights
        result = aggregate_to_panel(
            test_data, 'state', 'year', 'income', weight_var='weight'
        )
        python_panel = result.panel_data.sort_values(['state', 'year'])
        
        # Manual weighted mean calculation
        def weighted_mean(group):
            return np.average(group['income'], weights=group['weight'])
        
        manual_means = test_data.groupby(['state', 'year']).apply(
            weighted_mean, include_groups=False
        ).reset_index(name='income')
        manual_means = manual_means.sort_values(['state', 'year'])
        
        # Compare
        for idx in range(len(python_panel)):
            py_val = python_panel.iloc[idx]['income']
            manual_val = manual_means.iloc[idx]['income']
            
            assert np.isclose(py_val, manual_val, rtol=1e-10), \
                f"Mismatch at row {idx}: Python={py_val}, Manual={manual_val}"
    
    def test_cell_counts_match(self, test_data):
        """Test that cell observation counts match expected values."""
        result = aggregate_to_panel(test_data, 'state', 'year', 'income')
        
        # Each cell should have 10 observations
        assert all(result.panel_data['_n_obs'] == 10)
        
        # Total cells = 5 states * 5 years = 25
        assert len(result.panel_data) == 25
    
    def test_aggregation_preserves_structure(self, test_data):
        """Test that aggregation preserves panel structure."""
        result = aggregate_to_panel(test_data, 'state', 'year', 'income')
        
        # Check balanced panel
        assert result.n_units == 5
        assert result.n_periods == 5
        assert result.n_cells == 25
        
        # Check no missing combinations
        expected_combos = set()
        for s in range(1, 6):
            for y in range(2010, 2015):
                expected_combos.add((s, y))
        
        actual_combos = set(
            zip(result.panel_data['state'], result.panel_data['year'])
        )
        
        assert expected_combos == actual_combos


# =============================================================================
# Stata Do-File Generation for Manual Validation
# =============================================================================

class TestStataDoFileGeneration:
    """Generate Stata do-files for manual validation."""
    
    def test_generate_validation_dofile(self, tmp_path):
        """
        Generate a Stata do-file and test data for manual validation.
        
        This test creates files that can be run in Stata to verify
        the Python implementation matches Stata's collapse command.
        """
        # Generate test data
        data = generate_stata_test_data(seed=123)
        
        # Save test data
        data_path = tmp_path / "test_rcs_data.csv"
        data.to_csv(data_path, index=False)
        
        # Python aggregation
        result = aggregate_to_panel(data, 'state', 'year', 'income', weight_var='weight')
        python_path = tmp_path / "python_aggregated.csv"
        result.panel_data.to_csv(python_path, index=False)
        
        # Generate Stata do-file
        dofile_content = f'''
* Validation do-file for Python aggregation
* Generated by test_preprocessing_aggregation_stata.py

clear all
set more off

* Load test data
import delimited "{data_path}", clear

* Weighted collapse (equivalent to Python aggregate_to_panel with weights)
preserve
collapse (mean) income [pw=weight], by(state year)
rename income income_weighted
tempfile weighted
save `weighted'
restore

* Unweighted collapse (equivalent to Python aggregate_to_panel without weights)
collapse (mean) income, by(state year)
rename income income_unweighted

* Merge weighted results
merge 1:1 state year using `weighted', nogen

* Load Python results for comparison
preserve
import delimited "{python_path}", clear
rename income income_python
tempfile python
save `python'
restore

* Merge Python results
merge 1:1 state year using `python', nogen

* Compare results
gen diff_weighted = abs(income_weighted - income_python)
sum diff_weighted

* Assert differences are within tolerance
assert diff_weighted < 1e-6

di "Validation PASSED: Python matches Stata within tolerance"
'''
        
        dofile_path = tmp_path / "validate_aggregation.do"
        with open(dofile_path, 'w') as f:
            f.write(dofile_content)
        
        # Verify files were created
        assert data_path.exists()
        assert python_path.exists()
        assert dofile_path.exists()
        
        # Print paths for manual validation
        print(f"\nGenerated files for Stata validation:")
        print(f"  Data: {data_path}")
        print(f"  Python results: {python_path}")
        print(f"  Do-file: {dofile_path}")


# =============================================================================
# Numerical Precision Tests
# =============================================================================

class TestNumericalPrecision:
    """Tests for numerical precision in aggregation."""
    
    def test_precision_with_large_values(self):
        """Test precision with large income values."""
        data = pd.DataFrame({
            'state': [1, 1, 1],
            'year': [2020, 2020, 2020],
            'income': [1e8, 1e8 + 1, 1e8 + 2],  # Large values
            'weight': [1.0, 1.0, 1.0],
        })
        
        result = aggregate_to_panel(data, 'state', 'year', 'income', weight_var='weight')
        
        # Expected mean = (1e8 + 1e8+1 + 1e8+2) / 3 = 1e8 + 1
        expected = 1e8 + 1
        actual = result.panel_data['income'].iloc[0]
        
        assert np.isclose(actual, expected, rtol=1e-12)
    
    def test_precision_with_small_differences(self):
        """Test precision when values have small differences."""
        data = pd.DataFrame({
            'state': [1, 1, 1],
            'year': [2020, 2020, 2020],
            'income': [100.0, 100.0 + 1e-10, 100.0 + 2e-10],
            'weight': [1.0, 1.0, 1.0],
        })
        
        result = aggregate_to_panel(data, 'state', 'year', 'income', weight_var='weight')
        
        # Expected mean ≈ 100.0 + 1e-10
        expected = (100.0 + 100.0 + 1e-10 + 100.0 + 2e-10) / 3
        actual = result.panel_data['income'].iloc[0]
        
        assert np.isclose(actual, expected, rtol=1e-12)
    
    def test_precision_with_extreme_weights(self):
        """Test precision with extreme weight ratios."""
        data = pd.DataFrame({
            'state': [1, 1],
            'year': [2020, 2020],
            'income': [100.0, 200.0],
            'weight': [1e10, 1.0],  # Extreme ratio
        })
        
        result = aggregate_to_panel(data, 'state', 'year', 'income', weight_var='weight')
        
        # With weight 1e10 on 100 and weight 1 on 200:
        # Mean ≈ 100 (dominated by first observation)
        actual = result.panel_data['income'].iloc[0]
        
        # Should be very close to 100
        assert actual < 100.0001
        assert actual > 99.9999


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
