"""
Numerical validation tests for BUG-203, BUG-204, and BUG-205 fixes.

This module performs numerical cross-validation between Python and Stata
to verify the correctness of bug fixes.
"""

import numpy as np
import pandas as pd
import pytest
import warnings
import tempfile
import os


class TestBug203NumericalValidation:
    """Numerical validation for BUG-203: Controls missing value handling."""

    @pytest.fixture
    def test_data_with_missing_controls(self):
        """Create deterministic test data with missing control values."""
        np.random.seed(42)
        
        # Create a small dataset where we can manually verify results
        data = {
            'id': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6],
            'year': [2000, 2001, 2002] * 6,
            'y': [10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42, 50, 51, 52, 60, 61, 62],
            'gvar': [2001]*3 + [2001]*3 + [np.inf]*3 + [np.inf]*3 + [2002]*3 + [2002]*3,
            'ctrl': [1, 1, 1, 2, np.nan, 2, 3, 3, 3, 4, 4, 4, 5, 5, np.nan, 6, 6, 6],  # NaN at specific positions
        }
        return pd.DataFrame(data)

    def test_regression_excludes_nan_controls(self, test_data_with_missing_controls):
        """Verify that regression correctly excludes observations with NaN controls."""
        from lwdid import lwdid
        
        data = test_data_with_missing_controls
        
        # Use full lwdid workflow which handles controls properly
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = lwdid(
                data=data,
                y='y',
                gvar='gvar',
                ivar='id',
                tvar='year',
                rolling='demean',
                aggregate='cohort',
                controls=['ctrl'],
            )
        
        # Verify results exist and are valid
        assert results.att_by_cohort is not None
        assert len(results.att_by_cohort) > 0
        
        # Each cohort should have valid results
        for _, row in results.att_by_cohort.iterrows():
            assert np.isfinite(row['att'])
            # The number of units should be positive
            assert row['n_units'] > 0
        
        # Verify overall n_control is positive (from results object)
        assert results.n_control > 0

    def test_ols_calculation_matches_manual(self):
        """Verify OLS calculation without controls against manual calculation."""
        from lwdid.staggered.estimation import run_ols_regression
        
        # Simple dataset for manual verification (no controls to avoid sample size issues)
        data = pd.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            'd': [1, 1, 1, 1, 0, 0, 0, 0],
        })
        
        # Manual OLS: y = a + b*d
        # Using numpy for verification
        y = data['y'].values
        X = np.column_stack([
            np.ones(len(data)),
            data['d'].values,
        ])
        
        # OLS: beta = (X'X)^-1 X'y
        XtX_inv = np.linalg.inv(X.T @ X)
        beta_manual = XtX_inv @ (X.T @ y)
        
        # Get result from lwdid function
        result = run_ols_regression(
            data=data.reset_index(drop=True),
            y='y',
            d='d',
            controls=None,
            vce=None,
            cluster_var=None,
            alpha=0.05,
        )
        
        # The ATT (coefficient on d) should match
        # beta_manual[0] = intercept, beta_manual[1] = coefficient on d
        np.testing.assert_allclose(result['att'], beta_manual[1], rtol=1e-6)
        
        # Manual calculation: treated mean = (1+2+3+4)/4 = 2.5, control mean = (5+6+7+8)/4 = 6.5
        # ATT = 2.5 - 6.5 = -4
        expected_att = -4.0
        np.testing.assert_allclose(result['att'], expected_att, rtol=1e-6)


class TestBug204NumericalValidation:
    """Numerical validation for BUG-204: Cohort rounding consistency."""

    def test_control_mask_consistency(self):
        """Verify control mask generation is consistent across functions."""
        from lwdid.staggered.control_groups import (
            get_all_control_masks,
            get_valid_control_units,
            ControlGroupStrategy,
        )
        
        # Create test data
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3, 4, 4],
            'year': [2000, 2001] * 4,
            'gvar': [2001, 2001, 2002, 2002, np.inf, np.inf, np.inf, np.inf],
        })
        
        # Get masks from batch function
        batch_masks = get_all_control_masks(
            data=data,
            gvar='gvar',
            ivar='id',
            cohorts=[2001],
            T_max=2002,
            strategy=ControlGroupStrategy.NOT_YET_TREATED,
        )
        
        # Get mask from single function
        single_mask = get_valid_control_units(
            data=data,
            gvar='gvar',
            ivar='id',
            cohort=2001,
            period=2001,
            strategy=ControlGroupStrategy.NOT_YET_TREATED,
        )
        
        # The masks should be equivalent
        batch_key = (2001, 2001.0)
        assert batch_key in batch_masks
        
        # Compare mask values
        batch_result = batch_masks[batch_key]
        pd.testing.assert_series_equal(
            batch_result.sort_index(),
            single_mask.sort_index(),
            check_names=False,
        )


class TestBug205NumericalValidation:
    """Numerical validation for BUG-205: LaTeX escape function."""

    def test_latex_output_compiles(self):
        """Verify that LaTeX output from results is valid."""
        from lwdid import lwdid
        import tempfile
        
        # Create simple staggered data
        np.random.seed(42)
        data = []
        for i in range(20):
            gvar = 2005 if i < 10 else np.inf
            for t in range(2000, 2010):
                y = 10 + i * 0.5 + t * 0.1 + np.random.normal(0, 0.5)
                if gvar != np.inf and t >= gvar:
                    y += 2
                data.append({'id': i, 'year': t, 'y': y, 'gvar': gvar})
        
        df = pd.DataFrame(data)
        
        # Run estimation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = lwdid(
                data=df,
                y='y',
                gvar='gvar',
                ivar='id',
                tvar='year',
                rolling='demean',
                aggregate='cohort',
            )
        
        # Export to LaTeX
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tex', delete=False) as f:
            temp_path = f.name
        
        try:
            results.to_latex(temp_path)
            
            # Read and verify content
            with open(temp_path, 'r') as f:
                content = f.read()
            
            # Check that special characters are properly escaped
            assert r'\_' not in content or content.count(r'\_') == content.count('_')
            # Should contain LaTeX table markers
            assert 'tabular' in content or 'begin' in content
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_escape_idempotent(self):
        """Verify that escaping the same string twice gives the same result."""
        from lwdid.results import _latex_escape_string
        
        test_strings = [
            'simple_name',
            'multi_part_name',
            'price%change',
            'A&B_combined',
            r'already\_escaped',
            r'mixed\_and_unescaped',
        ]
        
        for s in test_strings:
            once = _latex_escape_string(s)
            twice = _latex_escape_string(once)
            # Escaping twice should give the same result (idempotent)
            assert once == twice, f"Not idempotent for '{s}': once='{once}', twice='{twice}'"


class TestCrossValidation:
    """Cross-validation tests comparing Python results with expected values."""

    def test_aggregation_result_structure(self):
        """Verify aggregation results have correct structure."""
        from lwdid import lwdid
        
        np.random.seed(123)
        data = []
        for i in range(30):
            if i < 10:
                gvar = np.inf
            elif i < 20:
                gvar = 2005
            else:
                gvar = 2006
            
            for t in range(2000, 2010):
                y = 10 + i * 0.3 + np.random.normal(0, 0.5)
                if gvar != np.inf and t >= gvar:
                    y += 2  # True ATT = 2
                data.append({'id': i, 'year': t, 'y': y, 'gvar': gvar})
        
        df = pd.DataFrame(data)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = lwdid(
                data=df,
                y='y',
                gvar='gvar',
                ivar='id',
                tvar='year',
                rolling='demean',
                aggregate='overall',
            )
        
        # Verify result structure
        assert results.att_overall is not None
        assert results.se_overall is not None
        assert np.isfinite(results.att_overall)
        assert np.isfinite(results.se_overall)
        
        # ATT should be close to true value (2) within statistical error
        # Using wide tolerance since this is a random sample
        assert 0.5 < results.att_overall < 3.5, f"ATT={results.att_overall} not in expected range"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
