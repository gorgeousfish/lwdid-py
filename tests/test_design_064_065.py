"""
Unit tests for DESIGN-064 and DESIGN-065 fixes.

DESIGN-064: OLS convergence checking in transformations.py
    - detrend_unit: Check for NaN/Inf coefficients
    - demeanq_unit: Check for NaN/Inf coefficients
    - detrendq_unit: Check for NaN/Inf coefficients

DESIGN-065: Absolute variance threshold in staggered/transformations.py
    - transform_staggered_detrend: Use absolute threshold VARIANCE_ABS_TOL = 1e-10

Note: statsmodels.OLS uses pseudo-inverse (pinv) internally, which means it rarely
produces NaN coefficients even for near-singular cases. The convergence checks
added in DESIGN-064 are defensive programming against potential future changes
or truly pathological data.
"""

import warnings
import numpy as np
import pandas as pd
import pytest

from lwdid.transformations import (
    detrend_unit,
    demeanq_unit,
    detrendq_unit,
    apply_rolling_transform,
)
from lwdid.staggered.transformations import transform_staggered_detrend


class TestDesign064OLSConvergenceCheck:
    """Tests for DESIGN-064: OLS convergence checking.
    
    Note: statsmodels.OLS handles near-singular cases gracefully using pinv,
    so these tests verify behavior rather than expecting NaN warnings.
    """
    
    def test_detrend_unit_constant_y_handles_gracefully(self):
        """Test that constant y values are handled gracefully by OLS."""
        # Create data where y is constant in pre-period
        data = pd.DataFrame({
            'unit': [1] * 5,
            'time': [1, 2, 3, 4, 5],
            'y': [10.0, 10.0, 10.0, 20.0, 25.0],  # Constant in pre-period
            'post': [0, 0, 0, 1, 1]
        })
        
        # Should not raise exception
        yhat, ydot = detrend_unit(data, 'y', 'time', 'post')
        
        # statsmodels uses pinv, so OLS still works
        # The slope will be near-zero, intercept will be y_mean
        assert not np.all(np.isnan(yhat))
        
        # Pre-treatment residuals should be near-zero (constant - constant ≈ 0)
        pre_residuals = ydot[:3]
        assert np.allclose(pre_residuals, 0, atol=1e-10)
    
    def test_detrend_unit_normal_case_produces_valid_results(self):
        """Test that normal data produces valid detrended results."""
        data = pd.DataFrame({
            'unit': [1] * 5,
            'time': [1, 2, 3, 4, 5],
            'y': [10.0, 12.0, 14.0, 20.0, 25.0],  # Normal increasing trend
            'post': [0, 0, 0, 1, 1]
        })
        
        yhat, ydot = detrend_unit(data, 'y', 'time', 'post')
        
        # Results should be valid
        assert not np.any(np.isnan(yhat))
        assert not np.any(np.isnan(ydot))
        
        # Pre-treatment residuals should be small (good fit)
        pre_residuals = ydot[:3]
        assert np.abs(pre_residuals).max() < 1.0
    
    def test_demeanq_unit_normal_case_produces_valid_results(self):
        """Test that normal data produces valid seasonally-adjusted results."""
        # Need n_valid > n_params. With 2 quarters, n_params = 2, so need at least 3 obs.
        # Use more pre-treatment observations to ensure sufficient df.
        data = pd.DataFrame({
            'unit': [1] * 10,
            'time': list(range(1, 11)),
            'y': [10.0, 12.0, 10.0, 12.0, 10.0, 12.0, 20.0, 25.0, 30.0, 35.0],
            'quarter': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],  # Only 2 distinct quarters
            'post': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  # 6 pre-treatment obs
        })
        
        yhat, ydot = demeanq_unit(data, 'y', 'quarter', 'post')
        
        # Results should be valid
        assert not np.any(np.isnan(yhat))
        assert not np.any(np.isnan(ydot))
    
    def test_detrendq_unit_normal_case_produces_valid_results(self):
        """Test that normal data produces valid seasonally-detrended results."""
        # Need n_valid > n_params. With 2 quarters, n_params = 1 + 2 = 3, so need at least 4 obs.
        # Use more pre-treatment observations to ensure sufficient df.
        data = pd.DataFrame({
            'unit': [1] * 10,
            'time': list(range(1, 11)),
            'y': [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 25.0, 30.0, 35.0, 40.0],
            'quarter': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],  # Only 2 distinct quarters
            'post': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  # 6 pre-treatment obs
        })
        
        yhat, ydot = detrendq_unit(data, 'y', 'time', 'quarter', 'post')
        
        # Results should be valid
        assert not np.any(np.isnan(yhat))
        assert not np.any(np.isnan(ydot))
    
    def test_detrend_unit_with_partial_nan_y_values(self):
        """Test handling of partial NaN values in outcome variable."""
        data = pd.DataFrame({
            'unit': [1] * 5,
            'time': [1, 2, 3, 4, 5],
            'y': [np.nan, 12.0, 14.0, 20.0, 25.0],  # One NaN in pre-period
            'post': [0, 0, 0, 1, 1]
        })
        
        # OLS with missing='drop' will fit on remaining data
        yhat, ydot = detrend_unit(data, 'y', 'time', 'post')
        
        # Results should be computed (OLS fits on available data)
        # First observation will have NaN residual due to NaN y
        assert np.isnan(ydot[0])  # First obs had NaN y
        assert not np.all(np.isnan(ydot[1:]))  # Others should be valid
    
    def test_ols_convergence_check_code_exists(self):
        """Verify that the OLS convergence check code is present in source."""
        import inspect
        source = inspect.getsource(detrend_unit)
        
        # Check that convergence checking code is present
        assert 'np.isnan(model.params)' in source
        assert 'np.isinf(model.params)' in source
        assert 'invalid coefficients' in source


class TestDesign065AbsoluteVarianceThreshold:
    """Tests for DESIGN-065: Absolute variance threshold in staggered detrend."""
    
    def test_transform_staggered_detrend_uses_absolute_threshold(self):
        """Verify that absolute variance threshold is used in code."""
        import inspect
        source = inspect.getsource(transform_staggered_detrend)
        
        # Check that absolute threshold is used
        assert 'VARIANCE_ABS_TOL' in source
        assert '1e-10' in source
        
        # Check that old relative threshold code is removed
        assert 't_range = t_vals.max() - t_vals.min()' not in source
    
    def test_transform_staggered_detrend_normal_data(self):
        """Test that normal data works correctly after threshold change."""
        np.random.seed(42)
        n_units = 10
        n_periods = 8
        
        # Create balanced panel
        units = np.repeat(range(1, n_units + 1), n_periods)
        times = np.tile(range(1, n_periods + 1), n_units)
        
        # Assign cohorts: some treated at t=4, some at t=5, some never-treated
        gvar_map = {
            1: 4, 2: 4, 3: 4,  # Cohort 4
            4: 5, 5: 5, 6: 5,  # Cohort 5
            7: 0, 8: 0, 9: 0, 10: 0  # Never-treated
        }
        gvar = [gvar_map[u] for u in units]
        
        # Generate outcomes with unit trends and treatment effects
        y = []
        for i, (u, t, g) in enumerate(zip(units, times, gvar)):
            base = 100 + u * 5 + t * 2 + np.random.randn() * 3
            if g != 0 and t >= g:
                base += 10  # Treatment effect
            y.append(base)
        
        data = pd.DataFrame({
            'unit': units,
            'time': times,
            'y': y,
            'gvar': gvar,
        })
        
        result = transform_staggered_detrend(data, 'y', 'unit', 'time', 'gvar')
        
        # Should have detrended columns for each (g, r) pair
        ycheck_cols = [col for col in result.columns if col.startswith('ycheck_')]
        assert len(ycheck_cols) > 0
        
        # Values should not all be NaN
        for col in ycheck_cols[:3]:  # Check first few columns
            non_nan_values = result[col].dropna()
            if len(non_nan_values) > 0:
                # Values should be reasonable (transformed outcomes)
                assert non_nan_values.abs().max() < 1000
    
    def test_transform_staggered_detrend_with_low_variance_unit(self):
        """Test handling of units with low time variance in pre-period."""
        # Create data where one unit only has 2 pre-treatment observations
        # with minimal time variance
        np.random.seed(42)
        
        data = pd.DataFrame({
            'unit': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            'time': [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
            'y': np.random.randn(12) * 10 + 100,
            'gvar': [3, 3, 3, 3, 4, 4, 4, 4, 0, 0, 0, 0],  # Unit 3 is never-treated
        })
        
        # Should not raise error
        result = transform_staggered_detrend(data, 'y', 'unit', 'time', 'gvar')
        
        # Check that output columns exist
        ycheck_cols = [col for col in result.columns if col.startswith('ycheck_')]
        assert len(ycheck_cols) > 0


class TestIntegration:
    """Integration tests combining DESIGN-064 and DESIGN-065 fixes."""
    
    def test_apply_rolling_transform_detrend_produces_valid_results(self):
        """Test apply_rolling_transform with detrend produces valid results."""
        data = pd.DataFrame({
            'unit': [1, 1, 1, 1, 2, 2, 2, 2],
            'time': [1, 2, 3, 4, 1, 2, 3, 4],
            'y': [10.0, 12.0, 14.0, 20.0, 15.0, 18.0, 21.0, 30.0],
            'post': [0, 0, 0, 1, 0, 0, 0, 1]
        })
        
        result = apply_rolling_transform(
            data, 'y', 'unit', 'time', 'post', 'detrend', tpost1=4
        )
        
        # Result should have ydot column
        assert 'ydot' in result.columns
        
        # All units should have valid ydot values
        assert not result['ydot'].isna().any()
        
        # Post-treatment averages should exist
        assert 'ydot_postavg' in result.columns
        assert result.loc[result['time'] == 4, 'ydot_postavg'].notna().all()
    
    def test_full_pipeline_detrend_with_varying_trends(self):
        """Test full detrending pipeline with units having different trends."""
        np.random.seed(42)
        
        # Unit 1: steep positive trend
        # Unit 2: flat trend
        # Unit 3: negative trend
        data_list = []
        for unit in [1, 2, 3]:
            if unit == 1:
                trend = 3.0  # Steep positive
            elif unit == 2:
                trend = 0.1  # Flat
            else:
                trend = -2.0  # Negative
            
            for t in range(1, 6):
                y = 100 + trend * t + np.random.randn() * 0.5
                post = 1 if t >= 4 else 0
                if post == 1:
                    y += 10  # Treatment effect
                data_list.append({
                    'unit': unit,
                    'time': t,
                    'y': y,
                    'post': post
                })
        
        data = pd.DataFrame(data_list)
        
        result = apply_rolling_transform(
            data, 'y', 'unit', 'time', 'post', 'detrend', tpost1=4
        )
        
        # All ydot values should be valid
        assert not result['ydot'].isna().any()
        
        # Pre-treatment ydot should be small (good fit removes trend)
        pre_ydot = result.loc[result['post'] == 0, 'ydot']
        assert pre_ydot.abs().mean() < 2.0  # Residuals should be small


class TestNumericalStability:
    """Tests for numerical stability of the transformations."""
    
    def test_detrend_with_large_time_values(self):
        """Test detrending with large time indices (e.g., years 2000-2025)."""
        data = pd.DataFrame({
            'unit': [1] * 6,
            'time': [2000, 2001, 2002, 2003, 2004, 2005],
            'y': [100.0, 102.0, 104.0, 110.0, 115.0, 120.0],
            'post': [0, 0, 0, 1, 1, 1]
        })
        
        yhat, ydot = detrend_unit(data, 'y', 'time', 'post')
        
        # Results should be numerically stable
        assert not np.any(np.isnan(yhat))
        assert not np.any(np.isnan(ydot))
        assert not np.any(np.isinf(yhat))
        assert not np.any(np.isinf(ydot))
        
        # Pre-treatment residuals should be small (good fit)
        pre_residuals = ydot[:3]
        assert np.abs(pre_residuals).max() < 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
