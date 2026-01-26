"""
Test BUG-060-A: OLS NaN handling in transformations.py

This test verifies that detrend_unit, demeanq_unit, and detrendq_unit
correctly handle NaN values in the outcome variable by using statsmodels'
missing='drop' parameter.

The fix ensures Python behavior matches Stata's regress command, which
automatically skips observations with missing values.
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from lwdid.transformations import (
    detrend_unit,
    demeanq_unit,
    detrendq_unit,
    apply_rolling_transform,
)


class TestDetrendUnitNaNHandling:
    """Test detrend_unit function with NaN values"""
    
    def test_detrend_unit_with_nan_in_pre_period(self):
        """Test detrend_unit correctly handles NaN in pre-treatment outcome"""
        # Create test data with NaN in pre-treatment period
        unit_data = pd.DataFrame({
            'y': [10.0, np.nan, 15.0, 20.0, 25.0, 30.0],  # NaN at t=2
            'tindex': [1, 2, 3, 4, 5, 6],
            'post': [0, 0, 0, 1, 1, 1]  # First 3 periods are pre-treatment
        })
        
        # Should not raise error and should return valid results
        yhat, ydot = detrend_unit(unit_data, 'y', 'tindex', 'post')
        
        # Verify coefficients are computed (not NaN)
        assert not np.isnan(yhat).all(), "All predictions are NaN - fix did not work"
        assert not np.isnan(ydot).all(), "All residuals are NaN - fix did not work"
        
        # Verify the linear trend is estimated from non-NaN observations only
        # Pre-treatment data without NaN: t=1,y=10 and t=3,y=15
        # Linear regression: slope = (15-10)/(3-1) = 2.5
        # intercept = 10 - 2.5*1 = 7.5 (using point t=1, y=10)
        # So yhat = 7.5 + 2.5*t
        expected_slope = 2.5
        expected_intercept = 7.5
        expected_yhat_at_t1 = expected_intercept + expected_slope * 1  # 10.0
        expected_yhat_at_t3 = expected_intercept + expected_slope * 3  # 15.0
        
        # Allow some tolerance for floating point
        assert np.abs(yhat[0] - expected_yhat_at_t1) < 0.01, f"yhat[0]={yhat[0]}, expected={expected_yhat_at_t1}"
        assert np.abs(yhat[2] - expected_yhat_at_t3) < 0.01, f"yhat[2]={yhat[2]}, expected={expected_yhat_at_t3}"
    
    def test_detrend_unit_without_nan(self):
        """Test detrend_unit works correctly with clean data (regression test)"""
        unit_data = pd.DataFrame({
            'y': [10.0, 12.5, 15.0, 20.0, 25.0, 30.0],
            'tindex': [1, 2, 3, 4, 5, 6],
            'post': [0, 0, 0, 1, 1, 1]
        })
        
        yhat, ydot = detrend_unit(unit_data, 'y', 'tindex', 'post')
        
        # Verify no NaN in output
        assert not np.isnan(yhat).any()
        assert not np.isnan(ydot).any()
        
        # Linear regression on t=1,2,3 with y=10,12.5,15 should give slope=2.5
        # intercept = mean(y) - slope * mean(t) = 12.5 - 2.5*2 = 7.5
        expected_slope = 2.5
        expected_intercept = 7.5
        
        assert np.abs(yhat[0] - (expected_intercept + expected_slope * 1)) < 0.01
        
    def test_detrend_unit_multiple_nan(self):
        """Test detrend_unit handles multiple NaN values"""
        unit_data = pd.DataFrame({
            'y': [10.0, np.nan, np.nan, 25.0, 30.0, 35.0],
            'tindex': [1, 2, 3, 4, 5, 6],
            'post': [0, 0, 0, 1, 1, 1]
        })
        
        # Only t=1, y=10 is available in pre-period
        # With only one observation, slope cannot be estimated reliably
        # but the function should not crash
        yhat, ydot = detrend_unit(unit_data, 'y', 'tindex', 'post')
        
        # Check it doesn't return all NaN
        assert not np.isnan(yhat).all()


class TestDemeanqUnitNaNHandling:
    """Test demeanq_unit function with NaN values"""
    
    def test_demeanq_unit_with_nan(self):
        """Test demeanq_unit correctly handles NaN in outcome"""
        unit_data = pd.DataFrame({
            'y': [10.0, np.nan, 15.0, 20.0, 22.0, 25.0, 28.0, 30.0],
            'quarter': [1, 2, 3, 4, 1, 2, 3, 4],
            'post': [0, 0, 0, 0, 1, 1, 1, 1]
        })
        
        yhat, ydot = demeanq_unit(unit_data, 'y', 'quarter', 'post')
        
        # Should not have all NaN
        assert not np.isnan(yhat).all(), "All predictions are NaN - fix did not work"
        assert not np.isnan(ydot).all(), "All residuals are NaN - fix did not work"
    
    def test_demeanq_unit_without_nan(self):
        """Test demeanq_unit works with clean data (regression test)"""
        unit_data = pd.DataFrame({
            'y': [10.0, 12.0, 15.0, 20.0, 22.0, 25.0, 28.0, 30.0],
            'quarter': [1, 2, 3, 4, 1, 2, 3, 4],
            'post': [0, 0, 0, 0, 1, 1, 1, 1]
        })
        
        yhat, ydot = demeanq_unit(unit_data, 'y', 'quarter', 'post')
        
        assert not np.isnan(yhat).any()
        assert not np.isnan(ydot).any()


class TestDetrendqUnitNaNHandling:
    """Test detrendq_unit function with NaN values"""
    
    def test_detrendq_unit_with_nan(self):
        """Test detrendq_unit correctly handles NaN in outcome"""
        # Create data with 8 pre-treatment periods (2 years of quarterly data)
        unit_data = pd.DataFrame({
            'y': [10.0, np.nan, 15.0, 18.0, 20.0, 22.0, 25.0, 28.0,
                  30.0, 32.0, 35.0, 38.0],
            'tindex': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            'quarter': [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
            'post': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
        })
        
        yhat, ydot = detrendq_unit(unit_data, 'y', 'tindex', 'quarter', 'post')
        
        # Should not have all NaN
        assert not np.isnan(yhat).all(), "All predictions are NaN - fix did not work"
        assert not np.isnan(ydot).all(), "All residuals are NaN - fix did not work"
    
    def test_detrendq_unit_without_nan(self):
        """Test detrendq_unit works with clean data (regression test)"""
        unit_data = pd.DataFrame({
            'y': [10.0, 12.0, 15.0, 18.0, 20.0, 22.0, 25.0, 28.0,
                  30.0, 32.0, 35.0, 38.0],
            'tindex': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            'quarter': [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
            'post': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
        })
        
        yhat, ydot = detrendq_unit(unit_data, 'y', 'tindex', 'quarter', 'post')
        
        assert not np.isnan(yhat).any()
        assert not np.isnan(ydot).any()


class TestNaNHandlingMatchesStata:
    """Test that Python NaN handling matches Stata's regress behavior"""
    
    def test_detrend_nan_vs_manual_drop(self):
        """Verify missing='drop' gives same result as manual dropna"""
        unit_data = pd.DataFrame({
            'y': [10.0, np.nan, 15.0, 20.0, 25.0, 30.0],
            'tindex': [1, 2, 3, 4, 5, 6],
            'post': [0, 0, 0, 1, 1, 1]
        })
        
        # Method 1: Use our fixed function (with missing='drop')
        yhat_fixed, ydot_fixed = detrend_unit(unit_data, 'y', 'tindex', 'post')
        
        # Method 2: Manual dropna and OLS (simulating Stata behavior)
        unit_pre = unit_data[unit_data['post'] == 0].dropna(subset=['y', 'tindex']).copy()
        X_pre = sm.add_constant(unit_pre['tindex'].values)
        y_pre = unit_pre['y'].values
        model_manual = sm.OLS(y_pre, X_pre).fit()
        
        X_all = sm.add_constant(unit_data['tindex'].values)
        yhat_manual = model_manual.predict(X_all)
        ydot_manual = unit_data['y'].values - yhat_manual
        
        # Results should match
        np.testing.assert_array_almost_equal(yhat_fixed, yhat_manual, decimal=10)
        # ydot may have NaN where y has NaN, compare non-NaN elements
        valid_mask = ~np.isnan(ydot_manual)
        np.testing.assert_array_almost_equal(
            ydot_fixed[valid_mask], 
            ydot_manual[valid_mask], 
            decimal=10
        )
    
    def test_computed_coefficients_accuracy(self):
        """Test exact coefficient computation with NaN data"""
        # Create data where we know exact expected coefficients
        # Pre-treatment: t=1,y=10 and t=3,y=16 (skip t=2 which has NaN)
        # Linear regression: slope = (16-10)/(3-1) = 3, intercept = 10 - 3*1 = 7
        unit_data = pd.DataFrame({
            'y': [10.0, np.nan, 16.0, 22.0, 28.0],
            'tindex': [1, 2, 3, 4, 5],
            'post': [0, 0, 0, 1, 1]
        })
        
        yhat, ydot = detrend_unit(unit_data, 'y', 'tindex', 'post')
        
        # Expected: yhat = 7 + 3*t
        expected_yhat = np.array([10.0, 13.0, 16.0, 19.0, 22.0])
        np.testing.assert_array_almost_equal(yhat, expected_yhat, decimal=10)
        
        # ydot = y - yhat
        expected_ydot = np.array([0.0, np.nan, 0.0, 3.0, 6.0])
        # Compare non-NaN elements
        valid_idx = [0, 2, 3, 4]
        np.testing.assert_array_almost_equal(
            ydot[valid_idx], 
            expected_ydot[valid_idx], 
            decimal=10
        )


class TestRegressionPreservation:
    """Test that fix doesn't break existing functionality"""
    
    def test_existing_test_data_still_works(self):
        """Ensure modification doesn't break clean data processing"""
        # Standard test data without NaN
        np.random.seed(42)
        n_units = 5
        n_periods = 8
        tpost1 = 5
        
        data = []
        for i in range(1, n_units + 1):
            for t in range(1, n_periods + 1):
                y = 10 + 2 * t + np.random.normal(0, 1)
                post = 1 if t >= tpost1 else 0
                d = 1 if i <= 2 else 0  # First 2 units are treated
                data.append({'id': i, 'year': t, 'y': y, 'post': post, 'd': d})
        
        df = pd.DataFrame(data)
        
        # Apply transformation - should work without error
        result = apply_rolling_transform(
            df.copy(), 
            y='y', 
            ivar='id', 
            tindex='year', 
            post='post', 
            rolling='detrend',
            tpost1=tpost1
        )
        
        # Verify output columns exist
        assert 'ydot' in result.columns
        assert 'ydot_postavg' in result.columns
        assert 'firstpost' in result.columns
        
        # Verify no unexpected NaN in ydot (all original data was clean)
        assert not result['ydot'].isna().any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
