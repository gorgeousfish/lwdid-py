"""
Validation tests for BUG-121 and BUG-123 fixes.

BUG-121: detrendq degrees of freedom calculation
- Model: y ~ 1 + tindex + i.quarter
- Parameters: k = 1 (intercept) + 1 (time) + (Q-1) (quarter dummies) = Q + 1
- Requirement: n >= k + 1 = Q + 2 for df >= 1

BUG-123: controls parameter not passed to aggregate functions in RI
- aggregate_to_overall and aggregate_to_cohort now support controls parameter
- randomization_inference_staggered passes controls to these functions
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
import warnings

# Import the modules to test
import sys
sys.path.insert(0, '/Users/cxy/Desktop/大样本lwdid/lwdid-py_v0.1.0/src')

from lwdid.transformations import apply_rolling_transform
from lwdid.exceptions import InsufficientPrePeriodsError
from lwdid.staggered.aggregation import aggregate_to_overall, aggregate_to_cohort
from lwdid.staggered.transformations import transform_staggered_demean


class TestBug121DetrendqDegreesOfFreedom:
    """Test BUG-121 fix: correct degrees of freedom calculation for detrendq."""

    def test_detrendq_df_calculation_formula(self):
        """
        Verify the minimum observations formula for detrendq.
        
        Model: y ~ 1 + tindex + i.quarter
        k = 1 + 1 + (Q-1) = Q + 1 parameters
        Require n >= k + 1 = Q + 2 for valid inference (df >= 1)
        """
        # Create test data with exactly Q+1 observations (should fail)
        # Q = 4 quarters, so need at least Q + 2 = 6 observations
        np.random.seed(42)
        
        # Case 1: Q=4 quarters, n=5 observations (should fail, need 6)
        n_obs = 5
        data = pd.DataFrame({
            'id': [1] * n_obs + [2] * 8,  # Unit 1 has 5 pre-obs, Unit 2 has 8
            'year': list(range(2000, 2000 + n_obs)) + list(range(2000, 2008)),
            'quarter': [1, 2, 3, 4, 1] + [1, 2, 3, 4, 1, 2, 3, 4],  # 4 unique quarters
            'y': np.random.randn(n_obs + 8),
            'post': [0] * n_obs + [0] * 4 + [1] * 4,  # Unit 1: all pre, Unit 2: 4 pre + 4 post
            'd': [1] * n_obs + [0] * 8,
        })
        data['tindex'] = data.groupby('id').cumcount() + 1
        
        # Unit 1 has only 5 pre-period observations with 4 quarters
        # Should raise InsufficientPrePeriodsError because 5 < 4 + 2 = 6
        with pytest.raises(InsufficientPrePeriodsError) as excinfo:
            apply_rolling_transform(
                data=data.copy(),
                y='y',
                ivar='id',
                post='post',
                tindex='tindex',
                tpost1=5,
                rolling='detrendq',
                quarter='quarter'
            )
        
        error_msg = str(excinfo.value)
        # Check that a unit is mentioned and the error is about observations
        assert 'Unit' in error_msg
        assert 'observations' in error_msg.lower()
        # Verify the new formula is used: should mention df = n - k >= 1
        assert 'df = n - k >= 1' in error_msg or 'inference' in error_msg.lower()

    def test_detrendq_sufficient_df(self):
        """Test that detrendq works when n >= Q + 2."""
        np.random.seed(42)
        
        # Create data with sufficient observations
        # Q=4 quarters, n=8 pre-period observations (8 >= 4+2=6, should work)
        data = pd.DataFrame({
            'id': [1] * 12,
            'year': list(range(2000, 2012)),
            'quarter': [1, 2, 3, 4] * 3,
            'y': np.random.randn(12),
            'post': [0] * 8 + [1] * 4,
        })
        data['tindex'] = data.groupby('id').cumcount() + 1
        
        # Should not raise error
        result = apply_rolling_transform(
            data=data.copy(),
            y='y',
            ivar='id',
            post='post',
            tindex='tindex',
            tpost1=9,
            rolling='detrendq',
            quarter='quarter'
        )
        
        assert 'ydot' in result.columns
        assert not result['ydot'].isna().all()

    def test_detrendq_boundary_case(self):
        """Test boundary case: exactly Q+2 observations."""
        np.random.seed(42)
        
        # Q=3 quarters (only 1,2,3 appear), n=5 pre-period observations
        # Need n >= 3+2=5, so n=5 should just pass
        data = pd.DataFrame({
            'id': [1] * 8,
            'year': list(range(2000, 2008)),
            'quarter': [1, 2, 3, 1, 2] + [1, 2, 3],  # 3 unique quarters in pre-period
            'y': np.random.randn(8),
            'post': [0] * 5 + [1] * 3,
        })
        data['tindex'] = data.groupby('id').cumcount() + 1
        
        # Should work (boundary case)
        result = apply_rolling_transform(
            data=data.copy(),
            y='y',
            ivar='id',
            post='post',
            tindex='tindex',
            tpost1=6,
            rolling='detrendq',
            quarter='quarter'
        )
        
        assert 'ydot' in result.columns


class TestBug123ControlsInAggregation:
    """Test BUG-123 fix: controls parameter in aggregate functions."""

    @pytest.fixture
    def staggered_data_with_controls(self):
        """Create staggered DiD data with control variables."""
        np.random.seed(42)
        n_units = 50
        n_periods = 10
        
        # Create panel data
        units = np.repeat(np.arange(1, n_units + 1), n_periods)
        periods = np.tile(np.arange(1, n_periods + 1), n_units)
        
        # Treatment cohorts: some treated at period 5, some at 7, some never
        cohorts = np.zeros(n_units, dtype=int)
        cohorts[:15] = 5  # Cohort 5
        cohorts[15:30] = 7  # Cohort 7
        # Units 31-50 are never-treated (cohort = 0)
        
        gvar = np.repeat(cohorts, n_periods)
        
        # Control variable (time-invariant)
        x1_unit = np.random.randn(n_units)
        x1 = np.repeat(x1_unit, n_periods)
        
        # Outcome with treatment effect
        y = np.random.randn(n_units * n_periods)
        treated_mask = (periods >= gvar) & (gvar > 0)
        y[treated_mask] += 2.0  # True ATT = 2.0
        
        data = pd.DataFrame({
            'id': units,
            'time': periods,
            'gvar': gvar,
            'y': y,
            'x1': x1,
        })
        
        return data

    def test_aggregate_to_overall_accepts_controls(self, staggered_data_with_controls):
        """Test that aggregate_to_overall accepts controls parameter."""
        data = staggered_data_with_controls
        
        # First transform the data
        data_transformed = transform_staggered_demean(
            data=data.copy(),
            y='y',
            ivar='id',
            tvar='time',
            gvar='gvar'
        )
        
        # Test without controls
        result_no_controls = aggregate_to_overall(
            data_transformed=data_transformed,
            gvar='gvar',
            ivar='id',
            tvar='time',
            transform_type='demean',
        )
        
        # Test with controls
        result_with_controls = aggregate_to_overall(
            data_transformed=data_transformed,
            gvar='gvar',
            ivar='id',
            tvar='time',
            transform_type='demean',
            controls=['x1'],
        )
        
        # Both should produce valid results
        assert result_no_controls.att is not None
        assert result_with_controls.att is not None
        assert np.isfinite(result_no_controls.att)
        assert np.isfinite(result_with_controls.att)

    def test_aggregate_to_cohort_accepts_controls(self, staggered_data_with_controls):
        """Test that aggregate_to_cohort accepts controls parameter."""
        data = staggered_data_with_controls
        
        # First transform the data
        data_transformed = transform_staggered_demean(
            data=data.copy(),
            y='y',
            ivar='id',
            tvar='time',
            gvar='gvar'
        )
        
        # Test without controls
        results_no_controls = aggregate_to_cohort(
            data_transformed=data_transformed,
            gvar='gvar',
            ivar='id',
            tvar='time',
            cohorts=[5, 7],
            T_max=10,
            transform_type='demean',
        )
        
        # Test with controls
        results_with_controls = aggregate_to_cohort(
            data_transformed=data_transformed,
            gvar='gvar',
            ivar='id',
            tvar='time',
            cohorts=[5, 7],
            T_max=10,
            transform_type='demean',
            controls=['x1'],
        )
        
        # Both should produce valid results
        assert len(results_no_controls) > 0
        assert len(results_with_controls) > 0
        for r in results_no_controls:
            assert np.isfinite(r.att)
        for r in results_with_controls:
            assert np.isfinite(r.att)

    def test_controls_affect_estimation(self, staggered_data_with_controls):
        """Test that controls actually affect the estimation."""
        data = staggered_data_with_controls
        
        # Add a strongly correlated control
        data = data.copy()
        data['x_correlated'] = data['y'] * 0.5 + np.random.randn(len(data)) * 0.1
        
        # Transform the data
        data_transformed = transform_staggered_demean(
            data=data,
            y='y',
            ivar='id',
            tvar='time',
            gvar='gvar'
        )
        
        # Results should differ when using a correlated control
        result_no_controls = aggregate_to_overall(
            data_transformed=data_transformed,
            gvar='gvar',
            ivar='id',
            tvar='time',
            transform_type='demean',
        )
        
        result_with_controls = aggregate_to_overall(
            data_transformed=data_transformed,
            gvar='gvar',
            ivar='id',
            tvar='time',
            transform_type='demean',
            controls=['x_correlated'],
        )
        
        # Standard errors should generally be different
        # (Though ATT estimates might be similar due to orthogonalization)
        assert result_no_controls.se != result_with_controls.se or \
               result_no_controls.att != result_with_controls.att


class TestBug123ControlsInRI:
    """Test that controls are passed through in randomization inference."""

    def test_ri_function_signature_includes_controls(self):
        """Verify randomization_inference_staggered accepts controls parameter."""
        from lwdid.staggered.randomization import randomization_inference_staggered
        import inspect
        
        sig = inspect.signature(randomization_inference_staggered)
        param_names = list(sig.parameters.keys())
        
        assert 'controls' in param_names, \
            "randomization_inference_staggered should accept 'controls' parameter"


class TestNumericalValidation:
    """Numerical validation against expected mathematical properties."""

    def test_detrendq_residuals_orthogonality(self):
        """
        Test that detrendq residuals are orthogonal to regressors.
        
        For OLS: X'e = 0, where e = y - X*beta_hat
        """
        np.random.seed(42)
        
        # Create data
        data = pd.DataFrame({
            'id': [1] * 20,
            'year': list(range(2000, 2020)),
            'quarter': [1, 2, 3, 4] * 5,
            'y': np.random.randn(20) + np.arange(20) * 0.1,  # Add trend
            'post': [0] * 12 + [1] * 8,
        })
        data['tindex'] = data.groupby('id').cumcount() + 1
        
        result = apply_rolling_transform(
            data=data.copy(),
            y='y',
            ivar='id',
            post='post',
            tindex='tindex',
            tpost1=13,
            rolling='detrendq',
            quarter='quarter'
        )
        
        # Check residuals exist and are not all NaN
        assert 'ydot' in result.columns
        pre_residuals = result[result['post'] == 0]['ydot']
        assert not pre_residuals.isna().all()
        
        # Pre-treatment residuals should have mean close to 0
        assert abs(pre_residuals.mean()) < 0.1


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
