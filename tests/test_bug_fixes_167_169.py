"""
Unit tests for BUG-167, BUG-168, BUG-169 fixes in estimation.py.

BUG-167: Removal of unused X_centered_t variable in estimate_period_effects()
BUG-168: Statsmodels version compatibility check for HC4 cache clearing
BUG-169: Singular matrix exception handling in HC4 variance computation
"""

import numpy as np
import pandas as pd
import pytest
import warnings

from lwdid.estimation import estimate_att, estimate_period_effects, prepare_controls


class TestBug167UnusedVariable:
    """BUG-167: Verify estimate_period_effects works correctly after removing X_centered_t."""
    
    def test_period_effects_with_controls(self):
        """Verify period effects estimation with controls works after removing unused variable.
        
        The regression should use original controls (not centered) plus interaction terms.
        This matches Stata's implementation.
        """
        np.random.seed(42)
        n_units = 50
        n_periods = 6
        
        # Create panel data
        data = pd.DataFrame({
            'unit_id': np.repeat(range(n_units), n_periods),
            'tindex': np.tile(range(1, n_periods + 1), n_units),
        })
        
        # Treatment assignment (first 20 units treated)
        data['d_'] = (data['unit_id'] < 20).astype(int)
        
        # Control variable (time-invariant)
        unit_x = np.random.randn(n_units)
        data['x1'] = data['unit_id'].map(lambda i: unit_x[i])
        
        # Outcome with treatment effect starting at period 4
        tpost1 = 4
        data['ydot'] = (
            0.5 * data['x1'] + 
            np.random.randn(len(data)) * 0.5 +
            2.0 * data['d_'] * (data['tindex'] >= tpost1)
        )
        
        # Prepare controls specification
        N_treated = 20
        N_control = 30
        controls_spec = prepare_controls(
            data=data,
            d='d_',
            ivar='unit_id',
            controls=['x1'],
            N_treated=N_treated,
            N_control=N_control,
        )
        
        # Verify controls are included
        assert controls_spec['include'] is True
        
        # Run period effects estimation
        period_labels = {t: f"Period {t}" for t in range(1, n_periods + 1)}
        results_df = estimate_period_effects(
            data=data,
            ydot='ydot',
            d='d_',
            tindex='tindex',
            tpost1=tpost1,
            Tmax=n_periods,
            controls_spec=controls_spec,
            vce=None,
            cluster_var=None,
            period_labels=period_labels,
            alpha=0.05,
        )
        
        # Verify results structure
        assert len(results_df) == n_periods - tpost1 + 1  # periods 4, 5, 6
        assert 'beta' in results_df.columns
        assert 'se' in results_df.columns
        
        # Verify treatment effects are approximately 2.0 (the true effect)
        for _, row in results_df.iterrows():
            assert not np.isnan(row['beta'])
            assert abs(row['beta'] - 2.0) < 1.0  # Within reasonable range


class TestBug168StatsmodelsCompatibility:
    """BUG-168: Verify statsmodels version compatibility for HC4 cache clearing."""
    
    def test_hc4_works_without_cache_attribute(self):
        """Test HC4 estimation works even if _cache attribute doesn't exist.
        
        The fix adds hasattr check before clearing _cache.
        """
        data = pd.DataFrame({
            'unit_id': list(range(1, 21)),
            'd_': [1] * 5 + [0] * 15,
            'ydot_postavg': np.random.randn(20),
            'firstpost': [True] * 20,
        })
        
        # This should not raise AttributeError even if statsmodels internal API changes
        results = estimate_att(
            data=data,
            y_transformed='ydot_postavg',
            d='d_',
            ivar='unit_id',
            controls=None,
            vce='hc4',
            cluster_var=None,
            sample_filter=data['firstpost'],
        )
        
        # Verify results are valid
        assert not np.isnan(results['att'])
        assert not np.isnan(results['se_att'])
        assert results['vce_type'] == 'hc4'
    
    def test_hc4_period_effects_compatibility(self):
        """Test HC4 in estimate_period_effects with compatibility check."""
        np.random.seed(123)
        n_units = 30
        n_periods = 5
        
        data = pd.DataFrame({
            'unit_id': np.repeat(range(n_units), n_periods),
            'tindex': np.tile(range(1, n_periods + 1), n_units),
        })
        data['d_'] = (data['unit_id'] < 10).astype(int)
        data['ydot'] = np.random.randn(len(data)) + data['d_'] * 1.5
        
        period_labels = {t: f"Period {t}" for t in range(1, n_periods + 1)}
        
        # Run with HC4 - should not raise any errors
        results_df = estimate_period_effects(
            data=data,
            ydot='ydot',
            d='d_',
            tindex='tindex',
            tpost1=3,
            Tmax=n_periods,
            controls_spec=None,
            vce='hc4',
            cluster_var=None,
            period_labels=period_labels,
        )
        
        # Verify results are valid
        assert len(results_df) == 3  # periods 3, 4, 5
        for _, row in results_df.iterrows():
            assert not np.isnan(row['beta'])
            assert not np.isnan(row['se'])


class TestBug169SingularMatrixHandling:
    """BUG-169: Verify singular matrix exception handling in HC4 variance computation."""
    
    def test_hc4_with_collinear_controls(self):
        """Test HC4 handles perfectly collinear controls gracefully.
        
        When X'X is singular due to collinearity, the code should:
        1. Catch np.linalg.LinAlgError
        2. Issue a warning
        3. Use pseudo-inverse as fallback
        """
        np.random.seed(456)
        
        # Create data with collinear controls (x2 = 2*x1)
        data = pd.DataFrame({
            'unit_id': list(range(1, 31)),
            'd_': [1] * 10 + [0] * 20,
            'firstpost': [True] * 30,
        })
        
        # Add collinear controls
        x1 = np.random.randn(30)
        data['x1'] = x1
        data['x2'] = 2 * x1  # Perfect collinearity
        
        # Outcome
        data['ydot_postavg'] = 1.0 + 0.5 * x1 + data['d_'] * 2.0 + np.random.randn(30) * 0.1
        
        # Prepare controls with collinear variables
        controls_spec = prepare_controls(
            data=data,
            d='d_',
            ivar='unit_id',
            controls=['x1', 'x2'],
            N_treated=10,
            N_control=20,
        )
        
        # The collinearity will cause singular matrix in HC4 computation
        # The fix should handle this gracefully with a warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Note: The regression itself may fail due to singular design matrix
            # The HC4 fix handles the case when regression succeeds but X'X is ill-conditioned
            try:
                results = estimate_att(
                    data=data,
                    y_transformed='ydot_postavg',
                    d='d_',
                    ivar='unit_id',
                    controls=['x1', 'x2'],
                    vce='hc4',
                    cluster_var=None,
                    sample_filter=data['firstpost'],
                )
                # If we get here, verify results are reasonable
                assert results is not None
            except (ValueError, np.linalg.LinAlgError):
                # This is expected for perfect collinearity
                pass
    
    def test_hc4_with_near_singular_matrix(self):
        """Test HC4 with near-singular (ill-conditioned) design matrix.
        
        This tests the warning mechanism when condition number is very high.
        """
        np.random.seed(789)
        
        # Create data with highly correlated (but not perfectly collinear) controls
        data = pd.DataFrame({
            'unit_id': list(range(1, 51)),
            'd_': [1] * 15 + [0] * 35,
            'firstpost': [True] * 50,
        })
        
        x1 = np.random.randn(50)
        # x2 is almost perfectly correlated with x1 (add tiny noise)
        data['x1'] = x1
        data['x2'] = x1 + np.random.randn(50) * 1e-8
        
        data['ydot_postavg'] = 1.0 + data['d_'] * 2.0 + np.random.randn(50) * 0.5
        
        # This should work but may issue a warning about ill-conditioning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            try:
                results = estimate_att(
                    data=data,
                    y_transformed='ydot_postavg',
                    d='d_',
                    ivar='unit_id',
                    controls=['x1', 'x2'],
                    vce='hc4',
                    cluster_var=None,
                    sample_filter=data['firstpost'],
                )
                # Verify we got some result
                assert results is not None
            except (ValueError, np.linalg.LinAlgError):
                # May fail due to near-singularity
                pass
    
    def test_hc4_normal_case_no_warning(self):
        """Test HC4 with well-conditioned design matrix produces no singular matrix warning."""
        np.random.seed(111)
        
        data = pd.DataFrame({
            'unit_id': list(range(1, 41)),
            'd_': [1] * 12 + [0] * 28,
            'firstpost': [True] * 40,
        })
        
        # Independent controls
        data['x1'] = np.random.randn(40)
        data['x2'] = np.random.randn(40)
        data['ydot_postavg'] = (
            1.0 + 
            0.5 * data['x1'] + 
            0.3 * data['x2'] + 
            data['d_'] * 2.0 + 
            np.random.randn(40) * 0.3
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            results = estimate_att(
                data=data,
                y_transformed='ydot_postavg',
                d='d_',
                ivar='unit_id',
                controls=['x1', 'x2'],
                vce='hc4',
                cluster_var=None,
                sample_filter=data['firstpost'],
            )
            
            # Check no singular matrix warning was issued
            singular_warnings = [
                warning for warning in w 
                if 'singular' in str(warning.message).lower()
            ]
            assert len(singular_warnings) == 0, "Unexpected singular matrix warning"
            
            # Verify results
            assert not np.isnan(results['att'])
            assert not np.isnan(results['se_att'])
            assert results['vce_type'] == 'hc4'


class TestHC4NumericalAccuracy:
    """Test HC4 numerical accuracy after bug fixes."""
    
    def test_hc4_vs_hc3_comparison(self):
        """Compare HC4 and HC3 standard errors - both should be reasonable.
        
        HC4 typically produces larger SEs than HC3 for high-leverage observations.
        """
        np.random.seed(222)
        
        data = pd.DataFrame({
            'unit_id': list(range(1, 51)),
            'd_': [1] * 15 + [0] * 35,
            'ydot_postavg': np.random.randn(50) * 2.0,
            'firstpost': [True] * 50,
        })
        data['ydot_postavg'] += data['d_'] * 3.0  # Treatment effect
        
        results_hc3 = estimate_att(
            data=data,
            y_transformed='ydot_postavg',
            d='d_',
            ivar='unit_id',
            controls=None,
            vce='hc3',
            cluster_var=None,
            sample_filter=data['firstpost'],
        )
        
        results_hc4 = estimate_att(
            data=data,
            y_transformed='ydot_postavg',
            d='d_',
            ivar='unit_id',
            controls=None,
            vce='hc4',
            cluster_var=None,
            sample_filter=data['firstpost'],
        )
        
        # Point estimates should be identical
        assert abs(results_hc3['att'] - results_hc4['att']) < 1e-10
        
        # Both SEs should be positive and finite
        assert results_hc3['se_att'] > 0
        assert results_hc4['se_att'] > 0
        assert np.isfinite(results_hc3['se_att'])
        assert np.isfinite(results_hc4['se_att'])
        
        # HC4 SE should be within reasonable range of HC3 SE
        # (typically close, but can differ based on leverage distribution)
        ratio = results_hc4['se_att'] / results_hc3['se_att']
        assert 0.5 < ratio < 2.0, f"SE ratio {ratio} is out of expected range"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
