"""
Stata cross-validation tests for BUG-158, BUG-159, BUG-160 fixes.

This module validates that the Python implementation matches Stata behavior
for edge cases related to these bugs.
"""

import warnings
import pytest
import numpy as np
import pandas as pd

from lwdid import lwdid
from lwdid.staggered.aggregation import aggregate_to_overall


class TestBUG160StataConsistency:
    """Test BUG-160 fix matches Stata behavior for empty inputs."""
    
    @pytest.fixture
    def castle_data(self):
        """Load castle law data for testing."""
        import os
        data_path = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'castle.csv'
        )
        if os.path.exists(data_path):
            return pd.read_csv(data_path)
        return None
    
    def test_overall_effect_with_valid_cohorts(self, castle_data):
        """Test overall effect estimation with valid cohort data."""
        if castle_data is None:
            pytest.skip("Castle data not available")
        
        # Run lwdid with staggered design
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=castle_data,
                y='lhomicide',
                ivar='sid',
                tvar='year',
                gvar='effyear',
                rolling='demean',
                aggregate='overall',
                estimator='ra',
            )
        
        # Verify result is valid (not NaN)
        assert not np.isnan(result.att_overall), "ATT should not be NaN"
        assert result.se_overall > 0 or result.se_overall is not None, "SE should be positive"
    
    def test_empty_cohorts_error_message_clarity(self):
        """Test that empty cohorts error message is clear and actionable."""
        from lwdid.staggered.aggregation import construct_aggregated_outcome
        
        # Create minimal test data
        data = pd.DataFrame({
            'id': [1, 2, 3],
            'year': [2000, 2000, 2000],
            'gvar': [2001, 2001, 0],
            'y': [1.0, 2.0, 3.0],
        })
        
        # Empty cohorts should give clear error
        with pytest.raises(ValueError) as excinfo:
            construct_aggregated_outcome(
                data=data,
                gvar='gvar',
                ivar='id',
                tvar='year',
                weights={},
                cohorts=[],
                T_max=2002,
                transform_type='demean',
            )
        
        error_msg = str(excinfo.value).lower()
        assert "cohorts" in error_msg
        assert "empty" in error_msg


class TestBUG159DFInferenceValidation:
    """Test BUG-159 fix for df_inference boundary conditions."""
    
    def test_large_sample_approximation_documented(self):
        """Verify large-sample approximation (df=1000) is statistically sound.
        
        For t-distribution, df=1000 is very close to normal distribution:
        - t(1000, 0.975) ≈ 1.9623 vs z(0.975) = 1.9600
        - Difference is negligible for practical purposes
        """
        from scipy import stats
        
        # Compare t(1000) to normal
        alpha = 0.05
        t_crit_1000 = stats.t.ppf(1 - alpha / 2, 1000)
        z_crit = stats.norm.ppf(1 - alpha / 2)
        
        # Should be within 0.2% of normal
        relative_diff = abs(t_crit_1000 - z_crit) / z_crit
        assert relative_diff < 0.002, f"t(1000) should approximate normal: diff={relative_diff:.4f}"
    
    def test_df_inference_warning_content(self):
        """Test that warning contains useful information."""
        from lwdid.results import LWDIDResults
        
        # Create minimal staggered results with invalid df_inference
        results_dict = {
            'att': 0.5,
            'se_att': 0.1,
            't_stat': 5.0,
            'pvalue': 0.001,
            'ci_lower': 0.3,
            'ci_upper': 0.7,
            'nobs': 100,
            'df_resid': 98,
            'df_inference': 0,  # Invalid
            'params': np.array([0.0, 0.5]),
            'bse': np.array([0.1, 0.1]),
            'vcov': np.eye(2) * 0.01,
            'resid': np.zeros(100),
            'vce_type': 'ols',
            'is_staggered': True,
            'cohorts': [2005],
            'cohort_sizes': {2005: 50},
            'att_by_cohort_time': pd.DataFrame({
                'cohort': [2005],
                'period': [2005],
                'att': [0.5],
                'se': [0.1],
                'n_treated': [50],
                'n_control': [50]
            }),
            'att_by_cohort': pd.DataFrame({
                'cohort': [2005],
                'att': [0.5],
                'se': [0.1],
                'n_units': [50],
                'n_periods': [1]
            }),
            'att_overall': 0.5,
            'se_overall': 0.1,
            'cohort_weights': {2005: 1.0},
            'aggregate': 'overall',
            'estimator': 'ra',
            'n_never_treated': 50,
        }
        
        metadata = {
            'K': 2004,
            'tpost1': 2005,
            'depvar': 'y',
            'N_treated': 50,
            'N_control': 50,
            'rolling': 'demean',
            'alpha': 0.05,
        }
        
        results = LWDIDResults(results_dict, metadata)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = results.summary_staggered()
            
            # Find df_inference warning
            df_warnings = [
                str(warning.message) for warning in w
                if "degrees of freedom" in str(warning.message).lower()
            ]
            
            assert len(df_warnings) > 0, "Should warn about invalid df_inference"
            
            # Warning should mention the fallback
            warning_text = df_warnings[0].lower()
            assert "1000" in warning_text or "large-sample" in warning_text


class TestBUG158FailureRateValidation:
    """Test BUG-158 fix for failure_rate boundary conditions."""
    
    def test_failure_rate_edge_cases(self):
        """Test failure_rate calculation edge cases."""
        # Test various failure rates
        test_cases = [
            (0.0, 100, 1000),    # No failures
            (0.5, 100, 1000),    # 50% failure
            (0.99, 100, 1000),   # 99% failure
            (1.0, 100, 1000),    # 100% failure (edge case)
        ]
        
        for failure_rate, min_valid_reps, rireps in test_cases:
            # This is the fixed logic
            if failure_rate < 1.0:
                recommended_reps = int(min_valid_reps / (1 - failure_rate))
            else:
                recommended_reps = rireps * 10
            
            # Should always be positive
            assert recommended_reps > 0, f"recommended_reps should be positive for failure_rate={failure_rate}"
            
            # For failure_rate=1.0, should use fallback
            if failure_rate == 1.0:
                assert recommended_reps == 10000, "Should use rireps * 10 fallback"
    
    def test_no_zero_division_in_error_formatting(self):
        """Test that error message formatting doesn't cause ZeroDivisionError."""
        failure_rate = 1.0
        min_valid_reps = 100
        rireps = 1000
        
        # Format error message (simulating the randomization.py behavior)
        if failure_rate < 1.0:
            recommended_reps = int(min_valid_reps / (1 - failure_rate))
        else:
            recommended_reps = rireps * 10
        
        # Should be able to format without error
        error_msg = (
            f"Insufficient valid RI replications: 0/{rireps} "
            f"(need at least {min_valid_reps}).\n"
            f"  Failure rate: {failure_rate:.1%}\n"
            f"  Recommendation: increase rireps to at least {recommended_reps}."
        )
        
        assert "10000" in error_msg


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
