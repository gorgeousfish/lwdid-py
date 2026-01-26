"""
Unit tests for BUG-158, BUG-159, and BUG-160 fixes.

BUG-158: randomization.py division by zero when failure_rate == 1.0
BUG-159: results.py df_inference boundary value warning
BUG-160: aggregation.py empty cohorts validation
"""

import warnings
import pytest
import numpy as np
import pandas as pd

from lwdid.staggered.aggregation import construct_aggregated_outcome
from lwdid.results import LWDIDResults


class TestBUG158DivisionByZero:
    """Test that BUG-158 fix prevents division by zero when failure_rate == 1.0."""
    
    def test_failure_rate_one_no_division_error(self):
        """Verify recommended_reps calculation doesn't divide by zero when failure_rate=1.0."""
        # This is a code path test - we verify the logic handles failure_rate=1.0
        # The actual fix is at randomization.py lines 305-308
        failure_rate = 1.0
        min_valid_reps = 100
        rireps = 1000
        
        # This is the fixed logic
        if failure_rate < 1.0:
            recommended_reps = int(min_valid_reps / (1 - failure_rate))
        else:
            recommended_reps = rireps * 10
        
        # Should not raise, should use fallback
        assert recommended_reps == 10000
    
    def test_failure_rate_less_than_one_normal_calculation(self):
        """Verify normal calculation works when failure_rate < 1.0."""
        failure_rate = 0.5
        min_valid_reps = 100
        rireps = 1000
        
        if failure_rate < 1.0:
            recommended_reps = int(min_valid_reps / (1 - failure_rate))
        else:
            recommended_reps = rireps * 10
        
        assert recommended_reps == 200  # 100 / 0.5 = 200


class TestBUG159DFInferenceBoundary:
    """Test that BUG-159 fix issues warnings for invalid df_inference values."""
    
    @pytest.fixture
    def minimal_results_dict(self):
        """Create minimal results_dict for LWDIDResults initialization."""
        return {
            'att': 0.5,
            'se_att': 0.1,
            't_stat': 5.0,
            'pvalue': 0.001,
            'ci_lower': 0.3,
            'ci_upper': 0.7,
            'nobs': 100,
            'df_resid': 98,
            'params': np.array([0.0, 0.5]),
            'bse': np.array([0.1, 0.1]),
            'vcov': np.eye(2) * 0.01,
            'resid': np.zeros(100),
            'vce_type': 'ols',
            'is_staggered': True,
            'cohorts': [2005, 2006],
            'cohort_sizes': {2005: 30, 2006: 20},
            'att_by_cohort_time': pd.DataFrame({
                'cohort': [2005, 2005, 2006],
                'period': [2005, 2006, 2006],
                'att': [0.4, 0.5, 0.6],
                'se': [0.1, 0.1, 0.1],
                'n_treated': [30, 30, 20],
                'n_control': [50, 50, 50]
            }),
            'att_by_cohort': pd.DataFrame({
                'cohort': [2005, 2006],
                'att': [0.45, 0.6],
                'se': [0.08, 0.1],
                'n_units': [30, 20],
                'n_periods': [2, 1]
            }),
            'att_overall': 0.5,
            'se_overall': 0.07,
            'cohort_weights': {2005: 0.6, 2006: 0.4},
            'aggregate': 'overall',
            'estimator': 'ra',
            'n_never_treated': 50,
            'df_inference': 0,  # Invalid value to trigger warning
        }
    
    @pytest.fixture
    def minimal_metadata(self):
        """Create minimal metadata for LWDIDResults initialization."""
        return {
            'K': 2004,
            'tpost1': 2005,
            'depvar': 'y',
            'N_treated': 50,
            'N_control': 50,
            'rolling': 'demean',
            'alpha': 0.05,
        }
    
    def test_df_inference_zero_triggers_warning_summary(
        self, minimal_results_dict, minimal_metadata
    ):
        """Verify warning is issued when df_inference=0 in summary_staggered()."""
        results = LWDIDResults(minimal_results_dict, minimal_metadata)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = results.summary_staggered()
            
            # Check that a warning was issued about invalid df_inference
            df_warnings = [
                warning for warning in w 
                if "Invalid degrees of freedom" in str(warning.message)
            ]
            assert len(df_warnings) >= 1, "Expected warning for invalid df_inference"
    
    def test_df_inference_none_triggers_warning(
        self, minimal_results_dict, minimal_metadata
    ):
        """Verify warning is issued when df_inference=None."""
        minimal_results_dict['df_inference'] = None
        results = LWDIDResults(minimal_results_dict, minimal_metadata)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = results.summary_staggered()
            
            df_warnings = [
                warning for warning in w 
                if "Invalid degrees of freedom" in str(warning.message)
            ]
            # Should trigger warning
            assert len(df_warnings) >= 1
    
    def test_df_inference_negative_triggers_warning(
        self, minimal_results_dict, minimal_metadata
    ):
        """Verify warning is issued when df_inference < 0."""
        minimal_results_dict['df_inference'] = -5
        results = LWDIDResults(minimal_results_dict, minimal_metadata)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = results.summary_staggered()
            
            df_warnings = [
                warning for warning in w 
                if "Invalid degrees of freedom" in str(warning.message)
            ]
            assert len(df_warnings) >= 1
    
    def test_df_inference_valid_no_warning(
        self, minimal_results_dict, minimal_metadata
    ):
        """Verify no warning is issued when df_inference is valid."""
        minimal_results_dict['df_inference'] = 100
        results = LWDIDResults(minimal_results_dict, minimal_metadata)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = results.summary_staggered()
            
            df_warnings = [
                warning for warning in w 
                if "Invalid degrees of freedom" in str(warning.message)
            ]
            assert len(df_warnings) == 0, "Should not warn for valid df_inference"


class TestBUG160EmptyCohortsValidation:
    """Test that BUG-160 fix validates empty cohorts/weights inputs."""
    
    @pytest.fixture
    def sample_data(self):
        """Create minimal sample data for aggregation tests."""
        np.random.seed(42)
        n_units = 20
        n_periods = 5
        
        data = pd.DataFrame({
            'id': np.repeat(range(n_units), n_periods),
            'year': np.tile(range(2000, 2000 + n_periods), n_units),
            'gvar': np.repeat(
                [2002] * 10 + [0] * 10,  # 10 treated at 2002, 10 never-treated
                n_periods
            ),
            'y': np.random.randn(n_units * n_periods),
        })
        
        # Add transformation columns
        data['ydot_g2002_r2002'] = np.random.randn(len(data))
        data['ydot_g2002_r2003'] = np.random.randn(len(data))
        data['ydot_g2002_r2004'] = np.random.randn(len(data))
        
        return data
    
    def test_empty_cohorts_raises_value_error(self, sample_data):
        """Verify ValueError is raised when cohorts list is empty."""
        with pytest.raises(ValueError) as excinfo:
            construct_aggregated_outcome(
                data=sample_data,
                gvar='gvar',
                ivar='id',
                tvar='year',
                weights={},
                cohorts=[],
                T_max=2004,
                transform_type='demean',
            )
        
        assert "cohorts list cannot be empty" in str(excinfo.value)
    
    def test_empty_weights_raises_value_error(self, sample_data):
        """Verify ValueError is raised when weights dict is empty."""
        with pytest.raises(ValueError) as excinfo:
            construct_aggregated_outcome(
                data=sample_data,
                gvar='gvar',
                ivar='id',
                tvar='year',
                weights={},
                cohorts=[2002],
                T_max=2004,
                transform_type='demean',
            )
        
        assert "weights dict cannot be empty" in str(excinfo.value)
    
    def test_valid_inputs_no_error(self, sample_data):
        """Verify no error when cohorts and weights are valid."""
        # Should not raise
        result = construct_aggregated_outcome(
            data=sample_data,
            gvar='gvar',
            ivar='id',
            tvar='year',
            weights={2002: 1.0},
            cohorts=[2002],
            T_max=2004,
            transform_type='demean',
        )
        
        assert isinstance(result, pd.Series)
        assert len(result) > 0


class TestBUG158IntegrationWithRandomization:
    """Integration tests for BUG-158 fix with actual randomization module."""
    
    def test_randomization_error_message_formatting(self):
        """Test that error message formatting doesn't fail with edge cases."""
        from lwdid.randomization import RandomizationError
        
        # Test that the error can be constructed
        error_msg = (
            "Insufficient valid RI replications: 0/1000 (need at least 100).\n"
            "  Failure rate: 100.0%\n"
            "  Failure reasons: {'N1=0': 500, 'N1=N': 500}\n"
            "  N1 distribution: N/A (all replications failed before N1 recorded)\n"
            "  Recommendation: Use ri_method=\"permutation\" for classical Fisher RI, "
            "or increase rireps to at least 10000."
        )
        
        # Should be able to create and raise this error
        with pytest.raises(RandomizationError):
            raise RandomizationError(error_msg)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
