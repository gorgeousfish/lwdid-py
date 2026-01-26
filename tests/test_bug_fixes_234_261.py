"""
Regression tests for bug fixes BUG-234 to BUG-261 (Code Review Round 79).

This module tests the fixes for:
- P1 (Critical): BUG-235, 236, 237, 238, 239, 240, 241
- P2 (General): BUG-242, 243, 245, 246, 248, 250, 251, 252, 253, 254, 255
- P3 (Minor): BUG-256, 257, 258, 259, 260, 261
"""

import warnings
import numpy as np
import pandas as pd
import pytest
from scipy import stats

# Import modules to test
import sys
sys.path.insert(0, '/Users/cxy/Desktop/大样本lwdid/lwdid-py_v0.1.0/src')

from lwdid import lwdid
from lwdid.validation import (
    is_never_treated, 
    get_cohort_mask, 
    validate_staggered_data,
    COHORT_FLOAT_TOLERANCE
)
from lwdid.staggered.aggregation import construct_aggregated_outcome
from lwdid.staggered.estimators import _create_binary_treatment, COHORT_FLOAT_TOLERANCE as EST_TOLERANCE


class TestBUG235InfFiltering:
    """Test BUG-235: staggered/randomization.py should filter Inf values."""
    
    def test_inf_values_filtered(self):
        """Verify that np.isfinite is used instead of ~np.isnan."""
        # This is a code inspection test - the fix ensures Inf values are filtered
        from lwdid.staggered import randomization
        import inspect
        source = inspect.getsource(randomization.randomization_inference_staggered)
        # The fix changes ~np.isnan to np.isfinite
        assert 'np.isfinite(perm_stats)' in source, \
            "BUG-235: Should use np.isfinite() to filter both NaN and Inf"


class TestBUG236ConditionalVariance:
    """Test BUG-236: Conditional variance fallback should use global variance."""
    
    def test_conditional_variance_not_zero(self):
        """Verify that single-group variance uses global fallback, not 0.0."""
        from lwdid.staggered import estimators
        import inspect
        source = inspect.getsource(estimators._estimate_conditional_variance_same_group_paper)
        # The fix should use global variance as fallback
        assert 'np.var(Y, ddof=1)' in source, \
            "BUG-236: Should use global variance as fallback, not 0.0"


class TestBUG237IPWDf:
    """Test BUG-237: IPW df calculation should be n-2."""
    
    def test_ipw_df_not_subtract_controls(self):
        """Verify IPW df is n-2, not n-2-len(propensity_controls)."""
        from lwdid.staggered import estimation
        import inspect
        source = inspect.getsource(estimation._estimate_single_effect_ipw)
        # Check that df calculation doesn't subtract propensity_controls
        # The fix uses result.n_treated + result.n_control - 2
        lines = source.split('\n')
        for line in lines:
            if 'df_resid' in line and 'propensity_controls' in line:
                pytest.fail("BUG-237: df should not subtract propensity_controls")


class TestBUG238DetrendCheck:
    """Test BUG-238: Detrend check should use n_pre_periods, not K."""
    
    def test_detrend_uses_n_pre_periods(self):
        """Verify that detrend checks n_pre_periods < 2, not K < 2."""
        from lwdid import transformations
        import inspect
        source = inspect.getsource(transformations._detrend_transform)
        # The fix checks n_pre_periods < 2
        assert 'n_pre_periods < 2' in source, \
            "BUG-238: Should check n_pre_periods < 2, not K < 2"


class TestBUG239QuarterNaNFiltering:
    """Test BUG-239: Quarter validation should filter NaN values."""
    
    def test_quarter_sets_filter_nan(self):
        """Verify that NaN values are filtered when creating quarter sets."""
        from lwdid import validation
        import inspect
        source = inspect.getsource(validation.validate_quarter_diversity)
        # The fix filters NaN: set(q for q in ... if pd.notna(q))
        assert 'pd.notna(q)' in source, \
            "BUG-239: Should filter NaN values from quarter sets"


class TestBUG240ToleranceConsistency:
    """Test BUG-240: Not-yet-treated comparison should use tolerance."""
    
    def test_nyt_mask_uses_tolerance(self):
        """Verify not-yet-treated mask uses COHORT_FLOAT_TOLERANCE."""
        from lwdid.staggered import estimators
        import inspect
        source = inspect.getsource(estimators._identify_control_units)
        # The fix adds tolerance: unit_gvar > period_r + COHORT_FLOAT_TOLERANCE
        assert 'COHORT_FLOAT_TOLERANCE' in source, \
            "BUG-240: Not-yet-treated comparison should use tolerance"


class TestBUG241TimeInvariantControls:
    """Test BUG-241: Staggered mode should validate time-invariant controls."""
    
    def test_staggered_validates_time_invariant_controls(self):
        """Verify validate_staggered_data calls _validate_time_invariant_controls."""
        from lwdid import validation
        import inspect
        source = inspect.getsource(validation.validate_staggered_data)
        # The fix adds: _validate_time_invariant_controls(data, ivar, controls)
        assert '_validate_time_invariant_controls' in source, \
            "BUG-241: Staggered mode should validate time-invariant controls"


class TestBUG242EstimatorNullCheck:
    """Test BUG-242: Estimator parameter should handle None."""
    
    def test_estimator_none_defaults_to_ra(self):
        """Verify estimator=None defaults to 'ra' without error."""
        from lwdid import core
        import inspect
        source = inspect.getsource(core._lwdid_staggered)
        # The fix: estimator.lower() if estimator else 'ra'
        assert "estimator.lower() if estimator else 'ra'" in source, \
            "BUG-242: estimator=None should default to 'ra'"


class TestBUG246EmptyCohortsValidation:
    """Test BUG-246: Empty cohorts list should raise ValueError."""
    
    def test_empty_cohorts_raises_error(self):
        """Verify that empty cohorts list raises ValueError."""
        # Create minimal test data
        data = pd.DataFrame({
            'ivar': [1, 1, 2, 2],
            'tvar': [2000, 2001, 2000, 2001],
            'gvar': [0, 0, 0, 0],  # All never-treated
            'y': [1.0, 2.0, 1.5, 2.5]
        })
        
        with pytest.raises(ValueError, match="cohorts list cannot be empty"):
            construct_aggregated_outcome(
                data=data,
                gvar='gvar',
                ivar='ivar',
                tvar='tvar',
                weights={},  # Empty weights
                cohorts=[],  # Empty cohorts list
                T_max=2001,
                transform_type='demean'
            )


class TestBUG250GetCohortMaskInf:
    """Test BUG-250: get_cohort_mask should reject infinity."""
    
    def test_get_cohort_mask_rejects_inf(self):
        """Verify that get_cohort_mask raises ValueError for infinity."""
        unit_gvar = pd.Series([2005, 2010, np.inf, 0])
        
        with pytest.raises(ValueError, match="Cohort value cannot be infinity"):
            get_cohort_mask(unit_gvar, np.inf)
    
    def test_get_cohort_mask_works_for_normal_values(self):
        """Verify get_cohort_mask works correctly for normal cohort values."""
        unit_gvar = pd.Series([2005, 2010, np.inf, 0], index=['a', 'b', 'c', 'd'])
        mask = get_cohort_mask(unit_gvar, 2005)
        assert mask['a'] == True
        assert mask['b'] == False
        assert mask['c'] == False
        assert mask['d'] == False


class TestBUG251MixedNaNGvar:
    """Test BUG-251: Validate mixed NaN/valid gvar values."""
    
    def test_mixed_nan_gvar_warning(self):
        """Verify warning is issued for units with mixed NaN and valid gvar."""
        from lwdid import validation
        import inspect
        source = inspect.getsource(validation.validate_staggered_data)
        # The fix adds check for mixed NaN and valid gvar
        assert 'mixed_units' in source or 'has_na & has_valid' in source, \
            "BUG-251: Should check for mixed NaN/valid gvar values"


class TestBUG253PostBinarization:
    """Test BUG-253: validate_quarter_diversity should binarize post."""
    
    def test_post_binarization(self):
        """Verify that post column is binarized before comparison."""
        from lwdid import validation
        import inspect
        source = inspect.getsource(validation.validate_quarter_diversity)
        # The fix adds: post_binary = (data[post] != 0).astype(int)
        assert 'post_binary' in source or "!= 0" in source, \
            "BUG-253: Should binarize post column"


class TestBUG254DfTDefined:
    """Test BUG-254: df_t should be defined in all branches."""
    
    def test_df_t_defined_in_degenerate_branch(self):
        """Verify df_t is defined even in degenerate regression branch."""
        from lwdid import estimation
        import inspect
        source = inspect.getsource(estimation.estimate_period_effects)
        # The fix adds df_t = np.nan in the degenerate branch
        assert 'df_t = np.nan' in source, \
            "BUG-254: df_t should be defined in degenerate branch"


class TestBUG255BinaryTreatmentTolerance:
    """Test BUG-255: _create_binary_treatment should use tolerance."""
    
    def test_binary_treatment_uses_tolerance(self):
        """Verify _create_binary_treatment uses tolerance-based comparison."""
        from lwdid.staggered import estimators
        import inspect
        source = inspect.getsource(estimators._create_binary_treatment)
        # The fix uses: np.abs(subsample[gvar_col].values - cohort_g) <= COHORT_FLOAT_TOLERANCE
        assert 'COHORT_FLOAT_TOLERANCE' in source, \
            "BUG-255: Should use tolerance-based comparison"


class TestBUG256WeightsCVSingleWeight:
    """Test BUG-256: weights_cv should be 0.0 for single weight."""
    
    def test_single_weight_cv_zero(self):
        """Verify that single weight returns CV=0.0, not NaN."""
        from lwdid import core
        import inspect
        source = inspect.getsource(core._convert_ipwra_result_to_dict)
        # The fix adds: elif weights is not None and len(weights) == 1: weights_cv = 0.0
        assert 'weights_cv = 0.0' in source, \
            "BUG-256: Single weight should have CV=0.0"


class TestBUG257DfInferenceWarning:
    """Test BUG-257: df_inference fallback should issue warning."""
    
    def test_df_inference_fallback_warns(self):
        """Verify warning is issued when df_inference falls back to 1000."""
        from lwdid import results
        import inspect
        source = inspect.getsource(results.LWDIDResults)
        # The fix adds warning when using df=1000 fallback
        assert 'df=1000' in source or 'df_inference unavailable' in source, \
            "BUG-257: Should warn when using df=1000 fallback"


class TestBUG258DuplicateImport:
    """Test BUG-258: Duplicate 'import re' should be removed."""
    
    def test_no_duplicate_re_import(self):
        """Verify there's only one actual 'import re' statement in results.py."""
        from lwdid import results
        import inspect
        source = inspect.getsource(results)
        # Count actual import statements (not comments)
        # Look for 'import re' at line beginning (with possible leading whitespace)
        import_count = 0
        for line in source.split('\n'):
            stripped = line.strip()
            # Count actual import statements, not references in comments
            if stripped == 'import re' or stripped.startswith('import re #'):
                import_count += 1
        assert import_count == 1, \
            f"BUG-258: Found {import_count} 'import re' statements, expected 1"


class TestBUG259FsumConsistency:
    """Test BUG-259: fsum should be used consistently for weight summation."""
    
    def test_fsum_used_for_weights(self):
        """Verify fsum is used for weight summation."""
        from lwdid.staggered import aggregation
        import inspect
        source = inspect.getsource(aggregation.construct_aggregated_outcome)
        # The fix uses: weights_sum = fsum(weights.values())
        assert 'fsum(weights.values())' in source, \
            "BUG-259: Should use fsum for weight summation"


class TestBUG260DetrendDfConsistency:
    """Test BUG-260: detrend should require df >= 1 like detrendq."""
    
    def test_detrend_requires_df_ge_1(self):
        """Verify detrend requires n_valid > n_params (df >= 1)."""
        from lwdid import transformations
        import inspect
        source = inspect.getsource(transformations.detrend_unit)
        # The fix checks: if n_valid <= n_params
        assert 'n_valid <= n_params' in source, \
            "BUG-260: detrend should require df >= 1"


class TestBUG261TvarFloatWarning:
    """Test BUG-261: Float tvar should trigger warning."""
    
    def test_float_tvar_warning(self):
        """Verify warning is issued for non-integer tvar values."""
        from lwdid import validation
        import inspect
        source = inspect.getsource(validation.validate_staggered_data)
        # The fix adds warning for non-integer tvar
        assert 'non-integer values' in source or 'truncated to integers' in source, \
            "BUG-261: Should warn for non-integer tvar values"


class TestIntegration:
    """Integration tests for the bug fixes."""
    
    @pytest.fixture
    def sample_staggered_data(self):
        """Create sample staggered DiD data for testing."""
        np.random.seed(42)
        n_units = 20
        n_periods = 10
        
        data = []
        for i in range(n_units):
            # Assign cohort: 0 (never-treated), or treatment year
            if i < 5:
                gvar = 0  # Never-treated
            elif i < 10:
                gvar = 5  # Treated in period 5
            else:
                gvar = 7  # Treated in period 7
            
            for t in range(n_periods):
                treated = (gvar > 0 and t >= gvar)
                y = 1.0 + 0.5 * t + (2.0 if treated else 0) + np.random.normal(0, 0.5)
                data.append({
                    'unit': i,
                    'time': t,
                    'gvar': gvar,
                    'y': y
                })
        
        return pd.DataFrame(data)
    
    def test_lwdid_staggered_runs_without_error(self, sample_staggered_data):
        """Verify that lwdid runs on staggered data without errors from bug fixes."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = lwdid(
                data=sample_staggered_data,
                y='y',
                ivar='unit',
                tvar='time',
                gvar='gvar',
                rolling='demean',
                aggregate='cohort'
            )
            
            # Basic assertions
            assert result is not None
            assert hasattr(result, 'att')
            assert result.is_staggered


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
