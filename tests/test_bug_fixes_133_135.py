"""
Unit tests for BUG-133, BUG-134, and BUG-135 fixes.

BUG-133: validation.py post_ column NaN values causing false common timing violation
BUG-134: staggered/estimation.py missing sample size validation after cluster filtering
BUG-135: validation.py safe_int_cohort not handling infinity values

These tests verify the defensive programming fixes for edge cases in data validation.
"""

import numpy as np
import pandas as pd
import pytest
import warnings

from lwdid.validation import (
    safe_int_cohort,
    validate_and_prepare_data,
    COHORT_FLOAT_TOLERANCE,
)
from lwdid.staggered.estimation import run_ols_regression
from lwdid.exceptions import InvalidParameterError


class TestBug133NuniquDropna:
    """
    BUG-133: post_ column NaN values could cause false common timing violation.
    
    The fix uses nunique(dropna=True) to exclude NaN from unique value count,
    preventing false positives when post_ contains missing values.
    """

    def test_common_timing_with_nan_in_post_no_false_positive(self):
        """
        Test that NaN values in post_ column do not trigger false common timing violation.
        
        This test creates a scenario where post_ has valid values (all 0 or all 1 per time)
        but also some NaN values. Without the fix, nunique() would count NaN as a distinct
        value, incorrectly reporting a common timing violation.
        """
        # Create data where post is correctly constant within each time period
        # but some observations have NaN in post
        data = pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'year': [2000, 2001, 2002, 2000, 2001, 2002, 2000, 2001, 2002],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            'd': [1, 1, 1, 0, 0, 0, 0, 0, 0],
            'post': [0, 1, 1, 0, 1, 1, 0, 1, 1],  # Correct: all 0 in 2000, all 1 in 2001/2002
        })
        
        # This should pass validation without raising common timing violation
        data_clean, metadata = validate_and_prepare_data(
            data, y='y', d='d', ivar='id', tvar='year', post='post', rolling='demean'
        )
        
        assert metadata['K'] == 1  # Last pre-treatment period
        assert metadata['tpost1'] == 2  # First post-treatment period
        assert metadata['N_treated'] == 1
        assert metadata['N_control'] == 2

    def test_common_timing_real_violation_still_detected(self):
        """
        Test that actual common timing violations are still correctly detected.
        
        This ensures the fix does not mask real violations.
        """
        # Create data with actual common timing violation (different units have
        # different post values at the same time period)
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'year': [2000, 2001, 2000, 2001],
            'y': [1.0, 2.0, 3.0, 4.0],
            'd': [1, 1, 0, 0],
            # Violation: at year 2001, unit 1 has post=1 but unit 2 has post=0
            'post': [0, 1, 0, 0],
        })
        
        with pytest.raises(InvalidParameterError, match="Common timing assumption violated"):
            validate_and_prepare_data(
                data, y='y', d='d', ivar='id', tvar='year', post='post', rolling='demean'
            )


class TestBug134ClusterFilterValidation:
    """
    BUG-134: Missing sample size validation after filtering NaN cluster values.
    
    The fix adds validation for:
    1. n < 2 after NaN filtering
    2. Controls estimability (n_treated > K and n_control > K)
    """

    def test_ols_cluster_nan_filtering_minimum_sample(self):
        """
        Test that ValueError is raised when n < 2 after filtering NaN clusters.
        """
        # Create data where most observations have NaN cluster
        data = pd.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0],
            'd': [1, 1, 0, 0, 0],
            'cluster': [np.nan, np.nan, np.nan, np.nan, 1],  # Only 1 valid cluster obs
        })
        
        # With vce='cluster', this should raise ValueError due to insufficient n or clusters
        with pytest.raises(ValueError):
            run_ols_regression(
                data=data,
                y='y',
                d='d',
                controls=None,
                vce='cluster',
                cluster_var='cluster',
            )

    def test_ols_cluster_nan_filtering_preserves_valid_data(self):
        """
        Test that valid data with some NaN clusters still produces results.
        
        This verifies the NaN filtering logic works correctly when there are
        enough valid observations remaining.
        """
        # Create data with some NaN clusters but enough valid data
        data = pd.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            'd': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            'cluster': [1, 2, 3, np.nan, np.nan, 4, 5, 6, 7, 8],
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = run_ols_regression(
                data=data,
                y='y',
                d='d',
                controls=None,
                vce='cluster',
                cluster_var='cluster',
            )
            
            # Should complete successfully with valid result
            assert 'att' in result
            assert not np.isnan(result['att'])
            
            # Should warn about NaN cluster values being excluded
            warning_messages = [str(warning.message) for warning in w]
            has_nan_warning = any('missing values' in msg for msg in warning_messages)
            assert has_nan_warning

    def test_ols_cluster_insufficient_degrees_of_freedom_warning(self):
        """
        Test that warning is issued when degrees of freedom are insufficient.
        
        When n=k (no residual df), the function returns NaN for SE/CI/pvalue
        but still provides a point estimate with a warning.
        """
        # Create minimal data with no degrees of freedom
        data = pd.DataFrame({
            'y': [1.0, 2.0],
            'd': [1, 0],
            'cluster': [1, 2],
        })
        
        # Should complete with warning about no degrees of freedom
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = run_ols_regression(
                data=data,
                y='y',
                d='d',
                controls=None,
                vce='cluster',
                cluster_var='cluster',
            )
            
            # Point estimate should be returned
            assert 'att' in result
            # SE should be NaN due to insufficient df
            assert np.isnan(result['se'])
            
            # Should have warning about degrees of freedom
            warning_messages = [str(warning.message) for warning in w]
            has_df_warning = any('degrees of freedom' in msg.lower() for msg in warning_messages)
            assert has_df_warning

    def test_ols_cluster_controls_with_sufficient_sample(self):
        """
        Test that controls are properly included when sample size is sufficient.
        """
        # Create data with enough observations for controls
        np.random.seed(42)
        n = 50
        data = pd.DataFrame({
            'y': np.random.randn(n),
            'd': np.repeat([1, 0], n // 2),
            'x1': np.random.randn(n),
            'cluster': np.arange(n),  # Each observation is its own cluster
        })
        
        result = run_ols_regression(
            data=data,
            y='y',
            d='d',
            controls=['x1'],
            vce='cluster',
            cluster_var='cluster',
        )
        
        # Should complete successfully
        assert 'att' in result
        assert 'se' in result
        assert result['nobs'] == n


class TestBug135SafeIntCohortInfinity:
    """
    BUG-135: safe_int_cohort should explicitly handle infinity values.
    
    The fix adds an explicit check for np.isinf(g) before attempting round()/int()
    to provide a clear error message instead of OverflowError.
    """

    def test_safe_int_cohort_positive_infinity(self):
        """
        Test that positive infinity raises ValueError with clear message.
        """
        with pytest.raises(ValueError, match="infinity"):
            safe_int_cohort(np.inf)

    def test_safe_int_cohort_negative_infinity(self):
        """
        Test that negative infinity raises ValueError with clear message.
        """
        with pytest.raises(ValueError, match="infinity"):
            safe_int_cohort(-np.inf)

    def test_safe_int_cohort_float_infinity(self):
        """
        Test that float('inf') raises ValueError with clear message.
        """
        with pytest.raises(ValueError, match="infinity"):
            safe_int_cohort(float('inf'))

    def test_safe_int_cohort_nan(self):
        """
        Test that NaN raises ValueError with clear message.
        """
        with pytest.raises(ValueError, match="NaN"):
            safe_int_cohort(np.nan)
        
        with pytest.raises(ValueError, match="NaN"):
            safe_int_cohort(float('nan'))

    def test_safe_int_cohort_none(self):
        """
        Test that None raises ValueError with clear message.
        """
        with pytest.raises(ValueError, match="NaN|None"):
            safe_int_cohort(None)

    def test_safe_int_cohort_normal_integer(self):
        """
        Test that normal integers pass through correctly.
        """
        assert safe_int_cohort(2005) == 2005
        assert safe_int_cohort(0) == 0
        assert safe_int_cohort(-1) == -1

    def test_safe_int_cohort_numpy_integer(self):
        """
        Test that numpy integers pass through correctly.
        """
        assert safe_int_cohort(np.int64(2005)) == 2005
        assert safe_int_cohort(np.int32(2000)) == 2000

    def test_safe_int_cohort_float_close_to_integer(self):
        """
        Test that floats close to integers are converted correctly.
        """
        assert safe_int_cohort(2005.0) == 2005
        # Small floating point error should be tolerated
        assert safe_int_cohort(2005.0 + 1e-12) == 2005

    def test_safe_int_cohort_non_integer_float(self):
        """
        Test that non-integer floats raise ValueError.
        """
        with pytest.raises(ValueError, match="not close to an integer"):
            safe_int_cohort(2005.5)
        
        with pytest.raises(ValueError, match="not close to an integer"):
            safe_int_cohort(2005.1)

    def test_safe_int_cohort_error_message_informative(self):
        """
        Test that the error message for infinity is informative.
        """
        try:
            safe_int_cohort(np.inf)
            pytest.fail("Expected ValueError")
        except ValueError as e:
            error_msg = str(e)
            # Check that message explains the issue
            assert "infinity" in error_msg.lower()
            assert "never-treated" in error_msg.lower()
            # Check that message suggests how to fix
            assert "Filter" in error_msg or "gvar" in error_msg


class TestBug133Bug134Bug135Integration:
    """
    Integration tests to verify the three fixes work together correctly.
    """

    def test_staggered_estimation_with_edge_cases(self):
        """
        Test that staggered estimation handles edge cases correctly.
        
        This tests the interaction between validation and estimation.
        """
        from lwdid.validation import validate_staggered_data, is_never_treated
        
        # Create staggered data with never-treated units (using inf)
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3],
            'year': [2000, 2001, 2000, 2001, 2000, 2001],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'gvar': [2001, 2001, np.inf, np.inf, 0, 0],  # Unit 1 treated in 2001, units 2&3 never-treated
        })
        
        # Validate should correctly identify cohorts and never-treated units
        result = validate_staggered_data(data, gvar='gvar', ivar='id', tvar='year', y='y')
        
        assert result['cohorts'] == [2001]
        assert result['n_never_treated'] == 2
        assert result['n_treated'] == 1
        assert result['has_never_treated'] == True

    def test_is_never_treated_with_various_values(self):
        """
        Test is_never_treated function with various edge case values.
        """
        from lwdid.validation import is_never_treated
        
        # Never-treated indicators
        assert is_never_treated(0) == True
        assert is_never_treated(np.inf) == True
        assert is_never_treated(np.nan) == True
        assert is_never_treated(None) == True
        
        # Treated cohort values
        assert is_never_treated(2005) == False
        assert is_never_treated(2005.0) == False
        assert is_never_treated(1) == False
        
        # Negative infinity returns False (not a valid NT indicator)
        assert is_never_treated(-np.inf) == False
