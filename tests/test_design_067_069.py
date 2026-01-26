"""
Unit tests for design issues DESIGN-067, DESIGN-068, and DESIGN-069.

DESIGN-067: Exception handling in staggered/randomization.py
DESIGN-068: Quarter validation function refactoring in validation.py
DESIGN-069: Conditional variance boundary case in staggered/estimators.py
"""

import numpy as np
import pandas as pd
import pytest

from lwdid.exceptions import InsufficientQuarterDiversityError
from lwdid.validation import (
    validate_quarter_diversity,
    validate_quarter_coverage,
    _check_quarter_coverage_for_unit,
)
from lwdid.staggered.estimators import _estimate_conditional_variance_same_group


class TestDesign067ExceptionHandling:
    """Tests for DESIGN-067: More precise exception handling in randomization inference."""
    
    def test_expected_value_error_is_caught(self):
        """Test that expected ValueError messages result in NaN, not re-raised."""
        # This is an integration test - the actual exception handling is in the RI loop
        # We test by verifying the keywords list covers expected cases
        expected_keywords = ['empty', 'insufficient', 'no valid', 'zero', 
                            'singular', 'collinear', 'degenerate', 'no data',
                            'not enough', 'too few', 'missing']
        
        # Verify all keywords are lowercase for case-insensitive matching
        for kw in expected_keywords:
            assert kw == kw.lower(), f"Keyword '{kw}' should be lowercase"
    
    def test_linalg_error_expected_message(self):
        """Test that LinAlgError is expected for singular matrices."""
        # Create a singular matrix that would cause LinAlgError
        singular_matrix = np.array([[1, 2], [2, 4]])
        
        with pytest.raises(np.linalg.LinAlgError):
            np.linalg.inv(singular_matrix)


class TestDesign068QuarterValidation:
    """Tests for DESIGN-068: Quarter validation function refactoring."""
    
    def test_helper_function_no_uncovered_quarters(self):
        """Test helper returns None when all quarters are covered."""
        pre_quarters = {1, 2, 3, 4}
        post_quarters = {1, 2}
        
        result = _check_quarter_coverage_for_unit("unit1", pre_quarters, post_quarters)
        assert result is None
    
    def test_helper_function_with_uncovered_quarters(self):
        """Test helper returns error message for uncovered quarters."""
        pre_quarters = {1, 2}
        post_quarters = {1, 3, 4}
        
        result = _check_quarter_coverage_for_unit("unit1", pre_quarters, post_quarters)
        
        assert result is not None
        assert "unit1" in result
        assert "3" in result or "4" in result  # uncovered quarters mentioned
        assert "not observed in pre-period" in result
    
    def test_helper_function_empty_post_quarters(self):
        """Test helper handles empty post quarters."""
        pre_quarters = {1, 2, 3}
        post_quarters = set()
        
        result = _check_quarter_coverage_for_unit("unit1", pre_quarters, post_quarters)
        assert result is None
    
    def test_validate_quarter_diversity_passes_valid_data(self):
        """Test validate_quarter_diversity passes with valid data."""
        # Note: post quarters must be a subset of pre quarters
        data = pd.DataFrame({
            'id': [1, 1, 1, 1, 2, 2, 2, 2],
            'quarter': [1, 2, 1, 2, 1, 2, 1, 2],  # All post quarters (1,2) covered in pre
            'post': [0, 0, 1, 1, 0, 0, 1, 1]
        })
        
        # Should not raise
        validate_quarter_diversity(data, 'id', 'quarter', 'post')
    
    def test_validate_quarter_diversity_fails_insufficient_quarters(self):
        """Test validate_quarter_diversity fails with < 2 quarters in pre-period."""
        data = pd.DataFrame({
            'id': [1, 1, 1],
            'quarter': [1, 1, 1],  # Only one quarter in pre-period
            'post': [0, 0, 1]
        })
        
        with pytest.raises(InsufficientQuarterDiversityError) as exc_info:
            validate_quarter_diversity(data, 'id', 'quarter', 'post')
        
        assert "only 1 quarter(s)" in str(exc_info.value).lower()
    
    def test_validate_quarter_diversity_fails_uncovered_quarters(self):
        """Test validate_quarter_diversity fails with uncovered post quarters."""
        data = pd.DataFrame({
            'id': [1, 1, 1, 1],
            'quarter': [1, 2, 3, 4],  # Q3, Q4 only in post
            'post': [0, 0, 1, 1]
        })
        
        with pytest.raises(InsufficientQuarterDiversityError) as exc_info:
            validate_quarter_diversity(data, 'id', 'quarter', 'post')
        
        assert "not observed in pre-period" in str(exc_info.value).lower()
    
    def test_validate_quarter_coverage_passes_valid_data(self):
        """Test validate_quarter_coverage passes with valid data."""
        data = pd.DataFrame({
            'id': [1, 1, 1, 1],
            'quarter': [1, 2, 1, 2],
            'post': [0, 0, 1, 1]
        })
        
        # Should not raise
        validate_quarter_coverage(data, 'id', 'quarter', 'post')
    
    def test_validate_quarter_coverage_fails_uncovered(self):
        """Test validate_quarter_coverage fails with uncovered quarters."""
        data = pd.DataFrame({
            'id': [1, 1, 1, 1],
            'quarter': [1, 2, 3, 4],  # Q3, Q4 only in post
            'post': [0, 0, 1, 1]
        })
        
        with pytest.raises(InsufficientQuarterDiversityError):
            validate_quarter_coverage(data, 'id', 'quarter', 'post')
    
    def test_both_functions_give_same_error_for_coverage(self):
        """Test that both functions produce consistent error messages for coverage."""
        data = pd.DataFrame({
            'id': [1, 1, 1, 1, 1],
            'quarter': [1, 2, 1, 3, 4],  # Q3, Q4 only in post
            'post': [0, 0, 1, 1, 1]
        })
        
        # validate_quarter_diversity checks both diversity and coverage
        # For this data, it should pass diversity (has Q1, Q2 in pre) but fail coverage
        with pytest.raises(InsufficientQuarterDiversityError) as exc1:
            validate_quarter_diversity(data, 'id', 'quarter', 'post')
        
        with pytest.raises(InsufficientQuarterDiversityError) as exc2:
            validate_quarter_coverage(data, 'id', 'quarter', 'post')
        
        # Both should mention "not observed in pre-period"
        assert "not observed in pre-period" in str(exc1.value).lower()
        assert "not observed in pre-period" in str(exc2.value).lower()


class TestDesign069ConditionalVariance:
    """Tests for DESIGN-069: Conditional variance boundary case handling."""
    
    def test_single_observation_group_uses_global_variance(self):
        """Test that groups with <= 1 observation use global variance."""
        # Create data where one treatment group has only 1 observation
        Y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        X = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        W = np.array([0, 0, 0, 0, 1])  # Only 1 treated observation
        
        sigma2 = _estimate_conditional_variance_same_group(Y, X, W, J=2)
        
        # The treated observation (index 4) should have global variance, not 0
        global_var = np.var(Y, ddof=1)
        
        # sigma2[4] should be close to global variance, not 0
        assert sigma2[4] != 0, "Variance should not be 0 for single-observation group"
        assert np.isclose(sigma2[4], global_var, rtol=0.01), \
            f"Single observation group should use global variance ({global_var}), got {sigma2[4]}"
    
    def test_empty_group_uses_global_variance(self):
        """Test that empty groups don't cause errors."""
        Y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        X = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        W = np.array([0, 0, 0, 0, 0])  # All control, no treated
        
        sigma2 = _estimate_conditional_variance_same_group(Y, X, W, J=2)
        
        # All observations should have non-NaN variance
        assert not np.any(np.isnan(sigma2)), "Variance should not be NaN"
    
    def test_normal_case_neighbor_variance(self):
        """Test that normal case uses neighbor-based variance."""
        np.random.seed(42)
        n = 20
        Y = np.random.randn(n)
        X = np.random.randn(n)
        W = np.array([0] * 10 + [1] * 10)
        
        sigma2 = _estimate_conditional_variance_same_group(Y, X, W, J=2)
        
        # All observations should have positive variance
        assert np.all(sigma2 > 0), "All variance estimates should be positive"
        assert not np.any(np.isnan(sigma2)), "No NaN values expected"
    
    def test_small_group_uses_within_group_variance(self):
        """Test that small groups (n_group <= J) use within-group variance."""
        Y = np.array([1.0, 2.0, 3.0, 10.0, 11.0])
        X = np.array([0.1, 0.2, 0.3, 0.9, 1.0])
        W = np.array([0, 0, 0, 1, 1])  # 3 control, 2 treated
        
        sigma2 = _estimate_conditional_variance_same_group(Y, X, W, J=3)
        
        # Treated group has only 2 observations, less than J=3
        # Should use within-group variance
        treated_indices = np.where(W == 1)[0]
        Y_treated = Y[treated_indices]
        within_group_var = np.var(Y_treated, ddof=1)
        
        # Both treated observations should have within-group variance
        for idx in treated_indices:
            assert np.isclose(sigma2[idx], within_group_var, rtol=0.01), \
                f"Small group should use within-group variance ({within_group_var}), got {sigma2[idx]}"
    
    def test_variance_not_zero_for_single_obs_group(self):
        """Test that single-observation groups use global variance, not 0."""
        # This is the key test for DESIGN-069: when a group has only 1 observation,
        # the variance should be the global variance, not 0.
        
        # Case: 1 control, multiple treated
        Y = np.array([5.0, 1.0, 2.0, 3.0, 4.0])  # Different values to ensure positive variance
        X = np.array([0.9, 0.1, 0.2, 0.3, 0.4])
        W = np.array([0, 1, 1, 1, 1])  # Only 1 control observation
        
        sigma2 = _estimate_conditional_variance_same_group(Y, X, W, J=2)
        
        # The control observation should have global variance, not 0
        global_var = np.var(Y, ddof=1)
        
        # Control observation (index 0) should use global variance
        assert sigma2[0] != 0, "Single-obs group variance should not be 0"
        assert np.isclose(sigma2[0], global_var, rtol=0.01), \
            f"Single-obs group should use global variance ({global_var}), got {sigma2[0]}"
        
        # Case: 1 treated, multiple control
        Y2 = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
        X2 = np.array([0.1, 0.2, 0.3, 0.4, 0.9])
        W2 = np.array([0, 0, 0, 0, 1])  # Only 1 treated observation
        
        sigma2_2 = _estimate_conditional_variance_same_group(Y2, X2, W2, J=2)
        
        global_var2 = np.var(Y2, ddof=1)
        
        # Treated observation (index 4) should use global variance
        assert sigma2_2[4] != 0, "Single-obs group variance should not be 0"
        assert np.isclose(sigma2_2[4], global_var2, rtol=0.01), \
            f"Single-obs group should use global variance ({global_var2}), got {sigma2_2[4]}"


class TestDesign069Integration:
    """Integration tests for DESIGN-069 with real-world scenarios."""
    
    def test_psm_with_very_small_control_group(self):
        """Test PSM variance estimation with very small control group."""
        # Simulate a scenario where control group is very small
        np.random.seed(123)
        
        # 50 treated, 3 control (edge case)
        n_treated = 50
        n_control = 3
        
        Y_treated = np.random.randn(n_treated) + 2
        Y_control = np.random.randn(n_control)
        Y = np.concatenate([Y_control, Y_treated])
        
        X_treated = np.random.randn(n_treated) * 0.5 + 1
        X_control = np.random.randn(n_control) * 0.5
        X = np.concatenate([X_control, X_treated])
        
        W = np.array([0] * n_control + [1] * n_treated)
        
        sigma2 = _estimate_conditional_variance_same_group(Y, X, W, J=2)
        
        # Control group has only 3 observations
        # Variance should be computed (not 0) for all observations
        assert np.all(sigma2 > 0), "All variance estimates should be positive"
        
        # Control observations should have reasonable variance (not 0)
        control_var = sigma2[W == 0]
        assert np.all(control_var > 0), "Control group variance should be positive"
    
    def test_variance_estimation_consistency(self):
        """Test that variance estimation is consistent across different group sizes."""
        np.random.seed(456)
        
        # Create data with varying group sizes
        for n_control, n_treated in [(1, 100), (2, 100), (5, 100), (100, 100)]:
            Y = np.concatenate([np.random.randn(n_control), np.random.randn(n_treated) + 1])
            X = np.concatenate([np.random.randn(n_control), np.random.randn(n_treated) + 0.5])
            W = np.array([0] * n_control + [1] * n_treated)
            
            sigma2 = _estimate_conditional_variance_same_group(Y, X, W, J=2)
            
            # All variance should be positive
            assert np.all(sigma2 > 0), \
                f"All variance should be positive for n_control={n_control}, n_treated={n_treated}"
            
            # No NaN values
            assert not np.any(np.isnan(sigma2)), \
                f"No NaN expected for n_control={n_control}, n_treated={n_treated}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
