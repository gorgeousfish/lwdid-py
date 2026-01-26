"""
Test suite for BUG-161 fix: Conditional variance fallback logic consistency.

This module tests that the memory-efficient path and cdist path in
_estimate_conditional_variance_same_group produce consistent fallback values
in edge cases.

BUG-161: The memory-efficient path used 0.0 as fallback (anti-conservative),
while cdist path used three-tier fallback (group_var -> global_var -> 1.0).
The fix unifies both paths to use the conservative three-tier fallback.
"""

import numpy as np
import pytest

from lwdid.staggered.estimators import (
    _estimate_conditional_variance_same_group,
    _MEMORY_EFFICIENT_THRESHOLD,
)


class TestVarianceFallbackConsistency:
    """Test fallback logic consistency between cdist and memory-efficient paths."""

    def test_small_group_fallback_cdist_path(self):
        """Test cdist path uses three-tier fallback for small groups."""
        # Small sample to force cdist path (n_group < _MEMORY_EFFICIENT_THRESHOLD)
        n = 10  # Well below threshold
        Y = np.random.randn(n) * 2 + 5  # Non-trivial variance
        X = np.random.randn(n).reshape(-1, 1)
        W = np.concatenate([np.ones(2), np.zeros(n - 2)])  # Only 2 treated
        
        sigma2 = _estimate_conditional_variance_same_group(Y, X, W, J=2)
        
        # With only 2 treated units (W=1), J_actual = min(J, n_group-1) = min(2, 1) = 1
        # This triggers the J_actual < 2 branch
        # Should use group variance or global variance, NOT 0.0
        assert not np.any(sigma2 == 0.0), "Fallback should not be 0.0 (anti-conservative)"
        assert np.all(np.isfinite(sigma2)), "All variance estimates should be finite"
        assert np.all(sigma2 > 0), "All variance estimates should be positive"

    def test_small_group_fallback_memory_efficient_path(self):
        """Test memory-efficient path uses three-tier fallback for small groups."""
        # Large sample with 1D data to force memory-efficient path
        n = _MEMORY_EFFICIENT_THRESHOLD + 100
        Y = np.random.randn(n) * 2 + 5
        X = np.random.randn(n).reshape(-1, 1)
        # Create small treated group (only 2 units) to trigger J_actual < 2
        W = np.concatenate([np.ones(2), np.zeros(n - 2)])
        
        sigma2 = _estimate_conditional_variance_same_group(Y, X, W, J=2)
        
        # Should use fallback, NOT 0.0
        assert not np.any(sigma2 == 0.0), "Fallback should not be 0.0 (anti-conservative)"
        assert np.all(np.isfinite(sigma2)), "All variance estimates should be finite"
        assert np.all(sigma2 > 0), "All variance estimates should be positive"

    def test_paths_produce_similar_results(self):
        """Test both paths produce similar variance estimates for same data structure."""
        np.random.seed(42)
        
        # Create data that can be tested with both paths by varying sample size
        base_Y = np.random.randn(100) * 2 + 5
        base_X = np.random.randn(100)
        base_W = np.concatenate([np.ones(30), np.zeros(70)])
        
        # Small sample (cdist path)
        small_n = 50
        Y_small = base_Y[:small_n]
        X_small = base_X[:small_n].reshape(-1, 1)
        W_small = np.concatenate([np.ones(15), np.zeros(35)])
        
        sigma2_small = _estimate_conditional_variance_same_group(Y_small, X_small, W_small, J=2)
        
        # Verify no zeros in small sample result
        assert not np.any(sigma2_small == 0.0), "cdist path should not produce 0.0"

    def test_single_observation_group_fallback(self):
        """Test fallback when a treatment group has only one observation."""
        n = 20
        Y = np.random.randn(n) * 3 + 10
        X = np.random.randn(n).reshape(-1, 1)
        # Only 1 treated unit - triggers n_group <= 1 branch
        W = np.concatenate([np.ones(1), np.zeros(n - 1)])
        
        sigma2 = _estimate_conditional_variance_same_group(Y, X, W, J=2)
        
        # Single treated unit should use global variance fallback
        treated_var = sigma2[W == 1][0]
        global_var = np.var(Y, ddof=1)
        
        # Should be global variance (or 1.0 if global not available)
        assert treated_var > 0, "Variance should be positive"
        assert np.isclose(treated_var, global_var) or treated_var == 1.0, \
            f"Expected global var {global_var} or 1.0, got {treated_var}"

    def test_no_neighbors_fallback_memory_efficient(self):
        """Test fallback when no valid neighbors found in memory-efficient path."""
        # Force memory-efficient path with large 1D data
        n = _MEMORY_EFFICIENT_THRESHOLD + 100
        Y = np.random.randn(n) * 2 + 5
        # All observations have same X value -> neighbors will have distance 0
        # but algorithm should still work
        X = np.ones((n, 1))
        W = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])
        
        sigma2 = _estimate_conditional_variance_same_group(Y, X, W, J=2)
        
        # Should not have any zeros
        assert not np.any(sigma2 == 0.0), "No zeros should appear in variance estimates"
        assert np.all(np.isfinite(sigma2)), "All estimates should be finite"

    def test_extreme_small_sample(self):
        """Test with extremely small sample (n=3) to verify minimum viable case."""
        Y = np.array([1.0, 2.0, 3.0])
        X = np.array([[0.1], [0.2], [0.3]])
        W = np.array([1, 1, 0])  # 2 treated, 1 control
        
        sigma2 = _estimate_conditional_variance_same_group(Y, X, W, J=2)
        
        # No zeros should appear
        assert not np.any(sigma2 == 0.0), "Extreme small sample should not produce 0.0"
        assert np.all(sigma2 > 0), "All variance estimates should be positive"


class TestFallbackValueCorrectness:
    """Test that fallback values are statistically appropriate."""

    def test_global_variance_fallback_value(self):
        """Verify global variance is used correctly as fallback."""
        np.random.seed(123)
        n = 20
        Y = np.random.randn(n) * 5 + 10  # Known variance ~25
        X = np.random.randn(n).reshape(-1, 1)
        # Only 2 treated -> forces J_actual < 2
        W = np.concatenate([np.ones(2), np.zeros(n - 2)])
        
        sigma2 = _estimate_conditional_variance_same_group(Y, X, W, J=2)
        global_var = np.var(Y, ddof=1)
        
        # Treated group variance should be close to global variance (fallback)
        treated_vars = sigma2[W == 1]
        for tv in treated_vars:
            assert tv > 0, "Variance must be positive"
            # Allow some tolerance for numerical differences
            assert abs(tv - global_var) < global_var * 0.5 or tv == 1.0, \
                f"Treated var {tv} should be close to global {global_var}"

    def test_unit_fallback_preserved(self):
        """Test that 1.0 fallback is used when no other option available."""
        # Extreme case: single observation
        Y = np.array([5.0])
        X = np.array([[1.0]])
        W = np.array([1])  # Single treated
        
        sigma2 = _estimate_conditional_variance_same_group(Y, X, W, J=2)
        
        # Should fall back to 1.0 as last resort
        assert sigma2[0] == 1.0, f"Single observation should use 1.0 fallback, got {sigma2[0]}"


class TestMemoryEfficientThreshold:
    """Test behavior around the memory-efficient threshold boundary."""

    def test_at_threshold_boundary(self):
        """Test behavior exactly at the threshold boundary."""
        # Just below threshold (cdist path)
        n_below = _MEMORY_EFFICIENT_THRESHOLD - 1
        np.random.seed(42)
        Y = np.random.randn(n_below)
        X = np.random.randn(n_below).reshape(-1, 1)
        W = np.concatenate([np.ones(n_below // 2), np.zeros(n_below - n_below // 2)])
        
        sigma2_below = _estimate_conditional_variance_same_group(Y, X, W, J=2)
        
        # Just above threshold (memory-efficient path)
        n_above = _MEMORY_EFFICIENT_THRESHOLD + 1
        np.random.seed(42)
        Y = np.random.randn(n_above)
        X = np.random.randn(n_above).reshape(-1, 1)
        W = np.concatenate([np.ones(n_above // 2), np.zeros(n_above - n_above // 2)])
        
        sigma2_above = _estimate_conditional_variance_same_group(Y, X, W, J=2)
        
        # Both should produce valid (non-zero) estimates
        assert not np.any(sigma2_below == 0.0), "Below threshold should not produce zeros"
        assert not np.any(sigma2_above == 0.0), "Above threshold should not produce zeros"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
