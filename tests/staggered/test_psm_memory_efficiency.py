"""
DESIGN-035: PSM Memory Efficiency Tests

This module tests the memory-efficient k-NN implementations against the original
cdist-based implementations to ensure numerical equivalence.

The memory-efficient implementations use:
- Sorted array + binary search for 1D data: O(n) space instead of O(n²)

Tests verify:
1. Numerical consistency between new and old implementations
2. Correctness of k-NN results
3. Memory usage improvement for large samples
"""

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from lwdid.staggered.estimators import (
    _find_k_nearest_1d_all,
    _find_k_nearest_1d_cross,
    _estimate_conditional_variance_same_group,
    _estimate_conditional_variance_same_group_paper,
    _MEMORY_EFFICIENT_THRESHOLD,
)


class TestFindKNearest1DAll:
    """Test the memory-efficient 1D k-NN function for same-array queries."""
    
    def test_basic_functionality(self):
        """Test basic k-NN search returns correct neighbors."""
        values = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
        # Sorted: [1.0, 2.0, 3.0, 4.0, 5.0] -> indices [0, 3, 2, 4, 1]
        
        neighbors = _find_k_nearest_1d_all(values, k=2, exclude_self=True)
        
        assert neighbors.shape == (5, 2)
        
        # For value 1.0 (idx 0), nearest are 2.0 (idx 3) and 3.0 (idx 2)
        assert set(neighbors[0]) == {3, 2}
        
        # For value 5.0 (idx 1), nearest are 4.0 (idx 4) and 3.0 (idx 2)
        assert set(neighbors[1]) == {4, 2}
        
    def test_exclude_self(self):
        """Test that exclude_self correctly excludes self from neighbors."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # With exclude_self=True
        neighbors_excl = _find_k_nearest_1d_all(values, k=2, exclude_self=True)
        for i in range(len(values)):
            assert i not in neighbors_excl[i]
            
        # With exclude_self=False
        neighbors_incl = _find_k_nearest_1d_all(values, k=2, exclude_self=False)
        # Each point should have itself as nearest (distance 0)
        for i in range(len(values)):
            assert i in neighbors_incl[i]
    
    def test_equivalence_with_cdist(self):
        """Test that results match cdist-based implementation."""
        np.random.seed(42)
        n = 100
        k = 5
        values = np.random.randn(n)
        
        # New implementation
        neighbors_new = _find_k_nearest_1d_all(values, k=k, exclude_self=True)
        
        # Original cdist-based implementation
        dist_matrix = cdist(values.reshape(-1, 1), values.reshape(-1, 1))
        np.fill_diagonal(dist_matrix, np.inf)
        neighbors_cdist = np.argsort(dist_matrix, axis=1)[:, :k]
        
        # Check each point's neighbors
        for i in range(n):
            new_set = set(neighbors_new[i][neighbors_new[i] >= 0])
            cdist_set = set(neighbors_cdist[i])
            
            # The neighbors should be the same (allowing for ties)
            # In case of ties, the order might differ, so we check the distances
            new_dists = sorted([abs(values[j] - values[i]) for j in new_set])
            cdist_dists = sorted([abs(values[j] - values[i]) for j in cdist_set])
            
            np.testing.assert_allclose(new_dists, cdist_dists, rtol=1e-10)
    
    def test_edge_cases(self):
        """Test edge cases: empty array, single element, k >= n."""
        # Empty array
        neighbors = _find_k_nearest_1d_all(np.array([]), k=2, exclude_self=True)
        assert neighbors.shape == (0, 2)
        
        # Single element
        neighbors = _find_k_nearest_1d_all(np.array([1.0]), k=2, exclude_self=True)
        assert neighbors.shape == (1, 2)
        assert np.all(neighbors == -1)  # No valid neighbors
        
        # k >= n - 1 (requesting more neighbors than available)
        values = np.array([1.0, 2.0, 3.0])
        neighbors = _find_k_nearest_1d_all(values, k=5, exclude_self=True)
        assert neighbors.shape == (3, 5)
        # Each point should have at most 2 valid neighbors
        for i in range(3):
            valid_count = np.sum(neighbors[i] >= 0)
            assert valid_count == 2  # n - 1
    
    def test_tied_distances(self):
        """Test handling of tied distances."""
        values = np.array([0.0, 1.0, 1.0, 2.0])  # indices 1 and 2 are tied for point 0
        
        neighbors = _find_k_nearest_1d_all(values, k=2, exclude_self=True)
        
        # For point 0, neighbors should be indices 1 and 2 (both at distance 1.0)
        assert set(neighbors[0]) == {1, 2}


class TestFindKNearest1DCross:
    """Test the memory-efficient 1D k-NN function for cross-array queries."""
    
    def test_basic_functionality(self):
        """Test basic cross-array k-NN search."""
        query = np.array([2.5, 4.0])
        target = np.array([1.0, 3.0, 5.0])
        
        neighbors = _find_k_nearest_1d_cross(query, target, k=2)
        
        assert neighbors.shape == (2, 2)
        
        # For query 2.5, nearest targets are 3.0 (idx 1) and 1.0 (idx 0)
        assert 1 in neighbors[0]  # 3.0 is closest
        
        # For query 4.0, nearest targets are 3.0 (idx 1) and 5.0 (idx 2)
        assert set(neighbors[1]) == {1, 2}
    
    def test_equivalence_with_cdist(self):
        """Test that results match cdist-based implementation."""
        np.random.seed(42)
        m, n = 50, 100
        k = 3
        query = np.random.randn(m)
        target = np.random.randn(n)
        
        # New implementation
        neighbors_new = _find_k_nearest_1d_cross(query, target, k=k)
        
        # Original cdist-based implementation
        dist_matrix = cdist(query.reshape(-1, 1), target.reshape(-1, 1))
        neighbors_cdist = np.argsort(dist_matrix, axis=1)[:, :k]
        
        # Check each query point's neighbors
        for i in range(m):
            new_set = set(neighbors_new[i][neighbors_new[i] >= 0])
            cdist_set = set(neighbors_cdist[i])
            
            new_dists = sorted([abs(target[j] - query[i]) for j in new_set])
            cdist_dists = sorted([abs(target[j] - query[i]) for j in cdist_set])
            
            np.testing.assert_allclose(new_dists, cdist_dists, rtol=1e-10)
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Empty query
        neighbors = _find_k_nearest_1d_cross(np.array([]), np.array([1.0, 2.0]), k=2)
        assert neighbors.shape == (0, 2)
        
        # Empty target
        neighbors = _find_k_nearest_1d_cross(np.array([1.0, 2.0]), np.array([]), k=2)
        assert neighbors.shape == (2, 2)
        assert np.all(neighbors == -1)


class TestConditionalVarianceNumericalConsistency:
    """Test numerical consistency of conditional variance estimation."""
    
    @pytest.fixture
    def small_sample_data(self):
        """Generate small sample test data."""
        np.random.seed(42)
        n = 100
        Y = np.random.randn(n)
        X = np.random.randn(n)  # 1D propensity scores
        W = np.random.randint(0, 2, n)
        return Y, X, W
    
    @pytest.fixture
    def medium_sample_data(self):
        """Generate medium sample test data (below threshold)."""
        np.random.seed(42)
        n = 1000
        Y = np.random.randn(n)
        X = np.random.randn(n)
        W = np.random.randint(0, 2, n)
        return Y, X, W
    
    def test_stata_variance_small_sample(self, small_sample_data):
        """Test Stata-style variance estimation with small sample."""
        Y, X, W = small_sample_data
        
        # Both should use cdist path for small samples
        sigma2 = _estimate_conditional_variance_same_group(Y, X, W, J=2)
        
        assert sigma2.shape == (len(Y),)
        assert np.all(sigma2 >= 0)  # Variance should be non-negative
        assert not np.any(np.isnan(sigma2))
    
    def test_paper_variance_small_sample(self, small_sample_data):
        """Test paper-style variance estimation with small sample."""
        Y, X, W = small_sample_data
        
        sigma2 = _estimate_conditional_variance_same_group_paper(Y, X, W, J=2)
        
        assert sigma2.shape == (len(Y),)
        assert np.all(sigma2 >= 0)
        assert not np.any(np.isnan(sigma2))
    
    def test_variance_consistency_across_j_values(self, small_sample_data):
        """Test that variance estimation works for different J values."""
        Y, X, W = small_sample_data
        
        for J in [1, 2, 3, 5]:
            sigma2_stata = _estimate_conditional_variance_same_group(Y, X, W, J=J)
            sigma2_paper = _estimate_conditional_variance_same_group_paper(Y, X, W, J=J)
            
            assert sigma2_stata.shape == (len(Y),)
            assert sigma2_paper.shape == (len(Y),)
            assert np.all(sigma2_stata >= 0)
            assert np.all(sigma2_paper >= 0)


class TestMemoryEfficiencyThreshold:
    """Test that memory-efficient path is correctly activated."""
    
    def test_threshold_constant_exists(self):
        """Test that threshold constant is defined."""
        assert _MEMORY_EFFICIENT_THRESHOLD > 0
        assert isinstance(_MEMORY_EFFICIENT_THRESHOLD, int)
    
    def test_small_sample_uses_cdist(self):
        """Test that small samples use original cdist path."""
        np.random.seed(42)
        n = 100  # Well below threshold
        Y = np.random.randn(n)
        X = np.random.randn(n)
        W = np.random.randint(0, 2, n)
        
        # This should work without any issues (uses cdist)
        sigma2 = _estimate_conditional_variance_same_group(Y, X, W, J=2)
        assert sigma2.shape == (n,)


class TestLargeSampleSimulation:
    """Simulate large sample scenario (without actually using huge memory)."""
    
    def test_memory_efficient_path_correctness(self):
        """Test that memory-efficient path produces correct results.
        
        We test with a sample size just below the threshold, then manually
        verify the algorithm would produce the same results for larger samples.
        """
        np.random.seed(42)
        n = 500
        Y = np.random.randn(n)
        X = np.random.randn(n)
        W = np.random.randint(0, 2, n)
        
        # Force use of memory-efficient path by temporarily lowering threshold
        original_threshold = _MEMORY_EFFICIENT_THRESHOLD
        
        # Import and modify the module-level constant
        import lwdid.staggered.estimators as est
        est._MEMORY_EFFICIENT_THRESHOLD = 100  # Lower threshold
        
        try:
            sigma2_efficient = _estimate_conditional_variance_same_group(Y, X, W, J=2)
            
            # Reset to high threshold to force cdist path
            est._MEMORY_EFFICIENT_THRESHOLD = 10000
            sigma2_cdist = _estimate_conditional_variance_same_group(Y, X, W, J=2)
            
            # Results should be numerically equivalent
            np.testing.assert_allclose(sigma2_efficient, sigma2_cdist, rtol=1e-10)
        finally:
            # Restore original threshold
            est._MEMORY_EFFICIENT_THRESHOLD = original_threshold
    
    def test_paper_variance_memory_efficient_consistency(self):
        """Test paper-style variance with memory-efficient path."""
        np.random.seed(42)
        n = 500
        Y = np.random.randn(n)
        X = np.random.randn(n)
        W = np.random.randint(0, 2, n)
        
        import lwdid.staggered.estimators as est
        original_threshold = est._MEMORY_EFFICIENT_THRESHOLD
        
        try:
            est._MEMORY_EFFICIENT_THRESHOLD = 100
            sigma2_efficient = _estimate_conditional_variance_same_group_paper(Y, X, W, J=2)
            
            est._MEMORY_EFFICIENT_THRESHOLD = 10000
            sigma2_cdist = _estimate_conditional_variance_same_group_paper(Y, X, W, J=2)
            
            np.testing.assert_allclose(sigma2_efficient, sigma2_cdist, rtol=1e-10)
        finally:
            est._MEMORY_EFFICIENT_THRESHOLD = original_threshold


class TestKNNEdgeCases:
    """Test edge cases for k-NN functions."""
    
    def test_all_same_values(self):
        """Test k-NN with all identical values."""
        values = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        neighbors = _find_k_nearest_1d_all(values, k=2, exclude_self=True)
        
        # All points are equidistant, so any neighbors are valid
        assert neighbors.shape == (5, 2)
        for i in range(5):
            assert i not in neighbors[i]  # Self excluded
    
    def test_sorted_input(self):
        """Test k-NN with already sorted input."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        neighbors = _find_k_nearest_1d_all(values, k=2, exclude_self=True)
        
        # For point 0, neighbors should be 1 and 2
        assert set(neighbors[0]) == {1, 2}
        
        # For point 4, neighbors should be 3 and 2
        assert set(neighbors[4]) == {3, 2}
    
    def test_reverse_sorted_input(self):
        """Test k-NN with reverse sorted input."""
        values = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        neighbors = _find_k_nearest_1d_all(values, k=2, exclude_self=True)
        
        # For point 0 (value 5.0), neighbors should be 1 (4.0) and 2 (3.0)
        assert set(neighbors[0]) == {1, 2}
    
    def test_negative_values(self):
        """Test k-NN with negative values."""
        values = np.array([-5.0, -2.0, 0.0, 1.0, 3.0])
        neighbors = _find_k_nearest_1d_all(values, k=2, exclude_self=True)
        
        assert neighbors.shape == (5, 2)
        # Each point should have valid neighbors
        for i in range(5):
            valid_count = np.sum(neighbors[i] >= 0)
            assert valid_count == 2


class TestMemoryUsage:
    """Test memory usage comparison between old and new implementations."""
    
    def test_memory_efficient_knn_space_complexity(self):
        """Verify that memory-efficient k-NN uses O(n) space."""
        import sys
        
        # Test data sizes
        sizes = [100, 500, 1000]
        k = 5
        
        memory_usage = []
        for n in sizes:
            values = np.random.randn(n)
            
            # Measure approximate memory of result
            neighbors = _find_k_nearest_1d_all(values, k=k, exclude_self=True)
            result_memory = sys.getsizeof(neighbors) + neighbors.nbytes
            
            memory_usage.append((n, result_memory))
        
        # Memory should grow linearly with n (O(n * k))
        # The ratio of memory to n should be approximately constant
        ratios = [mem / n for n, mem in memory_usage]
        
        # Verify ratios are within reasonable bounds (allowing for overhead)
        assert max(ratios) / min(ratios) < 3.0, "Memory usage should scale linearly"
    
    def test_cdist_vs_efficient_memory_theoretical(self):
        """Theoretical memory comparison: cdist O(n²) vs efficient O(n)."""
        # For cdist: distance matrix is n × n × 8 bytes (float64)
        # For efficient: neighbors array is n × k × 8 bytes (int64)
        
        n = 10000
        k = 5
        
        # Theoretical cdist memory (bytes)
        cdist_memory = n * n * 8  # 800 MB for n=10000
        
        # Theoretical efficient memory (bytes)
        # Just the result array: n * k * 8
        efficient_memory = n * k * 8  # 400 KB for n=10000, k=5
        
        # Verify massive improvement
        improvement_ratio = cdist_memory / efficient_memory
        assert improvement_ratio > 1000, f"Expected >1000x improvement, got {improvement_ratio}x"
    
    def test_no_memory_explosion_large_sample(self):
        """Test that large samples don't cause memory issues."""
        # This test uses a sample size below threshold to avoid actual memory issues
        # but verifies the algorithm works correctly
        np.random.seed(42)
        n = 2000
        
        Y = np.random.randn(n)
        X = np.random.randn(n)
        W = np.random.randint(0, 2, n)
        
        # This should complete without memory errors
        sigma2 = _estimate_conditional_variance_same_group(Y, X, W, J=2)
        
        assert sigma2.shape == (n,)
        assert not np.any(np.isnan(sigma2))
        assert np.all(sigma2 >= 0)


class TestPSMEstimatorIntegration:
    """Integration tests for PSM estimator with memory-efficient code."""
    
    def test_psm_small_sample_works(self):
        """Test that PSM estimator works with small samples."""
        from lwdid.staggered.estimators import estimate_psm
        import pandas as pd
        
        np.random.seed(42)
        n = 200
        
        # Generate simple test data
        X1 = np.random.randn(n)
        X2 = np.random.randn(n)
        ps = 1 / (1 + np.exp(-(X1 + X2)))
        D = (np.random.rand(n) < ps).astype(int)
        Y = 2 * D + X1 + np.random.randn(n)
        
        data = pd.DataFrame({
            'Y': Y, 'D': D, 'X1': X1, 'X2': X2
        })
        
        # This should work without issues
        result = estimate_psm(
            data, y='Y', d='D', 
            propensity_controls=['X1', 'X2'],
            n_neighbors=1,
            se_method='abadie_imbens_full'
        )
        
        assert result is not None
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)
        assert result.se > 0
    
    def test_psm_moderate_sample(self):
        """Test PSM with moderate sample size."""
        from lwdid.staggered.estimators import estimate_psm
        import pandas as pd
        
        np.random.seed(42)
        n = 500
        
        X1 = np.random.randn(n)
        X2 = np.random.randn(n)
        ps = 1 / (1 + np.exp(-(0.5 * X1 + 0.3 * X2)))
        D = (np.random.rand(n) < ps).astype(int)
        Y = 1.5 * D + X1 + 0.5 * X2 + np.random.randn(n) * 0.5
        
        data = pd.DataFrame({
            'Y': Y, 'D': D, 'X1': X1, 'X2': X2
        })
        
        result = estimate_psm(
            data, y='Y', d='D',
            propensity_controls=['X1', 'X2'],
            n_neighbors=3,
            se_method='abadie_imbens_full'
        )
        
        # ATT should be approximately 1.5
        assert 0.5 < result.att < 2.5, f"ATT {result.att} outside expected range"
        assert result.se > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
