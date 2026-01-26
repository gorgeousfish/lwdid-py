"""
Numerical validation tests for BUG-112 fix.

These tests verify that the match_rate boundary check fix does not affect
normal calculations by comparing results with known expected values.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(__file__).rsplit('/tests/', 1)[0] + '/src')

from lwdid.staggered.estimators import _compute_match_statistics, estimate_psm


class TestNumericalValidation:
    """Numerical validation tests for match_rate calculation."""
    
    def test_exact_computation_simple(self):
        """Test exact match_rate computation with known values."""
        # Case 1: All matched
        K_M = np.array([1, 1, 1, 1, 1])
        n_treated = 5
        n_dropped = 0
        
        match_rate, retention_rate, _, _ = _compute_match_statistics(K_M, n_treated, n_dropped)
        
        assert match_rate == 1.0, "All control units matched should give match_rate=1.0"
        assert retention_rate == 1.0, "No dropped units should give retention_rate=1.0"
        
    def test_exact_computation_partial(self):
        """Test exact match_rate computation with partial matching."""
        # 3 out of 5 control units matched, 2 out of 10 treated dropped
        K_M = np.array([2, 0, 1, 0, 3])  # 3 unique controls used
        n_treated = 10
        n_dropped = 2
        
        match_rate, retention_rate, _, _ = _compute_match_statistics(K_M, n_treated, n_dropped)
        
        expected_match_rate = 3 / 5  # 0.6
        expected_retention_rate = 8 / 10  # 0.8
        
        assert match_rate == pytest.approx(expected_match_rate, abs=1e-15)
        assert retention_rate == pytest.approx(expected_retention_rate, abs=1e-15)
        
    def test_precision_not_affected(self):
        """Test that precision is not affected by the boundary check."""
        # Test with values that should give exactly representable results
        K_M = np.array([1, 0, 1, 0])  # 2 out of 4 = 0.5
        n_treated = 8
        n_dropped = 2  # 6/8 = 0.75
        
        match_rate, retention_rate, _, _ = _compute_match_statistics(K_M, n_treated, n_dropped)
        
        # These values should be exactly representable in floating point
        assert match_rate == 0.5
        assert retention_rate == 0.75
        
    def test_no_rounding_error_introduced(self):
        """Test that boundary check doesn't introduce rounding errors."""
        # Test with values that might be affected by floating point precision
        K_M = np.array([1, 1, 1])  # 3 out of 3
        n_treated = 3
        n_dropped = 0
        
        for _ in range(1000):  # Run multiple times to check consistency
            match_rate, retention_rate, _, _ = _compute_match_statistics(K_M, n_treated, n_dropped)
            
            # Should be exactly 1.0, not 0.9999999999 or 1.0000000001
            assert match_rate == 1.0
            assert retention_rate == 1.0


class TestPSMEstimationNumerical:
    """Test PSM estimation numerical accuracy."""
    
    @pytest.fixture
    def psm_test_data(self):
        """Load PSM test data."""
        test_dir = Path(__file__).parent / 'data'
        data_path = test_dir / 'psm_test_n100.csv'
        
        if data_path.exists():
            return pd.read_csv(data_path)
        else:
            pytest.skip(f"Test data not found: {data_path}")
            
    def test_psm_estimation_match_counts(self, psm_test_data):
        """Test PSM estimation produces correct match counts."""
        data = psm_test_data
        
        result = estimate_psm(
            data=data,
            y='y',
            d='d',
            propensity_controls=['x1', 'x2'],
            n_neighbors=1,
            with_replacement=True,
            alpha=0.05,
            return_diagnostics=False,
        )
        
        # Verify match counts are consistent
        n_treated = int(data['d'].sum())
        n_control = int((data['d'] == 0).sum())
        
        assert result.n_treated == n_treated
        assert result.n_control == n_control
        
        # Match rate should be valid
        if hasattr(result, 'match_diagnostics') and result.match_diagnostics is not None:
            diag = result.match_diagnostics
            assert 0.0 <= diag.match_rate <= 1.0
            assert 0.0 <= diag.treatment_retention_rate <= 1.0
            
    def test_psm_estimation_with_caliper(self, psm_test_data):
        """Test PSM estimation with caliper produces valid match_rate."""
        data = psm_test_data
        
        # Use a strict caliper that will drop some units
        result = estimate_psm(
            data=data,
            y='y',
            d='d',
            propensity_controls=['x1', 'x2'],
            n_neighbors=1,
            caliper=0.1,  # Strict caliper
            with_replacement=True,
            alpha=0.05,
            return_diagnostics=True,
        )
        
        # n_dropped should be >= 0 and <= n_treated
        assert result.n_dropped >= 0
        assert result.n_dropped <= result.n_treated
        
        # Verify diagnostics if available
        if hasattr(result, 'match_diagnostics') and result.match_diagnostics is not None:
            diag = result.match_diagnostics
            assert 0.0 <= diag.match_rate <= 1.0
            assert 0.0 <= diag.treatment_retention_rate <= 1.0
            
            # treatment_retention_rate should match our formula
            expected_retention = (result.n_treated - result.n_dropped) / result.n_treated
            assert diag.treatment_retention_rate == pytest.approx(expected_retention, rel=1e-10)


class TestMathematicalProperties:
    """Test mathematical properties of match_rate calculation."""
    
    def test_monotonicity_wrt_dropped(self):
        """Test that retention_rate decreases as n_dropped increases."""
        K_M = np.array([1, 1, 1, 1])
        n_treated = 10
        
        prev_retention = 1.1  # Start above max
        for n_dropped in range(n_treated + 1):
            _, retention_rate, _, _ = _compute_match_statistics(K_M, n_treated, n_dropped)
            
            assert retention_rate <= prev_retention, \
                f"retention_rate should decrease as n_dropped increases: {retention_rate} > {prev_retention}"
            assert 0.0 <= retention_rate <= 1.0
            
            prev_retention = retention_rate
            
    def test_symmetry_property(self):
        """Test that match_rate is symmetric in K_M arrangement."""
        # Different arrangements of same values should give same result
        K_M1 = np.array([1, 0, 2, 0, 1])
        K_M2 = np.array([0, 1, 0, 2, 1])
        K_M3 = np.array([2, 1, 1, 0, 0])
        
        n_treated = 4
        n_dropped = 0
        
        mr1, rr1, ar1, max1 = _compute_match_statistics(K_M1, n_treated, n_dropped)
        mr2, rr2, ar2, max2 = _compute_match_statistics(K_M2, n_treated, n_dropped)
        mr3, rr3, ar3, max3 = _compute_match_statistics(K_M3, n_treated, n_dropped)
        
        # Same number of matched controls (3), same match counts
        assert mr1 == mr2 == mr3
        assert rr1 == rr2 == rr3
        assert ar1 == ar2 == ar3
        assert max1 == max2 == max3


class TestEdgeCasesNumerical:
    """Test numerical edge cases."""
    
    def test_very_small_sample(self):
        """Test with very small sample sizes."""
        K_M = np.array([1])
        n_treated = 1
        n_dropped = 0
        
        match_rate, retention_rate, _, _ = _compute_match_statistics(K_M, n_treated, n_dropped)
        
        assert match_rate == 1.0
        assert retention_rate == 1.0
        
    def test_very_large_sample(self):
        """Test with large sample sizes."""
        np.random.seed(42)
        K_M = np.random.randint(0, 10, size=10000)
        n_treated = 5000
        n_dropped = 100
        
        match_rate, retention_rate, _, _ = _compute_match_statistics(K_M, n_treated, n_dropped)
        
        assert 0.0 <= match_rate <= 1.0
        assert 0.0 <= retention_rate <= 1.0
        
        # Verify exact values
        expected_match_rate = np.sum(K_M > 0) / len(K_M)
        expected_retention = (5000 - 100) / 5000
        
        assert match_rate == pytest.approx(expected_match_rate, rel=1e-10)
        assert retention_rate == pytest.approx(expected_retention, rel=1e-10)
        
    def test_floating_point_edge(self):
        """Test edge cases that might cause floating point issues."""
        # Case where numerator might slightly exceed denominator due to FP errors
        K_M = np.ones(100, dtype=np.int64)
        n_treated = 100
        n_dropped = 0
        
        match_rate, retention_rate, _, _ = _compute_match_statistics(K_M, n_treated, n_dropped)
        
        # Should be exactly 1.0, not something like 1.0000000000000002
        assert match_rate == 1.0
        assert retention_rate == 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
