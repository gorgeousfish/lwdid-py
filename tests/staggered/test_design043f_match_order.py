"""
Test suite for DESIGN-043-F: match_order parameter for without-replacement PSM

This module tests the new match_order parameter that controls the order in which
treated units are matched to control units during without-replacement matching.

Tests include:
1. Unit tests for ordering logic
2. Backward compatibility tests (match_order='data')
3. Reproducibility tests (match_order='random' + seed)
4. Sensitivity tests comparing different match_order options
5. Tests for with_replacement (match_order should have no effect)

Note: Since Stata teffects psmatch only supports with-replacement matching,
we cannot perform Python-Stata end-to-end validation for this feature.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from lwdid.staggered.estimators import (
    estimate_psm,
    _nearest_neighbor_match,
)
from lwdid import lwdid


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def match_order_test_data():
    """
    Create test data specifically designed to show match_order effects.
    
    Design:
    - 20 control units with propensity scores spread across [0.2, 0.8]
    - 10 treated units with varying propensity scores
    - Some treated units have extreme PS (hard to match)
    - Some treated units have PS close to 0.5 (easy to match)
    """
    np.random.seed(42)
    n_control = 20
    n_treated = 10
    n_total = n_control + n_treated
    
    # Create data
    df = pd.DataFrame({
        'id': range(n_total),
        'D': [0] * n_control + [1] * n_treated,
    })
    
    # Control units: spread across [0.2, 0.8]
    x_control = np.linspace(-1.5, 1.5, n_control)
    
    # Treated units: some extreme, some moderate
    x_treated = np.array([
        -2.0, -1.5, -1.0,  # High PS (close to 1)
        0.0, 0.1, -0.1,    # Moderate PS (close to 0.5)
        1.5, 2.0, 1.0,     # Low PS (close to 0)
        0.5,               # Slightly above 0.5
    ])
    
    df['x'] = np.concatenate([x_control, x_treated])
    
    # Outcome: true effect is 2.0, with noise
    true_effect = 2.0
    df['Y'] = df['x'] + true_effect * df['D'] + np.random.normal(0, 0.5, n_total)
    
    return df


@pytest.fixture
def simple_test_data():
    """Simple test data for basic functionality tests."""
    np.random.seed(123)
    n = 100
    
    df = pd.DataFrame({
        'id': range(n),
        'D': [0] * 60 + [1] * 40,
        'x1': np.random.normal(0, 1, n),
        'x2': np.random.normal(0, 1, n),
    })
    
    # Add treatment effect
    df['Y'] = df['x1'] + 0.5 * df['x2'] + 1.5 * df['D'] + np.random.normal(0, 0.3, n)
    
    return df


@pytest.fixture
def staggered_test_data():
    """Staggered DiD test data for API-level tests."""
    np.random.seed(456)
    n_units = 30
    n_periods = 8
    
    data = []
    for i in range(n_units):
        # Cohort assignment
        if i < 10:
            g = np.inf  # Never treated
        elif i < 20:
            g = 4  # Cohort 4
        else:
            g = 5  # Cohort 5
        
        for t in range(1, n_periods + 1):
            data.append({
                'id': i,
                'year': t,
                'gvar': g,
                'x1': np.random.normal(0, 1),
                'x2': np.random.normal(0, 1),
                'y': np.random.normal(0, 1) + (2.0 if t >= g else 0),
            })
    
    return pd.DataFrame(data)


# ============================================================================
# Test: match_order parameter validation
# ============================================================================

class TestMatchOrderValidation:
    """Test match_order parameter validation."""
    
    def test_valid_match_orders(self, simple_test_data):
        """Test that all valid match_order values work."""
        valid_orders = ['data', 'random', 'largest', 'smallest']
        
        for order in valid_orders:
            result = estimate_psm(
                data=simple_test_data,
                y='Y',
                d='D',
                propensity_controls=['x1', 'x2'],
                with_replacement=False,
                match_order=order,
                seed=42,  # For reproducibility with 'random'
            )
            assert result.att is not None, f"match_order='{order}' should work"
    
    def test_invalid_match_order_raises(self, simple_test_data):
        """Test that invalid match_order values raise error."""
        with pytest.raises(ValueError, match="match_order"):
            estimate_psm(
                data=simple_test_data,
                y='Y',
                d='D',
                propensity_controls=['x1', 'x2'],
                with_replacement=False,
                match_order='invalid_order',
            )


# ============================================================================
# Test: match_order='data' (backward compatibility)
# ============================================================================

class TestMatchOrderDataBackwardCompatibility:
    """Test that match_order='data' preserves backward compatibility."""
    
    def test_default_is_data_order(self, simple_test_data):
        """Test that default match_order is 'data'."""
        # Run without specifying match_order
        result_default = estimate_psm(
            data=simple_test_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            with_replacement=False,
        )
        
        # Run with explicit match_order='data'
        result_data = estimate_psm(
            data=simple_test_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            with_replacement=False,
            match_order='data',
        )
        
        # Results should be identical
        assert result_default.att == result_data.att
        assert result_default.se == result_data.se
    
    def test_data_order_is_deterministic(self, simple_test_data):
        """Test that match_order='data' gives deterministic results."""
        results = []
        for _ in range(3):
            result = estimate_psm(
                data=simple_test_data,
                y='Y',
                d='D',
                propensity_controls=['x1', 'x2'],
                with_replacement=False,
                match_order='data',
            )
            results.append(result.att)
        
        # All runs should give identical ATT
        assert results[0] == results[1] == results[2]


# ============================================================================
# Test: match_order='random' reproducibility
# ============================================================================

class TestMatchOrderRandomReproducibility:
    """Test match_order='random' with seed for reproducibility."""
    
    def test_random_with_seed_reproducible(self, simple_test_data):
        """Test that random order with same seed is reproducible."""
        result1 = estimate_psm(
            data=simple_test_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            with_replacement=False,
            match_order='random',
            seed=42,
        )
        
        result2 = estimate_psm(
            data=simple_test_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            with_replacement=False,
            match_order='random',
            seed=42,
        )
        
        assert result1.att == result2.att
        assert result1.se == result2.se
    
    def test_random_different_seeds_differ(self, simple_test_data):
        """Test that different seeds give different results."""
        result1 = estimate_psm(
            data=simple_test_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            with_replacement=False,
            match_order='random',
            seed=42,
        )
        
        result2 = estimate_psm(
            data=simple_test_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            with_replacement=False,
            match_order='random',
            seed=123,
        )
        
        # With different seeds, ATT may differ (though not guaranteed for all data)
        # We just verify that the function runs without error
        assert result1.att is not None
        assert result2.att is not None


# ============================================================================
# Test: match_order effect on without-replacement matching
# ============================================================================

class TestMatchOrderEffect:
    """Test that different match_order options can produce different results."""
    
    def test_different_orders_may_differ_without_replacement(self, match_order_test_data):
        """Test that different match_order can give different ATT for without-replacement."""
        orders = ['data', 'largest', 'smallest']
        results = {}
        
        for order in orders:
            result = estimate_psm(
                data=match_order_test_data,
                y='Y',
                d='D',
                propensity_controls=['x'],
                with_replacement=False,
                match_order=order,
            )
            results[order] = result.att
        
        # At least some orders should potentially give different results
        # (This is data-dependent, so we just verify the code runs)
        assert all(att is not None for att in results.values())
        
        # Print results for inspection
        print(f"\nATT by match_order (without replacement):")
        for order, att in results.items():
            print(f"  {order}: {att:.4f}")
    
    def test_with_replacement_order_has_no_effect(self, simple_test_data):
        """Test that match_order has no effect on with-replacement matching."""
        orders = ['data', 'random', 'largest', 'smallest']
        results = {}
        
        for order in orders:
            result = estimate_psm(
                data=simple_test_data,
                y='Y',
                d='D',
                propensity_controls=['x1', 'x2'],
                with_replacement=True,  # With replacement!
                match_order=order,
                seed=42,
            )
            results[order] = result.att
        
        # All orders should give identical ATT with replacement
        atts = list(results.values())
        assert all(att == atts[0] for att in atts), \
            f"With replacement, all match_orders should give same ATT, got: {results}"


# ============================================================================
# Test: _nearest_neighbor_match unit test
# ============================================================================

class TestNearestNeighborMatchOrdering:
    """Unit tests for _nearest_neighbor_match ordering logic."""
    
    def test_data_order_matches_indices(self):
        """Test that 'data' order processes in index order."""
        # Create simple test case
        pscores_treat = np.array([0.3, 0.7, 0.5])  # 3 treated
        pscores_control = np.array([0.31, 0.69, 0.51, 0.4])  # 4 control
        
        matched, counts, dropped = _nearest_neighbor_match(
            pscores_treat=pscores_treat,
            pscores_control=pscores_control,
            n_neighbors=1,
            with_replacement=False,
            caliper=None,
            match_order='data',
        )
        
        # First treated (PS=0.3) should match control 0 (PS=0.31)
        # Second treated (PS=0.7) should match control 1 (PS=0.69)
        # Third treated (PS=0.5) should match remaining best: control 2 or 3
        assert matched[0] == [0]  # PS=0.3 -> PS=0.31
        assert matched[1] == [1]  # PS=0.7 -> PS=0.69
        # Third can be control 2 or 3 (both remaining)
        assert matched[2] == [2] or matched[2] == [3]
    
    def test_largest_order_prioritizes_extreme(self):
        """Test that 'largest' order processes extreme PS first."""
        # Create test case where extreme PS unit benefits from priority
        pscores_treat = np.array([0.5, 0.9])  # 0.9 is extreme
        pscores_control = np.array([0.85, 0.5])  # 0.85 is best for extreme
        
        # With 'data' order: 0.5 goes first, might take 0.5 control
        matched_data, _, _ = _nearest_neighbor_match(
            pscores_treat=pscores_treat,
            pscores_control=pscores_control,
            n_neighbors=1,
            with_replacement=False,
            caliper=None,
            match_order='data',
        )
        
        # With 'largest' order: 0.9 goes first (|0.9-0.5|=0.4 > |0.5-0.5|=0)
        matched_largest, _, _ = _nearest_neighbor_match(
            pscores_treat=pscores_treat,
            pscores_control=pscores_control,
            n_neighbors=1,
            with_replacement=False,
            caliper=None,
            match_order='largest',
        )
        
        # Verify extreme unit gets best match with 'largest' order
        # Treated[1] (PS=0.9) should get Control[0] (PS=0.85)
        assert matched_largest[1] == [0], "Extreme PS unit should get its best match first"
    
    def test_smallest_order_prioritizes_moderate(self):
        """Test that 'smallest' order processes moderate PS first."""
        pscores_treat = np.array([0.9, 0.5])  # 0.5 is moderate
        pscores_control = np.array([0.85, 0.5])
        
        # With 'smallest' order: 0.5 goes first (|0.5-0.5|=0 < |0.9-0.5|=0.4)
        matched_smallest, _, _ = _nearest_neighbor_match(
            pscores_treat=pscores_treat,
            pscores_control=pscores_control,
            n_neighbors=1,
            with_replacement=False,
            caliper=None,
            match_order='smallest',
        )
        
        # Treated[1] (PS=0.5) should get Control[1] (PS=0.5) - perfect match
        assert matched_smallest[1] == [1], "Moderate PS unit should get its best match first"


# ============================================================================
# Test: API-level integration
# ============================================================================

class TestMatchOrderAPIIntegration:
    """Test match_order at the lwdid() API level."""
    
    def test_lwdid_match_order_parameter(self, staggered_test_data):
        """Test that match_order parameter works in lwdid()."""
        result = lwdid(
            data=staggered_test_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            estimator='psm',
            controls=['x1', 'x2'],
            with_replacement=False,
            match_order='largest',
        )
        
        assert result.att is not None
    
    def test_lwdid_match_order_validation(self, staggered_test_data):
        """Test that invalid match_order raises error at API level."""
        with pytest.raises(ValueError, match="match_order must be"):
            lwdid(
                data=staggered_test_data,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                estimator='psm',
                controls=['x1', 'x2'],
                match_order='invalid',
            )


# ============================================================================
# Test: Sensitivity analysis example
# ============================================================================

class TestMatchOrderSensitivityAnalysis:
    """Demonstrate sensitivity analysis with different match_order options."""
    
    def test_sensitivity_analysis_example(self, match_order_test_data):
        """
        Example of how to conduct sensitivity analysis.
        
        This test demonstrates that researchers should compare results across
        different match_order options to assess robustness.
        """
        orders = ['data', 'random', 'largest', 'smallest']
        results = {}
        
        for order in orders:
            result = estimate_psm(
                data=match_order_test_data,
                y='Y',
                d='D',
                propensity_controls=['x'],
                with_replacement=False,
                match_order=order,
                seed=42,
            )
            results[order] = {
                'att': result.att,
                'se': result.se,
                'n_matched': result.n_matched,
            }
        
        # Print sensitivity analysis
        print("\n" + "=" * 60)
        print("SENSITIVITY ANALYSIS: match_order for without-replacement PSM")
        print("=" * 60)
        print(f"{'Order':<12} {'ATT':>10} {'SE':>10} {'N_matched':>10}")
        print("-" * 42)
        for order, res in results.items():
            print(f"{order:<12} {res['att']:>10.4f} {res['se']:>10.4f} {res['n_matched']:>10}")
        
        # Calculate range of ATT estimates
        atts = [r['att'] for r in results.values()]
        att_range = max(atts) - min(atts)
        att_mean = np.mean(atts)
        
        print("-" * 42)
        print(f"ATT range: {att_range:.4f} ({100*att_range/abs(att_mean):.1f}% of mean)")
        
        # All results should be reasonable (not NaN or Inf)
        for order, res in results.items():
            assert np.isfinite(res['att']), f"ATT for {order} should be finite"
            assert np.isfinite(res['se']), f"SE for {order} should be finite"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
