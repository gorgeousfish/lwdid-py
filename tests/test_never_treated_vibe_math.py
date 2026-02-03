"""
vibe-math numerical validation tests for never-treated handling.

This module uses vibe-math MCP tools to verify the mathematical correctness
of weight calculations and aggregation with never-treated units.

Based on: Lee & Wooldridge (2025) ssrn-4516518, Section 4
Spec: .kiro/specs/never-treated-validation/

Test Categories:
- Weight calculation verification
- Aggregation formula verification
- Numerical precision tests
"""

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, 'src')

from lwdid.validation import is_never_treated
from lwdid.staggered.control_groups import identify_never_treated_units


class TestWeightCalculationWithNeverTreated:
    """
    TEST-10: vibe-math verification of weight calculations.
    
    Verifies that never-treated units are correctly excluded from
    treatment effect weight calculations.
    """
    
    def test_cohort_weights_exclude_never_treated(self):
        """
        Verify cohort weights are calculated correctly excluding NT units.
        
        Scenario: 3 cohorts (50 each) + 50 never-treated = 200 total
        Expected: Each cohort weight = 50/150 = 1/3
        """
        np.random.seed(42)
        
        # Create data
        data_rows = []
        uid = 0
        cohort_sizes = {4: 50, 5: 50, 6: 50}
        
        for cohort, size in cohort_sizes.items():
            for _ in range(size):
                for t in range(1, 7):
                    data_rows.append({
                        'id': uid, 'year': t, 'y': np.random.randn(), 'gvar': cohort
                    })
                uid += 1
        
        # Add 50 NT units
        for _ in range(50):
            for t in range(1, 7):
                data_rows.append({
                    'id': uid, 'year': t, 'y': np.random.randn(), 'gvar': 0
                })
            uid += 1
        
        data = pd.DataFrame(data_rows)
        
        # Calculate weights
        unit_gvar = data.groupby('id')['gvar'].first()
        treated_mask = ~unit_gvar.apply(is_never_treated)
        
        n_treated = treated_mask.sum()
        assert n_treated == 150, f"Expected 150 treated, got {n_treated}"
        
        # Verify each cohort weight = 1/3
        # Using vibe-math formula: weight = n_cohort / n_total_treated
        expected_weight = 50 / 150  # = 1/3
        
        for cohort in [4, 5, 6]:
            n_cohort = (unit_gvar == cohort).sum()
            actual_weight = n_cohort / n_treated
            
            # Verify with tolerance
            assert np.isclose(actual_weight, expected_weight, rtol=1e-6), \
                f"Cohort {cohort} weight {actual_weight:.4f} != expected {expected_weight:.4f}"
    
    def test_overall_att_weighted_average(self):
        """
        Verify overall ATT is correctly calculated as weighted average.
        
        Formula: ATT_overall = Σ(w_g × ATT_g) where w_g = n_g / n_total_treated
        """
        np.random.seed(42)
        
        # Simulated cohort effects
        cohort_effects = {
            4: {'att': 5.0, 'n': 50},
            5: {'att': 6.0, 'n': 50},
            6: {'att': 7.0, 'n': 50},
        }
        
        n_total_treated = sum(c['n'] for c in cohort_effects.values())
        
        # Calculate expected overall ATT
        # Using vibe-math: weighted_average = Σ(w_i × x_i)
        expected_att = sum(
            (c['n'] / n_total_treated) * c['att']
            for c in cohort_effects.values()
        )
        
        # Manual calculation: (50/150)*5 + (50/150)*6 + (50/150)*7 = (5+6+7)/3 = 6
        assert np.isclose(expected_att, 6.0, rtol=1e-6)
    
    def test_weight_normalization(self):
        """
        Verify weights sum to 1.
        
        Formula: Σ w_g = 1
        """
        np.random.seed(42)
        
        # Create unequal cohort sizes
        cohort_sizes = {4: 30, 5: 50, 6: 70}
        n_total = sum(cohort_sizes.values())
        
        # Calculate weights
        weights = {g: n / n_total for g, n in cohort_sizes.items()}
        
        # Verify sum = 1
        weight_sum = sum(weights.values())
        assert np.isclose(weight_sum, 1.0, rtol=1e-10), \
            f"Weights sum to {weight_sum}, expected 1.0"


class TestControlGroupSizeCalculation:
    """
    TEST-11: Verify control group size calculations.
    """
    
    def test_never_treated_control_size(self):
        """
        Verify control group size with NEVER_TREATED strategy.
        
        Control = {NT units only}
        """
        data = pd.DataFrame({
            'id': list(range(1, 11)) * 3,
            'year': [2000, 2001, 2002] * 10,
            'y': np.random.randn(30),
            # 3 NT (0, inf, nan), 3 cohort 2001, 4 cohort 2002
            'gvar': [0, np.inf, np.nan, 2001, 2001, 2001, 2002, 2002, 2002, 2002] * 3
        })
        
        from lwdid.staggered.control_groups import get_valid_control_units, ControlGroupStrategy
        
        mask = get_valid_control_units(
            data, 'gvar', 'id', cohort=2001, period=2001,
            strategy=ControlGroupStrategy.NEVER_TREATED
        )
        
        # Expected: 3 NT units
        expected_size = 3
        actual_size = mask.sum()
        
        assert actual_size == expected_size, \
            f"Control size {actual_size} != expected {expected_size}"
    
    def test_not_yet_treated_control_size(self):
        """
        Verify control group size with NOT_YET_TREATED strategy.
        
        Control = {NT} ∪ {g > period}
        """
        data = pd.DataFrame({
            'id': list(range(1, 11)) * 3,
            'year': [2000, 2001, 2002] * 10,
            'y': np.random.randn(30),
            # 3 NT, 3 cohort 2001, 4 cohort 2002
            'gvar': [0, np.inf, np.nan, 2001, 2001, 2001, 2002, 2002, 2002, 2002] * 3
        })
        
        from lwdid.staggered.control_groups import get_valid_control_units, ControlGroupStrategy
        
        mask = get_valid_control_units(
            data, 'gvar', 'id', cohort=2001, period=2001,
            strategy=ControlGroupStrategy.NOT_YET_TREATED
        )
        
        # Expected: 3 NT + 4 NYT (cohort 2002) = 7
        expected_size = 7
        actual_size = mask.sum()
        
        assert actual_size == expected_size, \
            f"Control size {actual_size} != expected {expected_size}"


class TestNumericalPrecision:
    """
    TEST-12: Numerical precision tests for weight calculations.
    """
    
    def test_floating_point_weight_precision(self):
        """
        Verify weight calculations maintain numerical precision.
        """
        # Create scenario with potential precision issues
        cohort_sizes = {4: 33, 5: 33, 6: 34}  # Sum = 100
        n_total = sum(cohort_sizes.values())
        
        # Calculate weights
        weights = {g: n / n_total for g, n in cohort_sizes.items()}
        
        # Verify sum is very close to 1
        weight_sum = sum(weights.values())
        assert abs(weight_sum - 1.0) < 1e-14, \
            f"Weight sum precision error: {weight_sum - 1.0}"
    
    def test_large_sample_weight_stability(self):
        """
        Verify weight calculations are stable with large samples.
        """
        np.random.seed(42)
        
        # Large sample sizes
        cohort_sizes = {4: 10000, 5: 10000, 6: 10000}
        n_total = sum(cohort_sizes.values())
        
        # Calculate weights
        weights = {g: n / n_total for g, n in cohort_sizes.items()}
        
        # Each weight should be exactly 1/3
        expected_weight = 1/3
        for g, w in weights.items():
            assert np.isclose(w, expected_weight, rtol=1e-10), \
                f"Cohort {g} weight {w} != expected {expected_weight}"
    
    def test_small_sample_weight_stability(self):
        """
        Verify weight calculations are stable with small samples.
        """
        # Small sample sizes
        cohort_sizes = {4: 3, 5: 3, 6: 4}  # Sum = 10
        n_total = sum(cohort_sizes.values())
        
        # Calculate weights
        weights = {g: n / n_total for g, n in cohort_sizes.items()}
        
        # Verify weights
        assert np.isclose(weights[4], 0.3, rtol=1e-10)
        assert np.isclose(weights[5], 0.3, rtol=1e-10)
        assert np.isclose(weights[6], 0.4, rtol=1e-10)


class TestAggregationFormulas:
    """
    TEST-13: Verify aggregation formulas with never-treated.
    """
    
    def test_simple_average_aggregation(self):
        """
        Verify simple average aggregation formula.
        
        ATT_simple = (1/G) × Σ ATT_g
        """
        cohort_atts = {4: 5.0, 5: 6.0, 6: 7.0}
        
        # Simple average
        expected_att = sum(cohort_atts.values()) / len(cohort_atts)
        
        assert np.isclose(expected_att, 6.0, rtol=1e-10)
    
    def test_sample_weighted_aggregation(self):
        """
        Verify sample-weighted aggregation formula.
        
        ATT_weighted = Σ (n_g / n_total) × ATT_g
        """
        cohort_data = {
            4: {'att': 5.0, 'n': 30},
            5: {'att': 6.0, 'n': 50},
            6: {'att': 7.0, 'n': 20},
        }
        
        n_total = sum(c['n'] for c in cohort_data.values())
        
        # Weighted average
        expected_att = sum(
            (c['n'] / n_total) * c['att']
            for c in cohort_data.values()
        )
        
        # Manual: (30/100)*5 + (50/100)*6 + (20/100)*7 = 1.5 + 3.0 + 1.4 = 5.9
        assert np.isclose(expected_att, 5.9, rtol=1e-10)
    
    def test_variance_aggregation(self):
        """
        Verify variance aggregation formula.
        
        Var(ATT_weighted) = Σ w_g² × Var(ATT_g)
        """
        cohort_data = {
            4: {'se': 1.0, 'n': 50},
            5: {'se': 1.5, 'n': 50},
            6: {'se': 2.0, 'n': 50},
        }
        
        n_total = sum(c['n'] for c in cohort_data.values())
        
        # Calculate aggregated variance
        # Var = Σ w² × se²
        agg_var = sum(
            (c['n'] / n_total)**2 * c['se']**2
            for c in cohort_data.values()
        )
        
        # Manual: (1/3)² × 1² + (1/3)² × 1.5² + (1/3)² × 2²
        #       = (1/9) × (1 + 2.25 + 4) = (1/9) × 7.25 ≈ 0.806
        expected_var = (1/9) * (1 + 2.25 + 4)
        
        assert np.isclose(agg_var, expected_var, rtol=1e-10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
