"""
Numerical validation tests for never-treated handling.

This module contains numerical tests to verify the mathematical correctness
of never-treated identification and control group selection.

Based on: Lee & Wooldridge (2025) ssrn-4516518, Section 4
Spec: .kiro/specs/never-treated-validation/
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import sys
sys.path.insert(0, 'src')

from lwdid.validation import is_never_treated
from lwdid.staggered.control_groups import (
    identify_never_treated_units,
    get_valid_control_units,
    ControlGroupStrategy,
)


class TestWeightCalculationWithNeverTreated:
    """Verify weight calculations with never-treated units."""
    
    def test_never_treated_not_in_treatment_weights(self):
        """Verify NT units don't contribute to treatment effect weights."""
        # Create data: 3 cohorts (50 each) + 50 NT
        np.random.seed(42)
        
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
        
        # Verify cohort weights should be 1/3 each (NT not counted)
        unit_gvar = data.groupby('id')['gvar'].first()
        treated_mask = ~unit_gvar.apply(is_never_treated)
        
        n_treated = treated_mask.sum()
        assert n_treated == 150  # 50 * 3 cohorts
        
        # Each cohort weight = 50/150 = 1/3
        for cohort in [4, 5, 6]:
            cohort_weight = (unit_gvar == cohort).sum() / n_treated
            assert np.isclose(cohort_weight, 1/3, rtol=0.01)
    
    def test_control_group_size_calculation(self):
        """Verify control group size is correctly calculated."""
        data = pd.DataFrame({
            'id': list(range(1, 11)) * 3,
            'year': [2000, 2001, 2002] * 10,
            'y': np.random.randn(30),
            # 3 NT (0, inf, nan), 3 cohort 2001, 4 cohort 2002
            'gvar': [0, np.inf, np.nan, 2001, 2001, 2001, 2002, 2002, 2002, 2002] * 3
        })
        
        # For cohort=2001, period=2001:
        # Control = NT (3) + NYT (4 from cohort 2002)
        mask = get_valid_control_units(
            data, 'gvar', 'id', cohort=2001, period=2001,
            strategy=ControlGroupStrategy.NOT_YET_TREATED
        )
        
        assert mask.sum() == 7  # 3 NT + 4 NYT


class TestPaperFormula43Verification:
    """Verify paper formula (4.3) for control group definition."""
    
    def test_formula_43_basic(self):
        """
        Paper formula (4.3): A_{r+1} = {i : g_i > r} ∪ {i : g_i = ∞}
        
        Control group at period r consists of:
        1. Units not yet treated at r (g_i > r)
        2. Never-treated units (g_i ∈ {0, inf, NaN})
        
        Note: The control group includes units with g_i > r (strictly greater),
        meaning units treated at period r are NOT in the control group.
        """
        # Create test data
        data = pd.DataFrame({
            'id': list(range(1, 8)) * 5,
            'year': list(range(1, 6)) * 7,
            'y': np.random.randn(35),
            'gvar': [2, 3, 4, 5, 0, np.inf, np.nan] * 5
        })
        
        # Test for different periods
        # For cohort g at period r, control = {g_i > r} ∪ {NT}
        test_cases = [
            # (cohort, period, expected_control_ids)
            # For cohort=2, period=2: control = {g>2} ∪ {NT} = {3,4,5} ∪ {5,6,7}
            # Note: id=5 has gvar=0 (NT), not gvar=5
            (2, 2, [3, 4, 5, 6, 7]),  # g>2: {id with gvar 3,4,5}, NT: {id 5,6,7}
            (3, 3, [4, 5, 6, 7]),      # g>3: {id with gvar 4,5}, NT: {id 5,6,7}
            (4, 4, [5, 6, 7]),         # g>4: {id with gvar 5}, NT: {id 5,6,7}
        ]
        
        # First verify the data structure
        unit_gvar = data.groupby('id')['gvar'].first()
        # id 1: gvar=2, id 2: gvar=3, id 3: gvar=4, id 4: gvar=5
        # id 5: gvar=0 (NT), id 6: gvar=inf (NT), id 7: gvar=nan (NT)
        
        for cohort, period, expected_ids in test_cases:
            mask = get_valid_control_units(
                data, 'gvar', 'id', cohort=cohort, period=period,
                strategy=ControlGroupStrategy.NOT_YET_TREATED
            )
            
            actual_ids = sorted(mask[mask].index.tolist())
            
            # Recalculate expected based on actual data
            # Control = {g > period} ∪ {NT}
            expected_recalc = []
            for uid, gvar in unit_gvar.items():
                if is_never_treated(gvar) or (not is_never_treated(gvar) and gvar > period):
                    expected_recalc.append(uid)
            expected_recalc = sorted(expected_recalc)
            
            assert actual_ids == expected_recalc, \
                f"For cohort={cohort}, period={period}: expected {expected_recalc}, got {actual_ids}"
    
    def test_formula_43_never_treated_only(self):
        """Test NEVER_TREATED strategy (only NT units in control)."""
        data = pd.DataFrame({
            'id': list(range(1, 6)) * 3,
            'year': [2000, 2001, 2002] * 5,
            'y': np.random.randn(15),
            'gvar': [2001, 2002, 0, np.inf, np.nan] * 3
        })
        
        mask = get_valid_control_units(
            data, 'gvar', 'id', cohort=2001, period=2001,
            strategy=ControlGroupStrategy.NEVER_TREATED
        )
        
        # Only NT units (ids 3, 4, 5)
        expected_ids = [3, 4, 5]
        actual_ids = sorted(mask[mask].index.tolist())
        assert actual_ids == expected_ids


class TestATTEstimationWithNeverTreated:
    """Test ATT estimation correctness with never-treated control."""
    
    def test_simple_att_with_nt_control(self):
        """Test simple ATT estimation using NT control group."""
        np.random.seed(42)
        true_att = 5.0
        
        # Generate data
        data_rows = []
        uid = 0
        
        # 50 treated units (cohort 4)
        for _ in range(50):
            alpha_i = np.random.randn()
            for t in range(1, 7):
                treated = 1 if t >= 4 else 0
                y = alpha_i + 0.5*t + true_att*treated + np.random.randn()
                data_rows.append({
                    'id': uid, 'year': t, 'y': y, 'gvar': 4
                })
            uid += 1
        
        # 50 NT units
        for _ in range(50):
            alpha_i = np.random.randn()
            for t in range(1, 7):
                y = alpha_i + 0.5*t + np.random.randn()
                data_rows.append({
                    'id': uid, 'year': t, 'y': y, 'gvar': 0
                })
            uid += 1
        
        data = pd.DataFrame(data_rows)
        
        # Simple DiD estimation
        # Pre-treatment mean (t < 4)
        pre_data = data[data['year'] < 4]
        pre_means = pre_data.groupby('id')['y'].mean()
        
        # Post-treatment mean (t >= 4)
        post_data = data[data['year'] >= 4]
        post_means = post_data.groupby('id')['y'].mean()
        
        # Transformed outcome
        delta_y = post_means - pre_means
        
        # Separate by treatment status
        unit_gvar = data.groupby('id')['gvar'].first()
        treated_mask = unit_gvar == 4
        control_mask = unit_gvar.apply(is_never_treated)
        
        # ATT estimate
        att_est = delta_y[treated_mask].mean() - delta_y[control_mask].mean()
        
        # Should be close to true ATT
        assert abs(att_est - true_att) < 1.0, f"ATT estimate {att_est} too far from {true_att}"
    
    def test_att_unbiased_with_parallel_trends(self):
        """Test ATT is unbiased when parallel trends holds."""
        np.random.seed(123)
        true_att = 3.0
        n_simulations = 50
        att_estimates = []
        
        for sim in range(n_simulations):
            np.random.seed(sim + 1000)
            
            data_rows = []
            uid = 0
            
            # Common trend for all units
            common_trend = 0.5
            
            # Treated units
            for _ in range(30):
                alpha_i = np.random.randn() * 2
                for t in range(1, 7):
                    treated = 1 if t >= 4 else 0
                    y = alpha_i + common_trend*t + true_att*treated + np.random.randn()
                    data_rows.append({
                        'id': uid, 'year': t, 'y': y, 'gvar': 4
                    })
                uid += 1
            
            # NT units (same trend)
            for _ in range(30):
                alpha_i = np.random.randn() * 2
                for t in range(1, 7):
                    y = alpha_i + common_trend*t + np.random.randn()
                    data_rows.append({
                        'id': uid, 'year': t, 'y': y, 'gvar': 0
                    })
                uid += 1
            
            data = pd.DataFrame(data_rows)
            
            # Estimate ATT
            pre_means = data[data['year'] < 4].groupby('id')['y'].mean()
            post_means = data[data['year'] >= 4].groupby('id')['y'].mean()
            delta_y = post_means - pre_means
            
            unit_gvar = data.groupby('id')['gvar'].first()
            treated_mask = unit_gvar == 4
            control_mask = unit_gvar.apply(is_never_treated)
            
            att_est = delta_y[treated_mask].mean() - delta_y[control_mask].mean()
            att_estimates.append(att_est)
        
        # Check bias
        mean_att = np.mean(att_estimates)
        bias = mean_att - true_att
        
        assert abs(bias) < 0.5, f"Bias {bias} too large (mean ATT = {mean_att})"


class TestEdgeCasesNumerical:
    """Numerical edge case tests."""
    
    def test_single_nt_unit(self):
        """Test with single never-treated unit."""
        data = pd.DataFrame({
            'id': [1, 2, 3] * 3,
            'year': [2000, 2001, 2002] * 3,
            'y': np.random.randn(9),
            'gvar': [0, 2001, 2002] * 3  # Only 1 NT unit
        })
        
        mask = get_valid_control_units(
            data, 'gvar', 'id', cohort=2001, period=2001,
            strategy=ControlGroupStrategy.NEVER_TREATED
        )
        
        assert mask.sum() == 1
        assert mask.loc[1] == True
    
    def test_all_nt_units(self):
        """Test with all never-treated units."""
        data = pd.DataFrame({
            'id': [1, 2, 3] * 3,
            'year': [2000, 2001, 2002] * 3,
            'y': np.random.randn(9),
            'gvar': [0, np.inf, np.nan] * 3  # All NT
        })
        
        unit_gvar = data.groupby('id')['gvar'].first()
        n_nt = unit_gvar.apply(is_never_treated).sum()
        
        assert n_nt == 3
    
    def test_large_cohort_values(self):
        """Test with large cohort values (e.g., year 2050)."""
        data = pd.DataFrame({
            'id': [1, 2, 3] * 3,
            'year': [2040, 2045, 2050] * 3,
            'y': np.random.randn(9),
            'gvar': [0, 2045, 2050] * 3
        })
        
        mask = get_valid_control_units(
            data, 'gvar', 'id', cohort=2045, period=2045,
            strategy=ControlGroupStrategy.NOT_YET_TREATED
        )
        
        # NT (id=1) + NYT (id=3, gvar=2050 > 2045)
        assert mask.sum() == 2
        assert mask.loc[1] == True
        assert mask.loc[3] == True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
