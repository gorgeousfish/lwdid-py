"""
Paper end-to-end tests for never-treated handling.

This module verifies that the implementation matches the paper's
specifications for never-treated identification and control group selection.

Based on: Lee & Wooldridge (2025) ssrn-4516518, Section 4
Spec: .kiro/specs/never-treated-validation/

Test Categories:
- Paper Section 4 setup verification
- Paper formula (4.3) control group verification
- Paper Appendix C Monte Carlo DGP verification
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


class TestPaperSection4Setup:
    """
    TEST-7: Paper Section 4 example verification.
    
    Paper setup: T=6 periods, cohorts={4,5,6,∞}
    - Cohort 4: treated starting period 4
    - Cohort 5: treated starting period 5
    - Cohort 6: treated starting period 6
    - Cohort ∞: never-treated (gvar=0, inf, or NaN)
    """
    
    def test_paper_setup_t6_cohorts_456_inf(self):
        """Verify setup matches paper Section 4: T=6, cohorts={4,5,6,∞}."""
        np.random.seed(42)
        n_per_cohort = 50
        
        # Create paper Section 4 data
        data = self._create_paper_section4_data(n_per_cohort)
        
        # Verify period structure: T=6
        assert data['year'].nunique() == 6
        assert sorted(data['year'].unique()) == [1, 2, 3, 4, 5, 6]
        
        # Verify cohort structure: {4, 5, 6}
        unit_gvar = data.groupby('id')['gvar'].first()
        treated_cohorts = sorted(unit_gvar[unit_gvar > 0].unique())
        assert treated_cohorts == [4, 5, 6], f"Expected [4,5,6], got {treated_cohorts}"
        
        # Verify never-treated count
        n_nt = unit_gvar.apply(is_never_treated).sum()
        assert n_nt == n_per_cohort, f"Expected {n_per_cohort} NT units, got {n_nt}"
        
        # Verify total units: 4 cohorts × 50 = 200
        assert data['id'].nunique() == 4 * n_per_cohort
    
    def test_paper_cohort_sizes_equal(self):
        """Verify each cohort has equal size as in paper."""
        np.random.seed(42)
        n_per_cohort = 50
        
        data = self._create_paper_section4_data(n_per_cohort)
        unit_gvar = data.groupby('id')['gvar'].first()
        
        # Count units per cohort
        cohort_counts = {}
        for cohort in [4, 5, 6]:
            cohort_counts[cohort] = (unit_gvar == cohort).sum()
        
        # Count NT units
        cohort_counts['NT'] = unit_gvar.apply(is_never_treated).sum()
        
        # All should be equal
        for cohort, count in cohort_counts.items():
            assert count == n_per_cohort, f"Cohort {cohort} has {count} units, expected {n_per_cohort}"
    
    def test_paper_treatment_timing(self):
        """Verify treatment timing matches paper specification."""
        np.random.seed(42)
        data = self._create_paper_section4_data(n_per_cohort=50)
        
        # For each cohort, verify treatment starts at correct period
        for cohort in [4, 5, 6]:
            cohort_data = data[data['gvar'] == cohort]
            
            # Pre-treatment periods: t < cohort
            pre_data = cohort_data[cohort_data['year'] < cohort]
            assert len(pre_data) > 0, f"No pre-treatment data for cohort {cohort}"
            
            # Post-treatment periods: t >= cohort
            post_data = cohort_data[cohort_data['year'] >= cohort]
            assert len(post_data) > 0, f"No post-treatment data for cohort {cohort}"
    
    def _create_paper_section4_data(self, n_per_cohort: int = 50) -> pd.DataFrame:
        """Create data matching paper Section 4 setup."""
        data_rows = []
        uid = 0
        
        # Cohorts 4, 5, 6
        for cohort in [4, 5, 6]:
            for _ in range(n_per_cohort):
                for t in range(1, 7):
                    treated = 1 if t >= cohort else 0
                    y = 10 + 0.5*t + 5*treated + np.random.randn()
                    data_rows.append({
                        'id': uid, 'year': t, 'y': y, 'gvar': cohort
                    })
                uid += 1
        
        # Never-treated (gvar = 0)
        for _ in range(n_per_cohort):
            for t in range(1, 7):
                y = 10 + 0.5*t + np.random.randn()
                data_rows.append({
                    'id': uid, 'year': t, 'y': y, 'gvar': 0
                })
            uid += 1
        
        return pd.DataFrame(data_rows)


class TestPaperFormula43:
    """
    TEST-8: Paper formula (4.3) control group verification.
    
    Paper formula (4.3): A_{r+1} = {i : g_i > r} ∪ {i : g_i = ∞}
    
    Control group at period r consists of:
    1. Units not yet treated at r (g_i > r)
    2. Never-treated units (g_i ∈ {0, inf, NaN})
    """
    
    def test_control_group_for_cohort4_period4(self):
        """Verify control group for cohort=4, period=4."""
        np.random.seed(42)
        data = self._create_paper_section4_data(n_per_cohort=50)
        
        # For cohort=4, period=4:
        # A_5 = {g > 4} ∪ {NT} = {cohort 5, 6} ∪ {NT}
        mask = get_valid_control_units(
            data, 'gvar', 'id', cohort=4, period=4,
            strategy=ControlGroupStrategy.NOT_YET_TREATED
        )
        
        unit_gvar = data.groupby('id')['gvar'].first()
        
        for uid, gvar in unit_gvar.items():
            if is_never_treated(gvar) or gvar > 4:
                assert mask.loc[uid] == True, \
                    f"Unit {uid} with gvar={gvar} should be in control"
            else:
                assert mask.loc[uid] == False, \
                    f"Unit {uid} with gvar={gvar} should not be in control"
    
    def test_control_group_for_cohort5_period5(self):
        """Verify control group for cohort=5, period=5."""
        np.random.seed(42)
        data = self._create_paper_section4_data(n_per_cohort=50)
        
        # For cohort=5, period=5:
        # A_6 = {g > 5} ∪ {NT} = {cohort 6} ∪ {NT}
        mask = get_valid_control_units(
            data, 'gvar', 'id', cohort=5, period=5,
            strategy=ControlGroupStrategy.NOT_YET_TREATED
        )
        
        unit_gvar = data.groupby('id')['gvar'].first()
        
        for uid, gvar in unit_gvar.items():
            if is_never_treated(gvar) or gvar > 5:
                assert mask.loc[uid] == True
            else:
                assert mask.loc[uid] == False
    
    def test_control_group_for_cohort6_period6(self):
        """Verify control group for cohort=6, period=6."""
        np.random.seed(42)
        data = self._create_paper_section4_data(n_per_cohort=50)
        
        # For cohort=6, period=6:
        # A_7 = {g > 6} ∪ {NT} = {} ∪ {NT} = {NT only}
        mask = get_valid_control_units(
            data, 'gvar', 'id', cohort=6, period=6,
            strategy=ControlGroupStrategy.NOT_YET_TREATED
        )
        
        unit_gvar = data.groupby('id')['gvar'].first()
        
        for uid, gvar in unit_gvar.items():
            if is_never_treated(gvar):
                assert mask.loc[uid] == True
            else:
                assert mask.loc[uid] == False
    
    def test_never_treated_only_strategy(self):
        """Verify NEVER_TREATED strategy uses only NT units."""
        np.random.seed(42)
        data = self._create_paper_section4_data(n_per_cohort=50)
        
        # NEVER_TREATED strategy should only include NT units
        mask = get_valid_control_units(
            data, 'gvar', 'id', cohort=4, period=4,
            strategy=ControlGroupStrategy.NEVER_TREATED
        )
        
        unit_gvar = data.groupby('id')['gvar'].first()
        
        for uid, gvar in unit_gvar.items():
            if is_never_treated(gvar):
                assert mask.loc[uid] == True
            else:
                assert mask.loc[uid] == False
    
    def test_control_group_size_decreases_over_time(self):
        """Verify control group size decreases as more cohorts get treated."""
        np.random.seed(42)
        data = self._create_paper_section4_data(n_per_cohort=50)
        
        control_sizes = {}
        for period in [4, 5, 6]:
            mask = get_valid_control_units(
                data, 'gvar', 'id', cohort=4, period=period,
                strategy=ControlGroupStrategy.NOT_YET_TREATED
            )
            control_sizes[period] = mask.sum()
        
        # Control group should shrink: period 4 > period 5 > period 6
        assert control_sizes[4] >= control_sizes[5] >= control_sizes[6]
    
    def _create_paper_section4_data(self, n_per_cohort: int = 50) -> pd.DataFrame:
        """Create data matching paper Section 4 setup."""
        data_rows = []
        uid = 0
        
        for cohort in [4, 5, 6]:
            for _ in range(n_per_cohort):
                for t in range(1, 7):
                    treated = 1 if t >= cohort else 0
                    y = 10 + 0.5*t + 5*treated + np.random.randn()
                    data_rows.append({
                        'id': uid, 'year': t, 'y': y, 'gvar': cohort
                    })
                uid += 1
        
        for _ in range(n_per_cohort):
            for t in range(1, 7):
                y = 10 + 0.5*t + np.random.randn()
                data_rows.append({
                    'id': uid, 'year': t, 'y': y, 'gvar': 0
                })
            uid += 1
        
        return pd.DataFrame(data_rows)


class TestPaperAppendixC:
    """
    TEST-9: Paper Appendix C Monte Carlo DGP verification.
    
    Paper Appendix C specifies a DGP for Monte Carlo simulations
    with approximately 30% never-treated units.
    """
    
    def test_appendix_c_dgp_never_treated_ratio(self):
        """Verify DGP produces correct never-treated ratio (~30%)."""
        np.random.seed(42)
        
        # Generate DGP with target 30% NT ratio
        data = self._generate_appendix_c_dgp(n_units=1000, nt_ratio=0.3)
        
        unit_gvar = data.groupby('id')['gvar'].first()
        actual_nt_ratio = unit_gvar.apply(is_never_treated).mean()
        
        # Allow ±5% tolerance
        assert 0.25 <= actual_nt_ratio <= 0.35, \
            f"NT ratio {actual_nt_ratio:.2%} outside expected range [25%, 35%]"
    
    def test_appendix_c_dgp_parallel_trends(self):
        """Verify DGP satisfies parallel trends assumption."""
        np.random.seed(42)
        
        # Generate DGP with common trend
        data = self._generate_appendix_c_dgp(n_units=500, nt_ratio=0.3)
        
        # Calculate pre-treatment trends for treated and control
        pre_data = data[data['year'] < 4]
        
        unit_gvar = data.groupby('id')['gvar'].first()
        treated_ids = unit_gvar[unit_gvar == 4].index
        control_ids = unit_gvar[unit_gvar.apply(is_never_treated)].index
        
        # Calculate average trend for each group
        treated_trend = pre_data[pre_data['id'].isin(treated_ids)].groupby('year')['y'].mean()
        control_trend = pre_data[pre_data['id'].isin(control_ids)].groupby('year')['y'].mean()
        
        # Trends should be parallel (similar slope)
        treated_slope = np.polyfit(treated_trend.index, treated_trend.values, 1)[0]
        control_slope = np.polyfit(control_trend.index, control_trend.values, 1)[0]
        
        # Allow 20% tolerance in slope difference
        slope_diff = abs(treated_slope - control_slope) / max(abs(treated_slope), abs(control_slope))
        assert slope_diff < 0.2, f"Slopes differ by {slope_diff:.1%}"
    
    def test_appendix_c_dgp_treatment_effect(self):
        """Verify DGP produces correct treatment effect."""
        np.random.seed(42)
        true_att = 5.0
        
        # Generate DGP with known treatment effect
        data = self._generate_appendix_c_dgp(n_units=500, nt_ratio=0.3, true_att=true_att)
        
        # Simple DiD estimation
        pre_means = data[data['year'] < 4].groupby('id')['y'].mean()
        post_means = data[data['year'] >= 4].groupby('id')['y'].mean()
        delta_y = post_means - pre_means
        
        unit_gvar = data.groupby('id')['gvar'].first()
        treated_mask = unit_gvar == 4
        control_mask = unit_gvar.apply(is_never_treated)
        
        att_est = delta_y[treated_mask].mean() - delta_y[control_mask].mean()
        
        # Should be close to true ATT
        assert abs(att_est - true_att) < 1.0, \
            f"ATT estimate {att_est:.2f} too far from true {true_att}"
    
    def _generate_appendix_c_dgp(
        self, n_units: int, nt_ratio: float, true_att: float = 5.0
    ) -> pd.DataFrame:
        """Generate DGP matching paper Appendix C."""
        data_rows = []
        
        for i in range(n_units):
            # Assign treatment status
            if np.random.rand() < nt_ratio:
                gvar = 0  # never-treated
            else:
                gvar = 4  # treated at period 4
            
            # Unit fixed effect
            alpha_i = np.random.randn() * 2
            
            for t in range(1, 7):
                # Common time trend
                delta_t = 0.5 * t
                
                # Treatment effect
                treated = 1 if (gvar > 0 and t >= gvar) else 0
                
                # Error term
                epsilon = np.random.randn()
                
                # Outcome
                y = alpha_i + delta_t + true_att * treated + epsilon
                
                data_rows.append({
                    'id': i, 'year': t, 'y': y, 'gvar': gvar
                })
        
        return pd.DataFrame(data_rows)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
