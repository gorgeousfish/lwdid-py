"""
Unit tests for DESIGN-048: n_control statistics accuracy in staggered mode.

Verifies that n_control is correctly computed based on the actual control group
strategy used:
- 'never_treated': n_control = n_never_treated
- 'not_yet_treated': n_control = max(n_control across cohort-time effects)

Test coverage:
1. Never-treated control group strategy
2. Not-yet-treated control group strategy  
3. Edge cases (empty cohort_time_effects, single cohort)
4. Numerical validation against manual calculation
"""

import pytest
import numpy as np
import pandas as pd
from lwdid import lwdid


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def staggered_data_with_nt():
    """
    Create staggered test data with never-treated units.
    
    Structure:
    - 10 units total
    - Cohort 2002: units 1-3 (3 units)
    - Cohort 2003: units 4-6 (3 units)
    - Never-treated: units 7-10 (4 units)
    - Time periods: 2000-2004 (5 years)
    """
    np.random.seed(42)
    
    data = []
    for unit in range(1, 11):
        if unit <= 3:
            gvar = 2002  # Cohort 2002
        elif unit <= 6:
            gvar = 2003  # Cohort 2003
        else:
            gvar = 0  # Never treated
        
        for year in range(2000, 2005):
            y = 1.0 + 0.1 * unit + 0.05 * year + np.random.normal(0, 0.1)
            # Add treatment effect post-treatment
            if gvar > 0 and year >= gvar:
                y += 0.5
            
            data.append({
                'id': unit,
                'year': year,
                'y': y,
                'gvar': gvar
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def staggered_data_no_nt():
    """
    Create staggered test data without never-treated units.
    
    Structure:
    - 6 units total
    - Cohort 2002: units 1-3 (3 units)
    - Cohort 2003: units 4-6 (3 units)
    - Time periods: 2000-2004 (5 years)
    """
    np.random.seed(42)
    
    data = []
    for unit in range(1, 7):
        if unit <= 3:
            gvar = 2002  # Cohort 2002
        else:
            gvar = 2003  # Cohort 2003
        
        for year in range(2000, 2005):
            y = 1.0 + 0.1 * unit + 0.05 * year + np.random.normal(0, 0.1)
            if gvar > 0 and year >= gvar:
                y += 0.5
            
            data.append({
                'id': unit,
                'year': year,
                'y': y,
                'gvar': gvar
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def staggered_data_many_cohorts():
    """
    Create staggered data with many cohorts for comprehensive n_control testing.
    
    Structure:
    - 15 units total
    - Cohort 2002: units 1-3 (3 units)
    - Cohort 2003: units 4-6 (3 units)
    - Cohort 2004: units 7-9 (3 units)
    - Never-treated: units 10-15 (6 units)
    - Time periods: 2000-2005 (6 years)
    
    Control group sizes under 'not_yet_treated':
    - (g=2002, r=2002): NT(6) + cohort2003(3) + cohort2004(3) = 12
    - (g=2002, r=2003): NT(6) + cohort2004(3) = 9
    - (g=2002, r=2004): NT(6) = 6
    - (g=2003, r=2003): NT(6) + cohort2004(3) = 9
    - (g=2003, r=2004): NT(6) = 6
    - (g=2004, r=2004): NT(6) = 6
    
    Maximum n_control under 'not_yet_treated' = 12
    """
    np.random.seed(42)
    
    data = []
    for unit in range(1, 16):
        if unit <= 3:
            gvar = 2002
        elif unit <= 6:
            gvar = 2003
        elif unit <= 9:
            gvar = 2004
        else:
            gvar = 0  # Never treated
        
        for year in range(2000, 2006):
            y = 1.0 + 0.1 * unit + 0.05 * year + np.random.normal(0, 0.1)
            if gvar > 0 and year >= gvar:
                y += 0.5
            
            data.append({
                'id': unit,
                'year': year,
                'y': y,
                'gvar': gvar
            })
    
    return pd.DataFrame(data)


# =============================================================================
# Test: Never-Treated Control Group Strategy
# =============================================================================

class TestNeverTreatedControlGroup:
    """Tests for n_control when control_group='never_treated'."""
    
    def test_n_control_equals_n_never_treated(self, staggered_data_with_nt):
        """n_control should equal n_never_treated when using never_treated strategy."""
        result = lwdid(
            staggered_data_with_nt,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            control_group='never_treated',
            aggregate='cohort',
        )
        
        # With cohort/overall aggregation, control_group is auto-switched to never_treated
        assert result.control_group_used == 'never_treated'
        assert result.n_control == result.n_never_treated
        assert result.n_control == 4  # 4 never-treated units
    
    def test_n_control_consistency_across_aggregation_levels(self, staggered_data_with_nt):
        """n_control should be consistent across different aggregation levels."""
        # Cohort aggregation
        result_cohort = lwdid(
            staggered_data_with_nt,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            control_group='never_treated',
            aggregate='cohort',
        )
        
        # Overall aggregation
        result_overall = lwdid(
            staggered_data_with_nt,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            control_group='never_treated',
            aggregate='overall',
        )
        
        # Both should have same n_control = n_never_treated
        assert result_cohort.n_control == result_overall.n_control
        assert result_cohort.n_control == 4


# =============================================================================
# Test: Not-Yet-Treated Control Group Strategy
# =============================================================================

class TestNotYetTreatedControlGroup:
    """Tests for n_control when control_group='not_yet_treated'."""
    
    def test_n_control_greater_than_n_never_treated(self, staggered_data_no_nt):
        """
        n_control should be > 0 even without never-treated units
        when using not_yet_treated strategy.
        """
        result = lwdid(
            staggered_data_no_nt,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            control_group='not_yet_treated',
            aggregate='none',  # Must use 'none' without NT units
        )
        
        # With not_yet_treated strategy, n_control should reflect NYT units
        assert result.n_never_treated == 0
        assert result.n_control > 0  # Should have not-yet-treated controls
    
    def test_n_control_matches_max_cohort_time_effect(self, staggered_data_many_cohorts):
        """n_control should equal max(n_control) from cohort-time effects."""
        result = lwdid(
            staggered_data_many_cohorts,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            control_group='not_yet_treated',
            aggregate='none',
        )
        
        # Extract n_control from cohort-time effects
        att_df = result.att_by_cohort_time
        max_n_control_from_effects = att_df['n_control'].max()
        
        # Result n_control should equal max from effects
        assert result.n_control == max_n_control_from_effects
    
    def test_n_control_numerical_validation(self, staggered_data_many_cohorts):
        """
        Validate n_control calculation against manual computation.
        
        For (g=2002, r=2002) with not_yet_treated:
        - Control = NT(6) + cohort2003(3) + cohort2004(3) = 12
        This should be the maximum n_control.
        """
        result = lwdid(
            staggered_data_many_cohorts,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            control_group='not_yet_treated',
            aggregate='none',
        )
        
        # Manual calculation: max control group is for earliest cohort, earliest period
        # (g=2002, r=2002): NT(6) + g2003(3) + g2004(3) = 12
        expected_max_n_control = 12
        
        assert result.n_control == expected_max_n_control


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases in n_control computation."""
    
    def test_single_cohort_with_nt(self, staggered_data_with_nt):
        """Test n_control with single cohort (filtered data)."""
        # Filter to keep only cohort 2002 and NT units
        filtered_data = staggered_data_with_nt[
            (staggered_data_with_nt['gvar'] == 2002) |
            (staggered_data_with_nt['gvar'] == 0)
        ].copy()
        
        result = lwdid(
            filtered_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            control_group='never_treated',
            aggregate='cohort',
        )
        
        assert result.n_control == 4  # Only NT units
        assert result.n_treated == 3  # Cohort 2002 units
    
    def test_nobs_consistency(self, staggered_data_with_nt):
        """Verify nobs = n_treated + n_control."""
        result = lwdid(
            staggered_data_with_nt,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            control_group='never_treated',
            aggregate='cohort',
        )
        
        assert result.nobs == result.n_treated + result.n_control


# =============================================================================
# Test: Summary Output
# =============================================================================

class TestSummaryOutput:
    """Tests that summary() correctly displays n_control."""
    
    def test_summary_contains_n_control(self, staggered_data_with_nt):
        """Verify summary output includes correct n_control."""
        result = lwdid(
            staggered_data_with_nt,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            control_group='never_treated',
            aggregate='cohort',
        )
        
        summary_str = result.summary()
        
        # Check n_control appears in summary
        assert 'Control Units' in summary_str or 'N_control' in summary_str or 'n_control' in summary_str.lower()
        assert str(result.n_control) in summary_str


# =============================================================================
# Test: Detrend Transformation
# =============================================================================

class TestDetrendTransformation:
    """Tests n_control with detrend transformation."""
    
    def test_n_control_with_detrend(self, staggered_data_with_nt):
        """n_control should be computed correctly with detrend transformation."""
        result = lwdid(
            staggered_data_with_nt,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='detrend',
            control_group='never_treated',
            aggregate='cohort',
        )
        
        assert result.n_control == result.n_never_treated
        assert result.n_control == 4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
