"""
Numerical validation tests for DESIGN-048: n_control statistics.

This module performs comprehensive numerical validation of n_control calculation:
1. Manual calculation verification
2. Cross-validation with cohort-time effects
3. Monte Carlo simulation for statistical consistency
"""

import pytest
import numpy as np
import pandas as pd
from lwdid import lwdid


# =============================================================================
# Manual Calculation Verification
# =============================================================================

class TestManualCalculationVerification:
    """Verify n_control against manual calculation."""
    
    @pytest.fixture
    def known_structure_data(self):
        """
        Create data with exactly known control group structure.
        
        Structure:
        - 20 units total
        - Cohort 2002: units 1-5 (5 units)
        - Cohort 2003: units 6-10 (5 units)
        - Cohort 2004: units 11-15 (5 units)
        - Never-treated: units 16-20 (5 units)
        - Time periods: 2000-2005 (6 years)
        """
        np.random.seed(12345)
        
        data = []
        for unit in range(1, 21):
            if unit <= 5:
                gvar = 2002
            elif unit <= 10:
                gvar = 2003
            elif unit <= 15:
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
    
    def test_never_treated_manual_verification(self, known_structure_data):
        """
        Verify n_control for never_treated strategy matches manual count.
        
        Expected: n_control = 5 (units 16-20)
        """
        result = lwdid(
            known_structure_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            control_group='never_treated',
            aggregate='cohort',
        )
        
        # Manual calculation
        unit_gvar = known_structure_data.groupby('id')['gvar'].first()
        manual_n_nt = (unit_gvar == 0).sum()
        
        assert result.n_control == manual_n_nt
        assert result.n_control == 5
    
    def test_not_yet_treated_manual_verification(self, known_structure_data):
        """
        Verify n_control for not_yet_treated strategy matches manual calculation.
        
        Control group sizes by (g, r):
        - (g=2002, r=2002): NT(5) + g2003(5) + g2004(5) = 15
        - (g=2002, r=2003): NT(5) + g2004(5) = 10
        - (g=2002, r=2004): NT(5) = 5
        - (g=2002, r=2005): NT(5) = 5
        - (g=2003, r=2003): NT(5) + g2004(5) = 10
        - (g=2003, r=2004): NT(5) = 5
        - (g=2003, r=2005): NT(5) = 5
        - (g=2004, r=2004): NT(5) = 5
        - (g=2004, r=2005): NT(5) = 5
        
        Maximum n_control = 15 (for g=2002, r=2002)
        """
        result = lwdid(
            known_structure_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            control_group='not_yet_treated',
            aggregate='none',
        )
        
        # Expected maximum n_control
        expected_max_n_control = 15  # NT(5) + g2003(5) + g2004(5)
        
        # Verify result
        assert result.n_control == expected_max_n_control
        
        # Also verify it matches max from cohort-time effects
        att_df = result.att_by_cohort_time
        max_from_effects = att_df['n_control'].max()
        assert result.n_control == max_from_effects
    
    def test_control_group_sizes_by_cohort_time(self, known_structure_data):
        """
        Verify individual (g, r) control group sizes are correct.
        """
        result = lwdid(
            known_structure_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            control_group='not_yet_treated',
            aggregate='none',
        )
        
        att_df = result.att_by_cohort_time
        
        # Expected control group sizes
        expected = {
            (2002, 2002): 15,  # NT(5) + g2003(5) + g2004(5)
            (2002, 2003): 10,  # NT(5) + g2004(5)
            (2002, 2004): 5,   # NT(5)
            (2002, 2005): 5,   # NT(5)
            (2003, 2003): 10,  # NT(5) + g2004(5)
            (2003, 2004): 5,   # NT(5)
            (2003, 2005): 5,   # NT(5)
            (2004, 2004): 5,   # NT(5)
            (2004, 2005): 5,   # NT(5)
        }
        
        for _, row in att_df.iterrows():
            key = (row['cohort'], row['period'])
            if key in expected:
                assert row['n_control'] == expected[key], \
                    f"Mismatch for {key}: expected {expected[key]}, got {row['n_control']}"


# =============================================================================
# Cross-Validation Tests
# =============================================================================

class TestCrossValidation:
    """Cross-validate n_control across different calculation methods."""
    
    @pytest.fixture
    def random_staggered_data(self):
        """Create random staggered data for cross-validation."""
        np.random.seed(42)
        
        n_units = 50
        cohorts = [2002, 2003, 2004, 0]  # 0 = never treated
        
        data = []
        for unit in range(1, n_units + 1):
            gvar = np.random.choice(cohorts)
            
            for year in range(2000, 2006):
                y = np.random.normal(0, 1)
                if gvar > 0 and year >= gvar:
                    y += 0.5
                
                data.append({
                    'id': unit,
                    'year': year,
                    'y': y,
                    'gvar': gvar
                })
        
        return pd.DataFrame(data)
    
    def test_n_control_consistency_demean_vs_detrend(self, random_staggered_data):
        """n_control should be consistent across transformation methods."""
        result_demean = lwdid(
            random_staggered_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            control_group='never_treated',
            aggregate='cohort',
        )
        
        result_detrend = lwdid(
            random_staggered_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='detrend',
            control_group='never_treated',
            aggregate='cohort',
        )
        
        # n_control should be the same regardless of transformation
        assert result_demean.n_control == result_detrend.n_control
    
    def test_n_control_sum_equals_cohort_time_total(self, random_staggered_data):
        """
        For never_treated strategy, n_control should equal n_never_treated
        across all cohort-time effects.
        """
        result = lwdid(
            random_staggered_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            control_group='never_treated',
            aggregate='none',
        )
        
        # All cohort-time effects should have same n_control = n_never_treated
        att_df = result.att_by_cohort_time
        unique_n_controls = att_df['n_control'].unique()
        
        # Should all be the same for never_treated strategy
        assert len(unique_n_controls) == 1
        assert unique_n_controls[0] == result.n_never_treated


# =============================================================================
# Monte Carlo Simulation
# =============================================================================

class TestMonteCarloSimulation:
    """Monte Carlo simulation for n_control statistical properties."""
    
    def test_n_control_stability_across_random_seeds(self):
        """n_control should be deterministic given data structure."""
        
        def create_data(seed):
            np.random.seed(seed)
            data = []
            for unit in range(1, 21):
                if unit <= 5:
                    gvar = 2002
                elif unit <= 10:
                    gvar = 2003
                else:
                    gvar = 0
                
                for year in range(2000, 2005):
                    y = np.random.normal(0, 1)
                    data.append({
                        'id': unit,
                        'year': year,
                        'y': y,
                        'gvar': gvar
                    })
            return pd.DataFrame(data)
        
        n_controls = []
        for seed in [1, 42, 123, 456, 789]:
            data = create_data(seed)
            result = lwdid(
                data,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                rolling='demean',
                control_group='never_treated',
                aggregate='cohort',
            )
            n_controls.append(result.n_control)
        
        # n_control should be the same for all seeds (same data structure)
        assert len(set(n_controls)) == 1
        assert n_controls[0] == 10  # 10 never-treated units


# =============================================================================
# Boundary Condition Tests
# =============================================================================

class TestBoundaryConditions:
    """Test boundary conditions for n_control calculation."""
    
    def test_all_units_treated_eventually(self):
        """Test n_control when all units are eventually treated."""
        np.random.seed(42)
        
        data = []
        for unit in range(1, 11):
            if unit <= 5:
                gvar = 2002
            else:
                gvar = 2003
            
            for year in range(2000, 2005):
                y = np.random.normal(0, 1)
                data.append({
                    'id': unit,
                    'year': year,
                    'y': y,
                    'gvar': gvar
                })
        
        df = pd.DataFrame(data)
        
        result = lwdid(
            df,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            control_group='not_yet_treated',
            aggregate='none',
        )
        
        # n_control should be > 0 (not-yet-treated units exist)
        assert result.n_control > 0
        assert result.n_never_treated == 0
    
    def test_single_period_treatment(self):
        """Test n_control with single treatment cohort."""
        np.random.seed(42)
        
        data = []
        for unit in range(1, 11):
            if unit <= 5:
                gvar = 2003
            else:
                gvar = 0
            
            for year in range(2000, 2005):
                y = np.random.normal(0, 1)
                data.append({
                    'id': unit,
                    'year': year,
                    'y': y,
                    'gvar': gvar
                })
        
        df = pd.DataFrame(data)
        
        result = lwdid(
            df,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            control_group='never_treated',
            aggregate='cohort',
        )
        
        assert result.n_control == 5
        assert result.n_treated == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
