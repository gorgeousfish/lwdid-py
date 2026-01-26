"""
Tests for BUG-115 fix: Float comparison using tolerance in estimation.py.

BUG-115: control_groups.py and estimation.py used == for float comparison
    - Problem: Direct == comparison of floating-point cohort values can fail
      due to precision issues (e.g., 2005.0000000001 != 2005.0)
    - Fix: Use COHORT_FLOAT_TOLERANCE for cohort value comparisons

This test verifies that:
1. Cohort mask comparison works correctly with floating-point precision issues
2. Treatment indicator assignment uses tolerance comparison
3. Results are consistent when cohort values have minor floating-point errors
"""

import numpy as np
import pandas as pd
import pytest

from lwdid.validation import COHORT_FLOAT_TOLERANCE
from lwdid.staggered.estimation import estimate_cohort_time_effects
from lwdid.staggered.transformations import transform_staggered_demean
from lwdid.staggered.control_groups import count_control_units_by_strategy


# =============================================================================
# BUG-115 Tests: Float Comparison Tolerance in Estimation
# =============================================================================

class TestBug115FloatComparisonTolerance:
    """
    BUG-115 regression tests for floating-point cohort comparison.
    
    The estimation module must correctly identify treated units when:
    - Cohort values are exact integers (e.g., 2005)
    - Cohort values are floats (e.g., 2005.0)
    - Cohort values have minor floating-point errors (e.g., 2005.0000000001)
    """
    
    @pytest.fixture
    def panel_data(self):
        """Create a simple staggered panel dataset."""
        np.random.seed(42)
        
        # 10 units, 6 time periods (2000-2005)
        units = list(range(1, 11))
        years = list(range(2000, 2006))
        
        data = []
        for unit in units:
            # Assign cohorts: units 1-3 -> cohort 2003, units 4-6 -> cohort 2004
            # units 7-10 -> never treated (gvar=0)
            if unit <= 3:
                gvar = 2003
            elif unit <= 6:
                gvar = 2004
            else:
                gvar = 0  # never treated
            
            for year in years:
                # Simple outcome: base + unit effect + time effect + treatment effect
                y = 10 + unit * 0.5 + year * 0.1
                if gvar > 0 and year >= gvar:
                    y += 2.0  # treatment effect
                
                data.append({
                    'id': unit,
                    'year': year,
                    'gvar': float(gvar),  # Use float to allow precision tests
                    'y': y + np.random.normal(0, 0.1),
                })
        
        return pd.DataFrame(data)
    
    def test_count_control_units_with_exact_cohort(self, panel_data):
        """Control unit counting works with exact cohort values."""
        counts = count_control_units_by_strategy(
            panel_data, 
            gvar='gvar', 
            ivar='id',
            cohort=2003,
            period=2003
        )
        
        assert counts['never_treated'] == 4  # units 7-10
        assert counts['treatment_cohort'] == 3  # units 1-3
    
    def test_count_control_units_with_float_precision_error(self, panel_data):
        """Control unit counting works when cohort has precision error."""
        # Simulate floating-point precision error
        cohort_with_error = 2003 + COHORT_FLOAT_TOLERANCE / 2
        
        counts = count_control_units_by_strategy(
            panel_data,
            gvar='gvar',
            ivar='id',
            cohort=cohort_with_error,
            period=2003
        )
        
        # Should still correctly identify treatment cohort
        assert counts['treatment_cohort'] == 3
    
    def test_count_control_units_with_gvar_precision_error(self, panel_data):
        """Control unit counting works when gvar column has precision error."""
        # Modify gvar to have floating-point precision error
        data_with_error = panel_data.copy()
        data_with_error.loc[data_with_error['gvar'] == 2003.0, 'gvar'] = 2003 + COHORT_FLOAT_TOLERANCE / 2
        
        counts = count_control_units_by_strategy(
            data_with_error,
            gvar='gvar',
            ivar='id',
            cohort=2003,
            period=2003
        )
        
        # Should still correctly identify treatment cohort
        assert counts['treatment_cohort'] == 3
    
    def test_estimate_cohort_time_effects_with_exact_values(self, panel_data):
        """Estimation works correctly with exact cohort values."""
        # First transform the data
        transformed = transform_staggered_demean(
            panel_data, 'y', 'id', 'year', 'gvar'
        )
        
        # Then estimate effects
        results = estimate_cohort_time_effects(
            data_transformed=transformed,
            gvar='gvar',
            ivar='id',
            tvar='year',
            controls=None,
            estimator='ra',
        )
        
        # Should have results for cohort 2003 and 2004
        assert len(results) > 0
        
        # Check that we have estimates for expected (g, r) pairs
        estimated_pairs = [(e.cohort, e.period) for e in results]
        assert (2003, 2003) in estimated_pairs
        assert (2004, 2004) in estimated_pairs
    
    def test_estimate_cohort_time_effects_with_gvar_precision_error(self, panel_data):
        """Estimation works when gvar has floating-point precision errors."""
        # Modify gvar for cohort 2003 to have precision error
        data_with_error = panel_data.copy()
        data_with_error.loc[data_with_error['gvar'] == 2003.0, 'gvar'] = 2003 + COHORT_FLOAT_TOLERANCE / 2
        
        # Transform and estimate
        transformed = transform_staggered_demean(
            data_with_error, 'y', 'id', 'year', 'gvar'
        )
        
        results = estimate_cohort_time_effects(
            data_transformed=transformed,
            gvar='gvar',
            ivar='id',
            tvar='year',
            controls=None,
            estimator='ra',
        )
        
        # Should still produce results
        assert len(results) > 0
        
        # All effects should have valid ATT estimates
        for effect in results:
            assert not np.isnan(effect.att)
            assert not np.isinf(effect.att)
    
    def test_treatment_mask_with_tolerance(self, panel_data):
        """Treatment mask should use tolerance comparison."""
        # Transform data
        transformed = transform_staggered_demean(
            panel_data, 'y', 'id', 'year', 'gvar'
        )
        
        # Estimate effects
        results = estimate_cohort_time_effects(
            data_transformed=transformed,
            gvar='gvar',
            ivar='id',
            tvar='year',
            controls=None,
            estimator='ra',
        )
        
        # Verify n_treated for each cohort is correct
        # Cohort 2003 has 3 treated units (ids 1-3)
        # Cohort 2004 has 3 treated units (ids 4-6)
        for eff in results:
            if eff.cohort == 2003:
                assert eff.n_treated == 3, f"Cohort 2003 should have 3 treated, got {eff.n_treated}"
            elif eff.cohort == 2004:
                assert eff.n_treated == 3, f"Cohort 2004 should have 3 treated, got {eff.n_treated}"


class TestBug115BoundaryConditions:
    """
    Boundary condition tests for BUG-115 float comparison fix.
    
    Tests edge cases at the tolerance boundary:
    - Values exactly at COHORT_FLOAT_TOLERANCE boundary
    - Values just inside/outside the tolerance
    """
    
    def test_tolerance_boundary_inside(self):
        """Values just inside tolerance should match."""
        # Create data where gvar differs by less than tolerance
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3],
            'year': [2000, 2001, 2000, 2001, 2000, 2001],
            'gvar': [
                2001 + COHORT_FLOAT_TOLERANCE * 0.9,  # unit 1: within tolerance
                2001 + COHORT_FLOAT_TOLERANCE * 0.9,
                0.0, 0.0,  # unit 2: never treated
                0.0, 0.0,  # unit 3: never treated
            ],
            'y': [1.0, 2.0, 1.0, 1.5, 1.0, 1.5],
        })
        
        counts = count_control_units_by_strategy(
            data, gvar='gvar', ivar='id',
            cohort=2001, period=2001
        )
        
        # Unit 1 should be counted as treatment cohort
        assert counts['treatment_cohort'] == 1
    
    def test_tolerance_boundary_outside(self):
        """Values outside tolerance should not match."""
        # Create data where gvar differs by more than tolerance
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3],
            'year': [2000, 2001, 2000, 2001, 2000, 2001],
            'gvar': [
                2001 + COHORT_FLOAT_TOLERANCE * 2,  # unit 1: outside tolerance
                2001 + COHORT_FLOAT_TOLERANCE * 2,
                0.0, 0.0,  # unit 2: never treated
                0.0, 0.0,  # unit 3: never treated
            ],
            'y': [1.0, 2.0, 1.0, 1.5, 1.0, 1.5],
        })
        
        counts = count_control_units_by_strategy(
            data, gvar='gvar', ivar='id',
            cohort=2001, period=2001
        )
        
        # Unit 1 should NOT be counted as treatment cohort 2001
        # (it's actually cohort 2001 + 2*tol ≈ 2001.000000002)
        assert counts['treatment_cohort'] == 0


class TestBug115NeverTreatedComparison:
    """
    Tests for never-treated unit identification with float tolerance.
    
    Never-treated units are identified by gvar values of:
    - 0 (with tolerance)
    - np.inf
    - NaN
    """
    
    def test_zero_gvar_with_tolerance(self):
        """Near-zero gvar values should be treated as never-treated."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'year': [2000, 2001, 2000, 2001],
            'gvar': [
                COHORT_FLOAT_TOLERANCE / 2,  # unit 1: near zero (never treated)
                COHORT_FLOAT_TOLERANCE / 2,
                2001.0, 2001.0,  # unit 2: treated in 2001
            ],
            'y': [1.0, 1.5, 1.0, 2.0],
        })
        
        counts = count_control_units_by_strategy(
            data, gvar='gvar', ivar='id',
            cohort=2001, period=2001
        )
        
        # Unit 1 should be counted as never treated
        assert counts['never_treated'] == 1
        assert counts['treatment_cohort'] == 1


# =============================================================================
# Integration Tests
# =============================================================================

class TestBug115Integration:
    """
    Integration tests to verify BUG-115 fix doesn't break existing functionality.
    """
    
    def test_integer_gvar_still_works(self):
        """Integer gvar values should continue to work correctly."""
        np.random.seed(123)
        
        data = pd.DataFrame({
            'id': np.repeat(range(1, 11), 5),
            'year': np.tile(range(2000, 2005), 10),
            'gvar': np.repeat([2002, 2002, 2003, 2003, 2003, 0, 0, 0, 0, 0], 5),
            'y': np.random.normal(10, 1, 50),
        })
        
        # Transform and estimate
        transformed = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar'
        )
        
        results = estimate_cohort_time_effects(
            data_transformed=transformed,
            gvar='gvar',
            ivar='id',
            tvar='year',
            controls=None,
            estimator='ra',
        )
        
        assert len(results) > 0
        
        # Verify cohort counts
        cohort_2002_effects = [e for e in results if e.cohort == 2002]
        cohort_2003_effects = [e for e in results if e.cohort == 2003]
        
        assert len(cohort_2002_effects) > 0
        assert len(cohort_2003_effects) > 0
        
        # Check n_treated values
        for eff in cohort_2002_effects:
            assert eff.n_treated == 2  # 2 units in cohort 2002
        
        for eff in cohort_2003_effects:
            assert eff.n_treated == 3  # 3 units in cohort 2003
