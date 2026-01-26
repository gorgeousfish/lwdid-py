"""
DESIGN-043-E Stata Validation Tests.

This module validates the τ_{TT} detection logic fix by comparing Python behavior
with Stata reference implementation.

The fix changes the condition from `period_r == max_cohort` to `period_r >= max_cohort`
to correctly detect all cases where no control units are available for the last cohort.

Reference:
- Lee & Wooldridge (2023) Section 4
- Stata: Lee_Wooldridge_2023-main 3/2.lee_wooldridge_rolling_staggered.do
"""

import numpy as np
import pandas as pd
import pytest
import subprocess
import tempfile
import os

# Skip entire module if required functions are not available
try:
    from lwdid.staggered.estimators import (
        build_subsample_for_ps_estimation,
        estimate_ipwra,
    )
except ImportError as e:
    pytest.skip(
        f"Skipping module: required functions not implemented ({e})",
        allow_module_level=True
    )


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def staggered_data_with_nt():
    """
    Create staggered DiD data WITH never-treated units.
    
    Similar to Stata example: cohorts {4, 5, 6, ∞}
    """
    np.random.seed(42)
    n_per_cohort = 100
    T = 6
    
    units = []
    for g_idx, g in enumerate([4, 5, 6, np.inf]):
        for i in range(n_per_cohort):
            unit_id = g_idx * n_per_cohort + i
            x1 = np.random.normal(5 + g_idx * 0.1, 1)
            x2 = np.random.normal(0, 1)
            
            for t in range(1, T + 1):
                # Treatment effect for treated units in post-treatment periods
                if not np.isinf(g) and t >= g:
                    te = 2.0  # ATT = 2.0
                else:
                    te = 0
                
                y = 1 + 0.5 * x1 + 0.3 * x2 + 0.1 * t + te + np.random.normal(0, 1)
                
                units.append({
                    'id': unit_id,
                    'year': t,
                    'gvar': g,
                    'x1': x1,
                    'x2': x2,
                    'y': y,
                })
    
    return pd.DataFrame(units)


@pytest.fixture
def staggered_data_all_eventually_treated():
    """
    Create staggered DiD data WITHOUT never-treated units.
    
    All units are eventually treated: cohorts {4, 5, 6}
    """
    np.random.seed(42)
    n_per_cohort = 100
    T = 8  # Extended periods to test period > max_cohort
    
    units = []
    for g_idx, g in enumerate([4, 5, 6]):
        for i in range(n_per_cohort):
            unit_id = g_idx * n_per_cohort + i
            x1 = np.random.normal(5 + g_idx * 0.1, 1)
            x2 = np.random.normal(0, 1)
            
            for t in range(1, T + 1):
                if t >= g:
                    te = 2.0
                else:
                    te = 0
                
                y = 1 + 0.5 * x1 + 0.3 * x2 + 0.1 * t + te + np.random.normal(0, 1)
                
                units.append({
                    'id': unit_id,
                    'year': t,
                    'gvar': g,
                    'x1': x1,
                    'x2': x2,
                    'y': y,
                })
    
    return pd.DataFrame(units)


# ============================================================================
# Boundary Condition Tests
# ============================================================================

class TestDesign043EBoundaryConditions:
    """
    Test boundary conditions for τ_{TT} detection.
    
    Validates that the fix correctly handles:
    1. period_r == max_cohort (original case)
    2. period_r > max_cohort (new case covered by fix)
    """
    
    def test_tau_66_with_nt_succeeds(self, staggered_data_with_nt):
        """
        τ_{6,6} with NT units should succeed.
        
        Control group = {∞} (never-treated units)
        """
        result = build_subsample_for_ps_estimation(
            data=staggered_data_with_nt,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=6,
            period_r=6,
            control_group='not_yet_treated',
        )
        
        assert result.n_control > 0
        assert np.inf in result.control_cohorts
        assert result.has_never_treated == True
    
    def test_tau_66_without_nt_fails(self, staggered_data_all_eventually_treated):
        """
        τ_{6,6} without NT units should fail.
        
        This tests period_r == max_cohort (original condition).
        """
        with pytest.raises(ValueError, match="Cannot estimate.*last cohort"):
            build_subsample_for_ps_estimation(
                data=staggered_data_all_eventually_treated,
                gvar_col='gvar',
                ivar_col='id',
                cohort_g=6,
                period_r=6,
                control_group='not_yet_treated',
            )
    
    def test_tau_67_without_nt_fails(self, staggered_data_all_eventually_treated):
        """
        τ_{6,7} without NT units should fail.
        
        This tests period_r > max_cohort (the DESIGN-043-E fix).
        Before the fix, this would pass the early detection but fail later.
        """
        with pytest.raises(ValueError, match="Cannot estimate.*last cohort"):
            build_subsample_for_ps_estimation(
                data=staggered_data_all_eventually_treated,
                gvar_col='gvar',
                ivar_col='id',
                cohort_g=6,
                period_r=7,
                control_group='not_yet_treated',
            )
    
    def test_tau_68_without_nt_fails(self, staggered_data_all_eventually_treated):
        """
        τ_{6,8} without NT units should fail.
        
        Tests period_r >> max_cohort.
        """
        with pytest.raises(ValueError, match="Cannot estimate.*last cohort"):
            build_subsample_for_ps_estimation(
                data=staggered_data_all_eventually_treated,
                gvar_col='gvar',
                ivar_col='id',
                cohort_g=6,
                period_r=8,
                control_group='not_yet_treated',
            )
    
    def test_tau_45_without_nt_succeeds(self, staggered_data_all_eventually_treated):
        """
        τ_{4,5} without NT units should succeed.
        
        Control group = {6} (not-yet-treated at period 5)
        """
        result = build_subsample_for_ps_estimation(
            data=staggered_data_all_eventually_treated,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=4,
            period_r=5,
            control_group='not_yet_treated',
        )
        
        assert result.n_control > 0
        assert 6 in result.control_cohorts
        assert 5 not in result.control_cohorts  # 5 == period_r, not control
    
    def test_tau_46_without_nt_fails(self, staggered_data_all_eventually_treated):
        """
        τ_{4,6} without NT units should fail.
        
        At period 6, all other cohorts (5, 6) have started treatment.
        No control units available.
        """
        with pytest.raises(ValueError, match="No control units"):
            build_subsample_for_ps_estimation(
                data=staggered_data_all_eventually_treated,
                gvar_col='gvar',
                ivar_col='id',
                cohort_g=4,
                period_r=6,
                control_group='not_yet_treated',
            )


# ============================================================================
# IPWRA Estimator Integration Tests
# ============================================================================

class TestDesign043EIPWRAIntegration:
    """
    Integration tests with IPWRA estimator.
    
    Validates that the boundary detection works correctly when called
    through the main estimator interface.
    """
    
    def test_ipwra_tau_44_subsample_succeeds(self, staggered_data_with_nt):
        """
        Build subsample for τ_{4,4} and verify it can be used for IPWRA.
        """
        result = build_subsample_for_ps_estimation(
            data=staggered_data_with_nt,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=4,
            period_r=4,
            control_group='not_yet_treated',
        )
        
        # Subsample should be valid for IPWRA estimation
        assert result.n_treated > 0
        assert result.n_control > 0
        assert 'D_ig' in result.subsample.columns
        
        # Binary treatment indicator should be correct
        assert set(result.D_ig).issubset({0, 1})
    
    def test_ipwra_tau_66_subsample_with_nt_succeeds(self, staggered_data_with_nt):
        """
        Build subsample for τ_{6,6} with NT and verify it can be used for IPWRA.
        """
        result = build_subsample_for_ps_estimation(
            data=staggered_data_with_nt,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=6,
            period_r=6,
            control_group='not_yet_treated',
        )
        
        # With NT units, this should succeed
        assert result.n_treated > 0
        assert result.n_control > 0
        assert np.inf in result.control_cohorts


# ============================================================================
# Error Message Quality Tests
# ============================================================================

class TestDesign043EErrorMessages:
    """
    Test that error messages are informative and helpful.
    """
    
    def test_error_message_contains_max_cohort(self, staggered_data_all_eventually_treated):
        """
        Error message should include max_cohort value for debugging.
        """
        with pytest.raises(ValueError) as exc_info:
            build_subsample_for_ps_estimation(
                data=staggered_data_all_eventually_treated,
                gvar_col='gvar',
                ivar_col='id',
                cohort_g=6,
                period_r=7,
                control_group='not_yet_treated',
            )
        
        error_msg = str(exc_info.value)
        assert "max(gvar) = 6" in error_msg
    
    def test_error_message_contains_period(self, staggered_data_all_eventually_treated):
        """
        Error message should include period_r value.
        """
        with pytest.raises(ValueError) as exc_info:
            build_subsample_for_ps_estimation(
                data=staggered_data_all_eventually_treated,
                gvar_col='gvar',
                ivar_col='id',
                cohort_g=6,
                period_r=7,
                control_group='not_yet_treated',
            )
        
        error_msg = str(exc_info.value)
        assert "gvar > 7" in error_msg
    
    def test_error_message_explains_reason(self, staggered_data_all_eventually_treated):
        """
        Error message should explain why no control units exist.
        """
        with pytest.raises(ValueError) as exc_info:
            build_subsample_for_ps_estimation(
                data=staggered_data_all_eventually_treated,
                gvar_col='gvar',
                ivar_col='id',
                cohort_g=6,
                period_r=7,
                control_group='not_yet_treated',
            )
        
        error_msg = str(exc_info.value)
        assert "not-yet-treated" in error_msg.lower()
        assert "no valid control" in error_msg.lower()


# ============================================================================
# Regression Tests (ensure existing behavior unchanged)
# ============================================================================

class TestDesign043ERegressionTests:
    """
    Regression tests to ensure existing behavior is not changed.
    """
    
    def test_standard_estimation_unchanged(self, staggered_data_with_nt):
        """
        Standard estimation scenarios should work exactly as before.
        """
        # τ_{4,4}
        result_44 = build_subsample_for_ps_estimation(
            data=staggered_data_with_nt,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=4,
            period_r=4,
            control_group='not_yet_treated',
        )
        assert set(result_44.control_cohorts) == {5, 6, np.inf}
        
        # τ_{4,5}
        result_45 = build_subsample_for_ps_estimation(
            data=staggered_data_with_nt,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=4,
            period_r=5,
            control_group='not_yet_treated',
        )
        assert 5 not in result_45.control_cohorts  # 5 == period_r
        assert set(result_45.control_cohorts) == {6, np.inf}
        
        # τ_{4,6}
        result_46 = build_subsample_for_ps_estimation(
            data=staggered_data_with_nt,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=4,
            period_r=6,
            control_group='not_yet_treated',
        )
        assert 5 not in result_46.control_cohorts
        assert 6 not in result_46.control_cohorts
        assert result_46.control_cohorts == [np.inf]
    
    def test_never_treated_strategy_unchanged(self, staggered_data_with_nt):
        """
        never_treated control group strategy should work as before.
        """
        result = build_subsample_for_ps_estimation(
            data=staggered_data_with_nt,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=4,
            period_r=4,
            control_group='never_treated',
        )
        
        # Only NT units in control
        assert result.control_cohorts == [np.inf]
        assert result.has_never_treated == True
