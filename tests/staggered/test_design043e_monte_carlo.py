"""
DESIGN-043-E Monte Carlo Simulation Tests.

This module validates the τ_{TT} detection logic fix through Monte Carlo simulations.

The simulations verify that:
1. The boundary detection correctly identifies unestimable scenarios
2. Normal estimation scenarios continue to work correctly
3. The fix handles various data configurations properly

Reference:
- Lee & Wooldridge (2023) Section 4
"""

import numpy as np
import pandas as pd
import pytest
from typing import Tuple

# Skip entire module if required functions are not available
try:
    from lwdid.staggered.estimators import build_subsample_for_ps_estimation
except ImportError as e:
    pytest.skip(
        f"Skipping module: required functions not implemented ({e})",
        allow_module_level=True
    )


# ============================================================================
# Data Generation Functions
# ============================================================================

def generate_staggered_panel(
    n_units: int,
    n_periods: int,
    cohorts: list,
    include_never_treated: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a staggered DiD panel dataset.
    
    Parameters
    ----------
    n_units : int
        Number of units per cohort
    n_periods : int
        Number of time periods
    cohorts : list
        List of treatment start periods (excluding inf)
    include_never_treated : bool
        Whether to include never-treated units
    seed : int
        Random seed
        
    Returns
    -------
    pd.DataFrame
        Panel dataset with columns: id, year, gvar, x1, x2, y
    """
    np.random.seed(seed)
    
    all_cohorts = cohorts.copy()
    if include_never_treated:
        all_cohorts.append(np.inf)
    
    units = []
    for g_idx, g in enumerate(all_cohorts):
        for i in range(n_units):
            unit_id = g_idx * n_units + i
            x1 = np.random.normal(5 + g_idx * 0.1, 1)
            x2 = np.random.normal(0, 1)
            
            for t in range(1, n_periods + 1):
                if not np.isinf(g) and t >= g:
                    te = 2.0  # True ATT
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
# Monte Carlo Simulation Tests
# ============================================================================

class TestDesign043EMonteCarlo:
    """
    Monte Carlo simulation tests for DESIGN-043-E fix.
    """
    
    @pytest.mark.parametrize("seed", range(10))
    def test_boundary_detection_multiple_seeds(self, seed):
        """
        Test boundary detection with multiple random seeds.
        
        Ensures the fix works regardless of random data generation.
        """
        # Generate data without NT (all eventually treated)
        df = generate_staggered_panel(
            n_units=50,
            n_periods=8,
            cohorts=[4, 5, 6],
            include_never_treated=False,
            seed=seed,
        )
        
        # τ_{6,6} should fail (period == max_cohort)
        with pytest.raises(ValueError, match="Cannot estimate.*last cohort"):
            build_subsample_for_ps_estimation(
                data=df,
                gvar_col='gvar',
                ivar_col='id',
                cohort_g=6,
                period_r=6,
                control_group='not_yet_treated',
            )
        
        # τ_{6,7} should also fail (period > max_cohort) - THE FIX
        with pytest.raises(ValueError, match="Cannot estimate.*last cohort"):
            build_subsample_for_ps_estimation(
                data=df,
                gvar_col='gvar',
                ivar_col='id',
                cohort_g=6,
                period_r=7,
                control_group='not_yet_treated',
            )
    
    @pytest.mark.parametrize("seed", range(10))
    def test_valid_estimation_multiple_seeds(self, seed):
        """
        Test that valid estimations succeed with multiple random seeds.
        """
        df = generate_staggered_panel(
            n_units=50,
            n_periods=8,
            cohorts=[4, 5, 6],
            include_never_treated=True,
            seed=seed,
        )
        
        # τ_{4,4} should succeed
        result = build_subsample_for_ps_estimation(
            data=df,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=4,
            period_r=4,
            control_group='not_yet_treated',
        )
        assert result.n_control > 0
        
        # τ_{6,6} should succeed with NT units
        result = build_subsample_for_ps_estimation(
            data=df,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=6,
            period_r=6,
            control_group='not_yet_treated',
        )
        assert result.n_control > 0
    
    @pytest.mark.parametrize("max_cohort,period_r", [
        (6, 6),   # period == max_cohort
        (6, 7),   # period > max_cohort (THE FIX)
        (6, 8),   # period >> max_cohort
        (6, 10),  # period >>> max_cohort
        (10, 10), # Different max_cohort, period == max_cohort
        (10, 11), # Different max_cohort, period > max_cohort
    ])
    def test_boundary_parametric(self, max_cohort, period_r):
        """
        Parametric test for various boundary conditions.
        """
        # Generate data with cohorts up to max_cohort
        cohorts = list(range(4, max_cohort + 1))
        
        df = generate_staggered_panel(
            n_units=30,
            n_periods=period_r + 2,
            cohorts=cohorts,
            include_never_treated=False,
            seed=42,
        )
        
        # Last cohort at or after max_cohort should fail
        with pytest.raises(ValueError, match="Cannot estimate.*last cohort"):
            build_subsample_for_ps_estimation(
                data=df,
                gvar_col='gvar',
                ivar_col='id',
                cohort_g=max_cohort,
                period_r=period_r,
                control_group='not_yet_treated',
            )
    
    def test_simulation_sample_sizes(self):
        """
        Test with various sample sizes.
        """
        for n_units in [10, 50, 100, 200]:
            df = generate_staggered_panel(
                n_units=n_units,
                n_periods=8,
                cohorts=[4, 5, 6],
                include_never_treated=False,
                seed=42,
            )
            
            # Should always detect the boundary condition
            with pytest.raises(ValueError, match="Cannot estimate.*last cohort"):
                build_subsample_for_ps_estimation(
                    data=df,
                    gvar_col='gvar',
                    ivar_col='id',
                    cohort_g=6,
                    period_r=7,
                    control_group='not_yet_treated',
                )
    
    def test_simulation_cohort_configurations(self):
        """
        Test with various cohort configurations.
        """
        # Different number of cohorts
        for n_cohorts in [2, 3, 4, 5]:
            cohorts = list(range(4, 4 + n_cohorts))
            max_cohort = max(cohorts)
            
            df = generate_staggered_panel(
                n_units=30,
                n_periods=max_cohort + 2,
                cohorts=cohorts,
                include_never_treated=False,
                seed=42,
            )
            
            # Last cohort at period > max_cohort should fail
            with pytest.raises(ValueError, match="Cannot estimate.*last cohort"):
                build_subsample_for_ps_estimation(
                    data=df,
                    gvar_col='gvar',
                    ivar_col='id',
                    cohort_g=max_cohort,
                    period_r=max_cohort + 1,
                    control_group='not_yet_treated',
                )


# ============================================================================
# Consistency Tests
# ============================================================================

class TestDesign043EConsistency:
    """
    Test consistency of boundary detection across different scenarios.
    """
    
    def test_consistency_with_and_without_nt(self):
        """
        Test that behavior is consistent whether NT units exist or not.
        """
        np.random.seed(42)
        
        # With NT units
        df_with_nt = generate_staggered_panel(
            n_units=50,
            n_periods=8,
            cohorts=[4, 5, 6],
            include_never_treated=True,
            seed=42,
        )
        
        # Without NT units
        df_without_nt = generate_staggered_panel(
            n_units=50,
            n_periods=8,
            cohorts=[4, 5, 6],
            include_never_treated=False,
            seed=42,
        )
        
        # τ_{4,5} should succeed in both cases
        result_with = build_subsample_for_ps_estimation(
            data=df_with_nt,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=4,
            period_r=5,
            control_group='not_yet_treated',
        )
        
        result_without = build_subsample_for_ps_estimation(
            data=df_without_nt,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=4,
            period_r=5,
            control_group='not_yet_treated',
        )
        
        assert result_with.n_control > 0
        assert result_without.n_control > 0
        
        # Control cohorts should differ
        assert np.inf in result_with.control_cohorts
        assert np.inf not in result_without.control_cohorts
    
    def test_consistency_control_group_strategies(self):
        """
        Test that boundary detection works for both control group strategies.
        """
        df = generate_staggered_panel(
            n_units=50,
            n_periods=8,
            cohorts=[4, 5, 6],
            include_never_treated=True,
            seed=42,
        )
        
        # not_yet_treated strategy
        result_nyt = build_subsample_for_ps_estimation(
            data=df,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=4,
            period_r=4,
            control_group='not_yet_treated',
        )
        
        # never_treated strategy
        result_nt = build_subsample_for_ps_estimation(
            data=df,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=4,
            period_r=4,
            control_group='never_treated',
        )
        
        # Both should succeed with NT units
        assert result_nyt.n_control > 0
        assert result_nt.n_control > 0
        
        # But control cohorts differ
        assert len(result_nyt.control_cohorts) > len(result_nt.control_cohorts)


# ============================================================================
# Performance/Stress Tests
# ============================================================================

class TestDesign043EPerformance:
    """
    Performance tests to ensure the fix doesn't introduce overhead.
    """
    
    def test_large_scale_boundary_detection(self):
        """
        Test boundary detection with large dataset.
        """
        df = generate_staggered_panel(
            n_units=500,
            n_periods=12,
            cohorts=[4, 5, 6, 7, 8, 9, 10],
            include_never_treated=False,
            seed=42,
        )
        
        # Should quickly detect the boundary condition
        import time
        start = time.time()
        
        with pytest.raises(ValueError, match="Cannot estimate.*last cohort"):
            build_subsample_for_ps_estimation(
                data=df,
                gvar_col='gvar',
                ivar_col='id',
                cohort_g=10,
                period_r=11,
                control_group='not_yet_treated',
            )
        
        elapsed = time.time() - start
        
        # Should complete in reasonable time (< 1 second)
        assert elapsed < 1.0, f"Boundary detection took {elapsed:.2f}s, expected < 1s"
    
    def test_repeated_boundary_checks(self):
        """
        Test that repeated boundary checks are efficient.
        """
        df = generate_staggered_panel(
            n_units=100,
            n_periods=10,
            cohorts=[4, 5, 6],
            include_never_treated=False,
            seed=42,
        )
        
        import time
        start = time.time()
        
        # Repeat boundary check many times
        for _ in range(100):
            try:
                build_subsample_for_ps_estimation(
                    data=df,
                    gvar_col='gvar',
                    ivar_col='id',
                    cohort_g=6,
                    period_r=7,
                    control_group='not_yet_treated',
                )
            except ValueError:
                pass  # Expected
        
        elapsed = time.time() - start
        
        # 100 checks should complete in < 5 seconds
        assert elapsed < 5.0, f"100 boundary checks took {elapsed:.2f}s, expected < 5s"
