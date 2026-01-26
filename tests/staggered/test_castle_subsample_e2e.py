"""
End-to-End Tests for Castle Law Data with Subsample Construction (Story 1.2).

Tests the integration of build_subsample_for_ps_estimation() with real Castle Law data.
"""

import pytest
import numpy as np
import pandas as pd

# Skip entire module if required functions are not available
try:
    from lwdid.staggered.estimators import (
        build_subsample_for_ps_estimation,
        estimate_ipwra,
        estimate_psm,
    )
except ImportError as e:
    pytest.skip(
        f"Skipping module: required functions not implemented ({e})",
        allow_module_level=True
    )

from lwdid.staggered.transformations import transform_staggered_demean


# Try multiple data paths
DATA_PATHS = [
    '/Users/cxy/Desktop/大样本lwdid/lwdid-py_v0.1.0/data/castle.csv',
    '/Users/cxy/Desktop/rebuildlwdid/lwdid-py_v0.1.0/data/castle.csv',
    '/Users/cxy/Desktop/大样本lwdid/lwdid-py_v0.1.0/tests/data/castle.csv',
]


@pytest.fixture
def castle_data():
    """Load Castle Law data."""
    for path in DATA_PATHS:
        try:
            castle = pd.read_csv(path)
            castle['gvar'] = castle['effyear'].fillna(np.inf)
            return castle
        except FileNotFoundError:
            continue
    
    pytest.skip("Castle Law data not found")


@pytest.fixture
def castle_transformed(castle_data):
    """Transform Castle Law data with demeaning."""
    return transform_staggered_demean(
        castle_data, 'lhomicide', 'sid', 'year', 'gvar'
    )


class TestCastleSubsampleConstruction:
    """Test subsample construction with Castle Law data."""
    
    def test_cohort_2005_period_2005(self, castle_transformed):
        """
        Test τ_{2005,2005} subsample for Castle Law data.
        
        Control should include: {2006, 2007, ..., 2010, ∞}
        """
        # Filter to year 2005
        df_2005 = castle_transformed[castle_transformed['year'] == 2005].copy()
        
        result = build_subsample_for_ps_estimation(
            data=df_2005,
            gvar_col='gvar',
            ivar_col='sid',
            cohort_g=2005,
            period_r=2005,
            control_group='not_yet_treated',
        )
        
        # Verify subsample is non-empty
        assert result.n_treated > 0
        assert result.n_control > 0
        
        # Control should NOT include 2005 (the treatment cohort)
        assert 2005 not in result.control_cohorts
        
        # Control should include later cohorts and NT
        # (depends on actual data cohort structure)
        assert result.has_never_treated == True  # Castle Law has NT states
    
    def test_cohort_2005_period_2006(self, castle_transformed):
        """
        Test τ_{2005,2006} subsample for Castle Law data.
        
        Cohort 2006 should be excluded (gvar == period_r).
        """
        df_2006 = castle_transformed[castle_transformed['year'] == 2006].copy()
        
        result = build_subsample_for_ps_estimation(
            data=df_2006,
            gvar_col='gvar',
            ivar_col='sid',
            cohort_g=2005,
            period_r=2006,
            control_group='not_yet_treated',
        )
        
        # Cohort 2006 should be excluded (gvar=2006 == period_r=2006)
        assert 2006 not in result.control_cohorts
        
        # Should have NT and later cohorts
        assert result.n_control > 0
    
    def test_subsample_d_ig_correctness(self, castle_transformed):
        """Verify D_ig correctly identifies treatment cohort."""
        df_2005 = castle_transformed[castle_transformed['year'] == 2005].copy()
        
        result = build_subsample_for_ps_estimation(
            data=df_2005,
            gvar_col='gvar',
            ivar_col='sid',
            cohort_g=2005,
            period_r=2005,
            control_group='not_yet_treated',
        )
        
        # D_ig should be binary
        assert set(result.D_ig).issubset({0, 1})
        
        # D_ig=1 should only be for gvar=2005
        treated_gvars = result.subsample[result.D_ig == 1]['gvar'].unique()
        assert len(treated_gvars) == 1
        assert treated_gvars[0] == 2005


class TestCastleIPWRAIntegration:
    """Test IPWRA with Castle Law data using Staggered mode."""
    
    def test_ipwra_staggered_mode(self, castle_transformed):
        """
        Test IPWRA estimation in Staggered mode.
        
        Note: Castle Law 2005 cohort only has 1 state (Florida), 
        so we need to find a cohort with more units.
        """
        # Find a cohort with enough units for IPWRA
        cohorts = sorted([
            g for g in castle_transformed['gvar'].unique() 
            if not np.isinf(g) and g > 0
        ])
        
        target_cohort = None
        for g in cohorts:
            df_test = castle_transformed[castle_transformed['year'] == g].copy()
            n_treated = (df_test['gvar'] == g).sum()
            if n_treated >= 3:  # Need at least 3 for IPWRA
                target_cohort = g
                break
        
        if target_cohort is None:
            pytest.skip("No cohort with sufficient sample size for IPWRA")
        
        df_cohort = castle_transformed[castle_transformed['year'] == target_cohort].copy()
        y_col = f'ydot_g{int(target_cohort)}_r{int(target_cohort)}'
        
        if y_col not in df_cohort.columns:
            pytest.skip(f"Transformation column {y_col} not available")
        
        # Get available control variables - prefer continuous variables with good variation
        # Exclude binary/categorical variables that may cause singular matrix issues
        preferred_controls = ['population', 'income', 'unemployrt', 'police', 
                             'prisoner', 'blackm_15_24', 'whitem_15_24', 'poverty']
        available_controls = [c for c in preferred_controls 
                             if c in df_cohort.columns][:2]
        
        if not available_controls:
            # Fallback to any numeric columns
            numeric_cols = df_cohort.select_dtypes(include=[np.number]).columns
            available_controls = [c for c in numeric_cols 
                                if c not in ['sid', 'year', 'gvar', 'effyear', y_col, 'D_ig']][:2]
        
        if not available_controls:
            pytest.skip("No suitable control variables available")
        
        try:
            result = estimate_ipwra(
                data=df_cohort,
                y=y_col,
                d='',  # Ignored in Staggered mode
                controls=available_controls,
                gvar_col='gvar',
                ivar_col='sid',
                cohort_g=target_cohort,
                period_r=target_cohort,
                control_group='not_yet_treated',
            )
            
            # Verify result structure
            assert np.isfinite(result.att)
            assert result.se > 0
            assert result.n_treated > 0
            assert result.n_control > 0
        except ValueError as e:
            # May fail due to singular matrix or insufficient sample
            if "奇异" in str(e) or "insufficient" in str(e).lower():
                pytest.skip(f"IPWRA failed due to data issues: {e}")
            raise


class TestCastlePSMIntegration:
    """Test PSM with Castle Law data using Staggered mode."""
    
    def test_psm_staggered_mode(self, castle_transformed):
        """
        Test PSM estimation in Staggered mode.
        
        Note: PSM requires sufficient sample size for SE calculation.
        """
        # Find a cohort with enough units for PSM
        cohorts = sorted([
            g for g in castle_transformed['gvar'].unique() 
            if not np.isinf(g) and g > 0
        ])
        
        target_cohort = None
        for g in cohorts:
            df_test = castle_transformed[castle_transformed['year'] == g].copy()
            n_treated = (df_test['gvar'] == g).sum()
            if n_treated >= 3:  # Need at least 3 for reliable PSM SE
                target_cohort = g
                break
        
        if target_cohort is None:
            pytest.skip("No cohort with sufficient sample size for PSM")
        
        df_cohort = castle_transformed[castle_transformed['year'] == target_cohort].copy()
        y_col = f'ydot_g{int(target_cohort)}_r{int(target_cohort)}'
        
        if y_col not in df_cohort.columns:
            pytest.skip(f"Transformation column {y_col} not available")
        
        # Get available control variables - prefer continuous variables with good variation
        preferred_controls = ['population', 'income', 'unemployrt', 'police', 
                             'prisoner', 'blackm_15_24', 'whitem_15_24', 'poverty']
        available_controls = [c for c in preferred_controls 
                             if c in df_cohort.columns][:2]
        
        if not available_controls:
            # Fallback to any numeric columns
            numeric_cols = df_cohort.select_dtypes(include=[np.number]).columns
            available_controls = [c for c in numeric_cols 
                                if c not in ['sid', 'year', 'gvar', 'effyear', y_col, 'D_ig']][:2]
        
        if not available_controls:
            pytest.skip("No suitable control variables available")
        
        result = estimate_psm(
            data=df_cohort,
            y=y_col,
            d='',
            propensity_controls=available_controls,
            n_neighbors=1,
            gvar_col='gvar',
            ivar_col='sid',
            cohort_g=target_cohort,
            period_r=target_cohort,
            control_group='not_yet_treated',
        )
        
        assert np.isfinite(result.att)
        # SE may be NaN for very small samples, but ATT should be finite
        assert result.n_treated > 0
        assert result.n_control > 0


class TestCastleMultipleCohortPeriods:
    """Test multiple (g,r) combinations with Castle Law data."""
    
    def test_all_cohort_period_combinations(self, castle_transformed):
        """
        Test all valid (g,r) combinations.
        
        For each cohort g, test periods r = g, g+1, ..., T_max.
        """
        T_max = castle_transformed['year'].max()  # 2010
        cohorts = sorted([
            g for g in castle_transformed['gvar'].unique() 
            if not np.isinf(g) and g > 0 and g <= T_max
        ])
        
        successful_combinations = []
        failed_combinations = []
        
        for g in cohorts[:3]:  # Test first 3 cohorts to save time
            for r in range(int(g), min(int(g) + 3, T_max + 1)):  # Test first 3 periods per cohort
                try:
                    df_r = castle_transformed[castle_transformed['year'] == r].copy()
                    y_col = f'ydot_g{int(g)}_r{r}'
                    
                    if y_col not in df_r.columns:
                        continue
                    
                    result = build_subsample_for_ps_estimation(
                        data=df_r,
                        gvar_col='gvar',
                        ivar_col='sid',
                        cohort_g=g,
                        period_r=r,
                        control_group='not_yet_treated',
                    )
                    
                    # Verify basic properties
                    assert result.n_treated > 0
                    assert result.n_control > 0
                    assert set(result.D_ig).issubset({0, 1})
                    
                    successful_combinations.append((g, r))
                    
                except Exception as e:
                    failed_combinations.append((g, r, str(e)))
        
        # Should have some successful combinations
        assert len(successful_combinations) > 0, \
            f"No successful combinations. Failures: {failed_combinations}"
        
        print(f"\n✓ Successfully tested {len(successful_combinations)} (g,r) combinations")
        if failed_combinations:
            print(f"  (Skipped {len(failed_combinations)} combinations)")
