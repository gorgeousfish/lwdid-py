"""
Tests for the staggered (g,r)-specific effect estimation module.

Validates basic (g,r) effect estimation, sample construction logic,
control group exclusion, standard error types, edge cases, and the
Castle Law end-to-end pipeline.

Validates Section 4 (cohort-period estimation) of the Lee-Wooldridge
Difference-in-Differences framework for staggered adoption designs.

References
----------
Lee, S. & Wooldridge, J. M. (2025). A Simple Transformation Approach to
    Difference-in-Differences Estimation for Panel Data. SSRN 4516518.
Lee, S. & Wooldridge, J. M. (2026). Simple Approaches to Inference with
    DiD Estimators with Small Cross-Sectional Sample Sizes. SSRN 5325686.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from lwdid.staggered import (
    transform_staggered_demean,
    estimate_cohort_time_effects,
    CohortTimeEffect,
    results_to_dataframe,
    ControlGroupStrategy,
    get_valid_control_units,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def simple_test_data():
    """Simple test data with T=4, cohorts={3,4}, and never treated."""
    return pd.DataFrame({
        'id': [1,1,1,1, 2,2,2,2, 3,3,3,3],
        'year': [1,2,3,4, 1,2,3,4, 1,2,3,4],
        'y': [10.0, 12.0, 14.0, 20.0,  # unit 1: cohort 3
              15.0, 16.0, 17.0, 18.0,  # unit 2: cohort 4
              5.0, 6.0, 7.0, 8.0],     # unit 3: never treated
        'gvar': [3,3,3,3, 4,4,4,4, 0,0,0,0]
    })


@pytest.fixture
def multi_cohort_data():
    """Data with multiple cohorts to test control group exclusion."""
    return pd.DataFrame({
        'id': [1]*5 + [2]*5 + [3]*5 + [4]*5,
        'year': [1,2,3,4,5]*4,
        'y': [10.0]*5 + [20.0]*5 + [30.0]*5 + [40.0]*5,
        'gvar': [4]*5 + [5]*5 + [6]*5 + [0]*5  # cohorts 4,5,6 + NT
    })


@pytest.fixture
def castle_data():
    """Load castle.csv if available."""
    data_path = Path(__file__).parent.parent.parent / 'data' / 'castle.csv'
    if data_path.exists():
        data = pd.read_csv(data_path)
        data['gvar'] = data['effyear'].fillna(0).astype(int)
        return data
    return None


# ============================================================================
# Basic Functionality Tests
# ============================================================================

class TestCohortTimeEffectBasic:
    """Test basic (g,r) effect estimation."""
    
    def test_estimate_returns_list(self, simple_test_data):
        """Verify estimate returns a list of CohortTimeEffect."""
        transformed = transform_staggered_demean(
            simple_test_data, 'y', 'id', 'year', 'gvar'
        )
        results = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year',
            control_strategy='not_yet_treated'
        )
        
        assert isinstance(results, list)
        assert all(isinstance(r, CohortTimeEffect) for r in results)
    
    def test_result_structure(self, simple_test_data):
        """Verify CohortTimeEffect has all required fields."""
        transformed = transform_staggered_demean(
            simple_test_data, 'y', 'id', 'year', 'gvar'
        )
        results = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year'
        )
        
        assert len(results) > 0
        r = results[0]
        
        # Check all fields exist
        assert hasattr(r, 'cohort')
        assert hasattr(r, 'period')
        assert hasattr(r, 'event_time')
        assert hasattr(r, 'att')
        assert hasattr(r, 'se')
        assert hasattr(r, 'ci_lower')
        assert hasattr(r, 'ci_upper')
        assert hasattr(r, 't_stat')
        assert hasattr(r, 'pvalue')
        assert hasattr(r, 'n_treated')
        assert hasattr(r, 'n_control')
        assert hasattr(r, 'n_total')
    
    def test_event_time_calculation(self, simple_test_data):
        """Verify event_time = period - cohort."""
        transformed = transform_staggered_demean(
            simple_test_data, 'y', 'id', 'year', 'gvar'
        )
        results = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year'
        )
        
        for r in results:
            expected_et = r.period - r.cohort
            assert r.event_time == expected_et, \
                f"Event time wrong: cohort={r.cohort}, period={r.period}, " \
                f"expected={expected_et}, got={r.event_time}"
            assert r.event_time >= 0, "Event time should not be negative"
    
    def test_sample_counts_consistency(self, simple_test_data):
        """Verify n_total = n_treated + n_control."""
        transformed = transform_staggered_demean(
            simple_test_data, 'y', 'id', 'year', 'gvar'
        )
        results = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year'
        )
        
        for r in results:
            assert r.n_total == r.n_treated + r.n_control, \
                f"Sample counts inconsistent: {r.n_total} != {r.n_treated} + {r.n_control}"


# ============================================================================
# Sample Construction Tests
# ============================================================================

class TestSampleConstruction:
    """Test correct sample construction for estimation."""
    
    def test_control_group_exclusion(self, multi_cohort_data):
        """Test that gvar==period units are excluded from controls.
        
        For τ_{4,5}:
        - Include: cohort 4 (treatment) + cohort 6 (gvar=6>5) + NT
        - Exclude: cohort 5 (gvar=5==period=5, starts treatment in period 5)
        """
        transformed = transform_staggered_demean(
            multi_cohort_data, 'y', 'id', 'year', 'gvar'
        )
        results = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year',
            control_strategy='not_yet_treated'
        )
        
        # Find τ_{4,5}
        tau_4_5 = next((r for r in results if r.cohort == 4 and r.period == 5), None)
        
        if tau_4_5 is not None:
            # n_treated = 1 (cohort 4)
            # n_control = 2 (cohort 6 + NT, cohort 5 excluded)
            assert tau_4_5.n_treated == 1
            assert tau_4_5.n_control == 2, \
                f"Expected 2 controls (cohort 6 + NT), got {tau_4_5.n_control}. " \
                "Cohort 5 should be excluded."
    
    def test_gvar_equals_period_not_control(self):
        """Verify gvar==period units are NOT in control group."""
        data = pd.DataFrame({
            'id': [1, 2, 3],
            'year': [5, 5, 5],
            'gvar': [4, 5, 6]  # cohort 4 (treated), 5 (starts now), 6 (not yet)
        })
        
        # For cohort=4, period=5, controls should be gvar>5 only
        control_mask = get_valid_control_units(
            data, 'gvar', 'id', cohort=4, period=5,
            strategy=ControlGroupStrategy.NOT_YET_TREATED
        )
        
        # id=1 (gvar=4): treatment group, not control
        # id=2 (gvar=5==5): starts treatment in period 5, NOT control
        # id=3 (gvar=6>5): not yet treated, IS control
        assert control_mask.loc[1] == False  # treatment
        assert control_mask.loc[2] == False  # gvar==period, excluded!
        assert control_mask.loc[3] == True   # gvar > period, control
    
    def test_never_treated_always_control(self, simple_test_data):
        """Verify never treated units are always valid controls."""
        transformed = transform_staggered_demean(
            simple_test_data, 'y', 'id', 'year', 'gvar'
        )
        results = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year',
            control_strategy='not_yet_treated'
        )
        
        # Unit 3 (gvar=0) should always be in control group
        for r in results:
            assert r.n_control >= 1, \
                f"Expected at least 1 control (NT unit), got {r.n_control}"


# ============================================================================
# Standard Error Tests
# ============================================================================

class TestStandardErrors:
    """Test different standard error types."""
    
    def test_homoskedastic_se(self, simple_test_data):
        """Test homoskedastic standard errors (vce=None)."""
        transformed = transform_staggered_demean(
            simple_test_data, 'y', 'id', 'year', 'gvar'
        )
        results = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year',
            vce=None
        )
        
        for r in results:
            assert r.se > 0
            assert not np.isnan(r.se)
    
    def test_hc3_se(self, simple_test_data):
        """Test HC3 robust standard errors."""
        transformed = transform_staggered_demean(
            simple_test_data, 'y', 'id', 'year', 'gvar'
        )
        results = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year',
            vce='hc3'
        )
        
        for r in results:
            assert r.se > 0
            assert not np.isnan(r.se)
    
    def test_att_same_across_vce_types(self, simple_test_data):
        """ATT point estimate should be same regardless of VCE type."""
        transformed = transform_staggered_demean(
            simple_test_data, 'y', 'id', 'year', 'gvar'
        )
        
        results_homo = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year', vce=None
        )
        results_hc3 = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year', vce='hc3'
        )
        
        for r_homo, r_hc3 in zip(results_homo, results_hc3):
            assert np.isclose(r_homo.att, r_hc3.att, rtol=1e-10), \
                f"ATT differs: homo={r_homo.att}, hc3={r_hc3.att}"


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_treated_unit(self):
        """Test estimation with single treated unit."""
        data = pd.DataFrame({
            'id': [1,1,1, 2,2,2, 3,3,3],
            'year': [1,2,3, 1,2,3, 1,2,3],
            'y': [10.0, 12.0, 20.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            'gvar': [3,3,3, 0,0,0, 0,0,0]  # 1 treated, 2 NT
        })
        
        transformed = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        results = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year',
            min_treated=1
        )
        
        assert len(results) >= 1
        assert results[0].n_treated == 1
    
    def test_insufficient_controls_skipped(self):
        """Test that (g,r) pairs with no controls are skipped."""
        data = pd.DataFrame({
            'id': [1, 2],
            'year': [4, 4],
            'y': [10.0, 20.0],
            'gvar': [4, 4]  # All units are cohort 4, no controls
        })
        
        # Manually add transformation column
        data['ydot_g4_r4'] = [1.0, 2.0]
        
        results = estimate_cohort_time_effects(
            data, 'gvar', 'id', 'year',
            min_control=1
        )
        
        # No valid results due to no controls
        assert len(results) == 0
    
    def test_all_eventually_treated(self):
        """Test when all units eventually get treated (no NT)."""
        data = pd.DataFrame({
            'id': [1,1,1, 2,2,2, 3,3,3],
            'year': [1,2,3, 1,2,3, 1,2,3],
            'y': [10.0]*3 + [20.0]*3 + [30.0]*3,
            'gvar': [2,2,2, 3,3,3, 3,3,3]  # Cohorts 2,3 only, no NT
        })
        
        transformed = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        results = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year',
            control_strategy='not_yet_treated'
        )
        
        # Some pairs may be estimable using NYT controls
        # Cohort 2 in period 2 can use cohort 3 as control
        cohorts_estimated = {r.cohort for r in results}
        if len(results) > 0:
            assert 2 in cohorts_estimated, \
                "Cohort 2 should be estimable with cohort 3 as NYT control"


# ============================================================================
# Utility Function Tests
# ============================================================================

class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_results_to_dataframe(self, simple_test_data):
        """Test conversion to DataFrame."""
        transformed = transform_staggered_demean(
            simple_test_data, 'y', 'id', 'year', 'gvar'
        )
        results = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year'
        )
        
        df = results_to_dataframe(results)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(results)
        
        expected_cols = [
            'cohort', 'period', 'event_time', 'att', 'se',
            'ci_lower', 'ci_upper', 't_stat', 'pvalue',
            'n_treated', 'n_control', 'n_total'
        ]
        for col in expected_cols:
            assert col in df.columns
    
    def test_results_to_dataframe_empty(self):
        """Test conversion with empty results."""
        df = results_to_dataframe([])
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ============================================================================
# Castle Law End-to-End Test
# ============================================================================

@pytest.mark.integration
@pytest.mark.stata_alignment
class TestCastleLawEndToEnd:
    """End-to-end test using Castle Law data."""
    
    def test_castle_basic_estimation(self, castle_data):
        """Basic estimation on Castle Law data."""
        if castle_data is None:
            pytest.skip("Castle data not available")
        
        # Transform data
        transformed = transform_staggered_demean(
            castle_data, 
            y='lhomicide', 
            ivar='sid', 
            tvar='year', 
            gvar='gvar'
        )
        
        # Estimate effects
        results = estimate_cohort_time_effects(
            transformed, 
            gvar='gvar', 
            ivar='sid', 
            tvar='year',
            control_strategy='not_yet_treated'
        )
        
        # Basic assertions
        assert len(results) > 0
        
        # Expected cohorts: 2005, 2006, 2007, 2008, 2009
        cohorts = {r.cohort for r in results}
        assert cohorts.issubset({2005, 2006, 2007, 2008, 2009})
        
        # Check reasonable estimates
        for r in results:
            assert not np.isnan(r.att)
            assert not np.isnan(r.se)
            assert r.se > 0
    
    def test_castle_expected_count(self, castle_data):
        """Verify expected number of (g,r) pairs."""
        if castle_data is None:
            pytest.skip("Castle data not available")
        
        transformed = transform_staggered_demean(
            castle_data, 
            y='lhomicide', 
            ivar='sid', 
            tvar='year', 
            gvar='gvar'
        )
        
        results = estimate_cohort_time_effects(
            transformed, 
            gvar='gvar', 
            ivar='sid', 
            tvar='year',
            control_strategy='not_yet_treated'
        )
        
        # T_max = 2010
        # Cohort 2005: 6 periods (2005-2010)
        # Cohort 2006: 5 periods (2006-2010)
        # Cohort 2007: 4 periods (2007-2010)
        # Cohort 2008: 3 periods (2008-2010)
        # Cohort 2009: 2 periods (2009-2010)
        expected_max = 6 + 5 + 4 + 3 + 2  # = 20
        
        # May have fewer if some pairs lack sufficient data
        assert len(results) <= expected_max
        assert len(results) >= 15, \
            f"Expected at least 15 (g,r) pairs, got {len(results)}"
