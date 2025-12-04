"""
Comprehensive tests for E1-S3: (g,r)-specific effect estimation.

This module tests all acceptance criteria for Story 1.3:
1. estimate_cohort_time_effects() function implementation
2. Cross-sectional regression on transformed data
3. Sample construction: D_{ig} + A_{r+1} = 1
4. Multiple VCE types (homoskedastic, HC3, cluster)
5. Structured result output

Reference:
    - PRD-staggered-extension.md FR-3
    - epics-staggered-extension.md E1-S3
    - Lee & Wooldridge (2023) Procedure 4.1
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
def staggered_dgp_data():
    """
    Synthetic data matching Stata 2.lee_wooldridge_staggered_data.dta structure.
    
    Structure:
    - T=6 periods (years 1-6)
    - Cohorts: {4, 5, 6, inf}
    - N=40 units (10 per cohort)
    - Known treatment effect: tau=2 for all (g,r)
    """
    np.random.seed(42)
    n_per_cohort = 10
    cohorts = [4, 5, 6, 0]  # 0 = never treated
    T = 6
    
    data = []
    unit_id = 0
    
    for g in cohorts:
        for _ in range(n_per_cohort):
            unit_id += 1
            # Unit fixed effect
            alpha_i = np.random.normal(0, 1)
            
            for t in range(1, T + 1):
                # Base outcome
                y = alpha_i + 0.5 * t + np.random.normal(0, 0.5)
                
                # Add treatment effect for treated units in post-treatment periods
                if g > 0 and t >= g:
                    y += 2.0  # Known treatment effect = 2
                
                data.append({
                    'id': unit_id,
                    'year': t,
                    'y': y,
                    'gvar': g,
                    'x1': np.random.normal(0, 1),
                    'x2': np.random.normal(0, 1),
                })
    
    return pd.DataFrame(data)


@pytest.fixture
def castle_data():
    """Load Castle Law data."""
    data_path = Path(__file__).parent.parent.parent / 'data' / 'castle.csv'
    if data_path.exists():
        data = pd.read_csv(data_path)
        data['gvar'] = data['effyear'].fillna(0).astype(int)
        return data
    return None


# ============================================================================
# AC-1: estimate_cohort_time_effects() Function Tests
# ============================================================================

class TestEstimateCohortTimeEffects:
    """Test estimate_cohort_time_effects() function implementation."""
    
    def test_function_signature(self, staggered_dgp_data):
        """Verify function signature matches Epic specification."""
        # Function should accept these parameters
        transformed = transform_staggered_demean(
            staggered_dgp_data, 'y', 'id', 'year', 'gvar'
        )
        
        results = estimate_cohort_time_effects(
            data_transformed=transformed,
            gvar='gvar',
            ivar='id',
            tvar='year',
            controls=None,
            vce=None,
            cluster_var=None,
            control_strategy='not_yet_treated',
        )
        
        assert isinstance(results, list)
    
    def test_returns_correct_structure(self, staggered_dgp_data):
        """Verify return type is List[CohortTimeEffect]."""
        transformed = transform_staggered_demean(
            staggered_dgp_data, 'y', 'id', 'year', 'gvar'
        )
        
        results = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year'
        )
        
        assert all(isinstance(r, CohortTimeEffect) for r in results)
        
        # Check all required fields
        required_fields = [
            'cohort', 'period', 'event_time', 'att', 'se',
            'ci_lower', 'ci_upper', 't_stat', 'pvalue',
            'n_treated', 'n_control', 'n_total'
        ]
        for r in results:
            for field in required_fields:
                assert hasattr(r, field), f"Missing field: {field}"


# ============================================================================
# AC-2: Cross-sectional Regression on Transformed Data
# ============================================================================

class TestCrossSectionalRegression:
    """
    Test that estimation is cross-sectional regression.
    
    Key: For each (g, r), we extract period r's cross-section and run OLS.
    Each unit has exactly one row in the regression.
    """
    
    def test_sample_is_cross_sectional(self, staggered_dgp_data):
        """Verify estimation sample is cross-sectional (one row per unit)."""
        transformed = transform_staggered_demean(
            staggered_dgp_data, 'y', 'id', 'year', 'gvar'
        )
        
        results = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year'
        )
        
        # For each result, n_total should equal number of unique units
        # in the estimation sample (treatment + controls)
        for r in results:
            # n_total = n_treated + n_control
            # Each unit appears once in cross-section
            assert r.n_total == r.n_treated + r.n_control
    
    def test_regression_on_transformed_y(self, staggered_dgp_data):
        """Verify regression uses ydot_g{g}_r{r} as dependent variable."""
        transformed = transform_staggered_demean(
            staggered_dgp_data, 'y', 'id', 'year', 'gvar'
        )
        
        # Check that transformation columns exist for valid (g,r) pairs
        cohorts = [4, 5, 6]  # From fixture
        T_max = 6
        
        for g in cohorts:
            for r in range(g, T_max + 1):
                col_name = f'ydot_g{g}_r{r}'
                assert col_name in transformed.columns, \
                    f"Missing transformation column: {col_name}"


# ============================================================================
# AC-3: Sample Construction (D_{ig} + A_{r+1} = 1)
# ============================================================================

class TestSampleConstruction:
    """
    Test correct sample construction for estimation.
    
    Sample must satisfy: D_{ig} + A_{r+1} = 1
    - Treatment group: cohort g units
    - Control group: gvar > period (NYT) OR gvar in {0, inf, NaN} (NT)
    - Exclude: other already-treated cohorts
    """
    
    def test_treatment_group_correct(self, staggered_dgp_data):
        """Verify treatment group is exactly cohort g units."""
        transformed = transform_staggered_demean(
            staggered_dgp_data, 'y', 'id', 'year', 'gvar'
        )
        
        results = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year',
            control_strategy='not_yet_treated'
        )
        
        # Count units per cohort in fixture
        n_per_cohort = 10  # From fixture definition
        
        for r in results:
            assert r.n_treated == n_per_cohort, \
                f"Expected {n_per_cohort} treated units for cohort {r.cohort}, " \
                f"got {r.n_treated}"
    
    def test_gvar_equal_period_excluded(self):
        """
        ⚠️ CRITICAL: Verify gvar==period units are NOT in control group.
        
        Stata equivalent: `if f05 & ~g5` excludes cohort 5 when estimating
        τ_{4,5} because cohort 5 starts treatment in period 5.
        """
        data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'year': [5]*5,
            'gvar': [4, 5, 6, 7, 0]
        })
        
        # For cohort=4, period=5
        control_mask = get_valid_control_units(
            data, 'gvar', 'id', cohort=4, period=5,
            strategy=ControlGroupStrategy.NOT_YET_TREATED
        )
        
        # Check each unit
        assert control_mask.loc[1] == False  # gvar=4, treatment group
        assert control_mask.loc[2] == False  # gvar=5==5, NOT control!
        assert control_mask.loc[3] == True   # gvar=6>5, NYT control
        assert control_mask.loc[4] == True   # gvar=7>5, NYT control
        assert control_mask.loc[5] == True   # gvar=0, NT control
    
    def test_control_group_strategies(self):
        """Test all three control group strategies work correctly."""
        data = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'year': [4]*4,
            'gvar': [4, 5, 6, 0]
        })
        
        # NEVER_TREATED: only gvar=0
        nt_mask = get_valid_control_units(
            data, 'gvar', 'id', cohort=4, period=4,
            strategy=ControlGroupStrategy.NEVER_TREATED
        )
        assert nt_mask.sum() == 1
        assert nt_mask.loc[4] == True  # Only gvar=0
        
        # NOT_YET_TREATED: gvar>4 OR gvar=0
        nyt_mask = get_valid_control_units(
            data, 'gvar', 'id', cohort=4, period=4,
            strategy=ControlGroupStrategy.NOT_YET_TREATED
        )
        assert nyt_mask.sum() == 3
        assert nyt_mask.loc[2] == True   # gvar=5>4
        assert nyt_mask.loc[3] == True   # gvar=6>4
        assert nyt_mask.loc[4] == True   # gvar=0
        
        # AUTO: same as NYT when NYT available
        auto_mask = get_valid_control_units(
            data, 'gvar', 'id', cohort=4, period=4,
            strategy=ControlGroupStrategy.AUTO
        )
        assert (auto_mask == nyt_mask).all()


# ============================================================================
# AC-4: Standard Error Types
# ============================================================================

class TestStandardErrorTypes:
    """Test multiple standard error types."""
    
    def test_homoskedastic_se(self, staggered_dgp_data):
        """Test homoskedastic (OLS) standard errors."""
        transformed = transform_staggered_demean(
            staggered_dgp_data, 'y', 'id', 'year', 'gvar'
        )
        
        results = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year',
            vce=None
        )
        
        for r in results:
            assert r.se > 0
            assert not np.isnan(r.se)
            # t-stat should be reasonable
            assert abs(r.t_stat) < 100
    
    def test_hc3_se(self, staggered_dgp_data):
        """Test HC3 heteroskedasticity-robust standard errors."""
        transformed = transform_staggered_demean(
            staggered_dgp_data, 'y', 'id', 'year', 'gvar'
        )
        
        results = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year',
            vce='hc3'
        )
        
        for r in results:
            assert r.se > 0
            assert not np.isnan(r.se)
    
    def test_cluster_se(self, staggered_dgp_data):
        """Test cluster-robust standard errors."""
        # Add cluster variable
        staggered_dgp_data['cluster'] = staggered_dgp_data['id'] % 5
        
        transformed = transform_staggered_demean(
            staggered_dgp_data, 'y', 'id', 'year', 'gvar'
        )
        
        results = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year',
            vce='cluster',
            cluster_var='cluster'
        )
        
        for r in results:
            assert r.se > 0
            assert not np.isnan(r.se)
    
    def test_att_invariant_to_vce(self, staggered_dgp_data):
        """ATT point estimate should not change with VCE type."""
        transformed = transform_staggered_demean(
            staggered_dgp_data, 'y', 'id', 'year', 'gvar'
        )
        
        results_homo = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year', vce=None
        )
        results_hc3 = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year', vce='hc3'
        )
        
        for r_homo, r_hc3 in zip(results_homo, results_hc3):
            assert np.isclose(r_homo.att, r_hc3.att, rtol=1e-10)


# ============================================================================
# AC-5: Output Format Verification
# ============================================================================

class TestOutputFormat:
    """Test output format requirements."""
    
    def test_output_includes_event_time(self, staggered_dgp_data):
        """Verify event_time = period - cohort is included."""
        transformed = transform_staggered_demean(
            staggered_dgp_data, 'y', 'id', 'year', 'gvar'
        )
        
        results = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year'
        )
        
        for r in results:
            expected_et = r.period - r.cohort
            assert r.event_time == expected_et
            assert r.event_time >= 0
    
    def test_results_to_dataframe(self, staggered_dgp_data):
        """Test conversion to DataFrame format."""
        transformed = transform_staggered_demean(
            staggered_dgp_data, 'y', 'id', 'year', 'gvar'
        )
        
        results = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year'
        )
        
        df = results_to_dataframe(results)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(results)
        
        # Check all required columns
        required_cols = [
            'cohort', 'period', 'event_time', 'att', 'se',
            'ci_lower', 'ci_upper', 't_stat', 'pvalue',
            'n_treated', 'n_control', 'n_total'
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"
    
    def test_confidence_intervals(self, staggered_dgp_data):
        """Verify CI is [ATT - t*SE, ATT + t*SE]."""
        transformed = transform_staggered_demean(
            staggered_dgp_data, 'y', 'id', 'year', 'gvar'
        )
        
        results = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year'
        )
        
        for r in results:
            assert r.ci_lower < r.att < r.ci_upper
            # Symmetry check
            assert np.isclose(r.att - r.ci_lower, r.ci_upper - r.att, rtol=0.01)


# ============================================================================
# Castle Law End-to-End Test
# ============================================================================

class TestCastleLawEstimation:
    """End-to-end estimation tests using Castle Law data."""
    
    def test_castle_estimation_basic(self, castle_data):
        """Basic estimation on Castle Law data."""
        if castle_data is None:
            pytest.skip("Castle data not available")
        
        transformed = transform_staggered_demean(
            castle_data, 'lhomicide', 'sid', 'year', 'gvar'
        )
        
        results = estimate_cohort_time_effects(
            transformed, 'gvar', 'sid', 'year',
            control_strategy='not_yet_treated'
        )
        
        assert len(results) > 0
        
        # Verify cohorts
        cohorts_found = {r.cohort for r in results}
        expected_cohorts = {2005, 2006, 2007, 2008, 2009}
        assert cohorts_found.issubset(expected_cohorts)
    
    def test_castle_expected_pairs_count(self, castle_data):
        """
        Verify expected number of (g,r) pairs.
        
        Castle Law cohorts: 2005, 2006, 2007, 2008, 2009
        T_max = 2010
        Expected pairs: 6 + 5 + 4 + 3 + 2 = 20
        """
        if castle_data is None:
            pytest.skip("Castle data not available")
        
        transformed = transform_staggered_demean(
            castle_data, 'lhomicide', 'sid', 'year', 'gvar'
        )
        
        results = estimate_cohort_time_effects(
            transformed, 'gvar', 'sid', 'year',
            control_strategy='not_yet_treated'
        )
        
        # Count by cohort
        counts = {}
        for r in results:
            counts[r.cohort] = counts.get(r.cohort, 0) + 1
        
        # Cohort 2005: 2005-2010 = 6 periods
        # Cohort 2006: 2006-2010 = 5 periods
        # ...
        expected = {
            2005: 6,
            2006: 5,
            2007: 4,
            2008: 3,
            2009: 2,
        }
        
        for cohort, expected_count in expected.items():
            if cohort in counts:
                assert counts[cohort] == expected_count, \
                    f"Cohort {cohort}: expected {expected_count}, got {counts[cohort]}"
    
    def test_castle_florida_instantaneous_effect(self, castle_data):
        """
        Test Florida (only 2005 cohort state) instantaneous effect.
        
        Florida should have τ_{2005, 2005} (event time = 0).
        """
        if castle_data is None:
            pytest.skip("Castle data not available")
        
        transformed = transform_staggered_demean(
            castle_data, 'lhomicide', 'sid', 'year', 'gvar'
        )
        
        results = estimate_cohort_time_effects(
            transformed, 'gvar', 'sid', 'year',
            control_strategy='not_yet_treated'
        )
        
        # Find τ_{2005, 2005}
        tau_2005_2005 = next(
            (r for r in results if r.cohort == 2005 and r.period == 2005),
            None
        )
        
        assert tau_2005_2005 is not None
        assert tau_2005_2005.event_time == 0
        assert tau_2005_2005.n_treated == 1  # Only Florida
        assert not np.isnan(tau_2005_2005.att)
        assert tau_2005_2005.se > 0
    
    def test_castle_never_treated_strategy(self, castle_data):
        """Test estimation using only never treated as controls."""
        if castle_data is None:
            pytest.skip("Castle data not available")
        
        transformed = transform_staggered_demean(
            castle_data, 'lhomicide', 'sid', 'year', 'gvar'
        )
        
        results_nt = estimate_cohort_time_effects(
            transformed, 'gvar', 'sid', 'year',
            control_strategy='never_treated'
        )
        
        results_nyt = estimate_cohort_time_effects(
            transformed, 'gvar', 'sid', 'year',
            control_strategy='not_yet_treated'
        )
        
        # Both should produce results
        assert len(results_nt) > 0
        assert len(results_nyt) > 0
        
        # NT should have fewer controls for late periods
        tau_nt = next(r for r in results_nt if r.cohort == 2005 and r.period == 2010)
        tau_nyt = next(r for r in results_nyt if r.cohort == 2005 and r.period == 2010)
        
        # NT controls should be <= NYT controls
        assert tau_nt.n_control <= tau_nyt.n_control


# ============================================================================
# Edge Cases and Boundary Conditions
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_treated_unit(self):
        """Test estimation with single treated unit."""
        data = pd.DataFrame({
            'id': [1,1,1, 2,2,2, 3,3,3],
            'year': [1,2,3, 1,2,3, 1,2,3],
            'y': [10.0, 12.0, 20.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            'gvar': [3,3,3, 0,0,0, 0,0,0]
        })
        
        transformed = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        results = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year',
            min_treated=1
        )
        
        assert len(results) >= 1
        assert results[0].n_treated == 1
    
    def test_all_eventually_treated(self):
        """
        Test when all units eventually get treated.
        
        In this case:
        - Some (g,r) pairs can be estimated using NYT
        - Last cohort cannot be estimated (no controls)
        - Cohort/overall effects cannot be estimated
        """
        data = pd.DataFrame({
            'id': [1]*4 + [2]*4 + [3]*4,
            'year': [1,2,3,4]*3,
            'y': [10.0]*4 + [20.0]*4 + [30.0]*4,
            'gvar': [2,2,2,2, 3,3,3,3, 4,4,4,4]
        })
        
        transformed = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        results = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year',
            control_strategy='not_yet_treated'
        )
        
        # Cohort 2 can use cohorts 3,4 as controls at period 2
        # Cohort 4 has no controls (last cohort, all others already treated)
        cohorts_estimated = {r.cohort for r in results}
        
        # Cohort 4 should NOT be in results (no controls)
        if 4 in cohorts_estimated:
            tau_4 = [r for r in results if r.cohort == 4]
            for t in tau_4:
                # If cohort 4 is estimated, it must be in a period where
                # there are still NYT units
                assert t.n_control > 0
    
    def test_insufficient_observations_skipped(self):
        """Test that (g,r) pairs with insufficient data are skipped."""
        data = pd.DataFrame({
            'id': [1, 2],
            'year': [3, 3],
            'y': [10.0, 20.0],
            'gvar': [3, 3]  # All same cohort, no controls
        })
        
        data['ydot_g3_r3'] = [1.0, 2.0]
        
        results = estimate_cohort_time_effects(
            data, 'gvar', 'id', 'year',
            min_control=1
        )
        
        assert len(results) == 0


# ============================================================================
# Treatment Effect Recovery Test
# ============================================================================

class TestTreatmentEffectRecovery:
    """Test that known treatment effects can be recovered."""
    
    def test_recovers_constant_treatment_effect(self, staggered_dgp_data):
        """
        Test that known treatment effect (tau=2) can be recovered.
        
        DGP in fixture has tau=2 for all (g,r).
        """
        transformed = transform_staggered_demean(
            staggered_dgp_data, 'y', 'id', 'year', 'gvar'
        )
        
        results = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year'
        )
        
        # Average ATT should be close to 2
        atts = [r.att for r in results]
        avg_att = np.mean(atts)
        
        # Allow some statistical variation
        assert 1.0 < avg_att < 3.0, \
            f"Average ATT={avg_att:.2f}, expected close to 2.0"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
