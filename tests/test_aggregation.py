"""
Tests for the staggered effect aggregation module.

Tests cohort-specific (τ_g) and overall (τ_ω) effect aggregation,
including variance-weighted combination of (g,r)-specific estimates.

Validates Section 7 (aggregation procedures) of the Lee-Wooldridge
Difference-in-Differences framework.

References
----------
Lee, S. & Wooldridge, J. M. (2025). A Simple Transformation Approach to
    Difference-in-Differences Estimation for Panel Data. SSRN 4516518.
Lee, S. & Wooldridge, J. M. (2026). Simple Approaches to Inference with
    DiD Estimators with Small Cross-Sectional Sample Sizes. SSRN 5325686.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lwdid.staggered.transformations import (
    transform_staggered_demean,
    transform_staggered_detrend,
    get_cohorts,
)
from lwdid.staggered.aggregation import (
    aggregate_to_cohort,
    aggregate_to_overall,
    construct_aggregated_outcome,
    CohortEffect,
    OverallEffect,
    cohort_effects_to_dataframe,
)
from lwdid.exceptions import NoNeverTreatedError


# =============================================================================
# Test Fixtures
# =============================================================================

def create_test_data_with_cohorts(
    cohorts, 
    n_never_treated, 
    T_max, 
    treatment_effect=5.0, 
    base_y=10.0
):
    """
    Create staggered panel data for testing.
    
    Parameters
    ----------
    cohorts : list of int
        Treatment cohorts, e.g., [3, 4, 5]
    n_never_treated : int
        Number of never treated units
    T_max : int
        Maximum time period
    treatment_effect : float
        Treatment effect added to treated units post-treatment
    base_y : float
        Base Y value
        
    Returns
    -------
    DataFrame
        Long-format panel data with columns: id, year, y, gvar
    """
    data_rows = []
    unit_id = 1
    
    # Create treated units
    for g in cohorts:
        for t in range(1, T_max + 1):
            y = base_y + t * 2  # Linear trend
            if t >= g:
                y += treatment_effect  # Add treatment effect
            data_rows.append({
                'id': unit_id,
                'year': t,
                'y': y,
                'gvar': g
            })
        unit_id += 1
    
    # Create never treated units
    for _ in range(n_never_treated):
        for t in range(1, T_max + 1):
            y = base_y + t * 2  # Same trend, no treatment
            data_rows.append({
                'id': unit_id,
                'year': t,
                'y': y,
                'gvar': 0  # 0 = never treated
            })
        unit_id += 1
    
    return pd.DataFrame(data_rows)


def create_test_data():
    """Simple test data: 1 cohort, 2 NT units."""
    return create_test_data_with_cohorts([4], n_never_treated=2, T_max=6)


# =============================================================================
# Basic Functionality Tests
# =============================================================================

class TestAggregateToCohortBasic:
    """Basic tests for aggregate_to_cohort function."""
    
    def test_basic_three_cohorts(self):
        """Test basic cohort aggregation with 3 cohorts."""
        data = create_test_data_with_cohorts([3, 4, 5], n_never_treated=2, T_max=6)
        transformed = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        cohort_effects = aggregate_to_cohort(
            transformed, 
            gvar='gvar', 
            ivar='id', 
            tvar='year', 
            cohorts=[3, 4, 5], 
            T_max=6,
            never_treated_values=[0],
            transform_type='demean'
        )
        
        assert len(cohort_effects) == 3, f"Expected 3 cohort effects, got {len(cohort_effects)}"
        
        for effect in cohort_effects:
            assert hasattr(effect, 'cohort')
            assert hasattr(effect, 'att')
            assert hasattr(effect, 'se')
            assert effect.n_control == 2, f"NT control should be 2, got {effect.n_control}"
            assert not np.isnan(effect.att), f"Cohort {effect.cohort} ATT is NaN"
            assert effect.se > 0, f"Cohort {effect.cohort} SE should be > 0"
    
    def test_control_group_enforcement(self):
        """Test that cohort aggregation uses NT control group correctly."""
        data = create_test_data()  # cohort=[4], n_nt=2, T_max=6
        transformed = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        result = aggregate_to_cohort(
            transformed, 
            gvar='gvar', 
            ivar='id', 
            tvar='year', 
            cohorts=[4], 
            T_max=6,
            never_treated_values=[0]
        )
        
        assert len(result) == 1
        assert result[0].n_control == 2  # 2 NT units
    
    def test_no_never_treated_error(self):
        """Test error when no NT units available."""
        # Create data without NT units
        data = pd.DataFrame({
            'id': [1,1,1, 2,2,2, 3,3,3],
            'year': [1,2,3]*3,
            'y': [10,12,14, 15,17,19, 20,22,24],
            'gvar': [2,2,2, 3,3,3, 3,3,3]  # All treated, no NT
        })
        
        transformed = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        with pytest.raises(NoNeverTreatedError) as excinfo:
            aggregate_to_cohort(
                transformed, 
                gvar='gvar', 
                ivar='id', 
                tvar='year', 
                cohorts=[2, 3], 
                T_max=3,
                never_treated_values=[0]
            )
        
        error_msg = str(excinfo.value).lower()
        assert "never-treated" in error_msg


class TestAggregateToOverallBasic:
    """Basic tests for aggregate_to_overall function."""
    
    def test_basic_overall_effect(self):
        """Test basic overall effect estimation."""
        data = create_test_data_with_cohorts([4, 5], n_never_treated=3, T_max=6)
        transformed = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        result = aggregate_to_overall(
            transformed, 
            gvar='gvar', 
            ivar='id', 
            tvar='year',
            never_treated_values=[0]
        )
        
        assert hasattr(result, 'att')
        assert hasattr(result, 'se')
        assert hasattr(result, 'cohort_weights')
        
        # Weights should sum to 1
        assert np.isclose(sum(result.cohort_weights.values()), 1.0)
    
    def test_weights_calculation(self):
        """Test cohort weights calculation."""
        # Create data with specific cohort sizes
        # cohort 4: 2 units -> omega_4 = 2/5
        # cohort 5: 3 units -> omega_5 = 3/5
        data = pd.DataFrame({
            'id': [1,1, 2,2, 3,3, 4,4, 5,5, 6,6, 7,7],
            'year': [1,2]*7,
            'y': [10]*14,
            'gvar': [4,4, 4,4, 5,5, 5,5, 5,5, 0,0, 0,0]
        })
        
        unit_gvar = data.drop_duplicates('id').set_index('id')['gvar']
        
        cohort_4_count = (unit_gvar == 4).sum()
        cohort_5_count = (unit_gvar == 5).sum()
        
        assert cohort_4_count == 2
        assert cohort_5_count == 3
    
    def test_no_never_treated_error(self):
        """Test error when no NT units for overall effect."""
        data = pd.DataFrame({
            'id': [1,1, 2,2],
            'year': [1,2]*2,
            'y': [10,12, 15,17],
            'gvar': [2,2, 2,2]  # All treated
        })
        
        transformed = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        with pytest.raises(NoNeverTreatedError) as excinfo:
            aggregate_to_overall(
                transformed, 
                gvar='gvar', 
                ivar='id',
                tvar='year',
                never_treated_values=[0]
            )
        
        error_msg = str(excinfo.value).lower()
        assert "never-treated" in error_msg


# =============================================================================
# Numerical Accuracy Tests
# =============================================================================

class TestNumericalAccuracy:
    """Tests for numerical accuracy against manual calculations."""
    
    def test_cohort_effect_numerical(self):
        """Test cohort effect against manual calculation.
        
        Data:
        - cohort 3: 1 unit (id=1), treated at t=3
          y = [10, 12, 20, 25] (t=1,2,3,4)
          pre_mean_g3 = (10+12)/2 = 11
          ydot_g3_r3 = 20 - 11 = 9
          ydot_g3_r4 = 25 - 11 = 14
          Y_bar_i1_g3 = (9+14)/2 = 11.5
        
        - never treated: 1 unit (id=2)
          y = [5, 6, 7, 8] (t=1,2,3,4)
          pre_mean_g3 = (5+6)/2 = 5.5
          ydot_g3_r3 = 7 - 5.5 = 1.5
          ydot_g3_r4 = 8 - 5.5 = 2.5
          Y_bar_i2_g3 = (1.5+2.5)/2 = 2.0
        
        Expected τ_3 = 11.5 - 2.0 = 9.5
        
        Note: With only 2 units, SE is NaN due to df=0.
        """
        data = pd.DataFrame({
            'id': [1,1,1,1, 2,2,2,2],
            'year': [1,2,3,4, 1,2,3,4],
            'y': [10,12,20,25, 5,6,7,8],
            'gvar': [3,3,3,3, 0,0,0,0]
        })
        
        transformed = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        # Verify transformation values
        assert np.isclose(
            transformed.loc[(transformed['id']==1) & (transformed['year']==3), 'ydot_g3_r3'].iloc[0],
            9.0
        )
        assert np.isclose(
            transformed.loc[(transformed['id']==1) & (transformed['year']==4), 'ydot_g3_r4'].iloc[0],
            14.0
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore small sample warnings
            cohort_effects = aggregate_to_cohort(
                transformed, 
                gvar='gvar', 
                ivar='id', 
                tvar='year', 
                cohorts=[3], 
                T_max=4,
                never_treated_values=[0]
            )
        
        assert len(cohort_effects) == 1
        assert np.isclose(cohort_effects[0].att, 9.5, atol=1e-10)
        assert cohort_effects[0].n_units == 1
        assert cohort_effects[0].n_control == 1
        assert cohort_effects[0].n_periods == 2
        # SE is NaN with only 2 units
        assert np.isnan(cohort_effects[0].se)
    
    def test_overall_equals_weighted_cohort(self):
        """Verify overall effect equals weighted average of cohort effects."""
        data = create_test_data_with_cohorts([4, 5], n_never_treated=3, T_max=6)
        transformed = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        # Cohort effects
        cohort_effects = aggregate_to_cohort(
            transformed, 
            gvar='gvar', 
            ivar='id', 
            tvar='year', 
            cohorts=[4, 5], 
            T_max=6,
            never_treated_values=[0]
        )
        
        # Overall effect
        overall_effect = aggregate_to_overall(
            transformed, 
            gvar='gvar', 
            ivar='id', 
            tvar='year',
            never_treated_values=[0]
        )
        
        # Manual weighted average
        weighted_sum = 0.0
        for ce in cohort_effects:
            weight = overall_effect.cohort_weights[ce.cohort]
            weighted_sum += weight * ce.att
        
        # Should be very close (numerical precision)
        assert np.isclose(overall_effect.att, weighted_sum, atol=1e-6)


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_single_post_period(self):
        """Test cohort with only 1 post period (g = T_max)."""
        data = create_test_data_with_cohorts([5], n_never_treated=2, T_max=5)
        transformed = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        cohort_effects = aggregate_to_cohort(
            transformed, 'gvar', 'id', 'year', [5], 5, never_treated_values=[0]
        )
        
        assert len(cohort_effects) == 1
        assert cohort_effects[0].n_periods == 1
        assert not np.isnan(cohort_effects[0].att)
    
    def test_single_cohort(self):
        """Single cohort: overall effect should equal cohort effect."""
        data = create_test_data_with_cohorts([4], n_never_treated=3, T_max=6)
        transformed = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        cohort_effects = aggregate_to_cohort(
            transformed, 'gvar', 'id', 'year', [4], 6, never_treated_values=[0]
        )
        overall_effect = aggregate_to_overall(
            transformed, 'gvar', 'id', 'year', never_treated_values=[0]
        )
        
        assert len(cohort_effects) == 1
        assert np.isclose(overall_effect.att, cohort_effects[0].att, atol=1e-10)
        assert np.isclose(overall_effect.cohort_weights[4], 1.0)
    
    def test_single_nt_unit(self):
        """Test with only 1 NT unit (minimum viable).
        
        With 2 units (1 treated + 1 NT), ATT can be estimated but SE is NaN
        due to zero degrees of freedom.
        """
        data = create_test_data_with_cohorts([4], n_never_treated=1, T_max=6)
        transformed = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore warnings about small sample
            result = aggregate_to_overall(
                transformed, 'gvar', 'id', 'year', never_treated_values=[0]
            )
        
        assert result.n_control == 1
        assert not np.isnan(result.att)
        # SE is NaN due to df=0
        assert np.isnan(result.se)
    
    def test_all_eventually_treated_cohort_error(self):
        """All Eventually Treated: cohort effect should error."""
        data = pd.DataFrame({
            'id': [1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3],
            'year': [1,2,3,4,5]*3,
            'y': [10,12,18,22,26, 15,17,19,25,29, 20,22,24,26,35],
            'gvar': [3]*5 + [4]*5 + [5]*5  # No NT
        })
        
        transformed = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        with pytest.raises(NoNeverTreatedError) as excinfo:
            aggregate_to_cohort(
                transformed, 
                gvar='gvar', 
                ivar='id', 
                tvar='year', 
                cohorts=[3, 4, 5], 
                T_max=5,
                never_treated_values=[0]
            )
        
        assert "never-treated" in str(excinfo.value).lower()
    
    def test_all_eventually_treated_overall_error(self):
        """All Eventually Treated: overall effect should error."""
        data = pd.DataFrame({
            'id': [1,1,1, 2,2,2, 3,3,3],
            'year': [1,2,3]*3,
            'y': [10,12,18, 15,17,19, 20,22,24],
            'gvar': [2]*3 + [3]*3 + [3]*3  # No NT
        })
        
        transformed = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        with pytest.raises(NoNeverTreatedError) as excinfo:
            aggregate_to_overall(
                transformed, 
                gvar='gvar', 
                ivar='id', 
                tvar='year',
                never_treated_values=[0]
            )
        
        assert "never-treated" in str(excinfo.value).lower()


# =============================================================================
# Stata Comparison Tests
# =============================================================================

@pytest.mark.stata_alignment
class TestStataComparison:
    """Tests comparing to hand-calculated Stata-equivalent results."""
    
    def test_cohort_effect_manual_calc(self):
        """Compare cohort effect to manual calculation.
        
        Same as numerical test but with more detailed verification.
        Note: With 2 units per cohort estimation (1 treated + 1 NT), SE is NaN.
        """
        data = pd.DataFrame({
            'id': [1,1,1,1, 2,2,2,2, 3,3,3,3],
            'year': [1,2,3,4, 1,2,3,4, 1,2,3,4],
            'y': [10,12,20,25, 15,16,17,22, 5,6,7,8],
            'gvar': [3,3,3,3, 4,4,4,4, 0,0,0,0]
        })
        
        transformed = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        # Cohort 3: Unit 1, NT: Unit 3
        # Unit 1: pre_mean = (10+12)/2 = 11
        # ydot_g3_r3 = 20 - 11 = 9, ydot_g3_r4 = 25 - 11 = 14
        # Y_bar_i1_g3 = (9+14)/2 = 11.5
        
        # Unit 3 (NT): pre_mean = (5+6)/2 = 5.5
        # ydot_g3_r3 = 7 - 5.5 = 1.5, ydot_g3_r4 = 8 - 5.5 = 2.5
        # Y_bar_i3_g3 = (1.5+2.5)/2 = 2.0
        
        # tau_3 = 11.5 - 2.0 = 9.5
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore small sample warnings
            cohort_effects = aggregate_to_cohort(
                transformed, gvar='gvar', ivar='id', tvar='year',
                cohorts=[3], T_max=4, never_treated_values=[0]
            )
        
        assert len(cohort_effects) == 1
        expected_tau_3 = 9.5
        assert np.isclose(cohort_effects[0].att, expected_tau_3, atol=1e-10)
    
    def test_overall_effect_manual_calc(self):
        """Compare overall effect to manual calculation."""
        data = pd.DataFrame({
            'id': [1,1,1,1, 2,2,2,2, 3,3,3,3],
            'year': [1,2,3,4, 1,2,3,4, 1,2,3,4],
            'y': [10,12,20,25, 15,16,17,22, 5,6,7,8],
            'gvar': [3,3,3,3, 4,4,4,4, 0,0,0,0]
        })
        
        transformed = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        # N_3 = 1, N_4 = 1, N_treat = 2
        # omega_3 = 0.5, omega_4 = 0.5
        
        overall_effect = aggregate_to_overall(
            transformed, gvar='gvar', ivar='id', tvar='year',
            never_treated_values=[0]
        )
        
        # Verify weights
        assert np.isclose(overall_effect.cohort_weights[3], 0.5, atol=1e-10)
        assert np.isclose(overall_effect.cohort_weights[4], 0.5, atol=1e-10)


# =============================================================================
# Castle Law End-to-End Tests
# =============================================================================

@pytest.mark.integration
@pytest.mark.paper_validation
class TestCastleLaw:
    """End-to-end tests with Castle Law data."""
    
    @pytest.fixture
    def castle_data(self):
        """Load and preprocess Castle Law data."""
        # Try multiple paths
        possible_paths = [
            os.environ.get('LWDID_TEST_DATA_PATH', ''),
            'lwdid-py_v0.1.0/data/castle.csv',
            '../lwdid-py_v0.1.0/data/castle.csv',
            str(Path(__file__).parent.parent / 'data' / 'castle.csv'),
        ]
        
        data_path = None
        for path in possible_paths:
            if path and Path(path).exists():
                data_path = path
                break
        
        if data_path is None:
            pytest.skip("Castle Law data not found")
        
        data = pd.read_csv(data_path)
        data['gvar'] = data['effyear'].fillna(0).astype(int)
        
        return data
    
    def test_castle_data_structure(self, castle_data):
        """Verify Castle Law data structure."""
        assert 'sid' in castle_data.columns
        assert 'year' in castle_data.columns
        assert 'lhomicide' in castle_data.columns
        assert 'gvar' in castle_data.columns
        
        n_states = castle_data['sid'].nunique()
        assert n_states == 50, f"Expected 50 states, got {n_states}"
        
        years = sorted(castle_data['year'].unique())
        assert years == list(range(2000, 2011)), f"Unexpected years: {years}"
    
    def test_castle_cohort_distribution(self, castle_data):
        """Verify Castle Law cohort distribution matches paper."""
        unit_gvar = castle_data.drop_duplicates('sid').set_index('sid')['gvar']
        
        expected = {
            2005: 1,
            2006: 13,
            2007: 4,
            2008: 2,
            2009: 1,
        }
        
        for cohort, expected_count in expected.items():
            actual = (unit_gvar == cohort).sum()
            assert actual == expected_count, \
                f"Cohort {cohort}: expected {expected_count}, got {actual}"
        
        n_nt = (unit_gvar == 0).sum()
        assert n_nt == 29, f"NT units: expected 29, got {n_nt}"
    
    def test_castle_overall_effect(self, castle_data):
        """Test overall effect against paper results.
        
        Paper results (Lee & Wooldridge 2025 Section 7.2):
        - τ̂_ω ≈ 0.092
        - OLS SE ≈ 0.057
        - OLS t ≈ 1.61
        - HC3 t ≈ 1.50
        """
        transformed = transform_staggered_demean(
            castle_data, y='lhomicide', ivar='sid', tvar='year', gvar='gvar'
        )
        
        result = aggregate_to_overall(
            transformed, 
            gvar='gvar', 
            ivar='sid', 
            tvar='year',
            never_treated_values=[0],
            vce='hc3'
        )
        
        # Verify against paper
        ATT_EXPECTED = 0.092
        ATT_TOLERANCE = 0.02
        
        assert np.isclose(result.att, ATT_EXPECTED, atol=ATT_TOLERANCE), \
            f"ATT {result.att:.4f} not close to expected {ATT_EXPECTED}"
        
        assert result.se > 0
        assert 0.04 < result.se < 0.10
        assert 1.0 < result.t_stat < 2.5
        
        # Verify sample sizes
        assert result.n_treated == 21
        assert result.n_control == 29
    
    def test_castle_cohort_effects(self, castle_data):
        """Test cohort effects for Castle Law data."""
        transformed = transform_staggered_demean(
            castle_data, y='lhomicide', ivar='sid', tvar='year', gvar='gvar'
        )
        
        cohorts = [2005, 2006, 2007, 2008, 2009]
        T_max = 2010
        
        cohort_effects = aggregate_to_cohort(
            transformed, 
            gvar='gvar', 
            ivar='sid', 
            tvar='year', 
            cohorts=cohorts, 
            T_max=T_max,
            never_treated_values=[0],
            vce='hc3'
        )
        
        expected_n_periods = {
            2005: 6,  # 2005-2010
            2006: 5,  # 2006-2010
            2007: 4,  # 2007-2010
            2008: 3,  # 2008-2010
            2009: 2,  # 2009-2010
        }
        
        expected_n_units = {
            2005: 1,
            2006: 13,
            2007: 4,
            2008: 2,
            2009: 1,
        }
        
        assert len(cohort_effects) == 5
        
        for effect in cohort_effects:
            g = effect.cohort
            
            assert not np.isnan(effect.att)
            assert effect.se > 0
            assert effect.n_control == 29
            assert effect.n_units == expected_n_units[g]
            assert effect.n_periods == expected_n_periods[g]
    
    def test_castle_weighted_average_consistency(self, castle_data):
        """Verify overall = weighted average of cohort effects."""
        transformed = transform_staggered_demean(
            castle_data, y='lhomicide', ivar='sid', tvar='year', gvar='gvar'
        )
        
        cohort_effects = aggregate_to_cohort(
            transformed, 
            gvar='gvar', 
            ivar='sid', 
            tvar='year', 
            cohorts=[2005, 2006, 2007, 2008, 2009], 
            T_max=2010,
            never_treated_values=[0], 
            vce=None
        )
        
        overall_effect = aggregate_to_overall(
            transformed, 'gvar', 'sid', 'year',
            never_treated_values=[0], vce=None
        )
        
        # Manual weighted average
        weighted_sum = 0.0
        for effect in cohort_effects:
            weight = overall_effect.cohort_weights[effect.cohort]
            weighted_sum += weight * effect.att
        
        assert np.isclose(overall_effect.att, weighted_sum, atol=1e-6)
    
    def test_castle_cohort_weights(self, castle_data):
        """Verify cohort weights calculation."""
        transformed = transform_staggered_demean(
            castle_data, y='lhomicide', ivar='sid', tvar='year', gvar='gvar'
        )
        
        result = aggregate_to_overall(
            transformed, 'gvar', 'sid', 'year', never_treated_values=[0]
        )
        
        N_treat = 21
        expected_weights = {
            2005: 1/N_treat,
            2006: 13/N_treat,
            2007: 4/N_treat,
            2008: 2/N_treat,
            2009: 1/N_treat,
        }
        
        for g, expected_w in expected_weights.items():
            actual_w = result.cohort_weights.get(g, 0)
            assert np.isclose(actual_w, expected_w, atol=0.001)
        
        # Weights sum to 1
        assert np.isclose(sum(result.cohort_weights.values()), 1.0, atol=1e-10)


# =============================================================================
# Transform Type Tests
# =============================================================================

class TestTransformType:
    """Tests for different transform types (demean vs detrend)."""
    
    def test_detrend_transform_type(self):
        """Test transform_type='detrend' parameter."""
        data = create_test_data_with_cohorts([4, 5], n_never_treated=2, T_max=6)
        
        # Use detrend transform
        transformed = transform_staggered_detrend(data, 'y', 'id', 'year', 'gvar')
        
        # Check ycheck columns exist
        expected_cols = ['ycheck_g4_r4', 'ycheck_g4_r5', 'ycheck_g4_r6',
                         'ycheck_g5_r5', 'ycheck_g5_r6']
        for col in expected_cols:
            assert col in transformed.columns, f"Missing column: {col}"
        
        # Aggregate with transform_type='detrend'
        cohort_effects = aggregate_to_cohort(
            transformed,
            gvar='gvar',
            ivar='id',
            tvar='year',
            cohorts=[4, 5],
            T_max=6,
            never_treated_values=[0],
            transform_type='detrend'
        )
        
        assert len(cohort_effects) == 2
        for effect in cohort_effects:
            assert not np.isnan(effect.att)
            assert effect.se > 0


# =============================================================================
# Utility Function Tests
# =============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_cohort_effects_to_dataframe(self):
        """Test conversion to DataFrame."""
        data = create_test_data_with_cohorts([3, 4], n_never_treated=2, T_max=5)
        transformed = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        effects = aggregate_to_cohort(
            transformed, 'gvar', 'id', 'year',
            cohorts=[3, 4], T_max=5, never_treated_values=[0]
        )
        
        df = cohort_effects_to_dataframe(effects)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'cohort' in df.columns
        assert 'att' in df.columns
        assert 'se' in df.columns
    
    def test_empty_cohort_effects_to_dataframe(self):
        """Test conversion of empty list."""
        df = cohort_effects_to_dataframe([])
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert 'cohort' in df.columns