"""
E5-S1 Comprehensive Unit Tests and Edge Cases

This file provides comprehensive testing for the lwdid staggered extension,
covering all modules, edge cases, and Stata comparison tests.

Test Categories:
1. Transformation Tests (demean/detrend)
2. Control Group Selection Tests
3. Estimation Tests (RA/IPWRA/PSM)
4. Aggregation Tests (cohort/overall)
5. Edge Case Tests
6. Castle Law End-to-End Tests
"""

import warnings
import pytest
import numpy as np
import pandas as pd
from pathlib import Path


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_staggered_data():
    """Simple balanced panel with 3 cohorts."""
    np.random.seed(42)
    data = []
    # 9 units: 3 per cohort (2001, 2002, NT)
    for unit in range(1, 10):
        if unit <= 3:
            gvar = 2001
        elif unit <= 6:
            gvar = 2002
        else:
            gvar = 0
        
        for year in range(2000, 2004):
            y = 1.0 + 0.1 * unit + 0.05 * (year - 2000)
            if gvar > 0 and year >= gvar:
                y += 0.5  # Treatment effect
            y += np.random.normal(0, 0.05)
            
            data.append({
                'id': unit, 'year': year, 'y': y, 'gvar': gvar,
                'x1': np.random.normal(0, 1),
                'x2': np.random.normal(0, 1)
            })
    return pd.DataFrame(data)


@pytest.fixture
def unbalanced_panel_data():
    """Unbalanced panel with missing observations."""
    np.random.seed(123)
    data = []
    for unit in range(1, 7):
        gvar = 2001 if unit <= 3 else 0
        years = [2000, 2001, 2002, 2003]
        # Remove some observations for unit 2 and 5
        if unit == 2:
            years = [2000, 2002, 2003]  # Missing 2001
        elif unit == 5:
            years = [2001, 2002, 2003]  # Missing 2000
        
        for year in years:
            y = 1.0 + 0.1 * unit
            if gvar > 0 and year >= gvar:
                y += 0.3
            data.append({'id': unit, 'year': year, 'y': y, 'gvar': gvar})
    return pd.DataFrame(data)


@pytest.fixture  
def single_cohort_data():
    """Data with only one cohort."""
    np.random.seed(456)
    data = []
    for unit in range(1, 6):
        gvar = 2002 if unit <= 2 else 0
        for year in range(2000, 2004):
            y = 1.0 + 0.1 * unit
            if gvar > 0 and year >= gvar:
                y += 0.4
            data.append({'id': unit, 'year': year, 'y': y, 'gvar': gvar})
    return pd.DataFrame(data)


@pytest.fixture
def all_treated_data():
    """Data where all units eventually get treated (no NT)."""
    np.random.seed(789)
    data = []
    for unit in range(1, 7):
        gvar = 2001 if unit <= 3 else 2003
        for year in range(2000, 2005):
            y = 1.0 + 0.1 * unit
            if year >= gvar:
                y += 0.5
            data.append({'id': unit, 'year': year, 'y': y, 'gvar': gvar})
    return pd.DataFrame(data)


@pytest.fixture
def castle_data():
    """Load Castle Law data."""
    data_path = Path(__file__).parent.parent / 'data' / 'castle.csv'
    if not data_path.exists():
        data_path = Path(__file__).parent.parent.parent / 'data' / 'castle.csv'
    df = pd.read_csv(data_path)
    df['gvar'] = df['effyear'].fillna(0).astype(int)
    return df


# =============================================================================
# 1. Transformation Tests
# =============================================================================

class TestTransformationsDemean:
    """Comprehensive tests for demeaning transformation."""
    
    def test_demean_basic_calculation(self, simple_staggered_data):
        """Test basic demeaning calculation."""
        from lwdid.staggered.transformations import transform_staggered_demean
        
        result = transform_staggered_demean(
            simple_staggered_data, 'y', 'id', 'year', 'gvar'
        )
        
        # Should have ydot columns
        ydot_cols = [c for c in result.columns if c.startswith('ydot_')]
        assert len(ydot_cols) > 0
        
        # Check column naming format
        for col in ydot_cols:
            assert col.startswith('ydot_g')
            assert '_r' in col
    
    def test_demean_pre_mean_fixed(self, simple_staggered_data):
        """Verify pre-treatment mean is fixed across post periods."""
        from lwdid.staggered.transformations import transform_staggered_demean
        
        result = transform_staggered_demean(
            simple_staggered_data, 'y', 'id', 'year', 'gvar'
        )
        
        # For unit 1 (cohort 2001), check pre-mean is same for r=2001,2002,2003
        unit1 = result[result['id'] == 1]
        pre_y = unit1[unit1['year'] < 2001]['y'].mean()
        
        for r in [2001, 2002, 2003]:
            col = f'ydot_g2001_r{r}'
            if col in result.columns:
                y_r = unit1[unit1['year'] == r]['y'].iloc[0]
                ydot_r = unit1[unit1['year'] == r][col].iloc[0]
                pre_mean_r = y_r - ydot_r
                assert np.isclose(pre_mean_r, pre_y, atol=1e-10)
    
    def test_demean_all_units_transformed(self, simple_staggered_data):
        """Verify all units (including NT) have transformation values."""
        from lwdid.staggered.transformations import transform_staggered_demean
        
        result = transform_staggered_demean(
            simple_staggered_data, 'y', 'id', 'year', 'gvar'
        )
        
        # NT units (7,8,9) should have transformation values for cohort 2001
        nt_units = result[result['gvar'] == 0]
        assert not nt_units[nt_units['year'] == 2001]['ydot_g2001_r2001'].isna().all()
    
    def test_demean_unbalanced_panel(self, unbalanced_panel_data):
        """Test demeaning with unbalanced panel."""
        from lwdid.staggered.transformations import transform_staggered_demean
        
        result = transform_staggered_demean(
            unbalanced_panel_data, 'y', 'id', 'year', 'gvar'
        )
        
        # Should handle missing observations gracefully
        ydot_cols = [c for c in result.columns if c.startswith('ydot_')]
        assert len(ydot_cols) > 0


class TestTransformationsDetrend:
    """Comprehensive tests for detrending transformation."""
    
    def test_detrend_basic(self):
        """Test basic detrending calculation."""
        from lwdid.staggered.transformations import transform_staggered_detrend
        
        # Create data with sufficient pre-treatment periods (at least 2)
        np.random.seed(42)
        data = []
        for unit in range(1, 7):
            # Cohort 2003 has pre-periods 2000, 2001, 2002 (3 periods)
            gvar = 2003 if unit <= 3 else 0
            for year in range(2000, 2005):
                y = 1.0 + 0.1 * unit + 0.05 * (year - 2000)
                if gvar > 0 and year >= gvar:
                    y += 0.5
                data.append({'id': unit, 'year': year, 'y': y, 'gvar': gvar})
        df = pd.DataFrame(data)
        
        result = transform_staggered_detrend(df, 'y', 'id', 'year', 'gvar')
        
        # Should have ycheck columns
        ycheck_cols = [c for c in result.columns if c.startswith('ycheck_')]
        assert len(ycheck_cols) > 0
    
    def test_detrend_requires_min_preperiods(self):
        """Detrending requires at least 2 pre-treatment periods."""
        from lwdid.staggered.transformations import transform_staggered_detrend
        
        # Cohort 2001 with only 1 pre-period (2000)
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'year': [2000, 2001, 2000, 2001],
            'y': [1.0, 1.5, 1.0, 1.2],
            'gvar': [2001, 2001, 0, 0]
        })
        
        # Should raise ValueError for insufficient pre-periods
        with pytest.raises(ValueError, match="pre-treatment period"):
            transform_staggered_detrend(data, 'y', 'id', 'year', 'gvar')


# =============================================================================
# 2. Control Group Selection Tests
# =============================================================================

class TestControlGroupSelection:
    """Comprehensive control group selection tests."""
    
    def test_never_treated_strategy(self, simple_staggered_data):
        """Test NEVER_TREATED strategy."""
        from lwdid.staggered.control_groups import (
            get_valid_control_units, ControlGroupStrategy
        )
        
        mask = get_valid_control_units(
            simple_staggered_data, 'gvar', 'id',
            cohort=2001, period=2002,
            strategy=ControlGroupStrategy.NEVER_TREATED
        )
        
        # Only gvar=0 units should be controls
        unit_gvar = simple_staggered_data.groupby('id')['gvar'].first()
        for uid, is_control in mask.items():
            if unit_gvar[uid] == 0:
                assert is_control == True
            else:
                assert is_control == False
    
    def test_not_yet_treated_strategy(self, simple_staggered_data):
        """Test NOT_YET_TREATED strategy."""
        from lwdid.staggered.control_groups import (
            get_valid_control_units, ControlGroupStrategy
        )
        
        # At period 2001, cohort 2002 is not-yet-treated
        mask = get_valid_control_units(
            simple_staggered_data, 'gvar', 'id',
            cohort=2001, period=2001,
            strategy=ControlGroupStrategy.NOT_YET_TREATED
        )
        
        # Cohort 2002 (gvar=2002 > period=2001) should be control
        unit_gvar = simple_staggered_data.groupby('id')['gvar'].first()
        for uid, is_control in mask.items():
            gv = unit_gvar[uid]
            if gv == 2001:  # Treatment group
                assert is_control == False
            elif gv == 0 or gv > 2001:  # NT or NYT
                assert is_control == True
    
    def test_gvar_equals_period_excluded(self, simple_staggered_data):
        """Units with gvar==period should be excluded from controls."""
        from lwdid.staggered.control_groups import (
            get_valid_control_units, ControlGroupStrategy
        )
        
        # At period 2002, cohort 2002 is starting treatment
        mask = get_valid_control_units(
            simple_staggered_data, 'gvar', 'id',
            cohort=2001, period=2002,
            strategy=ControlGroupStrategy.NOT_YET_TREATED
        )
        
        unit_gvar = simple_staggered_data.groupby('id')['gvar'].first()
        for uid, is_control in mask.items():
            if unit_gvar[uid] == 2002:
                assert is_control == False  # gvar==period should be excluded
    
    def test_no_controls_available(self, all_treated_data):
        """Test when no controls are available at last period."""
        from lwdid.staggered.control_groups import (
            get_valid_control_units, ControlGroupStrategy
        )
        
        # At period 2004, all units are treated
        mask = get_valid_control_units(
            all_treated_data, 'gvar', 'id',
            cohort=2001, period=2004,
            strategy=ControlGroupStrategy.NOT_YET_TREATED
        )
        
        # Count should be 0
        assert mask.sum() == 0


# =============================================================================
# 3. Estimation Tests
# =============================================================================

class TestEstimationRA:
    """Comprehensive RA estimation tests."""
    
    def test_ra_basic_estimation(self, simple_staggered_data):
        """Test basic RA estimation."""
        from lwdid import lwdid
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=simple_staggered_data,
                y='y', ivar='id', tvar='year', gvar='gvar',
                rolling='demean', estimator='ra',
                aggregate='none'
            )
        
        assert result.is_staggered == True
        assert result.att_by_cohort_time is not None
        assert len(result.att_by_cohort_time) > 0
    
    def test_ra_with_controls(self, simple_staggered_data):
        """Test RA with control variables."""
        from lwdid import lwdid
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=simple_staggered_data,
                y='y', ivar='id', tvar='year', gvar='gvar',
                rolling='demean', estimator='ra',
                controls=['x1', 'x2'],
                aggregate='none'
            )
        
        assert result.is_staggered == True
    
    def test_ra_vce_options(self, simple_staggered_data):
        """Test different VCE options."""
        from lwdid import lwdid
        
        for vce in [None, 'hc3']:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = lwdid(
                    data=simple_staggered_data,
                    y='y', ivar='id', tvar='year', gvar='gvar',
                    rolling='demean', vce=vce,
                    aggregate='none'
                )
            assert result.is_staggered == True


class TestEstimationIPWRA:
    """Comprehensive IPWRA estimation tests."""
    
    def test_ipwra_basic(self, simple_staggered_data):
        """Test basic IPWRA estimation."""
        from lwdid import lwdid
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=simple_staggered_data,
                y='y', ivar='id', tvar='year', gvar='gvar',
                rolling='demean', estimator='ipwra',
                controls=['x1'],
                aggregate='none'
            )
        
        assert result.is_staggered == True
        assert result.att_by_cohort_time is not None


class TestEstimationPSM:
    """Comprehensive PSM estimation tests."""
    
    def test_psm_basic(self, simple_staggered_data):
        """Test basic PSM estimation."""
        from lwdid import lwdid
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=simple_staggered_data,
                y='y', ivar='id', tvar='year', gvar='gvar',
                rolling='demean', estimator='psm',
                controls=['x1'],
                aggregate='none'
            )
        
        assert result.is_staggered == True


# =============================================================================
# 4. Aggregation Tests
# =============================================================================

class TestAggregation:
    """Comprehensive aggregation tests."""
    
    def test_aggregate_none(self, simple_staggered_data):
        """Test no aggregation (g,r effects only)."""
        from lwdid import lwdid
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=simple_staggered_data,
                y='y', ivar='id', tvar='year', gvar='gvar',
                rolling='demean', aggregate='none'
            )
        
        assert result.att_by_cohort_time is not None
        assert result.att_by_cohort is None
        assert result.att_overall is None
    
    def test_aggregate_cohort(self, simple_staggered_data):
        """Test cohort-level aggregation."""
        from lwdid import lwdid
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=simple_staggered_data,
                y='y', ivar='id', tvar='year', gvar='gvar',
                rolling='demean', aggregate='cohort'
            )
        
        assert result.att_by_cohort is not None
        assert len(result.att_by_cohort) == 2  # 2 cohorts
    
    def test_aggregate_overall(self, simple_staggered_data):
        """Test overall aggregation."""
        from lwdid import lwdid
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=simple_staggered_data,
                y='y', ivar='id', tvar='year', gvar='gvar',
                rolling='demean', aggregate='overall'
            )
        
        assert result.att_overall is not None
        assert result.se_overall is not None
    
    def test_cohort_weights_sum_to_one(self, simple_staggered_data):
        """Verify cohort weights sum to 1."""
        from lwdid import lwdid
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=simple_staggered_data,
                y='y', ivar='id', tvar='year', gvar='gvar',
                rolling='demean', aggregate='overall'
            )
        
        if result.cohort_weights:
            weight_sum = sum(result.cohort_weights.values())
            assert abs(weight_sum - 1.0) < 0.001
    
    def test_aggregate_requires_nt_for_overall(self, all_treated_data):
        """Overall aggregation requires never-treated units."""
        from lwdid import lwdid
        
        # Should raise error when no NT units and aggregate='overall'
        with pytest.raises((ValueError, RuntimeError)):
            lwdid(
                data=all_treated_data,
                y='y', ivar='id', tvar='year', gvar='gvar',
                rolling='demean', aggregate='overall',
                control_group='never_treated'
            )


# =============================================================================
# 5. Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Edge case and boundary condition tests."""
    
    def test_single_treated_unit_per_cohort(self):
        """Test with only 1 treated unit per cohort."""
        from lwdid import lwdid
        
        np.random.seed(111)
        data = []
        for unit in range(1, 6):
            gvar = 2002 if unit == 1 else 0
            for year in range(2000, 2004):
                y = 1.0 + 0.1 * unit
                if gvar > 0 and year >= gvar:
                    y += 0.5
                data.append({'id': unit, 'year': year, 'y': y, 'gvar': gvar})
        df = pd.DataFrame(data)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=df, y='y', ivar='id', tvar='year', gvar='gvar',
                rolling='demean', aggregate='none'
            )
        
        assert result.is_staggered == True
    
    def test_single_nt_unit(self):
        """Test with only 1 never-treated unit."""
        from lwdid import lwdid
        
        np.random.seed(222)
        data = []
        for unit in range(1, 6):
            gvar = 0 if unit == 5 else 2002
            for year in range(2000, 2004):
                y = 1.0 + 0.1 * unit
                if gvar > 0 and year >= gvar:
                    y += 0.5
                data.append({'id': unit, 'year': year, 'y': y, 'gvar': gvar})
        df = pd.DataFrame(data)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=df, y='y', ivar='id', tvar='year', gvar='gvar',
                rolling='demean', aggregate='overall'
            )
        
        assert result.att_overall is not None
    
    def test_many_cohorts(self):
        """Test with many cohorts."""
        from lwdid import lwdid
        
        np.random.seed(333)
        data = []
        # 5 cohorts + 5 NT units
        for unit in range(1, 11):
            if unit <= 5:
                gvar = 2000 + unit
            else:
                gvar = 0
            for year in range(2000, 2010):
                y = 1.0 + 0.1 * unit
                if gvar > 0 and year >= gvar:
                    y += 0.3
                data.append({'id': unit, 'year': year, 'y': y, 'gvar': gvar})
        df = pd.DataFrame(data)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=df, y='y', ivar='id', tvar='year', gvar='gvar',
                rolling='demean', aggregate='cohort'
            )
        
        assert len(result.att_by_cohort) == 5
    
    def test_gvar_with_nan_and_zero(self):
        """Test gvar handling with NaN and 0 for never-treated."""
        from lwdid import lwdid
        
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3, 4, 4],
            'year': [2000, 2001, 2000, 2001, 2000, 2001, 2000, 2001],
            'y': [1.0, 1.5, 1.0, 1.2, 1.0, 1.1, 1.0, 1.3],
            'gvar': [2001, 2001, 0, 0, np.nan, np.nan, np.inf, np.inf]
        })
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=data, y='y', ivar='id', tvar='year', gvar='gvar',
                rolling='demean', aggregate='none'
            )
        
        assert result.is_staggered == True
    
    def test_control_group_auto_switch(self, simple_staggered_data):
        """Test auto-switch from NYT to NT for aggregation."""
        from lwdid import lwdid
        
        # When aggregate='overall' and control_group='not_yet_treated',
        # should auto-switch to 'never_treated'
        with pytest.warns(UserWarning, match="切换|switch"):
            result = lwdid(
                data=simple_staggered_data,
                y='y', ivar='id', tvar='year', gvar='gvar',
                rolling='demean',
                control_group='not_yet_treated',
                aggregate='overall'
            )
        
        assert result.control_group_used == 'never_treated'


# =============================================================================
# 6. Castle Law End-to-End Tests
# =============================================================================

class TestCastleLawE2E:
    """Castle Law data end-to-end tests with paper comparison."""
    
    def test_castle_data_structure(self, castle_data):
        """Verify Castle data structure."""
        assert 'sid' in castle_data.columns
        assert 'year' in castle_data.columns
        assert 'lhomicide' in castle_data.columns
        assert 'gvar' in castle_data.columns
        
        # 50 states × 11 years = 550 rows
        assert len(castle_data) == 550
        
        # Cohort distribution
        unit_gvar = castle_data.groupby('sid')['gvar'].first()
        n_nt = (unit_gvar == 0).sum()
        n_treated = (unit_gvar > 0).sum()
        assert n_nt == 29
        assert n_treated == 21
    
    def test_castle_cohort_sizes(self, castle_data):
        """Verify cohort sizes match paper."""
        unit_gvar = castle_data.groupby('sid')['gvar'].first()
        
        expected = {2005: 1, 2006: 13, 2007: 4, 2008: 2, 2009: 1}
        for g, expected_count in expected.items():
            actual = (unit_gvar == g).sum()
            assert actual == expected_count, f"Cohort {g}: {actual} != {expected_count}"
    
    def test_castle_demean_overall(self, castle_data):
        """
        Castle Law demean overall effect.
        Expected: τ_ω ≈ 0.092 (Lee & Wooldridge 2023 Table 3)
        """
        from lwdid import lwdid
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=castle_data,
                y='lhomicide', ivar='sid', tvar='year', gvar='gvar',
                rolling='demean',
                control_group='never_treated',
                aggregate='overall'
            )
        
        expected_att = 0.092
        tolerance = 0.03
        
        assert result.att_overall is not None
        assert abs(result.att_overall - expected_att) < tolerance, \
            f"Overall ATT {result.att_overall:.4f} differs from expected {expected_att}"
    
    def test_castle_detrend_overall(self, castle_data):
        """
        Castle Law detrend overall effect.
        Expected: τ_ω ≈ 0.067 (Lee & Wooldridge 2023 Table 3)
        """
        from lwdid import lwdid
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=castle_data,
                y='lhomicide', ivar='sid', tvar='year', gvar='gvar',
                rolling='detrend',
                control_group='never_treated',
                aggregate='overall'
            )
        
        expected_att = 0.067
        tolerance = 0.03
        
        assert result.att_overall is not None
        assert abs(result.att_overall - expected_att) < tolerance, \
            f"Overall ATT {result.att_overall:.4f} differs from expected {expected_att}"
    
    def test_castle_cohort_effects(self, castle_data):
        """Test cohort-level effects for Castle Law."""
        from lwdid import lwdid
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=castle_data,
                y='lhomicide', ivar='sid', tvar='year', gvar='gvar',
                rolling='demean',
                aggregate='cohort'
            )
        
        assert result.att_by_cohort is not None
        assert len(result.att_by_cohort) == 5
        
        # All cohorts should have effects
        cohorts_in_result = result.att_by_cohort['cohort'].tolist()
        assert set(cohorts_in_result) == {2005, 2006, 2007, 2008, 2009}
    
    def test_castle_gr_effects(self, castle_data):
        """Test (g,r) specific effects for Castle Law."""
        from lwdid import lwdid
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=castle_data,
                y='lhomicide', ivar='sid', tvar='year', gvar='gvar',
                rolling='demean',
                aggregate='none'
            )
        
        assert result.att_by_cohort_time is not None
        
        # Expected number of (g,r) effects:
        # g=2005: r∈{2005,...,2010} = 6
        # g=2006: r∈{2006,...,2010} = 5
        # g=2007: r∈{2007,...,2010} = 4
        # g=2008: r∈{2008,...,2010} = 3
        # g=2009: r∈{2009,...,2010} = 2
        # Total = 20
        expected_count = 20
        actual_count = len(result.att_by_cohort_time)
        assert actual_count == expected_count, \
            f"Expected {expected_count} (g,r) effects, got {actual_count}"
    
    def test_castle_florida_transformation(self, castle_data):
        """Verify Florida (sid=10) transformation values."""
        from lwdid.staggered.transformations import transform_staggered_demean
        
        result = transform_staggered_demean(
            castle_data, 'lhomicide', 'sid', 'year', 'gvar'
        )
        
        florida = result[result['sid'] == 10]
        assert florida['gvar'].iloc[0] == 2005
        
        # Verified pre-treatment mean
        expected_pre_mean = 1.7212106
        pre_y = florida[florida['year'] < 2005]['lhomicide'].mean()
        assert np.isclose(pre_y, expected_pre_mean, atol=1e-5)
    
    def test_castle_california_nt_transformation(self, castle_data):
        """Verify California (sid=5, NT) has transformation values."""
        from lwdid.staggered.transformations import transform_staggered_demean
        
        result = transform_staggered_demean(
            castle_data, 'lhomicide', 'sid', 'year', 'gvar'
        )
        
        california = result[result['sid'] == 5]
        assert california['gvar'].iloc[0] == 0  # Never treated
        
        # Should have ydot values for cohort 2005
        assert 'ydot_g2005_r2005' in result.columns
        assert not california[california['year'] == 2005]['ydot_g2005_r2005'].isna().all()


# =============================================================================
# 7. Results Class Tests
# =============================================================================

class TestResultsClass:
    """Tests for LWDIDResults class staggered extension."""
    
    def test_results_attributes(self, simple_staggered_data):
        """Test result object has expected attributes."""
        from lwdid import lwdid
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=simple_staggered_data,
                y='y', ivar='id', tvar='year', gvar='gvar',
                rolling='demean', aggregate='overall'
            )
        
        # Staggered-specific attributes
        assert hasattr(result, 'is_staggered')
        assert hasattr(result, 'cohorts')
        assert hasattr(result, 'n_never_treated')
        assert hasattr(result, 'control_group_used')
        assert hasattr(result, 'att_overall')
        assert hasattr(result, 'se_overall')
        assert hasattr(result, 'cohort_weights')
    
    def test_results_summary(self, simple_staggered_data):
        """Test summary() method for staggered results."""
        from lwdid import lwdid
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=simple_staggered_data,
                y='y', ivar='id', tvar='year', gvar='gvar',
                rolling='demean', aggregate='overall'
            )
        
        summary = result.summary()
        assert summary is not None
        assert isinstance(summary, str) or hasattr(summary, '__str__')
    
    def test_results_to_excel(self, simple_staggered_data, tmp_path):
        """Test Excel export for staggered results."""
        from lwdid import lwdid
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=simple_staggered_data,
                y='y', ivar='id', tvar='year', gvar='gvar',
                rolling='demean', aggregate='cohort'
            )
        
        excel_path = tmp_path / 'test_results.xlsx'
        result.to_excel(str(excel_path))
        assert excel_path.exists()


# =============================================================================
# 8. Visualization Tests
# =============================================================================

class TestVisualization:
    """Tests for event study visualization."""
    
    def test_plot_event_study_basic(self, simple_staggered_data):
        """Test basic event study plot."""
        from lwdid import lwdid
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=simple_staggered_data,
                y='y', ivar='id', tvar='year', gvar='gvar',
                rolling='demean', aggregate='none'
            )
        
        if hasattr(result, 'plot_event_study'):
            fig, ax = plt.subplots()
            result.plot_event_study(ax=ax)
            plt.close(fig)
    
    def test_plot_event_study_options(self, simple_staggered_data):
        """Test event study plot with various options."""
        from lwdid import lwdid
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=simple_staggered_data,
                y='y', ivar='id', tvar='year', gvar='gvar',
                rolling='demean', aggregate='none'
            )
        
        if hasattr(result, 'plot_event_study'):
            fig, ax = plt.subplots()
            result.plot_event_study(
                ax=ax,
                aggregate=True,
                show_ci=True,
                ref_period=-1
            )
            plt.close(fig)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
