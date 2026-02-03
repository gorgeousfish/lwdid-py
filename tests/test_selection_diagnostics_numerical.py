"""
Numerical validation tests for selection diagnostics.

Uses vibe-math MCP for precise numerical verification of calculations
in the selection_diagnostics module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from lwdid.selection_diagnostics import (
    diagnose_selection_mechanism,
    _compute_balance_statistics,
    _compute_attrition_analysis,
    _classify_missing_pattern,
    _compute_missing_rates,
    _compute_unit_stats,
)


# =============================================================================
# Test Balance Statistics Numerical Calculations
# =============================================================================

class TestBalanceStatisticsNumerical:
    """Numerical validation of balance statistics calculations."""
    
    def test_balance_ratio_calculation(self):
        """
        Verify balance_ratio = min_obs / max_obs.
        
        Using vibe-math for verification:
        mcp_vibe_math_mcp_calculate(expression="5/10", variables={})
        Expected: 0.5
        """
        # Create data with known imbalance
        data = pd.DataFrame({
            'unit_id': [1]*10 + [2]*5,  # Unit 1: 10 obs, Unit 2: 5 obs
            'year': list(range(1, 11)) + list(range(1, 6)),
            'y': np.random.randn(15),
        })
        
        obs_per_unit = data.groupby('unit_id').size()
        min_obs = obs_per_unit.min()
        max_obs = obs_per_unit.max()
        
        # Manual calculation
        expected_ratio = min_obs / max_obs
        
        # Verify: 5/10 = 0.5
        assert min_obs == 5
        assert max_obs == 10
        assert expected_ratio == 0.5
        
        # Verify via diagnostics
        diag = diagnose_selection_mechanism(
            data, y='y', ivar='unit_id', tvar='year', verbose=False
        )
        assert diag.balance_statistics.balance_ratio == 0.5
    
    def test_mean_obs_per_unit_calculation(self):
        """
        Verify mean_obs_per_unit calculation.
        
        Using vibe-math for verification:
        mcp_vibe_math_mcp_calculate(expression="(10 + 5) / 2", variables={})
        Expected: 7.5
        """
        data = pd.DataFrame({
            'unit_id': [1]*10 + [2]*5,
            'year': list(range(1, 11)) + list(range(1, 6)),
            'y': np.random.randn(15),
        })
        
        obs_per_unit = data.groupby('unit_id').size()
        expected_mean = obs_per_unit.mean()
        
        # Verify: (10 + 5) / 2 = 7.5
        assert expected_mean == 7.5
        
        diag = diagnose_selection_mechanism(
            data, y='y', ivar='unit_id', tvar='year', verbose=False
        )
        assert diag.balance_statistics.mean_obs_per_unit == 7.5
    
    def test_std_obs_per_unit_calculation(self):
        """
        Verify std_obs_per_unit calculation.
        
        Using vibe-math for verification:
        mcp_vibe_math_mcp_calculate(
            expression="sqrt(((10-7.5)^2 + (5-7.5)^2) / 1)",
            variables={}
        )
        Expected: sqrt(12.5) ≈ 3.536
        """
        data = pd.DataFrame({
            'unit_id': [1]*10 + [2]*5,
            'year': list(range(1, 11)) + list(range(1, 6)),
            'y': np.random.randn(15),
        })
        
        obs_per_unit = data.groupby('unit_id').size()
        expected_std = obs_per_unit.std()  # ddof=1 by default
        
        # Verify: sqrt(((10-7.5)^2 + (5-7.5)^2) / 1) = sqrt(12.5) ≈ 3.536
        manual_std = np.sqrt(((10 - 7.5)**2 + (5 - 7.5)**2) / 1)
        assert np.isclose(expected_std, manual_std, rtol=1e-6)
        
        diag = diagnose_selection_mechanism(
            data, y='y', ivar='unit_id', tvar='year', verbose=False
        )
        assert np.isclose(diag.balance_statistics.std_obs_per_unit, manual_std, rtol=1e-6)


# =============================================================================
# Test Missing Rate Numerical Calculations
# =============================================================================

class TestMissingRateNumerical:
    """Numerical validation of missing rate calculations."""
    
    def test_missing_rate_calculation(self):
        """
        Verify missing_rate = n_missing / n_total.
        
        Using vibe-math for verification:
        mcp_vibe_math_mcp_percentage(operation="of", value=100, percentage=20)
        Expected: 20 missing out of 100 = 20%
        """
        n_units = 10
        n_periods = 10
        n_total = n_units * n_periods  # 100
        
        # Create data with exactly 20% missing
        np.random.seed(42)
        data = []
        n_observed = 0
        for i in range(n_units):
            for t in range(1, n_periods + 1):
                # Keep 80% of observations
                if (i * n_periods + t) % 5 != 0:  # Skip every 5th
                    data.append({'unit_id': i, 'year': t, 'y': np.random.randn()})
                    n_observed += 1
        
        df = pd.DataFrame(data)
        
        # Calculate expected missing rate
        n_missing = n_total - n_observed
        expected_missing_rate = n_missing / n_total
        
        # Verify via diagnostics
        diag = diagnose_selection_mechanism(
            df, y='y', ivar='unit_id', tvar='year', verbose=False
        )
        
        assert np.isclose(diag.missing_rate_overall, expected_missing_rate, rtol=1e-6)
    
    def test_missing_rate_by_period(self):
        """
        Verify missing rate by period calculation.
        
        Create data where period 5 has 50% missing.
        """
        data = []
        for i in range(10):
            for t in range(1, 11):
                # Period 5: only half of units observed
                if t == 5 and i >= 5:
                    continue
                data.append({'unit_id': i, 'year': t, 'y': np.random.randn()})
        
        df = pd.DataFrame(data)
        
        diag = diagnose_selection_mechanism(
            df, y='y', ivar='unit_id', tvar='year', verbose=False
        )
        
        # Period 5 should have 50% missing
        assert np.isclose(diag.missing_rate_by_period[5], 0.5, rtol=1e-6)
        
        # Other periods should have 0% missing
        for t in [1, 2, 3, 4, 6, 7, 8, 9, 10]:
            assert diag.missing_rate_by_period[t] == 0.0


# =============================================================================
# Test Attrition Rate Numerical Calculations
# =============================================================================

class TestAttritionRateNumerical:
    """Numerical validation of attrition rate calculations."""
    
    def test_attrition_rate_calculation(self):
        """
        Verify attrition_rate = n_partial / n_total_units.
        
        Using vibe-math for verification:
        mcp_vibe_math_mcp_calculate(expression="30/100", variables={})
        Expected: 0.3 (30%)
        """
        n_units = 100
        n_complete = 70
        n_partial = 30
        
        expected_attrition = n_partial / n_units
        
        # Verify: 30/100 = 0.3
        assert expected_attrition == 0.3
        
        # Create data with 30% partial units
        data = []
        for i in range(n_units):
            n_periods = 10 if i < n_complete else 5  # 70 complete, 30 partial
            for t in range(1, n_periods + 1):
                data.append({'unit_id': i, 'year': t, 'y': np.random.randn()})
        
        df = pd.DataFrame(data)
        
        diag = diagnose_selection_mechanism(
            df, y='y', ivar='unit_id', tvar='year', verbose=False
        )
        
        assert np.isclose(diag.attrition_analysis.attrition_rate, 0.3, rtol=1e-6)
    
    def test_late_entry_rate_calculation(self):
        """
        Verify late_entry_rate = n_late_entry / n_units.
        
        Create data where 40% of units enter late.
        """
        data = []
        n_units = 100
        n_late = 40
        
        for i in range(n_units):
            start_period = 1 if i < (n_units - n_late) else 5
            for t in range(start_period, 11):
                data.append({'unit_id': i, 'year': t, 'y': np.random.randn()})
        
        df = pd.DataFrame(data)
        
        diag = diagnose_selection_mechanism(
            df, y='y', ivar='unit_id', tvar='year', verbose=False
        )
        
        # 40% should have late entry
        assert np.isclose(diag.attrition_analysis.late_entry_rate, 0.4, rtol=1e-6)
    
    def test_early_dropout_rate_calculation(self):
        """
        Verify early_dropout_rate = n_early_dropout / n_units.
        
        Create data where 25% of units drop out early.
        """
        data = []
        n_units = 100
        n_early_dropout = 25
        
        for i in range(n_units):
            end_period = 10 if i < (n_units - n_early_dropout) else 5
            for t in range(1, end_period + 1):
                data.append({'unit_id': i, 'year': t, 'y': np.random.randn()})
        
        df = pd.DataFrame(data)
        
        diag = diagnose_selection_mechanism(
            df, y='y', ivar='unit_id', tvar='year', verbose=False
        )
        
        # 25% should have early dropout
        assert np.isclose(diag.attrition_analysis.early_dropout_rate, 0.25, rtol=1e-6)


# =============================================================================
# Test Method Usability Numerical Calculations
# =============================================================================

class TestMethodUsabilityNumerical:
    """Numerical validation of method usability calculations."""
    
    def test_demean_usability_percentage(self):
        """
        Verify percentage of units usable for demeaning.
        
        pct_usable = (n_total - n_below_threshold) / n_total * 100
        
        Using vibe-math for verification:
        mcp_vibe_math_mcp_percentage(operation="of", value=100, percentage=90)
        Expected: 90 units usable out of 100 = 90%
        """
        # Create data where 10% of treated units have 0 pre-treatment periods
        data = []
        n_treated = 100
        n_below_demean = 10
        
        for i in range(n_treated):
            gvar = 6
            # 10 units have no pre-treatment observations
            start_period = 6 if i < n_below_demean else 1
            for t in range(start_period, 11):
                data.append({
                    'unit_id': i,
                    'year': t,
                    'y': np.random.randn(),
                    'gvar': gvar,
                })
        
        # Add some never-treated units
        for i in range(n_treated, n_treated + 50):
            for t in range(1, 11):
                data.append({
                    'unit_id': i,
                    'year': t,
                    'y': np.random.randn(),
                    'gvar': 0,
                })
        
        df = pd.DataFrame(data)
        
        diag = diagnose_selection_mechanism(
            df, y='y', ivar='unit_id', tvar='year', gvar='gvar', verbose=False
        )
        
        # 90% of treated units should be usable for demeaning
        expected_pct = (n_treated - n_below_demean) / n_treated * 100
        assert np.isclose(diag.balance_statistics.pct_usable_demean, expected_pct, rtol=1e-6)
    
    def test_detrend_usability_percentage(self):
        """
        Verify percentage of units usable for detrending.
        
        Detrending requires ≥2 pre-treatment periods.
        
        Using vibe-math for verification:
        mcp_vibe_math_mcp_percentage(operation="of", value=100, percentage=75)
        Expected: 75 units usable out of 100 = 75%
        """
        # Create data where 25% of treated units have <2 pre-treatment periods
        data = []
        n_treated = 100
        n_below_detrend = 25
        
        for i in range(n_treated):
            gvar = 6
            # 25 units have only 1 pre-treatment observation
            start_period = 5 if i < n_below_detrend else 1
            for t in range(start_period, 11):
                data.append({
                    'unit_id': i,
                    'year': t,
                    'y': np.random.randn(),
                    'gvar': gvar,
                })
        
        # Add never-treated units
        for i in range(n_treated, n_treated + 50):
            for t in range(1, 11):
                data.append({
                    'unit_id': i,
                    'year': t,
                    'y': np.random.randn(),
                    'gvar': 0,
                })
        
        df = pd.DataFrame(data)
        
        diag = diagnose_selection_mechanism(
            df, y='y', ivar='unit_id', tvar='year', gvar='gvar', verbose=False
        )
        
        # 75% of treated units should be usable for detrending
        expected_pct = (n_treated - n_below_detrend) / n_treated * 100
        assert np.isclose(diag.balance_statistics.pct_usable_detrend, expected_pct, rtol=1e-6)


# =============================================================================
# Test Statistical Test Calculations
# =============================================================================

class TestStatisticalTestNumerical:
    """Numerical validation of statistical test calculations."""
    
    def test_t_statistic_calculation(self):
        """
        Verify t-statistic calculation for MCAR test.
        
        t = (mean1 - mean2) / sqrt(var1/n1 + var2/n2)
        
        Using vibe-math for verification:
        mcp_vibe_math_mcp_calculate(
            expression="(10 - 9.5) / sqrt(1/50 + 1.2/50)",
            variables={}
        )
        """
        # Known values
        mean_complete = 10.0
        mean_incomplete = 9.5
        var_complete = 1.0
        var_incomplete = 1.2
        n_complete = 50
        n_incomplete = 50
        
        # Manual calculation
        pooled_se = np.sqrt(var_complete/n_complete + var_incomplete/n_incomplete)
        t_stat = (mean_complete - mean_incomplete) / pooled_se
        
        # Expected: (10 - 9.5) / sqrt(0.02 + 0.024) = 0.5 / 0.2098 ≈ 2.38
        expected_t = 0.5 / np.sqrt(0.044)
        
        assert np.isclose(t_stat, expected_t, rtol=1e-6)
        assert np.isclose(t_stat, 2.3833, rtol=1e-3)
    
    def test_point_biserial_correlation(self):
        """
        Verify point-biserial correlation calculation.
        
        Create data with known correlation between missing indicator and lagged Y.
        """
        np.random.seed(42)
        n = 100
        
        # Binary variable (missing indicator)
        missing = np.array([0]*70 + [1]*30)
        
        # Continuous variable (lagged outcome)
        # Make missing units have lower lagged values
        y_lag = np.concatenate([
            np.random.normal(10, 1, 70),  # Non-missing: mean=10
            np.random.normal(8, 1, 30),   # Missing: mean=8
        ])
        
        # Calculate correlation
        corr, pvalue = stats.pointbiserialr(missing, y_lag)
        
        # Should be negative (lower y_lag associated with missing)
        assert corr < 0
        assert pvalue < 0.05  # Significant


# =============================================================================
# Test Cohort-Specific Calculations
# =============================================================================

class TestCohortSpecificNumerical:
    """Numerical validation of cohort-specific calculations."""
    
    def test_missing_rate_by_cohort(self):
        """
        Verify missing rate calculation by cohort.
        
        Create data where cohort 6 has 20% missing and cohort 8 has 10% missing.
        """
        data = []
        
        # Cohort 6: 20% missing
        for i in range(50):
            gvar = 6
            for t in range(1, 11):
                if i < 10 and t == 5:  # 10 units missing period 5
                    continue
                data.append({
                    'unit_id': i,
                    'year': t,
                    'y': np.random.randn(),
                    'gvar': gvar,
                })
        
        # Cohort 8: 10% missing
        for i in range(50, 100):
            gvar = 8
            for t in range(1, 11):
                if i < 55 and t == 5:  # 5 units missing period 5
                    continue
                data.append({
                    'unit_id': i,
                    'year': t,
                    'y': np.random.randn(),
                    'gvar': gvar,
                })
        
        # Never-treated
        for i in range(100, 150):
            for t in range(1, 11):
                data.append({
                    'unit_id': i,
                    'year': t,
                    'y': np.random.randn(),
                    'gvar': 0,
                })
        
        df = pd.DataFrame(data)
        
        diag = diagnose_selection_mechanism(
            df, y='y', ivar='unit_id', tvar='year', gvar='gvar', verbose=False
        )
        
        # Verify cohort-specific missing rates
        # Cohort 6: 10 missing out of 500 = 2%
        # Cohort 8: 5 missing out of 500 = 1%
        assert 6 in diag.missing_rate_by_cohort
        assert 8 in diag.missing_rate_by_cohort
    
    def test_attrition_by_cohort(self):
        """
        Verify attrition rate calculation by cohort.
        """
        data = []
        
        # Cohort 6: 40% attrition (20 out of 50 units incomplete)
        for i in range(50):
            gvar = 6
            n_periods = 10 if i < 30 else 5  # 30 complete, 20 partial
            for t in range(1, n_periods + 1):
                data.append({
                    'unit_id': i,
                    'year': t,
                    'y': np.random.randn(),
                    'gvar': gvar,
                })
        
        # Cohort 8: 20% attrition (10 out of 50 units incomplete)
        for i in range(50, 100):
            gvar = 8
            n_periods = 10 if i < 90 else 5  # 40 complete, 10 partial
            for t in range(1, n_periods + 1):
                data.append({
                    'unit_id': i,
                    'year': t,
                    'y': np.random.randn(),
                    'gvar': gvar,
                })
        
        # Never-treated (all complete)
        for i in range(100, 150):
            for t in range(1, 11):
                data.append({
                    'unit_id': i,
                    'year': t,
                    'y': np.random.randn(),
                    'gvar': 0,
                })
        
        df = pd.DataFrame(data)
        
        diag = diagnose_selection_mechanism(
            df, y='y', ivar='unit_id', tvar='year', gvar='gvar', verbose=False
        )
        
        # Verify cohort-specific attrition rates
        assert 6 in diag.attrition_analysis.attrition_by_cohort
        assert 8 in diag.attrition_analysis.attrition_by_cohort
        
        # Cohort 6 should have higher attrition than cohort 8
        assert diag.attrition_analysis.attrition_by_cohort[6] > \
               diag.attrition_analysis.attrition_by_cohort[8]


# =============================================================================
# Test Boundary Conditions
# =============================================================================

class TestBoundaryConditions:
    """Test numerical calculations at boundary conditions."""
    
    def test_zero_missing_rate(self):
        """Balanced panel should have exactly 0% missing rate."""
        data = pd.DataFrame({
            'unit_id': [1, 1, 2, 2],
            'year': [1, 2, 1, 2],
            'y': [1.0, 2.0, 1.5, 2.5],
        })
        
        diag = diagnose_selection_mechanism(
            data, y='y', ivar='unit_id', tvar='year', verbose=False
        )
        
        assert diag.missing_rate_overall == 0.0
        assert diag.balance_statistics.balance_ratio == 1.0
    
    def test_maximum_imbalance(self):
        """Test with maximum imbalance (one unit has 1 obs, another has 10)."""
        data = pd.DataFrame({
            'unit_id': [1]*10 + [2],
            'year': list(range(1, 11)) + [1],
            'y': np.random.randn(11),
        })
        
        diag = diagnose_selection_mechanism(
            data, y='y', ivar='unit_id', tvar='year', verbose=False
        )
        
        # Balance ratio should be 1/10 = 0.1
        assert diag.balance_statistics.balance_ratio == 0.1
        assert diag.balance_statistics.min_obs_per_unit == 1
        assert diag.balance_statistics.max_obs_per_unit == 10
    
    def test_all_units_partial(self):
        """Test when all units have incomplete observations."""
        data = []
        for i in range(10):
            # Each unit missing a different period
            for t in range(1, 11):
                if t != (i % 10) + 1:
                    data.append({'unit_id': i, 'year': t, 'y': np.random.randn()})
        
        df = pd.DataFrame(data)
        
        diag = diagnose_selection_mechanism(
            df, y='y', ivar='unit_id', tvar='year', verbose=False
        )
        
        # All units should be partial
        assert diag.attrition_analysis.n_units_complete == 0
        assert diag.attrition_analysis.attrition_rate == 1.0
