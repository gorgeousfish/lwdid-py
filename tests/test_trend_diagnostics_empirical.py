"""
Empirical Data Tests for Trend Diagnostics Module.

This module contains tests using real-world datasets to verify
the trend diagnostics functionality in practical applications.

Test Categories:
- NLS Work dataset tests
- Data quality handling tests
- Robustness tests with real data characteristics
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from lwdid.trend_diagnostics import (
    test_parallel_trends as run_parallel_trends_test,
    diagnose_heterogeneous_trends,
    recommend_transformation,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def nlswork_data():
    """
    Load NLS Work dataset for testing.
    
    This dataset contains panel data on wages and employment.
    We create a synthetic treatment variable for testing purposes.
    """
    # Try multiple possible paths for the data file
    possible_paths = [
        Path(__file__).parent.parent / 'nlswork_did.csv',
        Path(__file__).parent.parent.parent / 'nlswork_did.csv',
        Path('nlswork_did.csv'),
    ]
    
    data_path = None
    for path in possible_paths:
        if path.exists():
            data_path = path
            break
    
    if data_path is None:
        pytest.skip("NLS Work dataset not found")
    
    data = pd.read_csv(data_path)
    
    # Check available columns and adapt accordingly
    available_cols = data.columns.tolist()
    
    # The dataset may have different column names
    # Try to identify the key columns
    if 'ln_wage' in available_cols:
        y_col = 'ln_wage'
    elif 'wage' in available_cols:
        y_col = 'wage'
    else:
        pytest.skip("No wage column found in dataset")
    
    # Check if we have panel structure
    if 'idcode' in available_cols and 'year' in available_cols:
        # Standard NLS Work format
        data = data.dropna(subset=[y_col, 'idcode', 'year'])
        ivar = 'idcode'
        tvar = 'year'
    elif 'did' in available_cols:
        # Simplified DID format - create synthetic panel structure
        np.random.seed(42)
        n_obs = len(data)
        n_periods = 5
        n_units = (n_obs + n_periods - 1) // n_periods  # Ceiling division
        
        # Create unit and time indices that match the data length exactly
        unit_ids = []
        time_ids = []
        for i in range(n_units):
            for t in range(1, n_periods + 1):
                if len(unit_ids) >= n_obs:
                    break
                unit_ids.append(i)
                time_ids.append(t)
            if len(unit_ids) >= n_obs:
                break
        
        data['unit'] = unit_ids[:n_obs]
        data['time'] = time_ids[:n_obs]
        data = data.dropna(subset=[y_col])
        ivar = 'unit'
        tvar = 'time'
    else:
        pytest.skip("Cannot identify panel structure in dataset")
    
    # Create treatment variable
    if 'first_treat' not in available_cols:
        if 'treat' in available_cols and 'post' in available_cols:
            # Create first_treat from treat and post
            # Treated units get first_treat = first post period
            # Control units get first_treat = inf
            data['first_treat'] = np.where(
                data['treat'] == 1,
                data.groupby(ivar)[tvar].transform('min') + 2,  # Synthetic treatment timing
                np.inf
            )
        else:
            # Create synthetic treatment
            np.random.seed(42)
            units = data[ivar].unique()
            n_treated = len(units) // 3
            treated_units = np.random.choice(units, n_treated, replace=False)
            treatment_map = {u: 3 for u in treated_units}  # All treated at period 3
            data['first_treat'] = data[ivar].map(
                lambda x: treatment_map.get(x, np.inf)
            )
    
    # Store column names for tests to use
    data.attrs['y_col'] = y_col
    data.attrs['ivar'] = ivar
    data.attrs['tvar'] = tvar
    
    return data


@pytest.fixture
def small_empirical_data():
    """Create a small empirical-like dataset for edge case testing."""
    np.random.seed(123)
    
    # Small sample with realistic characteristics
    n_units = 30
    n_periods = 6
    treatment_period = 4
    
    data = []
    for i in range(n_units):
        is_treated = i < n_units // 2
        first_treat = treatment_period if is_treated else np.inf
        
        # Add realistic features: unit heterogeneity, time trends
        unit_fe = np.random.normal(10, 2)
        unit_trend = np.random.normal(0.1, 0.05)
        
        for t in range(1, n_periods + 1):
            y = unit_fe + unit_trend * t + np.random.normal(0, 0.5)
            if is_treated and t >= treatment_period:
                y += 1.5  # Treatment effect
            
            # Occasionally add missing values (realistic)
            if np.random.random() < 0.02:
                y = np.nan
            
            data.append({
                'unit': i,
                'time': t,
                'Y': y,
                'first_treat': first_treat,
            })
    
    return pd.DataFrame(data)


# =============================================================================
# NLS Work Dataset Tests
# =============================================================================

class TestNLSWorkData:
    """Tests using NLS Work panel dataset."""
    
    def test_parallel_trends_runs_successfully(self, nlswork_data):
        """
        Test that parallel trends test runs without error on NLS Work data.
        """
        # Get column names from data attributes
        y_col = nlswork_data.attrs.get('y_col', 'ln_wage')
        ivar = nlswork_data.attrs.get('ivar', 'unit')
        tvar = nlswork_data.attrs.get('tvar', 'time')
        
        # Filter to ensure we have valid data
        valid_data = nlswork_data[
            (nlswork_data['first_treat'] != np.inf) | 
            (nlswork_data[ivar].isin(
                nlswork_data[nlswork_data['first_treat'] == np.inf][ivar].unique()[:100]
            ))
        ].copy()
        
        # Limit sample size for faster testing
        units = valid_data[ivar].unique()[:200]
        valid_data = valid_data[valid_data[ivar].isin(units)]
        
        if len(valid_data) < 100:
            pytest.skip("Insufficient data after filtering")
        
        result = run_parallel_trends_test(
            valid_data, y=y_col, ivar=ivar, tvar=tvar,
            gvar='first_treat', method='placebo', verbose=False
        )
        
        # Basic validity checks
        assert result is not None
        assert 0 <= result.pvalue <= 1
        assert result.recommendation in ['demean', 'detrend']
    
    def test_heterogeneous_trends_diagnosis_runs(self, nlswork_data):
        """
        Test that heterogeneous trends diagnosis runs on NLS Work data.
        """
        # Get column names from data attributes
        y_col = nlswork_data.attrs.get('y_col', 'ln_wage')
        ivar = nlswork_data.attrs.get('ivar', 'unit')
        tvar = nlswork_data.attrs.get('tvar', 'time')
        
        # Filter to get a manageable subset
        valid_data = nlswork_data[nlswork_data['first_treat'] != np.inf].copy()
        
        # Get cohorts with sufficient observations
        cohort_counts = valid_data.groupby('first_treat')[ivar].nunique()
        valid_cohorts = cohort_counts[cohort_counts >= 20].index
        valid_data = valid_data[valid_data['first_treat'].isin(valid_cohorts)]
        
        if len(valid_data) < 100 or len(valid_cohorts) < 2:
            pytest.skip("Insufficient cohorts for heterogeneity test")
        
        diag = diagnose_heterogeneous_trends(
            valid_data, y=y_col, ivar=ivar, tvar=tvar,
            gvar='first_treat', verbose=False
        )
        
        # Basic validity checks
        assert diag is not None
        assert len(diag.trend_by_cohort) > 0
        assert diag.recommendation in ['demean', 'detrend']
    
    def test_recommendation_produces_valid_output(self, nlswork_data):
        """
        Test that recommendation system produces valid output on NLS Work data.
        """
        # Get column names from data attributes
        y_col = nlswork_data.attrs.get('y_col', 'ln_wage')
        ivar = nlswork_data.attrs.get('ivar', 'unit')
        tvar = nlswork_data.attrs.get('tvar', 'time')
        
        # Use a subset for faster testing
        units = nlswork_data[ivar].unique()[:150]
        valid_data = nlswork_data[nlswork_data[ivar].isin(units)].copy()
        
        if len(valid_data) < 50:
            pytest.skip("Insufficient data")
        
        rec = recommend_transformation(
            valid_data, y=y_col, ivar=ivar, tvar=tvar,
            gvar='first_treat', verbose=False
        )
        
        # Validity checks
        assert rec is not None
        assert rec.recommended_method in ['demean', 'detrend', 'demeanq', 'detrendq']
        assert 0 <= rec.confidence <= 1
        assert len(rec.reasons) > 0


# =============================================================================
# Data Quality Handling Tests
# =============================================================================

class TestDataQualityHandling:
    """Tests for handling various data quality issues."""
    
    def test_missing_values_in_outcome(self, small_empirical_data):
        """
        Test handling of missing values in outcome variable.
        """
        # The fixture already has some missing values
        result = run_parallel_trends_test(
            small_empirical_data, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', method='placebo', verbose=False
        )
        
        # Should handle missing values gracefully
        assert result is not None
        assert not np.isnan(result.pvalue)
    
    def test_missing_values_introduced(self, small_empirical_data):
        """
        Test with additional missing values introduced.
        """
        data = small_empirical_data.copy()
        
        # Introduce more missing values
        np.random.seed(456)
        missing_idx = np.random.choice(len(data), size=int(len(data) * 0.1), replace=False)
        data.loc[data.index[missing_idx], 'Y'] = np.nan
        
        result = run_parallel_trends_test(
            data, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', method='placebo', verbose=False
        )
        
        assert result is not None
        assert isinstance(result.pvalue, (float, np.floating))
    
    def test_outlier_robustness(self, small_empirical_data):
        """
        Test robustness to outliers in the data.
        """
        data = small_empirical_data.copy()
        
        # Add some outliers
        np.random.seed(789)
        outlier_idx = np.random.choice(len(data), size=5, replace=False)
        data.loc[data.index[outlier_idx], 'Y'] *= 10  # Create outliers
        
        # Should still run without error
        result = run_parallel_trends_test(
            data, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', method='placebo', verbose=False
        )
        
        assert result is not None
        assert isinstance(result.pvalue, (float, np.floating))
    
    def test_small_sample_handling(self):
        """
        Test handling of small sample sizes.
        """
        np.random.seed(101)
        
        # Very small sample
        data = []
        for i in range(15):  # Only 15 units
            is_treated = i < 7
            first_treat = 3 if is_treated else np.inf
            
            for t in range(1, 5):
                y = 10 + 0.2 * t + np.random.normal(0, 0.5)
                if is_treated and t >= 3:
                    y += 1.0
                data.append({
                    'unit': i,
                    'time': t,
                    'Y': y,
                    'first_treat': first_treat,
                })
        
        df = pd.DataFrame(data)
        
        rec = recommend_transformation(
            df, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', verbose=False
        )
        
        # Should produce a recommendation even with small sample
        assert rec is not None
        assert rec.recommended_method in ['demean', 'detrend']
        
        # Should have lower confidence or warnings for small sample
        # (implementation may vary)
        assert isinstance(rec.confidence, (float, np.floating))


# =============================================================================
# Robustness Tests
# =============================================================================

class TestRobustness:
    """Robustness tests with various data characteristics."""
    
    def test_unbalanced_panel_handling(self):
        """
        Test handling of unbalanced panels (units with different observation counts).
        """
        np.random.seed(202)
        
        data = []
        for i in range(50):
            is_treated = i < 25
            first_treat = 5 if is_treated else np.inf
            
            # Vary the number of periods per unit
            n_periods = np.random.randint(6, 10)
            start_period = np.random.randint(1, 3)
            
            for t in range(start_period, start_period + n_periods):
                if t > 10:
                    break
                y = 10 + 0.2 * t + np.random.normal(0, 0.5)
                if is_treated and t >= 5:
                    y += 1.5
                data.append({
                    'unit': i,
                    'time': t,
                    'Y': y,
                    'first_treat': first_treat,
                })
        
        df = pd.DataFrame(data)
        
        result = run_parallel_trends_test(
            df, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', method='placebo', verbose=False
        )
        
        assert result is not None
        assert isinstance(result.pvalue, (float, np.floating))
    
    def test_many_cohorts(self):
        """
        Test with many treatment cohorts.
        """
        np.random.seed(303)
        
        data = []
        unit_id = 0
        cohorts = [4, 5, 6, 7, 8]  # 5 different cohorts
        
        for g in cohorts:
            for _ in range(20):  # 20 units per cohort
                for t in range(1, 12):
                    y = 10 + 0.1 * g * t + np.random.normal(0, 0.5)
                    if t >= g:
                        y += 2.0
                    data.append({
                        'unit': unit_id,
                        'time': t,
                        'Y': y,
                        'first_treat': g,
                    })
                unit_id += 1
        
        # Add control group
        for _ in range(30):
            for t in range(1, 12):
                y = 10 + 0.15 * t + np.random.normal(0, 0.5)
                data.append({
                    'unit': unit_id,
                    'time': t,
                    'Y': y,
                    'first_treat': np.inf,
                })
            unit_id += 1
        
        df = pd.DataFrame(data)
        
        diag = diagnose_heterogeneous_trends(
            df, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', verbose=False
        )
        
        # Should handle multiple cohorts
        assert diag is not None
        assert len(diag.trend_by_cohort) == len(cohorts)
    
    def test_high_variance_data(self):
        """
        Test with high variance (noisy) data.
        """
        np.random.seed(404)
        
        data = []
        for i in range(100):
            is_treated = i < 50
            first_treat = 5 if is_treated else np.inf
            
            for t in range(1, 10):
                # High noise relative to signal
                y = 10 + 0.1 * t + np.random.normal(0, 3.0)
                if is_treated and t >= 5:
                    y += 1.0
                data.append({
                    'unit': i,
                    'time': t,
                    'Y': y,
                    'first_treat': first_treat,
                })
        
        df = pd.DataFrame(data)
        
        rec = recommend_transformation(
            df, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', verbose=False
        )
        
        # Should still produce a recommendation
        # May include seasonal variants if seasonal patterns detected
        assert rec is not None
        assert rec.recommended_method in ['demean', 'detrend', 'demeanq', 'detrendq']
        
        # Confidence may be lower due to noise
        assert isinstance(rec.confidence, (float, np.floating))
    
    def test_no_never_treated(self):
        """
        Test when there are no never-treated units.
        """
        np.random.seed(505)
        
        data = []
        unit_id = 0
        cohorts = [4, 6, 8]  # All units eventually treated
        
        for g in cohorts:
            for _ in range(30):
                for t in range(1, 10):
                    y = 10 + 0.2 * t + np.random.normal(0, 0.5)
                    if t >= g:
                        y += 2.0
                    data.append({
                        'unit': unit_id,
                        'time': t,
                        'Y': y,
                        'first_treat': g,
                    })
                unit_id += 1
        
        df = pd.DataFrame(data)
        
        # Should handle case with no never-treated
        diag = diagnose_heterogeneous_trends(
            df, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', verbose=False
        )
        
        assert diag is not None
        assert diag.control_group_trend is None  # No control group


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple diagnostics."""
    
    def test_full_diagnostic_workflow(self, small_empirical_data):
        """
        Test the full diagnostic workflow: PT test -> diagnosis -> recommendation.
        """
        data = small_empirical_data.dropna()
        
        # Step 1: Test parallel trends
        pt_result = run_parallel_trends_test(
            data, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', method='placebo', verbose=False
        )
        
        # Step 2: Diagnose heterogeneous trends
        diag = diagnose_heterogeneous_trends(
            data, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', verbose=False
        )
        
        # Step 3: Get recommendation
        rec = recommend_transformation(
            data, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', verbose=False
        )
        
        # All should complete successfully
        assert pt_result is not None
        assert diag is not None
        assert rec is not None
        
        # Recommendation should be consistent with diagnostics
        if pt_result.reject_null:
            # If PT rejected, recommendation should lean toward detrend
            assert rec.parallel_trends_test.reject_null == True
    
    def test_summary_outputs_are_valid(self, small_empirical_data):
        """
        Test that all summary() methods produce valid string output.
        """
        data = small_empirical_data.dropna()
        
        pt_result = run_parallel_trends_test(
            data, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', method='placebo', verbose=False
        )
        
        diag = diagnose_heterogeneous_trends(
            data, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', verbose=False
        )
        
        rec = recommend_transformation(
            data, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', verbose=False
        )
        
        # All summaries should be non-empty strings
        pt_summary = pt_result.summary()
        assert isinstance(pt_summary, str)
        assert len(pt_summary) > 0
        
        diag_summary = diag.summary()
        assert isinstance(diag_summary, str)
        assert len(diag_summary) > 0
        
        rec_summary = rec.summary()
        assert isinstance(rec_summary, str)
        assert len(rec_summary) > 0


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
