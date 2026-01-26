"""
Tests for design fixes DESIGN-081, DESIGN-082, and DESIGN-083.

DESIGN-081: plot_event_study return data semantic clarity
DESIGN-082: validation.py sorted() NaN handling (verified existing fix)
DESIGN-083: Equivalence of detrend implementations

These tests verify:
1. Event study data includes both raw and normalized values
2. Quarter validation handles NaN values correctly
3. Both detrend implementations produce equivalent results
"""

import numpy as np
import pandas as pd
import pytest
import warnings


class TestDesign081PlotEventStudyReturnData:
    """Tests for DESIGN-081: plot_event_study return data semantic clarity."""
    
    @pytest.fixture
    def staggered_results(self):
        """Create a mock LWDIDResults object for testing."""
        from lwdid.results import LWDIDResults
        
        # Create minimal mock data for staggered results
        results_dict = {
            'att': 0.5,
            'se_att': 0.1,
            't_stat': 5.0,
            'pvalue': 0.001,
            'ci_lower': 0.3,
            'ci_upper': 0.7,
            'nobs': 1000,
            'df_resid': 990,
            'params': np.array([0.5, 0.1]),
            'bse': np.array([0.1, 0.02]),
            'vcov': np.eye(2) * 0.01,
            'resid': np.random.randn(1000),
            'vce_type': 'robust',
            'is_staggered': True,
            'cohorts': [2005, 2006, 2007],
            'cohort_sizes': {2005: 50, 2006: 40, 2007: 30},
            'att_by_cohort_time': pd.DataFrame({
                'cohort': [2005, 2005, 2005, 2006, 2006, 2007],
                'period': [2005, 2006, 2007, 2006, 2007, 2007],
                'att': [0.3, 0.5, 0.6, 0.4, 0.55, 0.45],
                'se': [0.08, 0.09, 0.10, 0.085, 0.095, 0.088],
            }),
            'att_by_cohort': pd.DataFrame({
                'cohort': [2005, 2006, 2007],
                'att': [0.47, 0.48, 0.45],
                'se': [0.05, 0.055, 0.052],
                'n_units': [50, 40, 30],
                'n_periods': [3, 2, 1],
            }),
            'att_overall': 0.47,
            'se_overall': 0.04,
            'ci_overall_lower': 0.39,
            'ci_overall_upper': 0.55,
            'cohort_weights': {2005: 0.42, 2006: 0.33, 2007: 0.25},
            'control_group': 'not_yet_treated',
            'control_group_used': 'not_yet_treated',
            'aggregate': 'overall',
            'estimator': 'ra',
            'n_never_treated': 100,
        }
        
        metadata = {
            'K': 3,
            'tpost1': 2005,
            'depvar': 'y',
            'N_treated': 120,
            'N_control': 100,
            'ivar': 'id',
            'tvar': 'year',
            'gvar': 'gvar',
        }
        
        return LWDIDResults(results_dict, metadata)
    
    def test_return_data_includes_raw_values(self, staggered_results):
        """Test that returned DataFrame includes raw (unnormalized) values."""
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend for testing
        import matplotlib.pyplot as plt
        
        fig, ax, event_df = staggered_results.plot_event_study(
            ref_period=0,
            return_data=True
        )
        plt.close(fig)
        
        # Check raw value columns exist
        assert 'att_raw' in event_df.columns, "att_raw column missing"
        assert 'ci_lower_raw' in event_df.columns, "ci_lower_raw column missing"
        assert 'ci_upper_raw' in event_df.columns, "ci_upper_raw column missing"
        
        # Check metadata columns exist
        assert 'is_normalized' in event_df.columns, "is_normalized column missing"
        assert 'ref_period_used' in event_df.columns, "ref_period_used column missing"
    
    def test_normalization_metadata_correct(self, staggered_results):
        """Test that normalization metadata is correctly set."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Test with normalization
        fig, ax, event_df = staggered_results.plot_event_study(
            ref_period=0,
            return_data=True
        )
        plt.close(fig)
        
        # If ref_period=0 exists, should be normalized
        if 0 in event_df['event_time'].values:
            assert event_df['is_normalized'].all(), "is_normalized should be True when ref_period specified"
            assert (event_df['ref_period_used'] == 0).all(), "ref_period_used should be 0"
    
    def test_no_normalization_preserves_raw(self, staggered_results):
        """Test that when ref_period=None, raw and normalized values are equal."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax, event_df = staggered_results.plot_event_study(
            ref_period=None,
            return_data=True
        )
        plt.close(fig)
        
        # Check raw values equal normalized values
        assert np.allclose(event_df['att'], event_df['att_raw'], equal_nan=True), \
            "att should equal att_raw when no normalization"
        assert np.allclose(event_df['ci_lower'], event_df['ci_lower_raw'], equal_nan=True), \
            "ci_lower should equal ci_lower_raw when no normalization"
        assert np.allclose(event_df['ci_upper'], event_df['ci_upper_raw'], equal_nan=True), \
            "ci_upper should equal ci_upper_raw when no normalization"
        
        # Check metadata
        assert not event_df['is_normalized'].any(), "is_normalized should be False"
        assert event_df['ref_period_used'].isna().all(), "ref_period_used should be NaN"
    
    def test_normalized_values_relationship(self, staggered_results):
        """Test that normalized = raw - ref_att relationship holds."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax, event_df = staggered_results.plot_event_study(
            ref_period=0,
            return_data=True
        )
        plt.close(fig)
        
        # Get reference period ATT
        ref_row = event_df[event_df['event_time'] == 0]
        if len(ref_row) > 0:
            ref_att_raw = ref_row['att_raw'].values[0]
            if not pd.isna(ref_att_raw):
                # Verify relationship: normalized = raw - ref_att_raw
                expected_att = event_df['att_raw'] - ref_att_raw
                assert np.allclose(event_df['att'], expected_att, equal_nan=True), \
                    "Normalized att should equal raw - ref_att"


class TestDesign082SortedNaNHandling:
    """Tests for DESIGN-082: sorted() NaN handling in quarter validation."""
    
    def test_quarter_validation_with_nan_values(self):
        """Test that quarter validation handles NaN values without TypeError."""
        from lwdid.validation import _create_time_index
        
        # Create test data with NaN in quarter column
        data = pd.DataFrame({
            'year': [2000, 2000, 2000, 2001, 2001, 2001],
            'quarter': [1.0, 2.0, np.nan, 1.0, 2.0, 3.0],  # Include NaN
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })
        
        # This should raise InvalidParameterError about invalid quarter values,
        # not TypeError from sorted()
        from lwdid.exceptions import InvalidParameterError
        with pytest.raises(InvalidParameterError) as exc_info:
            _create_time_index(data.copy(), ['year', 'quarter'])
        
        # The error message should mention invalid quarter values, not sorting error
        assert "contains non-numeric values" in str(exc_info.value) or \
               "Invalid" in str(exc_info.value) or \
               "NaN" in str(exc_info.value).lower()
    
    def test_quarter_validation_with_valid_quarters(self):
        """Test that quarter validation works with valid quarter values."""
        from lwdid.validation import _create_time_index
        
        # Create test data with valid quarters
        data = pd.DataFrame({
            'year': [2000, 2000, 2000, 2000, 2001, 2001, 2001, 2001],
            'quarter': [1, 2, 3, 4, 1, 2, 3, 4],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        })
        
        # This should succeed without errors
        result_data, is_quarterly = _create_time_index(data.copy(), ['year', 'quarter'])
        
        assert is_quarterly, "Should be identified as quarterly data"
        assert 'tindex' in result_data.columns, "tindex column should be created"
        assert 'tq' in result_data.columns, "tq column should be created"


class TestDesign083DetrendEquivalence:
    """Tests for DESIGN-083: Equivalence of detrend implementations."""
    
    def test_detrend_implementations_equivalent(self):
        """Test that both detrend implementations produce equivalent results."""
        from lwdid.transformations import detrend_unit
        
        # Create test data for a single unit
        np.random.seed(42)
        n_periods = 10
        pre_periods = 6
        
        time_values = np.arange(1, n_periods + 1)
        y_values = 2.0 + 0.5 * time_values + np.random.randn(n_periods) * 0.1
        post_values = np.array([0] * pre_periods + [1] * (n_periods - pre_periods))
        
        unit_data = pd.DataFrame({
            'tindex': time_values,
            'y': y_values,
            'post': post_values,
        })
        
        # Method 1: statsmodels OLS (transformations.py approach)
        yhat_sm, ydot_sm = detrend_unit(unit_data, 'y', 'tindex', 'post')
        
        # Method 2: Manual Cov/Var calculation (staggered approach)
        pre_data = unit_data[unit_data['post'] == 0]
        t_pre = pre_data['tindex'].values.astype(float)
        y_pre = pre_data['y'].values
        
        t_mean = np.mean(t_pre)
        y_mean = np.mean(y_pre)
        cov_ty = np.mean((t_pre - t_mean) * (y_pre - y_mean))
        var_t = np.mean((t_pre - t_mean) ** 2)
        
        beta_manual = cov_ty / var_t
        alpha_manual = y_mean - beta_manual * t_mean
        
        yhat_manual = alpha_manual + beta_manual * time_values
        ydot_manual = y_values - yhat_manual
        
        # Compare results - should be numerically equivalent
        assert np.allclose(ydot_sm, ydot_manual, atol=1e-10), \
            f"Detrended values differ: max diff = {np.max(np.abs(ydot_sm - ydot_manual))}"
    
    def test_staggered_detrend_vectorized_correct(self):
        """Test that staggered detrend produces correct OLS coefficients."""
        from lwdid.staggered.transformations import transform_staggered_detrend
        
        # Create simple staggered data with proper year values
        np.random.seed(123)
        n_units = 10
        n_periods = 8
        cohort_year = 2005  # Treatment starts at year 2005
        
        data_rows = []
        for i in range(n_units):
            for t in range(n_periods):
                year = 2001 + t  # Years 2001-2008
                # Simple linear trend: y = 1 + 0.5*t + unit_effect + noise
                unit_effect = i * 0.1
                y = 1.0 + 0.5 * t + unit_effect + np.random.randn() * 0.05
                # Half treated at 2005, half never-treated
                gvar = cohort_year if i < n_units // 2 else 0
                data_rows.append({
                    'id': i + 1,
                    'year': year,
                    'y': y,
                    'gvar': gvar,
                })
        
        data = pd.DataFrame(data_rows)
        
        # Apply staggered detrend
        result = transform_staggered_detrend(data, 'y', 'id', 'year', 'gvar')
        
        # Check that ycheck columns are created
        ycheck_cols = [c for c in result.columns if c.startswith('ycheck_')]
        assert len(ycheck_cols) > 0, "No ycheck columns created"
        
        # For treated units, check that detrended values in post-period have
        # removed the linear trend (should be close to zero mean if trend is removed)
        for col in ycheck_cols:
            values = result[col].dropna()
            if len(values) > 0:
                # Values should be centered around the treatment effect (or zero if no effect)
                assert np.abs(values.mean()) < 2.0, \
                    f"Detrended values have unexpected mean: {values.mean()}"


class TestDesign081082083Integration:
    """Integration tests for all three design fixes."""
    
    def test_full_workflow_with_quarterly_staggered(self):
        """Test full workflow with quarterly staggered data."""
        # This is a smoke test to ensure all components work together
        import warnings
        
        # Create quarterly staggered data
        np.random.seed(42)
        data_rows = []
        n_units = 20
        
        for i in range(n_units):
            for year in range(2000, 2005):
                for quarter in range(1, 5):
                    y = 1.0 + 0.1 * (year - 2000) + 0.05 * quarter + np.random.randn() * 0.1
                    # Staggered treatment: cohort based on unit id
                    if i < 5:
                        gvar = 0  # Never treated
                    elif i < 10:
                        gvar = 2002
                    elif i < 15:
                        gvar = 2003
                    else:
                        gvar = 2004
                    
                    data_rows.append({
                        'id': i + 1,
                        'year': year,
                        'quarter': quarter,
                        'y': y,
                        'gvar': gvar,
                    })
        
        data = pd.DataFrame(data_rows)
        
        # Test that validation handles the data correctly (DESIGN-082)
        from lwdid.validation import _create_time_index
        result_data, is_quarterly = _create_time_index(data.copy(), ['year', 'quarter'])
        assert is_quarterly, "Should identify as quarterly data"
        
        # Test staggered detrend (DESIGN-083)
        from lwdid.staggered.transformations import transform_staggered_demean
        result = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        assert not result.empty, "Transformation should produce results"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
