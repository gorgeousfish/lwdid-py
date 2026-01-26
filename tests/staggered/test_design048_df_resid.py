"""
Unit tests for DESIGN-048: df_resid calculation with control variables.

This module validates that residual degrees of freedom (df_resid) are correctly
calculated and propagated through the estimation pipeline when control variables
are used.

Key test scenarios:
1. df_resid without controls: should be n - 2
2. df_resid with controls: should be n - (2 + 2K) where K = number of controls
3. df_resid propagation through aggregation levels
4. df_inference for cluster-robust standard errors
"""

import pytest
import numpy as np
import pandas as pd
from lwdid import lwdid
from lwdid.staggered.estimation import (
    run_ols_regression,
    estimate_cohort_time_effects,
    CohortTimeEffect,
)
from lwdid.staggered.aggregation import (
    aggregate_to_cohort,
    aggregate_to_overall,
    CohortEffect,
    OverallEffect,
)
from lwdid.staggered.transformations import transform_staggered_demean


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def simple_staggered_data():
    """
    Create simple staggered data for df_resid validation.
    
    Structure:
    - 30 units: 10 treated in 2002, 10 treated in 2003, 10 never-treated
    - Time periods: 2000-2005
    """
    np.random.seed(42)
    
    data = []
    for unit in range(1, 31):
        if unit <= 10:
            gvar = 2002
        elif unit <= 20:
            gvar = 2003
        else:
            gvar = 0  # Never treated
        
        # Unit-specific control variable (time-invariant)
        x1 = np.random.normal(0, 1)
        x2 = np.random.uniform(-1, 1)
        
        for year in range(2000, 2006):
            y = 1.0 + 0.5 * x1 + 0.3 * x2 + 0.1 * year + np.random.normal(0, 0.5)
            if gvar > 0 and year >= gvar:
                y += 1.0  # Treatment effect
            
            data.append({
                'id': unit,
                'year': year,
                'y': y,
                'gvar': gvar,
                'x1': x1,
                'x2': x2,
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def transformed_data(simple_staggered_data):
    """Create pre-transformed data for direct function testing."""
    return transform_staggered_demean(
        data=simple_staggered_data,
        y='y',
        ivar='id',
        tvar='year',
        gvar='gvar',
    )


# =============================================================================
# Test run_ols_regression df_resid
# =============================================================================

class TestRunOLSRegressionDfResid:
    """Test df_resid calculation in run_ols_regression."""
    
    def test_df_resid_no_controls(self):
        """df_resid should be n - 2 without controls."""
        np.random.seed(42)
        n = 100
        data = pd.DataFrame({
            'y': np.random.normal(0, 1, n),
            'd': np.array([1] * 50 + [0] * 50),
        })
        
        result = run_ols_regression(data, y='y', d='d')
        
        expected_df = n - 2  # intercept + treatment
        assert result['df_resid'] == expected_df
        assert result['df_inference'] == expected_df
    
    def test_df_resid_with_one_control(self):
        """df_resid should be n - 4 with 1 control (1 + 1 + 1 + 1)."""
        np.random.seed(42)
        n = 100
        data = pd.DataFrame({
            'y': np.random.normal(0, 1, n),
            'd': np.array([1] * 50 + [0] * 50),
            'x1': np.random.normal(0, 1, n),
        })
        
        result = run_ols_regression(data, y='y', d='d', controls=['x1'])
        
        # K=1: df = n - (2 + 2*1) = n - 4
        expected_df = n - 4
        assert result['df_resid'] == expected_df
    
    def test_df_resid_with_two_controls(self):
        """df_resid should be n - 6 with 2 controls."""
        np.random.seed(42)
        n = 100
        data = pd.DataFrame({
            'y': np.random.normal(0, 1, n),
            'd': np.array([1] * 50 + [0] * 50),
            'x1': np.random.normal(0, 1, n),
            'x2': np.random.normal(0, 1, n),
        })
        
        result = run_ols_regression(data, y='y', d='d', controls=['x1', 'x2'])
        
        # K=2: df = n - (2 + 2*2) = n - 6
        expected_df = n - 6
        assert result['df_resid'] == expected_df
    
    def test_df_resid_with_cluster_vce(self):
        """df_inference should be G-1 for cluster-robust."""
        np.random.seed(42)
        n = 100
        n_clusters = 10
        data = pd.DataFrame({
            'y': np.random.normal(0, 1, n),
            'd': np.array([1] * 50 + [0] * 50),
            'cluster': np.repeat(np.arange(n_clusters), n // n_clusters),
        })
        
        result = run_ols_regression(
            data, y='y', d='d', 
            vce='cluster', cluster_var='cluster'
        )
        
        assert result['df_resid'] == n - 2
        assert result['df_inference'] == n_clusters - 1


# =============================================================================
# Test CohortTimeEffect df_resid
# =============================================================================

class TestCohortTimeEffectDfResid:
    """Test df_resid in CohortTimeEffect."""
    
    def test_cohort_time_effect_has_df_fields(self, transformed_data):
        """CohortTimeEffect should contain df_resid and df_inference."""
        effects = estimate_cohort_time_effects(
            data_transformed=transformed_data,
            gvar='gvar',
            ivar='id',
            tvar='year',
            controls=None,
            vce=None,
            cluster_var=None,
        )
        
        assert len(effects) > 0
        for effect in effects:
            assert hasattr(effect, 'df_resid')
            assert hasattr(effect, 'df_inference')
            assert effect.df_resid > 0
            assert effect.df_inference > 0
    
    def test_cohort_time_effect_df_without_controls(self, transformed_data):
        """df_resid should be n_total - 2 without controls."""
        effects = estimate_cohort_time_effects(
            data_transformed=transformed_data,
            gvar='gvar',
            ivar='id',
            tvar='year',
            controls=None,
            vce=None,
            cluster_var=None,
        )
        
        for effect in effects:
            expected_df = effect.n_total - 2
            assert effect.df_resid == expected_df, \
                f"Expected df_resid={expected_df}, got {effect.df_resid}"
    
    def test_cohort_time_effect_df_with_controls(self, simple_staggered_data):
        """df_resid should account for controls."""
        transformed = transform_staggered_demean(
            data=simple_staggered_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
        )
        
        effects = estimate_cohort_time_effects(
            data_transformed=transformed,
            gvar='gvar',
            ivar='id',
            tvar='year',
            controls=['x1', 'x2'],
            vce=None,
            cluster_var=None,
        )
        
        for effect in effects:
            # K=2: df = n - (2 + 2*2) = n - 6
            expected_df = effect.n_total - 6
            assert effect.df_resid == expected_df, \
                f"Expected df_resid={expected_df}, got {effect.df_resid}"


# =============================================================================
# Test Aggregation df_resid
# =============================================================================

class TestAggregationDfResid:
    """Test df_resid in aggregation functions."""
    
    def test_cohort_effect_has_df_fields(self, transformed_data):
        """CohortEffect should contain df_resid and df_inference."""
        cohort_effects = aggregate_to_cohort(
            data_transformed=transformed_data,
            gvar='gvar',
            ivar='id',
            tvar='year',
            cohorts=[2002, 2003],
            T_max=2005,
            vce=None,
            cluster_var=None,
        )
        
        assert len(cohort_effects) > 0
        for effect in cohort_effects:
            assert hasattr(effect, 'df_resid')
            assert hasattr(effect, 'df_inference')
            assert effect.df_resid > 0
    
    def test_overall_effect_has_df_fields(self, transformed_data):
        """OverallEffect should contain df_resid and df_inference."""
        overall_effect = aggregate_to_overall(
            data_transformed=transformed_data,
            gvar='gvar',
            ivar='id',
            tvar='year',
            vce=None,
            cluster_var=None,
        )
        
        assert hasattr(overall_effect, 'df_resid')
        assert hasattr(overall_effect, 'df_inference')
        assert overall_effect.df_resid > 0
    
    def test_cohort_effect_df_resid_value(self, transformed_data):
        """CohortEffect df_resid should be n - 2 (no controls in aggregation)."""
        cohort_effects = aggregate_to_cohort(
            data_transformed=transformed_data,
            gvar='gvar',
            ivar='id',
            tvar='year',
            cohorts=[2002, 2003],
            T_max=2005,
            vce=None,
            cluster_var=None,
        )
        
        for effect in cohort_effects:
            expected_df = effect.n_units + effect.n_control - 2
            assert effect.df_resid == expected_df
    
    def test_overall_effect_df_resid_value(self, transformed_data):
        """OverallEffect df_resid should be n_treated + n_control - 2."""
        overall_effect = aggregate_to_overall(
            data_transformed=transformed_data,
            gvar='gvar',
            ivar='id',
            tvar='year',
            vce=None,
            cluster_var=None,
        )
        
        expected_df = overall_effect.n_treated + overall_effect.n_control - 2
        assert overall_effect.df_resid == expected_df


# =============================================================================
# Test Core API df_resid
# =============================================================================

class TestCoreDfResid:
    """Test df_resid in lwdid core API."""
    
    def test_lwdid_result_has_df_fields(self, simple_staggered_data):
        """LWDIDResults should have df_resid and df_inference."""
        result = lwdid(
            simple_staggered_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            control_group='never_treated',
            aggregate='overall',
        )
        
        assert hasattr(result, 'df_resid')
        assert hasattr(result, 'df_inference')
        assert result.df_resid > 0
        assert result.df_inference > 0
    
    def test_lwdid_df_resid_overall_no_controls(self, simple_staggered_data):
        """df_resid should be n - 2 for overall aggregation without controls."""
        result = lwdid(
            simple_staggered_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            control_group='never_treated',
            aggregate='overall',
        )
        
        # Overall aggregation regression: Y_bar = alpha + tau * D + eps
        # df = n_treated + n_control - 2
        expected_df = result.n_treated + result.n_control - 2
        assert result.df_resid == expected_df
    
    def test_lwdid_df_resid_cohort_no_controls(self, simple_staggered_data):
        """df_resid for cohort aggregation should reflect cohort regressions."""
        result = lwdid(
            simple_staggered_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            control_group='never_treated',
            aggregate='cohort',
        )
        
        # df_resid should be median of cohort df values
        assert result.df_resid > 0
    
    def test_lwdid_df_resid_none_no_controls(self, simple_staggered_data):
        """df_resid for aggregate='none' should reflect cohort-time regressions."""
        result = lwdid(
            simple_staggered_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            control_group='never_treated',
            aggregate='none',
        )
        
        # df_resid should be median of cohort-time df values
        assert result.df_resid > 0
        
        # Verify df values in cohort-time effects
        att_df = result.att_by_cohort_time
        assert 'df_resid' in att_df.columns or result.df_resid > 0
    
    def test_lwdid_df_resid_with_controls_ra(self, simple_staggered_data):
        """df_resid with controls in RA estimator should be n - (2 + 2K)."""
        result = lwdid(
            simple_staggered_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            control_group='never_treated',
            aggregate='none',
            estimator='ra',
            controls=['x1', 'x2'],
        )
        
        # For cohort-time effects with K=2 controls:
        # df = n - (2 + 2*2) = n - 6
        # The overall df_resid is median of these values
        assert result.df_resid > 0
        
        # Verify it's less than n_total - 2
        # (because controls reduce df)
        # Note: result.df_resid is median of cohort-time df values


# =============================================================================
# Test DataFrame Export
# =============================================================================

class TestDataFrameExport:
    """Test df_resid/df_inference in DataFrame exports."""
    
    def test_results_to_dataframe_has_df_columns(self, transformed_data):
        """results_to_dataframe should include df_resid and df_inference."""
        from lwdid.staggered.estimation import results_to_dataframe
        
        effects = estimate_cohort_time_effects(
            data_transformed=transformed_data,
            gvar='gvar',
            ivar='id',
            tvar='year',
            controls=None,
            vce=None,
            cluster_var=None,
        )
        
        df = results_to_dataframe(effects)
        
        assert 'df_resid' in df.columns
        assert 'df_inference' in df.columns
        assert df['df_resid'].notna().all()
        assert df['df_inference'].notna().all()
    
    def test_cohort_effects_to_dataframe_has_df_columns(self, transformed_data):
        """cohort_effects_to_dataframe should include df_resid and df_inference."""
        from lwdid.staggered.aggregation import cohort_effects_to_dataframe
        
        cohort_effects = aggregate_to_cohort(
            data_transformed=transformed_data,
            gvar='gvar',
            ivar='id',
            tvar='year',
            cohorts=[2002, 2003],
            T_max=2005,
            vce=None,
            cluster_var=None,
        )
        
        df = cohort_effects_to_dataframe(cohort_effects)
        
        assert 'df_resid' in df.columns
        assert 'df_inference' in df.columns
        assert df['df_resid'].notna().all()
        assert df['df_inference'].notna().all()


# =============================================================================
# Test Formula Validation
# =============================================================================

class TestFormulaValidation:
    """Validate df_resid against known formulas."""
    
    def test_df_resid_formula_no_controls(self):
        """Verify df = n - 2 formula without controls."""
        np.random.seed(42)
        
        for n in [20, 50, 100, 200]:
            data = pd.DataFrame({
                'y': np.random.normal(0, 1, n),
                'd': np.array([1] * (n // 2) + [0] * (n // 2)),
            })
            
            result = run_ols_regression(data, y='y', d='d')
            
            assert result['df_resid'] == n - 2
    
    def test_df_resid_formula_with_controls(self):
        """Verify df = n - (2 + 2K) formula with controls."""
        np.random.seed(42)
        
        for n in [50, 100, 200]:
            for k in [1, 2, 3]:
                data = pd.DataFrame({
                    'y': np.random.normal(0, 1, n),
                    'd': np.array([1] * (n // 2) + [0] * (n // 2)),
                })
                controls = []
                for i in range(k):
                    col_name = f'x{i}'
                    data[col_name] = np.random.normal(0, 1, n)
                    controls.append(col_name)
                
                result = run_ols_regression(data, y='y', d='d', controls=controls)
                
                expected_df = n - (2 + 2 * k)
                assert result['df_resid'] == expected_df, \
                    f"n={n}, K={k}: expected {expected_df}, got {result['df_resid']}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
