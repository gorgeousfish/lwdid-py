"""
Stata-Python numerical validation tests for DESIGN-043-H diagnostics export.

Validates that:
1. Diagnostics exported via results_to_dataframe match direct object access
2. Diagnostics exported via diagnostics_to_dataframe are consistent
3. PS diagnostics values are reasonable against Stata reference data
4. Cross-validation between export methods

Uses Lee & Wooldridge (2023) staggered data for validation.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from lwdid.staggered import (
    transform_staggered_demean,
    estimate_cohort_time_effects,
    results_to_dataframe,
    diagnostics_to_dataframe,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def stata_data():
    """Load Stata Lee & Wooldridge staggered data."""
    data_path = Path("/Users/cxy/Desktop/大样本lwdid/Lee_Wooldridge_2023-main 3")
    dta_file = data_path / "2.lee_wooldridge_staggered_data.dta"
    
    if not dta_file.exists():
        pytest.skip(f"Stata data file not found: {dta_file}")
    
    df = pd.read_stata(dta_file)
    
    # Set panel structure
    df = df.sort_values(['id', 'year']).reset_index(drop=True)
    
    # Create gvar variable (from group)
    # group: 0=never treated, 4=cohort 4, 5=cohort 5, 6=cohort 6
    df['gvar'] = df['group'].fillna(0).astype(int)
    
    # Convert year from 2001-2006 to 1-6
    df['year'] = df['year'].astype(int) - 2000
    
    return df


# ============================================================================
# Cross-Validation Tests: results_to_dataframe vs direct access
# ============================================================================

class TestDiagnosticsExportConsistency:
    """Verify consistency between export methods and direct object access."""
    
    def test_results_to_dataframe_matches_direct_access(self, stata_data):
        """Exported diagnostics should match direct CohortTimeEffect.diagnostics access."""
        data = stata_data.copy()
        
        transformed = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar'
        )
        
        results = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year',
            estimator='ipw',
            propensity_controls=['x1', 'x2'],
            return_diagnostics=True,
        )
        
        df = results_to_dataframe(results, include_diagnostics=True)
        
        for i, r in enumerate(results):
            if r.diagnostics is not None:
                row = df[(df['cohort'] == r.cohort) & (df['period'] == r.period)].iloc[0]
                
                # Verify diagnostic values match
                assert row['diag_ps_mean'] == pytest.approx(r.diagnostics.ps_mean, rel=1e-10)
                assert row['diag_ps_std'] == pytest.approx(r.diagnostics.ps_std, rel=1e-10)
                assert row['diag_ps_min'] == pytest.approx(r.diagnostics.ps_min, rel=1e-10)
                assert row['diag_ps_max'] == pytest.approx(r.diagnostics.ps_max, rel=1e-10)
                assert row['diag_weights_cv'] == pytest.approx(r.diagnostics.weights_cv, rel=1e-10, nan_ok=True)
                assert row['diag_extreme_low_pct'] == pytest.approx(r.diagnostics.extreme_low_pct, rel=1e-10)
                assert row['diag_extreme_high_pct'] == pytest.approx(r.diagnostics.extreme_high_pct, rel=1e-10)
                assert row['diag_n_trimmed'] == r.diagnostics.n_trimmed
                assert row['diag_overlap_warning'] == r.diagnostics.overlap_warning
    
    def test_diagnostics_to_dataframe_matches_direct_access(self, stata_data):
        """diagnostics_to_dataframe should match direct access."""
        data = stata_data.copy()
        
        transformed = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar'
        )
        
        results = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year',
            estimator='ipwra',
            controls=['x1', 'x2'],
            propensity_controls=['x1', 'x2'],
            return_diagnostics=True,
        )
        
        diag_df = diagnostics_to_dataframe(results)
        
        # Count results with diagnostics
        results_with_diag = [r for r in results if r.diagnostics is not None]
        assert len(diag_df) == len(results_with_diag)
        
        for r in results_with_diag:
            row = diag_df[(diag_df['cohort'] == r.cohort) & (diag_df['period'] == r.period)].iloc[0]
            
            assert row['ps_mean'] == pytest.approx(r.diagnostics.ps_mean, rel=1e-10)
            assert row['ps_std'] == pytest.approx(r.diagnostics.ps_std, rel=1e-10)
            assert row['weights_cv'] == pytest.approx(r.diagnostics.weights_cv, rel=1e-10, nan_ok=True)
    
    def test_both_export_methods_consistent(self, stata_data):
        """results_to_dataframe and diagnostics_to_dataframe should be consistent."""
        data = stata_data.copy()
        
        transformed = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar'
        )
        
        results = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year',
            estimator='ipw',
            propensity_controls=['x1', 'x2'],
            return_diagnostics=True,
        )
        
        # Get diagnostics via both methods
        df_full = results_to_dataframe(results, include_diagnostics=True)
        df_diag = diagnostics_to_dataframe(results)
        
        # For each row in diagnostics_to_dataframe, verify consistency with results_to_dataframe
        for _, diag_row in df_diag.iterrows():
            full_row = df_full[
                (df_full['cohort'] == diag_row['cohort']) & 
                (df_full['period'] == diag_row['period'])
            ].iloc[0]
            
            assert diag_row['ps_mean'] == pytest.approx(full_row['diag_ps_mean'], rel=1e-10)
            assert diag_row['ps_std'] == pytest.approx(full_row['diag_ps_std'], rel=1e-10)
            assert diag_row['ps_min'] == pytest.approx(full_row['diag_ps_min'], rel=1e-10)
            assert diag_row['ps_max'] == pytest.approx(full_row['diag_ps_max'], rel=1e-10)


# ============================================================================
# Diagnostic Values Validation
# ============================================================================

class TestDiagnosticsValuesReasonable:
    """Verify diagnostic values are within expected ranges."""
    
    def test_ps_values_in_valid_range(self, stata_data):
        """Propensity scores should be in (0, 1) after trimming."""
        data = stata_data.copy()
        
        transformed = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar'
        )
        
        results = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year',
            estimator='ipw',
            propensity_controls=['x1', 'x2'],
            return_diagnostics=True,
            trim_threshold=0.01,
        )
        
        df = results_to_dataframe(results, include_diagnostics=True)
        
        # Filter rows with diagnostics
        df_diag = df[df['diag_ps_mean'].notna()]
        
        # PS values should be in valid range
        assert (df_diag['diag_ps_min'] >= 0.01).all(), "ps_min should be >= trim_threshold"
        assert (df_diag['diag_ps_max'] <= 0.99).all(), "ps_max should be <= 1-trim_threshold"
        assert (df_diag['diag_ps_mean'] > 0).all(), "ps_mean should be > 0"
        assert (df_diag['diag_ps_mean'] < 1).all(), "ps_mean should be < 1"
        assert (df_diag['diag_ps_std'] >= 0).all(), "ps_std should be >= 0"
    
    def test_extreme_pct_values_valid(self, stata_data):
        """Extreme percentage values should be in [0, 1]."""
        data = stata_data.copy()
        
        transformed = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar'
        )
        
        results = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year',
            estimator='ipw',
            propensity_controls=['x1', 'x2'],
            return_diagnostics=True,
        )
        
        diag_df = diagnostics_to_dataframe(results)
        
        assert (diag_df['extreme_low_pct'] >= 0).all()
        assert (diag_df['extreme_low_pct'] <= 1).all()
        assert (diag_df['extreme_high_pct'] >= 0).all()
        assert (diag_df['extreme_high_pct'] <= 1).all()
    
    def test_n_trimmed_non_negative(self, stata_data):
        """n_trimmed should be non-negative integer."""
        data = stata_data.copy()
        
        transformed = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar'
        )
        
        results = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year',
            estimator='ipw',
            propensity_controls=['x1', 'x2'],
            return_diagnostics=True,
        )
        
        diag_df = diagnostics_to_dataframe(results)
        
        assert (diag_df['n_trimmed'] >= 0).all()
        assert all(isinstance(n, (int, np.integer)) or n == int(n) for n in diag_df['n_trimmed'])
    
    def test_quantiles_monotonic(self, stata_data):
        """Quantiles should be monotonically increasing: q25 <= q50 <= q75."""
        data = stata_data.copy()
        
        transformed = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar'
        )
        
        results = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year',
            estimator='ipwra',
            controls=['x1', 'x2'],
            propensity_controls=['x1', 'x2'],
            return_diagnostics=True,
        )
        
        df = results_to_dataframe(results, include_diagnostics=True)
        df_diag = df[df['diag_ps_q25'].notna()]
        
        for _, row in df_diag.iterrows():
            assert row['diag_ps_q25'] <= row['diag_ps_q50'], "q25 should <= q50"
            assert row['diag_ps_q50'] <= row['diag_ps_q75'], "q50 should <= q75"


# ============================================================================
# Multiple Estimator Comparison
# ============================================================================

class TestMultipleEstimatorDiagnostics:
    """Compare diagnostics across different estimators."""
    
    def test_ipw_ipwra_psm_diagnostics_available(self, stata_data):
        """IPW, IPWRA, and PSM should all provide diagnostics."""
        data = stata_data.copy()
        
        transformed = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar'
        )
        
        for estimator in ['ipw', 'ipwra', 'psm']:
            if estimator == 'psm':
                results = estimate_cohort_time_effects(
                    transformed, 'gvar', 'id', 'year',
                    estimator=estimator,
                    propensity_controls=['x1', 'x2'],
                    return_diagnostics=True,
                    n_neighbors=1,
                )
            else:
                results = estimate_cohort_time_effects(
                    transformed, 'gvar', 'id', 'year',
                    estimator=estimator,
                    propensity_controls=['x1', 'x2'],
                    return_diagnostics=True,
                )
            
            diag_df = diagnostics_to_dataframe(results)
            assert len(diag_df) > 0, f"Estimator {estimator} should have diagnostics"
            
            # Verify basic diagnostic structure
            assert 'ps_mean' in diag_df.columns
            assert 'weights_cv' in diag_df.columns
    
    def test_ra_estimator_no_diagnostics(self, stata_data):
        """RA estimator should not have propensity score diagnostics."""
        data = stata_data.copy()
        
        transformed = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar'
        )
        
        results = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year',
            estimator='ra',
        )
        
        diag_df = diagnostics_to_dataframe(results)
        assert len(diag_df) == 0, "RA estimator should not have PS diagnostics"
        
        df = results_to_dataframe(results, include_diagnostics=True)
        assert df['diag_ps_mean'].isna().all()


# ============================================================================
# DataFrame Structure Validation
# ============================================================================

class TestDataFrameStructure:
    """Validate DataFrame structure and data types."""
    
    def test_results_to_dataframe_dtypes(self, stata_data):
        """Verify data types of exported DataFrame."""
        data = stata_data.copy()
        
        transformed = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar'
        )
        
        results = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year',
            estimator='ipw',
            propensity_controls=['x1', 'x2'],
            return_diagnostics=True,
        )
        
        df = results_to_dataframe(results, include_diagnostics=True)
        
        # Check numeric columns are float
        numeric_cols = [
            'att', 'se', 'ci_lower', 'ci_upper', 't_stat', 'pvalue',
            'diag_ps_mean', 'diag_ps_std', 'diag_ps_min', 'diag_ps_max',
            'diag_ps_q25', 'diag_ps_q50', 'diag_ps_q75', 'diag_weights_cv',
            'diag_extreme_low_pct', 'diag_extreme_high_pct'
        ]
        for col in numeric_cols:
            assert pd.api.types.is_float_dtype(df[col]) or df[col].isna().all(), f"{col} should be float"
        
        # Check integer columns
        int_cols = ['cohort', 'period', 'event_time', 'n_treated', 'n_control', 'n_total']
        for col in int_cols:
            # Allow either int or float (for compatibility)
            assert pd.api.types.is_numeric_dtype(df[col]), f"{col} should be numeric"
    
    def test_diagnostics_to_dataframe_dtypes(self, stata_data):
        """Verify data types of diagnostics DataFrame."""
        data = stata_data.copy()
        
        transformed = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar'
        )
        
        results = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year',
            estimator='ipw',
            propensity_controls=['x1', 'x2'],
            return_diagnostics=True,
        )
        
        df = diagnostics_to_dataframe(results)
        
        # Check numeric columns
        numeric_cols = [
            'ps_mean', 'ps_std', 'ps_min', 'ps_max',
            'ps_q25', 'ps_q50', 'ps_q75', 'weights_cv',
            'extreme_low_pct', 'extreme_high_pct'
        ]
        for col in numeric_cols:
            assert pd.api.types.is_float_dtype(df[col]) or df[col].isna().all(), f"{col} should be float"
        
        # n_trimmed should be integer-like
        assert pd.api.types.is_numeric_dtype(df['n_trimmed'])


# ============================================================================
# End-to-End Workflow Test
# ============================================================================

class TestEndToEndWorkflow:
    """Test complete workflow with diagnostics export."""
    
    def test_complete_analysis_workflow(self, stata_data):
        """Test complete analysis workflow with diagnostics."""
        data = stata_data.copy()
        
        # Step 1: Transform data
        transformed = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar'
        )
        
        # Step 2: Estimate with diagnostics
        results = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year',
            estimator='ipwra',
            controls=['x1', 'x2'],
            propensity_controls=['x1', 'x2'],
            return_diagnostics=True,
        )
        
        # Step 3: Export results with diagnostics
        df_results = results_to_dataframe(results, include_diagnostics=True)
        
        # Step 4: Export diagnostics separately for analysis
        df_diag = diagnostics_to_dataframe(results)
        
        # Verify workflow produces expected outputs
        assert len(df_results) > 0
        assert len(df_diag) > 0
        
        # Verify can analyze diagnostics
        # Example: Find pairs with high weight variation (CV > 2.0)
        high_cv = df_diag[df_diag['weights_cv'] > 2.0]
        # This is informational, not an assertion
        
        # Example: Check mean PS across cohorts
        ps_by_cohort = df_diag.groupby('cohort')['ps_mean'].mean()
        assert len(ps_by_cohort) > 0
        
        # Verify all expected columns present
        expected_diag_cols = [
            'cohort', 'period', 'event_time',
            'ps_mean', 'ps_std', 'ps_min', 'ps_max',
            'ps_q25', 'ps_q50', 'ps_q75', 'weights_cv',
            'extreme_low_pct', 'extreme_high_pct', 'n_trimmed',
            'overlap_warning'
        ]
        for col in expected_diag_cols:
            assert col in df_diag.columns, f"Missing column: {col}"
