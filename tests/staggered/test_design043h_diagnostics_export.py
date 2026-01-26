"""
Tests for DESIGN-043-H: diagnostics export in results_to_dataframe.

Verifies that:
1. results_to_dataframe(include_diagnostics=True) exports PS diagnostics columns
2. diagnostics_to_dataframe() extracts diagnostics to dedicated DataFrame
3. Proper handling when diagnostics are None (RA estimator)
4. Proper handling of empty results
5. Column naming consistency and completeness
"""

import numpy as np
import pandas as pd
import pytest

from lwdid.staggered import (
    transform_staggered_demean,
    estimate_cohort_time_effects,
    results_to_dataframe,
    diagnostics_to_dataframe,
    CohortTimeEffect,
    PropensityScoreDiagnostics,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def simple_staggered_data():
    """Simple staggered data for testing diagnostics export."""
    np.random.seed(42)
    n_units = 60
    T = 5
    
    # Unit-level characteristics
    # Cohort distribution: 50% NT, 25% g=3, 25% g=4
    cohort_probs = [0.50, 0.25, 0.25]
    cohort_values = [0, 3, 4]
    unit_cohorts = np.random.choice(cohort_values, size=n_units, p=cohort_probs)
    
    # Covariates for PS model
    x1 = np.random.randn(n_units)
    x2 = np.random.randn(n_units)
    
    # Generate panel data
    data_list = []
    for i in range(n_units):
        for t in range(1, T + 1):
            g = unit_cohorts[i]
            
            # Base outcome with unit and time effects
            y_base = 2.0 + 0.5 * t + 0.3 * x1[i] + 0.2 * x2[i] + np.random.randn() * 0.5
            
            # Treatment effect
            if g > 0 and t >= g:
                tau = 1.5 + 0.3 * (t - g)
                y = y_base + tau
            else:
                y = y_base
            
            data_list.append({
                'id': i + 1,
                'year': t,
                'y': y,
                'gvar': g,
                'x1': x1[i],
                'x2': x2[i],
            })
    
    return pd.DataFrame(data_list)


@pytest.fixture
def mock_cohort_time_effects_with_diagnostics():
    """Create mock CohortTimeEffect objects with diagnostics for unit testing."""
    diag1 = PropensityScoreDiagnostics(
        ps_mean=0.35,
        ps_std=0.15,
        ps_min=0.05,
        ps_max=0.85,
        ps_quantiles={'25%': 0.25, '50%': 0.35, '75%': 0.45},
        weights_cv=1.2,
        extreme_low_pct=0.02,
        extreme_high_pct=0.01,
        overlap_warning=None,
        n_trimmed=3,
    )
    
    diag2 = PropensityScoreDiagnostics(
        ps_mean=0.40,
        ps_std=0.18,
        ps_min=0.08,
        ps_max=0.90,
        ps_quantiles={'25%': 0.28, '50%': 0.40, '75%': 0.52},
        weights_cv=2.5,
        extreme_low_pct=0.05,
        extreme_high_pct=0.03,
        overlap_warning="High weight variation detected (CV > 2.0)",
        n_trimmed=8,
    )
    
    return [
        CohortTimeEffect(
            cohort=3, period=3, event_time=0,
            att=1.5, se=0.3, ci_lower=0.9, ci_upper=2.1,
            t_stat=5.0, pvalue=0.001,
            n_treated=15, n_control=30, n_total=45,
            diagnostics=diag1,
        ),
        CohortTimeEffect(
            cohort=3, period=4, event_time=1,
            att=1.8, se=0.35, ci_lower=1.1, ci_upper=2.5,
            t_stat=5.14, pvalue=0.0008,
            n_treated=15, n_control=25, n_total=40,
            diagnostics=diag2,
        ),
        CohortTimeEffect(
            cohort=4, period=4, event_time=0,
            att=1.6, se=0.32, ci_lower=0.95, ci_upper=2.25,
            t_stat=5.0, pvalue=0.001,
            n_treated=15, n_control=30, n_total=45,
            diagnostics=None,  # No diagnostics (e.g., RA estimator)
        ),
    ]


# ============================================================================
# Unit Tests: results_to_dataframe with include_diagnostics
# ============================================================================

class TestResultsToDataframeWithDiagnostics:
    """Tests for results_to_dataframe(include_diagnostics=True)."""
    
    def test_include_diagnostics_adds_columns(self, mock_cohort_time_effects_with_diagnostics):
        """include_diagnostics=True should add diagnostics columns."""
        results = mock_cohort_time_effects_with_diagnostics
        
        df = results_to_dataframe(results, include_diagnostics=True)
        
        # Check diagnostics columns are present
        diag_cols = [
            'diag_ps_mean', 'diag_ps_std', 'diag_ps_min', 'diag_ps_max',
            'diag_ps_q25', 'diag_ps_q50', 'diag_ps_q75', 'diag_weights_cv',
            'diag_extreme_low_pct', 'diag_extreme_high_pct', 'diag_n_trimmed',
            'diag_overlap_warning'
        ]
        for col in diag_cols:
            assert col in df.columns, f"Missing column: {col}"
    
    def test_include_diagnostics_false_excludes_columns(self, mock_cohort_time_effects_with_diagnostics):
        """include_diagnostics=False should not add diagnostics columns."""
        results = mock_cohort_time_effects_with_diagnostics
        
        df = results_to_dataframe(results, include_diagnostics=False)
        
        # Check diagnostics columns are NOT present
        diag_cols = [
            'diag_ps_mean', 'diag_ps_std', 'diag_ps_min', 'diag_ps_max',
            'diag_ps_q25', 'diag_ps_q50', 'diag_ps_q75', 'diag_weights_cv',
            'diag_extreme_low_pct', 'diag_extreme_high_pct', 'diag_n_trimmed',
            'diag_overlap_warning'
        ]
        for col in diag_cols:
            assert col not in df.columns, f"Unexpected column: {col}"
    
    def test_default_excludes_diagnostics(self, mock_cohort_time_effects_with_diagnostics):
        """Default behavior should not include diagnostics columns."""
        results = mock_cohort_time_effects_with_diagnostics
        
        df = results_to_dataframe(results)
        
        assert 'diag_ps_mean' not in df.columns
        assert 'diag_weights_cv' not in df.columns
    
    def test_diagnostics_values_correct(self, mock_cohort_time_effects_with_diagnostics):
        """Diagnostics values should be correctly extracted."""
        results = mock_cohort_time_effects_with_diagnostics
        
        df = results_to_dataframe(results, include_diagnostics=True)
        
        # Check first result (cohort=3, period=3) with diagnostics
        row0 = df[df['cohort'] == 3].iloc[0]
        assert row0['diag_ps_mean'] == pytest.approx(0.35)
        assert row0['diag_ps_std'] == pytest.approx(0.15)
        assert row0['diag_ps_min'] == pytest.approx(0.05)
        assert row0['diag_ps_max'] == pytest.approx(0.85)
        assert row0['diag_ps_q25'] == pytest.approx(0.25)
        assert row0['diag_ps_q50'] == pytest.approx(0.35)
        assert row0['diag_ps_q75'] == pytest.approx(0.45)
        assert row0['diag_weights_cv'] == pytest.approx(1.2)
        assert row0['diag_extreme_low_pct'] == pytest.approx(0.02)
        assert row0['diag_extreme_high_pct'] == pytest.approx(0.01)
        assert row0['diag_n_trimmed'] == 3
        assert row0['diag_overlap_warning'] is None
    
    def test_none_diagnostics_handled(self, mock_cohort_time_effects_with_diagnostics):
        """Results without diagnostics should have NaN/None values."""
        results = mock_cohort_time_effects_with_diagnostics
        
        df = results_to_dataframe(results, include_diagnostics=True)
        
        # Third result (cohort=4, period=4) has no diagnostics
        row2 = df[(df['cohort'] == 4) & (df['period'] == 4)].iloc[0]
        assert pd.isna(row2['diag_ps_mean'])
        assert pd.isna(row2['diag_weights_cv'])
        assert row2['diag_overlap_warning'] is None
    
    def test_overlap_warning_preserved(self, mock_cohort_time_effects_with_diagnostics):
        """Overlap warning string should be preserved."""
        results = mock_cohort_time_effects_with_diagnostics
        
        df = results_to_dataframe(results, include_diagnostics=True)
        
        # Second result (cohort=3, period=4) has overlap warning
        row1 = df[(df['cohort'] == 3) & (df['period'] == 4)].iloc[0]
        assert "High weight variation" in row1['diag_overlap_warning']
    
    def test_empty_results_with_diagnostics(self):
        """Empty results should return DataFrame with diagnostics columns."""
        df = results_to_dataframe([], include_diagnostics=True)
        
        assert len(df) == 0
        
        # Check all expected columns are present
        expected_cols = [
            'cohort', 'period', 'event_time', 'att', 'se',
            'ci_lower', 'ci_upper', 't_stat', 'pvalue',
            'n_treated', 'n_control', 'n_total',
            'diag_ps_mean', 'diag_ps_std', 'diag_ps_min', 'diag_ps_max',
            'diag_ps_q25', 'diag_ps_q50', 'diag_ps_q75', 'diag_weights_cv',
            'diag_extreme_low_pct', 'diag_extreme_high_pct', 'diag_n_trimmed',
            'diag_overlap_warning'
        ]
        for col in expected_cols:
            assert col in df.columns


# ============================================================================
# Unit Tests: diagnostics_to_dataframe
# ============================================================================

class TestDiagnosticsToDataframe:
    """Tests for diagnostics_to_dataframe() function."""
    
    def test_extracts_diagnostics_only(self, mock_cohort_time_effects_with_diagnostics):
        """Should only include rows with diagnostics."""
        results = mock_cohort_time_effects_with_diagnostics
        
        df = diagnostics_to_dataframe(results)
        
        # Third result has no diagnostics, should be excluded
        assert len(df) == 2
        assert set(df['cohort'].tolist()) == {3}  # Only cohort 3 has diagnostics
    
    def test_column_naming(self, mock_cohort_time_effects_with_diagnostics):
        """Column names should not have 'diag_' prefix."""
        results = mock_cohort_time_effects_with_diagnostics
        
        df = diagnostics_to_dataframe(results)
        
        # Check columns have clean names (no 'diag_' prefix)
        expected_cols = [
            'cohort', 'period', 'event_time',
            'ps_mean', 'ps_std', 'ps_min', 'ps_max',
            'ps_q25', 'ps_q50', 'ps_q75', 'weights_cv',
            'extreme_low_pct', 'extreme_high_pct', 'n_trimmed',
            'overlap_warning'
        ]
        for col in expected_cols:
            assert col in df.columns
    
    def test_values_correct(self, mock_cohort_time_effects_with_diagnostics):
        """Diagnostics values should be correct."""
        results = mock_cohort_time_effects_with_diagnostics
        
        df = diagnostics_to_dataframe(results)
        
        row0 = df[df['period'] == 3].iloc[0]
        assert row0['ps_mean'] == pytest.approx(0.35)
        assert row0['weights_cv'] == pytest.approx(1.2)
        assert row0['n_trimmed'] == 3
    
    def test_empty_when_no_diagnostics(self):
        """Should return empty DataFrame when no diagnostics available."""
        # Results without diagnostics
        results = [
            CohortTimeEffect(
                cohort=3, period=3, event_time=0,
                att=1.5, se=0.3, ci_lower=0.9, ci_upper=2.1,
                t_stat=5.0, pvalue=0.001,
                n_treated=15, n_control=30, n_total=45,
                diagnostics=None,
            ),
        ]
        
        df = diagnostics_to_dataframe(results)
        
        assert len(df) == 0
        assert 'ps_mean' in df.columns
        assert 'weights_cv' in df.columns
    
    def test_empty_results(self):
        """Empty results should return empty DataFrame with columns."""
        df = diagnostics_to_dataframe([])
        
        assert len(df) == 0
        expected_cols = [
            'cohort', 'period', 'event_time',
            'ps_mean', 'ps_std', 'ps_min', 'ps_max',
            'ps_q25', 'ps_q50', 'ps_q75', 'weights_cv',
            'extreme_low_pct', 'extreme_high_pct', 'n_trimmed',
            'overlap_warning'
        ]
        for col in expected_cols:
            assert col in df.columns


# ============================================================================
# Integration Tests: Full Pipeline
# ============================================================================

class TestDiagnosticsExportIntegration:
    """Integration tests with actual estimation."""
    
    def test_ra_estimator_no_diagnostics(self, simple_staggered_data):
        """RA estimator should have no diagnostics."""
        data = simple_staggered_data
        
        transformed = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar'
        )
        
        results = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year',
            estimator='ra',
        )
        
        # All diagnostics should be None
        for r in results:
            assert r.diagnostics is None
        
        # diagnostics_to_dataframe should return empty
        diag_df = diagnostics_to_dataframe(results)
        assert len(diag_df) == 0
        
        # results_to_dataframe with include_diagnostics should have NaN
        df = results_to_dataframe(results, include_diagnostics=True)
        assert df['diag_ps_mean'].isna().all()
    
    def test_ipw_estimator_with_diagnostics(self, simple_staggered_data):
        """IPW estimator with return_diagnostics=True should populate diagnostics."""
        data = simple_staggered_data
        
        transformed = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar'
        )
        
        results = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year',
            estimator='ipw',
            propensity_controls=['x1', 'x2'],
            return_diagnostics=True,
        )
        
        # At least some results should have diagnostics
        has_diagnostics = [r for r in results if r.diagnostics is not None]
        assert len(has_diagnostics) > 0
        
        # diagnostics_to_dataframe should have rows
        diag_df = diagnostics_to_dataframe(results)
        assert len(diag_df) > 0
        
        # results_to_dataframe with include_diagnostics should have values
        df = results_to_dataframe(results, include_diagnostics=True)
        assert not df['diag_ps_mean'].isna().all()
    
    def test_ipwra_estimator_with_diagnostics(self, simple_staggered_data):
        """IPWRA estimator with return_diagnostics=True should populate diagnostics."""
        data = simple_staggered_data
        
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
        
        # At least some results should have diagnostics
        has_diagnostics = [r for r in results if r.diagnostics is not None]
        assert len(has_diagnostics) > 0
        
        # diagnostics_to_dataframe should have rows
        diag_df = diagnostics_to_dataframe(results)
        assert len(diag_df) > 0
        
        # Verify diagnostic values are reasonable
        assert (diag_df['ps_mean'] > 0).all()
        assert (diag_df['ps_mean'] < 1).all()
        assert (diag_df['ps_std'] >= 0).all()
    
    def test_psm_estimator_with_diagnostics(self, simple_staggered_data):
        """PSM estimator with return_diagnostics=True should populate diagnostics."""
        data = simple_staggered_data
        
        transformed = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar'
        )
        
        results = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year',
            estimator='psm',
            propensity_controls=['x1', 'x2'],
            return_diagnostics=True,
            n_neighbors=1,
        )
        
        # At least some results should have diagnostics
        has_diagnostics = [r for r in results if r.diagnostics is not None]
        assert len(has_diagnostics) > 0
        
        # diagnostics_to_dataframe should have rows
        diag_df = diagnostics_to_dataframe(results)
        assert len(diag_df) > 0


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestDiagnosticsExportEdgeCases:
    """Edge case tests for diagnostics export."""
    
    def test_quantiles_missing_keys(self):
        """Handle ps_quantiles with missing keys gracefully."""
        diag = PropensityScoreDiagnostics(
            ps_mean=0.35,
            ps_std=0.15,
            ps_min=0.05,
            ps_max=0.85,
            ps_quantiles={'50%': 0.35},  # Missing 25% and 75%
            weights_cv=1.2,
            extreme_low_pct=0.02,
            extreme_high_pct=0.01,
            overlap_warning=None,
            n_trimmed=3,
        )
        
        results = [
            CohortTimeEffect(
                cohort=3, period=3, event_time=0,
                att=1.5, se=0.3, ci_lower=0.9, ci_upper=2.1,
                t_stat=5.0, pvalue=0.001,
                n_treated=15, n_control=30, n_total=45,
                diagnostics=diag,
            ),
        ]
        
        df = results_to_dataframe(results, include_diagnostics=True)
        
        assert df.iloc[0]['diag_ps_q50'] == pytest.approx(0.35)
        assert pd.isna(df.iloc[0]['diag_ps_q25'])
        assert pd.isna(df.iloc[0]['diag_ps_q75'])
        
        diag_df = diagnostics_to_dataframe(results)
        assert diag_df.iloc[0]['ps_q50'] == pytest.approx(0.35)
        assert pd.isna(diag_df.iloc[0]['ps_q25'])
    
    def test_nan_weights_cv(self):
        """Handle NaN weights_cv correctly."""
        diag = PropensityScoreDiagnostics(
            ps_mean=0.35,
            ps_std=0.15,
            ps_min=0.05,
            ps_max=0.85,
            ps_quantiles={'25%': 0.25, '50%': 0.35, '75%': 0.45},
            weights_cv=np.nan,  # NaN when control group < 2
            extreme_low_pct=0.02,
            extreme_high_pct=0.01,
            overlap_warning=None,
            n_trimmed=0,
        )
        
        results = [
            CohortTimeEffect(
                cohort=3, period=3, event_time=0,
                att=1.5, se=0.3, ci_lower=0.9, ci_upper=2.1,
                t_stat=5.0, pvalue=0.001,
                n_treated=15, n_control=30, n_total=45,
                diagnostics=diag,
            ),
        ]
        
        df = results_to_dataframe(results, include_diagnostics=True)
        assert pd.isna(df.iloc[0]['diag_weights_cv'])
        
        diag_df = diagnostics_to_dataframe(results)
        assert pd.isna(diag_df.iloc[0]['weights_cv'])


# ============================================================================
# Backward Compatibility Tests
# ============================================================================

class TestBackwardCompatibility:
    """Ensure backward compatibility with existing code."""
    
    def test_default_behavior_unchanged(self, mock_cohort_time_effects_with_diagnostics):
        """Default behavior should be unchanged from before."""
        results = mock_cohort_time_effects_with_diagnostics
        
        df = results_to_dataframe(results)
        
        # Check expected columns are present (original behavior)
        expected_cols = [
            'cohort', 'period', 'event_time', 'att', 'se',
            'ci_lower', 'ci_upper', 't_stat', 'pvalue',
            'n_treated', 'n_control', 'n_total'
        ]
        for col in expected_cols:
            assert col in df.columns
        
        # Diagnostics columns should NOT be present
        assert 'diag_ps_mean' not in df.columns
        
        # Check values are correct
        assert len(df) == 3
        assert df.iloc[0]['cohort'] == 3
        assert df.iloc[0]['att'] == pytest.approx(1.5)
    
    def test_existing_tests_pass(self, simple_staggered_data):
        """Existing test patterns should still work."""
        data = simple_staggered_data
        
        transformed = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar'
        )
        
        results = estimate_cohort_time_effects(
            transformed, 'gvar', 'id', 'year'
        )
        
        # Original test pattern
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
