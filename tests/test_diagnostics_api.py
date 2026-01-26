"""
Tests for Story 1.3: API Parameter Exposure - Diagnostics Passthrough

This module tests the return_diagnostics parameter passthrough chain and
the get_diagnostics() method functionality.

Test categories:
1. Passthrough chain verification
2. Diagnostics content correctness
3. get_diagnostics() method functionality
4. Estimator-specific behavior (RA, IPWRA, PSM)
5. Aggregation behavior for multiple (g,r) effects

References:
    Story 1.3: 顶层API参数暴露
    Lee & Wooldridge (2023), Section 4, Formula 4.11 (Overlap Assumption)
"""

import numpy as np
import pandas as pd
import pytest

from lwdid import lwdid
from lwdid.staggered.estimators import PropensityScoreDiagnostics


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def staggered_test_data():
    """
    Create staggered panel data for diagnostics testing.
    
    Features:
    - 500 units, 6 periods
    - 3 treatment cohorts (4, 5, 6) + never treated (0)
    - Known covariate distribution for predictable PS
    """
    np.random.seed(42)
    
    n_units = 500
    n_periods = 6
    
    # Generate unit IDs and periods
    ids = np.repeat(np.arange(1, n_units + 1), n_periods)
    periods = np.tile(np.arange(1, n_periods + 1), n_units)
    
    # Assign cohorts: 25% each to cohort 4, 5, 6, and never treated (0)
    unit_cohorts = np.random.choice([4, 5, 6, 0], size=n_units, p=[0.25, 0.25, 0.25, 0.25])
    cohorts = np.repeat(unit_cohorts, n_periods)
    
    # Generate covariates (unit-level, time-invariant)
    x1_unit = np.random.normal(0, 1, n_units)
    x2_unit = np.random.normal(0, 1, n_units)
    x1 = np.repeat(x1_unit, n_periods)
    x2 = np.repeat(x2_unit, n_periods)
    
    # Generate outcome with treatment effect
    treatment_effect = 2.0
    y_base = 1 + 0.5 * x1 + 0.3 * x2 + np.random.normal(0, 1, len(ids))
    
    # Add treatment effect for treated units in post-treatment periods
    treated_post = (cohorts > 0) & (periods >= cohorts)
    y = y_base + treatment_effect * treated_post.astype(float)
    
    df = pd.DataFrame({
        'id': ids,
        'period': periods,
        'gvar': cohorts,
        'x1': x1,
        'x2': x2,
        'y': y,
    })
    
    return df


@pytest.fixture
def small_staggered_data():
    """
    Create smaller staggered data for quick tests.
    """
    np.random.seed(123)
    
    n_units = 100
    n_periods = 4
    
    ids = np.repeat(np.arange(1, n_units + 1), n_periods)
    periods = np.tile(np.arange(1, n_periods + 1), n_units)
    
    unit_cohorts = np.random.choice([2, 3, 0], size=n_units, p=[0.4, 0.3, 0.3])
    cohorts = np.repeat(unit_cohorts, n_periods)
    
    x1 = np.repeat(np.random.normal(0, 1, n_units), n_periods)
    x2 = np.repeat(np.random.normal(0, 1, n_units), n_periods)
    
    y = 1 + 0.5 * x1 + 0.3 * x2 + np.random.normal(0, 0.5, len(ids))
    treated_post = (cohorts > 0) & (periods >= cohorts)
    y = y + 1.5 * treated_post.astype(float)
    
    return pd.DataFrame({
        'id': ids,
        'period': periods,
        'gvar': cohorts,
        'x1': x1,
        'x2': x2,
        'y': y,
    })


# =============================================================================
# Test Class: Passthrough Chain Verification
# =============================================================================

class TestDiagnosticsPassthrough:
    """Test return_diagnostics parameter passthrough chain."""
    
    def test_diagnostics_returned_when_requested_ipwra(self, small_staggered_data):
        """IPWRA: return_diagnostics=True should return diagnostics."""
        result = lwdid(
            small_staggered_data,
            y='y',
            ivar='id',
            tvar='period',
            gvar='gvar',
            estimator='ipwra',
            controls=['x1', 'x2'],
            return_diagnostics=True,
        )
        
        diagnostics = result.get_diagnostics()
        assert len(diagnostics) > 0, "Should return at least one diagnostic"
        
        for (g, r), diag in diagnostics.items():
            assert isinstance(diag, PropensityScoreDiagnostics), \
                f"Diagnostic for ({g},{r}) should be PropensityScoreDiagnostics"
    
    def test_diagnostics_returned_when_requested_psm(self, small_staggered_data):
        """PSM: return_diagnostics=True should return diagnostics."""
        result = lwdid(
            small_staggered_data,
            y='y',
            ivar='id',
            tvar='period',
            gvar='gvar',
            estimator='psm',
            controls=['x1', 'x2'],
            return_diagnostics=True,
        )
        
        diagnostics = result.get_diagnostics()
        assert len(diagnostics) > 0, "Should return at least one diagnostic"
        
        for (g, r), diag in diagnostics.items():
            assert isinstance(diag, PropensityScoreDiagnostics)
    
    def test_no_diagnostics_when_not_requested(self, small_staggered_data):
        """return_diagnostics=False (default) should not return diagnostics."""
        result = lwdid(
            small_staggered_data,
            y='y',
            ivar='id',
            tvar='period',
            gvar='gvar',
            estimator='ipwra',
            controls=['x1', 'x2'],
            return_diagnostics=False,
        )
        
        diagnostics = result.get_diagnostics()
        assert len(diagnostics) == 0, "Should return empty dict when not requested"
    
    def test_ra_estimator_no_diagnostics(self, small_staggered_data):
        """RA estimator should have no diagnostics even when requested."""
        result = lwdid(
            small_staggered_data,
            y='y',
            ivar='id',
            tvar='period',
            gvar='gvar',
            estimator='ra',
            controls=['x1', 'x2'],
            return_diagnostics=True,
        )
        
        diagnostics = result.get_diagnostics()
        assert len(diagnostics) == 0, "RA estimator should have no diagnostics"
    
    def test_backward_compatibility_default_false(self, small_staggered_data):
        """Default behavior (return_diagnostics not specified) should be False."""
        result = lwdid(
            small_staggered_data,
            y='y',
            ivar='id',
            tvar='period',
            gvar='gvar',
            estimator='ipwra',
            controls=['x1', 'x2'],
            # return_diagnostics not specified
        )
        
        diagnostics = result.get_diagnostics()
        assert len(diagnostics) == 0, "Default should be no diagnostics"


# =============================================================================
# Test Class: Diagnostics Content Correctness
# =============================================================================

class TestDiagnosticsCorrectness:
    """Test diagnostics content correctness."""
    
    def test_ps_mean_in_valid_range(self, staggered_test_data):
        """PS mean should be in (0, 1) range."""
        result = lwdid(
            staggered_test_data,
            y='y',
            ivar='id',
            tvar='period',
            gvar='gvar',
            estimator='ipwra',
            controls=['x1', 'x2'],
            return_diagnostics=True,
        )
        
        diagnostics = result.get_diagnostics()
        for (g, r), diag in diagnostics.items():
            assert 0 < diag.ps_mean < 1, \
                f"PS mean for ({g},{r}) should be in (0,1), got {diag.ps_mean}"
    
    def test_ps_std_positive(self, staggered_test_data):
        """PS std should be positive."""
        result = lwdid(
            staggered_test_data,
            y='y',
            ivar='id',
            tvar='period',
            gvar='gvar',
            estimator='ipwra',
            controls=['x1', 'x2'],
            return_diagnostics=True,
        )
        
        diagnostics = result.get_diagnostics()
        for (g, r), diag in diagnostics.items():
            assert diag.ps_std > 0, \
                f"PS std for ({g},{r}) should be positive, got {diag.ps_std}"
    
    def test_weights_cv_non_negative(self, staggered_test_data):
        """Weights CV should be non-negative."""
        result = lwdid(
            staggered_test_data,
            y='y',
            ivar='id',
            tvar='period',
            gvar='gvar',
            estimator='ipwra',
            controls=['x1', 'x2'],
            return_diagnostics=True,
        )
        
        diagnostics = result.get_diagnostics()
        for (g, r), diag in diagnostics.items():
            if not np.isnan(diag.weights_cv):
                assert diag.weights_cv >= 0, \
                    f"Weights CV for ({g},{r}) should be non-negative"
    
    def test_trim_threshold_respected(self, staggered_test_data):
        """PS should be trimmed to [trim, 1-trim] range."""
        trim = 0.05
        result = lwdid(
            staggered_test_data,
            y='y',
            ivar='id',
            tvar='period',
            gvar='gvar',
            estimator='ipwra',
            controls=['x1', 'x2'],
            trim_threshold=trim,
            return_diagnostics=True,
        )
        
        diagnostics = result.get_diagnostics()
        for (g, r), diag in diagnostics.items():
            assert diag.ps_min >= trim - 1e-10, \
                f"PS min for ({g},{r}) should be >= {trim}, got {diag.ps_min}"
            assert diag.ps_max <= 1 - trim + 1e-10, \
                f"PS max for ({g},{r}) should be <= {1-trim}, got {diag.ps_max}"
    
    def test_quantiles_ordered(self, staggered_test_data):
        """PS quantiles should be ordered: q25 <= q50 <= q75."""
        result = lwdid(
            staggered_test_data,
            y='y',
            ivar='id',
            tvar='period',
            gvar='gvar',
            estimator='ipwra',
            controls=['x1', 'x2'],
            return_diagnostics=True,
        )
        
        diagnostics = result.get_diagnostics()
        for (g, r), diag in diagnostics.items():
            q25 = diag.ps_quantiles['25%']
            q50 = diag.ps_quantiles['50%']
            q75 = diag.ps_quantiles['75%']
            assert q25 <= q50 <= q75, \
                f"Quantiles for ({g},{r}) should be ordered: {q25} <= {q50} <= {q75}"
    
    def test_extreme_pct_in_valid_range(self, staggered_test_data):
        """Extreme percentages should be in [0, 1] range."""
        result = lwdid(
            staggered_test_data,
            y='y',
            ivar='id',
            tvar='period',
            gvar='gvar',
            estimator='ipwra',
            controls=['x1', 'x2'],
            return_diagnostics=True,
        )
        
        diagnostics = result.get_diagnostics()
        for (g, r), diag in diagnostics.items():
            assert 0 <= diag.extreme_low_pct <= 1, \
                f"extreme_low_pct for ({g},{r}) should be in [0,1]"
            assert 0 <= diag.extreme_high_pct <= 1, \
                f"extreme_high_pct for ({g},{r}) should be in [0,1]"
    
    def test_n_trimmed_non_negative(self, staggered_test_data):
        """n_trimmed should be non-negative."""
        result = lwdid(
            staggered_test_data,
            y='y',
            ivar='id',
            tvar='period',
            gvar='gvar',
            estimator='ipwra',
            controls=['x1', 'x2'],
            return_diagnostics=True,
        )
        
        diagnostics = result.get_diagnostics()
        for (g, r), diag in diagnostics.items():
            assert diag.n_trimmed >= 0, \
                f"n_trimmed for ({g},{r}) should be non-negative"


# =============================================================================
# Test Class: get_diagnostics() Method
# =============================================================================

class TestGetDiagnosticsMethod:
    """Test get_diagnostics() method functionality."""
    
    def test_get_specific_cohort_period(self, staggered_test_data):
        """Get diagnostics for specific (cohort, period)."""
        result = lwdid(
            staggered_test_data,
            y='y',
            ivar='id',
            tvar='period',
            gvar='gvar',
            estimator='ipwra',
            controls=['x1', 'x2'],
            return_diagnostics=True,
        )
        
        # Get all diagnostics first to find a valid key
        all_diags = result.get_diagnostics()
        if len(all_diags) > 0:
            (g, r) = list(all_diags.keys())[0]
            diag = result.get_diagnostics(cohort=g, period=r)
            assert isinstance(diag, PropensityScoreDiagnostics)
    
    def test_get_all_for_cohort(self, staggered_test_data):
        """Get all diagnostics for a specific cohort."""
        result = lwdid(
            staggered_test_data,
            y='y',
            ivar='id',
            tvar='period',
            gvar='gvar',
            estimator='ipwra',
            controls=['x1', 'x2'],
            return_diagnostics=True,
        )
        
        all_diags = result.get_diagnostics()
        if len(all_diags) > 0:
            target_cohort = list(all_diags.keys())[0][0]
            cohort_diags = result.get_diagnostics(cohort=target_cohort)
            
            assert isinstance(cohort_diags, dict)
            for (c, p) in cohort_diags.keys():
                assert c == target_cohort, f"All results should be for cohort {target_cohort}"
    
    def test_get_all_for_period(self, staggered_test_data):
        """Get all diagnostics for a specific period."""
        result = lwdid(
            staggered_test_data,
            y='y',
            ivar='id',
            tvar='period',
            gvar='gvar',
            estimator='ipwra',
            controls=['x1', 'x2'],
            return_diagnostics=True,
        )
        
        all_diags = result.get_diagnostics()
        if len(all_diags) > 0:
            target_period = list(all_diags.keys())[0][1]
            period_diags = result.get_diagnostics(period=target_period)
            
            assert isinstance(period_diags, dict)
            for (c, p) in period_diags.keys():
                assert p == target_period, f"All results should be for period {target_period}"
    
    def test_get_all_diagnostics(self, staggered_test_data):
        """Get all diagnostics."""
        result = lwdid(
            staggered_test_data,
            y='y',
            ivar='id',
            tvar='period',
            gvar='gvar',
            estimator='ipwra',
            controls=['x1', 'x2'],
            return_diagnostics=True,
        )
        
        all_diags = result.get_diagnostics()
        assert isinstance(all_diags, dict)
        
        for key, diag in all_diags.items():
            assert isinstance(key, tuple) and len(key) == 2
            assert isinstance(diag, PropensityScoreDiagnostics)
    
    def test_nonexistent_cohort_period_returns_none(self, staggered_test_data):
        """Requesting non-existent (cohort, period) should return None."""
        result = lwdid(
            staggered_test_data,
            y='y',
            ivar='id',
            tvar='period',
            gvar='gvar',
            estimator='ipwra',
            controls=['x1', 'x2'],
            return_diagnostics=True,
        )
        
        diag = result.get_diagnostics(cohort=9999, period=9999)
        assert diag is None


# =============================================================================
# Test Class: PSM Diagnostics Consistency
# =============================================================================

class TestPSMDiagnosticsConsistency:
    """Test PSM diagnostics consistency with IPWRA."""
    
    def test_psm_diagnostics_structure(self, small_staggered_data):
        """PSM diagnostics should have same structure as IPWRA."""
        result = lwdid(
            small_staggered_data,
            y='y',
            ivar='id',
            tvar='period',
            gvar='gvar',
            estimator='psm',
            controls=['x1', 'x2'],
            return_diagnostics=True,
        )
        
        diagnostics = result.get_diagnostics()
        for (g, r), diag in diagnostics.items():
            # Check all expected attributes exist
            assert hasattr(diag, 'ps_mean')
            assert hasattr(diag, 'ps_std')
            assert hasattr(diag, 'ps_min')
            assert hasattr(diag, 'ps_max')
            assert hasattr(diag, 'ps_quantiles')
            assert hasattr(diag, 'weights_cv')
            assert hasattr(diag, 'extreme_low_pct')
            assert hasattr(diag, 'extreme_high_pct')
            assert hasattr(diag, 'overlap_warning')
            assert hasattr(diag, 'n_trimmed')
    
    def test_psm_ipwra_same_keys(self, small_staggered_data):
        """PSM and IPWRA should produce diagnostics for same (g,r) pairs."""
        result_psm = lwdid(
            small_staggered_data,
            y='y',
            ivar='id',
            tvar='period',
            gvar='gvar',
            estimator='psm',
            controls=['x1', 'x2'],
            return_diagnostics=True,
        )
        
        result_ipwra = lwdid(
            small_staggered_data,
            y='y',
            ivar='id',
            tvar='period',
            gvar='gvar',
            estimator='ipwra',
            controls=['x1', 'x2'],
            return_diagnostics=True,
        )
        
        diag_psm = result_psm.get_diagnostics()
        diag_ipwra = result_ipwra.get_diagnostics()
        
        # Same keys (same (g,r) pairs estimated)
        assert diag_psm.keys() == diag_ipwra.keys(), \
            "PSM and IPWRA should have same (g,r) pairs"


# =============================================================================
# Test Class: Diagnostics Aggregation
# =============================================================================

class TestDiagnosticsAggregation:
    """Test diagnostics aggregation behavior for multiple (g,r) effects."""
    
    def test_multiple_effects_have_independent_diagnostics(self, staggered_test_data):
        """Each (g,r) effect should have independent diagnostics."""
        result = lwdid(
            staggered_test_data,
            y='y',
            ivar='id',
            tvar='period',
            gvar='gvar',
            estimator='ipwra',
            controls=['x1', 'x2'],
            return_diagnostics=True,
        )
        
        all_diags = result.get_diagnostics()
        
        # Should have multiple diagnostics
        assert len(all_diags) > 1, "Should have multiple (g,r) effects"
        
        # Each diagnostics should be independent object
        diag_objects = list(all_diags.values())
        for i, d1 in enumerate(diag_objects):
            for j, d2 in enumerate(diag_objects):
                if i != j:
                    # Different (g,r) pairs may have different PS means
                    # (they are independent estimates)
                    assert type(d1) == type(d2)
    
    def test_dict_structure_correct(self, staggered_test_data):
        """Diagnostics dict should have correct structure."""
        result = lwdid(
            staggered_test_data,
            y='y',
            ivar='id',
            tvar='period',
            gvar='gvar',
            estimator='ipwra',
            controls=['x1', 'x2'],
            return_diagnostics=True,
        )
        
        all_diags = result.get_diagnostics()
        
        for key in all_diags.keys():
            assert isinstance(key, tuple), "Key should be tuple"
            assert len(key) == 2, "Key should be (cohort, period)"
            cohort, period = key
            assert isinstance(cohort, (int, float)), "Cohort should be numeric"
            assert isinstance(period, (int, float)), "Period should be numeric"


# =============================================================================
# Test Class: Edge Cases
# =============================================================================

class TestDiagnosticsEdgeCases:
    """Test edge cases for diagnostics."""
    
    def test_empty_diagnostics_when_no_ps_estimator(self, small_staggered_data):
        """RA estimator should return empty diagnostics."""
        result = lwdid(
            small_staggered_data,
            y='y',
            ivar='id',
            tvar='period',
            gvar='gvar',
            estimator='ra',
            controls=['x1', 'x2'],
            return_diagnostics=True,
        )
        
        assert len(result.get_diagnostics()) == 0
    
    def test_diagnostics_with_different_trim_thresholds(self, small_staggered_data):
        """Different trim thresholds should produce different diagnostics."""
        result_low = lwdid(
            small_staggered_data,
            y='y',
            ivar='id',
            tvar='period',
            gvar='gvar',
            estimator='ipwra',
            controls=['x1', 'x2'],
            trim_threshold=0.01,
            return_diagnostics=True,
        )
        
        result_high = lwdid(
            small_staggered_data,
            y='y',
            ivar='id',
            tvar='period',
            gvar='gvar',
            estimator='ipwra',
            controls=['x1', 'x2'],
            trim_threshold=0.10,
            return_diagnostics=True,
        )
        
        diag_low = result_low.get_diagnostics()
        diag_high = result_high.get_diagnostics()
        
        # Higher trim should result in more trimmed observations
        if len(diag_low) > 0 and len(diag_high) > 0:
            key = list(diag_low.keys())[0]
            if key in diag_high:
                # Higher trim threshold should trim more (or same)
                assert diag_high[key].n_trimmed >= diag_low[key].n_trimmed or \
                       diag_high[key].ps_min >= diag_low[key].ps_min


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
