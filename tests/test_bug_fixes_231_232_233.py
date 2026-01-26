"""
Tests for BUG-231, BUG-232, BUG-233 fixes.

BUG-231: ATT fallback calculation using n_treated-weighted average instead of simple mean
BUG-232: AI Robust SE covariance estimation numerical stability warning for small samples
BUG-233: Without-replacement PSM matching order sensitivity warning
"""

import warnings
import numpy as np
import pandas as pd
import pytest

from lwdid import lwdid
from lwdid.staggered.estimators import estimate_psm


class TestBug231WeightedATTFallback:
    """Test BUG-231: n_treated-weighted average for ATT fallback calculation."""

    @pytest.fixture
    def staggered_data_unequal_cohorts(self):
        """Create staggered DID data with unequal cohort sizes.
        
        This tests that the weighted average correctly accounts for
        different cohort sizes when computing the fallback ATT.
        """
        np.random.seed(42)
        
        # Create cohorts with different sizes
        # Cohort 3: 50 units, effect = 2.0
        # Cohort 4: 10 units, effect = 5.0
        # Never-treated: 40 units
        
        n_cohort3 = 50
        n_cohort4 = 10
        n_nt = 40
        n_total = n_cohort3 + n_cohort4 + n_nt
        
        records = []
        unit_id = 0
        
        # Cohort 3 (T=3,4,5,6)
        for _ in range(n_cohort3):
            for t in range(1, 7):
                y = 1.0 + np.random.randn() * 0.5
                if t >= 3:
                    y += 2.0  # ATT = 2.0
                records.append({
                    'unit': unit_id,
                    'time': t,
                    'gvar': 3,
                    'y': y,
                })
            unit_id += 1
        
        # Cohort 4 (T=4,5,6)
        for _ in range(n_cohort4):
            for t in range(1, 7):
                y = 1.0 + np.random.randn() * 0.5
                if t >= 4:
                    y += 5.0  # ATT = 5.0
                records.append({
                    'unit': unit_id,
                    'time': t,
                    'gvar': 4,
                    'y': y,
                })
            unit_id += 1
        
        # Never-treated
        for _ in range(n_nt):
            for t in range(1, 7):
                y = 1.0 + np.random.randn() * 0.5
                records.append({
                    'unit': unit_id,
                    'time': t,
                    'gvar': 0,
                    'y': y,
                })
            unit_id += 1
        
        return pd.DataFrame(records)

    def test_weighted_average_vs_simple_mean(self, staggered_data_unequal_cohorts):
        """Test that fallback ATT uses weighted average, not simple mean.
        
        With cohort sizes 50 (ATT=2.0) and 10 (ATT=5.0):
        - Simple mean: (2.0 + 5.0) / 2 = 3.5 (wrong)
        - Weighted mean: (50*2.0 + 10*5.0) / 60 = 2.5 (correct)
        """
        # Run with aggregate='none' to trigger fallback ATT calculation
        result = lwdid(
            staggered_data_unequal_cohorts,
            y='y',
            gvar='gvar',
            ivar='unit',
            tvar='time',
            rolling='demean',
            aggregate='none',
            control_group='never_treated',
        )
        
        # The 'att' field uses fallback calculation when aggregate='none'
        att_fallback = result.att
        
        # Get cohort-time effects for manual calculation
        ct_effects = result.att_by_cohort_time
        valid_mask = ct_effects['att'].notna()
        
        # Calculate both methods
        simple_mean = ct_effects.loc[valid_mask, 'att'].mean()
        
        weights = ct_effects.loc[valid_mask, 'n_treated']
        weighted_mean = np.average(ct_effects.loc[valid_mask, 'att'], weights=weights)
        
        # Verify fallback uses weighted mean
        assert att_fallback is not None
        assert np.isclose(att_fallback, weighted_mean, rtol=1e-10), \
            f"ATT fallback {att_fallback} != weighted mean {weighted_mean}"
        
        # Note: If cohort sizes differ significantly, simple mean != weighted mean
        if not np.isclose(simple_mean, weighted_mean, rtol=0.1):
            assert not np.isclose(att_fallback, simple_mean, rtol=1e-10), \
                "Fallback should not use simple mean when cohort sizes differ"


class TestBug232AIRobustSESmallSampleWarning:
    """Test BUG-232: Warning for small neighbor samples in AI Robust SE."""

    @pytest.fixture
    def small_sample_psm_data(self):
        """Create small sample data that triggers small-neighbor warning."""
        np.random.seed(42)
        n_treat = 5  # Very small treatment group
        n_control = 8
        
        # Treatment group
        x = np.random.randn(n_treat + n_control)
        d = np.concatenate([np.ones(n_treat), np.zeros(n_control)]).astype(int)
        
        # Outcome with treatment effect
        y = 1.0 + 2.0 * d + 0.5 * x + np.random.randn(n_treat + n_control) * 0.3
        
        return pd.DataFrame({
            'y': y,
            'd': d,
            'x': x,
        })

    def test_small_sample_warning_issued(self, small_sample_psm_data):
        """Verify warning is issued for small nearest neighbor samples."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = estimate_psm(
                small_sample_psm_data,
                y='y',
                d='d',
                propensity_controls=['x'],
                se_method='abadie_imbens_full',
                adjust_for_estimated_ps=True,
                variance_estimation_j=2,  # h=2, very small
            )
            
            # Check for the small sample warning
            small_sample_warnings = [
                warning for warning in w
                if 'Small nearest neighbor sample' in str(warning.message)
                or 'numerically unstable' in str(warning.message).lower()
            ]
            
            # With n_treat=5 and h=2, h_actual_treat = min(2, 4) = 2 < 4
            # Warning should be issued
            assert len(small_sample_warnings) > 0, \
                "Expected warning for small nearest neighbor sample"

    def test_no_warning_for_adequate_samples(self):
        """Verify no warning for adequate sample sizes."""
        np.random.seed(42)
        n_treat = 30
        n_control = 50
        
        x = np.random.randn(n_treat + n_control)
        d = np.concatenate([np.ones(n_treat), np.zeros(n_control)]).astype(int)
        y = 1.0 + 2.0 * d + 0.5 * x + np.random.randn(n_treat + n_control) * 0.3
        
        data = pd.DataFrame({'y': y, 'd': d, 'x': x})
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = estimate_psm(
                data,
                y='y',
                d='d',
                propensity_controls=['x'],
                se_method='abadie_imbens_full',
                adjust_for_estimated_ps=True,
                variance_estimation_j=4,  # h=4, adequate
            )
            
            small_sample_warnings = [
                warning for warning in w
                if 'Small nearest neighbor sample' in str(warning.message)
            ]
            
            assert len(small_sample_warnings) == 0, \
                "Unexpected warning for adequate sample size"


class TestBug233WithoutReplacementMatchOrderWarning:
    """Test BUG-233: Sensitivity analysis warning for without-replacement PSM."""

    @pytest.fixture
    def psm_data(self):
        """Create data for PSM tests."""
        np.random.seed(42)
        n_treat = 30
        n_control = 50
        
        x = np.random.randn(n_treat + n_control)
        d = np.concatenate([np.ones(n_treat), np.zeros(n_control)]).astype(int)
        y = 1.0 + 2.0 * d + 0.5 * x + np.random.randn(n_treat + n_control) * 0.3
        
        return pd.DataFrame({
            'y': y,
            'd': d,
            'x': x,
        })

    def test_warning_for_without_replacement_data_order(self, psm_data):
        """Verify warning for without-replacement with default 'data' order."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = estimate_psm(
                psm_data,
                y='y',
                d='d',
                propensity_controls=['x'],
                with_replacement=False,
                match_order='data',
            )
            
            match_order_warnings = [
                warning for warning in w
                if 'without-replacement' in str(warning.message).lower()
                or 'greedy' in str(warning.message).lower()
                or 'processing order' in str(warning.message).lower()
            ]
            
            assert len(match_order_warnings) > 0, \
                "Expected warning for without-replacement matching with data order"

    def test_no_warning_for_with_replacement(self, psm_data):
        """Verify no matching order warning for with-replacement matching."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = estimate_psm(
                psm_data,
                y='y',
                d='d',
                propensity_controls=['x'],
                with_replacement=True,
                match_order='data',  # Order doesn't matter for with-replacement
            )
            
            match_order_warnings = [
                warning for warning in w
                if 'without-replacement' in str(warning.message).lower()
                and 'greedy' in str(warning.message).lower()
            ]
            
            assert len(match_order_warnings) == 0, \
                "Unexpected warning for with-replacement matching"

    def test_no_warning_for_non_data_order(self, psm_data):
        """Verify no warning when using non-default match order."""
        for order in ['random', 'largest', 'smallest']:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                result = estimate_psm(
                    psm_data,
                    y='y',
                    d='d',
                    propensity_controls=['x'],
                    with_replacement=False,
                    match_order=order,
                    seed=42 if order == 'random' else None,
                )
                
                # Warning should mention 'data' order specifically
                data_order_warnings = [
                    warning for warning in w
                    if "Using default 'data' order" in str(warning.message)
                ]
                
                assert len(data_order_warnings) == 0, \
                    f"Unexpected 'data' order warning for match_order='{order}'"

    def test_match_order_affects_results(self, psm_data):
        """Verify that different match orders can produce different results."""
        results = {}
        for order in ['data', 'largest', 'smallest']:
            result = estimate_psm(
                psm_data,
                y='y',
                d='d',
                propensity_controls=['x'],
                with_replacement=False,
                match_order=order,
            )
            results[order] = result.att
        
        # Different orders should potentially produce different ATT estimates
        # (not guaranteed, but the mechanism should work)
        # Main test is that all orders complete successfully
        for order, att in results.items():
            assert not np.isnan(att), f"ATT is NaN for match_order='{order}'"


class TestBug231NumericalValidation:
    """Numerical validation tests for BUG-231 weighted average fix."""

    def test_weighted_average_formula_correctness(self):
        """Directly test the weighted average formula."""
        # Simulate cohort-time ATT estimates
        atts = np.array([2.0, 2.0, 5.0])  # Two estimates for cohort 3, one for cohort 4
        n_treated = np.array([50, 50, 10])  # Cohort 3 larger
        
        # Weighted average
        weighted = np.average(atts, weights=n_treated)
        # = (2.0*50 + 2.0*50 + 5.0*10) / (50+50+10)
        # = (100 + 100 + 50) / 110 = 250/110 ≈ 2.27
        
        expected = (2.0 * 50 + 2.0 * 50 + 5.0 * 10) / (50 + 50 + 10)
        
        assert np.isclose(weighted, expected), \
            f"Weighted average {weighted} != expected {expected}"
        
        # Simple mean would be different
        simple = np.mean(atts)
        # = (2.0 + 2.0 + 5.0) / 3 = 3.0
        
        assert not np.isclose(simple, weighted), \
            "Simple mean should differ from weighted mean for unequal weights"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
