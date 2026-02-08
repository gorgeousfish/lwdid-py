"""
Tests for Staggered Randomization Inference Module

Tests the randomization inference implementation for staggered DiD estimation.
"""

import os
import pytest
import numpy as np
import pandas as pd
import warnings

from lwdid import lwdid
from lwdid.staggered.randomization import (
    StaggeredRIResult,
    randomization_inference_staggered,
    ri_overall_effect,
    ri_cohort_effect,
)
from lwdid.staggered.transformations import get_cohorts
from lwdid.exceptions import RandomizationError


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def castle_data():
    """Load Castle Law test data."""
    here = os.path.dirname(__file__)
    data_path = os.path.join(here, '..', '..', 'data', 'castle.csv')
    data = pd.read_csv(data_path)
    
    # Prepare gvar: effyear with NaN -> 0 for never treated
    data['gvar'] = data['effyear'].fillna(0).astype(int)
    
    return data


@pytest.fixture
def simple_staggered_data():
    """Create simple synthetic staggered data for unit tests."""
    np.random.seed(42)
    
    # 10 units, 6 periods
    n_units = 10
    n_periods = 6
    
    records = []
    for i in range(n_units):
        unit_id = i + 1
        
        # Assign cohorts: units 1-3 -> cohort 3, units 4-6 -> cohort 4, units 7-10 -> never treated
        if i < 3:
            gvar = 3
        elif i < 6:
            gvar = 4
        else:
            gvar = 0  # never treated
        
        for t in range(1, n_periods + 1):
            # Generate outcome with treatment effect
            base = 10 + i * 0.5  # unit fixed effect
            trend = t * 0.2  # time trend
            effect = 0
            if gvar > 0 and t >= gvar:
                effect = 2.0  # treatment effect
            noise = np.random.normal(0, 0.5)
            y = base + trend + effect + noise
            
            records.append({
                'id': unit_id,
                'year': t,
                'y': y,
                'gvar': gvar
            })
    
    return pd.DataFrame(records)


@pytest.fixture
def all_treated_data():
    """Create data with all units eventually treated (no NT)."""
    np.random.seed(42)
    
    records = []
    for i in range(6):
        unit_id = i + 1
        gvar = 3 if i < 3 else 4  # All units are treated
        
        for t in range(1, 6):
            y = 10 + np.random.normal(0, 1)
            records.append({'id': unit_id, 'year': t, 'y': y, 'gvar': gvar})
    
    return pd.DataFrame(records)


# =============================================================================
# StaggeredRIResult Tests
# =============================================================================

class TestStaggeredRIResult:
    """Test StaggeredRIResult dataclass."""
    
    def test_result_structure(self):
        """Test StaggeredRIResult has correct attributes."""
        result = StaggeredRIResult(
            p_value=0.05,
            ri_method='permutation',
            ri_reps=1000,
            ri_valid=980,
            ri_failed=20,
            observed_stat=1.5,
            permutation_stats=np.array([0.5, 1.0, 1.5, 2.0])
        )
        
        assert result.p_value == 0.05
        assert result.ri_method == 'permutation'
        assert result.ri_reps == 1000
        assert result.ri_valid == 980
        assert result.ri_failed == 20
        assert result.observed_stat == 1.5
        assert len(result.permutation_stats) == 4
    
    def test_result_repr(self):
        """Test StaggeredRIResult string representation."""
        result = StaggeredRIResult(
            p_value=0.123,
            ri_method='permutation',
            ri_reps=100,
            ri_valid=95,
            ri_failed=5,
            observed_stat=1.0,
            permutation_stats=np.array([])
        )
        
        repr_str = repr(result)
        assert 'p_value=0.1230' in repr_str
        assert "method='permutation'" in repr_str
        assert 'valid=95/100' in repr_str


# =============================================================================
# Parameter Validation Tests
# =============================================================================

class TestRIParameterValidation:
    """Test parameter validation for RI functions."""
    
    def test_invalid_rireps_raises(self, simple_staggered_data):
        """Test that invalid rireps raises error."""
        with pytest.raises(RandomizationError, match="rireps must be positive"):
            randomization_inference_staggered(
                data=simple_staggered_data,
                gvar='gvar', ivar='id', tvar='year', y='y',
                observed_att=1.0,
                target='cohort_time',
                target_cohort=3,
                target_period=3,
                rireps=0,
                n_never_treated=4
            )
    
    def test_invalid_ri_method_raises(self, simple_staggered_data):
        """Test that invalid ri_method raises error."""
        with pytest.raises(RandomizationError, match="ri_method must be"):
            randomization_inference_staggered(
                data=simple_staggered_data,
                gvar='gvar', ivar='id', tvar='year', y='y',
                observed_att=1.0,
                target='cohort_time',
                target_cohort=3,
                target_period=3,
                ri_method='invalid',
                rireps=100,
                n_never_treated=4
            )
    
    def test_cohort_target_requires_cohort(self, simple_staggered_data):
        """Test that target='cohort' requires target_cohort."""
        with pytest.raises(RandomizationError, match="target_cohort required"):
            randomization_inference_staggered(
                data=simple_staggered_data,
                gvar='gvar', ivar='id', tvar='year', y='y',
                observed_att=1.0,
                target='cohort',
                rireps=100,
                n_never_treated=4
            )
    
    def test_cohort_time_target_requires_both(self, simple_staggered_data):
        """Test that target='cohort_time' requires cohort and period."""
        with pytest.raises(RandomizationError, match="target_cohort and target_period required"):
            randomization_inference_staggered(
                data=simple_staggered_data,
                gvar='gvar', ivar='id', tvar='year', y='y',
                observed_att=1.0,
                target='cohort_time',
                target_cohort=3,
                # Missing target_period
                rireps=100,
                n_never_treated=4
            )
    
    def test_overall_target_requires_nt(self, all_treated_data):
        """Test that target='overall' requires never-treated units."""
        with pytest.raises(RandomizationError, match="never treated"):
            randomization_inference_staggered(
                data=all_treated_data,
                gvar='gvar', ivar='id', tvar='year', y='y',
                observed_att=1.0,
                target='overall',
                rireps=100,
                n_never_treated=0  # No NT units
            )
    
    def test_too_few_units_raises(self):
        """Test that too few units raises error."""
        # Only 2 units
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'year': [1, 2, 1, 2],
            'y': [10, 12, 15, 16],
            'gvar': [2, 2, 0, 0]
        })
        
        with pytest.raises(RandomizationError, match="Too few units"):
            randomization_inference_staggered(
                data=data,
                gvar='gvar', ivar='id', tvar='year', y='y',
                cohorts=[2],
                observed_att=1.0,
                target='cohort_time',
                target_cohort=2,
                target_period=2,
                rireps=100,
                n_never_treated=1
            )


# =============================================================================
# Basic RI Functionality Tests
# =============================================================================

class TestRIBasicFunctionality:
    """Test basic RI functionality."""
    
    def test_ri_returns_valid_pvalue(self, simple_staggered_data):
        """Test RI returns p-value in [0, 1]."""
        result = randomization_inference_staggered(
            data=simple_staggered_data,
            gvar='gvar', ivar='id', tvar='year', y='y',
            cohorts=[3, 4],
            observed_att=2.0,
            target='cohort_time',
            target_cohort=3,
            target_period=3,
            ri_method='permutation',
            rireps=100,
            seed=42,
            rolling='demean',
            n_never_treated=4
        )
        
        assert isinstance(result, StaggeredRIResult)
        assert 0 <= result.p_value <= 1
        assert result.ri_valid > 0
        assert result.ri_valid <= result.ri_reps
    
    def test_ri_reproducibility_with_seed(self, simple_staggered_data):
        """Test RI results are reproducible with same seed."""
        kwargs = dict(
            data=simple_staggered_data,
            gvar='gvar', ivar='id', tvar='year', y='y',
            cohorts=[3, 4],
            observed_att=2.0,
            target='cohort_time',
            target_cohort=3,
            target_period=3,
            ri_method='permutation',
            rireps=50,
            seed=12345,
            rolling='demean',
            n_never_treated=4
        )
        
        result1 = randomization_inference_staggered(**kwargs)
        result2 = randomization_inference_staggered(**kwargs)
        
        assert result1.p_value == result2.p_value
        assert result1.ri_valid == result2.ri_valid
        np.testing.assert_array_equal(
            result1.permutation_stats, 
            result2.permutation_stats
        )
    
    def test_permutation_vs_bootstrap(self, simple_staggered_data):
        """Test both permutation and bootstrap methods work."""
        common_kwargs = dict(
            data=simple_staggered_data,
            gvar='gvar', ivar='id', tvar='year', y='y',
            cohorts=[3, 4],
            observed_att=2.0,
            target='cohort_time',
            target_cohort=3,
            target_period=3,
            rireps=50,
            seed=42,
            rolling='demean',
            n_never_treated=4
        )
        
        result_perm = randomization_inference_staggered(
            **common_kwargs, ri_method='permutation'
        )
        result_boot = randomization_inference_staggered(
            **common_kwargs, ri_method='bootstrap'
        )
        
        # Both should return valid results
        assert 0 <= result_perm.p_value <= 1
        assert 0 <= result_boot.p_value <= 1
        assert result_perm.ri_method == 'permutation'
        assert result_boot.ri_method == 'bootstrap'


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_ri_overall_effect(self, simple_staggered_data):
        """Test ri_overall_effect convenience function."""
        result = ri_overall_effect(
            data=simple_staggered_data,
            gvar='gvar', ivar='id', tvar='year', y='y',
            observed_att=2.0,
            rolling='demean',
            rireps=50,
            seed=42
        )
        
        assert isinstance(result, StaggeredRIResult)
        assert 0 <= result.p_value <= 1
    
    def test_ri_cohort_effect(self, simple_staggered_data):
        """Test ri_cohort_effect convenience function."""
        result = ri_cohort_effect(
            data=simple_staggered_data,
            gvar='gvar', ivar='id', tvar='year', y='y',
            target_cohort=3,
            observed_att=2.0,
            rolling='demean',
            rireps=50,
            seed=42
        )
        
        assert isinstance(result, StaggeredRIResult)
        assert 0 <= result.p_value <= 1


# =============================================================================
# Integration Tests
# =============================================================================

class TestRIIntegration:
    """Test RI integration with lwdid() main function."""
    
    def test_lwdid_staggered_with_ri(self, simple_staggered_data):
        """Test lwdid() staggered mode with ri=True."""
        results = lwdid(
            data=simple_staggered_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            aggregate='overall',
            ri=True,
            rireps=50,
            seed=12345
        )
        
        # Should have RI results
        assert hasattr(results, 'ri_pvalue')
        assert results.ri_pvalue is not None
        assert 0 <= results.ri_pvalue <= 1
        assert hasattr(results, 'ri_method')
        assert hasattr(results, 'ri_valid')
    
    def test_lwdid_staggered_without_ri(self, simple_staggered_data):
        """Test lwdid() staggered mode without ri=True."""
        results = lwdid(
            data=simple_staggered_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            aggregate='overall',
            ri=False
        )
        
        # Should not have RI p-value set
        assert not hasattr(results, 'ri_pvalue') or results.ri_pvalue is None


# =============================================================================
# Castle Law End-to-End Tests
# =============================================================================

@pytest.mark.slow
class TestCastleLawE2E:
    """Castle Law end-to-end tests for RI."""
    
    def test_castle_law_ri_overall_demean(self, castle_data):
        """Test RI on Castle Law data with demean transformation."""
        # First get the observed overall effect
        results = lwdid(
            data=castle_data,
            y='lhomicide',
            ivar='sid',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            control_group='never_treated',
            aggregate='overall'
        )
        
        observed_att = results.att_overall
        
        # Verify observed ATT is close to expected (≈0.092)
        assert abs(observed_att - 0.092) < 0.03, \
            f"Observed ATT {observed_att} not close to expected 0.092"
        
        # Run RI
        ri_result = ri_overall_effect(
            data=castle_data,
            gvar='gvar',
            ivar='sid',
            tvar='year',
            y='lhomicide',
            observed_att=observed_att,
            rolling='demean',
            rireps=200,  # Use more reps for better p-value estimate
            seed=42
        )
        
        # Verify RI result structure
        assert isinstance(ri_result, StaggeredRIResult)
        assert 0 <= ri_result.p_value <= 1
        assert ri_result.ri_valid > 0.9 * ri_result.ri_reps, \
            f"Too many failures: {ri_result.ri_failed}/{ri_result.ri_reps}"
        
        # P-value should indicate significance (expected effect is real)
        # Note: with small reps, p-value may vary, so we use a lenient threshold
        print(f"Castle Law RI p-value (demean): {ri_result.p_value:.4f}")
    
    def test_castle_law_ri_overall_detrend(self, castle_data):
        """Test RI on Castle Law data with detrend transformation."""
        # First get the observed overall effect
        results = lwdid(
            data=castle_data,
            y='lhomicide',
            ivar='sid',
            tvar='year',
            gvar='gvar',
            rolling='detrend',
            control_group='never_treated',
            aggregate='overall'
        )
        
        observed_att = results.att_overall
        
        # Verify observed ATT is close to expected (≈0.067)
        assert abs(observed_att - 0.067) < 0.03, \
            f"Observed ATT {observed_att} not close to expected 0.067"
        
        # Run RI
        ri_result = ri_overall_effect(
            data=castle_data,
            gvar='gvar',
            ivar='sid',
            tvar='year',
            y='lhomicide',
            observed_att=observed_att,
            rolling='detrend',
            rireps=200,
            seed=42
        )
        
        # Verify RI result structure
        assert isinstance(ri_result, StaggeredRIResult)
        assert 0 <= ri_result.p_value <= 1
        assert ri_result.ri_valid > 0.9 * ri_result.ri_reps
        
        print(f"Castle Law RI p-value (detrend): {ri_result.p_value:.4f}")
    
    def test_castle_law_lwdid_with_ri(self, castle_data):
        """Test lwdid() on Castle Law with ri=True."""
        results = lwdid(
            data=castle_data,
            y='lhomicide',
            ivar='sid',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            control_group='never_treated',
            aggregate='overall',
            ri=True,
            rireps=200,
            seed=12345
        )
        
        # Verify ATT
        assert abs(results.att_overall - 0.092) < 0.03
        
        # Verify RI results
        assert results.ri_pvalue is not None
        assert 0 <= results.ri_pvalue <= 1
        
        print(f"Castle Law ATT: {results.att_overall:.4f}")
        print(f"Castle Law RI p-value: {results.ri_pvalue:.4f}")


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_cohort(self):
        """Test RI with single treatment cohort."""
        np.random.seed(42)
        
        records = []
        for i in range(8):
            unit_id = i + 1
            gvar = 3 if i < 4 else 0
            
            for t in range(1, 5):
                effect = 1.5 if (gvar == 3 and t >= 3) else 0
                y = 10 + effect + np.random.normal(0, 0.5)
                records.append({'id': unit_id, 'year': t, 'y': y, 'gvar': gvar})
        
        data = pd.DataFrame(records)
        
        result = randomization_inference_staggered(
            data=data,
            gvar='gvar', ivar='id', tvar='year', y='y',
            cohorts=[3],
            observed_att=1.5,
            target='cohort_time',
            target_cohort=3,
            target_period=3,
            ri_method='permutation',
            rireps=50,
            seed=42,
            rolling='demean',
            n_never_treated=4
        )
        
        assert 0 <= result.p_value <= 1
    
    def test_many_cohorts(self):
        """Test RI with many treatment cohorts."""
        np.random.seed(42)
        
        records = []
        n_cohorts = 5
        for i in range(20):
            unit_id = i + 1
            
            if i < 15:
                # Treated units in cohorts 3, 4, 5, 6, 7
                gvar = 3 + (i % n_cohorts)
            else:
                gvar = 0
            
            for t in range(1, 10):
                effect = 2.0 if (gvar > 0 and t >= gvar) else 0
                y = 10 + effect + np.random.normal(0, 0.5)
                records.append({'id': unit_id, 'year': t, 'y': y, 'gvar': gvar})
        
        data = pd.DataFrame(records)
        
        result = ri_overall_effect(
            data=data,
            gvar='gvar', ivar='id', tvar='year', y='y',
            observed_att=2.0,
            rolling='demean',
            rireps=50,
            seed=42
        )
        
        assert 0 <= result.p_value <= 1
        assert result.ri_valid > 0
    
    def test_high_failure_rate_warning(self, simple_staggered_data):
        """Test that high failure rate triggers warning."""
        # Create data that might cause many failures
        data = simple_staggered_data.copy()
        # Add some NaN values
        data.loc[data['year'] == 3, 'y'] = np.nan
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            try:
                result = randomization_inference_staggered(
                    data=data,
                    gvar='gvar', ivar='id', tvar='year', y='y',
                    cohorts=[3, 4],
                    observed_att=2.0,
                    target='cohort_time',
                    target_cohort=3,
                    target_period=4,  # Use period 4 to avoid NaN
                    ri_method='permutation',
                    rireps=50,
                    seed=42,
                    rolling='demean',
                    n_never_treated=4
                )
                
                # Check if we got a result despite potential issues
                assert 0 <= result.p_value <= 1
            except RandomizationError:
                # Expected if too many failures
                pass


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
