"""
Tests for BUG-049: Common Timing RI Parameter Validation.

This module verifies that randomization inference (RI) parameters are properly
validated in common timing mode, matching the validation behavior of staggered mode.

Tests cover:
- Invalid rireps values (0, negative, non-integer)
- Invalid ri_method values (unknown methods)
- Edge cases (boundary values, type errors)
"""

import numpy as np
import pandas as pd
import pytest

from lwdid import lwdid


class TestCommonTimingRIParameterValidation:
    """Test RI parameter validation in common timing mode.
    
    BUG-049: Common timing mode was missing RI parameter validation that
    existed in staggered mode. Users could pass rireps=0 or invalid ri_method
    without early error detection.
    """
    
    @pytest.fixture
    def common_timing_data(self):
        """Generate minimal common timing panel data for validation tests."""
        np.random.seed(42)
        n_units = 20
        n_periods = 6
        treatment_period = 4
        
        data = []
        for i in range(n_units):
            treated = i < n_units // 2  # Half treated, half control
            for t in range(1, n_periods + 1):
                post = 1 if t >= treatment_period else 0
                y = 10 + 0.5 * t + (2.0 if treated and post else 0) + np.random.normal(0, 1)
                data.append({
                    'id': i + 1,
                    'year': 2000 + t,
                    'y': y,
                    'd': 1 if treated else 0,
                    'post': post
                })
        
        return pd.DataFrame(data)
    
    # =========================================================================
    # rireps Validation Tests
    # =========================================================================
    
    def test_rireps_zero_raises_error(self, common_timing_data):
        """rireps=0 should raise ValueError with informative message."""
        with pytest.raises(ValueError, match=r"Invalid rireps=0"):
            lwdid(
                common_timing_data,
                y='y', d='d', ivar='id', tvar='year', post='post',
                rolling='demean',
                ri=True,
                rireps=0
            )
    
    def test_rireps_negative_raises_error(self, common_timing_data):
        """Negative rireps should raise ValueError."""
        with pytest.raises(ValueError, match=r"Invalid rireps=-1"):
            lwdid(
                common_timing_data,
                y='y', d='d', ivar='id', tvar='year', post='post',
                rolling='demean',
                ri=True,
                rireps=-1
            )
    
    def test_rireps_float_raises_error(self, common_timing_data):
        """Float rireps should raise ValueError (must be integer)."""
        with pytest.raises(ValueError, match=r"rireps must be a positive integer"):
            lwdid(
                common_timing_data,
                y='y', d='d', ivar='id', tvar='year', post='post',
                rolling='demean',
                ri=True,
                rireps=100.5
            )
    
    def test_rireps_minimum_valid(self, common_timing_data):
        """rireps=100 is the minimum valid value for reliable inference.
        
        Note: The randomization module requires at least 100 valid replications
        for statistical reliability. rireps=1 passes parameter validation but
        fails at runtime due to insufficient replications requirement.
        """
        result = lwdid(
            common_timing_data,
            y='y', d='d', ivar='id', tvar='year', post='post',
            rolling='demean',
            ri=True,
            rireps=100,
            seed=42
        )
        assert hasattr(result, 'ri_pvalue')
        assert result.rireps == 100
    
    # =========================================================================
    # ri_method Validation Tests
    # =========================================================================
    
    def test_ri_method_invalid_raises_error(self, common_timing_data):
        """Invalid ri_method should raise ValueError with valid options."""
        with pytest.raises(ValueError, match=r"Invalid ri_method='invalid'"):
            lwdid(
                common_timing_data,
                y='y', d='d', ivar='id', tvar='year', post='post',
                rolling='demean',
                ri=True,
                rireps=100,
                ri_method='invalid'
            )
    
    def test_ri_method_typo_raises_error(self, common_timing_data):
        """Common typos in ri_method should raise informative error."""
        with pytest.raises(ValueError, match=r"Invalid ri_method"):
            lwdid(
                common_timing_data,
                y='y', d='d', ivar='id', tvar='year', post='post',
                rolling='demean',
                ri=True,
                rireps=100,
                ri_method='boostrap'  # Typo: missing 't'
            )
    
    def test_ri_method_bootstrap_valid(self, common_timing_data):
        """ri_method='bootstrap' should work correctly."""
        result = lwdid(
            common_timing_data,
            y='y', d='d', ivar='id', tvar='year', post='post',
            rolling='demean',
            ri=True,
            rireps=100,
            ri_method='bootstrap',
            seed=42
        )
        assert result.ri_method == 'bootstrap'
    
    def test_ri_method_permutation_valid(self, common_timing_data):
        """ri_method='permutation' should work correctly."""
        result = lwdid(
            common_timing_data,
            y='y', d='d', ivar='id', tvar='year', post='post',
            rolling='demean',
            ri=True,
            rireps=50,
            ri_method='permutation',
            seed=42
        )
        assert result.ri_method == 'permutation'
    
    def test_ri_method_case_insensitive(self, common_timing_data):
        """ri_method should be case-insensitive."""
        result = lwdid(
            common_timing_data,
            y='y', d='d', ivar='id', tvar='year', post='post',
            rolling='demean',
            ri=True,
            rireps=100,
            ri_method='BOOTSTRAP',  # Uppercase
            seed=42
        )
        assert result.ri_method == 'bootstrap'  # Should be normalized to lowercase
    
    # =========================================================================
    # ri=False Should Not Validate RI Parameters
    # =========================================================================
    
    def test_ri_false_ignores_invalid_rireps(self, common_timing_data):
        """When ri=False, invalid rireps should not cause error."""
        # This should NOT raise because ri=False means RI is disabled
        result = lwdid(
            common_timing_data,
            y='y', d='d', ivar='id', tvar='year', post='post',
            rolling='demean',
            ri=False,
            rireps=0  # Would be invalid if ri=True
        )
        assert not hasattr(result, 'ri_pvalue') or result.ri_pvalue is None or np.isnan(result.ri_pvalue) if hasattr(result, 'ri_pvalue') else True
    
    def test_ri_false_ignores_invalid_ri_method(self, common_timing_data):
        """When ri=False, invalid ri_method should not cause error."""
        # This should NOT raise because ri=False means RI is disabled
        result = lwdid(
            common_timing_data,
            y='y', d='d', ivar='id', tvar='year', post='post',
            rolling='demean',
            ri=False,
            ri_method='invalid'  # Would be invalid if ri=True
        )
        assert result.att is not None
    
    # =========================================================================
    # Error Message Quality Tests
    # =========================================================================
    
    def test_rireps_error_message_contains_recommendation(self, common_timing_data):
        """Error message for invalid rireps should include recommendation."""
        with pytest.raises(ValueError) as exc_info:
            lwdid(
                common_timing_data,
                y='y', d='d', ivar='id', tvar='year', post='post',
                rolling='demean',
                ri=True,
                rireps=0
            )
        error_msg = str(exc_info.value)
        assert "rireps >= 500" in error_msg.lower() or "500" in error_msg
    
    def test_ri_method_error_message_lists_valid_options(self, common_timing_data):
        """Error message for invalid ri_method should list valid options."""
        with pytest.raises(ValueError) as exc_info:
            lwdid(
                common_timing_data,
                y='y', d='d', ivar='id', tvar='year', post='post',
                rolling='demean',
                ri=True,
                rireps=100,
                ri_method='invalid'
            )
        error_msg = str(exc_info.value)
        assert 'bootstrap' in error_msg.lower()
        assert 'permutation' in error_msg.lower()


class TestRIValidationConsistencyWithStaggered:
    """Test that common timing RI validation matches staggered mode behavior.
    
    This ensures parity between the two modes as required by BUG-049.
    """
    
    @pytest.fixture
    def common_timing_data(self):
        """Generate common timing data."""
        np.random.seed(42)
        n_units = 20
        data = []
        for i in range(n_units):
            treated = i < 10
            for t in range(1, 7):
                post = 1 if t >= 4 else 0
                y = 10 + 0.5 * t + (2.0 if treated and post else 0) + np.random.normal(0, 1)
                data.append({
                    'id': i + 1,
                    'year': 2000 + t,
                    'y': y,
                    'd': 1 if treated else 0,
                    'post': post
                })
        return pd.DataFrame(data)
    
    @pytest.fixture
    def staggered_data(self):
        """Generate equivalent staggered data."""
        np.random.seed(42)
        n_units = 20
        data = []
        for i in range(n_units):
            # First 10 units treated at period 4, rest never treated
            gvar = 2004 if i < 10 else 0
            for t in range(1, 7):
                year = 2000 + t
                post = 1 if gvar > 0 and year >= gvar else 0
                y = 10 + 0.5 * t + (2.0 if post else 0) + np.random.normal(0, 1)
                data.append({
                    'id': i + 1,
                    'year': year,
                    'y': y,
                    'gvar': gvar
                })
        return pd.DataFrame(data)
    
    def test_same_rireps_error_format_both_modes(self, common_timing_data, staggered_data):
        """Both modes should raise similar errors for rireps=0."""
        # Common timing mode
        with pytest.raises(ValueError) as ct_exc:
            lwdid(
                common_timing_data,
                y='y', d='d', ivar='id', tvar='year', post='post',
                rolling='demean', ri=True, rireps=0
            )
        
        # Staggered mode
        with pytest.raises(ValueError) as sg_exc:
            lwdid(
                staggered_data,
                y='y', gvar='gvar', ivar='id', tvar='year',
                rolling='demean', ri=True, rireps=0
            )
        
        # Both should mention "rireps"
        assert "rireps" in str(ct_exc.value).lower()
        assert "rireps" in str(sg_exc.value).lower()
    
    def test_same_ri_method_error_format_both_modes(self, common_timing_data, staggered_data):
        """Both modes should raise similar errors for invalid ri_method."""
        # Common timing mode
        with pytest.raises(ValueError) as ct_exc:
            lwdid(
                common_timing_data,
                y='y', d='d', ivar='id', tvar='year', post='post',
                rolling='demean', ri=True, rireps=100, ri_method='invalid'
            )
        
        # Staggered mode
        with pytest.raises(ValueError) as sg_exc:
            lwdid(
                staggered_data,
                y='y', gvar='gvar', ivar='id', tvar='year',
                rolling='demean', ri=True, rireps=100, ri_method='invalid'
            )
        
        # Both should mention "ri_method"
        assert "ri_method" in str(ct_exc.value).lower()
        assert "ri_method" in str(sg_exc.value).lower()


class TestRIResultsIntegrity:
    """Test that RI results are correctly populated after validation passes."""
    
    @pytest.fixture
    def common_timing_data(self):
        """Generate common timing data for result validation."""
        np.random.seed(42)
        n_units = 30
        data = []
        for i in range(n_units):
            treated = i < 15
            for t in range(1, 8):
                post = 1 if t >= 5 else 0
                y = 10 + 0.5 * t + (2.5 if treated and post else 0) + np.random.normal(0, 1)
                data.append({
                    'id': i + 1,
                    'year': 2000 + t,
                    'y': y,
                    'd': 1 if treated else 0,
                    'post': post
                })
        return pd.DataFrame(data)
    
    def test_ri_results_populated_on_success(self, common_timing_data):
        """Successful RI should populate all expected result attributes."""
        result = lwdid(
            common_timing_data,
            y='y', d='d', ivar='id', tvar='year', post='post',
            rolling='demean',
            ri=True,
            rireps=100,
            ri_method='bootstrap',
            seed=42
        )
        
        # All RI attributes should be present
        assert hasattr(result, 'ri_pvalue')
        assert hasattr(result, 'rireps')
        assert hasattr(result, 'ri_seed')
        assert hasattr(result, 'ri_method')
        
        # Values should be valid
        assert 0 <= result.ri_pvalue <= 1
        assert result.rireps == 100
        assert result.ri_seed == 42
        assert result.ri_method == 'bootstrap'
    
    def test_ri_pvalue_in_valid_range(self, common_timing_data):
        """RI p-value should always be in [0, 1]."""
        result = lwdid(
            common_timing_data,
            y='y', d='d', ivar='id', tvar='year', post='post',
            rolling='demean',
            ri=True,
            rireps=200,
            seed=123
        )
        
        assert 0 <= result.ri_pvalue <= 1, \
            f"RI p-value {result.ri_pvalue} outside valid range [0, 1]"
    
    def test_ri_seed_reproducibility(self, common_timing_data):
        """Same seed should produce identical RI p-values."""
        seed = 54321
        
        result1 = lwdid(
            common_timing_data,
            y='y', d='d', ivar='id', tvar='year', post='post',
            rolling='demean',
            ri=True, rireps=100, seed=seed
        )
        
        result2 = lwdid(
            common_timing_data,
            y='y', d='d', ivar='id', tvar='year', post='post',
            rolling='demean',
            ri=True, rireps=100, seed=seed
        )
        
        assert result1.ri_pvalue == result2.ri_pvalue, \
            "Same seed should produce reproducible RI p-values"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
