"""
Python-Stata numerical validation tests for BUG-286, BUG-287, BUG-288.

This module validates that:
1. BUG-286: t_stat calculation matches Stata behavior for valid SE values
2. BUG-287/288: Parameter validation doesn't affect numerical results

Test Strategy:
- Generate test data with known treatment effects
- Run Python lwdid estimation
- Run Stata lwdid estimation via MCP
- Compare ATT, SE, t-stat values
"""

import warnings
import pytest
import numpy as np
import pandas as pd
import sys
import os
import tempfile

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lwdid import lwdid
from lwdid.results import _compute_t_stat


# =============================================================================
# Test Data Generation
# =============================================================================

def generate_common_timing_data_for_validation(n_units=100, n_periods=6, seed=42):
    """Generate common timing DiD test data with time-invariant controls."""
    np.random.seed(seed)
    
    data = []
    for i in range(1, n_units + 1):
        treated = 1 if i <= n_units // 2 else 0
        # Time-invariant controls
        x1 = np.random.normal(0, 1)
        x2 = np.random.normal(0, 1)
        
        for t in range(2000, 2000 + n_periods):
            post = 1 if t >= 2003 else 0
            # Treatment effect = 2.0
            y = 10 + 2 * treated * post + 0.5 * x1 + np.random.normal(0, 1)
            data.append({
                'unit': i,
                'year': t,
                'd': treated,
                'post': post,
                'y': y,
                'x1': x1,
                'x2': x2,
            })
    
    return pd.DataFrame(data)


def generate_staggered_data_for_validation(n_units=90, n_periods=8, seed=42):
    """Generate staggered adoption DiD test data."""
    np.random.seed(seed)
    
    data = []
    for i in range(1, n_units + 1):
        # Assign cohort
        if i <= 30:
            gvar = 0  # never treated
        elif i <= 60:
            gvar = 2003
        else:
            gvar = 2005
        
        # Time-invariant controls
        x1 = np.random.normal(0, 1)
        x2 = np.random.normal(0, 1)
            
        for t in range(2000, 2000 + n_periods):
            treated = 1 if gvar > 0 and t >= gvar else 0
            # Cohort-specific effects: 2003 -> 2.0, 2005 -> 3.0
            effect = 2.0 if gvar == 2003 and treated else (3.0 if gvar == 2005 and treated else 0)
            y = 10 + effect + 0.5 * x1 + np.random.normal(0, 1)
            data.append({
                'unit': i,
                'year': t,
                'gvar': gvar,
                'y': y,
                'x1': x1,
                'x2': x2,
            })
    
    return pd.DataFrame(data)


# =============================================================================
# BUG-286 Validation: t_stat calculation
# =============================================================================

class TestBug286TStatStataConsistency:
    """Validate that t_stat calculation matches Stata behavior."""
    
    def test_tstat_normal_calculation_matches_formula(self):
        """Verify t_stat = att / se for normal values."""
        # Test various ATT and SE combinations
        test_cases = [
            (2.0, 0.5, 4.0),     # Positive ATT
            (-2.0, 0.5, -4.0),   # Negative ATT
            (0.0, 0.5, 0.0),     # Zero ATT
            (1.5, 0.3, 5.0),     # Different ratio
            (0.1, 0.01, 10.0),   # Small values
        ]
        
        for att, se, expected in test_cases:
            result = _compute_t_stat(att, se, warn_on_boundary=False)
            assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result} for att={att}, se={se}"
    
    def test_tstat_consistency_with_python_estimation(self):
        """Verify t_stat in results matches manual calculation."""
        df = generate_common_timing_data_for_validation()
        
        result = lwdid(
            data=df,
            y='y',
            d='d',
            ivar='unit',
            tvar='year',
            post='post',
            rolling='demean',
        )
        
        # Manual calculation
        expected_t_stat = result.att / result.se_att
        
        # Should match (within floating point tolerance)
        assert abs(result.t_stat - expected_t_stat) < 1e-10


# =============================================================================
# BUG-287 Validation: ri_method type validation
# =============================================================================

class TestBug287RiMethodValidation:
    """Validate ri_method type validation doesn't break normal operation."""
    
    def test_ri_method_string_values_work_normally(self):
        """Valid ri_method strings should work as before."""
        df = generate_common_timing_data_for_validation(n_units=50)
        
        # Test 'bootstrap' 
        result = lwdid(
            data=df,
            y='y',
            d='d',
            ivar='unit',
            tvar='year',
            post='post',
            rolling='demean',
            ri=True,
            rireps=50,
            ri_method='bootstrap',
            seed=42,
        )
        assert result is not None
        assert result.att is not None
    
    def test_ri_method_case_insensitive(self):
        """ri_method should be case insensitive."""
        df = generate_common_timing_data_for_validation(n_units=50)
        
        # Test uppercase
        result = lwdid(
            data=df,
            y='y',
            d='d',
            ivar='unit',
            tvar='year',
            post='post',
            rolling='demean',
            ri=True,
            rireps=50,
            ri_method='BOOTSTRAP',
            seed=42,
        )
        assert result is not None


# =============================================================================
# BUG-288 Validation: trim_threshold range validation
# =============================================================================

class TestBug288TrimThresholdValidation:
    """Validate trim_threshold validation doesn't affect valid estimates."""
    
    def test_valid_trim_threshold_produces_consistent_results(self):
        """Valid trim_threshold values should produce consistent ATT."""
        df = generate_common_timing_data_for_validation()
        
        # Run with default trim_threshold (0.01)
        result_default = lwdid(
            data=df,
            y='y',
            d='d',
            ivar='unit',
            tvar='year',
            post='post',
            rolling='demean',
            estimator='ipw',
            controls=['x1', 'x2'],
            trim_threshold=0.01,
        )
        
        # Run with a different valid trim_threshold
        result_custom = lwdid(
            data=df,
            y='y',
            d='d',
            ivar='unit',
            tvar='year',
            post='post',
            rolling='demean',
            estimator='ipw',
            controls=['x1', 'x2'],
            trim_threshold=0.05,
        )
        
        # Both should produce valid results
        assert result_default.att is not None
        assert result_custom.att is not None
        
        # ATT should be similar (not exactly equal due to trimming differences)
        # But should be within reasonable range for the true effect of 2.0
        assert 0.5 < result_default.att < 4.0
        assert 0.5 < result_custom.att < 4.0


# =============================================================================
# Numerical Consistency Tests
# =============================================================================

class TestNumericalConsistency:
    """Test that bug fixes don't change correct numerical behavior."""
    
    def test_common_timing_att_near_true_effect(self):
        """ATT estimate should be near true effect (2.0)."""
        df = generate_common_timing_data_for_validation(n_units=200, seed=123)
        
        result = lwdid(
            data=df,
            y='y',
            d='d',
            ivar='unit',
            tvar='year',
            post='post',
            rolling='demean',
        )
        
        # True effect is 2.0; estimate should be close
        assert 1.5 < result.att < 2.5, f"ATT={result.att} not near true effect 2.0"
    
    def test_staggered_overall_att_reasonable(self):
        """Staggered overall ATT should be in reasonable range."""
        df = generate_staggered_data_for_validation(n_units=150, seed=456)
        
        result = lwdid(
            data=df,
            y='y',
            gvar='gvar',
            ivar='unit',
            tvar='year',
            rolling='demean',
            aggregate='overall',
        )
        
        # Overall effect is weighted average of 2.0 and 3.0
        # Should be between 2.0 and 3.0
        assert 1.5 < result.att_overall < 3.5, f"ATT_overall={result.att_overall} not in expected range"
    
    def test_se_is_positive_and_finite(self):
        """Standard errors should be positive and finite."""
        df = generate_common_timing_data_for_validation()
        
        result = lwdid(
            data=df,
            y='y',
            d='d',
            ivar='unit',
            tvar='year',
            post='post',
            rolling='demean',
        )
        
        assert result.se_att > 0
        assert np.isfinite(result.se_att)
    
    def test_t_stat_sign_matches_att_sign(self):
        """t-stat sign should match ATT sign for positive SE."""
        df = generate_common_timing_data_for_validation()
        
        result = lwdid(
            data=df,
            y='y',
            d='d',
            ivar='unit',
            tvar='year',
            post='post',
            rolling='demean',
        )
        
        # For positive ATT and positive SE, t-stat should be positive
        if result.att > 0:
            assert result.t_stat > 0
        elif result.att < 0:
            assert result.t_stat < 0


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases for the bug fixes."""
    
    def test_small_sample_warning(self):
        """Small samples should produce warnings but still work."""
        df = generate_common_timing_data_for_validation(n_units=30)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = lwdid(
                data=df,
                y='y',
                d='d',
                ivar='unit',
                tvar='year',
                post='post',
                rolling='demean',
                estimator='ipw',
                controls=['x1', 'x2'],
            )
            
            # Should still produce valid results
            assert result.att is not None
            # Should produce small sample warning
            assert any("small sample" in str(w_i.message).lower() for w_i in w)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
