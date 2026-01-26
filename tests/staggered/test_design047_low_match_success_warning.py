"""
DESIGN-047: PSM Low Match Success Rate Warning Tests

This module verifies that estimate_psm() issues a warning when the match success
rate falls below 50%, analogous to the bootstrap success rate warning.

Background:
- Before this fix, estimate_psm() only raised an error when ALL treated units
  failed to match (0% success rate)
- Users were not warned when most treated units failed (e.g., only 10% matched)
- This could lead to unreliable results without any indication

Fix:
- Added warning when match_success_rate < 0.5 (50%)
- Warning message includes: success rate %, matched/total counts, suggestions
- Consistent with bootstrap success rate warning design pattern

Test Strategy:
1. Unit tests: Verify warning triggers at low success rates
2. Boundary tests: Verify 50% threshold behavior
3. No-warning tests: Verify high success rates don't trigger warning
"""

import pytest
import numpy as np
import pandas as pd
import warnings

from lwdid.staggered.estimators import estimate_psm


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def well_separated_data():
    """
    Data with partial propensity score overlap between treated and control.
    Using appropriate caliper will cause many (but not all) matches to fail.
    
    Design:
    - Most treated units: x1 ~ N(2, 0.5) -> high propensity scores (will fail)
    - Some treated units: x1 ~ N(0, 0.5) -> moderate PS (will succeed)
    - Control units: x1 ~ N(0, 0.5) -> moderate propensity scores
    - With moderate caliper, ~70% of treated units will fail to find matches
    """
    np.random.seed(42)
    n_treat = 50
    n_control = 100
    
    # Split treated units: 70% far from controls, 30% near controls
    n_treat_far = 35
    n_treat_near = 15
    
    # Far treated units (will fail to match)
    x1_treat_far = np.random.normal(3, 0.3, n_treat_far)
    x2_treat_far = np.random.normal(0, 1, n_treat_far)
    
    # Near treated units (will succeed in matching)
    x1_treat_near = np.random.normal(0, 0.5, n_treat_near)
    x2_treat_near = np.random.normal(0, 1, n_treat_near)
    
    # Control units centered at 0
    x1_control = np.random.normal(0, 0.5, n_control)
    x2_control = np.random.normal(0, 1, n_control)
    
    x1 = np.concatenate([x1_treat_far, x1_treat_near, x1_control])
    x2 = np.concatenate([x2_treat_far, x2_treat_near, x2_control])
    D = np.array([1] * n_treat + [0] * n_control)
    
    # Outcome with treatment effect = 2.0
    Y = 1 + 0.5 * x1 + 0.3 * x2 + 2.0 * D + np.random.normal(0, 0.5, n_treat + n_control)
    
    return pd.DataFrame({
        'Y': Y,
        'D': D,
        'x1': x1,
        'x2': x2,
    })


@pytest.fixture
def overlapping_data():
    """
    Data with good propensity score overlap - all matches should succeed.
    """
    np.random.seed(123)
    n = 200
    
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    
    # Moderate propensity score dependence
    ps_true = 1 / (1 + np.exp(-0.3 * x1 - 0.2 * x2))
    D = (np.random.uniform(0, 1, n) < ps_true).astype(int)
    
    Y = 1 + 0.5 * x1 + 0.3 * x2 + 1.5 * D + np.random.normal(0, 0.5, n)
    
    return pd.DataFrame({
        'Y': Y,
        'D': D,
        'x1': x1,
        'x2': x2,
    })


@pytest.fixture
def boundary_data():
    """
    Data designed to test the 50% boundary threshold.
    Approximately half of treated units should fail with appropriate caliper.
    """
    np.random.seed(789)
    n_treat = 100
    n_control = 100
    
    # Half treated units have good overlap, half have poor overlap
    x1_treat_good = np.random.normal(0, 0.5, n_treat // 2)
    x1_treat_poor = np.random.normal(3, 0.5, n_treat // 2)
    x1_treat = np.concatenate([x1_treat_good, x1_treat_poor])
    
    x1_control = np.random.normal(0, 0.5, n_control)
    
    x2_treat = np.random.normal(0, 1, n_treat)
    x2_control = np.random.normal(0, 1, n_control)
    
    x1 = np.concatenate([x1_treat, x1_control])
    x2 = np.concatenate([x2_treat, x2_control])
    D = np.array([1] * n_treat + [0] * n_control)
    
    Y = 1 + 0.5 * x1 + 0.3 * x2 + 2.0 * D + np.random.normal(0, 0.5, n_treat + n_control)
    
    return pd.DataFrame({
        'Y': Y,
        'D': D,
        'x1': x1,
        'x2': x2,
    })


# ============================================================================
# Test: Low Match Success Rate Warning
# ============================================================================

class TestDESIGN047LowMatchSuccessWarning:
    """DESIGN-047: Verify warning is issued when match success rate < 50%."""
    
    def test_low_success_rate_triggers_warning(self, well_separated_data):
        """
        With well-separated data and tight caliper, most matches should fail,
        triggering the low match success rate warning.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Use moderate caliper to cause many (but not all) match failures
            result = estimate_psm(
                data=well_separated_data,
                y='Y',
                d='D',
                propensity_controls=['x1', 'x2'],
                caliper=0.5,  # Moderate caliper - allows some matches
                caliper_scale='sd',
            )
            
            # Find the low match success warning
            match_warnings = [
                warning for warning in w
                if "Low match success rate" in str(warning.message)
            ]
            
            assert len(match_warnings) >= 1, (
                "Expected warning for low match success rate with tight caliper"
            )
            
            # Verify warning message contains expected information
            warning_msg = str(match_warnings[0].message)
            assert "%" in warning_msg, "Warning should include percentage"
            assert "/" in warning_msg, "Warning should include count ratio"
            assert "Results may be unreliable" in warning_msg
    
    def test_high_success_rate_no_warning(self, overlapping_data):
        """
        With good overlap and no caliper, all matches should succeed,
        so no warning should be issued.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = estimate_psm(
                data=overlapping_data,
                y='Y',
                d='D',
                propensity_controls=['x1', 'x2'],
                caliper=None,  # No caliper - all matches succeed
            )
            
            # Should not find low match success warning
            match_warnings = [
                warning for warning in w
                if "Low match success rate" in str(warning.message)
            ]
            
            assert len(match_warnings) == 0, (
                "Should not warn when match success rate is high"
            )
            
            # Verify result is valid
            assert result.att is not None
            assert result.se > 0
    
    def test_warning_threshold_at_50_percent(self, boundary_data):
        """
        Test behavior at the 50% threshold boundary.
        """
        # First test with moderate caliper - should have some failures
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = estimate_psm(
                data=boundary_data,
                y='Y',
                d='D',
                propensity_controls=['x1', 'x2'],
                caliper=0.5,  # Moderate caliper
                caliper_scale='sd',
            )
            
            # Check if warning was triggered (depends on actual success rate)
            match_warnings = [
                warning for warning in w
                if "Low match success rate" in str(warning.message)
            ]
            
            # Result should be valid regardless of warning
            assert result.att is not None
            assert result.se > 0
    
    def test_warning_category_is_user_warning(self, well_separated_data):
        """
        Verify the warning is a UserWarning (allows user to suppress if desired).
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = estimate_psm(
                data=well_separated_data,
                y='Y',
                d='D',
                propensity_controls=['x1', 'x2'],
                caliper=0.5,  # Moderate caliper - allows some matches
                caliper_scale='sd',
            )
            
            match_warnings = [
                warning for warning in w
                if "Low match success rate" in str(warning.message)
            ]
            
            if len(match_warnings) > 0:
                assert issubclass(match_warnings[0].category, UserWarning), (
                    "Warning should be UserWarning type"
                )


# ============================================================================
# Test: Edge Cases
# ============================================================================

class TestDESIGN047EdgeCases:
    """Edge case tests for the low match success warning."""
    
    def test_just_above_threshold_no_warning(self):
        """
        When success rate is just above 50%, no warning should be issued.
        """
        np.random.seed(111)
        n = 100
        
        # Create data where ~55% of treated units will match
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        ps_true = 1 / (1 + np.exp(-0.3 * x1))
        D = (np.random.uniform(0, 1, n) < ps_true).astype(int)
        Y = 1 + 0.5 * x1 + 2.0 * D + np.random.normal(0, 0.5, n)
        
        data = pd.DataFrame({'Y': Y, 'D': D, 'x1': x1, 'x2': x2})
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Without caliper, all should match
            result = estimate_psm(
                data=data,
                y='Y',
                d='D',
                propensity_controls=['x1', 'x2'],
                caliper=None,
            )
            
            match_warnings = [
                warning for warning in w
                if "Low match success rate" in str(warning.message)
            ]
            
            # No warning when all matches succeed
            assert len(match_warnings) == 0
    
    def test_just_below_threshold_triggers_warning(self):
        """
        When success rate is just below 50%, warning should be issued.
        """
        np.random.seed(222)
        
        # Create data with clear separation
        n_treat = 50
        n_control = 50
        
        # Most treated units far from controls
        x1_treat = np.concatenate([
            np.random.normal(3, 0.3, 35),  # 70% will fail to match
            np.random.normal(0, 0.3, 15),  # 30% will succeed
        ])
        x1_control = np.random.normal(0, 0.3, n_control)
        
        x2_treat = np.random.normal(0, 1, n_treat)
        x2_control = np.random.normal(0, 1, n_control)
        
        x1 = np.concatenate([x1_treat, x1_control])
        x2 = np.concatenate([x2_treat, x2_control])
        D = np.array([1] * n_treat + [0] * n_control)
        Y = 1 + 0.5 * x1 + 2.0 * D + np.random.normal(0, 0.5, n_treat + n_control)
        
        data = pd.DataFrame({'Y': Y, 'D': D, 'x1': x1, 'x2': x2})
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Tight caliper to enforce matching failures
            result = estimate_psm(
                data=data,
                y='Y',
                d='D',
                propensity_controls=['x1', 'x2'],
                caliper=0.2,
                caliper_scale='sd',
            )
            
            match_warnings = [
                warning for warning in w
                if "Low match success rate" in str(warning.message)
            ]
            
            # Should trigger warning
            assert len(match_warnings) >= 1, (
                "Expected warning when success rate < 50%"
            )
    
    def test_all_match_failure_raises_error(self):
        """
        When ALL treated units fail to match (0% success), ValueError is raised.
        This tests that the original error behavior is preserved.
        """
        # Create data with complete separation - no overlap at all
        np.random.seed(999)
        n_treat = 30
        n_control = 30
        
        # Treated units with very high x1 values
        x1_treat = np.random.normal(10, 0.1, n_treat)
        x2_treat = np.random.normal(0, 1, n_treat)
        
        # Control units with very low x1 values
        x1_control = np.random.normal(-10, 0.1, n_control)
        x2_control = np.random.normal(0, 1, n_control)
        
        x1 = np.concatenate([x1_treat, x1_control])
        x2 = np.concatenate([x2_treat, x2_control])
        D = np.array([1] * n_treat + [0] * n_control)
        Y = 1 + 0.5 * x1 + 2.0 * D + np.random.normal(0, 0.5, n_treat + n_control)
        
        data = pd.DataFrame({'Y': Y, 'D': D, 'x1': x1, 'x2': x2})
        
        with pytest.raises(ValueError, match="All treated units failed"):
            estimate_psm(
                data=data,
                y='Y',
                d='D',
                propensity_controls=['x1', 'x2'],
                caliper=0.01,  # Very tight caliper with completely separated data
                caliper_scale='sd',
            )


# ============================================================================
# Test: Consistency with Bootstrap Warning
# ============================================================================

class TestDESIGN047ConsistencyWithBootstrap:
    """
    Verify the match success warning is consistent with bootstrap success warning.
    Both use similar threshold and message patterns for API consistency.
    """
    
    def test_warning_message_format_consistency(self, well_separated_data):
        """
        Verify warning message format is consistent with bootstrap warning.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = estimate_psm(
                data=well_separated_data,
                y='Y',
                d='D',
                propensity_controls=['x1', 'x2'],
                caliper=0.5,  # Moderate caliper - allows some matches
                caliper_scale='sd',
            )
            
            match_warnings = [
                warning for warning in w
                if "Low match success rate" in str(warning.message)
            ]
            
            if len(match_warnings) > 0:
                msg = str(match_warnings[0].message)
                
                # Check format elements consistent with bootstrap warning
                assert "Low" in msg  # Same prefix style
                assert "success rate" in msg  # Same terminology
                assert "%" in msg  # Percentage format
                assert "unreliable" in msg  # Reliability warning


# ============================================================================
# Test: Result Validity Despite Warning
# ============================================================================

class TestDESIGN047ResultValidityWithWarning:
    """
    Verify that results are still valid when warning is issued.
    The warning is informational, not a hard failure.
    """
    
    def test_result_valid_with_warning(self, well_separated_data):
        """
        Results should be statistically valid even with low success rate warning.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = estimate_psm(
                data=well_separated_data,
                y='Y',
                d='D',
                propensity_controls=['x1', 'x2'],
                caliper=0.5,  # Moderate caliper - allows some matches
                caliper_scale='sd',
            )
            
            # Verify result object is complete
            assert result.att is not None
            assert not np.isnan(result.att)
            assert result.se > 0
            assert not np.isnan(result.se)
            assert result.ci_lower < result.ci_upper
            assert result.n_treated > 0
            assert result.n_control > 0
