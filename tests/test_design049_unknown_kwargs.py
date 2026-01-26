"""Tests for DESIGN-049: Unknown kwargs warning mechanism.

This module tests that the lwdid() function properly warns users when unknown
keyword arguments are passed, helping catch parameter typos early.

Test Coverage:
1. Single unknown kwarg triggers warning
2. Multiple unknown kwargs triggers warning with all names
3. Known kwarg (riseed) does not trigger warning
4. Mixed known/unknown kwargs triggers warning only for unknown
5. Empty kwargs does not trigger warning
6. Warning does not affect estimation results
"""

import warnings
import numpy as np
import pandas as pd
import pytest

from lwdid import lwdid


def _create_simple_panel_data(n_units: int = 10, n_periods: int = 5) -> pd.DataFrame:
    """Create simple panel data for testing."""
    np.random.seed(42)
    
    data = []
    treatment_period = 3
    
    for i in range(1, n_units + 1):
        is_treated = i <= n_units // 2
        for t in range(1, n_periods + 1):
            post = 1 if t >= treatment_period else 0
            treatment_effect = 2.0 if (is_treated and post) else 0.0
            y = 1.0 + 0.5 * t + treatment_effect + np.random.normal(0, 0.5)
            
            data.append({
                'id': i,
                'year': 2000 + t,
                'd': 1 if is_treated else 0,
                'post': post,
                'y': y,
            })
    
    return pd.DataFrame(data)


class TestUnknownKwargsWarning:
    """Test that unknown kwargs trigger appropriate warnings."""
    
    @pytest.fixture
    def panel_data(self):
        """Create simple panel data for testing."""
        return _create_simple_panel_data()
    
    def test_single_unknown_kwarg_triggers_warning(self, panel_data):
        """A single unknown kwarg should trigger a warning."""
        with pytest.warns(UserWarning, match=r"Unknown keyword argument\(s\) ignored"):
            lwdid(
                panel_data, y='y', d='d', ivar='id', tvar='year',
                post='post', rolling='demean',
                seeed=123  # Typo: should be 'seed'
            )
    
    def test_multiple_unknown_kwargs_trigger_warning(self, panel_data):
        """Multiple unknown kwargs should all be listed in warning."""
        with pytest.warns(UserWarning) as record:
            lwdid(
                panel_data, y='y', d='d', ivar='id', tvar='year',
                post='post', rolling='demean',
                seeed=123,
                controlls=['x'],  # Typo
                vcee='robust',    # Typo
            )
        
        # Find the unknown kwargs warning
        unknown_warnings = [w for w in record if "Unknown keyword argument" in str(w.message)]
        assert len(unknown_warnings) >= 1
        
        warning_msg = str(unknown_warnings[0].message)
        assert 'seeed' in warning_msg
        assert 'controlls' in warning_msg
        assert 'vcee' in warning_msg
    
    def test_known_kwarg_riseed_no_warning(self, panel_data):
        """The known kwarg 'riseed' should not trigger warning."""
        # Capture all warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            lwdid(
                panel_data, y='y', d='d', ivar='id', tvar='year',
                post='post', rolling='demean',
                riseed=42  # Known kwarg for backward compatibility
            )
        
        # Check that no "Unknown keyword argument" warning was raised
        unknown_warnings = [warning for warning in w 
                          if "Unknown keyword argument" in str(warning.message)]
        assert len(unknown_warnings) == 0, \
            f"riseed should not trigger unknown kwarg warning, but got: {unknown_warnings}"
    
    def test_mixed_known_unknown_kwargs(self, panel_data):
        """Mixed known/unknown kwargs should only warn about unknown ones in 'ignored' list."""
        with pytest.warns(UserWarning) as record:
            lwdid(
                panel_data, y='y', d='d', ivar='id', tvar='year',
                post='post', rolling='demean',
                riseed=42,    # Known
                seeed=123,    # Unknown (typo)
            )
        
        unknown_warnings = [w for w in record if "Unknown keyword argument" in str(w.message)]
        assert len(unknown_warnings) >= 1
        
        warning_msg = str(unknown_warnings[0].message)
        # seeed should be in the ignored list
        assert 'seeed' in warning_msg
        # riseed should be in "Valid extra arguments" but NOT in "ignored" list
        assert "['seeed']" in warning_msg  # Only seeed in ignored list
    
    def test_empty_kwargs_no_warning(self, panel_data):
        """No kwargs should not trigger any unknown kwarg warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            lwdid(
                panel_data, y='y', d='d', ivar='id', tvar='year',
                post='post', rolling='demean'
            )
        
        unknown_warnings = [warning for warning in w 
                          if "Unknown keyword argument" in str(warning.message)]
        assert len(unknown_warnings) == 0
    
    def test_warning_does_not_affect_results(self, panel_data):
        """Unknown kwargs warning should not affect estimation results."""
        # Run without unknown kwargs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_normal = lwdid(
                panel_data, y='y', d='d', ivar='id', tvar='year',
                post='post', rolling='demean'
            )
        
        # Run with unknown kwargs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_with_unknown = lwdid(
                panel_data, y='y', d='d', ivar='id', tvar='year',
                post='post', rolling='demean',
                unknown_param=123
            )
        
        # Results should be identical
        assert result_normal.att == result_with_unknown.att
        assert result_normal.se_att == result_with_unknown.se_att
        assert result_normal.pvalue == result_with_unknown.pvalue


class TestWarningMessageContent:
    """Test the content and format of the warning message."""
    
    @pytest.fixture
    def panel_data(self):
        return _create_simple_panel_data()
    
    def test_warning_lists_valid_kwargs(self, panel_data):
        """Warning message should list valid extra arguments."""
        with pytest.warns(UserWarning) as record:
            lwdid(
                panel_data, y='y', d='d', ivar='id', tvar='year',
                post='post', rolling='demean',
                typo_param=123
            )
        
        unknown_warnings = [w for w in record if "Unknown keyword argument" in str(w.message)]
        warning_msg = str(unknown_warnings[0].message)
        
        # Should mention valid kwargs
        assert 'riseed' in warning_msg
        assert 'Valid extra arguments' in warning_msg
    
    def test_warning_shows_sorted_unknown_kwargs(self, panel_data):
        """Unknown kwargs should be sorted alphabetically in the warning."""
        with pytest.warns(UserWarning) as record:
            lwdid(
                panel_data, y='y', d='d', ivar='id', tvar='year',
                post='post', rolling='demean',
                zebra=1,
                apple=2,
                mango=3,
            )
        
        unknown_warnings = [w for w in record if "Unknown keyword argument" in str(w.message)]
        warning_msg = str(unknown_warnings[0].message)
        
        # Check alphabetical order
        apple_pos = warning_msg.find('apple')
        mango_pos = warning_msg.find('mango')
        zebra_pos = warning_msg.find('zebra')
        
        assert apple_pos < mango_pos < zebra_pos, \
            f"Unknown kwargs should be sorted: {warning_msg}"


class TestKwargsWithStaggeredMode:
    """Test kwargs handling in staggered adoption mode."""
    
    @pytest.fixture
    def staggered_data(self):
        """Create staggered adoption panel data."""
        np.random.seed(42)
        
        data = []
        # Cohorts must be > T_min to have pre-treatment periods
        # Time range: 2001-2006, so cohorts should be >= 2003
        cohorts = {1: 2004, 2: 2005, 3: 0, 4: 0, 5: 0}  # Units 3,4,5 never treated
        
        for unit_id, gvar in cohorts.items():
            for t in range(1, 7):
                year = 2000 + t
                post = 1 if gvar > 0 and year >= gvar else 0
                y = 1.0 + 0.5 * t + (2.0 if post else 0) + np.random.normal(0, 0.3)
                
                data.append({
                    'id': unit_id,
                    'year': year,
                    'gvar': gvar,
                    'y': y,
                })
        
        return pd.DataFrame(data)
    
    def test_unknown_kwargs_warning_in_staggered_mode(self, staggered_data):
        """Unknown kwargs should trigger warning in staggered mode too."""
        with pytest.warns(UserWarning, match=r"Unknown keyword argument\(s\) ignored"):
            lwdid(
                staggered_data, y='y', ivar='id', tvar='year',
                gvar='gvar', rolling='demean',
                typo_param=123
            )
    
    def test_riseed_works_in_staggered_mode(self, staggered_data):
        """riseed should work without warning in staggered mode."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            lwdid(
                staggered_data, y='y', ivar='id', tvar='year',
                gvar='gvar', rolling='demean',
                riseed=42
            )
        
        unknown_warnings = [warning for warning in w 
                          if "Unknown keyword argument" in str(warning.message)]
        assert len(unknown_warnings) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
