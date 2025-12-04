"""
Unit tests for lwdid() staggered mode API extension.

Tests Story E2-S1 acceptance criteria.
"""

import warnings
import pytest
import numpy as np
import pandas as pd

# Import will be done in fixtures to handle potential import errors gracefully


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def staggered_test_data():
    """Create simple staggered test data with 3 cohorts and 2 NT units."""
    np.random.seed(42)
    
    # 6 units: cohort 2001 (2 units), cohort 2002 (2 units), NT (2 units)
    data = []
    for unit in range(1, 7):
        if unit <= 2:
            gvar = 2001  # Cohort 2001
        elif unit <= 4:
            gvar = 2002  # Cohort 2002
        else:
            gvar = 0  # Never treated
        
        for year in range(2000, 2004):
            y = 1.0 + 0.1 * unit + 0.05 * year + np.random.normal(0, 0.1)
            # Add treatment effect
            if gvar > 0 and year >= gvar:
                y += 0.5  # Treatment effect
            
            data.append({
                'id': unit,
                'year': year,
                'y': y,
                'gvar': gvar
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def staggered_test_data_no_nt():
    """Create staggered test data without never-treated units."""
    np.random.seed(42)
    
    data = []
    for unit in range(1, 5):
        if unit <= 2:
            gvar = 2001
        else:
            gvar = 2002
        
        for year in range(2000, 2004):
            y = 1.0 + 0.1 * unit + np.random.normal(0, 0.1)
            if gvar > 0 and year >= gvar:
                y += 0.5
            
            data.append({
                'id': unit,
                'year': year,
                'y': y,
                'gvar': gvar
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def common_timing_data():
    """Create common timing test data for backward compatibility tests."""
    np.random.seed(42)
    
    data = []
    for unit in range(1, 6):
        treated = 1 if unit == 1 else 0
        
        for year in range(2000, 2005):
            post = 1 if year >= 2003 else 0
            y = 1.0 + 0.1 * unit + np.random.normal(0, 0.1)
            if treated and post:
                y += 0.5
            
            data.append({
                'id': unit,
                'year': year,
                'y': y,
                'd': treated,
                'post': post
            })
    
    return pd.DataFrame(data)


# =============================================================================
# Mode Detection Tests
# =============================================================================

class TestModeDetection:
    """Test mode detection logic."""
    
    def test_staggered_mode_with_gvar(self, staggered_test_data):
        """AC-2: Providing gvar should trigger staggered mode."""
        from lwdid import lwdid
        
        result = lwdid(
            data=staggered_test_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            aggregate='none'
        )
        
        assert result.is_staggered == True
        assert hasattr(result, 'cohorts')
        assert len(result.cohorts) > 0
    
    def test_common_timing_without_gvar(self, common_timing_data):
        """AC-2: Without gvar, should use common timing mode."""
        from lwdid import lwdid
        
        result = lwdid(
            data=common_timing_data,
            y='y',
            d='d',
            ivar='id',
            tvar='year',
            post='post',
            rolling='demean'
        )
        
        assert result.is_staggered == False
        assert result.att is not None
    
    def test_common_timing_missing_d_raises(self):
        """AC: Common timing mode missing d should raise error."""
        from lwdid import lwdid
        
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'year': [2000, 2001, 2000, 2001],
            'y': [1.0, 2.0, 1.5, 2.5],
            'post': [0, 1, 0, 1]
        })
        
        with pytest.raises(ValueError, match="'d'参数"):
            lwdid(data=data, y='y', ivar='id', tvar='year', post='post', rolling='demean')
    
    def test_common_timing_missing_post_raises(self):
        """AC: Common timing mode missing post should raise error."""
        from lwdid import lwdid
        
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'year': [2000, 2001, 2000, 2001],
            'y': [1.0, 2.0, 1.5, 2.5],
            'd': [1, 1, 0, 0]
        })
        
        with pytest.raises(ValueError, match="'post'参数"):
            lwdid(data=data, y='y', d='d', ivar='id', tvar='year', rolling='demean')


# =============================================================================
# Control Group Strategy Tests
# =============================================================================

class TestControlGroupStrategy:
    """Test control group strategy validation and auto-switching."""
    
    def test_control_group_auto_switch_for_cohort(self, staggered_test_data):
        """AC-4: aggregate='cohort' should force never_treated."""
        from lwdid import lwdid
        
        with pytest.warns(UserWarning, match="切换到'never_treated'"):
            result = lwdid(
                data=staggered_test_data,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                control_group='not_yet_treated',
                aggregate='cohort'
            )
        
        assert result.control_group == 'not_yet_treated'
        assert result.control_group_used == 'never_treated'
    
    def test_control_group_auto_switch_for_overall(self, staggered_test_data):
        """AC-4: aggregate='overall' should force never_treated."""
        from lwdid import lwdid
        
        with pytest.warns(UserWarning, match="切换到'never_treated'"):
            result = lwdid(
                data=staggered_test_data,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                control_group='not_yet_treated',
                aggregate='overall'
            )
        
        assert result.control_group_used == 'never_treated'
    
    def test_no_nt_raises_for_overall(self, staggered_test_data_no_nt):
        """AC-5: No NT units with aggregate='overall' should raise error."""
        from lwdid import lwdid
        
        with pytest.raises(ValueError, match="没有never treated单位"):
            lwdid(
                data=staggered_test_data_no_nt,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                aggregate='overall'
            )
    
    def test_no_nt_allows_none_aggregate(self, staggered_test_data_no_nt):
        """AC: No NT units with aggregate='none' should work."""
        from lwdid import lwdid
        
        result = lwdid(
            data=staggered_test_data_no_nt,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            control_group='not_yet_treated',
            aggregate='none'
        )
        
        assert result.is_staggered == True
        assert result.att_by_cohort_time is not None


# =============================================================================
# Mode Conflict Tests
# =============================================================================

class TestModeConflict:
    """Test handling of conflicting mode parameters."""
    
    def test_mode_conflict_warning(self, staggered_test_data):
        """AC-6: Providing both gvar and d/post should warn."""
        from lwdid import lwdid
        
        data = staggered_test_data.copy()
        data['d'] = 1
        data['post'] = 0
        
        with pytest.warns(UserWarning, match="同时提供了gvar和d/post"):
            result = lwdid(
                data=data,
                y='y',
                d='d',
                ivar='id',
                tvar='year',
                post='post',
                gvar='gvar',
                rolling='demean',
                aggregate='none'
            )
        
        # Should use staggered mode
        assert result.is_staggered == True


# =============================================================================
# Estimator Tests
# =============================================================================

class TestEstimator:
    """Test estimator parameter validation."""
    
    def test_ipwra_requires_controls(self, staggered_test_data):
        """AC-7: IPWRA without controls should raise ValueError."""
        from lwdid import lwdid
        
        # IPWRA is now implemented but requires controls parameter
        with pytest.raises(ValueError, match="controls"):
            lwdid(
                data=staggered_test_data,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                estimator='ipwra',
                aggregate='none'
            )
    
    def test_psm_requires_controls(self, staggered_test_data):
        """PSM without controls should raise ValueError."""
        from lwdid import lwdid
        
        with pytest.raises(ValueError, match="controls"):
            lwdid(
                data=staggered_test_data,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                estimator='psm',
                aggregate='none'
            )
    
    def test_ipwra_with_controls_works(self):
        """IPWRA with controls should work correctly."""
        from lwdid import lwdid
        
        np.random.seed(42)
        # Create data with controls
        data = []
        for unit in range(1, 11):
            if unit <= 4:
                gvar = 2001
            elif unit <= 7:
                gvar = 2002
            else:
                gvar = 0  # Never treated
            
            for year in range(2000, 2004):
                x1 = np.random.normal(unit * 0.1, 0.1)
                y = 1.0 + 0.1 * unit + 0.2 * x1 + np.random.normal(0, 0.1)
                if gvar > 0 and year >= gvar:
                    y += 0.5
                
                data.append({
                    'id': unit,
                    'year': year,
                    'y': y,
                    'gvar': gvar,
                    'x1': x1
                })
        
        df = pd.DataFrame(data)
        
        # IPWRA with controls should work
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=df,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                estimator='ipwra',
                controls=['x1'],
                aggregate='none'
            )
        
        assert result.is_staggered == True
        assert result.att_by_cohort_time is not None


# =============================================================================
# Rolling Method Tests
# =============================================================================

class TestRollingMethod:
    """Test rolling parameter validation for staggered mode."""
    
    def test_rolling_quarterly_not_supported(self, staggered_test_data):
        """AC-8: Quarterly rolling methods should not be supported in staggered."""
        from lwdid import lwdid
        
        with pytest.raises(ValueError, match="不支持"):
            lwdid(
                data=staggered_test_data,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                rolling='demeanq',
                aggregate='none'
            )


# =============================================================================
# Data Validation Tests
# =============================================================================

class TestDataValidation:
    """Test staggered data validation."""
    
    def test_gvar_column_not_exists(self):
        """AC-13: Missing gvar column should raise error."""
        from lwdid import lwdid
        
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'year': [2000, 2001, 2000, 2001],
            'y': [1.0, 2.0, 1.5, 2.5]
        })
        
        with pytest.raises(Exception, match="not found|不存在|columns"):
            lwdid(
                data=data,
                y='y',
                ivar='id',
                tvar='year',
                gvar='nonexistent',
                rolling='demean'
            )
    
    def test_gvar_negative_value(self):
        """AC-14: Negative gvar values should raise error."""
        from lwdid import lwdid
        from lwdid.exceptions import InvalidStaggeredDataError
        
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'year': [2000, 2001, 2000, 2001],
            'y': [1.0, 2.0, 1.5, 2.5],
            'gvar': [2001, 2001, -1, -1]
        })
        
        with pytest.raises((InvalidStaggeredDataError, ValueError), match="负|invalid|negative"):
            lwdid(
                data=data,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                rolling='demean'
            )
    
    def test_nt_units_warning_when_few(self, staggered_test_data):
        """AC-17: Few NT units should warn."""
        from lwdid import lwdid
        
        # Create data with only 1 NT unit
        data = staggered_test_data[staggered_test_data['id'] != 6].copy()
        
        # Should warn about few NT units
        with pytest.warns(UserWarning, match="单位数量过少|N=1"):
            result = lwdid(
                data=data,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                aggregate='overall'
            )


# =============================================================================
# Output Validation Tests
# =============================================================================

class TestOutputValidation:
    """Test output attributes for staggered mode."""
    
    def test_is_staggered_attribute(self, staggered_test_data):
        """AC-18: is_staggered should be True in staggered mode."""
        from lwdid import lwdid
        
        result = lwdid(
            data=staggered_test_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            aggregate='none'
        )
        
        assert result.is_staggered == True
    
    def test_cohorts_attribute(self, staggered_test_data):
        """AC-19: cohorts should contain valid cohort years."""
        from lwdid import lwdid
        
        result = lwdid(
            data=staggered_test_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            aggregate='none'
        )
        
        assert result.cohorts == [2001, 2002]
    
    def test_control_group_used_attribute(self, staggered_test_data):
        """AC-20: control_group_used should record actual strategy."""
        from lwdid import lwdid
        
        # Suppress the warning for this test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=staggered_test_data,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                control_group='not_yet_treated',
                aggregate='overall'
            )
        
        assert result.control_group_used == 'never_treated'
    
    def test_att_overall_not_none_for_overall(self, staggered_test_data):
        """AC-21: att_overall should not be None when aggregate='overall'."""
        from lwdid import lwdid
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=staggered_test_data,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                aggregate='overall'
            )
        
        assert result.att_overall is not None
        assert result.se_overall is not None


# =============================================================================
# Backward Compatibility Tests
# =============================================================================

class TestBackwardCompatibility:
    """Test backward compatibility with common timing usage."""
    
    def test_positional_args_still_work(self, common_timing_data):
        """AC-9, AC-12: Positional arguments should still work."""
        from lwdid import lwdid
        
        # This should work without breaking
        result = lwdid(
            common_timing_data,
            'y',
            'd',
            'id',
            'year',
            'post',
            'demean'
        )
        
        assert result.att is not None
        assert result.se_att is not None
    
    def test_keyword_args_still_work(self, common_timing_data):
        """AC-12: Keyword arguments should still work."""
        from lwdid import lwdid
        
        result = lwdid(
            data=common_timing_data,
            y='y',
            d='d',
            ivar='id',
            tvar='year',
            post='post',
            rolling='demean'
        )
        
        assert result.att is not None


# =============================================================================
# is_never_treated Function Tests
# =============================================================================

class TestIsNeverTreated:
    """Test is_never_treated() utility function."""
    
    def test_zero_is_nt(self):
        """0 should be identified as never treated."""
        from lwdid.validation import is_never_treated
        assert is_never_treated(0) == True
    
    def test_inf_is_nt(self):
        """np.inf should be identified as never treated."""
        from lwdid.validation import is_never_treated
        assert is_never_treated(np.inf) == True
    
    def test_nan_is_nt(self):
        """NaN should be identified as never treated."""
        from lwdid.validation import is_never_treated
        assert is_never_treated(np.nan) == True
    
    def test_positive_int_is_not_nt(self):
        """Positive integers should not be never treated."""
        from lwdid.validation import is_never_treated
        assert is_never_treated(2005) == False
        assert is_never_treated(2001) == False


# =============================================================================
# validate_staggered_data Function Tests
# =============================================================================

class TestValidateStaggeredData:
    """Test validate_staggered_data() function."""
    
    def test_basic_validation(self, staggered_test_data):
        """Basic validation should work."""
        from lwdid.validation import validate_staggered_data
        
        result = validate_staggered_data(
            data=staggered_test_data,
            gvar='gvar',
            ivar='id',
            tvar='year',
            y='y'
        )
        
        assert 'cohorts' in result
        assert 'n_never_treated' in result
        assert result['cohorts'] == [2001, 2002]
        assert result['n_never_treated'] == 2
    
    def test_missing_column_raises(self):
        """Missing column should raise error."""
        from lwdid.validation import validate_staggered_data
        from lwdid.exceptions import MissingRequiredColumnError
        
        data = pd.DataFrame({
            'id': [1, 2],
            'year': [2000, 2000],
            'y': [1.0, 2.0]
        })
        
        with pytest.raises(MissingRequiredColumnError):
            validate_staggered_data(data, 'gvar', 'id', 'year', 'y')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
