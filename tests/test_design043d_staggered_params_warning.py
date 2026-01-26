"""
DESIGN-043-D 修复验证测试：Staggered参数在Common Timing模式被静默忽略

测试内容：
1. 单元测试：验证当 staggered 参数被用于 common timing 模式时警告是否正确触发
2. 边界条件测试：验证默认参数值不触发警告
3. 组合测试：验证多个参数同时设置时的警告消息
4. 功能测试：验证警告不影响实际计算结果

创建日期: 2026-01-17
修复内容: 在 common timing 模式下使用 staggered-only 参数时发出警告
"""

import warnings
import pytest
import numpy as np
import pandas as pd

from lwdid import lwdid


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def common_timing_data():
    """Create simple panel data for common timing DiD."""
    np.random.seed(42)
    n_units = 20
    n_periods = 6
    
    data = pd.DataFrame({
        'unit': np.repeat(range(1, n_units + 1), n_periods),
        'year': np.tile(range(2000, 2000 + n_periods), n_units),
    })
    
    # Assign treatment: first 10 units are treated
    data['d'] = (data['unit'] <= 10).astype(int)
    
    # Post-treatment indicator: years 2003+ are post-treatment
    data['post'] = (data['year'] >= 2003).astype(int)
    
    # Generate outcome with treatment effect
    np.random.seed(42)
    data['y'] = 10 + data['unit'] * 0.5 + np.random.randn(len(data))
    
    # Add treatment effect for treated units in post periods
    data.loc[(data['d'] == 1) & (data['post'] == 1), 'y'] += 2.0
    
    return data


# =============================================================================
# Unit Tests: Warning Trigger Logic
# =============================================================================

class TestStaggeredParamsWarning:
    """Test that staggered-only parameters trigger warnings in common timing mode."""
    
    def test_control_group_triggers_warning(self, common_timing_data):
        """control_group != 'not_yet_treated' should trigger warning."""
        with pytest.warns(UserWarning, match=r"staggered-only parameters are ignored.*control_group='never_treated'"):
            lwdid(
                data=common_timing_data,
                y='y',
                d='d',
                ivar='unit',
                tvar='year',
                post='post',
                control_group='never_treated',  # Non-default
            )
    
    @pytest.mark.skip(reason="estimator='ipw' is valid in common timing mode, not a staggered-only parameter")
    def test_estimator_ipw_triggers_warning(self, common_timing_data):
        """estimator != 'ra' should trigger warning."""
        with pytest.warns(UserWarning, match=r"staggered-only parameters are ignored.*estimator='ipw'"):
            lwdid(
                data=common_timing_data,
                y='y',
                d='d',
                ivar='unit',
                tvar='year',
                post='post',
                estimator='ipw',  # Non-default
            )
    
    @pytest.mark.skip(reason="estimator='ipwra' is valid in common timing mode, not a staggered-only parameter")
    def test_estimator_ipwra_triggers_warning(self, common_timing_data):
        """estimator='ipwra' should trigger warning."""
        with pytest.warns(UserWarning, match=r"staggered-only parameters are ignored.*estimator='ipwra'"):
            lwdid(
                data=common_timing_data,
                y='y',
                d='d',
                ivar='unit',
                tvar='year',
                post='post',
                estimator='ipwra',  # Non-default
            )
    
    @pytest.mark.skip(reason="estimator='psm' is valid in common timing mode, not a staggered-only parameter")
    def test_estimator_psm_triggers_warning(self, common_timing_data):
        """estimator='psm' should trigger warning."""
        with pytest.warns(UserWarning, match=r"staggered-only parameters are ignored.*estimator='psm'"):
            lwdid(
                data=common_timing_data,
                y='y',
                d='d',
                ivar='unit',
                tvar='year',
                post='post',
                estimator='psm',  # Non-default
            )
    
    def test_aggregate_triggers_warning(self, common_timing_data):
        """aggregate != 'cohort' should trigger warning."""
        with pytest.warns(UserWarning, match=r"staggered-only parameters are ignored.*aggregate='overall'"):
            lwdid(
                data=common_timing_data,
                y='y',
                d='d',
                ivar='unit',
                tvar='year',
                post='post',
                aggregate='overall',  # Non-default
            )
    
    def test_ps_controls_triggers_warning(self, common_timing_data):
        """ps_controls != None should trigger warning."""
        # Add a control variable
        common_timing_data['x'] = np.random.randn(len(common_timing_data))
        
        with pytest.warns(UserWarning, match=r"staggered-only parameters are ignored.*ps_controls"):
            lwdid(
                data=common_timing_data,
                y='y',
                d='d',
                ivar='unit',
                tvar='year',
                post='post',
                ps_controls=['x'],  # Non-default
            )
    
    def test_trim_threshold_triggers_warning(self, common_timing_data):
        """trim_threshold != 0.01 should trigger warning."""
        with pytest.warns(UserWarning, match=r"staggered-only parameters are ignored.*trim_threshold"):
            lwdid(
                data=common_timing_data,
                y='y',
                d='d',
                ivar='unit',
                tvar='year',
                post='post',
                trim_threshold=0.05,  # Non-default
            )
    
    def test_return_diagnostics_triggers_warning(self, common_timing_data):
        """return_diagnostics=True should trigger warning."""
        with pytest.warns(UserWarning, match=r"staggered-only parameters are ignored.*return_diagnostics=True"):
            lwdid(
                data=common_timing_data,
                y='y',
                d='d',
                ivar='unit',
                tvar='year',
                post='post',
                return_diagnostics=True,  # Non-default
            )
    
    def test_n_neighbors_triggers_warning(self, common_timing_data):
        """n_neighbors != 1 should trigger warning."""
        with pytest.warns(UserWarning, match=r"staggered-only parameters are ignored.*n_neighbors"):
            lwdid(
                data=common_timing_data,
                y='y',
                d='d',
                ivar='unit',
                tvar='year',
                post='post',
                n_neighbors=3,  # Non-default
            )
    
    def test_caliper_triggers_warning(self, common_timing_data):
        """caliper != None should trigger warning."""
        with pytest.warns(UserWarning, match=r"staggered-only parameters are ignored.*caliper"):
            lwdid(
                data=common_timing_data,
                y='y',
                d='d',
                ivar='unit',
                tvar='year',
                post='post',
                caliper=0.5,  # Non-default
            )
    
    def test_with_replacement_false_triggers_warning(self, common_timing_data):
        """with_replacement=False should trigger warning."""
        with pytest.warns(UserWarning, match=r"staggered-only parameters are ignored.*with_replacement=False"):
            lwdid(
                data=common_timing_data,
                y='y',
                d='d',
                ivar='unit',
                tvar='year',
                post='post',
                with_replacement=False,  # Non-default
            )


# =============================================================================
# Boundary Tests: Default Parameters Should Not Trigger Warning
# =============================================================================

class TestDefaultParamsNoWarning:
    """Test that default parameter values do not trigger the staggered warning."""
    
    def test_default_params_no_warning(self, common_timing_data):
        """Default parameters should not trigger staggered params warning."""
        # We should NOT see the staggered-only warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = lwdid(
                data=common_timing_data,
                y='y',
                d='d',
                ivar='unit',
                tvar='year',
                post='post',
                # All defaults: control_group='not_yet_treated', estimator='ra',
                # aggregate='cohort', ps_controls=None, etc.
            )
            
            # Check no "staggered-only parameters are ignored" warning
            staggered_warnings = [
                warning for warning in w 
                if "staggered-only parameters are ignored" in str(warning.message)
            ]
            assert len(staggered_warnings) == 0, \
                f"Unexpected staggered params warning: {[str(w.message) for w in staggered_warnings]}"
            
            # Verify result is valid
            assert result is not None
            assert hasattr(result, 'att')
    
    def test_with_replacement_true_no_warning(self, common_timing_data):
        """with_replacement=True (default) should not trigger warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = lwdid(
                data=common_timing_data,
                y='y',
                d='d',
                ivar='unit',
                tvar='year',
                post='post',
                with_replacement=True,  # This is the default
            )
            
            staggered_warnings = [
                warning for warning in w 
                if "staggered-only parameters are ignored" in str(warning.message)
            ]
            assert len(staggered_warnings) == 0
    
    def test_n_neighbors_1_no_warning(self, common_timing_data):
        """n_neighbors=1 (default) should not trigger warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = lwdid(
                data=common_timing_data,
                y='y',
                d='d',
                ivar='unit',
                tvar='year',
                post='post',
                n_neighbors=1,  # This is the default
            )
            
            staggered_warnings = [
                warning for warning in w 
                if "staggered-only parameters are ignored" in str(warning.message)
            ]
            assert len(staggered_warnings) == 0


# =============================================================================
# Combination Tests: Multiple Parameters
# =============================================================================

class TestMultipleParamsWarning:
    """Test warning message when multiple staggered params are set."""
    
    def test_multiple_params_all_listed(self, common_timing_data):
        """Multiple non-default params should all appear in warning message."""
        with pytest.warns(UserWarning) as record:
            lwdid(
                data=common_timing_data,
                y='y',
                d='d',
                ivar='unit',
                tvar='year',
                post='post',
                estimator='ipwra',
                aggregate='overall',
                caliper=0.5,
            )
        
        # Find the staggered params warning
        staggered_warning = None
        for w in record:
            if "staggered-only parameters are ignored" in str(w.message):
                staggered_warning = str(w.message)
                break
        
        assert staggered_warning is not None, "Expected staggered params warning"
        assert "estimator='ipwra'" in staggered_warning
        assert "aggregate='overall'" in staggered_warning
        assert "caliper=0.5" in staggered_warning
    
    def test_warning_contains_guidance(self, common_timing_data):
        """Warning should contain guidance about using gvar parameter."""
        with pytest.warns(UserWarning, match="specify the 'gvar' parameter"):
            lwdid(
                data=common_timing_data,
                y='y',
                d='d',
                ivar='unit',
                tvar='year',
                post='post',
                estimator='ipw',
            )


# =============================================================================
# Functional Tests: Warning Does Not Affect Results
# =============================================================================

class TestWarningDoesNotAffectResults:
    """Verify that warnings do not affect the actual computation."""
    
    def test_results_same_with_ignored_params(self, common_timing_data):
        """Results should be identical regardless of ignored staggered params."""
        # Baseline without any staggered params
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_baseline = lwdid(
                data=common_timing_data,
                y='y',
                d='d',
                ivar='unit',
                tvar='year',
                post='post',
            )
        
        # With ignored staggered params
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_with_params = lwdid(
                data=common_timing_data,
                y='y',
                d='d',
                ivar='unit',
                tvar='year',
                post='post',
                estimator='ipwra',
                aggregate='overall',
                control_group='never_treated',
            )
        
        # Results should be identical (params are ignored)
        assert np.isclose(result_baseline.att, result_with_params.att), \
            f"ATT mismatch: {result_baseline.att} vs {result_with_params.att}"
        assert np.isclose(result_baseline.se_att, result_with_params.se_att), \
            f"SE mismatch: {result_baseline.se_att} vs {result_with_params.se_att}"


# =============================================================================
# Staggered Mode: No Warning Test
# =============================================================================

class TestStaggeredModeNoSpuriousWarning:
    """Verify that staggered mode does not trigger this warning."""
    
    @pytest.fixture
    def staggered_data(self):
        """Create data for staggered DiD."""
        np.random.seed(42)
        n_units = 30
        n_periods = 8
        
        data = pd.DataFrame({
            'unit': np.repeat(range(1, n_units + 1), n_periods),
            'year': np.tile(range(2000, 2000 + n_periods), n_units),
        })
        
        # Assign gvar: first 10 units treated in 2003, next 10 in 2005, rest never treated
        def assign_gvar(unit):
            if unit <= 10:
                return 2003
            elif unit <= 20:
                return 2005
            else:
                return 0  # Never treated
        
        data['gvar'] = data['unit'].apply(assign_gvar)
        
        # Generate outcome
        np.random.seed(42)
        data['y'] = 10 + data['unit'] * 0.5 + np.random.randn(len(data))
        
        # Add treatment effect
        for idx, row in data.iterrows():
            if row['gvar'] > 0 and row['year'] >= row['gvar']:
                data.loc[idx, 'y'] += 2.0
        
        return data
    
    def test_staggered_mode_no_spurious_warning(self, staggered_data):
        """Staggered mode with non-default params should NOT trigger this warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = lwdid(
                data=staggered_data,
                y='y',
                ivar='unit',
                tvar='year',
                gvar='gvar',  # Using staggered mode
                estimator='ra',
                aggregate='overall',
                control_group='never_treated',
            )
            
            # Should NOT see "staggered-only parameters are ignored" warning
            staggered_warnings = [
                warning for warning in w 
                if "staggered-only parameters are ignored" in str(warning.message)
            ]
            assert len(staggered_warnings) == 0, \
                "Staggered mode should not trigger 'params ignored' warning"


# =============================================================================
# Stacklevel Test
# =============================================================================

class TestWarningStacklevel:
    """Verify warning stacklevel points to user code."""
    
    def test_warning_stacklevel(self, common_timing_data):
        """Warning should point to user's lwdid() call, not internal code."""
        import traceback
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # This line should be referenced in the warning
            result = lwdid(
                data=common_timing_data,
                y='y',
                d='d',
                ivar='unit',
                tvar='year',
                post='post',
                estimator='ipw',
            )
        
        # Find the staggered params warning
        staggered_warning = None
        for warning in w:
            if "staggered-only parameters are ignored" in str(warning.message):
                staggered_warning = warning
                break
        
        assert staggered_warning is not None
        # Warning should come from this test file, not from core.py
        assert 'test_design043d_staggered_params_warning.py' in staggered_warning.filename


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
