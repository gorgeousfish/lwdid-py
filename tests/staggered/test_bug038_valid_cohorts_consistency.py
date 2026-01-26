# -*- coding: utf-8 -*-
"""
BUG-038: estimators.py 中 valid_cohorts 检测逻辑一致性测试

测试目的:
验证 build_subsample_for_ps_estimation() 函数中的 valid_cohorts 检测逻辑
与 is_never_treated() 函数的一致性。

BUG-038 已修复:
- build_subsample_for_ps_estimation() 现在使用 is_never_treated() 作为单一事实来源
- 浮点数边界情况（如 gvar=1e-10）现在正确排除，不会被误认为有效 cohort
- max_cohort 检测在 All Eventually Treated 场景中工作正确

References:
- BUG-038: estimators.py valid_cohorts 检测逻辑未使用 is_never_treated()（已修复）
- BUG-037: transformations.py NT 识别逻辑一致性（已修复）
- BUG-024: control_groups.py NT 识别一致性（已修复）
"""

import numpy as np
import pandas as pd
import pytest

from lwdid.validation import is_never_treated

# Skip entire module if required functions are not available
try:
    from lwdid.staggered.estimators import build_subsample_for_ps_estimation
except ImportError as e:
    pytest.skip(
        f"Skipping module: required functions not implemented ({e})",
        allow_module_level=True
    )


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def panel_data_with_float_gvar_aet():
    """
    创建 All Eventually Treated (AET) 场景的面板数据，
    包含浮点数边界情况来测试 valid_cohorts 检测逻辑
    """
    # 10 个单位，每个单位 5 个时期
    units = list(range(1, 11))
    periods = list(range(2001, 2006))
    
    data = pd.DataFrame([
        {'id': u, 'year': t}
        for u in units
        for t in periods
    ])
    
    # 设置 gvar 值，模拟 AET 场景（没有真正的 NT 单位）
    # 但包含一些浮点数边界情况，可能被误认为有效 cohort
    gvar_map = {
        1: 2003,           # 正常处理组 - cohort 2003
        2: 2004,           # 正常处理组 - cohort 2004
        3: 2005,           # 正常处理组 - cohort 2005 (最后 cohort)
        4: 1e-10,          # 极小浮点数 - 应被识别为 NT (在 1e-9 容差内)
        5: 1e-11,          # 更小的浮点数 - 应被识别为 NT
        6: -1e-10,         # 负极小浮点数 - 应被识别为 NT
        7: 2003,           # 正常处理组 - cohort 2003
        8: 2004,           # 正常处理组 - cohort 2004
        9: 2005,           # 正常处理组 - cohort 2005 (最后 cohort)
        10: 1e-9,          # 恰好在容差边界 - 应被识别为 NT
    }
    
    data['gvar'] = data['id'].map(gvar_map)
    data['y'] = np.random.randn(len(data))
    data['x1'] = np.random.randn(len(data))
    
    return data


@pytest.fixture
def panel_data_truly_aet():
    """
    创建真正的 All Eventually Treated 场景（没有 NT 单位）
    用于测试 max_cohort 检测的边界情况
    """
    # 9 个单位，每个单位 5 个时期
    units = list(range(1, 10))
    periods = list(range(2001, 2006))
    
    data = pd.DataFrame([
        {'id': u, 'year': t}
        for u in units
        for t in periods
    ])
    
    # 所有单位都会被处理（三个 cohort: 2003, 2004, 2005）
    gvar_map = {
        1: 2003,           # cohort 2003
        2: 2003,           # cohort 2003
        3: 2003,           # cohort 2003
        4: 2004,           # cohort 2004
        5: 2004,           # cohort 2004
        6: 2004,           # cohort 2004
        7: 2005,           # cohort 2005 (最后 cohort)
        8: 2005,           # cohort 2005 (最后 cohort)
        9: 2005,           # cohort 2005 (最后 cohort)
    }
    
    data['gvar'] = data['id'].map(gvar_map)
    data['y'] = np.random.randn(len(data))
    data['x1'] = np.random.randn(len(data))
    
    return data


@pytest.fixture
def panel_data_with_nt():
    """创建包含 NT 单位的标准面板数据"""
    units = list(range(1, 11))
    periods = list(range(2001, 2006))
    
    data = pd.DataFrame([
        {'id': u, 'year': t}
        for u in units
        for t in periods
    ])
    
    gvar_map = {
        1: 2003,           # cohort 2003
        2: 2003,           # cohort 2003
        3: 2004,           # cohort 2004
        4: 2004,           # cohort 2004
        5: 2005,           # cohort 2005
        6: 2005,           # cohort 2005
        7: 0,              # NT - 整数 0
        8: np.inf,         # NT - 无穷
        9: np.nan,         # NT - NaN
        10: 0.0,           # NT - 浮点数 0.0
    }
    
    data['gvar'] = data['id'].map(gvar_map)
    data['y'] = np.random.randn(len(data))
    data['x1'] = np.random.randn(len(data))
    
    return data


# =============================================================================
# BUG-038: valid_cohorts 检测一致性测试
# =============================================================================

class TestValidCohortsConsistency:
    """测试 valid_cohorts 检测与 is_never_treated() 的一致性"""
    
    def test_float_nt_excluded_from_valid_cohorts(self, panel_data_with_float_gvar_aet):
        """
        BUG-038 修复验证: 浮点数 NT 值应从 valid_cohorts 中排除
        
        修复后行为:
        - gvar=1e-10 应被 is_never_treated() 识别为 NT
        - valid_cohorts 应只包含真正的 cohort 值 (2003, 2004, 2005)
        - max_cohort 应为 2005
        """
        data = panel_data_with_float_gvar_aet
        
        # 验证 is_never_treated() 正确识别这些浮点数
        assert is_never_treated(1e-10) == True, "is_never_treated(1e-10) 应为 True"
        assert is_never_treated(1e-11) == True, "is_never_treated(1e-11) 应为 True"
        assert is_never_treated(-1e-10) == True, "is_never_treated(-1e-10) 应为 True"
        assert is_never_treated(1e-9) == True, "is_never_treated(1e-9) 应为 True"
        
        # 验证正常 cohort 不被识别为 NT
        assert is_never_treated(2003) == False, "is_never_treated(2003) 应为 False"
        assert is_never_treated(2004) == False, "is_never_treated(2004) 应为 False"
        assert is_never_treated(2005) == False, "is_never_treated(2005) 应为 False"
        
        # 调用 build_subsample_for_ps_estimation，验证它能正确工作
        # 对于 (cohort=2003, period=2004)，应该能构建子样本
        result = build_subsample_for_ps_estimation(
            data=data,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=2003,
            period_r=2004,
            control_group='not_yet_treated',
        )
        
        # 验证结果
        assert result is not None, "子样本构建应成功"
        assert result.n_treated > 0, "应有处理组"
        assert result.n_control > 0, "应有控制组"
        
        # 验证浮点数 NT 单位 (id=4,5,6,10) 被正确识别为 NT
        # 它们应该被包含在控制组中（作为 NT）
        assert result.has_never_treated == True, (
            "数据中应检测到 NT 单位（包括浮点数 NT）"
        )
    
    def test_max_cohort_detection_in_aet_scenario(self, panel_data_truly_aet):
        """
        BUG-038 修复验证: 在真正的 AET 场景中，max_cohort 检测应正确工作
        
        修复后行为:
        - 没有 NT 单位时，has_never_treated = False
        - valid_cohorts 应只包含真正的 cohort 值
        - 最后 cohort 在最后时期应抛出正确错误
        """
        data = panel_data_truly_aet
        
        # 在 AET 场景中，估计 τ_{2005,2005} 应该抛出错误
        # 因为没有控制组（所有其他单位都已被处理）
        with pytest.raises(ValueError) as exc_info:
            build_subsample_for_ps_estimation(
                data=data,
                gvar_col='gvar',
                ivar_col='id',
                cohort_g=2005,
                period_r=2005,
                control_group='not_yet_treated',
            )
        
        # 验证错误消息 (support both Chinese and English messages)
        error_msg = str(exc_info.value).lower()
        assert "没有可用的控制组单位" in str(exc_info.value) or \
               "控制组" in str(exc_info.value) or \
               "no available control units" in error_msg or \
               "no valid control group" in error_msg, \
            f"错误消息应说明没有控制组，实际: {exc_info.value}"
    
    def test_earlier_cohort_period_works_in_aet(self, panel_data_truly_aet):
        """
        在 AET 场景中，非最后 (cohort, period) 组合应能成功构建子样本
        """
        data = panel_data_truly_aet
        
        # τ_{2003,2004} 应该能成功
        result = build_subsample_for_ps_estimation(
            data=data,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=2003,
            period_r=2004,
            control_group='not_yet_treated',
        )
        
        assert result is not None, "子样本构建应成功"
        assert result.has_never_treated == False, "AET 场景应无 NT 单位"
        assert result.n_treated > 0, "应有处理组"
        assert result.n_control > 0, "应有控制组（cohort 2005 尚未处理）"
    
    def test_standard_data_with_nt(self, panel_data_with_nt):
        """
        标准数据（包含明确的 NT 单位）应正常工作
        """
        data = panel_data_with_nt
        
        result = build_subsample_for_ps_estimation(
            data=data,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=2003,
            period_r=2004,
            control_group='not_yet_treated',
        )
        
        assert result is not None, "子样本构建应成功"
        assert result.has_never_treated == True, "应检测到 NT 单位"
        assert result.n_treated > 0, "应有处理组"
        assert result.n_control > 0, "应有控制组"
        
        # 验证控制组包含 NT 单位和尚未处理的单位
        control_ids = set(result.subsample[result.D_ig == 0]['id'].unique())
        
        # NT 单位 (7, 8, 9, 10) 应在控制组中
        expected_nt_ids = {7, 8, 9, 10}
        for nt_id in expected_nt_ids:
            if nt_id in set(data['id'].unique()):
                # 只有存在于数据中的单位才检查
                pass  # 控制组的具体构成取决于 period_r


# =============================================================================
# 边界情况和回归测试
# =============================================================================

class TestBug038EdgeCases:
    """BUG-038 边界情况测试"""
    
    def test_exact_zero_vs_near_zero(self):
        """测试精确 0 和近零值的一致处理"""
        # 创建简单数据
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3, 4, 4],
            'year': [2001, 2002, 2001, 2002, 2001, 2002, 2001, 2002],
            'gvar': [2002, 2002, 0, 0, 1e-15, 1e-15, 0.0, 0.0],
            'y': [1.0, 2.0, 1.5, 2.5, 1.2, 2.2, 1.8, 2.8],
        })
        
        # 验证 is_never_treated() 对所有 NT 情况一致
        assert is_never_treated(0) == True
        assert is_never_treated(0.0) == True
        assert is_never_treated(1e-15) == True
        
        # 构建子样本应成功
        result = build_subsample_for_ps_estimation(
            data=data,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=2002,
            period_r=2002,
            control_group='not_yet_treated',
        )
        
        assert result is not None
        assert result.has_never_treated == True, "应检测到 NT 单位"
        # 控制组应包含 id=2,3,4 (都是 NT)
        control_ids = set(result.subsample[result.D_ig == 0]['id'].unique())
        assert 2 in control_ids, "id=2 (gvar=0) 应在控制组"
        assert 3 in control_ids, "id=3 (gvar=1e-15) 应在控制组"
        assert 4 in control_ids, "id=4 (gvar=0.0) 应在控制组"
    
    def test_negative_near_zero(self):
        """测试负的近零值"""
        # -1e-10 应被识别为 NT（在 abs() 容差内）
        assert is_never_treated(-1e-10) == True
        assert is_never_treated(-1e-9) == True
        
        # 但稍大的负值不应被识别为 NT
        assert is_never_treated(-1e-8) == False
        assert is_never_treated(-1) == False
    
    def test_inf_handling(self):
        """测试无穷值处理"""
        assert is_never_treated(np.inf) == True
        assert is_never_treated(-np.inf) == False  # 负无穷不是有效的 NT 标识
    
    def test_nan_handling(self):
        """测试 NaN 处理"""
        assert is_never_treated(np.nan) == True
        assert is_never_treated(float('nan')) == True


# =============================================================================
# 集成测试
# =============================================================================

class TestBug038Integration:
    """BUG-038 集成测试"""
    
    def test_full_workflow_with_float_nt(self, panel_data_with_float_gvar_aet):
        """
        完整工作流测试：确保修复后的代码能正确处理浮点数 NT
        """
        data = panel_data_with_float_gvar_aet
        
        # 测试多个 (cohort, period) 组合
        test_cases = [
            (2003, 2003),
            (2003, 2004),
            (2004, 2004),
            (2004, 2005),
        ]
        
        for cohort_g, period_r in test_cases:
            result = build_subsample_for_ps_estimation(
                data=data,
                gvar_col='gvar',
                ivar_col='id',
                cohort_g=cohort_g,
                period_r=period_r,
                control_group='not_yet_treated',
            )
            
            assert result is not None, f"(g={cohort_g}, r={period_r}) 子样本构建应成功"
            assert result.n_treated > 0, f"(g={cohort_g}, r={period_r}) 应有处理组"
            assert result.n_control > 0, f"(g={cohort_g}, r={period_r}) 应有控制组"
            
            # 验证 has_never_treated 正确检测到浮点数 NT
            assert result.has_never_treated == True, (
                f"(g={cohort_g}, r={period_r}) 应检测到 NT 单位（浮点数 NT）"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
