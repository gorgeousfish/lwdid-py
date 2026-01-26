# -*- coding: utf-8 -*-
"""
BUG-024: control_groups.py NT识别一致性测试

测试目的:
验证 control_groups.py 中的 NT（Never Treated）识别逻辑与
validation.py 中的 is_never_treated() 函数的一致性。

BUG-024 已修复:
- control_groups.py 现在使用 is_never_treated() 作为单一事实来源
- 浮点数边界情况（如 gvar=1e-10）现在处理一致
- 所有模块统一使用相同的 NT 识别逻辑

References:
- BUG-024: control_groups.py 未使用 is_never_treated() 函数（已修复）
- BUG-020: is_never_treated() 浮点比较问题（已修复）
"""

import numpy as np
import pandas as pd
import pytest

from lwdid.validation import is_never_treated
from lwdid.staggered.control_groups import (
    identify_never_treated_units,
    get_valid_control_units,
    ControlGroupStrategy,
)


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def panel_data_with_float_gvar():
    """创建包含浮点数边界情况的面板数据"""
    # 10 个单位，每个单位 5 个时期
    units = list(range(1, 11))
    periods = list(range(2001, 2006))
    
    data = pd.DataFrame([
        {'id': u, 'year': t}
        for u in units
        for t in periods
    ])
    
    # 设置 gvar 值，包含浮点数边界情况
    gvar_map = {
        1: 2003,           # 正常处理组
        2: 2004,           # 正常处理组
        3: 0,              # 整数 0 - 应为 NT
        4: 0.0,            # 浮点数 0.0 - 应为 NT
        5: 1e-10,          # 极小浮点数 - 应为 NT（BUG-024 关键测试）
        6: np.inf,         # 正无穷 - 应为 NT
        7: np.nan,         # NaN - 应为 NT
        8: 1e-9,           # 恰好在容差边界 - 应为 NT
        9: 1e-8,           # 超出容差边界 - 应为处理组
        10: -1e-10,        # 负极小浮点数 - 应为 NT
    }
    
    data['gvar'] = data['id'].map(gvar_map)
    data['y'] = np.random.randn(len(data))
    
    return data


@pytest.fixture
def expected_nt_status():
    """is_never_treated() 函数的预期结果"""
    return {
        1: False,   # 2003 - 处理组
        2: False,   # 2004 - 处理组
        3: True,    # 0 - NT
        4: True,    # 0.0 - NT
        5: True,    # 1e-10 - NT（在容差内）
        6: True,    # inf - NT
        7: True,    # NaN - NT
        8: True,    # 1e-9 - NT（边界值）
        9: False,   # 1e-8 - 处理组（超出容差）
        10: True,   # -1e-10 - NT（负值在容差内）
    }


# =============================================================================
# is_never_treated() 函数测试（基准）
# =============================================================================

class TestIsNeverTreatedBaseline:
    """验证 is_never_treated() 作为基准正确处理所有情况"""
    
    def test_integer_zero(self):
        """整数 0 应为 NT"""
        assert is_never_treated(0) == True
    
    def test_float_zero(self):
        """浮点数 0.0 应为 NT"""
        assert is_never_treated(0.0) == True
    
    def test_small_float_within_tolerance(self):
        """1e-10（在 1e-9 容差内）应为 NT"""
        assert is_never_treated(1e-10) == True
    
    def test_boundary_value(self):
        """1e-9（边界值）应为 NT"""
        assert is_never_treated(1e-9) == True
    
    def test_outside_tolerance(self):
        """1e-8（超出 1e-9 容差）应为处理组"""
        assert is_never_treated(1e-8) == False
    
    def test_negative_small_float(self):
        """-1e-10（负值在容差内）应为 NT"""
        assert is_never_treated(-1e-10) == True
    
    def test_inf(self):
        """正无穷应为 NT"""
        assert is_never_treated(np.inf) == True
    
    def test_nan(self):
        """NaN 应为 NT"""
        assert is_never_treated(np.nan) == True
    
    def test_normal_cohort(self):
        """正常 cohort 值应为处理组"""
        assert is_never_treated(2003) == False
        assert is_never_treated(2003.0) == False


# =============================================================================
# BUG-024: 一致性测试（验证修复）
# =============================================================================

class TestNTIdentificationConsistency:
    """测试 control_groups.py 与 is_never_treated() 的一致性（BUG-024 已修复）"""
    
    def test_identify_never_treated_vs_is_never_treated(
        self, panel_data_with_float_gvar, expected_nt_status
    ):
        """
        BUG-024 修复验证: identify_never_treated_units() 与 is_never_treated() 应一致
        
        修复后行为：
        - identify_never_treated_units() 使用 is_never_treated() 作为单一事实来源
        - 浮点数边界情况（如 1e-10）正确处理
        - 两者对所有输入应返回相同结果
        """
        data = panel_data_with_float_gvar
        
        # 使用 identify_never_treated_units()
        nt_mask_control_groups = identify_never_treated_units(
            data, 'gvar', 'id', never_treated_values=[0, np.inf]
        )
        
        # 使用 is_never_treated() 作为基准
        unit_gvar = data.groupby('id')['gvar'].first()
        nt_mask_validation = unit_gvar.apply(is_never_treated)
        
        # 检查每个单位的一致性
        for unit_id in expected_nt_status:
            control_groups_result = nt_mask_control_groups[unit_id]
            validation_result = nt_mask_validation[unit_id]
            expected = expected_nt_status[unit_id]
            
            assert control_groups_result == validation_result, (
                f"Unit {unit_id}: control_groups={control_groups_result}, "
                f"is_never_treated={validation_result} (不一致!)"
            )
            assert control_groups_result == expected, (
                f"Unit {unit_id}: 结果={control_groups_result}, 预期={expected}"
            )
    
    def test_float_boundary_case_1e10(self, panel_data_with_float_gvar):
        """
        BUG-024 修复验证: gvar=1e-10 应被识别为 NT
        
        修复后行为:
        - is_never_treated(1e-10) = True
        - identify_never_treated_units 对 gvar=1e-10 返回 True
        """
        data = panel_data_with_float_gvar
        
        # is_never_treated() 正确返回 True
        assert is_never_treated(1e-10) == True, "is_never_treated(1e-10) 应为 True"
        
        # identify_never_treated_units() 应也返回 True
        nt_mask = identify_never_treated_units(
            data, 'gvar', 'id', never_treated_values=[0, np.inf]
        )
        
        # Unit 5 的 gvar = 1e-10
        unit5_is_nt = nt_mask[5]
        assert unit5_is_nt == True, (
            "BUG-024 修复验证失败: identify_never_treated_units() 对 gvar=1e-10 "
            "应返回 True"
        )
    
    def test_get_valid_control_units_consistency(self, panel_data_with_float_gvar):
        """
        BUG-024 修复验证: get_valid_control_units() 的 NT 识别也应一致
        """
        data = panel_data_with_float_gvar
        
        # 获取 (cohort=2003, period=2004) 的控制单位掩码
        control_mask = get_valid_control_units(
            data, 'gvar', 'id',
            cohort=2003, period=2004,
            strategy=ControlGroupStrategy.NEVER_TREATED,
            never_treated_values=[0, np.inf]
        )
        
        # Unit 5 (gvar=1e-10) 应被识别为 NT 并成为控制组的一部分
        unit_gvar = data.groupby('id')['gvar'].first()
        
        # 验证 is_never_treated() 确实说这是 NT
        assert is_never_treated(unit_gvar[5]) == True
        
        # 检查 Unit 5 是否被正确识别为控制单位
        assert control_mask[5] == True, (
            "BUG-024 修复验证失败: get_valid_control_units() 未将 gvar=1e-10 "
            "识别为控制单位"
        )


# =============================================================================
# 修复后的行为验证测试
# =============================================================================

class TestExpectedBehaviorAfterFix:
    """
    BUG-024 修复后的行为验证
    
    这些测试验证修复后的正确行为。
    """
    
    def test_all_nt_cases_match(self, panel_data_with_float_gvar, expected_nt_status):
        """BUG-024 修复验证: 所有 NT 情况应匹配"""
        data = panel_data_with_float_gvar
        
        nt_mask = identify_never_treated_units(
            data, 'gvar', 'id', never_treated_values=[0, np.inf]
        )
        
        for unit_id, expected_is_nt in expected_nt_status.items():
            actual = nt_mask[unit_id]
            assert actual == expected_is_nt, (
                f"Unit {unit_id}: expected NT={expected_is_nt}, got {actual}"
            )


# =============================================================================
# 边界情况和回归测试
# =============================================================================

class TestEdgeCasesAndRegression:
    """边界情况和回归测试"""
    
    def test_normal_cases_still_work(self, panel_data_with_float_gvar):
        """确保正常情况不受影响"""
        data = panel_data_with_float_gvar
        
        nt_mask = identify_never_treated_units(
            data, 'gvar', 'id', never_treated_values=[0, np.inf]
        )
        
        # 正常 cohort 不应被识别为 NT
        assert nt_mask[1] == False, "Unit 1 (gvar=2003) 不应为 NT"
        assert nt_mask[2] == False, "Unit 2 (gvar=2004) 不应为 NT"
        
        # 明确的 NT 值应被识别
        assert nt_mask[3] == True, "Unit 3 (gvar=0) 应为 NT"
        assert nt_mask[6] == True, "Unit 6 (gvar=inf) 应为 NT"
        assert nt_mask[7] == True, "Unit 7 (gvar=NaN) 应为 NT"
    
    def test_empty_data_raises_error(self):
        """空数据应抛出 ValueError"""
        empty_data = pd.DataFrame(columns=['id', 'gvar', 'y'])
        
        with pytest.raises(ValueError, match="empty"):
            identify_never_treated_units(empty_data, 'gvar', 'id')
    
    def test_missing_column_raises_error(self, panel_data_with_float_gvar):
        """缺失列应抛出 KeyError"""
        data = panel_data_with_float_gvar
        
        with pytest.raises(KeyError):
            identify_never_treated_units(data, 'nonexistent', 'id')
