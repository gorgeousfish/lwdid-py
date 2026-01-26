"""BUG-060-M: K值计算不友好错误消息修复验证测试

验证 transformations.py 中的 _compute_max_pre_tindex 辅助函数
在处理边界情况时能够提供友好的错误消息，而不是抛出不明确的
ValueError: cannot convert float NaN to integer。

测试场景:
1. 空的处理前数据（没有 post==0 的观测）
2. 处理前 tindex 全为 NaN
3. 混合数据（部分 tindex 为 NaN，部分有效）
4. 正常数据处理确保不受影响
5. Stata 行为一致性验证

参考 Stata 实现 (lwdid.ado 第 187-192 行):
```stata
qui su `tindex' if `post_'==0, meanonly
if r(N)==0 {
    di as err "No pre-treatment observations (post==0)."
    exit 2000
}
local K = r(max)
```
"""

import numpy as np
import pandas as pd
import pytest

from lwdid.transformations import apply_rolling_transform, _compute_max_pre_tindex
from lwdid.exceptions import InsufficientPrePeriodsError


class TestComputeMaxPreTindex:
    """单元测试: _compute_max_pre_tindex 辅助函数"""

    def test_empty_pre_treatment_data(self):
        """测试空的处理前数据抛出友好错误消息。
        
        场景: 所有观测都是 post==1（处理后），没有 post==0 的观测。
        预期: 抛出 InsufficientPrePeriodsError，消息包含 "No pre-treatment observations"。
        """
        data = pd.DataFrame({
            'post': [1, 1, 1],
            'tindex': [1, 2, 3],
        })
        
        with pytest.raises(InsufficientPrePeriodsError) as excinfo:
            _compute_max_pre_tindex(data, 'post', 'tindex', 'demeanq')
        
        assert "No pre-treatment observations found" in str(excinfo.value)
        assert "post==0" in str(excinfo.value)

    def test_all_nan_tindex(self):
        """测试处理前 tindex 全为 NaN 抛出友好错误消息。
        
        场景: 有处理前观测，但 tindex 全为 NaN（可能由 dropna 过滤导致）。
        预期: 抛出 InsufficientPrePeriodsError，消息包含 "All pre-treatment time index values are NaN"。
        """
        data = pd.DataFrame({
            'post': [0, 0, 1, 1],
            'tindex': [np.nan, np.nan, 3, 4],
        })
        
        with pytest.raises(InsufficientPrePeriodsError) as excinfo:
            _compute_max_pre_tindex(data, 'post', 'tindex', 'detrend')
        
        assert "All pre-treatment time index values are NaN" in str(excinfo.value)

    def test_partial_nan_tindex_returns_valid_max(self):
        """测试部分 tindex 为 NaN 时返回有效的最大值。
        
        场景: 处理前数据中部分 tindex 为 NaN，部分有效。
        预期: 返回有效 tindex 的最大值，忽略 NaN。
        """
        data = pd.DataFrame({
            'post': [0, 0, 0, 1, 1],
            'tindex': [1.0, np.nan, 3.0, 4.0, 5.0],
        })
        
        K = _compute_max_pre_tindex(data, 'post', 'tindex', 'demeanq')
        assert K == 3

    def test_normal_data_returns_correct_max(self):
        """测试正常数据返回正确的最大值。
        
        场景: 标准面板数据，处理前有多个有效观测。
        预期: 返回处理前 tindex 的最大值。
        """
        data = pd.DataFrame({
            'post': [0, 0, 0, 1, 1],
            'tindex': [1, 2, 3, 4, 5],
        })
        
        K = _compute_max_pre_tindex(data, 'post', 'tindex', 'detrend')
        assert K == 3
        assert isinstance(K, int)

    def test_error_message_includes_method_name(self):
        """测试错误消息包含方法名称（demeanq/detrendq/detrend）。
        
        这有助于用户理解是哪个 rolling 方法遇到了问题。
        """
        data = pd.DataFrame({
            'post': [1, 1, 1],
            'tindex': [1, 2, 3],
        })
        
        for method in ['demeanq', 'detrendq', 'detrend']:
            with pytest.raises(InsufficientPrePeriodsError) as excinfo:
                _compute_max_pre_tindex(data, 'post', 'tindex', method)
            
            assert f"rolling('{method}')" in str(excinfo.value)


class TestApplyRollingTransformKValueErrors:
    """集成测试: apply_rolling_transform 中的 K 值计算错误处理"""

    def test_demeanq_empty_pre_treatment(self):
        """测试 demeanq 在空处理前数据时的错误处理。"""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'y': [10.0, 15.0, 5.0, 8.0],
            'd_': [1, 1, 0, 0],
            'post_': [1, 1, 1, 1],  # 所有都是处理后
            'tindex': [1, 2, 1, 2],
            'quarter': [1, 2, 1, 2],
        })
        
        with pytest.raises(InsufficientPrePeriodsError) as excinfo:
            apply_rolling_transform(
                data=data, y='y', ivar='id', tindex='tindex', post='post_',
                rolling='demeanq', tpost1=1, quarter='quarter'
            )
        
        assert "No pre-treatment observations" in str(excinfo.value)

    def test_detrendq_empty_pre_treatment(self):
        """测试 detrendq 在空处理前数据时的错误处理。"""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'y': [10.0, 15.0, 5.0, 8.0],
            'd_': [1, 1, 0, 0],
            'post_': [1, 1, 1, 1],  # 所有都是处理后
            'tindex': [1, 2, 1, 2],
            'quarter': [1, 2, 1, 2],
        })
        
        with pytest.raises(InsufficientPrePeriodsError) as excinfo:
            apply_rolling_transform(
                data=data, y='y', ivar='id', tindex='tindex', post='post_',
                rolling='detrendq', tpost1=1, quarter='quarter'
            )
        
        assert "No pre-treatment observations" in str(excinfo.value)

    def test_detrend_empty_pre_treatment(self):
        """测试 detrend 在空处理前数据时的错误处理。"""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'y': [10.0, 15.0, 5.0, 8.0],
            'd_': [1, 1, 0, 0],
            'post_': [1, 1, 1, 1],  # 所有都是处理后
            'tindex': [1, 2, 1, 2],
        })
        
        with pytest.raises(InsufficientPrePeriodsError) as excinfo:
            apply_rolling_transform(
                data=data, y='y', ivar='id', tindex='tindex', post='post_',
                rolling='detrend', tpost1=1
            )
        
        assert "No pre-treatment observations" in str(excinfo.value)

    def test_demeanq_all_nan_tindex(self):
        """测试 demeanq 在处理前 tindex 全为 NaN 时的错误处理。"""
        data = pd.DataFrame({
            'id': [1, 1, 1, 1, 2, 2, 2, 2],
            'y': [10.0, 11.0, 12.0, 15.0, 5.0, 6.0, 7.0, 8.0],
            'd_': [1, 1, 1, 1, 0, 0, 0, 0],
            'post_': [0, 0, 0, 1, 0, 0, 0, 1],
            'tindex': [np.nan, np.nan, np.nan, 4.0, np.nan, np.nan, np.nan, 4.0],
            'quarter': [1, 2, 3, 4, 1, 2, 3, 4],
        })
        
        with pytest.raises(InsufficientPrePeriodsError) as excinfo:
            apply_rolling_transform(
                data=data, y='y', ivar='id', tindex='tindex', post='post_',
                rolling='demeanq', tpost1=4, quarter='quarter'
            )
        
        assert "All pre-treatment time index values are NaN" in str(excinfo.value)


class TestRegressionNormalData:
    """回归测试: 确保正常数据处理不受影响"""

    def test_demeanq_normal_data_unchanged(self):
        """验证 demeanq 对正常数据的处理结果与修复前一致。"""
        data = pd.DataFrame({
            'id': [1,1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2,2],
            'year': [1,1,1,1,1,2,2,2,2, 1,1,1,1,1,2,2,2,2],
            'quarter': [1,2,3,4,1,2,3,4,1, 1,2,3,4,1,2,3,4,1],
            'y': [10.0, 12.0, 11.0, 13.0, 10.0, 15.0, 17.0, 16.0, 18.0,
                  5.0, 7.0, 6.0, 8.0, 5.0, 5.0, 7.0, 6.0, 8.0],
            'd_': [1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0],
            'post_': [0,0,0,0,0,1,1,1,1, 0,0,0,0,0,1,1,1,1],
            'tindex': [1,2,3,4,5,6,7,8,9, 1,2,3,4,5,6,7,8,9],
        })
        
        # 应该正常执行，不抛出异常
        result = apply_rolling_transform(
            data=data, y='y', ivar='id', tindex='tindex', post='post_',
            rolling='demeanq', tpost1=6, quarter='quarter'
        )
        
        # 验证输出列存在
        assert 'ydot' in result.columns
        assert 'ydot_postavg' in result.columns
        assert 'firstpost' in result.columns
        
        # 验证没有 NaN（正常数据应该全部计算成功）
        assert result['ydot'].notna().all()

    def test_detrend_normal_data_unchanged(self):
        """验证 detrend 对正常数据的处理结果与修复前一致。"""
        data = pd.DataFrame({
            'id': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
            'tindex': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            'y': [5.0, 7.0, 9.0, 16.0, 18.0,  # Unit 1: trend + treatment
                  2.0, 3.0, 4.0, 5.0, 6.0,    # Unit 2: linear trend
                  5.5, 7.0, 8.5, 10.0, 11.5], # Unit 3: linear trend
            'd_': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'post_': [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1],
        })
        
        # 应该正常执行，不抛出异常
        result = apply_rolling_transform(
            data=data, y='y', ivar='id', tindex='tindex', post='post_',
            rolling='detrend', tpost1=4
        )
        
        # 验证输出列存在
        assert 'ydot' in result.columns
        assert 'ydot_postavg' in result.columns
        
        # 验证没有 NaN
        assert result['ydot'].notna().all()


class TestErrorMessageConsistencyWithStata:
    """Stata 行为一致性测试
    
    验证 Python 实现的错误处理行为与 Stata lwdid.ado 一致。
    
    Stata 行为 (lwdid.ado 第 187-192 行):
    - 当没有处理前观测时，显示 "No pre-treatment observations (post==0)."
    - 退出代码 2000
    
    Python 行为:
    - 抛出 InsufficientPrePeriodsError
    - 错误消息包含类似信息
    """

    def test_error_message_similar_to_stata(self):
        """验证错误消息风格与 Stata 一致。"""
        data = pd.DataFrame({
            'post': [1, 1, 1],
            'tindex': [1, 2, 3],
        })
        
        with pytest.raises(InsufficientPrePeriodsError) as excinfo:
            _compute_max_pre_tindex(data, 'post', 'tindex', 'demeanq')
        
        error_msg = str(excinfo.value)
        
        # 验证错误消息包含关键信息（类似 Stata）
        assert "pre-treatment" in error_msg.lower()
        assert "post==0" in error_msg or "post" in error_msg.lower()


class TestOriginalBugScenario:
    """BUG-060-M 原始场景复现测试
    
    原始问题:
    当所有处理前数据的 tindex 都是 NaN（因为被 dropna 过滤掉了），
    max() 返回 NaN，而 int(nan) 抛出不友好的
    "ValueError: cannot convert float NaN to integer"。
    
    修复后:
    应该抛出 InsufficientPrePeriodsError，消息清晰说明问题。
    """

    def test_original_bug_scenario_demeanq(self):
        """复现 BUG-060-M 原始场景 (demeanq)。"""
        # 模拟 dropna 后 tindex 全为 NaN 的场景
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'y': [10.0, 15.0, 5.0, 8.0],
            'd_': [1, 1, 0, 0],
            'post_': [0, 1, 0, 1],
            'tindex': [np.nan, 2.0, np.nan, 2.0],  # 处理前的 tindex 全为 NaN
            'quarter': [1, 2, 1, 2],
        })
        
        # 修复前会抛出: ValueError: cannot convert float NaN to integer
        # 修复后应该抛出: InsufficientPrePeriodsError
        with pytest.raises(InsufficientPrePeriodsError) as excinfo:
            apply_rolling_transform(
                data=data, y='y', ivar='id', tindex='tindex', post='post_',
                rolling='demeanq', tpost1=2, quarter='quarter'
            )
        
        # 验证不是 ValueError
        assert "All pre-treatment time index values are NaN" in str(excinfo.value)

    def test_original_bug_scenario_detrendq(self):
        """复现 BUG-060-M 原始场景 (detrendq)。"""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'y': [10.0, 15.0, 5.0, 8.0],
            'd_': [1, 1, 0, 0],
            'post_': [0, 1, 0, 1],
            'tindex': [np.nan, 2.0, np.nan, 2.0],
            'quarter': [1, 2, 1, 2],
        })
        
        with pytest.raises(InsufficientPrePeriodsError) as excinfo:
            apply_rolling_transform(
                data=data, y='y', ivar='id', tindex='tindex', post='post_',
                rolling='detrendq', tpost1=2, quarter='quarter'
            )
        
        assert "All pre-treatment time index values are NaN" in str(excinfo.value)

    def test_original_bug_scenario_detrend(self):
        """复现 BUG-060-M 原始场景 (detrend)。"""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'y': [10.0, 15.0, 5.0, 8.0],
            'd_': [1, 1, 0, 0],
            'post_': [0, 1, 0, 1],
            'tindex': [np.nan, 2.0, np.nan, 2.0],
        })
        
        with pytest.raises(InsufficientPrePeriodsError) as excinfo:
            apply_rolling_transform(
                data=data, y='y', ivar='id', tindex='tindex', post='post_',
                rolling='detrend', tpost1=2
            )
        
        assert "All pre-treatment time index values are NaN" in str(excinfo.value)

    def test_no_valueerror_for_nan_tindex(self):
        """确保不会抛出原始的 ValueError。
        
        这是 BUG-060-M 的核心验证点：确保 int(nan) 不会被调用。
        """
        data = pd.DataFrame({
            'post': [0, 0, 1],
            'tindex': [np.nan, np.nan, 3.0],
        })
        
        # 不应该抛出 ValueError
        try:
            _compute_max_pre_tindex(data, 'post', 'tindex', 'test')
            assert False, "Should have raised InsufficientPrePeriodsError"
        except InsufficientPrePeriodsError:
            pass  # 预期的异常
        except ValueError as e:
            if "cannot convert float NaN to integer" in str(e):
                pytest.fail(
                    "BUG-060-M not fixed: still getting "
                    "'ValueError: cannot convert float NaN to integer'"
                )
            raise
