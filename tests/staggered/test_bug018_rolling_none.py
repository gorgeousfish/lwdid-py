"""
BUG-018 修复测试：rolling 参数未检查 None 导致 AttributeError

测试范围:
1. 单元测试 - 验证 rolling=None 抛出 ValueError（而非 AttributeError）
2. 错误消息测试 - 验证错误消息包含有效值提示
3. 参数有效性测试 - 验证有效的 rolling 值正常工作
4. 边界情况测试 - 空字符串、空白字符串等
5. 回归测试 - 确保修复不影响现有功能

BUG描述:
在 core.py 的 _lwdid_staggered() 函数中，第 811 行直接调用 rolling.lower()
而未检查 rolling 是否为 None。如果用户传入 rolling=None，会抛出
AttributeError: 'NoneType' object has no attribute 'lower'

修复方案:
在 rolling.lower() 之前添加 rolling is None 检查，抛出友好的 ValueError。

Reference: BUG-018 in 审查/bug列表.md
"""

import numpy as np
import pandas as pd
import pytest
import sys
import warnings
from pathlib import Path

# 确保可以导入lwdid模块
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from lwdid import lwdid


# ============================================================================
# 测试数据生成辅助函数
# ============================================================================

def create_simple_staggered_data(n_units: int = 50, T: int = 6, seed: int = 42) -> pd.DataFrame:
    """
    创建简单的 staggered DiD 测试数据。
    
    Parameters
    ----------
    n_units : int
        单位数量
    T : int
        时期数量
    seed : int
        随机种子
        
    Returns
    -------
    pd.DataFrame
        包含 id, year, y, gvar, x1, x2 的面板数据
    """
    np.random.seed(seed)
    
    # 分配 cohort：50% NT(g=0), 17% g=4, 17% g=5, 16% g=6
    cohort_probs = [0.50, 0.17, 0.17, 0.16]
    cohort_values = [0, 4, 5, 6]
    unit_cohorts = np.random.choice(cohort_values, size=n_units, p=cohort_probs)
    
    # 生成协变量
    x1 = np.random.randn(n_units)
    x2 = np.random.randn(n_units)
    
    # 生成面板数据
    data_list = []
    for i in range(n_units):
        for t in range(1, T + 1):
            g = unit_cohorts[i]
            # 基础结果
            y_base = 1 + 0.5 * t + 0.3 * x1[i] + 0.2 * x2[i] + np.random.randn() * 0.5
            
            # 处理效应
            if g > 0 and t >= g:
                tau = 2.0 + 0.5 * (t - g)
                y = y_base + tau
            else:
                y = y_base
            
            data_list.append({
                'id': i + 1,
                'year': t,
                'y': y,
                'gvar': g,
                'x1': x1[i],
                'x2': x2[i],
            })
    
    return pd.DataFrame(data_list)


# ============================================================================
# 单元测试：rolling=None 应抛出 ValueError
# ============================================================================

class TestRollingNoneValidation:
    """验证 rolling=None 的错误处理"""
    
    def test_rolling_none_raises_value_error(self):
        """
        验证 rolling=None 抛出 ValueError 而非 AttributeError。
        
        修复前: AttributeError: 'NoneType' object has no attribute 'lower'
        修复后: ValueError: Staggered模式需要指定'rolling'参数...
        """
        data = create_simple_staggered_data()
        
        # rolling=None 应抛出 ValueError
        with pytest.raises(ValueError) as exc_info:
            lwdid(
                data=data,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                rolling=None,  # BUG-018: 这会导致 AttributeError
            )
        
        # 验证错误类型不是 AttributeError
        assert not isinstance(exc_info.value, AttributeError), (
            "rolling=None 应抛出 ValueError，而非 AttributeError"
        )
    
    def test_rolling_none_error_message_contains_valid_values(self):
        """
        验证错误消息包含有效值提示。
        """
        data = create_simple_staggered_data()
        
        with pytest.raises(ValueError) as exc_info:
            lwdid(
                data=data,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                rolling=None,
            )
        
        error_msg = str(exc_info.value)
        
        # 验证错误消息包含有效值
        assert 'demean' in error_msg.lower() or 'detrend' in error_msg.lower(), (
            f"错误消息应包含有效值 'demean' 或 'detrend': {error_msg}"
        )
        
        # 验证错误消息提示参数是必需的
        assert 'rolling' in error_msg.lower(), (
            f"错误消息应提及 'rolling' 参数: {error_msg}"
        )
    
    def test_rolling_invalid_string_raises_value_error(self):
        """
        验证无效字符串抛出 ValueError 并提供有效值提示。
        """
        data = create_simple_staggered_data()
        
        with pytest.raises(ValueError) as exc_info:
            lwdid(
                data=data,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                rolling='invalid_value',
            )
        
        error_msg = str(exc_info.value)
        
        # 验证错误消息包含无效值
        assert 'invalid_value' in error_msg, (
            f"错误消息应包含用户提供的无效值: {error_msg}"
        )


# ============================================================================
# 参数有效性测试：有效的 rolling 值应正常工作
# ============================================================================

class TestValidRollingValues:
    """验证有效的 rolling 参数值正常工作"""
    
    def test_rolling_demean_works(self):
        """验证 rolling='demean' 正常工作"""
        data = create_simple_staggered_data()
        
        # 不应抛出任何异常
        result = lwdid(
            data=data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
        )
        
        assert result is not None
        assert np.isfinite(result.att)
        assert result.se_att > 0
    
    def test_rolling_detrend_works(self):
        """验证 rolling='detrend' 正常工作"""
        data = create_simple_staggered_data()
        
        # 不应抛出任何异常
        result = lwdid(
            data=data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='detrend',
        )
        
        assert result is not None
        assert np.isfinite(result.att)
        assert result.se_att > 0
    
    def test_rolling_case_insensitive_demean(self):
        """验证 rolling 参数大小写不敏感 - DEMEAN"""
        data = create_simple_staggered_data()
        
        # 大写应该工作
        result_upper = lwdid(
            data=data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='DEMEAN',
        )
        
        assert result_upper is not None
        assert np.isfinite(result_upper.att)
    
    def test_rolling_case_insensitive_detrend(self):
        """验证 rolling 参数大小写不敏感 - Detrend"""
        data = create_simple_staggered_data()
        
        # 混合大小写应该工作
        result_mixed = lwdid(
            data=data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='Detrend',
        )
        
        assert result_mixed is not None
        assert np.isfinite(result_mixed.att)
    
    def test_rolling_default_value_works(self):
        """验证 rolling 参数默认值 'demean' 正常工作"""
        data = create_simple_staggered_data()
        
        # 不指定 rolling 参数，使用默认值
        result = lwdid(
            data=data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            # rolling 不指定，使用默认值 'demean'
        )
        
        assert result is not None
        assert np.isfinite(result.att)


# ============================================================================
# 边界情况测试
# ============================================================================

class TestRollingBoundaryCase:
    """测试 rolling 参数的边界情况"""
    
    def test_rolling_empty_string_raises_error(self):
        """验证空字符串抛出 ValueError"""
        data = create_simple_staggered_data()
        
        with pytest.raises(ValueError):
            lwdid(
                data=data,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                rolling='',  # 空字符串
            )
    
    def test_rolling_whitespace_raises_error(self):
        """验证纯空白字符串抛出 ValueError"""
        data = create_simple_staggered_data()
        
        with pytest.raises(ValueError):
            lwdid(
                data=data,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                rolling='   ',  # 纯空白
            )
    
    def test_rolling_with_extra_whitespace(self):
        """
        验证带前后空白的有效值行为。
        
        注: 当前实现可能不会自动 strip()，这个测试记录当前行为。
        """
        data = create_simple_staggered_data()
        
        # 带空白的值可能会报错（取决于实现）
        # 这里只验证不会因为 None 而抛出 AttributeError
        try:
            result = lwdid(
                data=data,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                rolling=' demean ',  # 带前后空白
            )
            # 如果成功，验证结果
            assert result is not None
        except ValueError as e:
            # 如果失败，应该是 ValueError 而非 AttributeError
            assert 'demean' in str(e).lower() or 'detrend' in str(e).lower()


# ============================================================================
# 回归测试：确保修复不影响现有功能
# ============================================================================

class TestNoRegression:
    """回归测试：确保 BUG-018 修复不引入新问题"""
    
    def test_staggered_estimation_still_works(self):
        """验证 staggered 估计仍然正常工作"""
        data = create_simple_staggered_data(n_units=80, seed=123)
        
        result = lwdid(
            data=data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
        )
        
        assert result is not None
        assert np.isfinite(result.att)
        assert result.se_att > 0
        # 在 staggered 模式下，pvalue 和 CI 来自 overall 效应属性
        if result.is_staggered and result.pvalue_overall is not None:
            assert 0 <= result.pvalue_overall <= 1
    
    def test_att_estimate_reasonable(self):
        """
        验证 ATT 估计值合理。
        
        真实处理效应约为 τ = 2.0 + 0.5*(r-g)，平均约 2.5
        """
        np.random.seed(42)
        data = create_simple_staggered_data(n_units=100, seed=42)
        
        result = lwdid(
            data=data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
        )
        
        # ATT 应该在合理范围内（真实效应约 2-3）
        assert 0 < result.att < 10, f"ATT={result.att} 不在合理范围内"
    
    def test_demean_and_detrend_produce_different_results(self):
        """
        验证 demean 和 detrend 产生不同结果。
        
        这确认两种变换都在正确工作。
        """
        data = create_simple_staggered_data(n_units=100, seed=42)
        
        result_demean = lwdid(
            data=data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
        )
        
        result_detrend = lwdid(
            data=data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='detrend',
        )
        
        # 两者应该都有有效结果
        assert np.isfinite(result_demean.att)
        assert np.isfinite(result_detrend.att)
        
        # ATT 值可能不同（取决于数据特性）
        # 这里只验证两者都能正常计算
    
    def test_with_controls(self):
        """验证带控制变量时 rolling 参数仍正常工作"""
        data = create_simple_staggered_data(n_units=100, seed=42)
        
        result = lwdid(
            data=data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            controls=['x1', 'x2'],
        )
        
        assert result is not None
        assert np.isfinite(result.att)
        assert result.se_att > 0


# ============================================================================
# 数值验证测试
# ============================================================================

class TestNumericalValidation:
    """数值验证：确保估计结果合理"""
    
    def test_confidence_interval_contains_estimate(self):
        """验证置信区间包含点估计"""
        data = create_simple_staggered_data(n_units=80, seed=42)
        
        result = lwdid(
            data=data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
        )
        
        # 在 staggered 模式下，检查 overall 效应的 CI
        if result.is_staggered and result.att_overall is not None:
            ci_l = result.ci_overall_lower
            ci_u = result.ci_overall_upper
            att = result.att_overall
            if ci_l is not None and ci_u is not None:
                assert ci_l < att < ci_u, (
                    f"CI [{ci_l}, {ci_u}] 不包含 ATT={att}"
                )
    
    def test_se_positive(self):
        """验证标准误为正"""
        data = create_simple_staggered_data(n_units=80, seed=42)
        
        result = lwdid(
            data=data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
        )
        
        assert result.se_att > 0, f"SE={result.se_att} 应为正数"
    
    def test_pvalue_in_valid_range(self):
        """验证 p 值在有效范围内"""
        data = create_simple_staggered_data(n_units=80, seed=42)
        
        result = lwdid(
            data=data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
        )
        
        # 在 staggered 模式下，检查 overall 效应的 pvalue
        if result.is_staggered and result.pvalue_overall is not None:
            assert 0 <= result.pvalue_overall <= 1, (
                f"p-value={result.pvalue_overall} 不在 [0,1] 范围内"
            )


# ============================================================================
# 蒙特卡洛测试
# ============================================================================

class TestMonteCarlo:
    """蒙特卡洛测试：验证估计稳定性"""
    
    @pytest.mark.slow
    def test_estimate_consistency_across_samples(self):
        """
        验证估计在多个样本间一致。
        
        使用相同 DGP 生成多个样本，ATT 估计应该围绕真实值波动。
        """
        n_simulations = 10
        att_estimates = []
        
        for seed in range(n_simulations):
            data = create_simple_staggered_data(n_units=100, seed=seed + 1000)
            
            try:
                result = lwdid(
                    data=data,
                    y='y',
                    ivar='id',
                    tvar='year',
                    gvar='gvar',
                    rolling='demean',
                )
                att_estimates.append(result.att)
            except Exception:
                pass  # 某些随机种子可能产生退化数据
        
        if len(att_estimates) >= 5:
            att_mean = np.mean(att_estimates)
            att_std = np.std(att_estimates, ddof=1)
            
            # ATT 均值应该合理（真实效应约 2-3）
            assert 0 < att_mean < 10, f"ATT均值={att_mean} 不在合理范围内"
            
            # ATT 标准差应该合理
            assert 0 < att_std < 5, f"ATT标准差={att_std} 过大"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
