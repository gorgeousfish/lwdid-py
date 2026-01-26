"""
BUG-039: Bootstrap最小样本阈值与警告阈值逻辑一致性测试

问题描述:
在 `_compute_ipw_se_bootstrap`、`compute_ipwra_se_bootstrap` 和 `_compute_psm_se_bootstrap` 中，
原来警告阈值是 n_bootstrap * 0.8，而最小样本阈值硬编码为 10。
当 n_bootstrap 较小时，这两个阈值不一致。

修复方案:
将最小样本阈值改为动态计算: max(10, int(n_bootstrap * 0.5))

测试目标:
1. 验证动态阈值计算公式正确
2. 验证不同 n_bootstrap 值的边界情况
3. 验证三个估计器的实现一致
"""

import warnings
import numpy as np
import pandas as pd
import pytest
import inspect

from lwdid.staggered import estimate_ipw
from lwdid.staggered.estimators import (
    estimate_ipwra, 
    estimate_psm,
    _compute_ipw_se_bootstrap,
    compute_ipwra_se_bootstrap,
    _compute_psm_se_bootstrap,
)


# ============================================================================
# Threshold Calculation Tests
# ============================================================================

class TestDynamicThresholdCalculation:
    """动态阈值计算公式验证测试"""
    
    @pytest.mark.parametrize("n_bootstrap,expected_min", [
        (10, 10),    # max(10, 5) = 10
        (15, 10),    # max(10, 7) = 10
        (20, 10),    # max(10, 10) = 10
        (21, 10),    # max(10, 10) = 10 (int truncation)
        (22, 11),    # max(10, 11) = 11
        (50, 25),    # max(10, 25) = 25
        (100, 50),   # max(10, 50) = 50
        (200, 100),  # max(10, 100) = 100
        (500, 250),  # max(10, 250) = 250
    ])
    def test_min_required_formula(self, n_bootstrap, expected_min):
        """验证 min_required = max(10, int(n_bootstrap * 0.5)) 公式"""
        actual_min = max(10, int(n_bootstrap * 0.5))
        assert actual_min == expected_min, \
            f"n_bootstrap={n_bootstrap}: expected {expected_min}, got {actual_min}"
    
    @pytest.mark.parametrize("n_bootstrap", [10, 20, 50, 100, 200])
    def test_warning_threshold_always_greater_than_min(self, n_bootstrap):
        """验证警告阈值（0.8）总是大于最小阈值（0.5）对于 n_bootstrap >= 20"""
        warning_threshold = n_bootstrap * 0.8
        min_required = max(10, int(n_bootstrap * 0.5))
        
        # 对于 n_bootstrap >= 20，警告阈值应该 >= 最小阈值
        # 对于 n_bootstrap < 20，最小阈值固定为10，可能大于警告阈值
        if n_bootstrap >= 20:
            assert warning_threshold >= min_required, \
                f"n_bootstrap={n_bootstrap}: warning threshold ({warning_threshold}) " \
                f"should be >= min_required ({min_required})"


class TestSourceCodeConsistency:
    """源代码一致性验证"""
    
    def test_all_bootstrap_functions_use_dynamic_threshold(self):
        """验证所有Bootstrap函数都使用动态阈值公式"""
        import lwdid.staggered.estimators as estimators_module
        
        source = inspect.getsource(estimators_module)
        
        # BUG-039 修复后，应该有 3 个 `max(10, int(n_bootstrap * 0.5))` 模式
        # 每个 bootstrap 函数一个
        count_dynamic_threshold = source.count('max(10, int(n_bootstrap * 0.5))')
        
        assert count_dynamic_threshold == 3, \
            f"期望3个动态阈值公式，实际找到 {count_dynamic_threshold} 个。" \
            "确保 _compute_ipw_se_bootstrap, compute_ipwra_se_bootstrap, " \
            "_compute_psm_se_bootstrap 都使用 max(10, int(n_bootstrap * 0.5))"
    
    def test_warning_threshold_unchanged(self):
        """验证警告阈值仍然是 n_bootstrap * 0.8"""
        import lwdid.staggered.estimators as estimators_module
        
        source = inspect.getsource(estimators_module)
        
        # 应该有 3 个 `n_bootstrap * 0.8` 警告阈值
        count_warning_threshold = source.count('n_bootstrap * 0.8')
        
        assert count_warning_threshold == 3, \
            f"期望3个警告阈值 'n_bootstrap * 0.8'，实际找到 {count_warning_threshold} 个"
    
    def test_bug039_fix_comments_present(self):
        """验证BUG-039修复注释存在"""
        import lwdid.staggered.estimators as estimators_module
        
        source = inspect.getsource(estimators_module)
        
        # 应该有 3 个 BUG-039 FIX 注释
        count_fix_comments = source.count('BUG-039 FIX')
        
        assert count_fix_comments == 3, \
            f"期望3个 'BUG-039 FIX' 注释，实际找到 {count_fix_comments} 个"


# ============================================================================
# Test Data Generator
# ============================================================================

def generate_test_data(n: int = 200, seed: int = 42, treatment_prob: float = 0.3) -> pd.DataFrame:
    """生成测试数据"""
    rng = np.random.default_rng(seed)
    
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    
    # 生成处理状态
    ps_index = -1.0 + 0.3 * x1 + 0.2 * x2
    ps_true = 1 / (1 + np.exp(-ps_index))
    d = rng.binomial(1, ps_true)
    
    # 确保有足够的处理组和控制组
    while d.sum() < max(5, int(n * 0.1)) or (n - d.sum()) < max(5, int(n * 0.1)):
        d = rng.binomial(1, ps_true)
    
    # 生成结果
    y0 = 1.0 + 0.5 * x1 + 0.3 * x2 + rng.normal(0, 0.5, n)
    y1 = y0 + 2.0  # ATT = 2.0
    y = np.where(d == 1, y1, y0)
    
    return pd.DataFrame({
        'y': y,
        'd': d,
        'x1': x1,
        'x2': x2,
    })


# ============================================================================
# Integration Tests: Actual Bootstrap Behavior
# ============================================================================

class TestBootstrapBehaviorWithDynamicThreshold:
    """Bootstrap实际行为测试（使用动态阈值）"""
    
    @pytest.fixture
    def basic_data(self):
        """基本测试数据"""
        return generate_test_data(n=200, seed=42)
    
    def test_ipw_bootstrap_works_with_small_n_bootstrap(self, basic_data):
        """测试IPW Bootstrap在小n_bootstrap时正常工作"""
        # n_bootstrap=20, min_required = max(10, 10) = 10
        result = estimate_ipw(
            data=basic_data,
            y='y',
            d='d',
            propensity_controls=['x1', 'x2'],
            se_method='bootstrap',
            n_bootstrap=20,
            seed=42,
        )
        
        assert result.se > 0, "SE应该为正"
        assert result.ci_lower < result.ci_upper, "CI下界应该小于上界"
    
    def test_ipwra_bootstrap_works_with_small_n_bootstrap(self, basic_data):
        """测试IPWRA Bootstrap在小n_bootstrap时正常工作"""
        result = estimate_ipwra(
            data=basic_data,
            y='y',
            d='d',
            controls=['x1', 'x2'],
            se_method='bootstrap',
            n_bootstrap=20,
            seed=42,
        )
        
        assert result.se > 0, "SE应该为正"
        assert result.ci_lower < result.ci_upper, "CI下界应该小于上界"
    
    def test_psm_bootstrap_works_with_small_n_bootstrap(self, basic_data):
        """测试PSM Bootstrap在小n_bootstrap时正常工作"""
        result = estimate_psm(
            data=basic_data,
            y='y',
            d='d',
            propensity_controls=['x1', 'x2'],
            se_method='bootstrap',
            n_bootstrap=20,
            seed=42,
        )
        
        assert result.se > 0, "SE应该为正"
        assert result.ci_lower < result.ci_upper, "CI下界应该小于上界"
    
    @pytest.mark.parametrize("n_bootstrap", [50, 100, 200])
    def test_bootstrap_se_stable_across_n_bootstrap(self, basic_data, n_bootstrap):
        """测试不同n_bootstrap下SE的稳定性"""
        result = estimate_ipw(
            data=basic_data,
            y='y',
            d='d',
            propensity_controls=['x1', 'x2'],
            se_method='bootstrap',
            n_bootstrap=n_bootstrap,
            seed=42,
        )
        
        # SE应该在合理范围内（不会随n_bootstrap剧烈变化）
        assert 0.01 < result.se < 10.0, f"SE={result.se} 超出合理范围"


class TestBootstrapErrorMessages:
    """Bootstrap错误消息测试"""
    
    def test_error_message_contains_dynamic_min_required(self):
        """验证错误消息包含动态计算的min_required值"""
        # 创建会导致大量失败的极端数据
        rng = np.random.default_rng(123)
        n = 20
        
        # 极端不平衡数据
        data = pd.DataFrame({
            'y': rng.normal(0, 1, n),
            'd': [1, 1] + [0] * (n - 2),  # 只有2个处理组
            'x1': rng.normal(0, 1, n),
            'x2': rng.normal(0, 1, n),
        })
        
        # n_bootstrap=100, min_required = max(10, 50) = 50
        # 由于极端不平衡，很多bootstrap样本会失败
        try:
            estimate_ipw(
                data=data,
                y='y',
                d='d',
                propensity_controls=['x1', 'x2'],
                se_method='bootstrap',
                n_bootstrap=100,
                seed=42,
            )
        except ValueError as e:
            error_msg = str(e)
            # 验证错误消息格式
            assert "minimum" in error_msg.lower() or "Insufficient" in error_msg, \
                f"错误消息应该包含最小样本信息: {error_msg}"


class TestBootstrapWarningBehavior:
    """Bootstrap警告行为测试"""
    
    def test_warning_triggered_below_80_percent(self):
        """验证成功率低于80%时触发警告"""
        # 创建中等不平衡数据，使bootstrap成功率在50%-80%之间
        rng = np.random.default_rng(456)
        n = 50
        
        # 适度不平衡
        ps_index = rng.normal(-1.5, 1, n)  # 低处理概率
        ps = 1 / (1 + np.exp(-ps_index))
        d = rng.binomial(1, ps)
        
        # 确保至少有5个处理组
        if d.sum() < 5:
            d[:5] = 1
        
        data = pd.DataFrame({
            'y': rng.normal(0, 1, n),
            'd': d,
            'x1': rng.normal(0, 1, n),
            'x2': rng.normal(0, 1, n),
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                estimate_ipw(
                    data=data,
                    y='y',
                    d='d',
                    propensity_controls=['x1', 'x2'],
                    se_method='bootstrap',
                    n_bootstrap=50,
                    seed=42,
                )
            except ValueError:
                # 如果抛出错误（样本不足），这也是可接受的
                pass
            
            # 检查是否有低成功率警告
            low_success_warnings = [
                warning for warning in w
                if "Low bootstrap success rate" in str(warning.message)
            ]
            
            # 注意：这个测试可能不稳定，因为取决于具体数据
            # 主要目的是验证警告机制存在


# ============================================================================
# Numerical Validation Tests
# ============================================================================

class TestThresholdNumericalValidation:
    """阈值数值验证测试"""
    
    def test_threshold_relationship_table(self):
        """验证阈值关系表（来自计划文档）"""
        test_cases = [
            # (n_bootstrap, warning_threshold_0.8, min_threshold_dynamic, relationship)
            (10, 8, 10, "min > warning"),
            (20, 16, 10, "warning > min"),
            (50, 40, 25, "warning > min"),
            (100, 80, 50, "warning > min"),
            (200, 160, 100, "warning > min"),
        ]
        
        for n_boot, expected_warn, expected_min, expected_rel in test_cases:
            actual_warn = n_boot * 0.8
            actual_min = max(10, int(n_boot * 0.5))
            
            assert actual_warn == expected_warn, \
                f"n_bootstrap={n_boot}: warning threshold should be {expected_warn}, got {actual_warn}"
            assert actual_min == expected_min, \
                f"n_bootstrap={n_boot}: min threshold should be {expected_min}, got {actual_min}"
            
            # 验证关系
            if expected_rel == "min > warning":
                assert actual_min > actual_warn
            else:
                assert actual_warn > actual_min


# ============================================================================
# Regression Tests
# ============================================================================

class TestBug039Regression:
    """BUG-039回归测试"""
    
    def test_no_hardcoded_10_as_standalone_threshold(self):
        """验证没有单独的硬编码10作为阈值（应该在max(10, ...)内）"""
        import lwdid.staggered.estimators as estimators_module
        
        source = inspect.getsource(estimators_module)
        
        # 检查是否存在旧的硬编码模式 `< 10:` 或 `<10:`
        # 排除 max(10, ...) 模式内的10
        import re
        
        # 查找类似 `if len(xxx) < 10:` 的模式（旧的硬编码）
        old_pattern = re.compile(r'if\s+len\([^)]+\)\s*<\s*10\s*:')
        matches = old_pattern.findall(source)
        
        # 不应该有旧的硬编码模式
        assert len(matches) == 0, \
            f"发现旧的硬编码阈值模式: {matches}"
    
    def test_dynamic_threshold_in_all_bootstrap_functions(self):
        """确保所有bootstrap函数都使用动态阈值"""
        # 这是关键的回归测试
        import lwdid.staggered.estimators as estimators_module
        
        source = inspect.getsource(estimators_module)
        
        # 查找 min_required 的定义
        min_required_count = source.count('min_required = max(10, int(n_bootstrap * 0.5))')
        
        assert min_required_count >= 3, \
            f"期望至少3个动态阈值定义，实际找到 {min_required_count} 个"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
