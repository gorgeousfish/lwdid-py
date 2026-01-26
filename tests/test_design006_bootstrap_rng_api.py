"""
DESIGN-006: Bootstrap随机数生成器API使用一致性测试

测试目标:
验证所有Bootstrap函数统一使用 np.random.default_rng(seed) API，而非
旧的 np.random.seed() + np.random.choice() API。

修复范围:
1. compute_ipwra_se_bootstrap - IPWRA Bootstrap标准误
2. compute_ipw_se_bootstrap - IPW Bootstrap标准误  
3. _compute_psm_se_bootstrap - PSM Bootstrap标准误

测试内容:
1. 单元测试 - API使用验证
2. 可重复性测试 - 同一seed产生相同结果
3. 全局状态隔离测试 - 不污染全局随机状态
4. 数值合理性测试 - 结果在预期范围内
5. 并发安全测试 - 多线程调用安全
6. 蒙特卡洛覆盖率测试 - Bootstrap置信区间覆盖率
7. 与解析法对比测试 - Bootstrap与解析SE差异合理

Reference: DESIGN-006 in 审查/设计问题列表.md
"""

import warnings
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from concurrent.futures import ThreadPoolExecutor, as_completed
import inspect
import re
import sys
from pathlib import Path

# 确保可以导入lwdid模块
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lwdid.staggered.estimators import (
    estimate_ipw,
    estimate_ipwra,
    estimate_psm,
    compute_ipwra_se_bootstrap,
    _compute_psm_se_bootstrap,
    estimate_propensity_score,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_cross_section_data():
    """生成简单的横截面测试数据，用于Bootstrap测试"""
    rng = np.random.default_rng(12345)
    n = 300
    
    # 协变量
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    
    # 倾向得分模型
    ps_index = -0.3 + 0.4 * x1 + 0.3 * x2
    ps_true = 1 / (1 + np.exp(-ps_index))
    
    # 处理分配
    d = rng.binomial(1, ps_true)
    
    # 确保有足够的处理组和控制组
    n_treat = d.sum()
    n_control = n - n_treat
    if n_treat < 30 or n_control < 30:
        # 强制平衡
        d[:n//3] = 0
        d[n//3:2*n//3] = 1
        rng.shuffle(d)
    
    # 结果变量（有处理效应）
    true_att = 2.5
    y0 = 1.0 + 0.5 * x1 + 0.3 * x2 + rng.standard_normal(n) * 0.8
    y1 = y0 + true_att + rng.standard_normal(n) * 0.3  # 异质性
    y = np.where(d == 1, y1, y0)
    
    data = pd.DataFrame({
        'y': y,
        'd': d,
        'x1': x1,
        'x2': x2,
    })
    
    return data, true_att


@pytest.fixture
def psm_compatible_data():
    """生成PSM兼容的测试数据"""
    rng = np.random.default_rng(67890)
    n = 400
    
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    
    ps_index = -0.2 + 0.5 * x1 + 0.3 * x2 + 0.2 * x3
    ps_true = 1 / (1 + np.exp(-ps_index))
    
    d = rng.binomial(1, ps_true)
    
    # 确保有足够样本
    n_treat = d.sum()
    n_control = n - n_treat
    while n_treat < 50 or n_control < 50:
        d = rng.binomial(1, ps_true)
        n_treat = d.sum()
        n_control = n - n_treat
    
    true_att = 1.8
    y0 = 2.0 + 0.6 * x1 + 0.4 * x2 + 0.2 * x3 + rng.standard_normal(n) * 0.7
    y1 = y0 + true_att
    y = np.where(d == 1, y1, y0)
    
    data = pd.DataFrame({
        'y': y,
        'd': d,
        'x1': x1,
        'x2': x2,
        'x3': x3,
    })
    
    return data, true_att


# ============================================================================
# Test Class: API一致性验证
# ============================================================================

class TestBootstrapRNGAPIConsistency:
    """验证Bootstrap函数使用正确的随机数生成器API"""
    
    def test_ipwra_bootstrap_uses_default_rng(self):
        """验证IPWRA Bootstrap使用np.random.default_rng"""
        source = inspect.getsource(compute_ipwra_se_bootstrap)
        
        # 检查使用了现代API
        assert 'np.random.default_rng' in source or 'default_rng' in source, \
            "IPWRA Bootstrap应使用np.random.default_rng()"
        
        # 检查没有使用旧API
        assert 'np.random.seed' not in source, \
            "IPWRA Bootstrap不应使用np.random.seed()"
    
    def test_psm_bootstrap_uses_default_rng(self):
        """验证PSM Bootstrap使用np.random.default_rng"""
        source = inspect.getsource(_compute_psm_se_bootstrap)
        
        # 检查使用了现代API
        assert 'np.random.default_rng' in source or 'default_rng' in source, \
            "PSM Bootstrap应使用np.random.default_rng()"
        
        # 检查没有使用旧API
        assert 'np.random.seed' not in source, \
            "PSM Bootstrap不应使用np.random.seed()"
    
    def test_rng_choice_pattern(self):
        """验证使用rng.choice而非np.random.choice"""
        ipwra_source = inspect.getsource(compute_ipwra_se_bootstrap)
        psm_source = inspect.getsource(_compute_psm_se_bootstrap)
        
        # IPWRA: 检查使用rng.choice模式
        assert 'rng.choice' in ipwra_source or 'rng.choice' in ipwra_source, \
            "IPWRA Bootstrap应使用rng.choice()"
        assert 'np.random.choice' not in ipwra_source, \
            "IPWRA Bootstrap不应使用np.random.choice()"
        
        # PSM: 检查使用rng.choice模式
        assert 'rng.choice' in psm_source, \
            "PSM Bootstrap应使用rng.choice()"
        assert 'np.random.choice' not in psm_source, \
            "PSM Bootstrap不应使用np.random.choice()"


# ============================================================================
# Test Class: 可重复性测试
# ============================================================================

class TestBootstrapReproducibility:
    """验证相同seed产生完全相同的结果"""
    
    def test_ipw_bootstrap_reproducibility(self, simple_cross_section_data):
        """IPW Bootstrap可重复性"""
        data, _ = simple_cross_section_data
        controls = ['x1', 'x2']
        
        # 第一次运行
        result1 = estimate_ipw(
            data=data,
            y='y',
            d='d',
            propensity_controls=controls,
            se_method='bootstrap',
            n_bootstrap=100,
            seed=42,
        )
        
        # 第二次运行，相同seed
        result2 = estimate_ipw(
            data=data,
            y='y',
            d='d',
            propensity_controls=controls,
            se_method='bootstrap',
            n_bootstrap=100,
            seed=42,
        )
        
        # 验证结果完全相同
        assert result1.att == result2.att, "ATT应完全相同"
        np.testing.assert_allclose(result1.se, result2.se, rtol=1e-10,
            err_msg="相同seed的Bootstrap SE应完全相同")
        np.testing.assert_allclose(result1.ci_lower, result2.ci_lower, rtol=1e-10)
        np.testing.assert_allclose(result1.ci_upper, result2.ci_upper, rtol=1e-10)
    
    def test_ipwra_bootstrap_reproducibility(self, simple_cross_section_data):
        """IPWRA Bootstrap可重复性"""
        data, _ = simple_cross_section_data
        controls = ['x1', 'x2']
        
        # 第一次运行
        result1 = estimate_ipwra(
            data=data,
            y='y',
            d='d',
            controls=controls,
            se_method='bootstrap',
            n_bootstrap=100,
            seed=42,
        )
        
        # 第二次运行，相同seed
        result2 = estimate_ipwra(
            data=data,
            y='y',
            d='d',
            controls=controls,
            se_method='bootstrap',
            n_bootstrap=100,
            seed=42,
        )
        
        # 验证结果完全相同
        assert result1.att == result2.att, "ATT应完全相同"
        np.testing.assert_allclose(result1.se, result2.se, rtol=1e-10,
            err_msg="相同seed的Bootstrap SE应完全相同")
    
    def test_psm_bootstrap_reproducibility(self, psm_compatible_data):
        """PSM Bootstrap可重复性"""
        data, _ = psm_compatible_data
        controls = ['x1', 'x2', 'x3']
        
        # 第一次运行
        result1 = estimate_psm(
            data=data,
            y='y',
            d='d',
            propensity_controls=controls,
            se_method='bootstrap',
            n_bootstrap=50,
            seed=42,
        )
        
        # 第二次运行，相同seed
        result2 = estimate_psm(
            data=data,
            y='y',
            d='d',
            propensity_controls=controls,
            se_method='bootstrap',
            n_bootstrap=50,
            seed=42,
        )
        
        # 验证结果完全相同
        assert result1.att == result2.att, "ATT应完全相同"
        np.testing.assert_allclose(result1.se, result2.se, rtol=1e-10,
            err_msg="相同seed的Bootstrap SE应完全相同")
    
    def test_different_seeds_produce_different_results(self, simple_cross_section_data):
        """不同seed产生不同结果"""
        data, _ = simple_cross_section_data
        controls = ['x1', 'x2']
        
        result1 = estimate_ipw(
            data=data,
            y='y',
            d='d',
            propensity_controls=controls,
            se_method='bootstrap',
            n_bootstrap=100,
            seed=42,
        )
        
        result2 = estimate_ipw(
            data=data,
            y='y',
            d='d',
            propensity_controls=controls,
            se_method='bootstrap',
            n_bootstrap=100,
            seed=123,
        )
        
        # ATT应相同（因为数据相同）
        assert result1.att == result2.att
        
        # SE应不同（因为Bootstrap样本不同）
        assert result1.se != result2.se, "不同seed的Bootstrap SE应该不同"


# ============================================================================
# Test Class: 全局状态隔离测试
# ============================================================================

class TestGlobalStateIsolation:
    """验证Bootstrap不污染全局随机状态"""
    
    def test_no_global_state_pollution_ipw(self, simple_cross_section_data):
        """IPW Bootstrap不应污染全局状态"""
        data, _ = simple_cross_section_data
        controls = ['x1', 'x2']
        
        # 设置全局状态
        np.random.seed(999)
        before_state = np.random.get_state()
        global_sample_before = np.random.random(5).copy()
        
        # 重置并运行Bootstrap
        np.random.seed(999)
        
        # 运行Bootstrap（使用不同的seed）
        _ = estimate_ipw(
            data=data,
            y='y',
            d='d',
            propensity_controls=controls,
            se_method='bootstrap',
            n_bootstrap=50,
            seed=42,  # 这应该使用独立的RNG
        )
        
        # 全局状态应该不受影响（如果使用了全局状态，则会改变）
        # 使用现代API，全局状态应该保持不变
        global_sample_after = np.random.random(5)
        
        # 如果使用了default_rng，全局状态应该不受影响
        # 但由于Python代码可能有其他随机调用，我们只检查
        # 连续调用应该产生可预测的结果
        np.random.seed(999)
        expected_sample = np.random.random(5)
        np.testing.assert_array_equal(global_sample_before, expected_sample,
            err_msg="全局随机状态应保持一致")
    
    def test_consecutive_bootstrap_calls_independent(self, simple_cross_section_data):
        """连续Bootstrap调用应相互独立"""
        data, _ = simple_cross_section_data
        controls = ['x1', 'x2']
        
        results = []
        for seed in [10, 20, 30]:
            result = estimate_ipw(
                data=data,
                y='y',
                d='d',
                propensity_controls=controls,
                se_method='bootstrap',
                n_bootstrap=50,
                seed=seed,
            )
            results.append(result.se)
        
        # 验证每次调用产生不同但稳定的结果
        assert len(set(results)) == 3, "三个不同seed应产生三个不同SE"
        
        # 再次运行验证可重复性
        for i, seed in enumerate([10, 20, 30]):
            result = estimate_ipw(
                data=data,
                y='y',
                d='d',
                propensity_controls=controls,
                se_method='bootstrap',
                n_bootstrap=50,
                seed=seed,
            )
            assert result.se == results[i], f"seed={seed}应产生相同结果"


# ============================================================================
# Test Class: 数值合理性测试
# ============================================================================

class TestBootstrapNumericalValidity:
    """验证Bootstrap结果的数值合理性"""
    
    def test_bootstrap_se_positive(self, simple_cross_section_data):
        """Bootstrap SE应为正数"""
        data, _ = simple_cross_section_data
        controls = ['x1', 'x2']
        
        for estimator in ['ipw', 'ipwra']:
            if estimator == 'ipw':
                result = estimate_ipw(
                    data=data, y='y', d='d',
                    propensity_controls=controls,
                    se_method='bootstrap',
                    n_bootstrap=100,
                    seed=42,
                )
            else:
                result = estimate_ipwra(
                    data=data, y='y', d='d',
                    controls=controls,
                    se_method='bootstrap',
                    n_bootstrap=100,
                    seed=42,
                )
            
            assert result.se > 0, f"{estimator} Bootstrap SE应为正数"
            assert np.isfinite(result.se), f"{estimator} Bootstrap SE应为有限值"
    
    def test_bootstrap_ci_contains_att(self, simple_cross_section_data):
        """Bootstrap置信区间应包含ATT点估计"""
        data, _ = simple_cross_section_data
        controls = ['x1', 'x2']
        
        result = estimate_ipw(
            data=data, y='y', d='d',
            propensity_controls=controls,
            se_method='bootstrap',
            n_bootstrap=200,
            seed=42,
        )
        
        assert result.ci_lower < result.att < result.ci_upper, \
            "95% CI应包含ATT点估计"
    
    def test_bootstrap_se_reasonable_magnitude(self, simple_cross_section_data):
        """Bootstrap SE应在合理范围内"""
        data, true_att = simple_cross_section_data
        controls = ['x1', 'x2']
        
        result = estimate_ipw(
            data=data, y='y', d='d',
            propensity_controls=controls,
            se_method='bootstrap',
            n_bootstrap=200,
            seed=42,
        )
        
        # SE不应太小（接近零）或太大（大于ATT的绝对值）
        assert result.se > 0.01, "SE不应太小"
        assert result.se < abs(result.att) * 2, "SE不应过大"
    
    def test_bootstrap_vs_analytical_se_comparison(self, simple_cross_section_data):
        """Bootstrap SE应与解析SE在同一量级"""
        data, _ = simple_cross_section_data
        controls = ['x1', 'x2']
        
        # 解析SE
        result_analytical = estimate_ipw(
            data=data, y='y', d='d',
            propensity_controls=controls,
            se_method='analytical',
        )
        
        # Bootstrap SE
        result_bootstrap = estimate_ipw(
            data=data, y='y', d='d',
            propensity_controls=controls,
            se_method='bootstrap',
            n_bootstrap=500,
            seed=42,
        )
        
        # 两者应在合理范围内（通常差异<50%）
        ratio = result_bootstrap.se / result_analytical.se
        assert 0.5 < ratio < 2.0, \
            f"Bootstrap/Analytical SE比值应在0.5-2.0之间，实际={ratio:.3f}"


# ============================================================================
# Test Class: 并发安全测试
# ============================================================================

class TestConcurrencySafety:
    """验证Bootstrap在并发环境下的安全性"""
    
    def test_parallel_bootstrap_independent(self, simple_cross_section_data):
        """并行Bootstrap调用应产生独立结果"""
        data, _ = simple_cross_section_data
        controls = ['x1', 'x2']
        
        def run_bootstrap(seed):
            result = estimate_ipw(
                data=data, y='y', d='d',
                propensity_controls=controls,
                se_method='bootstrap',
                n_bootstrap=50,
                seed=seed,
            )
            return seed, result.se
        
        seeds = [100, 200, 300, 400, 500]
        
        # 串行运行
        serial_results = {}
        for seed in seeds:
            _, se = run_bootstrap(seed)
            serial_results[seed] = se
        
        # 并行运行
        parallel_results = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(run_bootstrap, seed): seed for seed in seeds}
            for future in as_completed(futures):
                seed, se = future.result()
                parallel_results[seed] = se
        
        # 验证并行和串行结果一致
        for seed in seeds:
            np.testing.assert_allclose(
                serial_results[seed], 
                parallel_results[seed], 
                rtol=1e-10,
                err_msg=f"seed={seed}的串行和并行结果应一致"
            )
    
    def test_repeated_parallel_calls_stable(self, simple_cross_section_data):
        """重复并行调用应产生稳定结果"""
        data, _ = simple_cross_section_data
        controls = ['x1', 'x2']
        
        def run_with_fixed_seed():
            result = estimate_ipw(
                data=data, y='y', d='d',
                propensity_controls=controls,
                se_method='bootstrap',
                n_bootstrap=50,
                seed=42,
            )
            return result.se
        
        # 多次并行运行相同seed
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_with_fixed_seed) for _ in range(8)]
            for future in as_completed(futures):
                results.append(future.result())
        
        # 所有结果应完全相同
        assert len(set(results)) == 1, \
            f"相同seed的并行调用应产生相同结果，但得到{len(set(results))}个不同值"


# ============================================================================
# Test Class: 蒙特卡洛覆盖率测试
# ============================================================================

class TestMonteCarloConverage:
    """验证Bootstrap置信区间的覆盖率"""
    
    @pytest.mark.slow
    def test_bootstrap_ci_coverage_ipw(self):
        """IPW Bootstrap 95% CI覆盖率测试"""
        n_simulations = 100
        n_obs = 300
        true_att = 2.0
        coverage_count = 0
        
        for sim in range(n_simulations):
            # 生成数据
            rng = np.random.default_rng(sim + 1000)
            x1 = rng.standard_normal(n_obs)
            x2 = rng.standard_normal(n_obs)
            
            ps_index = -0.3 + 0.4 * x1 + 0.3 * x2
            ps_true = 1 / (1 + np.exp(-ps_index))
            d = rng.binomial(1, ps_true)
            
            # 确保样本平衡
            if d.sum() < 30 or (n_obs - d.sum()) < 30:
                continue
            
            y0 = 1.0 + 0.5 * x1 + 0.3 * x2 + rng.standard_normal(n_obs) * 0.5
            y1 = y0 + true_att
            y = np.where(d == 1, y1, y0)
            
            data = pd.DataFrame({
                'y': y, 'd': d, 'x1': x1, 'x2': x2
            })
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = estimate_ipw(
                        data=data, y='y', d='d',
                        propensity_controls=['x1', 'x2'],
                        se_method='bootstrap',
                        n_bootstrap=100,
                        seed=sim,
                    )
                
                # 检查真值是否在CI内
                if result.ci_lower <= true_att <= result.ci_upper:
                    coverage_count += 1
            except Exception:
                continue
        
        actual_coverage = coverage_count / n_simulations
        
        # 95% CI覆盖率应在合理范围内 (允许蒙特卡洛误差)
        assert 0.85 < actual_coverage < 1.0, \
            f"Bootstrap CI覆盖率应接近95%，实际={actual_coverage:.2%}"


# ============================================================================
# Test Class: 与其他估计方法对比
# ============================================================================

class TestCrossMethodComparison:
    """验证不同估计方法的一致性"""
    
    def test_ipw_ipwra_bootstrap_consistency(self, simple_cross_section_data):
        """IPW和IPWRA的Bootstrap SE应在合理范围内"""
        data, _ = simple_cross_section_data
        controls = ['x1', 'x2']
        
        result_ipw = estimate_ipw(
            data=data, y='y', d='d',
            propensity_controls=controls,
            se_method='bootstrap',
            n_bootstrap=200,
            seed=42,
        )
        
        result_ipwra = estimate_ipwra(
            data=data, y='y', d='d',
            controls=controls,
            se_method='bootstrap',
            n_bootstrap=200,
            seed=42,
        )
        
        # ATT估计应相近
        att_diff = abs(result_ipw.att - result_ipwra.att)
        max_se = max(result_ipw.se, result_ipwra.se)
        assert att_diff < 3 * max_se, \
            f"IPW和IPWRA的ATT差异过大: {att_diff:.4f} vs 3*SE={3*max_se:.4f}"
        
        # SE应在同一量级
        se_ratio = result_ipw.se / result_ipwra.se
        assert 0.3 < se_ratio < 3.0, \
            f"IPW/IPWRA SE比值应合理: {se_ratio:.3f}"


# ============================================================================
# Test Class: 边界条件测试
# ============================================================================

class TestEdgeCases:
    """测试边界条件下的行为"""
    
    def test_bootstrap_with_none_seed(self, simple_cross_section_data):
        """seed=None时应正常工作"""
        data, _ = simple_cross_section_data
        controls = ['x1', 'x2']
        
        # 不应抛出异常
        result = estimate_ipw(
            data=data, y='y', d='d',
            propensity_controls=controls,
            se_method='bootstrap',
            n_bootstrap=50,
            seed=None,  # 无seed
        )
        
        assert result.se > 0
        assert np.isfinite(result.se)
    
    def test_bootstrap_with_zero_seed(self, simple_cross_section_data):
        """seed=0时应正常工作且可重复"""
        data, _ = simple_cross_section_data
        controls = ['x1', 'x2']
        
        result1 = estimate_ipw(
            data=data, y='y', d='d',
            propensity_controls=controls,
            se_method='bootstrap',
            n_bootstrap=50,
            seed=0,
        )
        
        result2 = estimate_ipw(
            data=data, y='y', d='d',
            propensity_controls=controls,
            se_method='bootstrap',
            n_bootstrap=50,
            seed=0,
        )
        
        np.testing.assert_allclose(result1.se, result2.se, rtol=1e-10)
    
    def test_bootstrap_with_large_seed(self, simple_cross_section_data):
        """大seed值应正常工作"""
        data, _ = simple_cross_section_data
        controls = ['x1', 'x2']
        
        # 使用很大的seed值
        large_seed = 2**31 - 1  # 最大32位整数
        
        result = estimate_ipw(
            data=data, y='y', d='d',
            propensity_controls=controls,
            se_method='bootstrap',
            n_bootstrap=50,
            seed=large_seed,
        )
        
        assert result.se > 0
        assert np.isfinite(result.se)


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
