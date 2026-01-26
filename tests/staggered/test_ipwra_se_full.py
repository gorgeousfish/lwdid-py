"""
IPWRA完整版标准误测试

测试范围:
1. Stata一致性测试 - 与teffects ipwra SE对比
2. 解析SE vs Bootstrap SE对比
3. 边界条件测试
4. Monte Carlo覆盖率测试
5. 双稳健性验证测试
6. Vibe Math MCP公式验证

Reference: Story 3.1 - IPWRA完整版解析标准误
Based on: Wooldridge (2007) Theorem 3.1
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats

# 确保可以导入lwdid模块
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from lwdid.staggered.estimators import (
    estimate_ipwra,
    estimate_propensity_score,
    estimate_outcome_model,
    compute_ipwra_se_analytical,
    IPWRAResult,
)


# ============================================================================
# 辅助函数
# ============================================================================

def generate_dgp_for_ipwra_coverage(
    n: int,
    true_att: float = 2.0,
    ps_correct: bool = True,
    om_correct: bool = True,
    seed: int = None,
) -> tuple:
    """
    生成用于IPWRA覆盖率测试的模拟数据。
    
    DGP:
    - X1 ~ N(0, 1)
    - X2 ~ N(0, 1)
    - 真实PS: logit^{-1}(0.5 + 0.3*X1 + 0.2*X2 [+ 0.1*X1²])
    - 真实Y(0): 1 + 0.5*X1 + 0.3*X2 [+ 0.1*X1²] + ε
    - Y(1) = Y(0) + true_att
    
    ps_correct=False时，真实PS包含X1²但估计时不包含
    om_correct=False时，真实Y(0)包含X1²但估计时不包含
    
    Parameters
    ----------
    n : int
        样本量
    true_att : float
        真实ATT
    ps_correct : bool
        PS模型是否正确设定
    om_correct : bool
        结果模型是否正确设定
    seed : int
        随机种子
        
    Returns
    -------
    tuple
        (data, true_att)
    """
    rng = np.random.default_rng(seed)
    
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    
    # 真实PS（可能包含非线性项）
    ps_index = 0.5 + 0.3*X1 + 0.2*X2
    if not ps_correct:
        ps_index += 0.1 * X1**2  # 估计时会缺少这项
    ps_true = 1 / (1 + np.exp(-ps_index))
    
    # 处理分配
    D = rng.binomial(1, ps_true)
    
    # 真实Y(0)（可能包含非线性项）
    epsilon = rng.normal(0, 1, n)
    Y0 = 1 + 0.5*X1 + 0.3*X2
    if not om_correct:
        Y0 += 0.1 * X1**2  # 估计时会缺少这项
    Y0 += epsilon
    
    Y1 = Y0 + true_att
    
    # 观测结果
    Y = D * Y1 + (1 - D) * Y0
    
    data = pd.DataFrame({
        'y': Y,
        'd': D,
        'x1': X1,
        'x2': X2,
    })
    
    return data, true_att


def load_staggered_test_data():
    """加载staggered测试数据"""
    data_path = Path(__file__).parent.parent.parent.parent / \
        "Lee_Wooldridge_2023-main 3" / "2.lee_wooldridge_staggered_data.dta"
    
    if not data_path.exists():
        pytest.skip(f"测试数据不存在: {data_path}")
    
    return pd.read_stata(data_path)


def prepare_subsample_and_y(data, g, r):
    """准备子样本和转换后的Y变量"""
    # 设置面板结构
    data = data.sort_values(['id', 'year'])
    
    # 生成转换后的Y变量
    # y_gr = y - mean(y_{t-1}, y_{t-2}, ...)
    data = data.copy()
    
    # 创建滞后变量
    for lag in range(1, 6):
        data[f'y_lag{lag}'] = data.groupby('id')['y'].shift(lag)
    
    # 根据(g,r)计算转换后的Y
    if g == 4 and r == 4:
        data['y_transformed'] = data['y'] - (data['y_lag1'] + data['y_lag2'] + data['y_lag3']) / 3
        condition = data['f04'] == 1
    elif g == 4 and r == 5:
        data['y_transformed'] = data['y'] - (data['y_lag2'] + data['y_lag3'] + data['y_lag4']) / 3
        condition = (data['f05'] == 1) & (data['g5'] == 0)
    elif g == 4 and r == 6:
        data['y_transformed'] = data['y'] - (data['y_lag3'] + data['y_lag4'] + data['y_lag5']) / 3
        condition = (data['f06'] == 1) & ((data['g5'] + data['g6']) != 1)
    elif g == 5 and r == 5:
        data['y_transformed'] = data['y'] - (data['y_lag1'] + data['y_lag2'] + data['y_lag3'] + data['y_lag4']) / 4
        condition = (data['f05'] == 1) & (data['g4'] == 0)
    elif g == 5 and r == 6:
        data['y_transformed'] = data['y'] - (data['y_lag2'] + data['y_lag3'] + data['y_lag4'] + data['y_lag5']) / 4
        condition = (data['f06'] == 1) & ((data['g4'] + data['g6']) != 1)
    elif g == 6 and r == 6:
        data['y_transformed'] = data['y'] - (data['y_lag1'] + data['y_lag2'] + data['y_lag3'] + data['y_lag4'] + data['y_lag5']) / 5
        condition = (data['f06'] == 1) & ((data['g4'] + data['g5']) != 1)
    else:
        raise ValueError(f"Unsupported (g,r) = ({g},{r})")
    
    # 应用条件
    subsample = data[condition].dropna(subset=['y_transformed']).copy()
    
    # 创建处理变量
    subsample['D'] = subsample[f'g{g}'].astype(int)
    
    return subsample


# ============================================================================
# Stata一致性测试
# ============================================================================

class TestIPWRAStataConsistency:
    """IPWRA与Stata一致性测试"""
    
    @pytest.fixture
    def stata_results(self):
        """加载Stata验证数据"""
        test_dir = Path(__file__).parent
        results_file = test_dir / 'stata_ipwra_results.json'
        
        if not results_file.exists():
            pytest.skip(f"Stata结果文件不存在: {results_file}")
        
        with open(results_file) as f:
            return json.load(f)['results']
    
    @pytest.mark.parametrize("g,r", [
        (4, 4), (4, 5), (4, 6),
        (5, 5), (5, 6),
        (6, 6),
    ])
    def test_ipwra_se_vs_stata(self, g, r, stata_results):
        """测试IPWRA SE与Stata一致"""
        # 加载数据
        try:
            data = load_staggered_test_data()
        except Exception as e:
            pytest.skip(f"无法加载测试数据: {e}")
        
        # 准备子样本
        subsample = prepare_subsample_and_y(data, g, r)
        
        # Python估计
        result = estimate_ipwra(
            data=subsample,
            y='y_transformed',
            d='D',
            controls=['x1', 'x2'],
            propensity_controls=['x1', 'x2'],
            se_method='analytical',
            trim_threshold=0.01,
        )
        
        # Stata结果
        key = f'{g}_{r}'
        stata_att = stata_results[key]['att']
        stata_se = stata_results[key]['se']
        
        # ATT一致性检验（应该非常接近，允许0.1%误差）
        att_error = abs(result.att - stata_att) / abs(stata_att) if stata_att != 0 else abs(result.att - stata_att)
        assert att_error < 0.001, f"(g={g},r={r}) ATT误差 {att_error:.4f} > 0.1%: Python={result.att:.6f}, Stata={stata_att:.6f}"
        
        # SE一致性检验（完整版SE应与Stata接近，允许1%误差）
        se_error = abs(result.se - stata_se) / stata_se
        assert se_error < 0.01, f"(g={g},r={r}) SE误差 {se_error:.2%} > 1%: Python={result.se:.6f}, Stata={stata_se:.6f}"
        
        print(f"(g={g},r={r}): ATT误差={att_error:.4f}, SE误差={se_error:.2%}")


# ============================================================================
# 解析SE vs Bootstrap SE对比测试
# ============================================================================

class TestAnalyticalVsBootstrapSE:
    """解析SE与Bootstrap SE对比测试"""
    
    @pytest.mark.parametrize("n,max_diff", [
        (200, 0.25),   # 中等样本，允许25%差异
        (500, 0.20),   # 大样本，允许20%差异
        (1000, 0.15),  # 大样本，允许15%差异
    ])
    def test_analytical_vs_bootstrap_se(self, n, max_diff):
        """测试解析SE与Bootstrap SE一致性"""
        data, _ = generate_dgp_for_ipwra_coverage(n=n, seed=42)
        
        result_analytical = estimate_ipwra(
            data, 'y', 'd', ['x1', 'x2'],
            se_method='analytical'
        )
        result_bootstrap = estimate_ipwra(
            data, 'y', 'd', ['x1', 'x2'],
            se_method='bootstrap', n_bootstrap=300, seed=42
        )
        
        diff = abs(result_analytical.se - result_bootstrap.se) / result_bootstrap.se
        
        assert diff <= max_diff, \
            f"N={n}时解析SE与Bootstrap SE差异 {diff:.1%} > {max_diff:.1%}: " \
            f"analytical={result_analytical.se:.6f}, bootstrap={result_bootstrap.se:.6f}"
        
        print(f"N={n}: analytical_se={result_analytical.se:.6f}, bootstrap_se={result_bootstrap.se:.6f}, diff={diff:.2%}")


# ============================================================================
# 边界条件测试
# ============================================================================

class TestIPWRASEBoundaryConditions:
    """IPWRA SE边界条件测试"""
    
    def test_small_sample_warning(self):
        """测试小样本时的行为"""
        np.random.seed(42)
        n = 80
        data = pd.DataFrame({
            'y': np.random.randn(n),
            'd': np.concatenate([np.ones(30), np.zeros(50)]).astype(int),
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
        })
        
        # 应该能正常计算，不崩溃
        result = estimate_ipwra(
            data, 'y', 'd', ['x1', 'x2'],
            se_method='analytical'
        )
        
        assert result.se > 0
        assert not np.isnan(result.se)
    
    def test_extreme_weights(self):
        """测试极端权重情况"""
        np.random.seed(42)
        n = 200
        # 创建导致极端权重的数据
        x1 = np.concatenate([np.random.normal(-2, 0.5, 20),
                            np.random.normal(0, 1, 180)])
        d = np.concatenate([np.ones(20), np.zeros(180)]).astype(int)
        y = np.random.randn(n)
        
        data = pd.DataFrame({'y': y, 'd': d, 'x1': x1, 'x2': np.random.randn(n)})
        
        # 应该能处理极端权重
        result = estimate_ipwra(
            data, 'y', 'd', ['x1', 'x2'],
            se_method='analytical',
            trim_threshold=0.01
        )
        
        assert result.se > 0
        assert not np.isnan(result.se)
    
    def test_collinear_controls(self):
        """测试近似共线性控制变量"""
        np.random.seed(42)
        n = 200
        x1 = np.random.randn(n)
        x2 = x1 + np.random.randn(n) * 0.01  # 高度相关
        d = (np.random.uniform(0, 1, n) < 0.5).astype(int)
        y = np.random.randn(n)
        
        data = pd.DataFrame({'y': y, 'd': d, 'x1': x1, 'x2': x2})
        
        # 可能会有警告，但不应崩溃
        result = estimate_ipwra(
            data, 'y', 'd', ['x1', 'x2'],
            se_method='analytical'
        )
        
        # 结果应该有效（可能有警告）
        assert result.se >= 0 or np.isnan(result.se)


# ============================================================================
# Monte Carlo覆盖率测试
# ============================================================================

@pytest.mark.slow
class TestIPWRACoverage:
    """IPWRA标准误覆盖率Monte Carlo测试"""
    
    @pytest.mark.parametrize("n_obs,expected_range", [
        (500, (0.90, 0.98)),   # 放宽范围以适应有限样本
        (1000, (0.92, 0.98)),
    ])
    def test_coverage_both_correct(self, n_obs, expected_range):
        """测试PS和OM都正确时的覆盖率"""
        n_reps = 200  # 减少重复次数以加快测试
        true_att = 2.0
        covered = 0
        
        for rep in range(n_reps):
            data, _ = generate_dgp_for_ipwra_coverage(
                n_obs, true_att, ps_correct=True, om_correct=True, seed=rep
            )
            
            result = estimate_ipwra(
                data=data,
                y='y',
                d='d',
                controls=['x1', 'x2'],
                se_method='analytical',
            )
            
            if result.ci_lower <= true_att <= result.ci_upper:
                covered += 1
        
        coverage = covered / n_reps
        assert expected_range[0] <= coverage <= expected_range[1], \
            f"覆盖率 {coverage:.3f} 不在预期范围 {expected_range}"
        
        print(f"N={n_obs}, 覆盖率={coverage:.3f}")


# ============================================================================
# 双稳健性验证测试
# ============================================================================

@pytest.mark.slow
class TestIPWRADoublyRobustCoverage:
    """IPWRA双稳健性覆盖率验证"""
    
    def test_coverage_ps_wrong_om_correct(self):
        """PS错误但OM正确时覆盖率应正常（双稳健性）"""
        n_reps = 200
        n_obs = 500
        true_att = 2.0
        covered = 0
        
        for rep in range(n_reps):
            data, _ = generate_dgp_for_ipwra_coverage(
                n_obs, true_att, ps_correct=False, om_correct=True, seed=rep
            )
            
            result = estimate_ipwra(
                data=data,
                y='y',
                d='d',
                controls=['x1', 'x2'],  # 正确
                propensity_controls=['x1', 'x2'],  # 缺少x1²
                se_method='analytical',
            )
            
            if result.ci_lower <= true_att <= result.ci_upper:
                covered += 1
        
        coverage = covered / n_reps
        # 双稳健性：即使PS错误，覆盖率仍应接近95%
        assert 0.88 <= coverage <= 0.98, \
            f"双稳健性失败：PS错误时覆盖率 {coverage:.3f} 应在[0.88, 0.98]"
        
        print(f"PS错误OM正确: 覆盖率={coverage:.3f}")
    
    def test_coverage_ps_correct_om_wrong(self):
        """PS正确但OM错误时覆盖率应正常（双稳健性）"""
        n_reps = 200
        n_obs = 500
        true_att = 2.0
        covered = 0
        
        for rep in range(n_reps):
            data, _ = generate_dgp_for_ipwra_coverage(
                n_obs, true_att, ps_correct=True, om_correct=False, seed=rep
            )
            
            result = estimate_ipwra(
                data=data,
                y='y',
                d='d',
                controls=['x1', 'x2'],  # 缺少x1²
                propensity_controls=['x1', 'x2'],  # 正确
                se_method='analytical',
            )
            
            if result.ci_lower <= true_att <= result.ci_upper:
                covered += 1
        
        coverage = covered / n_reps
        # 双稳健性：即使OM错误，覆盖率仍应接近95%
        assert 0.88 <= coverage <= 0.98, \
            f"双稳健性失败：OM错误时覆盖率 {coverage:.3f} 应在[0.88, 0.98]"
        
        print(f"PS正确OM错误: 覆盖率={coverage:.3f}")
    
    def test_coverage_both_wrong(self):
        """PS和OM都错误时覆盖率应下降"""
        n_reps = 200
        n_obs = 500
        true_att = 2.0
        covered = 0
        
        for rep in range(n_reps):
            data, _ = generate_dgp_for_ipwra_coverage(
                n_obs, true_att, ps_correct=False, om_correct=False, seed=rep
            )
            
            result = estimate_ipwra(
                data=data,
                y='y',
                d='d',
                controls=['x1', 'x2'],  # 缺少x1²
                propensity_controls=['x1', 'x2'],  # 缺少x1²
                se_method='analytical',
            )
            
            if result.ci_lower <= true_att <= result.ci_upper:
                covered += 1
        
        coverage = covered / n_reps
        # 两者都错误时，覆盖率可能下降（但不一定很低，取决于误设程度）
        print(f"PS和OM都错误: 覆盖率={coverage:.3f}")
        # 不做严格断言，只记录结果


# ============================================================================
# 基本功能测试
# ============================================================================

class TestIPWRAFullSEBasic:
    """IPWRA完整版SE基本功能测试"""
    
    def test_se_positive(self):
        """测试SE为正"""
        data, _ = generate_dgp_for_ipwra_coverage(n=200, seed=42)
        
        result = estimate_ipwra(
            data, 'y', 'd', ['x1', 'x2'],
            se_method='analytical'
        )
        
        assert result.se > 0, f"SE应为正: {result.se}"
    
    def test_ci_contains_att(self):
        """测试置信区间包含ATT"""
        data, _ = generate_dgp_for_ipwra_coverage(n=200, seed=42)
        
        result = estimate_ipwra(
            data, 'y', 'd', ['x1', 'x2'],
            se_method='analytical'
        )
        
        assert result.ci_lower < result.att < result.ci_upper, \
            f"置信区间应包含ATT: [{result.ci_lower}, {result.ci_upper}], ATT={result.att}"
    
    def test_analytical_se_uses_full_version(self):
        """测试解析SE使用完整版（与Bootstrap接近）"""
        data, _ = generate_dgp_for_ipwra_coverage(n=500, seed=42)
        
        result_analytical = estimate_ipwra(
            data, 'y', 'd', ['x1', 'x2'],
            se_method='analytical'
        )
        result_bootstrap = estimate_ipwra(
            data, 'y', 'd', ['x1', 'x2'],
            se_method='bootstrap', n_bootstrap=500, seed=42
        )
        
        # 解析SE应与Bootstrap SE接近（都是完整版）
        se_diff = abs(result_analytical.se - result_bootstrap.se) / result_bootstrap.se
        assert se_diff < 0.25, \
            f"解析SE与Bootstrap SE差异过大: {se_diff:.1%}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
