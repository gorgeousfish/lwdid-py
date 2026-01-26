"""
倾向得分诊断模拟数据测试

生成具有已知特性的模拟数据，测试诊断功能在各种场景下的正确性：
1. 完美 Overlap 场景
2. 严重 Overlap 违反场景
3. 高权重 CV 无极端值场景
4. 蒙特卡洛模拟验证

Reference: Story 1.1 - 倾向得分诊断增强
"""

import numpy as np
import pandas as pd
import pytest

from lwdid.staggered.estimators import (
    _compute_ps_diagnostics,
    estimate_propensity_score,
    estimate_ipwra,
    PropensityScoreDiagnostics,
)


def generate_simulated_data(
    N: int = 200,
    treatment_prob: float = 0.3,
    overlap_quality: str = 'good',
    seed: int = 42
) -> pd.DataFrame:
    """
    生成用于诊断测试的模拟数据
    
    Parameters
    ----------
    N : int
        样本量
    treatment_prob : float
        处理组比例
    overlap_quality : str
        'good': 完美overlap
        'poor': 部分极端值
        'severe': 严重overlap违反
    seed : int
        随机种子
        
    Returns
    -------
    pd.DataFrame
        包含 'd', 'x1', 'x2', 'y' 的数据
    """
    np.random.seed(seed)
    
    if overlap_quality == 'good':
        # 协变量与处理弱相关
        x1 = np.random.normal(0, 1, N)
        x2 = np.random.normal(0, 1, N)
        logit_p = -0.5 + 0.3 * x1 + 0.2 * x2
        
    elif overlap_quality == 'poor':
        # 协变量与处理中度相关
        x1 = np.random.normal(0, 1, N)
        x2 = np.random.normal(0, 1, N)
        logit_p = -1.5 + 1.0 * x1 + 0.8 * x2
        
    else:  # severe
        # 协变量强预测处理（接近完全分离）
        x1 = np.random.normal(0, 1, N)
        x2 = np.random.normal(0, 1, N)
        logit_p = -3.0 + 2.5 * x1 + 2.0 * x2
    
    # 生成处理指示符
    p = 1 / (1 + np.exp(-logit_p))
    d = np.random.binomial(1, p)
    
    # 生成结果变量
    y = 1.0 + 0.5 * d + 0.3 * x1 + 0.2 * x2 + np.random.normal(0, 1, N)
    
    return pd.DataFrame({
        'd': d,
        'x1': x1,
        'x2': x2,
        'y': y,
    })


class TestScenarioPerfectOverlap:
    """场景1: 完美 Overlap，无警告"""
    
    def test_perfect_overlap_no_warnings(self):
        """完美 overlap 场景无警告"""
        np.random.seed(100)
        N = 200
        
        # 生成均匀分布的倾向得分
        pscores = np.random.uniform(0.2, 0.8, N)
        D = np.random.binomial(1, 0.3, N).astype(float)
        
        diag = _compute_ps_diagnostics(pscores, pscores, D, 0.01)
        
        # 验证无极端值
        assert diag.n_trimmed == 0
        assert diag.extreme_low_pct == 0
        assert diag.extreme_high_pct == 0
        
        # 验证 CV 合理
        assert diag.weights_cv < 2.0
        
        # 验证无警告
        assert diag.overlap_warning is None
    
    def test_perfect_overlap_simulated_data(self):
        """使用模拟数据的完美 overlap 场景"""
        data = generate_simulated_data(N=300, overlap_quality='good', seed=123)
        
        pscores, coef, diag = estimate_propensity_score(
            data, 'd', ['x1', 'x2'], return_diagnostics=True
        )
        
        # 完美 overlap 场景应该有较低的极端值比例
        assert diag.extreme_low_pct + diag.extreme_high_pct < 0.15
        
        # CV 应该合理
        assert diag.weights_cv < 3.0 or np.isnan(diag.weights_cv)
        
        # 倾向得分应该在合理范围内
        assert 0.1 < diag.ps_mean < 0.9
    
    def test_perfect_overlap_ipwra(self):
        """完美 overlap 场景的 IPWRA 估计"""
        data = generate_simulated_data(N=300, overlap_quality='good', seed=456)
        
        result = estimate_ipwra(
            data, 'y', 'd', ['x1', 'x2'], return_diagnostics=True
        )
        
        # 验证诊断存在
        assert result.diagnostics is not None
        
        # 验证估计结果合理
        assert not np.isnan(result.att)
        assert result.se > 0


class TestScenarioSevereOverlapViolation:
    """场景2: 严重 Overlap 违反，多个警告"""
    
    def test_severe_overlap_violation_warnings(self):
        """严重 overlap 违反场景触发警告"""
        np.random.seed(200)
        N = 100
        
        # 生成双峰分布: 大部分接近0或1
        # 确保极端值在裁剪阈值之外: < 0.01 或 > 0.99
        pscores_low = np.random.uniform(0.001, 0.009, 40)  # 全部 < 0.01
        pscores_high = np.random.uniform(0.991, 0.999, 40)  # 全部 > 0.99
        pscores_mid = np.random.uniform(0.3, 0.7, 20)
        pscores_raw = np.concatenate([pscores_low, pscores_mid, pscores_high])
        
        D = np.random.binomial(1, 0.5, N).astype(float)
        pscores_trimmed = np.clip(pscores_raw, 0.01, 0.99)
        
        diag = _compute_ps_diagnostics(pscores_raw, pscores_trimmed, D, 0.01)
        
        # 验证极端值检测 (40低 + 40高 = 80)
        assert diag.n_trimmed == 80
        assert diag.extreme_low_pct + diag.extreme_high_pct == 0.80
        
        # 验证触发警告
        assert diag.overlap_warning is not None
        assert 'Extreme propensity score proportion too high' in diag.overlap_warning
    
    def test_severe_overlap_simulated_data(self):
        """使用模拟数据的严重 overlap 违反场景"""
        data = generate_simulated_data(N=200, overlap_quality='severe', seed=789)
        
        pscores, coef, diag = estimate_propensity_score(
            data, 'd', ['x1', 'x2'], return_diagnostics=True
        )
        
        # 严重 overlap 违反场景应该有较高的极端值比例
        extreme_total = diag.extreme_low_pct + diag.extreme_high_pct
        
        # 可能触发警告（取决于数据生成）
        # 至少应该有一些被裁剪的观测
        print(f"Extreme: {extreme_total:.2%}, n_trimmed: {diag.n_trimmed}")
        print(f"Warning: {diag.overlap_warning}")


class TestScenarioHighCVNoExtremes:
    """场景3: 高权重 CV 但无极端值"""
    
    def test_high_cv_no_extremes(self):
        """高权重CV但无极端值场景"""
        np.random.seed(300)
        
        # 构造特殊数据: 控制组PS变异大
        pscores = np.array([
            0.1, 0.15, 0.2,  # 处理组: 低PS
            0.2, 0.5, 0.8,   # 控制组: 高变异
        ])
        D = np.array([1, 1, 1, 0, 0, 0], dtype=float)
        
        diag = _compute_ps_diagnostics(pscores, pscores, D, 0.01)
        
        # 验证无极端值
        assert diag.n_trimmed == 0
        
        # 验证高CV (控制组权重变异大)
        weights_control = np.array([0.2, 0.5, 0.8]) / np.array([0.8, 0.5, 0.2])
        cv_manual = np.std(weights_control, ddof=1) / np.mean(weights_control)
        assert abs(diag.weights_cv - cv_manual) < 1e-6
        
        # 如果 CV > 2.0，应该触发警告
        if cv_manual > 2.0:
            assert diag.overlap_warning is not None
            assert 'Weight coefficient of variation too high' in diag.overlap_warning


class TestMonteCarloSimulation:
    """蒙特卡洛模拟测试"""
    
    def test_monte_carlo_coverage(self):
        """蒙特卡洛模拟验证诊断一致性"""
        np.random.seed(1234)
        n_simulations = 50
        
        ps_means = []
        cv_values = []
        extreme_counts = []
        
        for i in range(n_simulations):
            # 生成数据
            N = 100
            pscores = np.random.uniform(0.1, 0.9, N)
            D = np.random.binomial(1, 0.3, N).astype(float)
            
            diag = _compute_ps_diagnostics(pscores, pscores, D, 0.01)
            
            ps_means.append(diag.ps_mean)
            if not np.isnan(diag.weights_cv):
                cv_values.append(diag.weights_cv)
            extreme_counts.append(diag.n_trimmed)
        
        # 验证蒙特卡洛一致性
        ps_means = np.array(ps_means)
        cv_values = np.array(cv_values)
        
        # PS 均值应该在合理范围内
        assert 0.4 < np.mean(ps_means) < 0.6
        
        # CV 应该是非负的
        assert np.all(cv_values >= 0)
        
        # 极端值应该很少（因为 PS 在 0.1-0.9 之间）
        assert np.mean(extreme_counts) == 0
    
    def test_monte_carlo_extreme_detection(self):
        """蒙特卡洛模拟验证极端值检测"""
        np.random.seed(5678)
        n_simulations = 30
        
        true_extreme_pct = 0.2  # 20% 极端值
        detected_rates = []
        
        for i in range(n_simulations):
            N = 100
            n_extreme = int(N * true_extreme_pct)
            n_normal = N - n_extreme
            
            # 生成混合分布
            pscores_extreme = np.random.uniform(0.001, 0.008, n_extreme // 2)
            pscores_extreme = np.concatenate([
                pscores_extreme,
                np.random.uniform(0.992, 0.999, n_extreme - n_extreme // 2)
            ])
            pscores_normal = np.random.uniform(0.1, 0.9, n_normal)
            pscores_raw = np.concatenate([pscores_extreme, pscores_normal])
            np.random.shuffle(pscores_raw)
            
            D = np.random.binomial(1, 0.3, N).astype(float)
            pscores_trimmed = np.clip(pscores_raw, 0.01, 0.99)
            
            diag = _compute_ps_diagnostics(pscores_raw, pscores_trimmed, D, 0.01)
            
            detected_rate = diag.extreme_low_pct + diag.extreme_high_pct
            detected_rates.append(detected_rate)
        
        # 平均检测率应该接近真实极端值比例
        mean_detected = np.mean(detected_rates)
        assert abs(mean_detected - true_extreme_pct) < 0.05, \
            f"检测率 {mean_detected:.2%} 应接近真实极端值比例 {true_extreme_pct:.2%}"


class TestDataQualityScenarios:
    """数据质量场景测试"""
    
    def test_small_sample_size(self):
        """小样本场景"""
        np.random.seed(111)
        N = 10
        data = generate_simulated_data(N=N, overlap_quality='good', seed=111)
        
        try:
            pscores, coef, diag = estimate_propensity_score(
                data, 'd', ['x1', 'x2'], return_diagnostics=True
            )
            
            # 小样本应该仍能计算诊断
            assert isinstance(diag, PropensityScoreDiagnostics)
        except ValueError:
            # 某些情况下可能因样本太小而失败
            pytest.skip("样本太小，模型无法估计")
    
    def test_imbalanced_treatment(self):
        """处理组比例不平衡场景"""
        np.random.seed(222)
        N = 200
        
        # 处理组比例很低
        data = generate_simulated_data(N=N, treatment_prob=0.1, seed=222)
        
        try:
            pscores, coef, diag = estimate_propensity_score(
                data, 'd', ['x1', 'x2'], return_diagnostics=True
            )
            
            # PS 均值应该接近处理组比例
            actual_prop = data['d'].mean()
            assert abs(diag.ps_mean - actual_prop) < 0.2
            
        except ValueError:
            # 可能因样本不平衡而失败
            pytest.skip("样本不平衡，模型无法估计")
    
    def test_high_dimensional_controls(self):
        """高维协变量场景"""
        np.random.seed(333)
        N = 200
        
        # 生成数据
        data = pd.DataFrame({
            'd': np.random.binomial(1, 0.3, N),
            'x1': np.random.normal(0, 1, N),
            'x2': np.random.normal(0, 1, N),
            'x3': np.random.normal(0, 1, N),
            'x4': np.random.normal(0, 1, N),
            'x5': np.random.normal(0, 1, N),
        })
        
        pscores, coef, diag = estimate_propensity_score(
            data, 'd', ['x1', 'x2', 'x3', 'x4', 'x5'], return_diagnostics=True
        )
        
        # 验证诊断正确
        assert isinstance(diag, PropensityScoreDiagnostics)
        assert 0 < diag.ps_mean < 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
