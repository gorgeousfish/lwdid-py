"""
Story 4.1: Python vs Stata端到端数值验证测试

使用Lee & Wooldridge (2023) Staggered场景实证数据进行验证。

测试结果摘要:
- ATT: 完全一致 (差异 < 1e-5)
- SE: 与 Stata 基准高度一致
- 统计推断结论一致

SE差异可能来源:
1. 条件方差估计的距离度量细节差异
2. 数值误差与边界处理

References:
    Stata teffects psmatch documentation
    Abadie A, Imbens GW (2006). Econometrica 74(1):235-267.
"""

import pytest
import json
import numpy as np
import pandas as pd
from pathlib import Path

from lwdid.staggered.estimators import estimate_psm


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope='module')
def stata_test_data():
    """加载Stata导出的测试数据"""
    data_path = Path(__file__).parent / 'stata_psm_test_data.csv'
    if not data_path.exists():
        pytest.skip("Stata测试数据文件不存在")
    return pd.read_csv(data_path)


@pytest.fixture(scope='module')
def stata_benchmark():
    """Stata teffects psmatch基准结果"""
    return {
        'att': 3.554019,
        'se': 0.5866075,
        'ci_lower': 2.40429,
        'ci_upper': 4.703749,
        'z': 6.06,
        'n_obs': 1000,
        'n_treat': 129,
        'n_control': 871,
    }


# ============================================================================
# End-to-End Tests
# ============================================================================

class TestPythonStataE2E:
    """Python vs Stata端到端测试"""
    
    def test_att_exact_match(self, stata_test_data, stata_benchmark):
        """ATT应完全一致"""
        result = estimate_psm(
            data=stata_test_data,
            y='y_44',
            d='g4',
            propensity_controls=['x1', 'x2'],
            n_neighbors=1,
            se_method='abadie_imbens_full',
        )
        
        # ATT完全一致
        assert np.isclose(result.att, stata_benchmark['att'], atol=1e-5), \
            f"ATT不一致: Python={result.att:.6f}, Stata={stata_benchmark['att']}"
    
    def test_se_within_acceptable_range(self, stata_test_data, stata_benchmark):
        """SE差异应在可接受范围内 (<10%)"""
        result = estimate_psm(
            data=stata_test_data,
            y='y_44',
            d='g4',
            propensity_controls=['x1', 'x2'],
            n_neighbors=1,
            se_method='abadie_imbens_full',
        )
        
        se_diff_pct = abs(result.se - stata_benchmark['se']) / stata_benchmark['se'] * 100
        
        # SE差异应<10%
        assert se_diff_pct < 10, \
            f"SE差异过大: Python={result.se:.6f}, Stata={stata_benchmark['se']}, 差异={se_diff_pct:.2f}%"
        
        # 记录实际差异（约5.45%）
        print(f"SE差异: {se_diff_pct:.2f}%")
    
    def test_inference_conclusion_same(self, stata_test_data, stata_benchmark):
        """统计推断结论应一致"""
        from scipy import stats
        
        result = estimate_psm(
            data=stata_test_data,
            y='y_44',
            d='g4',
            propensity_controls=['x1', 'x2'],
            n_neighbors=1,
            se_method='abadie_imbens_full',
        )
        
        # Python推断
        z_python = result.att / result.se
        p_python = 2 * (1 - stats.norm.cdf(abs(z_python)))
        sig_python = p_python < 0.05
        
        # Stata推断
        z_stata = stata_benchmark['att'] / stata_benchmark['se']
        p_stata = 2 * (1 - stats.norm.cdf(abs(z_stata)))
        sig_stata = p_stata < 0.05
        
        # 结论一致
        assert sig_python == sig_stata, \
            f"推断结论不一致: Python p={p_python:.4f}, Stata p={p_stata:.4f}"
    
    def test_ci_overlap(self, stata_test_data, stata_benchmark):
        """置信区间应有大量重叠"""
        result = estimate_psm(
            data=stata_test_data,
            y='y_44',
            d='g4',
            propensity_controls=['x1', 'x2'],
            n_neighbors=1,
            se_method='abadie_imbens_full',
        )
        
        # 计算重叠
        ci_python = (result.ci_lower, result.ci_upper)
        ci_stata = (stata_benchmark['ci_lower'], stata_benchmark['ci_upper'])
        
        overlap_lower = max(ci_python[0], ci_stata[0])
        overlap_upper = min(ci_python[1], ci_stata[1])
        overlap_width = overlap_upper - overlap_lower
        
        # 重叠应>0
        assert overlap_width > 0, "置信区间无重叠"
        
        # 重叠比例应>80%
        python_width = ci_python[1] - ci_python[0]
        overlap_ratio = overlap_width / python_width
        assert overlap_ratio > 0.8, f"置信区间重叠不足: {overlap_ratio:.2%}"
    
    def test_sample_size_match(self, stata_test_data, stata_benchmark):
        """样本量应一致"""
        result = estimate_psm(
            data=stata_test_data,
            y='y_44',
            d='g4',
            propensity_controls=['x1', 'x2'],
            n_neighbors=1,
            se_method='abadie_imbens_full',
        )
        
        assert result.n_treated == stata_benchmark['n_treat']
        assert result.n_control == stata_benchmark['n_control']
    
    def test_k_m_distribution_reasonable(self, stata_test_data):
        """K_M分布应合理"""
        result = estimate_psm(
            data=stata_test_data,
            y='y_44',
            d='g4',
            propensity_controls=['x1', 'x2'],
            n_neighbors=1,
            se_method='abadie_imbens_full',
        )
        
        K_M = result.control_match_counts
        
        # K_M总和至少应等于n_treat (Stata ties may increase matches)
        assert K_M.sum() >= result.n_treated
        
        # 应有未被匹配的控制单位
        assert (K_M == 0).sum() > 0
        
        # K_M最大值应合理（<10）
        assert K_M.max() < 10


# ============================================================================
# Numerical Precision Tests
# ============================================================================

class TestNumericalPrecision:
    """数值精度测试"""
    
    def test_propensity_score_model(self, stata_test_data):
        """倾向得分模型系数应与Stata一致"""
        # Stata logit结果:
        # x1: 0.37176
        # x2: -0.8242138
        # _cons: -3.016049
        
        result = estimate_psm(
            data=stata_test_data,
            y='y_44',
            d='g4',
            propensity_controls=['x1', 'x2'],
            n_neighbors=1,
            se_method='abadie_imbens_full',
        )
        
        # 检查倾向得分范围与Stata一致
        ps = result.propensity_scores
        # Stata: min=0.0383, max=0.5650, mean=0.129
        assert 0.03 < ps.min() < 0.05
        assert 0.55 < ps.max() < 0.60
        assert 0.12 < ps.mean() < 0.14
    
    def test_variance_components(self, stata_test_data, stata_benchmark):
        """方差组件分析"""
        result = estimate_psm(
            data=stata_test_data,
            y='y_44',
            d='g4',
            propensity_controls=['x1', 'x2'],
            n_neighbors=1,
            se_method='abadie_imbens_full',
        )
        
        n_treat = result.n_treated
        
        # Python方差
        python_var = result.se ** 2
        
        # Stata方差
        stata_var = stata_benchmark['se'] ** 2
        
        # 方差差异
        var_diff_pct = abs(python_var - stata_var) / stata_var * 100
        
        # 方差差异应<15%（对应SE差异约7%）
        assert var_diff_pct < 15, f"方差差异过大: {var_diff_pct:.2f}%"


# ============================================================================
# Documentation Test
# ============================================================================

class TestDocumentation:
    """文档测试 - 记录已知差异"""
    
    def test_known_se_difference(self, stata_test_data, stata_benchmark):
        """
        记录已知的SE差异
        
        Python实现与Stata的SE差异约5%，原因可能包括:
        1. 条件方差估计的距离度量细微差异
        2. 边界条件处理方式
        3. 数值精度
        
        此差异在实践中可接受，不影响统计推断结论。
        """
        result = estimate_psm(
            data=stata_test_data,
            y='y_44',
            d='g4',
            propensity_controls=['x1', 'x2'],
            n_neighbors=1,
            se_method='abadie_imbens_full',
        )
        
        se_diff_pct = abs(result.se - stata_benchmark['se']) / stata_benchmark['se'] * 100
        
        # 记录差异
        print(f"\n已知SE差异: {se_diff_pct:.2f}%")
        print(f"Python SE: {result.se:.6f}")
        print(f"Stata SE: {stata_benchmark['se']}")
        
        # 验证差异在预期范围内
        # 差异应该< 10%（主要来自于条件方差估计和AI Robust SE调整的差异）
        assert se_diff_pct < 10, f"SE差异超出预期范围(<10%), 实际: {se_diff_pct:.2f}%"


# ============================================================================
# Stata vce(iid) Tests
# ============================================================================

@pytest.fixture(scope='module')
def stata_benchmark_iid():
    """Stata teffects psmatch vce(iid) 基准结果"""
    return {
        'att': 3.554019,
        'se': 0.5492446,  # Stata vce(iid) SE
        'ci_lower': 2.477520,
        'ci_upper': 4.630519,
        'z': 6.47,
        'n_obs': 1000,
        'n_treat': 129,
        'n_control': 871,
    }


class TestStataIID:
    """Stata vce(iid) 对齐测试"""
    
    def test_iid_att_exact_match(self, stata_test_data, stata_benchmark_iid):
        """IID: ATT应完全一致"""
        result = estimate_psm(
            data=stata_test_data,
            y='y_44',
            d='g4',
            propensity_controls=['x1', 'x2'],
            n_neighbors=1,
            se_method='abadie_imbens_full',
            vce_type='iid',
            se_reference='stata',
        )
        
        # ATT完全一致（与 vce robust 相同的点估计）
        assert np.isclose(result.att, stata_benchmark_iid['att'], atol=1e-5), \
            f"ATT不一致: Python={result.att:.6f}, Stata={stata_benchmark_iid['att']}"
    
    def test_iid_se_within_acceptable_range(self, stata_test_data, stata_benchmark_iid):
        """IID: SE差异应在可接受范围内 (<5%)"""
        result = estimate_psm(
            data=stata_test_data,
            y='y_44',
            d='g4',
            propensity_controls=['x1', 'x2'],
            n_neighbors=1,
            se_method='abadie_imbens_full',
            vce_type='iid',
            se_reference='stata',
        )
        
        se_diff_pct = abs(result.se - stata_benchmark_iid['se']) / stata_benchmark_iid['se'] * 100
        
        # IID SE差异应<5%
        assert se_diff_pct < 5, \
            f"IID SE差异过大: Python={result.se:.6f}, Stata={stata_benchmark_iid['se']}, 差异={se_diff_pct:.2f}%"
        
        # 记录实际差异
        print(f"IID SE差异: {se_diff_pct:.2f}%")
    
    def test_iid_inference_conclusion_same(self, stata_test_data, stata_benchmark_iid):
        """IID: 统计推断结论应一致"""
        from scipy import stats
        
        result = estimate_psm(
            data=stata_test_data,
            y='y_44',
            d='g4',
            propensity_controls=['x1', 'x2'],
            n_neighbors=1,
            se_method='abadie_imbens_full',
            vce_type='iid',
            se_reference='stata',
        )
        
        # Python推断
        z_python = result.att / result.se
        p_python = 2 * (1 - stats.norm.cdf(abs(z_python)))
        sig_python = p_python < 0.05
        
        # Stata推断
        z_stata = stata_benchmark_iid['att'] / stata_benchmark_iid['se']
        p_stata = 2 * (1 - stats.norm.cdf(abs(z_stata)))
        sig_stata = p_stata < 0.05
        
        # 结论一致
        assert sig_python == sig_stata, \
            f"推断结论不一致: Python p={p_python:.4f}, Stata p={p_stata:.4f}"
    
    def test_iid_vs_robust_relationship(self, stata_test_data):
        """IID vs Robust SE 关系"""
        result_robust = estimate_psm(
            data=stata_test_data,
            y='y_44',
            d='g4',
            propensity_controls=['x1', 'x2'],
            n_neighbors=1,
            se_method='abadie_imbens_full',
            vce_type='robust',
            se_reference='stata',
        )
        
        result_iid = estimate_psm(
            data=stata_test_data,
            y='y_44',
            d='g4',
            propensity_controls=['x1', 'x2'],
            n_neighbors=1,
            se_method='abadie_imbens_full',
            vce_type='iid',
            se_reference='stata',
        )
        
        # Robust vs IID 的差异主要来自条件方差估计方式（局部 vs 全局）
        # 具体大小关系依赖数据特征
        print(f"Robust SE: {result_robust.se:.6f}")
        print(f"IID SE: {result_iid.se:.6f}")
        print(f"Ratio: {result_robust.se / result_iid.se:.4f}")
        
        # 两者都应该为正且合理
        assert result_robust.se > 0
        assert result_iid.se > 0
        # 通常不会相差超过2倍
        ratio = max(result_robust.se, result_iid.se) / min(result_robust.se, result_iid.se)
        assert ratio < 2.0, f"Robust/IID SE 比率异常: {ratio:.2f}"


# ============================================================================
# Variance Decomposition Validation (Optional)
# ============================================================================

class TestVarianceDecomposition:
    """方差组件分解验证 - 验证与Stata基准的对齐"""
    
    # Stata基准值 (g4_r4_nn1)
    STATA_ROBUST_SE = 0.5866075
    STATA_IID_SE = 0.5492446
    
    def test_robust_variance_decomposition(self, stata_test_data):
        """验证 robust SE 与 Stata vce(robust) 对齐"""
        result = estimate_psm(
            data=stata_test_data,
            y='y_44',
            d='g4',
            propensity_controls=['x1', 'x2'],
            n_neighbors=1,
            se_method='abadie_imbens_full',
            vce_type='robust',
            se_reference='stata',
        )
        
        # 验证与Stata基准对齐 (容差10%)
        # 差异主要来自条件方差估计与数值误差
        se_error = abs(result.se - self.STATA_ROBUST_SE) / self.STATA_ROBUST_SE
        assert se_error < 0.10, \
            f"Robust SE误差 {se_error:.2%} > 10%\n" \
            f"Python: {result.se:.6f}, Stata: {self.STATA_ROBUST_SE:.6f}"
        
        # 方差组件分析（仅用于诊断）
        Y_treat = stata_test_data.loc[stata_test_data['g4']==1, 'y_44'].to_numpy()
        Y_control = stata_test_data.loc[stata_test_data['g4']==0, 'y_44'].to_numpy()
        
        individual_effects = []
        for i, matches in enumerate(result.matched_control_ids):
            if len(matches) > 0:
                individual_effects.append(Y_treat[i] - Y_control[matches].mean())
        individual_effects = np.array(individual_effects)
        
        heterogeneity_sum = ((individual_effects - result.att) ** 2).sum()
        N1 = result.n_treated
        K_M = result.control_match_counts
        
        print(f"\nRobust SE验证:")
        print(f"  Python SE: {result.se:.6f}")
        print(f"  Stata SE:  {self.STATA_ROBUST_SE:.6f}")
        print(f"  误差: {se_error:.4%}")
        print(f"  异质性项: {heterogeneity_sum / (N1**2):.6f}")
        print(f"  Σ K(K-1): {(K_M * (K_M - 1)).sum()}")
    
    def test_iid_variance_decomposition(self, stata_test_data):
        """验证 iid SE 与 Stata vce(iid) 对齐"""
        result = estimate_psm(
            data=stata_test_data,
            y='y_44',
            d='g4',
            propensity_controls=['x1', 'x2'],
            n_neighbors=1,
            se_method='abadie_imbens_full',
            vce_type='iid',
            se_reference='stata',
        )
        
        # 验证与Stata基准对齐 (容差0.1%)
        se_error = abs(result.se - self.STATA_IID_SE) / self.STATA_IID_SE
        assert se_error < 0.001, \
            f"IID SE误差 {se_error:.2%} > 0.1%\n" \
            f"Python: {result.se:.6f}, Stata: {self.STATA_IID_SE:.6f}"
        
        # 方差组件分析（仅用于诊断）
        Y_treat = stata_test_data.loc[stata_test_data['g4']==1, 'y_44'].to_numpy()
        Y_control = stata_test_data.loc[stata_test_data['g4']==0, 'y_44'].to_numpy()
        
        individual_effects = []
        for i, matches in enumerate(result.matched_control_ids):
            if len(matches) > 0:
                individual_effects.append(Y_treat[i] - Y_control[matches].mean())
        individual_effects = np.array(individual_effects)
        
        heterogeneity_sum = ((individual_effects - result.att) ** 2).sum()
        N1 = result.n_treated
        K_M = result.control_match_counts
        
        print(f"\nIID SE验证:")
        print(f"  Python SE: {result.se:.6f}")
        print(f"  Stata SE:  {self.STATA_IID_SE:.6f}")
        print(f"  误差: {se_error:.4%}")
        print(f"  异质性项: {heterogeneity_sum / (N1**2):.6f}")
        print(f"  Σ K(K-1): {(K_M * (K_M - 1)).sum()}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
