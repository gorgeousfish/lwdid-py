"""
Story 4.1: 完整版Abadie-Imbens标准误测试

测试PSM估计量的完整版Abadie-Imbens (2006)标准误实现，包括：
- K_M(j) 匹配次数计算
- 同组条件方差估计
- 完整版SE公式
- 与Stata一致性验证

References:
    Abadie A, Imbens GW (2006). "Large Sample Properties of Matching
    Estimators for Average Treatment Effects." Econometrica 74(1):235-267.
"""

import pytest
import numpy as np
import pandas as pd
import warnings

from lwdid.staggered.estimators import (
    estimate_psm,
    PSMResult,
    _compute_control_match_counts,
    _compute_control_match_weights,
    _estimate_conditional_variance_same_group,
    _compute_psm_se_abadie_imbens_full,
    _compute_psm_se_abadie_imbens_simple,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_psm_data():
    """
    简单的PSM测试数据
    
    DGP: Y = 1 + 0.5*x1 + 0.3*x2 + 2.0*D + ε
    真实ATT = 2.0
    """
    np.random.seed(42)
    n = 200
    
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    
    ps_true = 1 / (1 + np.exp(-0.5 * x1 - 0.3 * x2))
    D = (np.random.uniform(0, 1, n) < ps_true).astype(int)
    
    Y = 1 + 0.5 * x1 + 0.3 * x2 + 2.0 * D + np.random.normal(0, 0.5, n)
    
    return pd.DataFrame({
        'Y': Y,
        'D': D,
        'x1': x1,
        'x2': x2,
    })


@pytest.fixture
def large_psm_data():
    """较大的PSM测试数据，用于精度测试"""
    np.random.seed(123)
    n = 500
    
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    
    ps_true = 1 / (1 + np.exp(-0.3 * x1 - 0.2 * x2))
    D = (np.random.uniform(0, 1, n) < ps_true).astype(int)
    
    Y = 1 + 0.5 * x1 + 0.3 * x2 + 1.5 * D + np.random.normal(0, 0.5, n)
    
    return pd.DataFrame({
        'Y': Y,
        'D': D,
        'x1': x1,
        'x2': x2,
    })


# ============================================================================
# Test: K_M Computation
# ============================================================================

class TestControlMatchCounts:
    """K_M(j) 计算测试"""
    
    def test_no_replacement_all_k_m_le_1(self):
        """无放回匹配时，所有K_M ≤ 1"""
        matched = [[0], [1], [2]]  # 3个处理单位各匹配不同控制
        K_M = _compute_control_match_counts(matched, n_control=3)
        
        assert np.all(K_M <= 1)
        assert K_M.sum() == 3
    
    def test_with_replacement_k_m_can_exceed_1(self):
        """有放回匹配时，K_M可以>1"""
        matched = [[0], [0], [0]]  # 都匹配到控制单位0
        K_M = _compute_control_match_counts(matched, n_control=3)
        
        assert K_M[0] == 3
        assert K_M[1] == 0
        assert K_M[2] == 0
    
    def test_sum_equals_total_matches(self):
        """K_M总和等于总匹配数"""
        matched = [[0, 1], [1, 2], [0, 2]]  # 3个处理单位，各匹配2个
        K_M = _compute_control_match_counts(matched, n_control=3)
        
        n_treat = len(matched)
        n_neighbors = len(matched[0])
        assert K_M.sum() == n_treat * n_neighbors
    
    def test_length_equals_n_control(self):
        """返回数组长度等于n_control"""
        matched = [[0, 1], [2, 3]]
        K_M = _compute_control_match_counts(matched, n_control=5)
        
        assert len(K_M) == 5
    
    def test_empty_matches(self):
        """空匹配列表正确处理"""
        matched = [[], [0], []]
        K_M = _compute_control_match_counts(matched, n_control=3)
        
        assert K_M.sum() == 1
        assert K_M[0] == 1
    
    def test_mixed_match_sizes(self):
        """不同大小的匹配列表"""
        matched = [[0], [0, 1, 2], [1]]  # 不同数量的匹配
        K_M = _compute_control_match_counts(matched, n_control=3)
        
        assert K_M[0] == 2  # 被匹配2次
        assert K_M[1] == 2  # 被匹配2次
        assert K_M[2] == 1  # 被匹配1次
        assert K_M.sum() == 5


# ============================================================================
# Test: Conditional Variance Estimation
# ============================================================================

class TestConditionalVariance:
    """同组条件方差估计测试"""
    
    def test_variance_positive(self):
        """方差估计非负"""
        np.random.seed(42)
        Y = np.random.randn(100)
        X = np.random.randn(100, 2)
        W = np.random.binomial(1, 0.5, 100)
        
        sigma2 = _estimate_conditional_variance_same_group(Y, X, W, J=2)
        
        assert np.all(sigma2 >= 0)
    
    def test_same_group_matching(self):
        """验证同组内匹配 - 条件方差应反映组内变异"""
        np.random.seed(42)
        n = 100
        
        # 处理组Y均值高，控制组Y均值低，但组内方差小
        Y_treat = 10 + np.random.randn(50) * 0.1
        Y_control = 0 + np.random.randn(50) * 0.1
        Y = np.concatenate([Y_treat, Y_control])
        
        X = np.random.randn(n, 2)
        W = np.concatenate([np.ones(50), np.zeros(50)])
        
        sigma2 = _estimate_conditional_variance_same_group(Y, X, W, J=2)
        
        # 条件方差应该反映组内变异（小），而非组间差异（大）
        assert sigma2.mean() < 1.0  # 组内方差应该小
    
    def test_j_parameter(self):
        """测试J参数影响"""
        np.random.seed(42)
        Y = np.random.randn(100)
        X = np.random.randn(100, 2)
        W = np.random.binomial(1, 0.5, 100)
        
        sigma2_j2 = _estimate_conditional_variance_same_group(Y, X, W, J=2)
        sigma2_j5 = _estimate_conditional_variance_same_group(Y, X, W, J=5)
        
        # 两者都应该有效
        assert sigma2_j2.shape == sigma2_j5.shape
        assert np.all(sigma2_j2 >= 0)
        assert np.all(sigma2_j5 >= 0)
    
    def test_1d_x_input(self):
        """测试1D协变量输入（如倾向得分）"""
        np.random.seed(42)
        Y = np.random.randn(100)
        X = np.random.randn(100)  # 1D
        W = np.random.binomial(1, 0.5, 100)
        
        sigma2 = _estimate_conditional_variance_same_group(Y, X, W, J=2)
        
        assert len(sigma2) == 100
        assert np.all(sigma2 >= 0)
    
    def test_small_group_fallback(self):
        """小样本组使用全局方差fallback"""
        np.random.seed(42)
        n = 20
        Y = np.random.randn(n)
        X = np.random.randn(n, 2)
        # 极度不平衡：只有2个处理单位
        W = np.array([1, 1] + [0] * 18)
        
        sigma2 = _estimate_conditional_variance_same_group(Y, X, W, J=3)
        
        # 应该不会崩溃
        assert len(sigma2) == n


# ============================================================================
# Test: Full Abadie-Imbens SE
# ============================================================================

class TestAbadieImbensFullSE:
    """完整版Abadie-Imbens SE测试"""
    
    def test_se_positive(self, simple_psm_data):
        """SE为正"""
        result = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            se_method='abadie_imbens_full',
        )
        
        assert result.se > 0
        assert result.ci_lower < result.ci_upper
    
    def test_returns_k_m(self, simple_psm_data):
        """返回K_M数组"""
        result = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            se_method='abadie_imbens_full',
        )
        
        assert result.control_match_counts is not None
        assert len(result.control_match_counts) == result.n_control
        # K_M总和至少应等于 n_treat * n_neighbors (Stata ties may increase matches)
        assert result.control_match_counts.sum() >= result.n_treated * 1  # n_neighbors=1
    
    def test_returns_sigma2(self, simple_psm_data):
        """返回条件方差估计"""
        result = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            se_method='abadie_imbens_full',
        )
        
        assert result.sigma2_estimates is not None
        # 长度应为 n_treat + n_control
        assert len(result.sigma2_estimates) == result.n_treated + result.n_control
    
    def test_se_method_field(self, simple_psm_data):
        """SE方法字段正确设置"""
        result = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            se_method='abadie_imbens_full',
        )
        
        assert result.se_method == 'abadie_imbens_full'
    
    def test_vce_type_robust(self, simple_psm_data):
        """Robust VCE类型"""
        result = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            se_method='abadie_imbens_full',
            vce_type='robust',
        )
        
        assert result.vce_type == 'robust'
    
    def test_vce_type_iid(self, simple_psm_data):
        """IID VCE类型"""
        result = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            se_method='abadie_imbens_full',
            vce_type='iid',
        )
        
        assert result.vce_type == 'iid'
    
    def test_variance_estimation_j(self, simple_psm_data):
        """variance_estimation_j参数"""
        result_j2 = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            se_method='abadie_imbens_full',
            variance_estimation_j=2,
        )
        
        result_j5 = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            se_method='abadie_imbens_full',
            variance_estimation_j=5,
        )
        
        # 两者都应该有效，但可能略有不同
        assert result_j2.se > 0
        assert result_j5.se > 0


# ============================================================================
# Test: Full vs Simple SE Comparison
# ============================================================================

class TestFullVsSimpleSE:
    """完整版vs简化版SE对比测试"""
    
    def test_full_ge_simple_with_replacement(self, large_psm_data):
        """有放回匹配时，完整版SE通常≥简化版SE"""
        result_full = estimate_psm(
            data=large_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            with_replacement=True,
            se_method='abadie_imbens_full',
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result_simple = estimate_psm(
                data=large_psm_data,
                y='Y',
                d='D',
                propensity_controls=['x1', 'x2'],
                with_replacement=True,
                se_method='abadie_imbens_simple',
            )
        
        # robust 使用局部条件方差，iid 使用全局方差；大小关系依赖数据
        # 允许一些浮动，但完整版通常不会显著低于简化版
        assert result_full.se >= result_simple.se * 0.8
    
    def test_both_positive(self, simple_psm_data):
        """两种方法都返回正的SE"""
        result_full = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            se_method='abadie_imbens_full',
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result_simple = estimate_psm(
                data=simple_psm_data,
                y='Y',
                d='D',
                propensity_controls=['x1', 'x2'],
                se_method='abadie_imbens_simple',
            )
        
        assert result_full.se > 0
        assert result_simple.se > 0


# ============================================================================
# Test: API Backward Compatibility
# ============================================================================

class TestBackwardCompatibility:
    """向后兼容性测试"""
    
    def test_default_se_method_is_full(self, simple_psm_data):
        """默认使用完整版SE"""
        result = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
        )
        
        assert result.se_method == 'abadie_imbens_full'
    
    def test_abadie_imbens_alias_works(self, simple_psm_data):
        """'abadie_imbens'参数作为'abadie_imbens_full'的别名工作 (DESIGN-009修复)"""
        # DESIGN-009: 'abadie_imbens' 现在指向完整版，不再触发 DeprecationWarning
        result = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            se_method='abadie_imbens',  # 现在是 'abadie_imbens_full' 的别名
        )
        
        assert result.se_method == 'abadie_imbens_full'  # DESIGN-009: 现在指向完整版
        assert result.se > 0
    
    def test_new_fields_have_defaults(self, simple_psm_data):
        """新字段在简化版中也有值"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = estimate_psm(
                data=simple_psm_data,
                y='Y',
                d='D',
                propensity_controls=['x1', 'x2'],
                se_method='abadie_imbens_simple',
            )
        
        # 即使使用简化版，也应该有K_M
        assert result.control_match_counts is not None


# ============================================================================
# Test: Edge Cases
# ============================================================================

class TestEdgeCases:
    """边界条件测试"""
    
    def test_unmatched_controls_k_m_zero(self):
        """未被匹配的控制单位K_M=0"""
        matched = [[0], [0]]  # 只有控制单位0被匹配
        K_M = _compute_control_match_counts(matched, n_control=5)
        
        assert K_M[0] == 2
        assert K_M[1] == 0
        assert K_M[2] == 0
        assert K_M[3] == 0
        assert K_M[4] == 0
    
    def test_multiple_neighbors(self, simple_psm_data):
        """多邻居匹配"""
        result = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            n_neighbors=3,
            se_method='abadie_imbens_full',
        )
        
        # K_M总和至少应等于 n_treat * n_neighbors (Stata ties may increase matches)
        assert result.control_match_counts.sum() >= result.n_treated * 3
        assert result.se > 0


# ============================================================================
# Test: Formula Verification with Vibe Math
# ============================================================================

class TestFormulaVerification:
    """公式验证测试（手工计算）"""
    
    def test_k_m_computation_manual(self):
        """手工验证K_M计算"""
        # 简单案例：3个处理单位，各匹配2个控制
        matched = [[0, 1], [1, 2], [0, 2]]
        K_M = _compute_control_match_counts(matched, n_control=3)
        
        # 手工计算：
        # 控制0: 出现在匹配列表[0,1]和[0,2]中 -> 2次
        # 控制1: 出现在匹配列表[0,1]和[1,2]中 -> 2次
        # 控制2: 出现在匹配列表[1,2]和[0,2]中 -> 2次
        assert K_M[0] == 2
        assert K_M[1] == 2
        assert K_M[2] == 2
    
    def test_variance_decomposition(self):
        """方差分解公式验证"""
        # 构造简单数据进行手工验证
        np.random.seed(42)
        n_treat = 10
        n_control = 20
        M = 2  # 2个邻居
        
        Y_treat = np.random.randn(n_treat) + 5
        Y_control = np.random.randn(n_control) + 3
        X_treat = np.random.randn(n_treat, 2)
        X_control = np.random.randn(n_control, 2)
        
        # 构造匹配
        matched = [[i % n_control, (i + 1) % n_control] for i in range(n_treat)]
        
        # 计算ATT
        att = np.mean([Y_treat[i] - np.mean(Y_control[matched[i]]) for i in range(n_treat)])
        
        # 调用完整版SE
        se, ci_lower, ci_upper, K_M, sigma2 = _compute_psm_se_abadie_imbens_full(
            Y_treat=Y_treat,
            Y_control=Y_control,
            X_treat=X_treat,
            X_control=X_control,
            matched_control_ids=matched,
            att=att,
            n_neighbors=M,
            variance_estimation_j=2,
            vce_type='robust',
        )
        
        # 验证基本性质
        assert se > 0
        assert K_M.sum() == n_treat * M
        assert len(sigma2) == n_treat + n_control


# ============================================================================
# Test: BUG-001 Fix - Robust SE Formula Verification
# ============================================================================

class TestBUG001RobustSEFormula:
    """
    BUG-001 修复验证测试
    
    验证 robust SE 使用正确的公式（对齐 Stata），
    而不是经验拟合的魔法数字。
    
    Stata vce(robust) 公式:
    V̂ = [Σ_{i:D=1}(τ_i-τ)^2 + Σ_{j:D=0} ξ^2_j (K_m(j)^2 - K0_m(j))] / N₁²
    """
    
    def test_robust_se_no_magic_numbers(self, simple_psm_data):
        """验证 robust SE 不依赖魔法数字"""
        result = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            se_method='abadie_imbens_full',
            vce_type='robust',
            se_reference='stata',
        )
        
        # SE 应该是正数且合理
        assert result.se > 0
        assert not np.isnan(result.se)
        assert not np.isinf(result.se)
    
    def test_robust_formula_components(self):
        """手工验证 Stata robust 公式的各个组件（含异质性项）"""
        np.random.seed(42)
        n_treat = 20
        n_control = 40
        M = 2  # 2个邻居
        
        Y_treat = np.random.randn(n_treat) + 5
        Y_control = np.random.randn(n_control) + 3
        X_treat = np.random.randn(n_treat, 2)
        X_control = np.random.randn(n_control, 2)
        
        # 构造匹配（有重复使用控制单位）
        matched = [[i % n_control, (i + 5) % n_control] for i in range(n_treat)]
        
        # 计算ATT
        individual_effects = []
        for i in range(n_treat):
            Y_matched = np.mean(Y_control[matched[i]])
            individual_effects.append(Y_treat[i] - Y_matched)
        att = np.mean(individual_effects)
        
        # 调用完整版SE (stata 参考)
        se, ci_lower, ci_upper, K_M, sigma2 = _compute_psm_se_abadie_imbens_full(
            Y_treat=Y_treat,
            Y_control=Y_control,
            X_treat=X_treat,
            X_control=X_control,
            matched_control_ids=matched,
            att=att,
            n_neighbors=M,
            variance_estimation_j=2,
            vce_type='robust',
            se_reference='stata',
        )
        
        # 手工计算各组件
        # 1. 异质性项: Σ_{i:D=1}(τ_i-τ)^2
        heterogeneity_sum = np.sum((np.array(individual_effects) - att) ** 2)
        
        # 2. 控制组条件方差（加权）: Σⱼ ξ^2_j (K_m(j)^2 - K0_m(j))
        sigma2_control = sigma2[n_treat:]
        K_m, K0_m = _compute_control_match_weights(matched, n_control)
        sum_weighted_sigma2_control = np.sum((K_m ** 2 - K0_m) * sigma2_control)
        
        # 3. 手工计算方差 (Stata 公式)
        var_manual = (heterogeneity_sum + sum_weighted_sigma2_control) / (n_treat ** 2)
        se_manual = np.sqrt(var_manual)
        
        # 验证SE接近手工计算结果
        relative_diff = abs(se - se_manual) / se
        assert relative_diff < 0.01, \
            f"SE与手工计算差异过大: {relative_diff:.2%}"
    
    def test_robust_se_with_high_k_m(self):
        """测试高 K_M 值场景（某些控制单位被频繁匹配）"""
        np.random.seed(123)
        n_treat = 30
        n_control = 10  # 少量控制单位，导致高 K_M
        M = 2
        
        Y_treat = np.random.randn(n_treat) + 5
        Y_control = np.random.randn(n_control) + 3
        X_treat = np.random.randn(n_treat, 2)
        X_control = np.random.randn(n_control, 2)
        
        # 所有处理单位都匹配到前几个控制单位
        matched = [[0, 1] for _ in range(n_treat)]
        
        # 计算ATT
        att = np.mean([Y_treat[i] - np.mean(Y_control[matched[i]]) for i in range(n_treat)])
        
        # 调用完整版SE - 应该正常处理高 K_M 情况
        se, ci_lower, ci_upper, K_M, sigma2 = _compute_psm_se_abadie_imbens_full(
            Y_treat=Y_treat,
            Y_control=Y_control,
            X_treat=X_treat,
            X_control=X_control,
            matched_control_ids=matched,
            att=att,
            n_neighbors=M,
            variance_estimation_j=2,
            vce_type='robust',
            se_reference='stata',
        )
        
        # 验证K_M正确
        assert K_M[0] == n_treat  # 控制单位0被匹配n_treat次
        assert K_M[1] == n_treat  # 控制单位1被匹配n_treat次
        
        # SE应该仍然是正数且合理
        assert se > 0
        assert not np.isnan(se)
        assert not np.isinf(se)
    
    def test_stata_robust_vs_iid_relationship(self, large_psm_data):
        """
        Stata 实现: robust SE 通常 <= iid SE
        
        因为 Stata robust 使用局部条件方差（更精确），
        而 iid 使用全局条件方差（可能高估）。
        """
        result_robust = estimate_psm(
            data=large_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            se_method='abadie_imbens_full',
            vce_type='robust',
            se_reference='stata',
        )
        
        result_iid = estimate_psm(
            data=large_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            se_method='abadie_imbens_full',
            vce_type='iid',
            se_reference='stata',
        )
        
        # 两者都应该是正数
        assert result_robust.se > 0
        assert result_iid.se > 0
        
        # Stata 实现中，robust 可以小于 iid（使用局部方差 vs 全局方差）
        # 这里只验证两者都合理，不强制特定关系
    
    def test_paper_vs_stata_reference(self, simple_psm_data):
        """验证 paper 和 stata 参考的差异"""
        result_paper = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            se_method='abadie_imbens_full',
            vce_type='robust',
            se_reference='paper',
        )
        
        result_stata = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            se_method='abadie_imbens_full',
            vce_type='robust',
            se_reference='stata',
        )
        
        # 两者都包含异质性项，差异来自条件方差估计方法
        assert result_paper.se >= result_stata.se * 0.8, \
            "Paper SE should be >= 0.8 * Stata SE"


# ============================================================================
# Test: BUG-040 Regression Tests
# ============================================================================

class TestBUG040VarianceEstimationJValidation:
    """BUG-040: variance_estimation_j 参数验证在 paper 路径下应与 stata 一致"""
    
    def test_robust_j1_paper_raises_error(self, simple_psm_data):
        """BUG-040: vce_type='robust' + j=1 + se_reference='paper' 应抛出错误"""
        with pytest.raises(ValueError, match="variance_estimation_j must be >= 2"):
            estimate_psm(
                data=simple_psm_data,
                y='Y',
                d='D',
                propensity_controls=['x1', 'x2'],
                se_method='abadie_imbens_full',
                vce_type='robust',
                variance_estimation_j=1,
                se_reference='paper',
            )
    
    def test_robust_j1_stata_raises_error(self, simple_psm_data):
        """BUG-040: vce_type='robust' + j=1 + se_reference='stata' 应抛出错误"""
        with pytest.raises(ValueError, match="variance_estimation_j must be >= 2"):
            estimate_psm(
                data=simple_psm_data,
                y='Y',
                d='D',
                propensity_controls=['x1', 'x2'],
                se_method='abadie_imbens_full',
                vce_type='robust',
                variance_estimation_j=1,
                se_reference='stata',
            )
    
    def test_iid_j1_paper_allowed(self, simple_psm_data):
        """BUG-040: vce_type='iid' + j=1 应该被允许"""
        result = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            se_method='abadie_imbens_full',
            vce_type='iid',
            variance_estimation_j=1,
            se_reference='paper',
        )
        assert result.se > 0
    
    def test_iid_j1_stata_allowed(self, simple_psm_data):
        """BUG-040: vce_type='iid' + j=1 + se_reference='stata' 应该被允许"""
        result = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            se_method='abadie_imbens_full',
            vce_type='iid',
            variance_estimation_j=1,
            se_reference='stata',
        )
        assert result.se > 0
    
    def test_robust_j2_both_refs_allowed(self, simple_psm_data):
        """BUG-040: vce_type='robust' + j=2 在两种 se_reference 下都应工作"""
        for ref in ['stata', 'paper']:
            result = estimate_psm(
                data=simple_psm_data,
                y='Y',
                d='D',
                propensity_controls=['x1', 'x2'],
                se_method='abadie_imbens_full',
                vce_type='robust',
                variance_estimation_j=2,
                se_reference=ref,
            )
            assert result.se > 0, f"SE should be positive for se_reference='{ref}'"
    
    def test_validation_consistency_between_paths(self, simple_psm_data):
        """BUG-040: paper 和 stata 路径应有相同的参数验证行为"""
        # 验证两种 se_reference 对相同无效输入有相同的行为
        for ref in ['stata', 'paper']:
            with pytest.raises(ValueError):
                estimate_psm(
                    data=simple_psm_data,
                    y='Y',
                    d='D',
                    propensity_controls=['x1', 'x2'],
                    se_method='abadie_imbens_full',
                    vce_type='robust',
                    variance_estimation_j=1,
                    se_reference=ref,
                )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
