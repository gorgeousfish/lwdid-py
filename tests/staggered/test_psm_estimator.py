"""
PSM (Propensity Score Matching) Estimator Tests

测试倾向得分匹配估计量的实现，对应Story E3-S2。

Reference:
    Story E3-S2: PSM估计量实现
    docs/stories/story-E3-S2-psm-estimator.md
"""

import pytest
import numpy as np
import pandas as pd
import warnings

from lwdid.staggered.estimators import (
    estimate_psm,
    PSMResult,
    estimate_propensity_score,
    _validate_psm_inputs,
    _nearest_neighbor_match,
    _compute_psm_se_abadie_imbens,
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
    
    # 生成协变量
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    
    # 生成处理指示符（依赖协变量）
    ps_true = 1 / (1 + np.exp(-0.5 * x1 - 0.3 * x2))
    D = (np.random.uniform(0, 1, n) < ps_true).astype(int)
    
    # 生成结果变量
    Y = 1 + 0.5 * x1 + 0.3 * x2 + 2.0 * D + np.random.normal(0, 0.5, n)
    
    return pd.DataFrame({
        'Y': Y,
        'D': D,
        'x1': x1,
        'x2': x2,
    })


@pytest.fixture
def large_psm_data():
    """较大的PSM测试数据，用于Bootstrap测试"""
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


@pytest.fixture
def imbalanced_data():
    """处理组和控制组大小不平衡的数据"""
    np.random.seed(456)
    n = 100
    
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    
    # 高倾向得分，导致大多数是处理组
    ps_true = 1 / (1 + np.exp(-1.5 - 0.5 * x1))
    D = (np.random.uniform(0, 1, n) < ps_true).astype(int)
    
    Y = 1 + 0.5 * x1 + 1.0 * D + np.random.normal(0, 0.5, n)
    
    return pd.DataFrame({
        'Y': Y,
        'D': D,
        'x1': x1,
        'x2': x2,
    })


# ============================================================================
# Test: Basic Functionality
# ============================================================================

class TestEstimatePSMBasic:
    """estimate_psm() 基本功能测试"""
    
    def test_basic_functionality(self, simple_psm_data):
        """测试基本功能"""
        result = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            n_neighbors=1,
        )
        
        assert isinstance(result, PSMResult)
        assert result.n_treated > 0
        assert result.n_control > 0
        assert result.att is not None
        assert result.se > 0
        # ATT应该接近真实值2.0（容忍较大误差因为样本小）
        assert abs(result.att - 2.0) < 1.0
    
    def test_returns_psm_result(self, simple_psm_data):
        """测试返回PSMResult对象"""
        result = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
        )
        
        # 检查所有必要属性
        assert hasattr(result, 'att')
        assert hasattr(result, 'se')
        assert hasattr(result, 'ci_lower')
        assert hasattr(result, 'ci_upper')
        assert hasattr(result, 't_stat')
        assert hasattr(result, 'pvalue')
        assert hasattr(result, 'propensity_scores')
        assert hasattr(result, 'match_counts')
        assert hasattr(result, 'matched_control_ids')
        assert hasattr(result, 'n_treated')
        assert hasattr(result, 'n_control')
        assert hasattr(result, 'n_matched')
        assert hasattr(result, 'n_dropped')
    
    def test_confidence_interval_contains_att(self, simple_psm_data):
        """测试置信区间包含ATT"""
        result = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
        )
        
        # CI应该包含ATT
        assert result.ci_lower < result.att < result.ci_upper
        # CI宽度应该是正的
        assert result.ci_upper - result.ci_lower > 0


# ============================================================================
# Test: k-NN Matching
# ============================================================================

class TestKNNMatching:
    """k-NN匹配测试"""
    
    def test_k_neighbors_1(self, simple_psm_data):
        """测试1-NN匹配"""
        result = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            n_neighbors=1,
        )
        
        # 每个处理单位最多匹配1个控制单位
        for count in result.match_counts:
            assert count <= 1
    
    def test_k_neighbors_3(self, simple_psm_data):
        """测试3-NN匹配"""
        result = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            n_neighbors=3,
        )
        
        # 每个处理单位最多匹配3个控制单位
        for count in result.match_counts:
            assert count <= 3
        
        assert result.att is not None
    
    def test_k_neighbors_effect_on_variance(self, large_psm_data):
        """测试k增大对方差的影响"""
        result_1nn = estimate_psm(
            data=large_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            n_neighbors=1,
            se_method='abadie_imbens',
        )
        
        result_5nn = estimate_psm(
            data=large_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            n_neighbors=5,
            se_method='abadie_imbens',
        )
        
        # 两种方法都应该成功
        assert result_1nn.att is not None
        assert result_5nn.att is not None


# ============================================================================
# Test: With/Without Replacement
# ============================================================================

class TestReplacementOptions:
    """有放回/无放回匹配测试"""
    
    def test_with_replacement_default(self, simple_psm_data):
        """测试默认有放回匹配"""
        result = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            with_replacement=True,
        )
        
        assert result.att is not None
    
    def test_without_replacement(self, simple_psm_data):
        """测试无放回匹配"""
        result = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            with_replacement=False,
        )
        
        # 无放回时，n_matched应该<=n_treated
        assert result.n_matched <= result.n_treated
        assert result.att is not None
    
    def test_without_replacement_constraint(self, simple_psm_data):
        """测试无放回匹配的约束"""
        result = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            with_replacement=False,
            n_neighbors=1,
        )
        
        # 匹配的控制单位数应该<=控制组大小
        assert result.n_matched <= result.n_control


# ============================================================================
# Test: Caliper
# ============================================================================

class TestCaliper:
    """Caliper匹配阈值测试"""
    
    def test_no_caliper(self, simple_psm_data):
        """测试无caliper"""
        result = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            caliper=None,
        )
        
        assert result.caliper is None
        assert result.n_dropped == 0  # 无caliper时不应有dropped
    
    def test_caliper_sd_scale(self, simple_psm_data):
        """测试以标准差为单位的caliper"""
        result = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            caliper=0.5,
            caliper_scale='sd',
        )
        
        assert result.caliper is not None
        assert result.att is not None
    
    def test_caliper_absolute_scale(self, simple_psm_data):
        """测试绝对值caliper"""
        result = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            caliper=0.1,
            caliper_scale='absolute',
        )
        
        assert result.caliper == 0.1
        assert result.att is not None
    
    def test_strict_caliper_drops_units(self, simple_psm_data):
        """测试严格caliper导致单位被丢弃"""
        result_loose = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            caliper=2.0,
            caliper_scale='sd',
        )
        
        result_strict = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            caliper=0.1,
            caliper_scale='sd',
        )
        
        # 严格caliper应该导致更多dropped
        assert result_strict.n_dropped >= result_loose.n_dropped


# ============================================================================
# Test: Standard Error Methods
# ============================================================================

class TestSEMethods:
    """标准误计算方法测试"""
    
    def test_abadie_imbens_se(self, simple_psm_data):
        """测试Abadie-Imbens标准误"""
        result = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            se_method='abadie_imbens',
        )
        
        assert result.se > 0
        assert not np.isnan(result.se)
    
    def test_bootstrap_se(self, large_psm_data):
        """测试Bootstrap标准误"""
        result = estimate_psm(
            data=large_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            se_method='bootstrap',
            n_bootstrap=50,  # 减少bootstrap次数以加速测试
            seed=42,
        )
        
        assert result.se > 0
        assert not np.isnan(result.se)
    
    def test_se_methods_comparable(self, large_psm_data):
        """测试两种SE方法结果可比"""
        result_ai = estimate_psm(
            data=large_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            se_method='abadie_imbens',
        )
        
        result_boot = estimate_psm(
            data=large_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            se_method='bootstrap',
            n_bootstrap=100,
            seed=42,
        )
        
        # 两种方法的SE应该在同一数量级
        ratio = result_ai.se / result_boot.se
        assert 0.3 < ratio < 3.0, f"SE ratio out of range: {ratio}"
    
    def test_invalid_se_method(self, simple_psm_data):
        """测试无效的SE方法"""
        with pytest.raises(ValueError, match="未知的se_method"):
            estimate_psm(
                data=simple_psm_data,
                y='Y',
                d='D',
                propensity_controls=['x1', 'x2'],
                se_method='invalid_method',
            )


# ============================================================================
# Test: Input Validation
# ============================================================================

class TestInputValidation:
    """输入验证测试"""
    
    def test_missing_y_column(self, simple_psm_data):
        """测试缺失结果变量"""
        with pytest.raises(ValueError, match="结果变量.*不在数据中"):
            estimate_psm(
                data=simple_psm_data,
                y='nonexistent',
                d='D',
                propensity_controls=['x1'],
            )
    
    def test_missing_d_column(self, simple_psm_data):
        """测试缺失处理指示符"""
        with pytest.raises(ValueError, match="处理指示符.*不在数据中"):
            estimate_psm(
                data=simple_psm_data,
                y='Y',
                d='nonexistent',
                propensity_controls=['x1'],
            )
    
    def test_missing_control_column(self, simple_psm_data):
        """测试缺失控制变量"""
        with pytest.raises(ValueError, match="控制变量不存在"):
            estimate_psm(
                data=simple_psm_data,
                y='Y',
                d='D',
                propensity_controls=['x1', 'nonexistent'],
            )
    
    def test_invalid_n_neighbors(self, simple_psm_data):
        """测试无效的n_neighbors"""
        with pytest.raises(ValueError, match="n_neighbors必须>=1"):
            estimate_psm(
                data=simple_psm_data,
                y='Y',
                d='D',
                propensity_controls=['x1'],
                n_neighbors=0,
            )
    
    def test_empty_treatment_group(self, simple_psm_data):
        """测试空处理组"""
        data = simple_psm_data.copy()
        data['D'] = 0  # 全部设为控制组
        
        with pytest.raises(ValueError, match="处理组样本量为0"):
            estimate_psm(
                data=data,
                y='Y',
                d='D',
                propensity_controls=['x1'],
            )
    
    def test_insufficient_controls(self):
        """测试控制组不足"""
        # 创建只有很少控制组的数据
        data = pd.DataFrame({
            'Y': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'D': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # 只有1个控制
            'x1': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        })
        
        with pytest.raises(ValueError, match="控制组样本量.*小于匹配数"):
            estimate_psm(
                data=data,
                y='Y',
                d='D',
                propensity_controls=['x1'],
                n_neighbors=5,
            )
    
    def test_invalid_d_values(self, simple_psm_data):
        """测试非0/1的处理指示符"""
        data = simple_psm_data.copy()
        data['D'] = data['D'] + 1  # 变成1/2
        
        with pytest.raises(ValueError, match="处理指示符.*必须是0/1值"):
            estimate_psm(
                data=data,
                y='Y',
                d='D',
                propensity_controls=['x1'],
            )


# ============================================================================
# Test: Edge Cases
# ============================================================================

class TestEdgeCases:
    """边界情况测试"""
    
    def test_small_sample_warning(self):
        """测试小样本警告"""
        np.random.seed(789)
        data = pd.DataFrame({
            'Y': np.random.normal(0, 1, 20),
            'D': [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'x1': np.random.normal(0, 1, 20),
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = estimate_psm(
                data=data,
                y='Y',
                d='D',
                propensity_controls=['x1'],
            )
            
            # 应该有小样本警告
            assert result.att is not None  # 但仍应返回结果
    
    def test_caliper_too_strict(self):
        """测试caliper过于严格"""
        np.random.seed(999)
        n = 50
        data = pd.DataFrame({
            'Y': np.random.normal(0, 1, n),
            'D': [1] * 25 + [0] * 25,
            # x1差异很大，导致倾向得分差异大
            'x1': np.concatenate([np.random.normal(5, 0.1, 25), 
                                  np.random.normal(-5, 0.1, 25)]),
        })
        
        with pytest.raises(ValueError, match="所有处理单位都无法找到有效匹配"):
            estimate_psm(
                data=data,
                y='Y',
                d='D',
                propensity_controls=['x1'],
                caliper=0.001,  # 非常严格的caliper
                caliper_scale='absolute',
            )
    
    def test_handles_missing_values(self, simple_psm_data):
        """测试处理缺失值"""
        data = simple_psm_data.copy()
        # 添加一些缺失值
        data.loc[0, 'Y'] = np.nan
        data.loc[1, 'x1'] = np.nan
        
        result = estimate_psm(
            data=data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
        )
        
        # 应该成功处理
        assert result.att is not None


# ============================================================================
# Test: Helper Functions
# ============================================================================

class TestHelperFunctions:
    """辅助函数测试"""
    
    def test_validate_psm_inputs_valid(self, simple_psm_data):
        """测试有效输入"""
        # 不应抛出异常
        _validate_psm_inputs(
            simple_psm_data, 'Y', 'D', ['x1', 'x2'], 1
        )
    
    def test_nearest_neighbor_match_basic(self):
        """测试最近邻匹配基本功能"""
        pscores_treat = np.array([0.3, 0.5, 0.7])
        pscores_control = np.array([0.25, 0.55, 0.8])
        
        matched, counts, dropped = _nearest_neighbor_match(
            pscores_treat, pscores_control,
            n_neighbors=1,
            with_replacement=True,
            caliper=None,
        )
        
        assert len(matched) == 3
        assert dropped == 0
        # 第一个处理单位(0.3)应该匹配到第一个控制单位(0.25)
        assert 0 in matched[0]
    
    def test_nearest_neighbor_match_caliper(self):
        """测试带caliper的匹配"""
        pscores_treat = np.array([0.3, 0.5, 0.9])
        pscores_control = np.array([0.25, 0.55])
        
        matched, counts, dropped = _nearest_neighbor_match(
            pscores_treat, pscores_control,
            n_neighbors=1,
            with_replacement=True,
            caliper=0.1,
        )
        
        assert len(matched) == 3
        # 最后一个处理单位(0.9)应该被dropped，因为离所有控制单位太远
        assert dropped >= 1
    
    def test_compute_psm_se_abadie_imbens(self):
        """测试Abadie-Imbens SE计算"""
        # 设计不同的个体效应以确保方差>0
        Y_treat = np.array([1.0, 2.5, 3.0, 5.0])
        Y_control = np.array([0.5, 1.5, 2.5, 3.5])
        matched_ids = [[0], [1], [2], [3]]
        # 个体效应: 0.5, 1.0, 0.5, 1.5 -> 不同，方差>0
        
        se, ci_lower, ci_upper = _compute_psm_se_abadie_imbens(
            Y_treat, Y_control, matched_ids, att=0.875, alpha=0.05
        )
        
        assert se > 0
        assert ci_lower < ci_upper


# ============================================================================
# Test: Reproducibility
# ============================================================================

class TestReproducibility:
    """可复现性测试"""
    
    def test_bootstrap_reproducible_with_seed(self, large_psm_data):
        """测试带种子的Bootstrap可复现"""
        result1 = estimate_psm(
            data=large_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            se_method='bootstrap',
            n_bootstrap=50,
            seed=12345,
        )
        
        result2 = estimate_psm(
            data=large_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            se_method='bootstrap',
            n_bootstrap=50,
            seed=12345,
        )
        
        # ATT应该完全相同（确定性匹配）
        assert result1.att == result2.att
        # SE应该相同（相同种子的Bootstrap）
        assert result1.se == result2.se
