"""
Estimator Comparison Tests

对比RA、IPWRA、PSM三种估计量的一致性。

Reference:
    Story E3-S2: PSM估计量实现
    docs/stories/story-E3-S2-psm-estimator.md Section 5.3
"""

import pytest
import numpy as np
import pandas as pd


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def dgp_data():
    """
    生成已知DGP的测试数据
    
    DGP: Y = 1 + 0.5*x1 + 0.3*x2 + 2.0*D + ε
    真实ATT = 2.0
    """
    np.random.seed(12345)
    n = 500
    
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    
    # 倾向得分依赖协变量
    ps_true = 1 / (1 + np.exp(-0.3 * x1 - 0.2 * x2))
    D = (np.random.uniform(0, 1, n) < ps_true).astype(int)
    
    # 结果变量
    Y = 1 + 0.5 * x1 + 0.3 * x2 + 2.0 * D + np.random.normal(0, 0.5, n)
    
    return pd.DataFrame({
        'Y': Y,
        'D': D,
        'x1': x1,
        'x2': x2,
    })


@pytest.fixture
def dgp_data_no_confounding():
    """
    无混淆的DGP数据（倾向得分不依赖协变量）
    
    DGP: Y = 1 + 0.5*x1 + 0.3*x2 + 1.5*D + ε
    真实ATT = 1.5
    """
    np.random.seed(54321)
    n = 400
    
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    
    # 随机分配处理（无混淆）
    D = (np.random.uniform(0, 1, n) < 0.5).astype(int)
    
    Y = 1 + 0.5 * x1 + 0.3 * x2 + 1.5 * D + np.random.normal(0, 0.5, n)
    
    return pd.DataFrame({
        'Y': Y,
        'D': D,
        'x1': x1,
        'x2': x2,
    })


# ============================================================================
# Test: All Estimators Consistent
# ============================================================================

class TestEstimatorConsistency:
    """
    验证三种估计量都能恢复真实ATT
    """
    
    def test_all_estimators_near_true_att(self, dgp_data):
        """
        测试三种估计量都接近真实ATT=2.0
        """
        from lwdid.staggered.estimation import run_ols_regression
        from lwdid.staggered.estimators import estimate_ipwra, estimate_psm
        
        true_att = 2.0
        
        # RA估计 (无控制变量)
        ra_result = run_ols_regression(
            data=dgp_data,
            y='Y',
            d='D',
            controls=None,
        )
        
        # RA估计 (有控制变量)
        ra_result_ctrl = run_ols_regression(
            data=dgp_data,
            y='Y',
            d='D',
            controls=['x1', 'x2'],
        )
        
        # IPWRA估计
        ipwra_result = estimate_ipwra(
            data=dgp_data,
            y='Y',
            d='D',
            controls=['x1', 'x2'],
            se_method='analytical',
        )
        
        # PSM估计
        psm_result = estimate_psm(
            data=dgp_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            n_neighbors=1,
            se_method='abadie_imbens',
        )
        
        # RA无控制变量可能有偏（因为有混淆）
        # RA有控制变量应该接近真实值
        assert abs(ra_result_ctrl['att'] - true_att) < 0.5, \
            f"RA(ctrl)偏差过大: {ra_result_ctrl['att']:.4f}"
        
        # IPWRA应该接近真实值
        assert abs(ipwra_result.att - true_att) < 0.5, \
            f"IPWRA偏差过大: {ipwra_result.att:.4f}"
        
        # PSM应该接近真实值
        assert abs(psm_result.att - true_att) < 0.5, \
            f"PSM偏差过大: {psm_result.att:.4f}"
    
    def test_no_confounding_all_estimators_similar(self, dgp_data_no_confounding):
        """
        无混淆情况下，所有估计量应该非常接近
        """
        from lwdid.staggered.estimation import run_ols_regression
        from lwdid.staggered.estimators import estimate_ipwra, estimate_psm
        
        data = dgp_data_no_confounding
        true_att = 1.5
        
        # RA（无控制变量时也应该无偏）
        ra_result = run_ols_regression(
            data=data,
            y='Y',
            d='D',
            controls=None,
        )
        
        # IPWRA
        ipwra_result = estimate_ipwra(
            data=data,
            y='Y',
            d='D',
            controls=['x1', 'x2'],
            se_method='analytical',
        )
        
        # PSM
        psm_result = estimate_psm(
            data=data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            n_neighbors=1,
        )
        
        # 无混淆时，RA无控制变量也应该接近真实值
        assert abs(ra_result['att'] - true_att) < 0.3, \
            f"RA偏差过大: {ra_result['att']:.4f}"
        
        assert abs(ipwra_result.att - true_att) < 0.3, \
            f"IPWRA偏差过大: {ipwra_result.att:.4f}"
        
        assert abs(psm_result.att - true_att) < 0.3, \
            f"PSM偏差过大: {psm_result.att:.4f}"


# ============================================================================
# Test: SE Comparison
# ============================================================================

class TestSEComparison:
    """
    标准误对比测试
    """
    
    def test_se_all_positive(self, dgp_data):
        """
        所有估计量的SE都应该是正的
        """
        from lwdid.staggered.estimation import run_ols_regression
        from lwdid.staggered.estimators import estimate_ipwra, estimate_psm
        
        ra_result = run_ols_regression(
            data=dgp_data,
            y='Y',
            d='D',
            controls=['x1', 'x2'],
        )
        
        ipwra_result = estimate_ipwra(
            data=dgp_data,
            y='Y',
            d='D',
            controls=['x1', 'x2'],
        )
        
        psm_result = estimate_psm(
            data=dgp_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
        )
        
        assert ra_result['se'] > 0
        assert ipwra_result.se > 0
        assert psm_result.se > 0
    
    def test_se_same_order_of_magnitude(self, dgp_data):
        """
        不同估计量的SE应该在同一数量级
        """
        from lwdid.staggered.estimation import run_ols_regression
        from lwdid.staggered.estimators import estimate_ipwra, estimate_psm
        
        ra_result = run_ols_regression(
            data=dgp_data,
            y='Y',
            d='D',
            controls=['x1', 'x2'],
        )
        
        ipwra_result = estimate_ipwra(
            data=dgp_data,
            y='Y',
            d='D',
            controls=['x1', 'x2'],
        )
        
        psm_result = estimate_psm(
            data=dgp_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
        )
        
        # SE应该在同一数量级（0.3x到3x范围内）
        se_ra = ra_result['se']
        se_ipwra = ipwra_result.se
        se_psm = psm_result.se
        
        assert 0.3 < se_ipwra / se_ra < 3, f"IPWRA vs RA SE比率异常: {se_ipwra/se_ra}"
        assert 0.3 < se_psm / se_ra < 3, f"PSM vs RA SE比率异常: {se_psm/se_ra}"


# ============================================================================
# Test: Coverage
# ============================================================================

class TestCoverage:
    """
    置信区间覆盖测试
    """
    
    def test_ci_contains_true_att(self, dgp_data):
        """
        置信区间应该包含真实ATT（大多数情况下）
        """
        from lwdid.staggered.estimation import run_ols_regression
        from lwdid.staggered.estimators import estimate_ipwra, estimate_psm
        
        true_att = 2.0
        
        ra_result = run_ols_regression(
            data=dgp_data,
            y='Y',
            d='D',
            controls=['x1', 'x2'],
        )
        
        ipwra_result = estimate_ipwra(
            data=dgp_data,
            y='Y',
            d='D',
            controls=['x1', 'x2'],
        )
        
        psm_result = estimate_psm(
            data=dgp_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
        )
        
        # 检查CI包含真实值
        # 由于是随机数据，可能不总是包含，但应该大多数时候包含
        ra_covers = ra_result['ci_lower'] < true_att < ra_result['ci_upper']
        ipwra_covers = ipwra_result.ci_lower < true_att < ipwra_result.ci_upper
        psm_covers = psm_result.ci_lower < true_att < psm_result.ci_upper
        
        # 至少应该有一些估计量的CI包含真实值
        n_covers = int(ra_covers) + int(ipwra_covers) + int(psm_covers)
        # 由于随机性，可能不是所有都覆盖，但至少有1个应该覆盖
        assert n_covers >= 1, f"太少的CI包含真实值: {n_covers}/3"


# ============================================================================
# Test: Robustness
# ============================================================================

class TestRobustness:
    """
    稳健性测试
    """
    
    def test_small_sample(self):
        """
        小样本测试
        """
        from lwdid.staggered.estimators import estimate_psm
        
        np.random.seed(999)
        n = 50
        
        x1 = np.random.normal(0, 1, n)
        D = (np.random.uniform(0, 1, n) < 0.5).astype(int)
        Y = 1 + D + np.random.normal(0, 0.5, n)
        
        data = pd.DataFrame({'Y': Y, 'D': D, 'x1': x1})
        
        result = estimate_psm(
            data=data,
            y='Y',
            d='D',
            propensity_controls=['x1'],
            n_neighbors=1,
        )
        
        # 应该成功运行
        assert result.att is not None
        assert result.n_treated > 0
    
    def test_imbalanced_groups(self):
        """
        不平衡处理组测试
        """
        from lwdid.staggered.estimators import estimate_psm
        
        np.random.seed(888)
        n = 100
        
        x1 = np.random.normal(0, 1, n)
        # 不平衡：处理组比例低
        D = (np.random.uniform(0, 1, n) < 0.2).astype(int)
        Y = 1 + D + np.random.normal(0, 0.5, n)
        
        data = pd.DataFrame({'Y': Y, 'D': D, 'x1': x1})
        
        result = estimate_psm(
            data=data,
            y='Y',
            d='D',
            propensity_controls=['x1'],
            n_neighbors=1,
        )
        
        assert result.att is not None
        # 处理组应该很小
        assert result.n_treated < result.n_control


# ============================================================================
# Test: Numerical Values Check
# ============================================================================

class TestNumericalCheck:
    """
    数值检查测试
    """
    
    def test_no_nan_in_results(self, dgp_data):
        """
        结果中不应该有NaN（除了边界情况）
        """
        from lwdid.staggered.estimators import estimate_psm
        
        result = estimate_psm(
            data=dgp_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
        )
        
        assert not np.isnan(result.att)
        assert not np.isnan(result.se)
        assert not np.isnan(result.t_stat)
        assert not np.isnan(result.pvalue)
        assert not np.isnan(result.ci_lower)
        assert not np.isnan(result.ci_upper)
    
    def test_pvalue_in_valid_range(self, dgp_data):
        """
        p值应该在[0,1]范围内
        """
        from lwdid.staggered.estimators import estimate_psm
        
        result = estimate_psm(
            data=dgp_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
        )
        
        assert 0 <= result.pvalue <= 1
