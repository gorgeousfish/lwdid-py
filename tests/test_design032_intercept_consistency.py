"""
DESIGN-032: API 一致性测试 - 截距系数命名

验证所有返回系数字典的函数都使用一致的 '_intercept' 键名。
"""

import numpy as np
import pandas as pd
import pytest


class TestInterceptNamingConsistency:
    """验证所有估计器函数返回的系数字典使用一致的 '_intercept' 键名"""
    
    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)
        n = 200
        data = pd.DataFrame({
            'd': np.random.binomial(1, 0.4, n),
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
        })
        data['y'] = 1 + 2*data['x1'] + 0.5*data['x2'] + 3*data['d'] + np.random.randn(n)
        return data
    
    def test_estimate_ra_uses_intercept(self, sample_data):
        """验证 estimate_ra 返回的 outcome_model_coef 使用 '_intercept'"""
        from lwdid.staggered import estimate_ra
        
        result = estimate_ra(sample_data, 'y', 'd', ['x1', 'x2'])
        
        assert '_intercept' in result.outcome_model_coef, \
            f"estimate_ra 应使用 '_intercept'，实际键: {list(result.outcome_model_coef.keys())}"
        assert '_cons' not in result.outcome_model_coef, \
            "estimate_ra 不应包含 '_cons' 键"
    
    def test_estimate_ipwra_uses_intercept(self, sample_data):
        """验证 estimate_ipwra 返回的系数字典使用 '_intercept'"""
        from lwdid.staggered import estimate_ipwra
        
        result = estimate_ipwra(sample_data, 'y', 'd', ['x1', 'x2'])
        
        # 检查 outcome_model_coef
        assert '_intercept' in result.outcome_model_coef, \
            f"estimate_ipwra.outcome_model_coef 应使用 '_intercept'，实际键: {list(result.outcome_model_coef.keys())}"
        assert '_cons' not in result.outcome_model_coef, \
            "estimate_ipwra.outcome_model_coef 不应包含 '_cons' 键"
        
        # 检查 propensity_model_coef
        assert '_intercept' in result.propensity_model_coef, \
            f"estimate_ipwra.propensity_model_coef 应使用 '_intercept'，实际键: {list(result.propensity_model_coef.keys())}"
        assert '_cons' not in result.propensity_model_coef, \
            "estimate_ipwra.propensity_model_coef 不应包含 '_cons' 键"
    
    def test_estimate_propensity_score_uses_intercept(self, sample_data):
        """验证 estimate_propensity_score 返回的系数字典使用 '_intercept'"""
        from lwdid.staggered.estimators import estimate_propensity_score
        
        controls = ['x1', 'x2']
        
        # estimate_propensity_score 期望 DataFrame 作为第一个参数
        pscores, coef_dict = estimate_propensity_score(sample_data, 'd', controls)
        
        assert '_intercept' in coef_dict, \
            f"estimate_propensity_score 应使用 '_intercept'，实际键: {list(coef_dict.keys())}"
        assert '_cons' not in coef_dict, \
            "estimate_propensity_score 不应包含 '_cons' 键"
    
    def test_estimate_outcome_model_uses_intercept(self, sample_data):
        """验证 estimate_outcome_model 返回的系数字典使用 '_intercept'"""
        from lwdid.staggered.estimators import estimate_outcome_model
        
        controls = ['x1', 'x2']
        
        # estimate_outcome_model 期望 DataFrame 作为第一个参数
        m0_hat, coef_dict = estimate_outcome_model(sample_data, 'y', 'd', controls)
        
        assert '_intercept' in coef_dict, \
            f"estimate_outcome_model 应使用 '_intercept'，实际键: {list(coef_dict.keys())}"
        assert '_cons' not in coef_dict, \
            "estimate_outcome_model 不应包含 '_cons' 键"
    
    def test_compute_ra_se_analytical_accepts_intercept(self, sample_data):
        """验证 compute_ra_se_analytical 可以正确使用 '_intercept' 键"""
        from lwdid.staggered import estimate_ra
        from lwdid.staggered.estimators import compute_ra_se_analytical
        
        result = estimate_ra(sample_data, 'y', 'd', ['x1', 'x2'])
        
        # 使用 estimate_ra 返回的系数计算 SE
        se, ci_lower, ci_upper = compute_ra_se_analytical(
            sample_data, 'y', 'd', ['x1', 'x2'],
            att=result.att,
            outcome_coef=result.outcome_model_coef
        )
        
        # 验证可以正确计算
        assert se > 0, "SE 应为正数"
        assert np.isfinite(se), "SE 应为有限值"
        assert ci_lower < result.att < ci_upper, "ATT 应在置信区间内"
    
    def test_all_functions_consistent_intercept_value(self, sample_data):
        """验证不同函数返回的截距值在数值上是合理的"""
        from lwdid.staggered import estimate_ra, estimate_ipwra
        from lwdid.staggered.estimators import estimate_propensity_score, estimate_outcome_model
        
        controls = ['x1', 'x2']
        
        # 获取所有截距值
        result_ra = estimate_ra(sample_data, 'y', 'd', controls)
        result_ipwra = estimate_ipwra(sample_data, 'y', 'd', controls)
        
        # 使用 DataFrame 接口调用
        _, ps_coef = estimate_propensity_score(sample_data, 'd', controls)
        _, outcome_coef = estimate_outcome_model(sample_data, 'y', 'd', controls)
        
        # RA 和 IPWRA 的 outcome_model 应该相近（但不完全相同，因为 IPWRA 可能使用权重）
        ra_intercept = result_ra.outcome_model_coef['_intercept']
        ipwra_outcome_intercept = result_ipwra.outcome_model_coef['_intercept']
        direct_outcome_intercept = outcome_coef['_intercept']
        
        # 验证数值合理性
        assert np.isfinite(ra_intercept), "RA 截距应为有限值"
        assert np.isfinite(ipwra_outcome_intercept), "IPWRA outcome 截距应为有限值"
        assert np.isfinite(direct_outcome_intercept), "直接调用 outcome_model 截距应为有限值"
        assert np.isfinite(ps_coef['_intercept']), "PS 截距应为有限值"
        
        # RA 和直接调用 outcome_model 应该完全一致
        np.testing.assert_allclose(
            ra_intercept, direct_outcome_intercept, rtol=1e-10,
            err_msg="estimate_ra 和 estimate_outcome_model 的截距应一致"
        )


class TestInterceptNamingDocumentation:
    """验证文档和代码中的截距命名一致性"""
    
    def test_compute_ra_se_analytical_docstring_mentions_intercept(self):
        """验证 compute_ra_se_analytical 文档中提到 '_intercept'"""
        from lwdid.staggered.estimators import compute_ra_se_analytical
        
        docstring = compute_ra_se_analytical.__doc__
        assert docstring is not None, "函数应有文档字符串"
        assert '_intercept' in docstring, \
            "compute_ra_se_analytical 文档应提到 '_intercept' 键名"


class TestInterceptNamingEdgeCases:
    """边界情况测试"""
    
    def test_single_control_variable(self):
        """测试单个控制变量时截距命名正确"""
        from lwdid.staggered import estimate_ra
        
        np.random.seed(42)
        n = 100
        data = pd.DataFrame({
            'd': np.random.binomial(1, 0.5, n),
            'x1': np.random.randn(n),
        })
        data['y'] = 1 + 2*data['x1'] + 3*data['d'] + np.random.randn(n)
        
        result = estimate_ra(data, 'y', 'd', ['x1'])
        
        assert '_intercept' in result.outcome_model_coef
        assert len(result.outcome_model_coef) == 2  # _intercept + x1
    
    def test_many_control_variables(self):
        """测试多个控制变量时截距命名正确"""
        from lwdid.staggered import estimate_ra
        
        np.random.seed(42)
        n = 200
        n_controls = 5
        
        data = pd.DataFrame({
            'd': np.random.binomial(1, 0.5, n),
        })
        for i in range(n_controls):
            data[f'x{i+1}'] = np.random.randn(n)
        
        data['y'] = 1 + sum(data[f'x{i+1}'] for i in range(n_controls)) + 3*data['d'] + np.random.randn(n)
        
        controls = [f'x{i+1}' for i in range(n_controls)]
        result = estimate_ra(data, 'y', 'd', controls)
        
        assert '_intercept' in result.outcome_model_coef
        assert len(result.outcome_model_coef) == n_controls + 1  # _intercept + controls
        
        # 验证所有控制变量都在字典中
        for c in controls:
            assert c in result.outcome_model_coef, f"控制变量 {c} 应在系数字典中"
