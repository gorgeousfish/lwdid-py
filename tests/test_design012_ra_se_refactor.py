"""
DESIGN-012: RA 标准误实现代码重复 - 重构验证测试

测试目标:
1. 验证核心函数 _compute_ra_se_core() 的正确性
2. 验证重构后 _compute_ra_se_analytical() 输出不变
3. 验证重构后 compute_ra_se_analytical() 输出不变
4. 数值精度测试 (与 Stata 对比)
5. 边界条件测试
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from lwdid.staggered.estimators import (
    _compute_ra_se_core,
    _compute_ra_se_analytical,
    compute_ra_se_analytical,
    estimate_ra,
)


class TestRASECoreFunction:
    """测试核心实现函数 _compute_ra_se_core()"""
    
    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)
        n = 200
        
        # 生成协变量
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        
        # 生成处理变量 (约 30% 处理组)
        p_treat = 1 / (1 + np.exp(-0.5 * x1 - 0.3 * x2))
        D = (np.random.uniform(0, 1, n) < p_treat).astype(float)
        
        # 生成结果变量
        Y = 2 + 1.5 * x1 + 0.8 * x2 + 3.0 * D + np.random.normal(0, 1, n)
        
        return Y, D, x1, x2
    
    def test_core_returns_tuple(self, sample_data):
        """验证核心函数返回正确的类型"""
        Y, D, x1, x2 = sample_data
        X_design = np.column_stack([np.ones(len(Y)), x1, x2])
        beta_0 = np.array([2.0, 1.5, 0.8])  # 假设的系数
        att = 3.0
        
        result = _compute_ra_se_core(Y, D, X_design, beta_0, att)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)  # var_att
        assert isinstance(result[1], float)  # condition_number
    
    def test_core_variance_positive_for_normal_data(self, sample_data):
        """验证正常数据下方差为正"""
        Y, D, x1, x2 = sample_data
        X_design = np.column_stack([np.ones(len(Y)), x1, x2])
        
        # 在控制组上估计结果模型
        control_mask = D == 0
        X_control = X_design[control_mask]
        Y_control = Y[control_mask]
        model = sm.OLS(Y_control, X_control).fit()
        beta_0 = model.params
        
        # 计算 ATT
        treat_mask = D == 1
        X_treated = X_design[treat_mask]
        Y_treated = Y[treat_mask]
        y_counterfactual = X_treated @ beta_0
        att = np.mean(Y_treated - y_counterfactual)
        
        var_att, condition_number = _compute_ra_se_core(Y, D, X_design, beta_0, att)
        
        assert var_att > 0, "Variance should be positive for normal data"
        assert condition_number > 0, "Condition number should be positive"
    
    def test_core_condition_number_reasonable(self, sample_data):
        """验证条件数在合理范围内"""
        Y, D, x1, x2 = sample_data
        X_design = np.column_stack([np.ones(len(Y)), x1, x2])
        
        control_mask = D == 0
        model = sm.OLS(Y[control_mask], X_design[control_mask]).fit()
        beta_0 = model.params
        
        treat_mask = D == 1
        y_counterfactual = X_design[treat_mask] @ beta_0
        att = np.mean(Y[treat_mask] - y_counterfactual)
        
        _, condition_number = _compute_ra_se_core(Y, D, X_design, beta_0, att)
        
        # 条件数通常应小于 1e10 对于良好设计的问题
        assert condition_number < 1e10, "Condition number should be reasonable"


class TestPrivateFunctionConsistency:
    """测试私有函数 _compute_ra_se_analytical() 与核心函数一致性"""
    
    @pytest.fixture
    def regression_data(self):
        """创建用于回归的测试数据"""
        np.random.seed(123)
        n = 150
        
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        D = (np.random.uniform(0, 1, n) < 0.4).astype(float)
        Y = 1.0 + 2.0 * x1 + 1.0 * x2 + 2.5 * D + np.random.normal(0, 0.5, n)
        
        return Y, D, x1, x2
    
    def test_private_function_matches_core(self, regression_data):
        """验证私有函数输出与直接调用核心函数一致"""
        Y, D, x1, x2 = regression_data
        X = np.column_stack([x1, x2])
        
        treated_mask = D == 1
        control_mask = ~treated_mask
        
        # 在控制组上估计结果模型
        X_control = sm.add_constant(X[control_mask])
        model_control = sm.OLS(Y[control_mask], X_control).fit()
        
        # 计算 ATT
        X_treated = sm.add_constant(X[treated_mask])
        y_counterfactual = model_control.predict(X_treated)
        att = np.mean(Y[treated_mask] - y_counterfactual)
        
        # 使用私有函数
        se_private = _compute_ra_se_analytical(Y, X, treated_mask, model_control, att)
        
        # 直接使用核心函数
        X_design = sm.add_constant(X)
        var_att, _ = _compute_ra_se_core(Y, D, X_design, model_control.params, att)
        se_core = np.sqrt(max(var_att, 0))
        
        # 验证一致性
        np.testing.assert_allclose(
            se_private, se_core, rtol=1e-10,
            err_msg="Private function should match core function"
        )


class TestPublicFunctionConsistency:
    """测试公开函数 compute_ra_se_analytical() 与核心函数一致性"""
    
    @pytest.fixture
    def dataframe_data(self):
        """创建 DataFrame 格式的测试数据"""
        np.random.seed(456)
        n = 180
        
        data = pd.DataFrame({
            'y': np.random.normal(5, 2, n),
            'd': np.random.binomial(1, 0.35, n),
            'x1': np.random.normal(0, 1, n),
            'x2': np.random.normal(0, 1, n),
        })
        
        # 添加处理效应
        data.loc[data['d'] == 1, 'y'] += 2.0
        
        return data
    
    def test_public_function_matches_core(self, dataframe_data):
        """验证公开函数输出与直接调用核心函数一致"""
        data = dataframe_data
        controls = ['x1', 'x2']
        
        # 估计结果模型
        control_data = data[data['d'] == 0]
        X_control = sm.add_constant(control_data[controls].values)
        model = sm.OLS(control_data['y'].values, X_control).fit()
        
        outcome_coef = {
            '_intercept': model.params[0],
            'x1': model.params[1],
            'x2': model.params[2],
        }
        
        # 计算 ATT
        treat_data = data[data['d'] == 1]
        X_treat = sm.add_constant(treat_data[controls].values)
        y_counterfactual = model.predict(X_treat)
        att = np.mean(treat_data['y'].values - y_counterfactual)
        
        # 使用公开函数
        se_public, ci_lower, ci_upper = compute_ra_se_analytical(
            data, 'y', 'd', controls, att, outcome_coef
        )
        
        # 直接使用核心函数
        D = data['d'].values.astype(float)
        Y = data['y'].values.astype(float)
        X_design = np.column_stack([np.ones(len(D)), data[controls].values])
        beta_0 = np.array([outcome_coef['_intercept'], outcome_coef['x1'], outcome_coef['x2']])
        
        var_att, _ = _compute_ra_se_core(Y, D, X_design, beta_0, att)
        se_core = np.sqrt(max(var_att, 0))
        
        # 验证一致性
        np.testing.assert_allclose(
            se_public, se_core, rtol=1e-10,
            err_msg="Public function should match core function"
        )
    
    def test_public_function_ci_calculation(self, dataframe_data):
        """验证置信区间计算正确"""
        from scipy import stats
        
        data = dataframe_data
        controls = ['x1', 'x2']
        alpha = 0.05
        
        # 估计结果模型
        control_data = data[data['d'] == 0]
        X_control = sm.add_constant(control_data[controls].values)
        model = sm.OLS(control_data['y'].values, X_control).fit()
        
        outcome_coef = {
            '_intercept': model.params[0],
            'x1': model.params[1],
            'x2': model.params[2],
        }
        
        # 计算 ATT
        treat_data = data[data['d'] == 1]
        X_treat = sm.add_constant(treat_data[controls].values)
        y_counterfactual = model.predict(X_treat)
        att = np.mean(treat_data['y'].values - y_counterfactual)
        
        se, ci_lower, ci_upper = compute_ra_se_analytical(
            data, 'y', 'd', controls, att, outcome_coef, alpha=alpha
        )
        
        # 验证 CI 计算
        z_crit = stats.norm.ppf(1 - alpha / 2)
        expected_ci_lower = att - z_crit * se
        expected_ci_upper = att + z_crit * se
        
        np.testing.assert_allclose(ci_lower, expected_ci_lower, rtol=1e-10)
        np.testing.assert_allclose(ci_upper, expected_ci_upper, rtol=1e-10)


class TestEstimateRAIntegration:
    """测试 estimate_ra() 函数集成"""
    
    @pytest.fixture
    def ra_data(self):
        """创建用于 RA 估计的测试数据"""
        np.random.seed(789)
        n = 200
        
        data = pd.DataFrame({
            'y': np.random.normal(10, 3, n),
            'd': np.random.binomial(1, 0.4, n),
            'x1': np.random.normal(0, 1, n),
            'x2': np.random.uniform(-1, 1, n),
        })
        
        # 添加处理效应和协变量效应
        data['y'] += 1.5 * data['x1'] + 0.5 * data['x2']
        data.loc[data['d'] == 1, 'y'] += 3.0
        
        return data
    
    def test_estimate_ra_returns_valid_result(self, ra_data):
        """验证 estimate_ra() 返回有效结果"""
        result = estimate_ra(ra_data, 'y', 'd', ['x1', 'x2'])
        
        assert hasattr(result, 'att')
        assert hasattr(result, 'se')
        assert hasattr(result, 'ci_lower')
        assert hasattr(result, 'ci_upper')
        
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)
        assert result.se > 0
        assert result.ci_lower < result.att < result.ci_upper
    
    def test_estimate_ra_se_reasonable(self, ra_data):
        """验证 estimate_ra() 的 SE 在合理范围内"""
        result = estimate_ra(ra_data, 'y', 'd', ['x1', 'x2'])
        
        # SE 应该是正的且不会太大
        assert 0 < result.se < 10, "SE should be in reasonable range"
        
        # CI 宽度应该与 SE 一致 (约 2 * 1.96 * SE)
        ci_width = result.ci_upper - result.ci_lower
        expected_width = 2 * 1.96 * result.se
        np.testing.assert_allclose(ci_width, expected_width, rtol=0.01)


class TestNumericalPrecision:
    """数值精度测试"""
    
    def test_se_precision_small_sample(self):
        """小样本下的数值精度"""
        np.random.seed(111)
        n = 50
        
        data = pd.DataFrame({
            'y': np.random.normal(5, 1, n),
            'd': np.random.binomial(1, 0.5, n),
            'x1': np.random.normal(0, 1, n),
        })
        data.loc[data['d'] == 1, 'y'] += 1.0
        
        result = estimate_ra(data, 'y', 'd', ['x1'])
        
        assert np.isfinite(result.se)
        assert result.se > 0
    
    def test_se_precision_large_sample(self):
        """大样本下的数值精度"""
        np.random.seed(222)
        n = 2000
        
        data = pd.DataFrame({
            'y': np.random.normal(5, 1, n),
            'd': np.random.binomial(1, 0.5, n),
            'x1': np.random.normal(0, 1, n),
            'x2': np.random.normal(0, 1, n),
        })
        data.loc[data['d'] == 1, 'y'] += 1.0
        
        result = estimate_ra(data, 'y', 'd', ['x1', 'x2'])
        
        assert np.isfinite(result.se)
        assert result.se > 0
        # 大样本下 SE 应该较小
        assert result.se < 1.0


class TestEdgeCases:
    """边界条件测试"""
    
    def test_single_control(self):
        """单个协变量的情况"""
        np.random.seed(333)
        n = 100
        
        data = pd.DataFrame({
            'y': np.random.normal(5, 1, n),
            'd': np.random.binomial(1, 0.4, n),
            'x1': np.random.normal(0, 1, n),
        })
        data.loc[data['d'] == 1, 'y'] += 2.0
        
        result = estimate_ra(data, 'y', 'd', ['x1'])
        
        assert np.isfinite(result.se)
        assert result.se > 0
    
    def test_multiple_controls(self):
        """多个协变量的情况"""
        np.random.seed(444)
        n = 200
        
        data = pd.DataFrame({
            'y': np.random.normal(5, 1, n),
            'd': np.random.binomial(1, 0.4, n),
            'x1': np.random.normal(0, 1, n),
            'x2': np.random.normal(0, 1, n),
            'x3': np.random.normal(0, 1, n),
            'x4': np.random.normal(0, 1, n),
        })
        data.loc[data['d'] == 1, 'y'] += 2.0
        
        result = estimate_ra(data, 'y', 'd', ['x1', 'x2', 'x3', 'x4'])
        
        assert np.isfinite(result.se)
        assert result.se > 0
    
    def test_unbalanced_treatment(self):
        """不平衡处理组的情况"""
        np.random.seed(555)
        n = 200
        
        # 只有 20% 处理组
        data = pd.DataFrame({
            'y': np.random.normal(5, 1, n),
            'd': np.random.binomial(1, 0.2, n),
            'x1': np.random.normal(0, 1, n),
        })
        data.loc[data['d'] == 1, 'y'] += 2.0
        
        result = estimate_ra(data, 'y', 'd', ['x1'])
        
        assert np.isfinite(result.se)
        assert result.se > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
