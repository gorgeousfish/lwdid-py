# -*- coding: utf-8 -*-
"""
HC标准误测试模块

测试Story 6.1的HC标准误计算:
- HC0, HC1 (robust), HC2, HC3标准误
- 与Stata的OLS回归vce选项一致
- 与staggered模块的HC计算一致
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from .conftest import (
    compute_transformed_outcome_common_timing,
    assert_se_close_to_stata,
)


class TestHCStataConsistency:
    """HC标准误与Stata一致性测试"""
    
    # Stata OLS回归验证结果 (reg y_dot d x1 x2 if f04, vce(xxx))
    # 这些是OLS回归中D系数的标准误
    STATA_RESULTS = {
        'ols': {'se_d': 0.4013443},      # 同方差SE
        'hc1': {'se_d': 0.4022646},      # HC1 (robust)
        'hc2': {'se_d': 0.4029862},      # HC2
        'hc3': {'se_d': 0.4045209},      # HC3
    }
    
    def _compute_hc_se(
        self,
        data: pd.DataFrame,
        y: str,
        d: str,
        controls: list,
        hc_type: str,
    ) -> float:
        """
        计算指定HC类型的标准误
        
        Parameters
        ----------
        data : pd.DataFrame
            数据
        y : str
            结果变量
        d : str
            处理变量
        controls : list
            控制变量
        hc_type : str
            HC类型: 'ols', 'hc0', 'hc1', 'hc2', 'hc3'
            
        Returns
        -------
        float
            D系数的标准误
        """
        y_vals = data[y].values.astype(float)
        d_vals = data[d].values.astype(float)
        X_controls = data[controls].values.astype(float)
        
        # 设计矩阵: [1, D, X1, X2]
        X = np.column_stack([np.ones(len(data)), d_vals, X_controls])
        n, k = X.shape
        
        # OLS估计
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ (X.T @ y_vals)
        
        # 残差
        y_hat = X @ beta
        residuals = y_vals - y_hat
        
        # 杠杆值 (hat matrix diagonal)
        H_diag = np.sum(X * (X @ XtX_inv), axis=1)
        
        if hc_type == 'ols':
            # 同方差方差
            sigma2 = np.sum(residuals ** 2) / (n - k)
            var_beta = sigma2 * XtX_inv
        elif hc_type == 'hc0':
            # HC0: 无自由度调整
            omega = residuals ** 2
            meat = (X.T * omega) @ X
            var_beta = XtX_inv @ meat @ XtX_inv
        elif hc_type == 'hc1':
            # HC1: n/(n-k) 自由度调整 (Stata robust)
            correction = n / (n - k)
            omega = residuals ** 2 * correction
            meat = (X.T * omega) @ X
            var_beta = XtX_inv @ meat @ XtX_inv
        elif hc_type == 'hc2':
            # HC2: 1/(1-h_ii) 调整
            omega = residuals ** 2 / (1 - H_diag)
            meat = (X.T * omega) @ X
            var_beta = XtX_inv @ meat @ XtX_inv
        elif hc_type == 'hc3':
            # HC3: 1/(1-h_ii)^2 调整
            omega = residuals ** 2 / ((1 - H_diag) ** 2)
            meat = (X.T * omega) @ X
            var_beta = XtX_inv @ meat @ XtX_inv
        else:
            raise ValueError(f"Unknown HC type: {hc_type}")
        
        se_d = np.sqrt(var_beta[1, 1])
        return se_d
    
    def test_ols_se_vs_stata(self, transformed_data):
        """测试同方差SE与Stata一致"""
        period_data = transformed_data[transformed_data['period'] == 4].copy()
        
        se = self._compute_hc_se(period_data, 'y_dot', 'd', ['x1', 'x2'], 'ols')
        
        assert_se_close_to_stata(
            se,
            self.STATA_RESULTS['ols']['se_d'],
            tolerance=0.01,
            description="OLS SE"
        )
    
    def test_hc1_se_vs_stata(self, transformed_data):
        """测试HC1 (robust) SE与Stata一致"""
        period_data = transformed_data[transformed_data['period'] == 4].copy()
        
        se = self._compute_hc_se(period_data, 'y_dot', 'd', ['x1', 'x2'], 'hc1')
        
        assert_se_close_to_stata(
            se,
            self.STATA_RESULTS['hc1']['se_d'],
            tolerance=0.01,
            description="HC1 SE"
        )
    
    def test_hc2_se_vs_stata(self, transformed_data):
        """测试HC2 SE与Stata一致"""
        period_data = transformed_data[transformed_data['period'] == 4].copy()
        
        se = self._compute_hc_se(period_data, 'y_dot', 'd', ['x1', 'x2'], 'hc2')
        
        assert_se_close_to_stata(
            se,
            self.STATA_RESULTS['hc2']['se_d'],
            tolerance=0.01,
            description="HC2 SE"
        )
    
    def test_hc3_se_vs_stata(self, transformed_data):
        """测试HC3 SE与Stata一致"""
        period_data = transformed_data[transformed_data['period'] == 4].copy()
        
        se = self._compute_hc_se(period_data, 'y_dot', 'd', ['x1', 'x2'], 'hc3')
        
        assert_se_close_to_stata(
            se,
            self.STATA_RESULTS['hc3']['se_d'],
            tolerance=0.01,
            description="HC3 SE"
        )


class TestHCOrdering:
    """HC标准误顺序测试"""
    
    def test_hc_ordering(self, transformed_data):
        """
        验证HC标准误的典型顺序
        
        通常: HC0 < HC1 < HC2 < HC3
        """
        period_data = transformed_data[transformed_data['period'] == 4].copy()
        
        y_vals = period_data['y_dot'].values.astype(float)
        X = sm.add_constant(np.column_stack([
            period_data['d'].values.astype(float),
            period_data['x1'].values.astype(float),
            period_data['x2'].values.astype(float),
        ]))
        
        model = sm.OLS(y_vals, X).fit()
        
        # 获取不同HC类型的SE
        se_hc0 = model.get_robustcov_results(cov_type='HC0').bse[1]
        se_hc1 = model.get_robustcov_results(cov_type='HC1').bse[1]
        se_hc2 = model.get_robustcov_results(cov_type='HC2').bse[1]
        se_hc3 = model.get_robustcov_results(cov_type='HC3').bse[1]
        
        # 验证顺序
        assert se_hc0 <= se_hc1 * 1.01, "HC0 should be <= HC1"
        assert se_hc1 <= se_hc2 * 1.01, "HC1 should be <= HC2"
        assert se_hc2 <= se_hc3 * 1.01, "HC2 should be <= HC3"
    
    def test_hc_all_positive(self, transformed_data):
        """验证所有HC类型都产生正的SE"""
        period_data = transformed_data[transformed_data['period'] == 4].copy()
        
        y_vals = period_data['y_dot'].values.astype(float)
        X = sm.add_constant(np.column_stack([
            period_data['d'].values.astype(float),
            period_data['x1'].values.astype(float),
            period_data['x2'].values.astype(float),
        ]))
        
        model = sm.OLS(y_vals, X).fit()
        
        for hc_type in ['HC0', 'HC1', 'HC2', 'HC3']:
            robust_model = model.get_robustcov_results(cov_type=hc_type)
            se = robust_model.bse[1]
            assert se > 0, f"{hc_type} SE should be positive"
            assert not np.isnan(se), f"{hc_type} SE should not be NaN"


class TestHCWithStatsmodels:
    """使用statsmodels验证HC实现"""
    
    def test_hc1_matches_statsmodels(self, transformed_data):
        """验证自定义HC1实现与statsmodels一致"""
        period_data = transformed_data[transformed_data['period'] == 4].copy()
        
        y_vals = period_data['y_dot'].values.astype(float)
        X = sm.add_constant(np.column_stack([
            period_data['d'].values.astype(float),
            period_data['x1'].values.astype(float),
            period_data['x2'].values.astype(float),
        ]))
        
        # statsmodels HC1
        model = sm.OLS(y_vals, X).fit(cov_type='HC1')
        se_statsmodels = model.bse[1]
        
        # 自定义实现
        n, k = X.shape
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ (X.T @ y_vals)
        residuals = y_vals - X @ beta
        correction = n / (n - k)
        omega = residuals ** 2 * correction
        meat = (X.T * omega) @ X
        var_beta = XtX_inv @ meat @ XtX_inv
        se_custom = np.sqrt(var_beta[1, 1])
        
        # 验证一致
        rel_diff = abs(se_custom - se_statsmodels) / se_statsmodels
        assert rel_diff < 1e-10, \
            f"HC1 mismatch: custom={se_custom}, statsmodels={se_statsmodels}"


class TestHCFormulas:
    """HC公式验证测试"""
    
    def test_hc0_formula(self, transformed_data):
        """
        验证HC0公式
        
        HC0: Var(β) = (X'X)^{-1} X' diag(e²) X (X'X)^{-1}
        """
        period_data = transformed_data[transformed_data['period'] == 4].copy()
        
        y_vals = period_data['y_dot'].values.astype(float)
        d_vals = period_data['d'].values.astype(float)
        X = sm.add_constant(np.column_stack([
            d_vals,
            period_data['x1'].values.astype(float),
            period_data['x2'].values.astype(float),
        ]))
        
        n, k = X.shape
        
        # OLS估计
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ (X.T @ y_vals)
        
        # 残差
        residuals = y_vals - X @ beta
        
        # HC0方差
        e2 = residuals ** 2
        meat = X.T @ np.diag(e2) @ X
        var_hc0 = XtX_inv @ meat @ XtX_inv
        
        # 验证正定
        eigenvalues = np.linalg.eigvalsh(var_hc0)
        assert np.all(eigenvalues > 0), "HC0 variance should be positive definite"
        
        # 验证对称
        assert np.allclose(var_hc0, var_hc0.T), "HC0 variance should be symmetric"
    
    def test_hc_leverage_values(self, transformed_data):
        """
        验证杠杆值计算
        
        h_ii = X_i' (X'X)^{-1} X_i
        
        性质:
        - 0 < h_ii < 1
        - Σ h_ii = k (参数个数)
        """
        period_data = transformed_data[transformed_data['period'] == 4].copy()
        
        X = sm.add_constant(np.column_stack([
            period_data['d'].values.astype(float),
            period_data['x1'].values.astype(float),
            period_data['x2'].values.astype(float),
        ]))
        
        n, k = X.shape
        
        # 计算杠杆值
        XtX_inv = np.linalg.inv(X.T @ X)
        H_diag = np.sum(X * (X @ XtX_inv), axis=1)
        
        # 验证性质
        assert np.all(H_diag > 0), "Leverage values should be positive"
        assert np.all(H_diag < 1), "Leverage values should be less than 1"
        assert abs(H_diag.sum() - k) < 1e-10, f"Sum of leverage should equal k={k}"


class TestHCStaggeredModule:
    """与staggered模块HC实现的一致性测试"""
    
    def test_staggered_hc1_consistency(self, transformed_data):
        """测试与staggered模块的HC1计算一致"""
        from lwdid.staggered.estimation import _compute_hc1_variance
        
        period_data = transformed_data[transformed_data['period'] == 4].copy()
        
        y_vals = period_data['y_dot'].values.astype(float)
        X = sm.add_constant(np.column_stack([
            period_data['d'].values.astype(float),
            period_data['x1'].values.astype(float),
            period_data['x2'].values.astype(float),
        ]))
        
        n, k = X.shape
        
        # OLS估计
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ (X.T @ y_vals)
        residuals = y_vals - X @ beta
        
        # 使用staggered模块的HC1计算 (需要XtX_inv作为第三个参数)
        var_staggered = _compute_hc1_variance(X, residuals, XtX_inv)
        
        # 自定义HC1
        correction = n / (n - k)
        omega = residuals ** 2 * correction
        meat = (X.T * omega) @ X
        var_custom = XtX_inv @ meat @ XtX_inv
        
        # 验证一致
        assert np.allclose(var_staggered, var_custom, rtol=1e-10), \
            "Staggered module HC1 should match custom implementation"
