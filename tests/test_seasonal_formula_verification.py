"""
Formula Verification Tests for Seasonal Adjustment.

Task 4.5: Verify OLS normal equations and seasonal dummy encoding.

This test uses numpy/scipy to verify the mathematical formulas match
the paper's specifications exactly.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import linalg

from lwdid.transformations import demeanq_unit, detrendq_unit


class TestOLSNormalEquations:
    """Verify OLS normal equations: β̂ = (X'X)^{-1} X'y"""
    
    def test_demeanq_ols_solution(self):
        """
        Verify demeanq uses correct OLS solution.
        
        For demeanq with Q=4, the design matrix is:
        X = [1, D_2, D_3, D_4] where D_q = 1 if quarter=q
        
        The OLS solution is: β̂ = (X'X)^{-1} X'y
        """
        # Create data with 8 pre-treatment periods (2 complete years)
        data = pd.DataFrame({
            't': list(range(1, 13)),
            'quarter': [1, 2, 3, 4] * 3,
            'y': [100, 105, 110, 103, 101, 106, 111, 104, 99, 104, 109, 102],
            'post': [0] * 8 + [1] * 4
        })
        
        # Build design matrix manually for pre-treatment
        pre_data = data[data['post'] == 0]
        n_pre = len(pre_data)
        
        # X = [1, D_2, D_3, D_4]
        X = np.zeros((n_pre, 4))
        X[:, 0] = 1  # Intercept
        for i, q in enumerate(pre_data['quarter']):
            if q == 2:
                X[i, 1] = 1
            elif q == 3:
                X[i, 2] = 1
            elif q == 4:
                X[i, 3] = 1
        
        y = pre_data['y'].values
        
        # Compute OLS solution: β̂ = (X'X)^{-1} X'y
        XtX = X.T @ X
        Xty = X.T @ y
        beta_manual = linalg.solve(XtX, Xty)
        
        # Apply demeanq
        yhat, ydot = demeanq_unit(data, 'y', 'quarter', 'post', Q=4)
        
        # Verify fitted values match manual calculation
        for i in range(n_pre):
            q = pre_data.iloc[i]['quarter']
            expected_yhat = beta_manual[0]  # mu
            if q == 2:
                expected_yhat += beta_manual[1]
            elif q == 3:
                expected_yhat += beta_manual[2]
            elif q == 4:
                expected_yhat += beta_manual[3]
            
            assert_allclose(yhat[i], expected_yhat, atol=1e-10,
                           err_msg=f"Fitted value at index {i} doesn't match OLS solution")
    
    def test_detrendq_ols_solution(self):
        """
        Verify detrendq uses correct OLS solution.
        
        For detrendq with Q=4, the design matrix is:
        X = [1, t, D_2, D_3, D_4]
        
        With time centering: t_centered = t - mean(t_pre)
        """
        # Create data
        data = pd.DataFrame({
            't': list(range(1, 13)),
            'quarter': [1, 2, 3, 4] * 3,
            'y': [100 + 0.5*t + {1:0, 2:5, 3:10, 4:3}[((t-1)%4)+1] 
                  for t in range(1, 13)],
            'post': [0] * 8 + [1] * 4
        })
        
        # Build design matrix for pre-treatment
        pre_data = data[data['post'] == 0]
        n_pre = len(pre_data)
        
        # Time centering
        t_pre = pre_data['t'].values
        t_mean = np.mean(t_pre)
        t_centered = t_pre - t_mean
        
        # X = [1, t_centered, D_2, D_3, D_4]
        X = np.zeros((n_pre, 5))
        X[:, 0] = 1  # Intercept
        X[:, 1] = t_centered  # Centered time
        for i, q in enumerate(pre_data['quarter']):
            if q == 2:
                X[i, 2] = 1
            elif q == 3:
                X[i, 3] = 1
            elif q == 4:
                X[i, 4] = 1
        
        y = pre_data['y'].values
        
        # Compute OLS solution
        XtX = X.T @ X
        Xty = X.T @ y
        beta_manual = linalg.solve(XtX, Xty)
        
        # Apply detrendq
        yhat, ydot = detrendq_unit(data, 'y', 't', 'quarter', 'post', Q=4)
        
        # Verify pre-treatment residuals are near zero
        pre_residuals = ydot[:n_pre]
        assert_allclose(pre_residuals, 0, atol=1e-8,
                       err_msg="Pre-treatment residuals should be near zero")


class TestSeasonalDummyEncoding:
    """Verify seasonal dummy variable encoding."""
    
    def test_reference_category_q1(self):
        """
        Verify Q=1 is the reference category (no dummy).
        
        The model is: Y = μ + γ_2 D_2 + γ_3 D_3 + γ_4 D_4 + ε
        where D_q = 1 if quarter = q, 0 otherwise.
        
        This means γ_1 = 0 (absorbed into μ).
        """
        # Create data with enough pre-treatment observations (need > Q parameters)
        # 8 pre-treatment periods (2 complete years), 4 post-treatment
        data = pd.DataFrame({
            't': list(range(1, 13)),
            'quarter': [1, 2, 3, 4] * 3,
            'y': [100, 110, 120, 105] * 3,  # Perfect pattern
            'post': [0] * 8 + [1] * 4
        })
        
        yhat, ydot = demeanq_unit(data, 'y', 'quarter', 'post', Q=4)
        
        # Fitted values for Q1 should be 100 (the intercept)
        q1_indices = [i for i in range(8) if data.iloc[i]['quarter'] == 1]
        for idx in q1_indices:
            assert_allclose(yhat[idx], 100, atol=1e-10,
                           err_msg="Q1 fitted value should equal intercept")
        
        # Fitted values for Q2 should be 110 (intercept + γ_2)
        q2_indices = [i for i in range(8) if data.iloc[i]['quarter'] == 2]
        for idx in q2_indices:
            assert_allclose(yhat[idx], 110, atol=1e-10,
                           err_msg="Q2 fitted value should be 110")
    
    def test_monthly_encoding_q12(self):
        """
        Verify monthly encoding with Q=12.
        
        Reference category is month 1.
        """
        # Create monthly data with enough pre-treatment observations
        # Need > 12 pre-treatment observations, so use 24 months pre-treatment
        gamma = {m: m * 5 for m in range(1, 13)}  # gamma_m = 5*m
        gamma[1] = 0  # Reference
        mu = 100
        
        data = []
        for t in range(1, 49):  # 48 months total
            month = ((t - 1) % 12) + 1
            y = mu + gamma[month]
            post = 1 if t > 24 else 0  # 24 pre-treatment, 24 post-treatment
            data.append({'t': t, 'month': month, 'y': y, 'post': post})
        
        df = pd.DataFrame(data)
        
        yhat, ydot = demeanq_unit(df, 'y', 'month', 'post', Q=12)
        
        # Verify fitted values for pre-treatment
        for idx in range(24):  # Pre-treatment
            m = df.iloc[idx]['month']
            expected = mu + gamma[m]
            assert_allclose(yhat[idx], expected, atol=1e-10,
                           err_msg=f"Month {m} fitted value incorrect")


class TestResidualCalculation:
    """Verify residual calculation: ε̂ = y - ŷ"""
    
    def test_residual_formula_demeanq(self):
        """
        Verify: Ẏ_{it} = Y_{it} - ŷ_{it}
        where ŷ_{it} = μ̂ + Σ_{q=2}^{Q} γ̂_q D_q
        """
        # Need more pre-treatment observations than parameters (> 4)
        data = pd.DataFrame({
            't': list(range(1, 13)),
            'quarter': [1, 2, 3, 4] * 3,
            'y': [100, 105, 110, 103, 102, 107, 112, 105, 101, 106, 111, 104],
            'post': [0] * 8 + [1] * 4
        })
        
        yhat, ydot = demeanq_unit(data, 'y', 'quarter', 'post', Q=4)
        
        # Verify residual = y - yhat
        y = data['y'].values
        expected_residuals = y - yhat
        
        assert_allclose(ydot, expected_residuals, atol=1e-10,
                       err_msg="Residuals should equal y - yhat")
    
    def test_residual_formula_detrendq(self):
        """
        Verify: Ÿ_{it} = Y_{it} - ŷ_{it}
        where ŷ_{it} = α̂ + β̂·t + Σ_{q=2}^{Q} γ̂_q D_q
        """
        # Need more pre-treatment observations than parameters (> 5)
        data = pd.DataFrame({
            't': list(range(1, 13)),
            'quarter': [1, 2, 3, 4] * 3,
            'y': [100 + 0.5*t + {1:0, 2:5, 3:10, 4:3}[((t-1)%4)+1] 
                  for t in range(1, 13)],
            'post': [0] * 8 + [1] * 4
        })
        
        yhat, ydot = detrendq_unit(data, 'y', 't', 'quarter', 'post', Q=4)
        
        # Verify residual = y - yhat
        y = data['y'].values
        expected_residuals = y - yhat
        
        assert_allclose(ydot, expected_residuals, atol=1e-10,
                       err_msg="Residuals should equal y - yhat")


class TestNumericalStability:
    """Test numerical stability for high-dimensional cases."""
    
    def test_condition_number_q52(self):
        """
        Verify design matrix condition number is reasonable for Q=52.
        
        A well-conditioned matrix has condition number close to 1.
        For seasonal dummies, condition number should be O(sqrt(Q)).
        """
        # Create weekly data
        n_weeks = 156  # 3 years
        data = pd.DataFrame({
            't': list(range(1, n_weeks + 1)),
            'week': [((t - 1) % 52) + 1 for t in range(1, n_weeks + 1)],
            'y': [100 + np.sin(2 * np.pi * t / 52) * 10 for t in range(1, n_weeks + 1)],
            'post': [0] * 104 + [1] * 52
        })
        
        # Build design matrix
        pre_data = data[data['post'] == 0]
        n_pre = len(pre_data)
        
        X = np.zeros((n_pre, 52))
        X[:, 0] = 1  # Intercept
        for i, w in enumerate(pre_data['week']):
            if w > 1:
                X[i, w - 1] = 1
        
        # Compute condition number
        cond = np.linalg.cond(X)
        
        # Condition number should be reasonable (< 1000 for Q=52)
        assert cond < 1000, f"Condition number {cond} too high for Q=52"
    
    def test_no_numerical_overflow_q52(self):
        """
        Verify no numerical overflow for Q=52.
        """
        # Create weekly data with large values
        n_weeks = 156
        data = pd.DataFrame({
            't': list(range(1, n_weeks + 1)),
            'week': [((t - 1) % 52) + 1 for t in range(1, n_weeks + 1)],
            'y': [1e6 + np.sin(2 * np.pi * t / 52) * 1e4 for t in range(1, n_weeks + 1)],
            'post': [0] * 104 + [1] * 52
        })
        
        # Should not raise any errors
        yhat, ydot = demeanq_unit(data, 'y', 'week', 'post', Q=52)
        
        # Verify no NaN or Inf
        assert not np.any(np.isnan(yhat)), "yhat contains NaN"
        assert not np.any(np.isinf(yhat)), "yhat contains Inf"
        assert not np.any(np.isnan(ydot)), "ydot contains NaN"
        assert not np.any(np.isinf(ydot)), "ydot contains Inf"


class TestDesignMatrixProperties:
    """Test design matrix mathematical properties."""
    
    def test_design_matrix_rank_demeanq(self):
        """
        Verify design matrix has full column rank.
        
        For demeanq with Q=4, X is n×4 and should have rank 4.
        """
        # Need at least 4 pre-treatment observations with all quarters represented
        data = pd.DataFrame({
            't': list(range(1, 13)),
            'quarter': [1, 2, 3, 4] * 3,
            'y': [100, 105, 110, 103, 101, 106, 111, 104, 99, 104, 109, 102],
            'post': [0] * 8 + [1] * 4
        })
        
        pre_data = data[data['post'] == 0]
        n_pre = len(pre_data)
        
        # Build design matrix
        X = np.zeros((n_pre, 4))
        X[:, 0] = 1
        for i, q in enumerate(pre_data['quarter']):
            if q == 2:
                X[i, 1] = 1
            elif q == 3:
                X[i, 2] = 1
            elif q == 4:
                X[i, 3] = 1
        
        # Check rank
        rank = np.linalg.matrix_rank(X)
        assert rank == 4, f"Design matrix should have rank 4, got {rank}"
    
    def test_xtx_invertible(self):
        """
        Verify X'X is invertible (positive definite).
        """
        # Need at least 4 pre-treatment observations with all quarters
        data = pd.DataFrame({
            't': list(range(1, 13)),
            'quarter': [1, 2, 3, 4] * 3,
            'y': [100, 105, 110, 103, 101, 106, 111, 104, 99, 104, 109, 102],
            'post': [0] * 8 + [1] * 4
        })
        
        pre_data = data[data['post'] == 0]
        n_pre = len(pre_data)
        
        X = np.zeros((n_pre, 4))
        X[:, 0] = 1
        for i, q in enumerate(pre_data['quarter']):
            if q == 2:
                X[i, 1] = 1
            elif q == 3:
                X[i, 2] = 1
            elif q == 4:
                X[i, 3] = 1
        
        XtX = X.T @ X
        
        # Check positive definiteness via eigenvalues
        eigenvalues = np.linalg.eigvalsh(XtX)
        assert np.all(eigenvalues > 0), "X'X should be positive definite"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
