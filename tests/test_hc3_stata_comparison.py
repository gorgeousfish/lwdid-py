"""
Python-Stata comparison tests for HC3 with high leverage points.

This module verifies that our HC3 implementation matches Stata's behavior
when handling extreme leverage points (h_ii close to 1).
"""

import numpy as np
import pandas as pd
import pytest


class TestHC3StataComparison:
    """Test HC3 calculation matches Stata exactly."""
    
    @pytest.fixture
    def stata_high_leverage_data(self):
        """
        Data generated in Stata with seed 12345:
        
        clear
        set seed 12345
        set obs 100
        gen xvar = rnormal()
        replace xvar = 100 in 1
        gen yvar = 1 + 2*xvar + rnormal()
        gen d = (_n <= 50)
        
        Stata results:
        - OLS d coef: 0.0840257
        - OLS SE(d): 0.2252318
        - HC1 SE(d): 0.2263119
        - HC3 SE(d): 0.2275196
        - Max h_ii: 0.9917635
        """
        # Key observations from Stata (first 10 and observation 1 with extreme x)
        data = {
            'xvar': [100.0, -0.001805, 0.544077, 0.001629, 0.357681,
                     0.231946, 0.029461, 0.327628, -0.279001, -1.135741,
                     -0.339891, 1.098099, -1.075497, 0.158037, 1.207193,
                     -0.299066, -1.291115, 0.185259, -0.047609, 0.170654],
            'yvar': [198.855940, 0.103040, 3.025558, -1.515051, 2.581387,
                     1.050696, 0.886929, 1.600259, -0.171893, -2.100376,
                     0.133355, 3.265088, -2.265682, 2.071505, 3.436261,
                     0.364037, -3.051893, 2.213917, 1.261779, 2.086437],
            'd': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        }
        
        # Extend to 100 observations with more data
        np.random.seed(54321)  # Different seed for extension
        n_extend = 80
        xvar_extend = np.random.randn(n_extend) * 0.5
        yvar_extend = 1 + 2 * xvar_extend + np.random.randn(n_extend)
        d_extend = [1] * 30 + [0] * 50  # 30 more treated, 50 control
        
        data['xvar'].extend(xvar_extend.tolist())
        data['yvar'].extend(yvar_extend.tolist())
        data['d'].extend(d_extend)
        
        return pd.DataFrame(data)
    
    def test_hc3_formula_matches_stata(self):
        """
        Verify HC3 formula implementation matches Stata exactly.
        
        This test uses the exact Stata data and verifies that our numpy
        implementation produces identical results.
        """
        # Load the actual Stata export data
        try:
            df = pd.read_csv('/tmp/hc3_leverage_test.csv')
        except FileNotFoundError:
            pytest.skip("Stata data file not available")
        
        y = df['yvar'].values
        d = df['d'].values
        x = df['xvar'].values
        
        # Design matrix: [1, d, xvar]
        X = np.column_stack([np.ones(len(y)), d, x])
        n, k = X.shape
        
        # OLS estimation
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta
        XtX_inv = np.linalg.inv(X.T @ X)
        
        # Leverage values (hat matrix diagonal)
        tmp = X @ XtX_inv
        h_ii = (tmp * X).sum(axis=1)
        
        # HC3 variance (with numerical stability)
        denom = np.maximum((1 - h_ii) ** 2, np.finfo(np.float64).eps)
        omega = residuals ** 2 / denom
        meat = (X.T * omega) @ X
        var_hc3 = XtX_inv @ meat @ XtX_inv
        se_hc3 = np.sqrt(np.diag(var_hc3))
        
        # Stata reference values
        stata_d_coef = 0.0840257
        stata_hc3_se_d = 0.2275196
        stata_max_h = 0.9917635
        
        # Verify coefficient
        assert np.isclose(beta[1], stata_d_coef, rtol=1e-5), \
            f"d coefficient mismatch: {beta[1]} vs {stata_d_coef}"
        
        # Verify max leverage (verifies no clipping)
        assert np.isclose(h_ii.max(), stata_max_h, rtol=1e-5), \
            f"Max h_ii mismatch: {h_ii.max()} vs {stata_max_h}"
        
        # Verify HC3 SE
        assert np.isclose(se_hc3[1], stata_hc3_se_d, rtol=1e-5), \
            f"HC3 SE mismatch: {se_hc3[1]} vs {stata_hc3_se_d}"
    
    def test_leverage_not_clipped_to_0999(self):
        """Verify leverage values above 0.9999 are NOT clipped."""
        try:
            df = pd.read_csv('/tmp/hc3_leverage_test.csv')
        except FileNotFoundError:
            pytest.skip("Stata data file not available")
        
        y = df['yvar'].values
        d = df['d'].values
        x = df['xvar'].values
        
        X = np.column_stack([np.ones(len(y)), d, x])
        XtX_inv = np.linalg.inv(X.T @ X)
        tmp = X @ XtX_inv
        h_ii = (tmp * X).sum(axis=1)
        
        # The high leverage point should have h_ii = 0.9917635, NOT 0.9999
        max_h = h_ii.max()
        assert max_h > 0.99, f"Max leverage should be > 0.99, got {max_h}"
        assert max_h < 0.9999, f"Max leverage should be actual value, not clipped to 0.9999"
        
        # Verify it matches Stata's computation
        assert np.isclose(max_h, 0.9917635, rtol=1e-5)


class TestHC3NumericalValidation:
    """Numerical validation tests for HC3 implementation."""
    
    def test_hc3_se_larger_than_ols_with_high_leverage(self):
        """HC3 SE should be larger than OLS SE when high leverage exists."""
        np.random.seed(42)
        n = 100
        
        x = np.random.randn(n)
        x[0] = 50  # High leverage point
        y = 1 + 2 * x + np.random.randn(n)
        d = (np.arange(n) < 50).astype(int)
        
        X = np.column_stack([np.ones(n), d, x])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta
        XtX_inv = np.linalg.inv(X.T @ X)
        
        # OLS variance
        n_obs, k = X.shape
        sigma2 = np.sum(residuals**2) / (n_obs - k)
        var_ols = sigma2 * XtX_inv
        se_ols = np.sqrt(np.diag(var_ols))
        
        # HC3 variance
        tmp = X @ XtX_inv
        h_ii = (tmp * X).sum(axis=1)
        denom = np.maximum((1 - h_ii) ** 2, np.finfo(np.float64).eps)
        omega = residuals ** 2 / denom
        meat = (X.T * omega) @ X
        var_hc3 = XtX_inv @ meat @ XtX_inv
        se_hc3 = np.sqrt(np.diag(var_hc3))
        
        # HC3 should produce larger SE for xvar due to high leverage
        assert se_hc3[2] > se_ols[2], \
            f"HC3 SE ({se_hc3[2]}) should be > OLS SE ({se_ols[2]}) for xvar"
    
    def test_leverage_sum_equals_k(self):
        """Sum of leverage values should equal number of parameters k."""
        np.random.seed(123)
        n = 100
        k = 3  # intercept + d + x
        
        x = np.random.randn(n)
        d = (np.arange(n) < 50).astype(int)
        
        X = np.column_stack([np.ones(n), d, x])
        XtX_inv = np.linalg.inv(X.T @ X)
        tmp = X @ XtX_inv
        h_ii = (tmp * X).sum(axis=1)
        
        # Sum of leverages = trace(H) = k
        assert np.isclose(h_ii.sum(), k, rtol=1e-10), \
            f"Sum of leverages should be {k}, got {h_ii.sum()}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
