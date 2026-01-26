"""
BUG-060-L: Python-Stata End-to-End Validation

Validates that Python results match Stata exactly at small degrees of freedom,
and that Python correctly issues warnings while Stata does not.

Test data and expected values are from actual Stata runs.
"""

import numpy as np
import pandas as pd
import warnings
import pytest

from lwdid.staggered.estimation import run_ols_regression


class TestStataComparison:
    """Compare Python results with Stata at small df"""
    
    def test_df1_stata_match(self):
        """
        df=1 case: n=3, k=2
        
        Stata code:
            clear all
            input y d
            1.0 1
            2.0 0
            3.0 0
            end
            regress y d
        
        Stata results:
            ATT = -1.5
            SE = 0.8660254
            df = 1
            t-stat = -1.7320508
            p-value = 0.33333333
            95% CI = [-12.5039, 9.503896]
        """
        df = pd.DataFrame({
            'y': [1.0, 2.0, 3.0],
            'd': [1, 0, 0],
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = run_ols_regression(df, 'y', 'd')
            
            # Verify numerical results match Stata
            np.testing.assert_allclose(result['att'], -1.5, rtol=1e-6)
            np.testing.assert_allclose(result['se'], 0.8660254, rtol=1e-5)
            assert result['df_inference'] == 1
            np.testing.assert_allclose(result['t_stat'], -1.7320508, rtol=1e-5)
            np.testing.assert_allclose(result['pvalue'], 0.33333333, rtol=1e-5)
            np.testing.assert_allclose(result['ci_lower'], -12.5039, rtol=1e-3)
            np.testing.assert_allclose(result['ci_upper'], 9.5039, rtol=1e-3)
            
            # Python should issue warning, Stata does not
            df_warnings = [x for x in w if 'degrees of freedom' in str(x.message).lower()]
            assert len(df_warnings) == 1
    
    def test_df2_stata_match(self):
        """
        df=2 case: n=4, k=2
        
        Stata code:
            clear all
            input y d
            1.0 1
            2.0 1
            3.0 0
            4.0 0
            end
            regress y d
        
        Stata results:
            ATT = -2
            SE = 0.70710678
            df = 2
            t-stat = -2.8284271
            p-value = 0.10557281
            95% CI = [-5.042435, 1.042435]
        """
        df = pd.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0],
            'd': [1, 1, 0, 0],
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = run_ols_regression(df, 'y', 'd')
            
            # Verify numerical results match Stata
            np.testing.assert_allclose(result['att'], -2.0, rtol=1e-6)
            np.testing.assert_allclose(result['se'], 0.70710678, rtol=1e-5)
            assert result['df_inference'] == 2
            np.testing.assert_allclose(result['t_stat'], -2.8284271, rtol=1e-5)
            np.testing.assert_allclose(result['pvalue'], 0.10557281, rtol=1e-4)
            np.testing.assert_allclose(result['ci_lower'], -5.042435, rtol=1e-4)
            np.testing.assert_allclose(result['ci_upper'], 1.042435, rtol=1e-4)
            
            # Python should issue warning
            df_warnings = [x for x in w if 'degrees of freedom' in str(x.message).lower()]
            assert len(df_warnings) == 1
    
    def test_df3_stata_match_no_warning(self):
        """
        df=3 case: n=5, k=2 - should NOT trigger warning
        """
        df = pd.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0],
            'd': [1, 1, 0, 0, 0],
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = run_ols_regression(df, 'y', 'd')
            
            assert result['df_inference'] == 3
            
            # Should NOT have df warning
            df_warnings = [x for x in w if 'degrees of freedom' in str(x.message).lower()]
            assert len(df_warnings) == 0


class TestCriticalValuesMatch:
    """Verify t-distribution critical values match Stata"""
    
    def test_critical_values_match_stata(self):
        """
        Stata values from:
            display invttail(1, 0.025)  -> 12.706205
            display invttail(2, 0.025)  -> 4.3026527
            display invttail(3, 0.025)  -> 3.1824463
            display invttail(10, 0.025) -> 2.2281389
            display invttail(30, 0.025) -> 2.0422725
            display invnormal(0.975)    -> 1.959964
        """
        from scipy import stats
        
        # Stata uses invttail which gives the value c such that P(T > c) = p
        # This is equivalent to ppf(1-p) for upper tail
        # For 95% CI, we need invttail(df, 0.025) = ppf(0.975)
        
        stata_values = {
            1: 12.706205,
            2: 4.3026527,
            3: 3.1824463,
            10: 2.2281389,
            30: 2.0422725,
        }
        
        for df, stata_t in stata_values.items():
            python_t = stats.t.ppf(0.975, df)
            np.testing.assert_allclose(python_t, stata_t, rtol=1e-5,
                err_msg=f"df={df}: Python {python_t} != Stata {stata_t}")
        
        # Normal distribution
        python_z = stats.norm.ppf(0.975)
        stata_z = 1.959964
        np.testing.assert_allclose(python_z, stata_z, rtol=1e-5)


class TestClusterSmallDfStata:
    """Test cluster-robust SE at small df"""
    
    def test_cluster_df1_matches_stata_behavior(self):
        """
        With 2 clusters, df_inference = G - 1 = 1
        
        Note: Stata would compute cluster-robust SE but with df=1,
        inference is based on Cauchy distribution.
        """
        df = pd.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'd': [1, 1, 1, 0, 0, 0],
            'cluster': [1, 1, 1, 2, 2, 2],
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = run_ols_regression(
                df, 'y', 'd', vce='cluster', cluster_var='cluster'
            )
            
            # df_inference should be G - 1 = 1
            assert result['df_inference'] == 1
            
            # Should have df warning
            df_warnings = [x for x in w if 'degrees of freedom' in str(x.message).lower()]
            assert len(df_warnings) == 1
    
    def test_cluster_df2_matches_stata_behavior(self):
        """
        With 3 clusters, df_inference = G - 1 = 2
        """
        df = pd.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'd': [1, 1, 0, 0, 0, 0],
            'cluster': [1, 1, 2, 2, 3, 3],
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = run_ols_regression(
                df, 'y', 'd', vce='cluster', cluster_var='cluster'
            )
            
            assert result['df_inference'] == 2
            
            df_warnings = [x for x in w if 'degrees of freedom' in str(x.message).lower()]
            assert len(df_warnings) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
