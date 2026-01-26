"""
BUG-060-L: 极小自由度推断警告测试

测试 run_ols_regression() 函数在 df_inference <= 2 时是否正确发出警告。

统计学背景：
- df=1: t分布退化为Cauchy分布，没有有限的均值和方差
- df=2: t分布有有限均值但方差是无穷大
- 在这些情况下，p值和置信区间的统计推断极不可靠

论文依据（Lee & Wooldridge 2023）：
"Provided N > K + 2, (2.18) can be estimated by OLS, and, under the conditional normality 
assumption, the t statistic has an exact T_{N-K-2} distribution"
"""

import warnings
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from lwdid.staggered.estimation import run_ols_regression


class TestSmallDfWarningBasic:
    """测试极小自由度警告的基本功能"""
    
    def test_df_1_triggers_warning(self):
        """测试 df=1 时触发警告（Cauchy分布）"""
        # 创建只有3个观测值、无控制变量的数据
        # df = n - k = 3 - 2 = 1 (截距和D系数)
        np.random.seed(42)
        df = pd.DataFrame({
            'y': [1.0, 2.0, 3.0],
            'd': [1, 0, 0],
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = run_ols_regression(df, 'y', 'd')
            
            # 应该有警告
            assert len(w) >= 1
            # 检查是否有自由度警告
            df_warnings = [x for x in w if 'degrees of freedom' in str(x.message).lower()]
            assert len(df_warnings) == 1
            
            msg = str(df_warnings[0].message)
            assert 'df=1' in msg
            assert 'Cauchy' in msg
            assert 'inference' in msg.lower()
    
    def test_df_2_triggers_warning(self):
        """测试 df=2 时触发警告（无穷方差）"""
        # 创建只有4个观测值、无控制变量的数据
        # df = n - k = 4 - 2 = 2
        np.random.seed(42)
        df = pd.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0],
            'd': [1, 1, 0, 0],
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = run_ols_regression(df, 'y', 'd')
            
            df_warnings = [x for x in w if 'degrees of freedom' in str(x.message).lower()]
            assert len(df_warnings) == 1
            
            msg = str(df_warnings[0].message)
            assert 'df=2' in msg
            assert 'infinite' in msg.lower()
    
    def test_df_3_no_warning(self):
        """测试 df=3 时不触发警告"""
        # 创建有5个观测值的数据
        # df = n - k = 5 - 2 = 3
        np.random.seed(42)
        df = pd.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0],
            'd': [1, 1, 0, 0, 0],
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = run_ols_regression(df, 'y', 'd')
            
            # 不应该有自由度警告
            df_warnings = [x for x in w if 'degrees of freedom' in str(x.message).lower()]
            assert len(df_warnings) == 0
    
    def test_df_10_no_warning(self):
        """测试较大 df 时不触发警告"""
        np.random.seed(42)
        n = 50
        df = pd.DataFrame({
            'y': np.random.randn(n),
            'd': [1] * 25 + [0] * 25,
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = run_ols_regression(df, 'y', 'd')
            
            df_warnings = [x for x in w if 'degrees of freedom' in str(x.message).lower()]
            assert len(df_warnings) == 0


class TestSmallDfWarningWithControls:
    """测试带控制变量时的自由度警告"""
    
    def test_df_1_with_controls(self):
        """测试带控制变量时 df=1 触发警告"""
        # n=4, k=3 (截距、D、1个控制变量), df=4-3=1
        np.random.seed(42)
        df = pd.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0],
            'd': [1, 1, 0, 0],
            'x': [0.5, 0.6, 0.4, 0.3],
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = run_ols_regression(df, 'y', 'd', controls=['x'])
            
            df_warnings = [x for x in w if 'degrees of freedom' in str(x.message).lower()]
            assert len(df_warnings) == 1
            assert 'df=1' in str(df_warnings[0].message)
    
    def test_df_2_with_multiple_controls(self):
        """测试带多个控制变量时 df=2 触发警告"""
        # n=8, k=6 (截距、D、2控制变量、2个D*X交互项), df=8-6=2
        # 需要足够的样本量确保控制变量被包含 (N_treated > K+1, N_control > K+1)
        np.random.seed(42)
        df = pd.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            'd': [1, 1, 1, 1, 0, 0, 0, 0],  # 4 treated, 4 control -> both > K+1=3
            'x1': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            'x2': [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = run_ols_regression(df, 'y', 'd', controls=['x1', 'x2'])
            
            # 验证控制变量确实被包含且 df=2
            assert result['df_inference'] == 2, f"Expected df=2, got df={result['df_inference']}"
            
            df_warnings = [x for x in w if 'degrees of freedom' in str(x.message).lower()]
            assert len(df_warnings) == 1
            assert 'df=2' in str(df_warnings[0].message)


class TestSmallDfWarningWithCluster:
    """测试聚类标准误时的自由度警告"""
    
    def test_cluster_df_1_warning(self):
        """测试聚类时 G-1=1 触发警告"""
        # 只有2个聚类时, df_inference = G - 1 = 1
        np.random.seed(42)
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
            
            # 应该有自由度警告
            df_warnings = [x for x in w if 'degrees of freedom' in str(x.message).lower()]
            assert len(df_warnings) == 1
            assert 'df=1' in str(df_warnings[0].message)
    
    def test_cluster_df_2_warning(self):
        """测试聚类时 G-1=2 触发警告"""
        # 只有3个聚类时, df_inference = G - 1 = 2
        np.random.seed(42)
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
            
            df_warnings = [x for x in w if 'degrees of freedom' in str(x.message).lower()]
            assert len(df_warnings) == 1
            assert 'df=2' in str(df_warnings[0].message)
    
    def test_cluster_df_3_no_warning(self):
        """测试聚类时 G-1=3 不触发警告"""
        # 4个聚类时, df_inference = G - 1 = 3
        np.random.seed(42)
        df = pd.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            'd': [1, 1, 0, 0, 0, 0, 0, 0],
            'cluster': [1, 1, 2, 2, 3, 3, 4, 4],
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = run_ols_regression(
                df, 'y', 'd', vce='cluster', cluster_var='cluster'
            )
            
            # 不应该有自由度警告（但可能有少聚类警告）
            df_warnings = [x for x in w if 'degrees of freedom' in str(x.message).lower()]
            assert len(df_warnings) == 0


class TestSmallDfWarningMessage:
    """测试警告消息内容"""
    
    def test_warning_mentions_cauchy(self):
        """测试 df=1 警告提及Cauchy分布"""
        np.random.seed(42)
        df = pd.DataFrame({
            'y': [1.0, 2.0, 3.0],
            'd': [1, 0, 0],
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = run_ols_regression(df, 'y', 'd')
            
            df_warnings = [x for x in w if 'degrees of freedom' in str(x.message).lower()]
            msg = str(df_warnings[0].message)
            assert 'Cauchy' in msg
            assert 'no finite moments' in msg or 'no finite mean' in msg
    
    def test_warning_mentions_infinite_variance(self):
        """测试 df=2 警告提及无穷方差"""
        np.random.seed(42)
        df = pd.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0],
            'd': [1, 1, 0, 0],
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = run_ols_regression(df, 'y', 'd')
            
            df_warnings = [x for x in w if 'degrees of freedom' in str(x.message).lower()]
            msg = str(df_warnings[0].message)
            assert 'infinite' in msg.lower()
    
    def test_warning_mentions_unreliable_inference(self):
        """测试警告提及推断不可靠"""
        np.random.seed(42)
        df = pd.DataFrame({
            'y': [1.0, 2.0, 3.0],
            'd': [1, 0, 0],
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = run_ols_regression(df, 'y', 'd')
            
            df_warnings = [x for x in w if 'degrees of freedom' in str(x.message).lower()]
            msg = str(df_warnings[0].message)
            assert 'unreliable' in msg.lower() or 'may be' in msg.lower()


class TestSmallDfResultsValidity:
    """测试极小自由度情况下结果的有效性"""
    
    def test_df_1_returns_valid_att(self):
        """测试 df=1 时仍返回有效的ATT点估计"""
        np.random.seed(42)
        df = pd.DataFrame({
            'y': [1.0, 2.0, 3.0],
            'd': [1, 0, 0],
        })
        
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = run_ols_regression(df, 'y', 'd')
        
        assert np.isfinite(result['att'])
        assert np.isfinite(result['se'])
        assert result['df_inference'] == 1
    
    def test_df_2_returns_valid_ci(self):
        """测试 df=2 时仍返回有效的置信区间"""
        np.random.seed(42)
        df = pd.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0],
            'd': [1, 1, 0, 0],
        })
        
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = run_ols_regression(df, 'y', 'd')
        
        # CI应该有效但可能非常宽
        assert np.isfinite(result['ci_lower'])
        assert np.isfinite(result['ci_upper'])
        assert result['ci_upper'] > result['ci_lower']
        assert result['df_inference'] == 2
    
    def test_df_1_ci_is_very_wide(self):
        """测试 df=1 时置信区间非常宽"""
        np.random.seed(42)
        df_small = pd.DataFrame({
            'y': [1.0, 2.0, 3.0],
            'd': [1, 0, 0],
        })
        
        # 比较 df=1 和 df=50 的置信区间宽度
        df_large = pd.DataFrame({
            'y': np.concatenate([[1.0], np.random.randn(51) + 2]),
            'd': [1] + [0] * 51,
        })
        
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result_small = run_ols_regression(df_small, 'y', 'd')
            result_large = run_ols_regression(df_large, 'y', 'd')
        
        ci_width_small = result_small['ci_upper'] - result_small['ci_lower']
        ci_width_large = result_large['ci_upper'] - result_large['ci_lower']
        
        # df=1 时 CI 应该更宽（相对于SE）
        # t_crit(df=1, 0.975) ≈ 12.71
        # t_crit(df=50, 0.975) ≈ 2.01
        assert result_small['df_inference'] == 1
        assert result_large['df_inference'] == 50


class TestSmallDfWithHeteroskedasticityRobust:
    """测试异方差稳健标准误时的自由度警告"""
    
    def test_hc0_df_1_warning(self):
        """测试 HC0 标准误 df=1 触发警告"""
        np.random.seed(42)
        df = pd.DataFrame({
            'y': [1.0, 2.0, 3.0],
            'd': [1, 0, 0],
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = run_ols_regression(df, 'y', 'd', vce='hc0')
            
            df_warnings = [x for x in w if 'degrees of freedom' in str(x.message).lower()]
            assert len(df_warnings) == 1
            assert 'df=1' in str(df_warnings[0].message)
    
    def test_hc3_df_2_warning(self):
        """测试 HC3 标准误 df=2 触发警告"""
        np.random.seed(42)
        df = pd.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0],
            'd': [1, 1, 0, 0],
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = run_ols_regression(df, 'y', 'd', vce='hc3')
            
            df_warnings = [x for x in w if 'degrees of freedom' in str(x.message).lower()]
            assert len(df_warnings) == 1
            assert 'df=2' in str(df_warnings[0].message)
    
    def test_robust_df_3_no_warning(self):
        """测试 robust 标准误 df=3 不触发警告"""
        np.random.seed(42)
        df = pd.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0],
            'd': [1, 1, 0, 0, 0],
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = run_ols_regression(df, 'y', 'd', vce='robust')
            
            df_warnings = [x for x in w if 'degrees of freedom' in str(x.message).lower()]
            assert len(df_warnings) == 0


class TestTDistributionMathematicalProperties:
    """
    验证 t 分布在小自由度时的数学特性。
    
    这些测试验证警告的统计学依据是正确的。
    """
    
    def test_t_df1_is_cauchy(self):
        """验证 df=1 时 t 分布等价于 Cauchy 分布"""
        # t(df=1) = Cauchy(0, 1)
        np.random.seed(42)
        
        # 从两个分布采样
        t_samples = stats.t.rvs(df=1, size=100000)
        cauchy_samples = stats.cauchy.rvs(size=100000)
        
        # 比较分位数 (Cauchy 没有均值和方差，所以比较分位数)
        t_quantiles = np.percentile(t_samples, [5, 25, 50, 75, 95])
        cauchy_quantiles = np.percentile(cauchy_samples, [5, 25, 50, 75, 95])
        
        # 应该非常接近（允许采样误差）
        np.testing.assert_array_almost_equal(
            t_quantiles, cauchy_quantiles, decimal=1
        )
    
    def test_t_df1_no_finite_mean(self):
        """验证 df=1 时 t 分布的样本均值不稳定"""
        np.random.seed(42)
        
        # 多次采样，观察样本均值的不稳定性
        means = []
        for _ in range(100):
            sample = stats.t.rvs(df=1, size=1000)
            means.append(np.mean(sample))
        
        # df=1 时样本均值应该有很大的变异
        mean_std = np.std(means)
        # 与 df=30 比较
        means_df30 = []
        for _ in range(100):
            sample = stats.t.rvs(df=30, size=1000)
            means_df30.append(np.mean(sample))
        mean_std_df30 = np.std(means_df30)
        
        # df=1 的样本均值变异应该远大于 df=30
        assert mean_std > mean_std_df30 * 5
    
    def test_t_df2_infinite_variance(self):
        """验证 df=2 时 t 分布的样本方差不稳定"""
        np.random.seed(42)
        
        # 多次采样，观察样本方差的增长
        variances = []
        for _ in range(100):
            sample = stats.t.rvs(df=2, size=1000)
            variances.append(np.var(sample))
        
        # df=2 时方差无穷大，样本方差变异很大
        var_std = np.std(variances)
        
        # 与 df=30 比较 (理论方差 = df/(df-2) = 30/28 ≈ 1.07)
        variances_df30 = []
        for _ in range(100):
            sample = stats.t.rvs(df=30, size=1000)
            variances_df30.append(np.var(sample))
        var_std_df30 = np.std(variances_df30)
        
        # df=2 的样本方差变异应该远大于 df=30
        assert var_std > var_std_df30 * 3
    
    def test_t_critical_values_increase_with_small_df(self):
        """验证小自由度时 t 临界值急剧增加"""
        alpha = 0.05
        t_crit_df1 = stats.t.ppf(1 - alpha/2, df=1)
        t_crit_df2 = stats.t.ppf(1 - alpha/2, df=2)
        t_crit_df3 = stats.t.ppf(1 - alpha/2, df=3)
        t_crit_df30 = stats.t.ppf(1 - alpha/2, df=30)
        t_crit_inf = stats.norm.ppf(1 - alpha/2)  # ≈ 1.96
        
        # 验证临界值的单调性
        assert t_crit_df1 > t_crit_df2 > t_crit_df3 > t_crit_df30 > t_crit_inf
        
        # df=1 的临界值应该非常大
        assert t_crit_df1 > 10  # 实际约 12.71
        assert t_crit_df2 > 4   # 实际约 4.30
        assert t_crit_df3 > 3   # 实际约 3.18


class TestWarningStacklevel:
    """测试警告的 stacklevel 正确"""
    
    def test_warning_points_to_caller(self):
        """测试警告指向调用者而不是函数内部"""
        np.random.seed(42)
        df = pd.DataFrame({
            'y': [1.0, 2.0, 3.0],
            'd': [1, 0, 0],
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = run_ols_regression(df, 'y', 'd')  # line to capture
            
            df_warnings = [x for x in w if 'degrees of freedom' in str(x.message).lower()]
            assert len(df_warnings) == 1
            
            # 警告应该指向这个文件，而不是 estimation.py 内部
            # （由于 stacklevel=2）
            warning_obj = df_warnings[0]
            assert issubclass(warning_obj.category, UserWarning)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
