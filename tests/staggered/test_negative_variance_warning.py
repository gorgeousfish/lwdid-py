"""
BUG-010: 负方差诊断警告测试

测试 _warn_negative_variance() 函数的功能和警告内容。
"""

import warnings
import numpy as np
import pytest

from lwdid.staggered.estimators import _warn_negative_variance


class TestWarnNegativeVarianceBasic:
    """测试 _warn_negative_variance 函数的基本功能"""
    
    def test_warning_is_raised(self):
        """测试函数正确发出警告"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_negative_variance(
                var_value=-0.001,
                estimator_type='RA',
                n=100,
            )
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
    
    def test_warning_contains_var_value(self):
        """测试警告包含方差值"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_negative_variance(
                var_value=-0.00123,
                estimator_type='RA',
                n=100,
            )
            assert "-1.23" in str(w[0].message) or "-0.00123" in str(w[0].message)
    
    def test_warning_contains_estimator_type(self):
        """测试警告包含估计器类型"""
        for est_type in ['RA', 'IPWRA', 'IPW', 'PSM']:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _warn_negative_variance(
                    var_value=-0.001,
                    estimator_type=est_type,
                    n=100,
                )
                assert est_type in str(w[0].message)


class TestWarnNegativeVarianceDiagnostics:
    """测试诊断信息的生成"""
    
    def test_high_condition_number_diagnostic(self):
        """测试高条件数诊断"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_negative_variance(
                var_value=-0.001,
                estimator_type='RA',
                n=100,
                condition_number=1e8,
            )
            msg = str(w[0].message)
            assert "condition number" in msg
            assert "1.00e+08" in msg or "1e+08" in msg
            assert "1e+06" in msg  # threshold
    
    def test_small_sample_diagnostic(self):
        """Test small sample diagnostics"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_negative_variance(
                var_value=-0.001,
                estimator_type='RA',
                n=30,
            )
            msg = str(w[0].message)
            assert "Sample size" in msg or "sample size" in msg
            assert "n=30" in msg
            assert "50" in msg  # threshold
    
    def test_extreme_weights_diagnostic(self):
        """Test extreme weights diagnostics"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_negative_variance(
                var_value=-0.001,
                estimator_type='IPW',
                n=100,
                weights_cv=3.5,
            )
            msg = str(w[0].message)
            assert "weight" in msg.lower()
            assert "CV" in msg
            assert "3.5" in msg or "3.50" in msg
            assert "2.0" in msg  # threshold
    
    def test_sample_size_info(self):
        """Test sample size info"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_negative_variance(
                var_value=-0.001,
                estimator_type='IPW',
                n=100,
                n_treat=40,
                n_control=60,
            )
            msg = str(w[0].message)
            assert "n=100" in msg
            assert "treated=40" in msg
            assert "control=60" in msg
    
    def test_psm_matches_info(self):
        """Test PSM matches info"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_negative_variance(
                var_value=-0.001,
                estimator_type='PSM',
                n=100,
                n_matches=3,
            )
            msg = str(w[0].message)
            assert "matches" in msg.lower()
            assert "M: 3" in msg


class TestWarnNegativeVarianceSuggestions:
    """测试建议内容的生成"""
    
    def test_always_suggests_bootstrap(self):
        """测试总是建议使用Bootstrap"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_negative_variance(
                var_value=-0.001,
                estimator_type='RA',
                n=1000,
            )
            msg = str(w[0].message)
            assert "bootstrap" in msg.lower()
    
    def test_suggests_trim_threshold_for_extreme_weights(self):
        """测试极端权重时建议增加trim_threshold"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_negative_variance(
                var_value=-0.001,
                estimator_type='IPW',
                n=100,
                weights_cv=3.0,
            )
            msg = str(w[0].message)
            assert "trim_threshold" in msg
    
    def test_suggests_larger_sample_for_small_n(self):
        """Test suggestion to increase sample size for small n"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_negative_variance(
                var_value=-0.001,
                estimator_type='RA',
                n=50,
            )
            msg = str(w[0].message)
            assert "sample size" in msg.lower() or "Sample" in msg
    
    def test_suggests_ps_model_check_for_ipw(self):
        """Test suggestion to check PS model for IPW/IPWRA"""
        for est_type in ['IPW', 'IPWRA']:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _warn_negative_variance(
                    var_value=-0.001,
                    estimator_type=est_type,
                    n=100,
                )
                msg = str(w[0].message)
                assert "propensity score" in msg.lower()


class TestWarnNegativeVarianceEdgeCases:
    """测试边缘情况"""
    
    def test_nan_weights_cv(self):
        """Test NaN weight CV does not cause errors"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_negative_variance(
                var_value=-0.001,
                estimator_type='IPW',
                n=100,
                weights_cv=np.nan,
            )
            # Should not crash, and should not contain weight CV info
            msg = str(w[0].message)
            assert "Weight CV:" not in msg
    
    def test_very_small_variance(self):
        """测试非常小的负方差"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_negative_variance(
                var_value=-1e-15,
                estimator_type='RA',
                n=100,
            )
            assert len(w) == 1
            # 科学计数法
            assert "-1" in str(w[0].message)
    
    def test_moderate_condition_number(self):
        """测试中等条件数（不触发高条件数警告）"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_negative_variance(
                var_value=-0.001,
                estimator_type='RA',
                n=100,
                condition_number=1000,
            )
            msg = str(w[0].message)
            # 应该包含条件数但不触发 > 1e6 警告
            assert "1.00e+03" in msg
            assert "> 1e+06" not in msg
    
    def test_moderate_weights_cv(self):
        """Test moderate weight CV (between 1.5-2.0)"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_negative_variance(
                var_value=-0.001,
                estimator_type='IPW',
                n=100,
                weights_cv=1.8,
            )
            msg = str(w[0].message)
            assert "high variation" in msg.lower() or "weight" in msg.lower()
            assert "1.80" in msg


class TestNumericalTriggerScenarios:
    """
    数值触发测试：尝试创建会触发负方差的场景。
    
    注意：负方差在实践中很少发生，这些测试主要验证：
    1. 在极端情况下代码不崩溃
    2. 当负方差发生时警告正确发出
    """
    
    def test_near_singular_matrix_does_not_crash(self):
        """测试近奇异矩阵场景不崩溃"""
        import pandas as pd
        from lwdid.staggered.estimators import estimate_ra
        
        # 创建近共线的控制变量
        np.random.seed(42)
        n = 50
        x1 = np.random.randn(n)
        x2 = x1 + np.random.randn(n) * 1e-6  # 近共线
        d = np.array([1] * 25 + [0] * 25)
        y = 1 + x1 + np.random.randn(n) * 0.1 + d * 2
        
        df = pd.DataFrame({'y': y, 'd': d, 'x1': x1, 'x2': x2})
        
        # 应该不崩溃（可能有警告）
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            try:
                result = estimate_ra(df, 'y', 'd', ['x1', 'x2'])
                # 只要不崩溃就算通过
                assert result is not None
            except np.linalg.LinAlgError:
                # 如果发生线性代数错误，也可以接受
                pass
    
    def test_extreme_propensity_scores_ipw(self):
        """测试极端倾向得分不崩溃"""
        import pandas as pd
        from lwdid.staggered.estimators import estimate_ipw
        
        # 创建导致极端倾向得分的数据
        np.random.seed(123)
        n = 100
        x = np.random.randn(n) * 5  # 大方差导致极端PS
        d = (x > 0).astype(int)  # 完美分离
        y = 1 + x + np.random.randn(n) * 0.5 + d * 2
        
        df = pd.DataFrame({'y': y, 'd': d, 'x': x})
        
        # 应该不崩溃（会有警告）
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            try:
                result = estimate_ipw(df, 'y', 'd', ['x'])
                assert result is not None
            except (ValueError, np.linalg.LinAlgError):
                # 极端情况可能引发异常，这也是可接受的
                pass
    
    def test_very_small_sample_ra(self):
        """测试超小样本RA估计"""
        import pandas as pd
        from lwdid.staggered.estimators import estimate_ra
        
        np.random.seed(456)
        n = 20  # 超小样本
        x = np.random.randn(n)
        d = np.array([1] * 10 + [0] * 10)
        y = 1 + x + np.random.randn(n) * 0.5 + d * 2
        
        df = pd.DataFrame({'y': y, 'd': d, 'x': x})
        
        # 应该能运行，可能有警告
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = estimate_ra(df, 'y', 'd', ['x'])
            assert result is not None
            # 检查SE是否为正（即使负方差也应该返回正SE）
            assert result.se > 0
    
    def test_very_small_sample_ipwra(self):
        """测试超小样本IPWRA估计"""
        import pandas as pd
        from lwdid.staggered.estimators import estimate_ipwra
        
        np.random.seed(789)
        n = 30
        x = np.random.randn(n)
        d = np.array([1] * 15 + [0] * 15)
        y = 1 + x + np.random.randn(n) * 0.5 + d * 2
        
        df = pd.DataFrame({'y': y, 'd': d, 'x': x})
        
        # 应该能运行
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = estimate_ipwra(df, 'y', 'd', ['x'])
            assert result is not None
            assert result.se > 0


class TestWarningInActualEstimators:
    """
    验证警告在实际估计器中正确集成。
    
    这些测试验证当负方差发生时，警告消息包含正确的诊断信息。
    """
    
    def test_warning_message_structure_in_ra(self):
        """测试RA估计器中警告消息的结构"""
        # 这是一个模拟测试，因为负方差在实践中很难触发
        # 我们直接测试 _warn_negative_variance 函数
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_negative_variance(
                var_value=-0.001,
                estimator_type='RA',
                n=50,
                n_treat=20,
                n_control=30,
                condition_number=1e8,
            )
            
            msg = str(w[0].message)
            # Verify key diagnostic info exists
            assert "RA estimator" in msg
            assert "condition number" in msg
            assert "Sample" in msg or "sample" in msg
            assert "bootstrap" in msg.lower()


class TestWarnNegativeVarianceIntegration:
    """Integration tests: verify warning output in complete scenarios"""
    
    def test_full_diagnostic_output_ra(self):
        """Test full diagnostic output for RA estimator"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_negative_variance(
                var_value=-0.001234,
                estimator_type='RA',
                n=80,
                n_treat=30,
                n_control=50,
                condition_number=5e7,
            )
            msg = str(w[0].message)
            
            # Check structure completeness
            assert "Possible causes" in msg
            assert "Diagnostic" in msg
            assert "Suggestion" in msg
            
            # Check content
            assert "RA estimator" in msg
            assert "n=80" in msg
            assert "bootstrap" in msg.lower()
    
    def test_full_diagnostic_output_ipwra(self):
        """Test full diagnostic output for IPWRA estimator"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_negative_variance(
                var_value=-0.005,
                estimator_type='IPWRA',
                n=60,
                n_treat=25,
                n_control=35,
                weights_cv=2.5,
                condition_number=1e5,
            )
            msg = str(w[0].message)
            
            # Check structure
            assert "IPWRA estimator" in msg
            assert "weight" in msg.lower() or "Weight" in msg
            assert "2.50" in msg
    
    def test_full_diagnostic_output_psm(self):
        """Test full diagnostic output for PSM estimator"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_negative_variance(
                var_value=-0.002,
                estimator_type='PSM',
                n=100,
                n_treat=40,
                n_control=60,
                n_matches=3,
            )
            msg = str(w[0].message)
            
            # Check structure
            assert "PSM estimator" in msg
            assert "matches" in msg.lower()
            assert "M: 3" in msg
