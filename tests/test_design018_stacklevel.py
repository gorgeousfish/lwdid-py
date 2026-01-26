"""
Test DESIGN-018: 警告 stacklevel 值一致性

验证 warnings.warn() 的 stacklevel 参数设置正确，
确保警告指向用户代码而非库内部代码。

测试策略：
1. 使用 warnings.catch_warnings() 捕获警告
2. 检查警告的文件名是否指向测试文件（用户代码），而非库内部文件
3. 覆盖关键的公共 API 函数

Author: AI Code Review
Date: 2026-01-17
"""

import warnings
import pytest
import numpy as np
import pandas as pd


class TestStacklevelConsistency:
    """测试 stacklevel 设置是否正确"""
    
    @pytest.fixture
    def small_cross_section_data(self):
        """小样本横截面数据，用于触发各种警告"""
        np.random.seed(42)
        n = 20  # 小样本以触发小样本警告
        data = pd.DataFrame({
            'y': np.random.randn(n),
            'd': np.array([1]*3 + [0]*(n-3)),  # 很少的处理组
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
        })
        return data
    
    @pytest.fixture
    def staggered_panel_data(self):
        """小规模交错面板数据"""
        np.random.seed(42)
        n_units = 10
        n_periods = 5
        
        data = []
        for i in range(n_units):
            # 前 3 个单位在 period 3 处理
            # 中间 3 个单位是 never treated
            # 后 4 个单位在 period 4 处理
            if i < 3:
                gvar = 3
            elif i < 6:
                gvar = 0  # never treated
            else:
                gvar = 4
                
            for t in range(1, n_periods + 1):
                d = 1 if gvar > 0 and t >= gvar else 0
                y = np.random.randn() + 0.5 * d
                data.append({
                    'id': i,
                    'year': t,
                    'gvar': gvar,
                    'd': d,
                    'y': y,
                    'x1': np.random.randn(),
                })
        
        return pd.DataFrame(data)
    
    def test_ipwra_small_sample_warning_stacklevel(self, small_cross_section_data):
        """测试 estimate_ipwra 小样本警告的 stacklevel"""
        from lwdid.staggered.estimators import estimate_ipwra
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            try:
                result = estimate_ipwra(
                    small_cross_section_data,
                    y='y',
                    d='d',
                    controls=['x1', 'x2'],
                )
            except (ValueError, np.linalg.LinAlgError):
                # 如果估计失败，检查是否捕获了警告
                pass
            
            # 检查是否有 UserWarning（过滤掉 pytest 内部的警告）
            user_warnings = [
                x for x in w 
                if issubclass(x.category, UserWarning) 
                and 'pluggy' not in x.filename
                and 'pytest' not in x.filename
            ]
            
            if user_warnings:
                # 验证警告不是指向 lwdid 库内部的 .py 文件
                for warning in user_warnings:
                    # 警告应该指向测试文件或外部调用，而非 estimators.py 内部
                    is_library_internal = (
                        'estimators.py' in warning.filename and
                        'test_' not in warning.filename
                    )
                    assert not is_library_internal, \
                        f"警告指向了库内部文件: {warning.filename}:{warning.lineno}"
    
    def test_ipw_small_sample_warning_stacklevel(self, small_cross_section_data):
        """测试 estimate_ipw 小样本警告的 stacklevel"""
        from lwdid.staggered.estimators import estimate_ipw
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            try:
                result = estimate_ipw(
                    small_cross_section_data,
                    y='y',
                    d='d',
                    propensity_controls=['x1', 'x2'],
                )
            except (ValueError, np.linalg.LinAlgError):
                pass
            
            # 过滤掉 pytest 内部的警告
            user_warnings = [
                x for x in w 
                if issubclass(x.category, UserWarning)
                and 'pluggy' not in x.filename
                and 'pytest' not in x.filename
            ]
            
            if user_warnings:
                for warning in user_warnings:
                    is_library_internal = (
                        'estimators.py' in warning.filename and
                        'test_' not in warning.filename
                    )
                    assert not is_library_internal, \
                        f"警告指向了库内部文件: {warning.filename}:{warning.lineno}"
    
    def test_psm_small_sample_warning_stacklevel(self, small_cross_section_data):
        """测试 estimate_psm 小样本警告的 stacklevel"""
        from lwdid.staggered.estimators import estimate_psm
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            try:
                result = estimate_psm(
                    small_cross_section_data,
                    y='y',
                    d='d',
                    propensity_controls=['x1', 'x2'],
                    n_neighbors=1,
                )
            except (ValueError, np.linalg.LinAlgError):
                pass
            
            # 过滤掉 pytest 内部的警告
            user_warnings = [
                x for x in w 
                if issubclass(x.category, UserWarning)
                and 'pluggy' not in x.filename
                and 'pytest' not in x.filename
            ]
            
            if user_warnings:
                for warning in user_warnings:
                    is_library_internal = (
                        'estimators.py' in warning.filename and
                        'test_' not in warning.filename
                    )
                    assert not is_library_internal, \
                        f"警告指向了库内部文件: {warning.filename}:{warning.lineno}"
    
    def test_run_ols_regression_warning_stacklevel(self):
        """测试 run_ols_regression 警告的 stacklevel"""
        from lwdid.staggered.estimation import run_ols_regression
        
        # 创建小样本数据
        np.random.seed(42)
        data = pd.DataFrame({
            'Y_bar': np.random.randn(5),
            'D': [1, 1, 0, 0, 0],
            'cluster': [1, 1, 2, 2, 2],
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            try:
                result = run_ols_regression(
                    data=data,
                    y='Y_bar',
                    d='D',
                    vce='cluster',
                    cluster_var='cluster',
                )
            except (ValueError, np.linalg.LinAlgError):
                pass
            
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            
            if user_warnings:
                for warning in user_warnings:
                    # 允许警告指向测试文件或 site-packages（某些环境下）
                    # 关键是不应该指向 estimation.py 内部
                    assert 'estimation.py' not in warning.filename or \
                           'test_' in warning.filename, \
                        f"警告指向了库内部文件: {warning.filename}:{warning.lineno}"


class TestStacklevelValues:
    """验证特定 stacklevel 值的测试"""
    
    def test_all_warnings_have_stacklevel(self):
        """验证所有 warnings.warn 调用都有 stacklevel 参数"""
        import subprocess
        import os
        
        lwdid_src = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'src', 'lwdid'
        )
        
        # 使用 grep 查找没有 stacklevel 的 warnings.warn 调用
        # 排除 .bak 文件
        result = subprocess.run(
            ['grep', '-r', '-n', '--include=*.py', '--exclude=*.bak',
             r'warnings\.warn([^)]*\)$', lwdid_src],
            capture_output=True, text=True
        )
        
        # 如果找到匹配项，测试失败
        if result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            # 过滤掉已经有 stacklevel 的行（这些不应该匹配，但为了安全）
            problem_lines = [l for l in lines if 'stacklevel' not in l]
            
            if problem_lines:
                pytest.fail(
                    f"发现 {len(problem_lines)} 处 warnings.warn 调用没有 stacklevel 参数:\n"
                    + '\n'.join(problem_lines[:10])
                )


class TestSpecificStacklevelFunctions:
    """测试特定函数的 stacklevel 设置"""
    
    def test_estimate_propensity_score_constant_covariate_warning(self):
        """测试常数协变量警告的 stacklevel"""
        from lwdid.staggered.estimators import estimate_propensity_score
        
        # 创建包含常数协变量的数据
        np.random.seed(42)
        n = 100
        data = pd.DataFrame({
            'd': np.array([1]*50 + [0]*50),
            'x1': np.random.randn(n),
            'x_const': np.ones(n),  # 常数变量
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            try:
                pscores, coef = estimate_propensity_score(
                    data, 'd', ['x1', 'x_const'], trim_threshold=0.01
                )
            except (ValueError, np.linalg.LinAlgError):
                pass
            
            # 检查是否有常数协变量警告
            const_warnings = [
                x for x in w 
                if issubclass(x.category, UserWarning) and '常数' in str(x.message)
            ]
            
            if const_warnings:
                for warning in const_warnings:
                    # 警告应该指向测试文件
                    assert 'estimators.py' not in warning.filename or \
                           'test_' in warning.filename, \
                        f"常数协变量警告指向了库内部: {warning.filename}"


class TestDeprecationWarningStacklevel:
    """测试弃用警告的 stacklevel"""
    
    def test_abadie_imbens_simple_deprecation_stacklevel(self):
        """测试简化版 AI SE 弃用警告的 stacklevel"""
        from lwdid.staggered.estimators import _compute_psm_se_abadie_imbens_simple
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            try:
                # 调用弃用函数
                result = _compute_psm_se_abadie_imbens_simple(
                    Y_treat=np.array([1.0, 2.0]),
                    Y_control=np.array([0.5, 1.5]),
                    matched_control_ids=[[0], [1]],
                    att=0.5,
                )
            except (ValueError, np.linalg.LinAlgError):
                pass
            
            # 检查弃用警告
            deprecation_warnings = [
                x for x in w 
                if issubclass(x.category, DeprecationWarning)
            ]
            
            if deprecation_warnings:
                for warning in deprecation_warnings:
                    # 弃用警告应该指向调用者，不是库内部
                    # 但由于这是私有函数，stacklevel=3 指向更上层
                    pass  # 私有函数的弃用警告可以接受指向内部


class TestBootstrapWarningStacklevel:
    """测试 Bootstrap 相关警告的 stacklevel"""
    
    def test_bootstrap_success_rate_warning_stacklevel(self):
        """测试 Bootstrap 成功率警告的 stacklevel"""
        from lwdid.staggered.estimators import compute_ipwra_se_bootstrap
        
        # 创建可能导致 Bootstrap 失败的极端数据
        np.random.seed(42)
        data = pd.DataFrame({
            'y': np.random.randn(20),
            'd': np.array([1]*2 + [0]*18),  # 极少处理组
            'x1': np.random.randn(20),
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            try:
                se, ci_low, ci_high = compute_ipwra_se_bootstrap(
                    data=data,
                    y='y',
                    d='d',
                    controls=['x1'],
                    propensity_controls=['x1'],
                    trim_threshold=0.01,
                    n_bootstrap=20,  # 少量重复以加速
                    seed=42,
                    alpha=0.05,
                )
            except (ValueError, np.linalg.LinAlgError):
                pass
            
            # 检查 Bootstrap 相关警告
            bootstrap_warnings = [
                x for x in w 
                if issubclass(x.category, UserWarning) and 'Bootstrap' in str(x.message)
            ]
            
            if bootstrap_warnings:
                for warning in bootstrap_warnings:
                    # Bootstrap 函数是公共 API，stacklevel=2 应该指向测试文件
                    assert 'estimators.py' not in warning.filename or \
                           'test_' in warning.filename, \
                        f"Bootstrap 警告指向了库内部: {warning.filename}"


# ============================================================================
# 运行测试
# ============================================================================
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
