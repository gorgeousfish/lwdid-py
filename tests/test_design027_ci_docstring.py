"""
DESIGN-027: 验证 staggered 子模块中 CI docstring 不再硬编码 95%

测试目标：
1. 验证所有 CI 相关 docstring 使用通用描述
2. 验证没有硬编码 "95% CI" 的 docstring
3. 验证数据类和函数的 docstring 一致性
"""

import ast
import inspect
import re
import pytest
from pathlib import Path


# ============================================================================
# Section 1: Docstring 硬编码检查测试
# ============================================================================

class TestDocstringNoHardcoded95:
    """验证 staggered 子模块中没有硬编码 95% CI 的 docstring"""
    
    def test_estimation_no_95_hardcoded(self):
        """estimation.py 中不应有 '95%' 硬编码在 CI 相关 docstring 中"""
        from lwdid.staggered import estimation
        source = inspect.getsource(estimation)
        
        # 查找所有 "95%" 相关的模式（但排除合理用法如分位数描述）
        pattern = r'95%\s*(?:CI|confidence|置信)'
        matches = re.findall(pattern, source, re.IGNORECASE)
        
        assert len(matches) == 0, (
            f"Found hardcoded '95% CI' in estimation.py: {matches}"
        )
    
    def test_aggregation_no_95_hardcoded(self):
        """aggregation.py 中不应有 '95%' 硬编码在 CI 相关 docstring 中"""
        from lwdid.staggered import aggregation
        source = inspect.getsource(aggregation)
        
        pattern = r'95%\s*(?:CI|confidence|置信)'
        matches = re.findall(pattern, source, re.IGNORECASE)
        
        assert len(matches) == 0, (
            f"Found hardcoded '95% CI' in aggregation.py: {matches}"
        )
    
    def test_estimators_no_95_hardcoded(self):
        """estimators.py 中不应有 '95%' 硬编码在 CI 相关 docstring 中"""
        from lwdid.staggered import estimators
        source = inspect.getsource(estimators)
        
        pattern = r'95%\s*(?:CI|confidence|置信)'
        matches = re.findall(pattern, source, re.IGNORECASE)
        
        assert len(matches) == 0, (
            f"Found hardcoded '95% CI' in estimators.py: {matches}"
        )
    
    def test_no_chinese_95_ci(self):
        """staggered 子模块中不应有 '95% CI下界/上界' 硬编码"""
        from lwdid.staggered import estimation, aggregation, estimators
        
        for module in [estimation, aggregation, estimators]:
            source = inspect.getsource(module)
            pattern = r'95%\s*CI[下上]界'
            matches = re.findall(pattern, source)
            
            assert len(matches) == 0, (
                f"Found hardcoded '95% CI下界/上界' in {module.__name__}: {matches}"
            )


# ============================================================================
# Section 2: Docstring 正确格式验证
# ============================================================================

class TestDocstringCorrectFormat:
    """验证 CI 相关 docstring 使用正确的通用格式"""
    
    def test_cohort_time_effect_ci_docstring(self):
        """CohortTimeEffect 数据类的 CI 属性应使用通用描述"""
        from lwdid.staggered.estimation import CohortTimeEffect
        
        # 检查类的 docstring
        docstring = CohortTimeEffect.__doc__
        assert docstring is not None
        
        # 应该包含 "Confidence interval" 而不是 "95% confidence interval"
        assert 'ci_lower' in docstring.lower() or 'confidence' in docstring.lower()
        assert '95% confidence interval' not in docstring
    
    def test_ipw_result_ci_docstring(self):
        """IPWResult 数据类的 CI 属性应使用通用描述"""
        from lwdid.staggered.estimators import IPWResult
        
        docstring = IPWResult.__doc__
        assert docstring is not None
        assert '95% CI下界' not in docstring
        assert '95% CI上界' not in docstring
    
    def test_ipwra_result_ci_docstring(self):
        """IPWRAResult 数据类的 CI 属性应使用通用描述"""
        from lwdid.staggered.estimators import IPWRAResult
        
        docstring = IPWRAResult.__doc__
        assert docstring is not None
        assert '95% CI下界' not in docstring
        assert '95% CI上界' not in docstring
    
    def test_ra_result_ci_docstring(self):
        """RAResult 数据类的 CI 属性应使用通用描述"""
        from lwdid.staggered.estimators import RAResult
        
        docstring = RAResult.__doc__
        assert docstring is not None
        assert '95% CI下界' not in docstring
        assert '95% CI上界' not in docstring
    
    def test_psm_result_ci_docstring(self):
        """PSMResult 数据类的 CI 属性应使用通用描述"""
        from lwdid.staggered.estimators import PSMResult
        
        docstring = PSMResult.__doc__
        assert docstring is not None
        assert '95% CI下界' not in docstring
        assert '95% CI上界' not in docstring


# ============================================================================
# Section 3: Summary 方法输出格式验证
# ============================================================================

class TestSummaryMethodFormat:
    """验证 summary() 方法输出不硬编码 95%"""
    
    def test_ipw_result_summary_no_95(self):
        """IPWResult.summary() 输出应使用 'CI:' 而不是 '95% CI:'"""
        import numpy as np
        from lwdid.staggered.estimators import IPWResult
        
        # 创建一个虚拟结果对象
        result = IPWResult(
            att=0.1,
            se=0.05,
            ci_lower=0.0,
            ci_upper=0.2,
            t_stat=2.0,
            pvalue=0.05,
            propensity_scores=np.array([0.5, 0.5]),
            weights=np.array([1.0, 1.0]),
            propensity_model_coef={'const': 0.0},
            n_treated=100,
            n_control=100,
            weights_cv=0.1
        )
        
        summary = result.summary()
        
        # 应该包含 "CI:" 而不是 "95% CI:"
        assert 'CI:' in summary
        assert '95% CI:' not in summary


# ============================================================================
# Section 4: Alpha 参数说明验证
# ============================================================================

class TestAlphaParameterDocstring:
    """验证 alpha 参数的 docstring 使用通用描述"""
    
    def test_aggregate_to_cohort_alpha_docstring(self):
        """aggregate_to_cohort 的 alpha 参数应使用通用描述"""
        from lwdid.staggered.aggregation import aggregate_to_cohort
        
        docstring = aggregate_to_cohort.__doc__
        assert docstring is not None
        
        # 不应包含 "default 0.05 for 95% CI" 这样的硬编码
        assert '0.05 for 95% CI' not in docstring
        assert 'default 0.05 for 95%' not in docstring
    
    def test_aggregate_to_overall_alpha_docstring(self):
        """aggregate_to_overall 的 alpha 参数应使用通用描述"""
        from lwdid.staggered.aggregation import aggregate_to_overall
        
        docstring = aggregate_to_overall.__doc__
        assert docstring is not None
        
        assert '0.05 for 95% CI' not in docstring
        assert 'default 0.05 for 95%' not in docstring


# ============================================================================
# Section 5: 与 results.py 一致性验证
# ============================================================================

class TestConsistencyWithResults:
    """验证与 results.py 中修复后的格式一致"""
    
    def test_ci_docstring_style_consistency(self):
        """所有 CI 相关 docstring 应使用一致的风格"""
        from lwdid.results import LWDIDResults
        from lwdid.staggered.estimators import IPWResult, IPWRAResult, RAResult, PSMResult
        
        # 获取 results.py 中的 ci_lower/ci_upper 属性 docstring
        results_ci_lower_doc = LWDIDResults.ci_lower.fget.__doc__
        results_ci_upper_doc = LWDIDResults.ci_upper.fget.__doc__
        
        # 验证 results.py 使用的格式
        assert 'Confidence interval' in results_ci_lower_doc
        assert '95%' not in results_ci_lower_doc
        
        # 验证 staggered 数据类使用类似格式（不包含 95%）
        for cls in [IPWResult, IPWRAResult, RAResult, PSMResult]:
            docstring = cls.__doc__
            if docstring:
                assert '95% CI下界' not in docstring, f"{cls.__name__} still has hardcoded 95%"
                assert '95% CI上界' not in docstring, f"{cls.__name__} still has hardcoded 95%"


# ============================================================================
# Section 6: 源文件扫描测试
# ============================================================================

class TestSourceFileScan:
    """直接扫描源文件验证"""
    
    def test_scan_staggered_directory(self):
        """扫描整个 staggered 目录，确保没有遗漏的 95% CI 硬编码"""
        import lwdid.staggered as staggered_pkg
        staggered_dir = Path(staggered_pkg.__file__).parent
        
        problematic_files = []
        pattern = re.compile(r'95%\s*(?:CI|confidence|置信|CI下界|CI上界)', re.IGNORECASE)
        
        for py_file in staggered_dir.glob('*.py'):
            if py_file.name.endswith('.bak'):
                continue
            
            content = py_file.read_text(encoding='utf-8')
            matches = pattern.findall(content)
            
            if matches:
                problematic_files.append((py_file.name, matches))
        
        assert len(problematic_files) == 0, (
            f"Found hardcoded 95% CI patterns in staggered files: {problematic_files}"
        )
    
    def test_no_backup_files_exist(self):
        """确保没有 .bak 备份文件存在"""
        import lwdid.staggered as staggered_pkg
        staggered_dir = Path(staggered_pkg.__file__).parent
        
        bak_files = list(staggered_dir.glob('*.bak'))
        
        assert len(bak_files) == 0, (
            f"Found backup files in staggered directory: {[f.name for f in bak_files]}"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
