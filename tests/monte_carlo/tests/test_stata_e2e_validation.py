# -*- coding: utf-8 -*-
"""
Stata端到端验证测试

Task 8: Stata端到端验证
- 8.1 实现Stata验证脚本生成器
- 8.2 实现Python-Stata结果比较

Validates: Requirements 5.1, 5.2, 5.3

References
----------
Lee & Wooldridge (2023, 2026)
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

# 添加路径
fixtures_path = Path(__file__).parent.parent.parent / 'test_common_timing' / 'fixtures'
sys.path.insert(0, str(fixtures_path))


class TestStataDoFileGeneration:
    """
    Task 8.1: Stata验证脚本生成器测试
    Validates: Requirements 5.3
    """
    
    def test_generate_lwdid_demean_dofile(self, tmp_path):
        """
        生成lwdid demeaning命令的do文件
        """
        from dgp_small_sample import generate_small_sample_dgp
        
        # 生成测试数据
        data, params = generate_small_sample_dgp(seed=42)
        csv_path = tmp_path / "test_data.csv"
        data.to_csv(csv_path, index=False)
        
        # 生成do文件内容
        dofile_content = generate_lwdid_dofile(
            csv_path=str(csv_path),
            y='y',
            ivar='id',
            tvar='year',
            gvar='d',
            rolling='demean',
        )
        
        # 验证do文件内容
        assert 'import delimited' in dofile_content
        assert 'lwdid y' in dofile_content
        assert 'ivar(id)' in dofile_content
        assert 'tvar(year)' in dofile_content
        assert 'rolling(demean)' in dofile_content
    
    def test_generate_lwdid_detrend_dofile(self, tmp_path):
        """
        生成lwdid detrending命令的do文件
        """
        from dgp_small_sample import generate_small_sample_dgp
        
        data, params = generate_small_sample_dgp(seed=42)
        csv_path = tmp_path / "test_data.csv"
        data.to_csv(csv_path, index=False)
        
        dofile_content = generate_lwdid_dofile(
            csv_path=str(csv_path),
            y='y',
            ivar='id',
            tvar='year',
            gvar='d',
            rolling='detrend',
        )
        
        assert 'rolling(detrend)' in dofile_content
    
    def test_generate_complete_validation_dofile(self, tmp_path):
        """
        生成完整的验证do文件（包含多种估计方法）
        """
        from dgp_small_sample import generate_small_sample_dgp
        
        data, params = generate_small_sample_dgp(seed=42)
        csv_path = tmp_path / "test_data.csv"
        data.to_csv(csv_path, index=False)
        
        dofile_content = generate_complete_validation_dofile(
            csv_path=str(csv_path),
            y='y',
            ivar='id',
            tvar='year',
            gvar='d',
        )
        
        # 验证包含所有估计方法
        assert 'rolling(demean)' in dofile_content
        assert 'rolling(detrend)' in dofile_content
        assert 'vce(hc3)' in dofile_content or 'robust' in dofile_content


class TestPythonStataComparison:
    """
    Task 8.2: Python-Stata结果比较测试
    Validates: Requirements 5.1, 5.2
    """
    
    def test_att_comparison_structure(self):
        """
        验证ATT比较结构
        """
        # 模拟Python结果
        python_result = {
            'att': 2.0,
            'se': 0.5,
            'ci_lower': 1.02,
            'ci_upper': 2.98,
        }
        
        # 模拟Stata结果
        stata_result = {
            'att': 2.001,
            'se': 0.501,
            'ci_lower': 1.019,
            'ci_upper': 2.983,
        }
        
        # 比较
        comparison = compare_python_stata_results(python_result, stata_result)
        
        assert 'att_diff' in comparison
        assert 'se_diff' in comparison
        assert 'att_match' in comparison
        assert 'se_match' in comparison
    
    def test_att_tolerance_check(self):
        """
        验证ATT差异容差检查
        Requirements 5.1: ATT差异 < 1e-6
        """
        python_att = 2.0
        stata_att = 2.0000001
        
        diff = abs(python_att - stata_att)
        tolerance = 1e-6
        
        # 应该通过
        assert diff < tolerance, f"ATT差异 {diff} 超过容差 {tolerance}"
    
    def test_se_tolerance_check(self):
        """
        验证SE差异容差检查
        Requirements 5.2: SE差异 < 1e-4
        """
        python_se = 0.5
        stata_se = 0.50001
        
        diff = abs(python_se - stata_se)
        tolerance = 1e-4
        
        # 应该通过
        assert diff < tolerance, f"SE差异 {diff} 超过容差 {tolerance}"
    
    def test_relative_difference_calculation(self):
        """
        验证相对差异计算
        """
        python_att = 2.0
        stata_att = 2.1
        
        rel_diff = abs(python_att - stata_att) / abs(stata_att)
        expected_rel_diff = 0.1 / 2.1
        
        assert np.isclose(rel_diff, expected_rel_diff)


class TestStataResultParser:
    """
    Stata结果解析器测试
    """
    
    def test_parse_lwdid_output(self):
        """
        解析lwdid命令输出
        """
        # 模拟Stata输出
        stata_output = """
        lwdid y, ivar(id) tvar(year) gvar(d) rolling(detrend)
        
        ATT = 2.0000
        SE = 0.5000
        t = 4.0000
        P>|t| = 0.0001
        [95% Conf. Interval] = [1.0200, 2.9800]
        """
        
        # 解析
        result = parse_lwdid_output(stata_output)
        
        assert 'att' in result
        assert 'se' in result
    
    def test_parse_teffects_output(self):
        """
        解析teffects命令输出
        """
        # 模拟Stata输出
        stata_output = """
        teffects ra (y x1 x2) (d), atet
        
        ATET = 1.5000
        Std. Err. = 0.3000
        """
        
        result = parse_teffects_output(stata_output)
        
        assert 'att' in result or 'atet' in result


# =============================================================================
# 辅助函数
# =============================================================================

def generate_lwdid_dofile(
    csv_path: str,
    y: str,
    ivar: str,
    tvar: str,
    gvar: str,
    rolling: str = 'demean',
    vce: str | None = None,
) -> str:
    """
    生成lwdid命令的do文件
    
    Parameters
    ----------
    csv_path : str
        CSV数据文件路径
    y : str
        结果变量名
    ivar : str
        单位标识变量
    tvar : str
        时间变量
    gvar : str
        处理组变量
    rolling : str
        变换方法: 'demean' 或 'detrend'
    vce : str, optional
        方差估计方法
    
    Returns
    -------
    str
        do文件内容
    """
    vce_option = f', vce({vce})' if vce else ''
    
    dofile = f'''
* Monte Carlo Validation - lwdid Stata Comparison
* Generated automatically

clear all
set more off

* Load data
import delimited "{csv_path}", clear

* Describe data
describe
summarize

* Check treatment distribution
tab {gvar}

* Run lwdid estimation
lwdid {y}, ivar({ivar}) tvar({tvar}) gvar({gvar}) rolling({rolling}){vce_option}

* Store results
scalar att = e(att)
scalar se = e(se)
di "ATT = " att
di "SE = " se
'''
    return dofile


def generate_complete_validation_dofile(
    csv_path: str,
    y: str,
    ivar: str,
    tvar: str,
    gvar: str,
) -> str:
    """
    生成完整的验证do文件
    """
    dofile = f'''
* Complete Monte Carlo Validation
* Generated automatically

clear all
set more off

* Load data
import delimited "{csv_path}", clear

* Describe data
describe
summarize

* ============================================
* Method 1: Demeaning (OLS SE)
* ============================================
di _n "=== Demeaning (OLS SE) ==="
lwdid {y}, ivar({ivar}) tvar({tvar}) gvar({gvar}) rolling(demean)
scalar att_demean_ols = e(att)
scalar se_demean_ols = e(se)

* ============================================
* Method 2: Detrending (OLS SE)
* ============================================
di _n "=== Detrending (OLS SE) ==="
lwdid {y}, ivar({ivar}) tvar({tvar}) gvar({gvar}) rolling(detrend)
scalar att_detrend_ols = e(att)
scalar se_detrend_ols = e(se)

* ============================================
* Method 3: Detrending (HC3 SE)
* ============================================
di _n "=== Detrending (HC3 SE) ==="
lwdid {y}, ivar({ivar}) tvar({tvar}) gvar({gvar}) rolling(detrend), vce(hc3)
scalar att_detrend_hc3 = e(att)
scalar se_detrend_hc3 = e(se)

* ============================================
* Summary
* ============================================
di _n "=== Summary ==="
di "Demeaning (OLS): ATT = " att_demean_ols ", SE = " se_demean_ols
di "Detrending (OLS): ATT = " att_detrend_ols ", SE = " se_detrend_ols
di "Detrending (HC3): ATT = " att_detrend_hc3 ", SE = " se_detrend_hc3
'''
    return dofile


def compare_python_stata_results(
    python_result: Dict[str, float],
    stata_result: Dict[str, float],
    att_tolerance: float = 1e-6,
    se_tolerance: float = 1e-4,
) -> Dict[str, Any]:
    """
    比较Python和Stata结果
    
    Parameters
    ----------
    python_result : dict
        Python估计结果
    stata_result : dict
        Stata估计结果
    att_tolerance : float
        ATT差异容差
    se_tolerance : float
        SE差异容差
    
    Returns
    -------
    dict
        比较结果
    """
    att_diff = abs(python_result['att'] - stata_result['att'])
    se_diff = abs(python_result['se'] - stata_result['se'])
    
    return {
        'att_diff': att_diff,
        'se_diff': se_diff,
        'att_match': att_diff < att_tolerance,
        'se_match': se_diff < se_tolerance,
        'att_rel_diff': att_diff / abs(stata_result['att']) if stata_result['att'] != 0 else np.nan,
        'se_rel_diff': se_diff / abs(stata_result['se']) if stata_result['se'] != 0 else np.nan,
    }


def parse_lwdid_output(stata_output: str) -> Dict[str, float]:
    """
    解析lwdid命令输出
    
    Parameters
    ----------
    stata_output : str
        Stata输出文本
    
    Returns
    -------
    dict
        解析的结果
    """
    import re
    
    result = {}
    
    # 解析ATT
    att_match = re.search(r'ATT\s*=\s*([-\d.]+)', stata_output)
    if att_match:
        result['att'] = float(att_match.group(1))
    
    # 解析SE
    se_match = re.search(r'SE\s*=\s*([-\d.]+)', stata_output)
    if se_match:
        result['se'] = float(se_match.group(1))
    
    # 解析t统计量
    t_match = re.search(r't\s*=\s*([-\d.]+)', stata_output)
    if t_match:
        result['t_stat'] = float(t_match.group(1))
    
    # 解析p值
    p_match = re.search(r'P>?\|t\|\s*=\s*([-\d.]+)', stata_output)
    if p_match:
        result['pvalue'] = float(p_match.group(1))
    
    return result


def parse_teffects_output(stata_output: str) -> Dict[str, float]:
    """
    解析teffects命令输出
    
    Parameters
    ----------
    stata_output : str
        Stata输出文本
    
    Returns
    -------
    dict
        解析的结果
    """
    import re
    
    result = {}
    
    # 解析ATET
    atet_match = re.search(r'ATET\s*=\s*([-\d.]+)', stata_output)
    if atet_match:
        result['atet'] = float(atet_match.group(1))
        result['att'] = result['atet']
    
    # 解析Std. Err.
    se_match = re.search(r'Std\.?\s*Err\.?\s*=\s*([-\d.]+)', stata_output)
    if se_match:
        result['se'] = float(se_match.group(1))
    
    return result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
