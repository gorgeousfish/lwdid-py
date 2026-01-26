# -*- coding: utf-8 -*-
"""
Common Timing测试共享fixtures

提供:
- common_timing_data: 从Stata .dta文件加载的面板数据
- stata_results: Stata验证基准结果
- transformed_data: 预计算变换后的数据
- 参数化fixtures用于period和estimator
"""

import json
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# 路径配置
# =============================================================================

# 测试目录路径
TEST_DIR = Path(__file__).parent
FIXTURES_DIR = TEST_DIR / "fixtures"

# 数据文件路径 (相对于项目根目录)
PROJECT_ROOT = TEST_DIR.parent.parent.parent  # lwdid-py_v0.1.0/../..
DATA_DIR = PROJECT_ROOT / "Lee_Wooldridge_2023-main 3"
COMMON_DATA_FILE = DATA_DIR / "1.lee_wooldridge_common_data.dta"

# Stata验证结果JSON
STATA_RESULTS_FILE = FIXTURES_DIR / "stata_common_timing_results.json"


# =============================================================================
# 数据加载Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def common_timing_data() -> pd.DataFrame:
    """
    加载Common Timing测试数据 (Lee & Wooldridge 2023示例数据)
    
    数据结构:
    - id: 单位标识符 (1-1000)
    - year: 时期 (2001-2006)
    - period: 时期索引 (1-6，通过year-2000计算)
    - y: 结果变量
    - d: 处理指示符 (unit-level, time-invariant)
    - f04, f05, f06: period dummy indicators
    - x1, x2: 协变量
    
    注意: 原始数据年份是2001-2006，映射为period 1-6
    首次处理期S对应2004年(period=4)
    
    Returns
    -------
    pd.DataFrame
        面板数据 (6000 rows = 1000 units × 6 periods)
    """
    if not COMMON_DATA_FILE.exists():
        pytest.skip(
            f"Common timing data file not found: {COMMON_DATA_FILE}\n"
            f"Please ensure the Lee_Wooldridge_2023-main 3 folder is in the project root."
        )
    
    data = pd.read_stata(str(COMMON_DATA_FILE))
    
    # 验证数据结构
    required_cols = ['id', 'year', 'y', 'd', 'x1', 'x2', 'f04', 'f05', 'f06']
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        pytest.fail(f"Missing required columns in data: {missing}")
    
    # 确保数据类型正确
    data['id'] = data['id'].astype(int)
    data['year'] = data['year'].astype(int)
    data['d'] = data['d'].astype(int)
    
    # 添加period列 (year - 2000，使2001=1, ..., 2006=6)
    data['period'] = data['year'] - 2000
    
    return data


@pytest.fixture(scope="module")
def stata_results() -> Dict[str, Any]:
    """
    加载Stata验证结果JSON
    
    包含:
    - metadata: 数据来源信息
    - data_info: T, S, first_treat_period等
    - sample_info: n_obs, n_treated, n_control
    - estimators: ra, ipwra, psm各period的ATT和SE
    - tolerances: 验证误差阈值
    
    Returns
    -------
    dict
        Stata验证基准数据
    """
    if not STATA_RESULTS_FILE.exists():
        pytest.fail(f"Stata results file not found: {STATA_RESULTS_FILE}")
    
    with open(STATA_RESULTS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture(scope="module")
def sample_info(stata_results: Dict[str, Any]) -> Dict[str, int]:
    """提取样本信息"""
    return stata_results['sample_info']


@pytest.fixture(scope="module")
def data_info(stata_results: Dict[str, Any]) -> Dict[str, Any]:
    """提取数据配置信息"""
    return stata_results['data_info']


@pytest.fixture(scope="module")
def tolerances(stata_results: Dict[str, Any]) -> Dict[str, float]:
    """提取误差阈值"""
    return stata_results['tolerances']


# =============================================================================
# 变换数据Fixtures
# =============================================================================

def compute_transformed_outcome_common_timing(
    data: pd.DataFrame,
    y_col: str,
    id_col: str,
    period_col: str,
    first_treat_period: int,
    target_period: int,
) -> pd.Series:
    """
    计算Common Timing场景的变换后结果变量。
    
    公式 (论文3.2): 
        ŷ_{ir} = Y_{ir} - (1/(S-1)) × Σ_{q=1}^{S-1} Y_{iq}
               = Y_{ir} - mean(Y_{i,pre})
    
    关键点:
    - pre-mean对所有r相同，仅依赖S
    - 分母是 (S-1)，不是 (r-1)
    
    Parameters
    ----------
    data : pd.DataFrame
        面板数据
    y_col : str
        结果变量列名
    id_col : str
        单位标识符列名
    period_col : str
        时期列名 (应该是period或year列)
    first_treat_period : int
        首次处理期 S (本数据中S=4 for period或2004 for year)
    target_period : int
        目标评估期 r (4, 5, 或 6 for period; 2004,2005,2006 for year)
    
    Returns
    -------
    pd.Series
        变换后的结果变量，index为id
    """
    # 处理年份列 (如果使用year列，值是2001-2006)
    # 转换为period索引 1-6 进行计算
    period_data = data.copy()
    
    # 检测是否使用年份格式
    period_values = period_data[period_col].unique()
    if period_values.min() > 100:  # 年份格式 (2001-2006)
        # 转换为period索引
        year_offset = 2000
        period_data['_period_idx'] = period_data[period_col] - year_offset
        period_col_use = '_period_idx'
        first_treat_idx = first_treat_period - year_offset if first_treat_period > 100 else first_treat_period
        target_idx = target_period - year_offset if target_period > 100 else target_period
    else:
        period_col_use = period_col
        first_treat_idx = first_treat_period
        target_idx = target_period
    
    # 获取target period r的结果
    period_mask = period_data[period_col_use] == target_idx
    Y_r = period_data[period_mask].set_index(id_col)[y_col]
    
    # pre-treatment periods: 1, 2, ..., S-1
    pre_periods = list(range(1, first_treat_idx))  # [1, 2, 3] for S=4
    
    # 计算pre-treatment均值
    pre_data = period_data[period_data[period_col_use].isin(pre_periods)]
    pre_mean = pre_data.groupby(id_col)[y_col].mean()
    
    # 变换: Y_r - pre_mean
    y_transformed = Y_r - pre_mean
    
    return y_transformed


@pytest.fixture(scope="module")
def transformed_data(common_timing_data: pd.DataFrame) -> pd.DataFrame:
    """
    预计算所有post-treatment periods的变换后数据
    
    为每个period (4, 5, 6 对应年份2004,2005,2006)计算y_dot并添加到数据中。
    
    Returns
    -------
    pd.DataFrame
        包含y_dot列的数据（筛选到post-treatment periods）
    """
    data = common_timing_data.copy()
    S = 4  # first treatment period (period索引，对应2004年)
    
    # 年份与period索引的映射
    year_to_period = {2001: 1, 2002: 2, 2003: 3, 2004: 4, 2005: 5, 2006: 6}
    period_to_year = {v: k for k, v in year_to_period.items()}
    
    # 为每个period计算变换变量
    transformed_list = []
    
    for r in [4, 5, 6]:  # period索引
        target_year = period_to_year[r]  # 对应年份
        
        # 计算变换 (使用period列)
        y_dot = compute_transformed_outcome_common_timing(
            data, 'y', 'id', 'period',
            first_treat_period=S,
            target_period=r
        )
        
        # 获取该年份的数据
        period_data = data[data['year'] == target_year].copy()
        period_data['y_dot'] = period_data['id'].map(y_dot)
        # period列已经存在，不需要重新添加
        
        transformed_list.append(period_data)
    
    # 合并所有periods
    result = pd.concat(transformed_list, ignore_index=True)
    
    return result


@pytest.fixture
def period_4_data(transformed_data: pd.DataFrame) -> pd.DataFrame:
    """筛选到period 4 (year 2004)的横截面数据"""
    return transformed_data[transformed_data['period'] == 4].copy()


@pytest.fixture
def period_5_data(transformed_data: pd.DataFrame) -> pd.DataFrame:
    """筛选到period 5 (year 2005)的横截面数据"""
    return transformed_data[transformed_data['period'] == 5].copy()


@pytest.fixture
def period_6_data(transformed_data: pd.DataFrame) -> pd.DataFrame:
    """筛选到period 6 (year 2006)的横截面数据"""
    return transformed_data[transformed_data['period'] == 6].copy()


# =============================================================================
# 参数化Fixtures
# =============================================================================

@pytest.fixture(params=[4, 5, 6], ids=['f04', 'f05', 'f06'])
def period(request) -> int:
    """参数化period fixture"""
    return request.param


@pytest.fixture(params=['ra', 'ipwra', 'psm'], ids=['RA', 'IPWRA', 'PSM'])
def estimator(request) -> str:
    """参数化estimator fixture"""
    return request.param


# =============================================================================
# 辅助Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def controls() -> list:
    """控制变量列表"""
    return ['x1', 'x2']


@pytest.fixture(scope="module")
def ps_controls() -> list:
    """倾向得分模型控制变量"""
    return ['x1', 'x2']


@pytest.fixture
def get_stata_result(stata_results: Dict[str, Any]):
    """
    工厂fixture: 获取特定estimator和period的Stata结果
    
    Usage:
        def test_ra(get_stata_result):
            stata = get_stata_result('ra', 4)
            assert abs(result.att - stata['att']) < 1e-6
    """
    def _get_result(estimator: str, period: int) -> Dict[str, float]:
        period_key = f'f{period:02d}'
        return stata_results['estimators'][estimator][period_key]
    
    return _get_result


# =============================================================================
# 验证辅助函数
# =============================================================================

def assert_att_close_to_stata(
    computed_att: float,
    stata_att: float,
    tolerance: float = 1e-6,
    description: str = ""
) -> None:
    """
    断言ATT与Stata结果一致
    
    Parameters
    ----------
    computed_att : float
        计算得到的ATT
    stata_att : float
        Stata基准ATT
    tolerance : float
        相对误差阈值
    description : str
        错误信息描述
    """
    if stata_att == 0:
        abs_error = abs(computed_att - stata_att)
        assert abs_error < tolerance, \
            f"{description}: 绝对误差 {abs_error:.2e} > {tolerance}"
    else:
        rel_error = abs(computed_att - stata_att) / abs(stata_att)
        assert rel_error < tolerance, \
            f"{description}: 相对误差 {rel_error:.2e} > {tolerance}\n" \
            f"  Computed: {computed_att}\n  Stata: {stata_att}"


def assert_se_close_to_stata(
    computed_se: float,
    stata_se: float,
    tolerance: float = 0.05,
    description: str = ""
) -> None:
    """
    断言SE与Stata结果一致
    
    Parameters
    ----------
    computed_se : float
        计算得到的SE
    stata_se : float
        Stata基准SE
    tolerance : float
        相对误差阈值 (default: 5%)
    description : str
        错误信息描述
    """
    if stata_se == 0:
        abs_error = abs(computed_se - stata_se)
        assert abs_error < tolerance, \
            f"{description}: 绝对误差 {abs_error:.2e} > {tolerance}"
    else:
        rel_error = abs(computed_se - stata_se) / stata_se
        assert rel_error < tolerance, \
            f"{description}: 相对误差 {rel_error:.1%} > {tolerance:.1%}\n" \
            f"  Computed: {computed_se}\n  Stata: {stata_se}"


# 导出辅助函数供测试模块使用
__all__ = [
    'common_timing_data',
    'stata_results',
    'transformed_data',
    'period_4_data',
    'period_5_data',
    'period_6_data',
    'period',
    'estimator',
    'controls',
    'ps_controls',
    'get_stata_result',
    'compute_transformed_outcome_common_timing',
    'assert_att_close_to_stata',
    'assert_se_close_to_stata',
]
