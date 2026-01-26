# -*- coding: utf-8 -*-
"""
Staggered端到端测试共享fixtures

提供:
- staggered_data: 从Stata .dta文件加载的面板数据
- stata_results: Stata验证基准结果
- transformed_data: 预计算变换后的数据
- 参数化fixtures用于(g,r)组合和estimator
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
STAGGERED_DATA_FILE = DATA_DIR / "2.lee_wooldridge_staggered_data.dta"

# Stata验证结果JSON
STATA_RESULTS_FILE = FIXTURES_DIR / "stata_staggered_results.json"


# =============================================================================
# 常量定义
# =============================================================================

# 所有(g,r)组合
GR_COMBINATIONS = [(4, 4), (4, 5), (4, 6), (5, 5), (5, 6), (6, 6)]

# Stata已验证IPWRA结果 (从2.lee_wooldridge_rolling_staggered.do输出)
STATA_IPWRA_RESULTS = {
    (4, 4): {'att': 4.3029238, 'se': 0.42367713, 'n_obs': 1000, 'n_treated': 129, 'n_control': 871},
    (4, 5): {'att': 6.6112909, 'se': 0.43215951, 'n_obs': 891, 'n_treated': 129, 'n_control': 762},
    (4, 6): {'att': 8.3343553, 'se': 0.44138304, 'n_obs': 781, 'n_treated': 129, 'n_control': 652},
    (5, 5): {'att': 3.0283627, 'se': 0.42077459, 'n_obs': 871, 'n_treated': 109, 'n_control': 762},
    (5, 6): {'att': 4.9326076, 'se': 0.44026846, 'n_obs': 761, 'n_treated': 109, 'n_control': 652},
    (6, 6): {'att': 2.4200472, 'se': 0.48314986, 'n_obs': 762, 'n_treated': 110, 'n_control': 652},
}

# Stata已验证RA结果 (从Stata MCP获取)
STATA_RA_RESULTS = {
    (4, 4): {'att': 4.24825, 'se': 0.4246985, 'n_obs': 1000, 'n_treated': 129, 'n_control': 871},
    (4, 5): {'att': 6.470576, 'se': 0.410519, 'n_obs': 891, 'n_treated': 129, 'n_control': 762},
    (4, 6): {'att': 8.155675, 'se': 0.4354133, 'n_obs': 781, 'n_treated': 129, 'n_control': 652},
    (5, 5): {'att': 2.948093, 'se': 0.4055165, 'n_obs': 871, 'n_treated': 109, 'n_control': 762},
    (5, 6): {'att': 4.808718, 'se': 0.4387234, 'n_obs': 761, 'n_treated': 109, 'n_control': 652},
    (6, 6): {'att': 2.274351, 'se': 0.4641536, 'n_obs': 762, 'n_treated': 110, 'n_control': 652},
}

# Stata已验证PSM结果 (从Stata MCP获取)
STATA_PSM_RESULTS = {
    (4, 4): {'att': 3.554019, 'se': 0.5866075, 'n_obs': 1000, 'n_treated': 129, 'n_control': 871},
    (4, 5): {'att': 7.801825, 'se': 0.6506115, 'n_obs': 891, 'n_treated': 129, 'n_control': 762},
    (4, 6): {'att': 8.543539, 'se': 0.5665262, 'n_obs': 781, 'n_treated': 129, 'n_control': 652},
    (5, 5): {'att': 3.837064, 'se': 0.5198648, 'n_obs': 871, 'n_treated': 109, 'n_control': 762},
    (5, 6): {'att': 5.314513, 'se': 0.6649302, 'n_obs': 761, 'n_treated': 109, 'n_control': 652},
    (6, 6): {'att': 1.934457, 'se': 0.6671495, 'n_obs': 762, 'n_treated': 110, 'n_control': 652},
}

# 样本信息
SAMPLE_INFO = {
    'n_total': 1000,
    'n_g4': 129,
    'n_g5': 109,
    'n_g6': 110,
    'n_ginf': 652,
    'T': 6,
    'cohorts': [4, 5, 6],
}

# 预期控制组cohorts（用于验证）
EXPECTED_CONTROL_COHORTS = {
    (4, 4): {5, 6, float('inf')},  # g∞, g5, g6
    (4, 5): {6, float('inf')},      # g∞, g6
    (4, 6): {float('inf')},         # g∞ only
    (5, 5): {6, float('inf')},      # g∞, g6
    (5, 6): {float('inf')},         # g∞ only
    (6, 6): {float('inf')},         # g∞ only
}

# 容差阈值
TOLERANCES = {
    'att_relative_error': 1e-6,
    'se_relative_error_ra_ipwra': 0.05,  # 5%
    'se_relative_error_psm': 0.10,        # 10%
    'transformation_absolute_error': 1e-10,
}


# =============================================================================
# 数据加载Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def staggered_data() -> pd.DataFrame:
    """
    加载Staggered测试数据 (Lee & Wooldridge 2023示例数据)
    
    数据结构:
    - id: 单位标识符 (1-1000)
    - year: 原始年份 (2001-2006)
    - period: 时期索引 (1-6, 从year转换)
    - y: 结果变量
    - g4, g5, g6, ginf: cohort指示符
    - f04, f05, f06: period指示符
    - x1, x2: 协变量
    - first_treat: 首次处理期 (4/5/6/inf)
    
    Returns
    -------
    pd.DataFrame
        面板数据 (6000 rows = 1000 units × 6 periods)
    """
    if not STAGGERED_DATA_FILE.exists():
        pytest.skip(
            f"Staggered data file not found: {STAGGERED_DATA_FILE}\n"
            f"Please ensure the Lee_Wooldridge_2023-main 3 folder is in the project root."
        )
    
    data = pd.read_stata(str(STAGGERED_DATA_FILE))
    
    # 确保数据类型正确
    data['id'] = data['id'].astype(int)
    data['year'] = data['year'].astype(int)
    
    # 添加period列 (year - 2000，使2001=1, ..., 2006=6)
    data['period'] = data['year'] - 2000
    
    # 创建first_treat变量
    # g4=1 表示 first_treat=4 (2004年开始处理)
    # ginf=1 或 g0=1 表示 never treated (first_treat=inf)
    if 'g4' in data.columns:
        data['first_treat'] = np.where(data['g4'] == 1, 4,
                              np.where(data['g5'] == 1, 5,
                              np.where(data['g6'] == 1, 6, float('inf'))))
    elif 'group' in data.columns:
        # 备选: 使用group列
        data['first_treat'] = np.where(data['group'] == 4, 4,
                              np.where(data['group'] == 5, 5,
                              np.where(data['group'] == 6, 6, float('inf'))))
    else:
        pytest.fail("Cannot create first_treat: no g4/g5/g6 or group columns found")
    
    # 验证数据结构
    required_cols = ['id', 'year', 'y', 'x1', 'x2']
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        pytest.fail(f"Missing required columns in data: {missing}")
    
    return data


@pytest.fixture(scope="module")
def stata_results() -> Dict[str, Any]:
    """
    加载Stata验证结果JSON
    
    包含:
    - metadata: 数据来源信息
    - data_info: T, cohorts等
    - sample_info: n_obs, n_by_cohort
    - estimators: ra, ipwra, psm各(g,r)的ATT和SE
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
def sample_info() -> Dict[str, Any]:
    """提取样本信息"""
    return SAMPLE_INFO


@pytest.fixture(scope="module")
def tolerances() -> Dict[str, float]:
    """提取误差阈值"""
    return TOLERANCES


@pytest.fixture(scope="module")
def stata_ipwra_results() -> Dict[Tuple[int, int], Dict[str, float]]:
    """返回Stata IPWRA验证结果字典"""
    return STATA_IPWRA_RESULTS


@pytest.fixture(scope="module")
def stata_ra_results() -> Dict[Tuple[int, int], Dict[str, float]]:
    """返回Stata RA验证结果字典"""
    return STATA_RA_RESULTS


@pytest.fixture(scope="module")
def stata_psm_results() -> Dict[Tuple[int, int], Dict[str, float]]:
    """返回Stata PSM验证结果字典"""
    return STATA_PSM_RESULTS


@pytest.fixture(scope="module")
def expected_control_cohorts() -> Dict[Tuple[int, int], set]:
    """返回预期控制组cohorts字典"""
    return EXPECTED_CONTROL_COHORTS


# =============================================================================
# 变换数据辅助函数
# =============================================================================

def compute_transformed_outcome_staggered(
    data: pd.DataFrame,
    y_col: str,
    id_col: str,
    period_col: str,
    gvar_col: str,
    cohort_g: int,
    target_period: int,
) -> pd.Series:
    """
    计算Staggered场景的变换后结果变量。
    
    公式 (论文4.12): 
        ŷ_{irg} = Y_{ir} - (1/(g-1)) × Σ_{s=1}^{g-1} Y_{is}
               = Y_{ir} - mean(Y_{i,pre(g)})
    
    **关键约束**:
    - 分母是 (g-1)（cohort固定），不是 (r-1)
    - 同一cohort的所有post periods使用相同的pre-mean
    
    Parameters
    ----------
    data : pd.DataFrame
        完整面板数据（需要包含所有时期）
    y_col : str
        结果变量列名
    id_col : str
        单位标识符列名
    period_col : str
        时期列名
    gvar_col : str
        首次处理期列名
    cohort_g : int
        目标cohort g（首次处理期）
    target_period : int
        目标评估期 r (r >= g)
    
    Returns
    -------
    pd.Series
        变换后的结果变量，index为id
        
    Notes
    -----
    Stata lag对照:
    - y_44 at f04: L1+L2+L3 → periods 3,2,1, 分母=3 (=4-1)
    - y_45 at f05: L2+L3+L4 → periods 3,2,1, 分母=3 (=4-1)
    - y_55 at f05: L1+L2+L3+L4 → periods 4,3,2,1, 分母=4 (=5-1)
    """
    assert target_period >= cohort_g, \
        f"target_period {target_period} must >= cohort_g {cohort_g}"
    
    # 获取period r的结果
    Y_r = data[data[period_col] == target_period].set_index(id_col)[y_col]
    
    # pre-treatment periods: 1, 2, ..., g-1
    pre_periods = list(range(1, cohort_g))  # [1,2,3] for g=4, [1,2,3,4] for g=5
    
    if len(pre_periods) == 0:
        raise ValueError(f"cohort_g={cohort_g} 没有pre-treatment期")
    
    # 计算pre-treatment均值 (分母 = g-1)
    pre_data = data[data[period_col].isin(pre_periods)]
    pre_mean = pre_data.groupby(id_col)[y_col].mean()
    
    # 变换: Y_r - pre_mean
    # 只保留两边都有值的单位
    common_ids = Y_r.index.intersection(pre_mean.index)
    y_transformed = Y_r.loc[common_ids] - pre_mean.loc[common_ids]
    
    return y_transformed


def build_subsample_for_gr(
    data: pd.DataFrame,
    cohort_g: int,
    period_r: int,
    gvar_col: str = 'first_treat',
    period_col: str = 'period',
    id_col: str = 'id',
    control_group: str = 'not_yet_treated',
) -> pd.DataFrame:
    """
    为特定(g,r)组合构建子样本，完全匹配Stata筛选条件。
    
    实现论文Procedure 4.1的子样本构建。
    
    Parameters
    ----------
    data : pd.DataFrame
        面板数据
    cohort_g : int
        目标处理cohort
    period_r : int
        目标评估时期
    gvar_col : str
        cohort变量列名
    period_col : str
        时期变量列名
    id_col : str
        单位标识列名
    control_group : str
        控制组类型: 'not_yet_treated' (NYT+NT) 或 'never_treated' (NT only)
    
    Returns
    -------
    pd.DataFrame
        包含'd'列（处理指示符）的子样本
        d=1 表示处理组（gvar == g）
        d=0 表示控制组（gvar > period 或 NT）
    
    Notes
    -----
    Stata对照:
    - (4,4): if f04 → period==4, 控制组={g5, g6, g∞}
    - (4,5): if f05 & ~g5 → period==5, 控制组={g6, g∞}
    - (4,6): if f06 & (g5+g6!=1) → period==6, 控制组={g∞}
    
    **关键约束**: 使用严格不等式 gvar > period
    """
    # Step 1: 筛选到period r的观测
    data_r = data[data[period_col] == period_r].copy()
    
    if len(data_r) == 0:
        raise ValueError(f"period={period_r}不存在于数据中")
    
    # Step 2: 获取单位级gvar
    unit_gvar = data_r.set_index(id_col)[gvar_col]
    
    # Step 3: 识别处理组 (D_g = 1): gvar == cohort_g
    is_treated = (unit_gvar == cohort_g)
    
    # Step 4: 识别控制组
    # CRITICAL: 使用严格不等式 gvar > period_r
    is_never_treated = np.isinf(unit_gvar)
    
    if control_group == 'never_treated':
        # 仅使用Never Treated
        is_control = is_never_treated
    else:  # 'not_yet_treated'
        # NYT + NT: gvar > period_r
        is_not_yet_treated = (unit_gvar > period_r)
        is_control = is_not_yet_treated  # 包含NT (inf > any number)
    
    # Step 5: 构建子样本掩码
    subsample_mask = is_treated | is_control
    valid_units = subsample_mask[subsample_mask].index
    
    # Step 6: 筛选数据
    subsample = data_r[data_r[id_col].isin(valid_units)].copy()
    
    # Step 7: 创建处理指示符
    subsample['d'] = (subsample[gvar_col] == cohort_g).astype(int)
    
    # Step 8: 验证
    n_treated = (subsample['d'] == 1).sum()
    n_control = (subsample['d'] == 0).sum()
    
    if n_treated == 0:
        raise ValueError(
            f"(g={cohort_g}, r={period_r}): 处理组为空。"
        )
    
    if n_control == 0:
        raise ValueError(
            f"(g={cohort_g}, r={period_r}): 控制组为空。"
        )
    
    return subsample


# =============================================================================
# 变换数据Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def transformed_data(staggered_data: pd.DataFrame) -> pd.DataFrame:
    """
    预计算所有(g,r)组合的变换后数据
    
    为每个(g,r)组合计算y_dot并添加到数据中。
    
    Returns
    -------
    pd.DataFrame
        包含y_dot_g{g}_r{r}列的数据
    """
    data = staggered_data.copy()
    
    for g, r in GR_COMBINATIONS:
        y_dot = compute_transformed_outcome_staggered(
            data, 'y', 'id', 'period', 'first_treat',
            cohort_g=g, target_period=r
        )
        col_name = f'y_dot_g{g}_r{r}'
        data[col_name] = data['id'].map(y_dot)
    
    return data


# =============================================================================
# 参数化Fixtures
# =============================================================================

@pytest.fixture(params=GR_COMBINATIONS, ids=[f'g{g}_r{r}' for g, r in GR_COMBINATIONS])
def gr_combination(request) -> Tuple[int, int]:
    """参数化(g,r)组合fixture"""
    return request.param


@pytest.fixture(params=['ra', 'ipwra', 'psm'], ids=['RA', 'IPWRA', 'PSM'])
def estimator(request) -> str:
    """参数化estimator fixture"""
    return request.param


# =============================================================================
# 辅助Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def controls() -> List[str]:
    """控制变量列表"""
    return ['x1', 'x2']


@pytest.fixture(scope="module")
def ps_controls() -> List[str]:
    """倾向得分模型控制变量"""
    return ['x1', 'x2']


@pytest.fixture
def get_stata_ipwra_result(stata_ipwra_results: Dict):
    """
    工厂fixture: 获取特定(g,r)的Stata IPWRA结果
    
    Usage:
        def test_ipwra(get_stata_ipwra_result):
            stata = get_stata_ipwra_result(4, 4)
            assert abs(result.att - stata['att']) < 1e-6
    """
    def _get_result(cohort_g: int, period_r: int) -> Dict[str, float]:
        return stata_ipwra_results[(cohort_g, period_r)]
    
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


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    # Constants
    'GR_COMBINATIONS',
    'STATA_IPWRA_RESULTS',
    'SAMPLE_INFO',
    'EXPECTED_CONTROL_COHORTS',
    'TOLERANCES',
    # Functions
    'compute_transformed_outcome_staggered',
    'build_subsample_for_gr',
    'assert_att_close_to_stata',
    'assert_se_close_to_stata',
]
