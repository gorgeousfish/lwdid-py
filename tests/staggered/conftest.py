"""
Staggered DiD测试共享fixtures和辅助函数。

提供：
- Stata Staggered测试数据加载
- Stata验证结果加载
- 子样本构建辅助函数
- 变换变量计算辅助函数
"""
import json
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytest


# ============================================================================
# 数据路径辅助函数
# ============================================================================

def get_test_data_path(filename: str) -> str:
    """获取测试数据文件的完整路径。"""
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(tests_dir, filename)


def get_stata_data_path(filename: str) -> str:
    """获取Stata参考数据的完整路径。"""
    # 项目根目录: /Users/cxy/Desktop/大样本lwdid
    # 数据路径: /Users/cxy/Desktop/大样本lwdid/Lee_Wooldridge_2023-main 3/
    # 当前文件: .../lwdid-py_v0.1.0/tests/staggered/conftest.py
    tests_dir = os.path.dirname(os.path.abspath(__file__))  # tests/staggered
    package_root = os.path.dirname(os.path.dirname(tests_dir))  # lwdid-py_v0.1.0
    project_root = os.path.dirname(package_root)  # 大样本lwdid
    return os.path.join(project_root, 'Lee_Wooldridge_2023-main 3', filename)


# ============================================================================
# Stata验证数据（硬编码，与Stata输出完全一致）
# ============================================================================

STATA_IPWRA_RESULTS = {
    # (cohort_g, period_r): {att, se, n_obs, n_treated, n_control}
    (4, 4): {
        'att': 4.3029238,
        'se': 0.42367713,
        'n_obs': 1000,
        'n_treated': 129,
        'n_control': 871,
        'condition': 'f04',
    },
    (4, 5): {
        'att': 6.6112909,
        'se': 0.43215951,
        'n_obs': 891,
        'n_treated': 129,
        'n_control': 762,
        'condition': 'f05 & ~g5',
    },
    (4, 6): {
        'att': 8.3343553,
        'se': 0.44138304,
        'n_obs': 781,
        'n_treated': 129,
        'n_control': 652,
        'condition': 'f06 & (g5 + g6 != 1)',
    },
    (5, 5): {
        'att': 3.0283627,
        'se': 0.42077459,
        'n_obs': 871,
        'n_treated': 109,
        'n_control': 762,
        'condition': 'f05 & ~g4',
    },
    (5, 6): {
        'att': 4.9326076,
        'se': 0.44026846,
        'n_obs': 761,
        'n_treated': 109,
        'n_control': 652,
        'condition': 'f06 & (g4 + g6 != 1)',
    },
    (6, 6): {
        'att': 2.4200472,
        'se': 0.48314986,
        'n_obs': 762,
        'n_treated': 110,
        'n_control': 652,
        'condition': 'f06 & (g4 + g5 != 1)',
    },
}

# 预期控制组cohorts（用于验证）
EXPECTED_CONTROL_COHORTS = {
    (4, 4): {5, 6, 0},  # 0 表示NT
    (4, 5): {6, 0},
    (4, 6): {0},        # 仅NT
    (5, 5): {6, 0},
    (5, 6): {0},
    (6, 6): {0},
}


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def staggered_data():
    """
    加载Staggered测试数据（Lee & Wooldridge 2023）。
    
    原始数据结构:
    - id: 单位标识符
    - year: 时期 (2001-2006)
    - y: 结果变量
    - group: cohort值 (0=NT, 4=g4, 5=g5, 6=g6)
    - g0, g4, g5, g6: cohort指示符
    - f01-f06: 时期指示符
    - x1, x2: 协变量
    - w: 处理状态
    
    转换后:
    - 将year从2001-2006转为1-6
    - 创建gvar列（从group列）
    """
    data_path = get_stata_data_path('2.lee_wooldridge_staggered_data.dta')
    
    if not os.path.exists(data_path):
        pytest.skip(f"Staggered测试数据不存在: {data_path}")
    
    data = pd.read_stata(data_path)
    
    # 确保数据类型正确
    data['id'] = data['id'].astype(int)
    
    # 将year从2001-2006转为1-6（与Stata do文件一致）
    # Stata中 f04 表示 year==2004，对应我们的 period==4
    data['year'] = data['year'].astype(int) - 2000
    
    # 创建gvar列：从group列
    # group=0 表示NT, group=4 表示cohort4 (year 2004对应period 4)
    data['gvar'] = data['group'].fillna(0).astype(int)
    
    return data


@pytest.fixture(scope="module")
def stata_ipwra_results():
    """返回Stata IPWRA验证结果字典。"""
    return STATA_IPWRA_RESULTS


@pytest.fixture(scope="module")
def expected_control_cohorts():
    """返回预期控制组cohorts字典。"""
    return EXPECTED_CONTROL_COHORTS


@pytest.fixture
def small_staggered_data():
    """创建小型测试数据（用于快速单元测试）。"""
    n_units = 100
    T = 6
    
    np.random.seed(42)
    
    # 生成单位级特征
    # cohort分配: 12% g=4, 11% g=5, 11% g=6, 66% NT(g=0)
    cohort_probs = [0.66, 0.12, 0.11, 0.11]
    cohort_values = [0, 4, 5, 6]
    unit_cohorts = np.random.choice(cohort_values, size=n_units, p=cohort_probs)
    
    # 生成协变量
    x1 = np.random.randn(n_units)
    x2 = np.random.randn(n_units)
    
    # 生成面板数据
    data_list = []
    for i in range(n_units):
        for t in range(1, T + 1):
            # 基础结果（线性时间趋势）
            y_base = 1 + 0.5 * t + 0.3 * x1[i] + 0.2 * x2[i] + np.random.randn()
            
            # 处理效应
            g = unit_cohorts[i]
            if g > 0 and t >= g:
                # τ_{g,r} = 1.5 + 0.5*(r-g) + 0.3*(g-4)
                tau = 1.5 + 0.5 * (t - g) + 0.3 * (g - 4)
                y = y_base + tau
                treated = 1
            else:
                y = y_base
                treated = 0
            
            data_list.append({
                'id': i + 1,
                'year': t,
                'y': y,
                'gvar': g,
                'x1': x1[i],
                'x2': x2[i],
                'treated': treated,
                # 创建Stata风格的指示变量
                'g4': 1 if g == 4 else 0,
                'g5': 1 if g == 5 else 0,
                'g6': 1 if g == 6 else 0,
                'f04': 1 if t == 4 else 0,
                'f05': 1 if t == 5 else 0,
                'f06': 1 if t == 6 else 0,
            })
    
    return pd.DataFrame(data_list)


# ============================================================================
# 子样本构建辅助函数
# ============================================================================

def build_subsample_for_gr(
    data: pd.DataFrame,
    cohort_g: int,
    period_r: int,
    gvar_col: str = 'gvar',
    period_col: str = 'year',
    id_col: str = 'id',
    never_treated_values: Optional[List] = None,
) -> pd.DataFrame:
    """
    为特定(g,r)组合构建子样本，完全匹配Stata筛选条件。
    
    实现论文Procedure 4.1的子样本构建，等价于Stata条件筛选。
    
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
    never_treated_values : List, optional
        NT单位的gvar值，默认 [0]
    
    Returns
    -------
    pd.DataFrame
        包含'd'列（处理指示符）的子样本
        d=1 表示处理组（cohort == g）
        d=0 表示控制组（gvar > period 或 NT）
    
    Raises
    ------
    ValueError
        当处理组或控制组为空时
    
    Notes
    -----
    Stata对照:
    - (4,4): if f04 → period==4
    - (4,5): if f05 & ~g5 → period==5 & gvar!=5
    - (4,6): if f06 & (g5+g6!=1) → period==6 & ~gvar.isin([5,6])
    - (5,5): if f05 & ~g4 → period==5 & gvar!=4
    - (5,6): if f06 & (g4+g6!=1) → period==6 & ~gvar.isin([4,6])
    - (6,6): if f06 & (g4+g5!=1) → period==6 & ~gvar.isin([4,5])
    
    关键约束:
    - 控制组使用 gvar > period (严格不等式)
    - gvar == period 表示当期开始处理，属于处理组
    """
    if never_treated_values is None:
        never_treated_values = [0]
    
    # Step 1: 筛选到period r的观测
    data_r = data[data[period_col] == period_r].copy()
    
    if len(data_r) == 0:
        raise ValueError(f"period={period_r}不存在于数据中")
    
    # Step 2: 获取单位级gvar
    unit_gvar = data_r.set_index(id_col)[gvar_col]
    
    # Step 3: 识别处理组 (D_g = 1)
    is_treated = (unit_gvar == cohort_g)
    
    # Step 4: 识别控制组 (A_{r+1} = 1)
    # CRITICAL: 使用严格不等式 gvar > period_r
    is_never_treated = unit_gvar.isin(never_treated_values)
    is_not_yet_treated = (unit_gvar > period_r)  # 严格大于
    is_control = is_never_treated | is_not_yet_treated
    
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
            f"cohort {cohort_g} 在 period {period_r} 不存在。"
        )
    
    if n_control == 0:
        raise ValueError(
            f"(g={cohort_g}, r={period_r}): 控制组为空。"
            f"所有潜在控制单位在 period {period_r} 都已被处理。"
        )
    
    return subsample


def compute_transformed_outcome(
    data: pd.DataFrame,
    y_col: str,
    id_col: str,
    period_col: str,
    cohort_g: int,
    period_r: int,
) -> pd.Series:
    """
    计算变换后的结果变量 ŷ_{irg}。
    
    公式: ŷ_{irg} = Y_{ir} - (1/(g-1)) * Σ_{s=1}^{g-1} Y_{is}
    
    **关键**: 分母是 g-1（cohort固定），不是 r-1！
    
    Parameters
    ----------
    data : pd.DataFrame
        完整面板数据（需要包含所有时期）
    y_col : str
        结果变量列名
    id_col : str
        单位标识列名
    period_col : str
        时期变量列名
    cohort_g : int
        处理cohort（决定pre-treatment期数）
    period_r : int
        评估时期
    
    Returns
    -------
    pd.Series
        变换后的结果变量，index为id
    
    Notes
    -----
    Stata对照 (lag转换):
    - y_44 at r=4: L1.y + L2.y + L3.y → periods 3,2,1, 分母=3 (=4-1)
    - y_45 at r=5: L2.y + L3.y + L4.y → periods 3,2,1, 分母=3 (=4-1)
    - y_55 at r=5: L1.y + ... + L4.y → periods 4,3,2,1, 分母=4 (=5-1)
    - y_66 at r=6: L1.y + ... + L5.y → periods 5,4,3,2,1, 分母=5 (=6-1)
    """
    # 获取period r的结果
    Y_r = data[data[period_col] == period_r].set_index(id_col)[y_col]
    
    # pre-treatment periods: 1, 2, ..., g-1
    pre_periods = list(range(1, cohort_g))  # [1, 2, ..., g-1]
    
    if len(pre_periods) == 0:
        raise ValueError(f"cohort_g={cohort_g} 没有pre-treatment期")
    
    # 计算pre-treatment均值
    pre_data = data[data[period_col].isin(pre_periods)]
    pre_mean = pre_data.groupby(id_col)[y_col].mean()
    
    # 变换: Y_r - pre_mean
    # 只保留两边都有值的单位
    common_ids = Y_r.index.intersection(pre_mean.index)
    y_transformed = Y_r.loc[common_ids] - pre_mean.loc[common_ids]
    
    return y_transformed


def validate_subsample(
    subsample: pd.DataFrame,
    cohort_g: int,
    period_r: int,
    d_col: str = 'd',
    min_control_size: int = 10,
    max_imbalance_ratio: float = 20.0,
) -> Tuple[bool, List[str]]:
    """
    验证子样本并返回警告列表。
    
    Parameters
    ----------
    subsample : pd.DataFrame
        构建好的子样本
    cohort_g : int
        目标cohort
    period_r : int
        目标period
    d_col : str
        处理指示符列名
    min_control_size : int
        控制组最小样本量警告阈值
    max_imbalance_ratio : float
        最大不平衡比例警告阈值
    
    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, warnings)
    
    Raises
    ------
    ValueError
        当子样本无效时（控制组或处理组为空）
    """
    warnings_list = []
    
    n_treated = (subsample[d_col] == 1).sum()
    n_control = (subsample[d_col] == 0).sum()
    
    # 检查控制组为空
    if n_control == 0:
        raise ValueError(
            f"(g={cohort_g}, r={period_r}): 控制组为空。"
        )
    
    # 检查处理组为空
    if n_treated == 0:
        raise ValueError(
            f"(g={cohort_g}, r={period_r}): 处理组为空。"
        )
    
    # 警告：控制组样本量过小
    if n_control < min_control_size:
        warnings_list.append(
            f"(g={cohort_g}, r={period_r}): 控制组样本量较小 (n={n_control})。"
        )
    
    # 警告：极端不平衡
    ratio = n_treated / n_control
    if ratio < 1 / max_imbalance_ratio or ratio > max_imbalance_ratio:
        warnings_list.append(
            f"(g={cohort_g}, r={period_r}): 处理组/控制组比例极端 ({ratio:.2f})。"
        )
    
    return True, warnings_list
