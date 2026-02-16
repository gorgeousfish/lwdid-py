"""
Staggered DiD 数据生成过程 (DGP) 生成器

基于 Lee & Wooldridge (2023) 论文设定，用于Monte Carlo模拟。

DGP参数（论文Section 7.2）:
- N = 1000 units, T = 6 periods
- Cohort shares: g4=12%, g5=11%, g6=11%, NT=66%
- 真实ATT: τ_{g,r} = 1.5 + 0.5*(r-g) + 0.3*(g-4)

使用方法:
>>> from fixtures.dgp_generator import StaggeredDGP
>>> dgp = StaggeredDGP(n_units=1000, seed=42)
>>> data = dgp.generate()
>>> true_att = dgp.get_true_att(g=4, r=5)
"""
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class StaggeredDGP:
    """
    Staggered DiD 数据生成过程。
    
    实现Lee & Wooldridge (2023)论文中的DGP设定，
    用于Monte Carlo验证。
    
    Attributes
    ----------
    n_units : int
        单位数量
    n_periods : int
        时期数量
    cohort_shares : Dict[int, float]
        各cohort的份额 {cohort: share}
    base_att : float
        基础处理效应
    exposure_coef : float
        暴露时间系数
    cohort_coef : float
        cohort系数
    seed : int
        随机种子
    """
    
    def __init__(
        self,
        n_units: int = 1000,
        n_periods: int = 6,
        cohort_shares: Optional[Dict[int, float]] = None,
        base_att: float = 1.5,
        exposure_coef: float = 0.5,
        cohort_coef: float = 0.3,
        seed: Optional[int] = None,
    ):
        """
        初始化DGP。
        
        Parameters
        ----------
        n_units : int
            单位数量
        n_periods : int
            时期数量 (T)
        cohort_shares : Dict[int, float], optional
            各cohort的份额。默认: {0: 0.66, 4: 0.12, 5: 0.11, 6: 0.11}
            cohort=0 表示 Never Treated
        base_att : float
            基础处理效应 τ_0
        exposure_coef : float
            暴露时间系数 β (r-g的系数)
        cohort_coef : float
            cohort系数 γ (g-4的系数)
        seed : int, optional
            随机种子
        """
        self.n_units = n_units
        self.n_periods = n_periods
        
        if cohort_shares is None:
            # 默认份额（论文设定）
            self.cohort_shares = {0: 0.66, 4: 0.12, 5: 0.11, 6: 0.11}
        else:
            self.cohort_shares = cohort_shares
        
        self.base_att = base_att
        self.exposure_coef = exposure_coef
        self.cohort_coef = cohort_coef
        
        if seed is not None:
            np.random.seed(seed)
        self.seed = seed
        
        # 验证份额和为1
        total_share = sum(self.cohort_shares.values())
        if abs(total_share - 1.0) > 1e-10:
            raise ValueError(f"cohort_shares必须和为1，当前: {total_share}")
    
    def get_true_att(self, g: int, r: int) -> float:
        """
        计算真实ATT τ_{g,r}。
        
        公式: τ_{g,r} = τ_0 + β*(r-g) + γ*(g-4)
        
        Parameters
        ----------
        g : int
            Cohort (处理开始时期)
        r : int
            评估时期
        
        Returns
        -------
        float
            真实ATT
        """
        if g == 0:  # Never Treated
            return 0.0
        if r < g:  # Pre-treatment
            return 0.0
        
        # τ_{g,r} = 1.5 + 0.5*(r-g) + 0.3*(g-4)
        return self.base_att + self.exposure_coef * (r - g) + self.cohort_coef * (g - 4)
    
    def generate(self) -> pd.DataFrame:
        """
        生成模拟数据。
        
        Returns
        -------
        pd.DataFrame
            面板数据，包含列:
            - id: 单位标识
            - year: 时期 (1, 2, ..., T)
            - y: 结果变量
            - gvar: cohort (0=NT, 4, 5, 6)
            - x1, x2: 协变量
            - treated: 处理状态 (0/1)
        """
        # Step 1: 分配cohorts
        cohorts = list(self.cohort_shares.keys())
        probs = list(self.cohort_shares.values())
        unit_cohorts = np.random.choice(cohorts, size=self.n_units, p=probs)
        
        # Step 2: 生成单位级协变量
        x1 = np.random.randn(self.n_units)
        x2 = np.random.randn(self.n_units)
        
        # Step 3: 生成面板数据
        data_list = []
        
        for i in range(self.n_units):
            g = unit_cohorts[i]
            
            for t in range(1, self.n_periods + 1):
                # 基础结果模型: Y = α + β_1*t + β_2*x1 + β_3*x2 + ε
                y_base = 1.0 + 0.5 * t + 0.3 * x1[i] + 0.2 * x2[i]
                
                # 个体效应 (固定效应的一种模拟)
                individual_effect = 0.1 * i / self.n_units
                
                # 添加随机误差
                epsilon = np.random.randn()
                
                # 处理效应
                if g > 0 and t >= g:
                    tau = self.get_true_att(g, t)
                    treated = 1
                else:
                    tau = 0.0
                    treated = 0
                
                y = y_base + individual_effect + tau + epsilon
                
                data_list.append({
                    'id': i + 1,
                    'year': t,
                    'y': y,
                    'gvar': g,
                    'x1': x1[i],
                    'x2': x2[i],
                    'treated': treated,
                    # 便于与Stata对照的指示变量
                    'g0': 1 if g == 0 else 0,
                    'g4': 1 if g == 4 else 0,
                    'g5': 1 if g == 5 else 0,
                    'g6': 1 if g == 6 else 0,
                    'f04': 1 if t == 4 else 0,
                    'f05': 1 if t == 5 else 0,
                    'f06': 1 if t == 6 else 0,
                })
        
        return pd.DataFrame(data_list)
    
    def get_all_true_atts(self) -> Dict[Tuple[int, int], float]:
        """
        返回所有有效(g,r)组合的真实ATT。
        
        Returns
        -------
        Dict[Tuple[int, int], float]
            {(g, r): true_att}
        """
        true_atts = {}
        for g in [c for c in self.cohort_shares.keys() if c > 0]:
            for r in range(g, self.n_periods + 1):
                true_atts[(g, r)] = self.get_true_att(g, r)
        return true_atts


def generate_staggered_data(
    n_units: int = 1000,
    n_periods: int = 6,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[Tuple[int, int], float]]:
    """
    便捷函数：生成staggered数据和真实ATT。
    
    Parameters
    ----------
    n_units : int
        单位数量
    n_periods : int
        时期数量
    seed : int, optional
        随机种子
    
    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        (数据, {(g,r): true_att})
    """
    dgp = StaggeredDGP(n_units=n_units, n_periods=n_periods, seed=seed)
    data = dgp.generate()
    true_atts = dgp.get_all_true_atts()
    return data, true_atts


if __name__ == '__main__':
    # 测试DGP
    dgp = StaggeredDGP(n_units=100, seed=42)
    data = dgp.generate()
    
    print("数据形状:", data.shape)
    print("\nCohort分布:")
    print(data.groupby('gvar')['id'].nunique())
    
    print("\n真实ATT:")
    for (g, r), att in dgp.get_all_true_atts().items():
        print(f"  τ_({g},{r}) = {att:.4f}")
