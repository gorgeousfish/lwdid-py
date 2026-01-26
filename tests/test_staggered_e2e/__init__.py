# -*- coding: utf-8 -*-
"""
Story 6.2: Staggered端到端测试模块

验证Staggered（交错处理）场景下所有(g,r)组合的正确性，
确保与Stata teffects命令完全一致。

测试内容:
- 变换变量计算 (y_{gr})
- 控制组选择 (NYT + NT)
- RA估计量一致性
- IPWRA估计量一致性
- PSM估计量一致性
- Cohort和Overall聚合
- HC标准误验证
- Monte Carlo验证
- Vibe Math公式验证

论文参考: Lee & Wooldridge (2023) Section 4, Procedure 4.1
Stata参考: Lee_Wooldridge_2023-main 3/2.lee_wooldridge_rolling_staggered.do
"""
