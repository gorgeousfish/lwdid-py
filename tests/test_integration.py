"""
集成测试 - 端到端Stata对齐验证

测试完整lwdid()函数，对比Stata基准值（论文表3）
Story 1.2: 新增T003-T004（detrend总体ATT）和T010-T012（逐期效应）
"""

import numpy as np
import pandas as pd
import pytest

from lwdid import lwdid
from lwdid.exceptions import (
    NoControlUnitsError,
    NoTreatedUnitsError,
    TimeDiscontinuityError,
)

# 使用conftest.py中定义的fixture


class TestSmokingDataIntegration:
    """Smoking数据集端到端测试（对比论文表3）"""
    
    def test_T001_smoking_demean_ols(self, smoking_data):
        """T001: smoking + demean + vce=None
        
        核心端到端测试，验证与Stata完全对齐
        
        对应Story 1.1, AC6, Task 6.6
        对应PRD第5.1.2节测试矩阵T001
        
        Stata命令:
            lwdid lcigsale d, ivar(state) tvar(year) post(post) rolling(demean)
        
        基准值（论文表3，Procedure 2.1列）:
            ATT = -0.422, SE = 0.121, t ≈ -3.49, p ≈ 0.001, N = 39
        
        关键列名:
            - y='lcigsale' (对数香烟销量，论文使用)
            - d='d' (处理指示，1=California)
            - post='post' (后期指示，1=1989年及以后)
        """
        # 使用fixture加载smoking数据
        data = smoking_data
        
        # 执行lwdid估计
        results = lwdid(
            data,
            y='lcigsale',  # ⚠️ 必须使用对数值
            d='d',
            ivar='state',
            tvar='year',
            post='post',
            rolling='demean',
            vce=None,
        )
        
        # === 验证点估计（论文值四舍五入到3位小数）===
        # 实际: ATT ≈ -0.42217, SE ≈ 0.12080
        # 论文表3: ATT = -0.422, SE = 0.121（四舍五入）
        # 使用5e-4容差（考虑论文四舍五入）
        assert abs(results.att - (-0.422)) < 5e-4, \
            f"ATT mismatch: {results.att} vs -0.422"
        
        # === 验证标准误 ===
        assert abs(results.se_att - 0.121) < 5e-4, \
            f"SE mismatch: {results.se_att} vs 0.121"
        
        # === 验证t统计量（论文值四舍五入到2位小数）===
        # 实际: t ≈ -3.4948
        # 论文表3: t ≈ -3.49（四舍五入）
        assert abs(results.t_stat - (-3.49)) < 1e-2, \
            f"t-stat mismatch: {results.t_stat} vs -3.49"
        
        # === 验证p值（论文值四舍五入到3位小数）===
        # 实际: p ≈ 0.00125
        # 论文表3: p ≈ 0.001（四舍五入）
        assert abs(results.pvalue - 0.001) < 1e-3, \
            f"p-value mismatch: {results.pvalue} vs 0.001"
        
        # === 验证样本量（L0层，精确匹配）===
        assert results.nobs == 39, \
            f"N mismatch: {results.nobs} vs 39"
        
        # === 验证时间结构（L0层，精确匹配）===
        # K = 19（1988年，最后前期对应tindex=19）
        # tpost1 = 20（1989年，首个后期对应tindex=20）
        assert results.K == 19, f"K mismatch: {results.K} vs 19"
        assert results.tpost1 == 20, f"tpost1 mismatch: {results.tpost1} vs 20"
        
        # === 验证单位数（精确匹配）===
        assert results.n_treated == 1, "应有1个处理单位（California）"
        assert results.n_control == 38, "应有38个对照单位"
        
        # === 验证方法元数据 ===
        assert results.cmd == 'lwdid'
        assert results.rolling == 'demean'
        assert results.vce_type == 'ols'
        assert results.depvar == 'lcigsale'
    
    def test_T002_smoking_demean_hc3(self, smoking_data):
        """T002: smoking + demean + vce='hc3'
        
        对应PRD第5.1.2节测试矩阵T002
        
        注意：smoking数据N₁=1（单处理单位California），HC3在此场景下
        因杠杆值h=1而产生极大的SE（论文建议HC3需要"a handful of treated units"）
        
        本测试验证：
        1. HC3功能可运行（不报错）
        2. ATT系数与OLS相同（vce不影响点估计）
        3. 方法元数据正确
        
        不验证SE具体值（因单处理单位时HC3不稳定）
        """
        data = smoking_data
        
        results = lwdid(
            data,
            y='lcigsale',
            d='d',
            ivar='state',
            tvar='year',
            post='post',
            rolling='demean',
            vce='hc3',
        )
        
        # ATT系数应与T001相同（vce不影响点估计）
        assert abs(results.att - (-0.422)) < 5e-4
        
        # HC3能运行并返回有效SE（不验证具体值）
        # 原因：N₁=1时杠杆值h=1，HC3 SE异常大（符合理论预期）
        assert results.se_att > 0
        assert not np.isnan(results.se_att)
        
        # 验证方差估计类型
        assert results.vce_type == 'hc3'
        
        # N, K, tpost1应与T001相同
        assert results.nobs == 39
        assert results.K == 19
        assert results.tpost1 == 20
    
    def test_T003_smoking_detrend_ols(self, smoking_data):
        """T003: smoking + detrend + vce=None
        
        对应Story 1.2, AC6, Task 5.2
        对应PRD第5.1.2节测试矩阵T003
        
        Stata命令:
            lwdid lcigsale d, ivar(state) tvar(year) post(post) rolling(detrend)
        
        基准值（论文表3，Procedure 3.1列）:
            ATT = -0.227, SE = 0.094, t ≈ -2.41, p ≈ 0.021, N = 39
        
        验证点:
        - 总体平均ATT对齐论文表3
        - att_by_period DataFrame结构正确
        - average行在首行
        - 共13行（1 average + 12 periods，1989-2000）
        """
        # 加载smoking数据
        data = smoking_data
        
        # 执行lwdid估计（使用detrend方法）
        results = lwdid(
            data,
            y='lcigsale',  # ⚠️ 使用对数香烟销量
            d='d',
            ivar='state',
            tvar='year',
            post='post',
            rolling='detrend',
            vce=None,
        )
        
        # === 验证总体平均ATT（Stata精确基准值）===
        # Stata基准: ATT = -0.2269887, SE = 0.0940689
        assert abs(results.att - (-0.2269887)) < 1e-7, \
            f"ATT mismatch: {results.att} vs -0.2269887"
        
        assert abs(results.se_att - 0.0940689) < 1e-7, \
            f"SE mismatch: {results.se_att} vs 0.0940689"
        
        # t统计量: -2.413005
        assert abs(results.t_stat - (-2.413005)) < 1e-4, \
            f"t-stat mismatch: {results.t_stat} vs -2.413005"
        
        # p值: 0.0208919
        assert abs(results.pvalue - 0.0208919) < 1e-4, \
            f"p-value mismatch: {results.pvalue} vs 0.0208919"
        
        # === 验证样本量和时间结构 ===
        assert results.nobs == 39
        assert results.K == 19
        assert results.tpost1 == 20
        
        # === 验证att_by_period DataFrame ===
        assert results.att_by_period is not None, "att_by_period应该存在"
        assert isinstance(results.att_by_period, pd.DataFrame)
        
        # 验证行数：1 average + 12 periods (1989-2000)
        assert len(results.att_by_period) == 13, \
            f"att_by_period应有13行，实际{len(results.att_by_period)}行"
        
        # === 验证第一行（average行）===
        avg_row = results.att_by_period.iloc[0]
        assert avg_row['period'] == 'average'
        assert avg_row['tindex'] == '-'
        assert abs(avg_row['beta'] - (-0.227)) < 1e-4
        assert abs(avg_row['se'] - 0.094) < 1e-4
        
        # === 验证列类型 ===
        assert results.att_by_period['period'].dtype == object  # str
        assert results.att_by_period['tindex'].dtype == object  # str
        assert results.att_by_period['N'].dtype in [np.int64, int]
        
        # === 验证列顺序（对应Stata第489行）===
        expected_cols = ['period', 'tindex', 'beta', 'se', 'ci_lower', 'ci_upper', 'tstat', 'pval', 'N']
        assert list(results.att_by_period.columns) == expected_cols
    
    def test_T004_smoking_detrend_hc3(self, smoking_data):
        """T004: smoking + detrend + vce='hc3'
        
        对应Story 1.2, AC6, Task 5.3
        对应PRD第5.1.2节测试矩阵T004
        
        验证点:
        - ATT系数与T003相同（vce不影响点估计）
        - HC3标准误有效（>0且非NaN）
        - att_by_period存在
        """
        data = smoking_data
        
        results = lwdid(
            data,
            y='lcigsale',
            d='d',
            ivar='state',
            tvar='year',
            post='post',
            rolling='detrend',
            vce='hc3',
        )
        
        # ATT系数应与T003相同（vce不影响点估计）
        assert abs(results.att - (-0.227)) < 1e-4
        
        # HC3能运行并返回有效SE
        assert results.se_att > 0
        assert not np.isnan(results.se_att)
        
        # 验证方差估计类型
        assert results.vce_type == 'hc3'
        
        # att_by_period应存在
        assert results.att_by_period is not None
        assert len(results.att_by_period) == 13
    
    def test_T010_period_effect_1989(self, smoking_data):
        """T010: 1989年逐期效应验证
        
        对应Story 1.2, AC7, Task 6.1
        对应PRD第5.1.2节测试矩阵T010
        
        基准值（论文表3，τ_{1989}行）:
            beta = -0.043, se = 0.059, t ≈ -0.73, p ≈ 0.470
        
        验证点:
        - 筛选period='1989'行
        - beta, se对齐论文值
        """
        data = smoking_data
        
        results = lwdid(
            data, y='lcigsale', d='d', ivar='state', 
            tvar='year', post='post', rolling='detrend', vce=None
        )
        
        # 筛选1989年行
        row_1989 = results.att_by_period[results.att_by_period['period'] == '1989']
        assert len(row_1989) == 1, "应有恰好1行1989年数据"
        
        # 验证beta和se（Stata精确基准值）
        # Stata基准: beta = -0.04226831, se = 0.0592916
        assert abs(row_1989['beta'].values[0] - (-0.04226831)) < 1e-7, \
            f"1989年beta不匹配: {row_1989['beta'].values[0]} vs -0.04226831"
        
        assert abs(row_1989['se'].values[0] - 0.0592916) < 1e-7, \
            f"1989年se不匹配: {row_1989['se'].values[0]} vs 0.0592916"
        
        # 验证t统计量和p值
        # Stata基准: tstat = -0.7128885, pval = 0.4803869
        assert abs(row_1989['tstat'].values[0] - (-0.7128885)) < 1e-4
        assert abs(row_1989['pval'].values[0] - 0.4803869) < 1e-4
    
    def test_T011_period_effect_1995(self, smoking_data):
        """T011: 1995年逐期效应验证
        
        对应Story 1.2, AC7, Task 6.2
        
        Stata基准值:
            beta = -0.28203907, se = 0.1121333, t = -2.515211, p = 0.016369
        """
        data = smoking_data
        
        results = lwdid(
            data, y='lcigsale', d='d', ivar='state', 
            tvar='year', post='post', rolling='detrend'
        )
        
        # 筛选1995年行
        row_1995 = results.att_by_period[results.att_by_period['period'] == '1995']
        
        # 验证beta和se（L2精度）
        assert abs(row_1995['beta'].values[0] - (-0.28203907)) < 1e-7
        assert abs(row_1995['se'].values[0] - 0.1121333) < 1e-7
    
    def test_T012_period_effect_2000(self, smoking_data):
        """T012: 2000年逐期效应验证（最后一期）
        
        对应Story 1.2, AC7, Task 6.3
        
        Stata基准值:
            beta = -0.40287678, se = 0.1524529, t = -2.642632, p = 0.0119885
        """
        data = smoking_data
        
        results = lwdid(
            data, y='lcigsale', d='d', ivar='state', 
            tvar='year', post='post', rolling='detrend'
        )
        
        # 筛选2000年行
        row_2000 = results.att_by_period[results.att_by_period['period'] == '2000']
        
        # 验证beta和se（L2精度）
        assert abs(row_2000['beta'].values[0] - (-0.40287678)) < 1e-7
        assert abs(row_2000['se'].values[0] - 0.1524529) < 1e-7


class TestMVEIntegration:
    """MVE数据端到端测试"""
    
    def test_mve_demean_end_to_end(self, smoking_data):
        """MVE端到端测试：完整lwdid()调用
        
        对应PRD第5.4.3节: MVE端到端验证
        
        期望ATT = 3.5（手工计算）
        """
        data = pd.read_csv('tests/data/mve_demean.csv')
        
        results = lwdid(
            data,
            y='y',
            d='d',
            ivar='id',
            tvar='year',
            post='post',
            rolling='demean',
        )
        
        # 验证ATT
        assert abs(results.att - 3.5) < 1e-7
        
        # 验证样本信息
        assert results.nobs == 3
        assert results.n_treated == 1
        assert results.n_control == 2
        assert results.K == 2
        assert results.tpost1 == 3
    
    def test_mve_detrend_end_to_end(self, smoking_data):
        """MVE Detrend端到端测试：完整lwdid()调用
        
        对应Story 1.2, Task 1.3
        对应PRD第5.4.1节: MVE验证（完美线性趋势）
        
        数据: N=3, T=5（前3后2）
        - 单位1（处理）: y=3+2t+5·post，期望ATT=5
        - 单位2,3（对照）: 完美线性趋势，残差=0
        
        手工计算期望值:
        - ydot_postavg₁ = 5.0
        - ydot_postavg₂ = 0.0
        - ydot_postavg₃ = 0.0
        - ATT = 5.0 - 0.0 = 5.0
        """
        data = pd.read_csv('tests/data/mve_detrend.csv')

        # Fix BUG: Drop 'tindex' column if present (reserved column name)
        # The CSV file contains 'tindex' which is a reserved column name used internally by lwdid
        if 'tindex' in data.columns:
            data = data.drop(columns=['tindex'])

        results = lwdid(
            data,
            y='y',
            d='d',
            ivar='id',
            tvar='year',
            post='post',
            rolling='detrend',
        )
        
        # 验证ATT（完美线性数据，精度应极高）
        assert abs(results.att - 5.0) < 1e-7
        
        # 验证样本信息
        assert results.nobs == 3
        assert results.n_treated == 1
        assert results.n_control == 2
        assert results.K == 3  # 前期tindex=1,2,3
        assert results.tpost1 == 4  # 首个后期tindex=4
        
        # 验证att_by_period
        assert results.att_by_period is not None
        # 1 average + 2 periods (tindex=4,5)
        assert len(results.att_by_period) == 3


class TestBoundaryConditions:
    """边界条件测试"""
    
    def test_B001_minimum_sample(self, smoking_data):
        """B001: 最小样本（N=3, N₀=2, N₁=1）应正常运行"""
        data = pd.read_csv('tests/data/mve_demean.csv')
        
        # N=3是最小允许样本
        results = lwdid(data, 'y', 'd', 'id', 'year', 'post', 'demean')
        
        assert results.nobs == 3
        assert results.n_treated == 1
        assert results.n_control == 2
        # 应返回有效的ATT和SE
        assert not np.isnan(results.att)
        assert not np.isnan(results.se_att)
    
    def test_B005_no_control_units(self, smoking_data):
        """B005: 无对照单位应抛出NoControlUnitsError"""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3],
            'year': [1, 2, 1, 2, 1, 2],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'd': [1, 1, 1, 1, 1, 1],  # 全为处理
            'post': [0, 1, 0, 1, 0, 1],
        })
        
        with pytest.raises(NoControlUnitsError):
            lwdid(data, 'y', 'd', 'id', 'year', 'post', 'demean')
    
    def test_B006_no_treated_units(self, smoking_data):
        """B006: 无处理单位应抛出NoTreatedUnitsError"""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3],
            'year': [1, 2, 1, 2, 1, 2],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'd': [0, 0, 0, 0, 0, 0],  # 全为对照
            'post': [0, 1, 0, 1, 0, 1],
        })
        
        with pytest.raises(NoTreatedUnitsError):
            lwdid(data, 'y', 'd', 'id', 'year', 'post', 'demean')
    
    def test_B008_string_id_conversion(self, smoking_data):
        """B008: 字符串ID应自动转换"""
        data = pd.DataFrame({
            'state': ['CA', 'CA', 'TX', 'TX', 'NY', 'NY'],
            'year': [1, 2, 1, 2, 1, 2],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'd': [1, 1, 0, 0, 0, 0],
            'post': [0, 1, 0, 1, 0, 1],
        })
        
        # 应正常运行（字符串ID自动转换）
        results = lwdid(data, 'y', 'd', 'state', 'year', 'post', 'demean')
        
        assert results.nobs == 3
        # 结果应有效
        assert not np.isnan(results.att)
    
    def test_B009_missing_values_dropped(self, smoking_data):
        """B009: 缺失值应被正确删除"""
        data = pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'year': [1, 2, 3, 1, 2, 3, 1, 2, 3],
            'y': [1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            'd': [1, 1, 1, 0, 0, 0, 0, 0, 0],
            'post': [0, 0, 1, 0, 0, 1, 0, 0, 1],
        })
        
        # 应发出警告并删除缺失行
        with pytest.warns(UserWarning, match="Dropped 1 observations"):
            results = lwdid(data, 'y', 'd', 'id', 'year', 'post', 'demean')
        
        # 删除后单位1只有2行（t=1和t=3），前期只有1个观测
        # 仍应能运行（T0=1对demean是允许的）
        assert not np.isnan(results.att)


class TestResultsObject:
    """结果对象测试"""
    
    def test_results_attributes(self, smoking_data):
        """验证LWDIDResults所有属性可访问"""
        data = pd.read_csv('tests/data/mve_demean.csv')
        results = lwdid(data, 'y', 'd', 'id', 'year', 'post', 'demean')
        
        # 标量属性
        assert hasattr(results, 'att')
        assert hasattr(results, 'se_att')
        assert hasattr(results, 't_stat')
        assert hasattr(results, 'pvalue')
        assert hasattr(results, 'ci_lower')
        assert hasattr(results, 'ci_upper')
        
        # 模型信息
        assert hasattr(results, 'K')
        assert hasattr(results, 'tpost1')
        assert hasattr(results, 'nobs')
        assert hasattr(results, 'n_treated')
        assert hasattr(results, 'n_control')
        assert hasattr(results, 'df_resid')
        
        # 方法元数据
        assert hasattr(results, 'cmd')
        assert hasattr(results, 'depvar')
        assert hasattr(results, 'rolling')
        assert hasattr(results, 'vce_type')
        
        # 矩阵属性
        assert hasattr(results, 'params')
        assert hasattr(results, 'bse')
        assert hasattr(results, 'vcov')
    
    def test_summary_output(self, smoking_data):
        """验证summary()方法输出"""
        data = pd.read_csv('tests/data/mve_demean.csv')
        results = lwdid(data, 'y', 'd', 'id', 'year', 'post', 'demean')
        
        summary = results.summary()
        
        # 应包含关键信息
        assert 'lwdid Results' in summary
        assert 'demean' in summary
        assert 'ATT:' in summary
        assert 'Std. Err.:' in summary
        assert str(results.nobs) in summary
    
    def test_repr_methods(self, smoking_data):
        """验证__repr__()和__str__()方法"""
        data = pd.read_csv('tests/data/mve_demean.csv')
        results = lwdid(data, 'y', 'd', 'id', 'year', 'post', 'demean')
        
        # __repr__() - 简洁表示
        repr_str = repr(results)
        assert 'LWDIDResults' in repr_str
        assert 'att=' in repr_str
        
        # __str__() - 详细表示（应调用summary()）
        str_str = str(results)
        assert 'lwdid Results' in str_str


class TestQuarterlyData:
    """季度数据测试（Story 1.3）"""
    
    def test_T014_smoking_quarterly_demeanq(self, smoking_data):
        """测试T014: smoking_quarterly + demeanq
        
        对应Story 1.3, Task 7.1, AC6
        验证：period格式"1989q1"，DataFrame 49行
        """
        from lwdid import lwdid
        
        # 加载季度数据
        data = pd.read_csv('tests/data/smoking_quarterly.csv')
        
        # 执行lwdid（注意tvar为列表）
        results = lwdid(
            data, y='lcigsale', d='d', ivar='state', 
            tvar=['year', 'quarter'], post='post',
            rolling='demeanq', vce=None
        )
        
        # 验证基本运行成功
        assert results.att is not None
        assert results.se_att is not None
        assert results.rolling == 'demeanq'
        
        # 验证att_by_period的period列格式（季度）
        # 第一个后期应为"1989q1"（小写q，无空格）
        assert results.att_by_period.iloc[1]['period'] == '1989q1', \
            f"期望'1989q1'，得到'{results.att_by_period.iloc[1]['period']}'"
        
        # 最后一期应为"2000q4"
        last_period = results.att_by_period.iloc[-1]['period']
        assert last_period == '2000q4', f"期望'2000q4'，得到'{last_period}'"
        
        # 验证DataFrame行数
        # 1 average + 12 years × 4 quarters = 1 + 48 = 49行
        assert len(results.att_by_period) == 49, \
            f"期望49行，得到{len(results.att_by_period)}行"
        
        # 验证ATT和SE（需Stata基准，或合理性检查）
        assert not np.isnan(results.att)
        assert results.se_att > 0
    
    def test_T015_smoking_quarterly_detrendq(self, smoking_data):
        """测试T015: smoking_quarterly + detrendq
        
        对应Story 1.3, Task 7.2, AC6
        """
        from lwdid import lwdid
        
        # 加载季度数据
        data = pd.read_csv('tests/data/smoking_quarterly.csv')
        
        # 执行lwdid
        results = lwdid(
            data, y='lcigsale', d='d', ivar='state',
            tvar=['year', 'quarter'], post='post',
            rolling='detrendq', vce=None
        )
        
        # 验证基本运行
        assert results.rolling == 'detrendq'
        assert results.att is not None
        
        # 验证period格式
        assert results.att_by_period.iloc[1]['period'] == '1989q1'
        assert len(results.att_by_period) == 49
        
        # 验证ATT和SE有合理值
        assert not np.isnan(results.att)
        assert results.se_att > 0


class TestControlVariables:
    """控制变量测试（Story 1.3）"""
    
    def test_T017_controls_included(self, smoking_data):
        """测试T017: 控制变量包含（N₁=10, N₀=10, K_ctrl=2）
        
        对应Story 1.3, Task 7.3, AC6
        """
        from lwdid import lwdid
        
        # 加载控制变量数据（large）
        data = pd.read_csv('tests/data/smoking_controls_large.csv')
        
        # 执行lwdid
        results = lwdid(
            data, y='lcigsale', d='d', ivar='state',
            tvar='year', post='post',
            rolling='detrend', vce=None,
            controls=['x1', 'x2']
        )
        
        # 验证：无警告消息（10 > K_ctrl+1=3，满足条件）
        # （警告在上面的调用中如果有会被pytest捕获）
        
        # 验证results.params包含控制变量系数
        # 设计矩阵：[const, d_, x1, x2, d_x1_c, d_x2_c]
        # params长度应为6
        assert len(results.params) == 6, \
            f"期望6个系数（const,d,x1,x2,d_x1_c,d_x2_c），得到{len(results.params)}"
        
        # ATT = results.params[1]（索引1始终是d_，无论是否有控制变量）
        assert results.att is not None
        assert not np.isnan(results.att)
        
        # 验证控制变量被使用
        # results对象应有controls属性或controls_used标志
        # （这取决于LWDIDResults类的实现，可能需要调整）
    
    def test_T018_controls_warning(self, smoking_data):
        """测试T018: 控制变量警告（N₁=3, N₀=3, K_ctrl=2）
        
        对应Story 1.3, Task 7.4, AC6
        """
        from lwdid import lwdid
        
        # 加载控制变量数据（small）
        data = pd.read_csv('tests/data/smoking_controls_small.csv')
        
        # 执行lwdid，应触发警告
        with pytest.warns(UserWarning, match="Controls not applied"):
            results = lwdid(
                data, y='lcigsale', d='d', ivar='state',
                tvar='year', post='post',
                rolling='detrend', vce=None,
                controls=['x1', 'x2']
            )
        
        # 验证：控制变量被忽略
        # params仅包含2个系数：const, d_
        assert len(results.params) == 2, \
            f"期望2个系数（const,d），得到{len(results.params)}"
        
        # ATT值应不同于T017（未调整控制变量）
        assert results.att is not None
        assert not np.isnan(results.att)


class TestBoundaryConditionsStory13:
    """边界测试（Story 1.3）"""
    
    def test_B007_invalid_quarter(self, smoking_data):
        """测试B007: quarter=5非法值

        对应Story 1.3, Task 3.3
        """
        from lwdid import lwdid
        from lwdid.exceptions import InvalidParameterError

        # 构造包含非法quarter值的数据
        data = pd.DataFrame({
            'id': [1,1,1, 2,2,2],
            'year': [1,1,2, 1,1,2],
            'quarter': [1,5,1, 1,2,1],  # quarter=5非法
            'y': [10.0, 12.0, 15.0, 5.0, 6.0, 7.0],
            'd': [1,1,1, 0,0,0],
            'post': [0,0,1, 0,0,1],
        })

        # 应该在validation阶段抛出InvalidParameterError（数据质量问题）
        with pytest.raises(InvalidParameterError,
                          match="Quarter variable 'quarter' contains invalid values"):
            lwdid(data, y='y', d='d', ivar='id',
                  tvar=['year', 'quarter'], post='post', rolling='demeanq')
    
    def test_B020_controls_boundary(self, smoking_data):
        """测试B020: 控制变量边界（N=K_ctrl+2）
        
        对应Story 1.3, Task 7.5
        """
        from lwdid import lwdid
        
        # 从large数据选择N₁=4, N₀=4的子集
        # K_ctrl=2，nk=3，条件4>3（恰好满足）
        data_large = pd.read_csv('tests/data/smoking_controls_large.csv')
        
        # 选择前4个处理州和前4个对照州
        treated_subset = data_large[data_large['d']==1]['state'].unique()[:4]
        control_subset = data_large[data_large['d']==0]['state'].unique()[:4]
        
        data_subset = data_large[
            data_large['state'].isin(list(treated_subset) + list(control_subset))
        ].copy()
        
        # 执行lwdid（应包含控制变量，无警告）
        results = lwdid(
            data_subset, y='lcigsale', d='d', ivar='state',
            tvar='year', post='post',
            rolling='detrend', vce=None,
            controls=['x1', 'x2']
        )
        
        # 验证：包含控制变量（无警告）
        assert len(results.params) == 6, "应包含控制变量（6个系数）"
        assert not np.isnan(results.att)

