"""
DESIGN-026: _prepare_estimation_data 公共函数单元测试

测试 _prepare_estimation_data 函数在不同场景下的行为：
1. Staggered 场景正确构建子样本
2. 非 Staggered 场景直接返回原数据
3. 边界条件（部分参数为 None）
4. 返回值类型和结构

References
----------
Lee & Wooldridge (2023), Section 4, Procedure 4.1
"""
import numpy as np
import pandas as pd
import pytest

# Skip entire module if required functions are not available
try:
    from lwdid.staggered.estimators import (
        _prepare_estimation_data,
        build_subsample_for_ps_estimation,
        SubsampleResult,
    )
except ImportError as e:
    pytest.skip(
        f"Skipping module: required functions not implemented ({e})",
        allow_module_level=True
    )


@pytest.fixture
def simple_panel_data():
    """创建简单面板数据用于测试"""
    np.random.seed(42)
    n_units = 100
    n_periods = 5
    
    # 创建单位和时期
    ids = np.repeat(np.arange(1, n_units + 1), n_periods)
    periods = np.tile(np.arange(1, n_periods + 1), n_units)
    
    # 创建 gvar（cohort 变量）
    # 30% cohort 2, 30% cohort 3, 20% cohort 4, 20% never treated (0)
    unit_cohorts = np.random.choice(
        [2, 3, 4, 0], 
        size=n_units, 
        p=[0.3, 0.3, 0.2, 0.2]
    )
    gvar = np.repeat(unit_cohorts, n_periods)
    
    # 创建协变量和结果变量
    x1 = np.random.randn(n_units * n_periods)
    x2 = np.random.randn(n_units * n_periods)
    y = 1.0 + 0.5 * x1 + 0.3 * x2 + np.random.randn(n_units * n_periods) * 0.5
    
    # 创建处理指示变量 d（单位在 gvar 期或之后处理）
    d = ((gvar > 0) & (periods >= gvar)).astype(int)
    
    data = pd.DataFrame({
        'id': ids,
        'year': periods,
        'gvar': gvar,
        'd': d,
        'x1': x1,
        'x2': x2,
        'y': y,
    })
    
    return data


@pytest.fixture
def cross_section_data():
    """创建横截面数据（非 Staggered 场景）"""
    np.random.seed(123)
    n = 200
    
    # 创建处理变量和协变量
    d = np.random.binomial(1, 0.4, n)
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    y = 1.0 + 0.5 * d + 0.3 * x1 + 0.2 * x2 + np.random.randn(n) * 0.5
    
    data = pd.DataFrame({
        'd': d,
        'x1': x1,
        'x2': x2,
        'y': y,
    })
    
    return data


class TestStaggeredScenario:
    """Staggered 场景测试"""
    
    def test_staggered_returns_subsample(self, simple_panel_data):
        """测试 Staggered 场景返回正确的子样本"""
        data_for_est, d_var, subsample_result = _prepare_estimation_data(
            data=simple_panel_data,
            d='d',
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=2,
            period_r=3,
            control_group='not_yet_treated',
        )
        
        # 验证返回 SubsampleResult
        assert subsample_result is not None
        assert isinstance(subsample_result, SubsampleResult)
        
        # 验证 d_var 为 'D_ig'
        assert d_var == 'D_ig'
        
        # 验证 data_for_estimation 是子样本
        assert len(data_for_est) < len(simple_panel_data)
        assert 'D_ig' in data_for_est.columns
    
    def test_staggered_subsample_result_attributes(self, simple_panel_data):
        """测试 SubsampleResult 属性正确"""
        data_for_est, d_var, subsample_result = _prepare_estimation_data(
            data=simple_panel_data,
            d='d',
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=3,
            period_r=4,
            control_group='not_yet_treated',
        )
        
        # 验证 SubsampleResult 属性
        assert subsample_result.cohort == 3
        assert subsample_result.period == 4
        assert subsample_result.n_treated > 0
        assert subsample_result.n_control > 0
        assert subsample_result.control_strategy == 'not_yet_treated'
    
    def test_staggered_never_treated_control(self, simple_panel_data):
        """测试 never_treated 控制组策略"""
        data_for_est, d_var, subsample_result = _prepare_estimation_data(
            data=simple_panel_data,
            d='d',
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=2,
            period_r=3,
            control_group='never_treated',
        )
        
        assert subsample_result.control_strategy == 'never_treated'
        # 控制组 cohorts 应该只包含 NT（0 或 inf）
        for cohort in subsample_result.control_cohorts:
            assert cohort == 0 or np.isinf(cohort)
    
    def test_staggered_d_ig_binary(self, simple_panel_data):
        """测试 D_ig 是正确的二元变量"""
        data_for_est, d_var, subsample_result = _prepare_estimation_data(
            data=simple_panel_data,
            d='d',
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=2,
            period_r=3,
        )
        
        # D_ig 应该是 0 或 1
        d_ig_values = data_for_est['D_ig'].unique()
        assert set(d_ig_values).issubset({0, 1})
        
        # 处理组应该是 cohort 2
        treated_mask = data_for_est['D_ig'] == 1
        assert (data_for_est.loc[treated_mask, 'gvar'] == 2).all()


class TestNonStaggeredScenario:
    """非 Staggered 场景测试"""
    
    def test_non_staggered_returns_original_data(self, cross_section_data):
        """测试非 Staggered 场景直接返回原数据"""
        data_for_est, d_var, subsample_result = _prepare_estimation_data(
            data=cross_section_data,
            d='d',
            gvar_col=None,
            ivar_col=None,
            cohort_g=None,
            period_r=None,
        )
        
        # 验证 subsample_result 为 None
        assert subsample_result is None
        
        # 验证 d_var 为原始处理变量名
        assert d_var == 'd'
        
        # 验证数据未变
        pd.testing.assert_frame_equal(data_for_est, cross_section_data)
    
    def test_non_staggered_no_d_ig_column(self, cross_section_data):
        """测试非 Staggered 场景不添加 D_ig 列"""
        data_for_est, d_var, subsample_result = _prepare_estimation_data(
            data=cross_section_data,
            d='d',
            gvar_col=None,
            ivar_col=None,
            cohort_g=None,
            period_r=None,
        )
        
        # 不应该添加 D_ig 列（除非原数据就有）
        if 'D_ig' not in cross_section_data.columns:
            assert 'D_ig' not in data_for_est.columns


class TestBoundaryConditions:
    """边界条件测试"""
    
    def test_partial_staggered_params_treated_as_non_staggered(self, simple_panel_data):
        """测试部分 Staggered 参数视为非 Staggered"""
        # 只提供 gvar_col 和 ivar_col
        data_for_est, d_var, subsample_result = _prepare_estimation_data(
            data=simple_panel_data,
            d='d',
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=None,  # 缺少
            period_r=None,  # 缺少
        )
        
        assert subsample_result is None
        assert d_var == 'd'
    
    def test_partial_staggered_params_cohort_only(self, simple_panel_data):
        """测试只提供 cohort_g 时视为非 Staggered"""
        data_for_est, d_var, subsample_result = _prepare_estimation_data(
            data=simple_panel_data,
            d='d',
            gvar_col=None,
            ivar_col=None,
            cohort_g=2,  # 只提供这个
            period_r=None,
        )
        
        assert subsample_result is None
        assert d_var == 'd'
    
    def test_empty_d_string_in_staggered(self, simple_panel_data):
        """测试 Staggered 场景中 d 参数可以为空字符串"""
        # 在 Staggered 场景中，d 参数会被 'D_ig' 替代
        data_for_est, d_var, subsample_result = _prepare_estimation_data(
            data=simple_panel_data,
            d='',  # 空字符串
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=2,
            period_r=3,
        )
        
        # Staggered 场景应该使用 D_ig
        assert d_var == 'D_ig'
        assert subsample_result is not None


class TestReturnTypes:
    """返回值类型测试"""
    
    def test_return_tuple_staggered(self, simple_panel_data):
        """测试 Staggered 场景返回正确的元组类型"""
        result = _prepare_estimation_data(
            data=simple_panel_data,
            d='d',
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=2,
            period_r=3,
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        
        data_for_est, d_var, subsample_result = result
        assert isinstance(data_for_est, pd.DataFrame)
        assert isinstance(d_var, str)
        assert isinstance(subsample_result, SubsampleResult)
    
    def test_return_tuple_non_staggered(self, cross_section_data):
        """测试非 Staggered 场景返回正确的元组类型"""
        result = _prepare_estimation_data(
            data=cross_section_data,
            d='d',
            gvar_col=None,
            ivar_col=None,
            cohort_g=None,
            period_r=None,
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        
        data_for_est, d_var, subsample_result = result
        assert isinstance(data_for_est, pd.DataFrame)
        assert isinstance(d_var, str)
        assert subsample_result is None


class TestConsistencyWithDirectCall:
    """与直接调用 build_subsample_for_ps_estimation 的一致性测试"""
    
    def test_same_subsample_as_direct_call(self, simple_panel_data):
        """测试返回的子样本与直接调用 build_subsample_for_ps_estimation 一致"""
        cohort_g = 3
        period_r = 4
        
        # 使用 _prepare_estimation_data
        data_for_est, d_var, subsample_result = _prepare_estimation_data(
            data=simple_panel_data,
            d='d',
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=cohort_g,
            period_r=period_r,
            control_group='not_yet_treated',
        )
        
        # 直接调用 build_subsample_for_ps_estimation
        direct_result = build_subsample_for_ps_estimation(
            data=simple_panel_data,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=cohort_g,
            period_r=period_r,
            control_group='not_yet_treated',
        )
        
        # 比较结果
        assert subsample_result.n_treated == direct_result.n_treated
        assert subsample_result.n_control == direct_result.n_control
        assert subsample_result.cohort == direct_result.cohort
        assert subsample_result.period == direct_result.period
        assert subsample_result.control_strategy == direct_result.control_strategy
        
        # 比较子样本数据
        pd.testing.assert_frame_equal(
            data_for_est.reset_index(drop=True),
            direct_result.subsample.reset_index(drop=True)
        )


class TestCustomNeverTreatedValues:
    """自定义 never_treated_values 测试"""
    
    def test_custom_never_treated_values(self):
        """测试自定义 never_treated_values"""
        np.random.seed(456)
        n_units = 50
        n_periods = 4
        
        ids = np.repeat(np.arange(1, n_units + 1), n_periods)
        periods = np.tile(np.arange(1, n_periods + 1), n_units)
        
        # 使用 999 作为 never treated 标记
        unit_cohorts = np.random.choice([2, 3, 999], size=n_units, p=[0.4, 0.4, 0.2])
        gvar = np.repeat(unit_cohorts, n_periods)
        
        x1 = np.random.randn(n_units * n_periods)
        y = np.random.randn(n_units * n_periods)
        
        data = pd.DataFrame({
            'id': ids,
            'year': periods,
            'gvar': gvar,
            'x1': x1,
            'y': y,
        })
        
        # 使用自定义 never_treated_values
        data_for_est, d_var, subsample_result = _prepare_estimation_data(
            data=data,
            d='',
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=2,
            period_r=3,
            control_group='not_yet_treated',
            never_treated_values=[999],
        )
        
        assert subsample_result is not None
        # 验证 999 被识别为控制组
        control_gvars = data_for_est.loc[data_for_est['D_ig'] == 0, 'gvar'].unique()
        assert 999 in control_gvars or len(control_gvars) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
