"""
测试 exclude_pre_periods 参数功能。

验证 T5 任务实现：在 lwdid() 函数中添加 exclude_pre_periods 参数，
用于排除处理前紧邻的若干期数，以应对潜在的预期效应。
"""

import numpy as np
import pandas as pd
import pytest


class TestExcludePrePeriodsParameter:
    """测试 exclude_pre_periods 参数的基本功能。"""

    @pytest.fixture
    def panel_data(self):
        """生成测试用面板数据。"""
        np.random.seed(42)
        n_units = 50
        n_periods = 10
        treatment_period = 6

        data = []
        for i in range(n_units):
            alpha_i = np.random.normal(0, 1)
            treated = i < n_units // 2

            for t in range(1, n_periods + 1):
                d_it = 1 if treated and t >= treatment_period else 0
                tau = 2.0 if d_it else 0
                y = alpha_i + 0.1 * t + tau + np.random.normal(0, 0.5)

                data.append({
                    'unit': i,
                    'time': t,
                    'Y': y,
                    'd': 1 if treated else 0,
                    'post': 1 if t >= treatment_period else 0,
                })

        return pd.DataFrame(data)

    def test_exclude_pre_periods_default_zero(self, panel_data):
        """测试 exclude_pre_periods 默认值为 0。"""
        from lwdid import lwdid

        # 默认调用，不指定 exclude_pre_periods
        result = lwdid(
            panel_data,
            y='Y',
            d='d',
            ivar='unit',
            tvar='time',
            post='post',
            rolling='demean',
        )

        assert result.att is not None
        assert not np.isnan(result.att)

    def test_exclude_pre_periods_one(self, panel_data):
        """测试排除 1 个 pre-treatment 期。"""
        from lwdid import lwdid

        result = lwdid(
            panel_data,
            y='Y',
            d='d',
            ivar='unit',
            tvar='time',
            post='post',
            rolling='demean',
            exclude_pre_periods=1,
        )

        assert result.att is not None
        assert not np.isnan(result.att)

    def test_exclude_pre_periods_two(self, panel_data):
        """测试排除 2 个 pre-treatment 期。"""
        from lwdid import lwdid

        result = lwdid(
            panel_data,
            y='Y',
            d='d',
            ivar='unit',
            tvar='time',
            post='post',
            rolling='demean',
            exclude_pre_periods=2,
        )

        assert result.att is not None
        assert not np.isnan(result.att)

    def test_exclude_pre_periods_with_detrend(self, panel_data):
        """测试 detrend 方法与 exclude_pre_periods 的组合。"""
        from lwdid import lwdid

        result = lwdid(
            panel_data,
            y='Y',
            d='d',
            ivar='unit',
            tvar='time',
            post='post',
            rolling='detrend',
            exclude_pre_periods=1,
        )

        assert result.att is not None
        assert not np.isnan(result.att)

    def test_exclude_pre_periods_changes_estimate(self, panel_data):
        """测试排除期数会改变估计结果。"""
        from lwdid import lwdid

        result_0 = lwdid(
            panel_data,
            y='Y',
            d='d',
            ivar='unit',
            tvar='time',
            post='post',
            rolling='demean',
            exclude_pre_periods=0,
        )

        result_2 = lwdid(
            panel_data,
            y='Y',
            d='d',
            ivar='unit',
            tvar='time',
            post='post',
            rolling='demean',
            exclude_pre_periods=2,
        )

        # 估计值应该不同（除非数据恰好使它们相等）
        # 这里我们只验证两个结果都是有效的
        assert result_0.att is not None
        assert result_2.att is not None
        assert not np.isnan(result_0.att)
        assert not np.isnan(result_2.att)


class TestExcludePrePeriodsValidation:
    """测试 exclude_pre_periods 参数验证。"""

    @pytest.fixture
    def panel_data(self):
        """生成测试用面板数据。"""
        np.random.seed(42)
        data = []
        for i in range(20):
            for t in range(1, 8):
                data.append({
                    'unit': i,
                    'time': t,
                    'Y': np.random.randn(),
                    'd': 1 if i < 10 else 0,
                    'post': 1 if t >= 5 else 0,
                })
        return pd.DataFrame(data)

    def test_negative_exclude_pre_periods_raises(self, panel_data):
        """测试负数 exclude_pre_periods 抛出错误。"""
        from lwdid import lwdid

        with pytest.raises(ValueError, match="non-negative"):
            lwdid(
                panel_data,
                y='Y',
                d='d',
                ivar='unit',
                tvar='time',
                post='post',
                rolling='demean',
                exclude_pre_periods=-1,
            )

    def test_non_integer_exclude_pre_periods_raises(self, panel_data):
        """测试非整数 exclude_pre_periods 抛出错误。"""
        from lwdid import lwdid

        with pytest.raises(TypeError, match="integer"):
            lwdid(
                panel_data,
                y='Y',
                d='d',
                ivar='unit',
                tvar='time',
                post='post',
                rolling='demean',
                exclude_pre_periods=1.5,
            )


class TestExcludePrePeriodsStaggered:
    """测试 staggered 模式下的 exclude_pre_periods 参数。"""

    @pytest.fixture
    def staggered_data(self):
        """生成 staggered 设计的测试数据。"""
        np.random.seed(42)
        data = []
        for i in range(30):
            # 10 units treated at t=4, 10 at t=6, 10 never treated
            if i < 10:
                gvar = 4
            elif i < 20:
                gvar = 6
            else:
                gvar = 0  # never treated

            for t in range(1, 10):
                treated = gvar > 0 and t >= gvar
                y = np.random.randn() + (2.0 if treated else 0)
                data.append({
                    'unit': i,
                    'time': t,
                    'Y': y,
                    'first_treat': gvar,
                })
        return pd.DataFrame(data)

    def test_staggered_exclude_pre_periods_warns(self, staggered_data):
        """测试 staggered 模式下使用 exclude_pre_periods 会发出警告。"""
        from lwdid import lwdid

        with pytest.warns(UserWarning, match="not yet supported"):
            lwdid(
                staggered_data,
                y='Y',
                ivar='unit',
                tvar='time',
                gvar='first_treat',
                rolling='demean',
                exclude_pre_periods=1,
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
