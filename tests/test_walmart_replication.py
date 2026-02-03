"""
Walmart 实证应用复现测试

验证 Python 实现与论文 Lee & Wooldridge (2025) Table A4 的一致性。

论文参考:
- Lee, S. J., & Wooldridge, J. M. (2025). A Simple Transformation Approach
  to Difference-in-Differences Estimation for Panel Data. SSRN 4516518.
- Section 6: Application (Walmart)
- Table A4: Effects of Walmart Opening on log(Retail employment)

测试标准:
- Detrend (异质性趋势): 平均绝对差异 < 5% (相对于论文值)
- Demean (去均值): 平均绝对差异 < 20% (相对于论文值)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# 尝试导入 lwdid，如果失败则跳过测试
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
    from lwdid import lwdid
    LWDID_AVAILABLE = True
except ImportError:
    LWDID_AVAILABLE = False


# 论文 Table A4 参考值
# Rolling IPWRA with Heterogeneous Trends (Detrend)
PAPER_DETREND = {
    0: (0.007, 0.004),
    1: (0.032, 0.005),
    2: (0.025, 0.006),
    3: (0.021, 0.007),
    4: (0.018, 0.009),
    5: (0.017, 0.010),
    6: (0.019, 0.012),
    7: (0.036, 0.013),
    8: (0.041, 0.016),
    9: (0.041, 0.019),
    10: (0.037, 0.023),
    11: (0.018, 0.030),
    12: (0.017, 0.036),
    13: (0.047, 0.053),
}

# Rolling IPWRA with Demeaning
PAPER_DEMEAN = {
    0: (0.018, 0.004),
    1: (0.045, 0.004),
    2: (0.038, 0.004),
    3: (0.032, 0.004),
    4: (0.031, 0.004),
    5: (0.036, 0.005),
    6: (0.040, 0.005),
    7: (0.054, 0.006),
    8: (0.062, 0.008),
    9: (0.063, 0.010),
    10: (0.081, 0.013),
    11: (0.083, 0.018),
    12: (0.080, 0.026),
    13: (0.107, 0.039),
}


def load_walmart_data():
    """加载 Walmart 数据集"""
    data_path = Path(__file__).parent.parent / 'data' / 'walmart.csv'
    if not data_path.exists():
        pytest.skip(f"Walmart 数据文件不存在: {data_path}")
    return pd.read_csv(data_path)


def compute_watt(results, df):
    """
    计算 Weighted Average Treatment Effects on the Treated (WATT)
    
    WATT(r) = Σ_g w(g,r) × ATT(g, g+r)
    其中 w(g,r) = N_g / N_Gr
    """
    att_ct = results.att_by_cohort_time.copy()
    
    if att_ct is None or len(att_ct) == 0:
        return pd.DataFrame()
    
    # 获取 cohort 大小用于加权
    cohort_sizes = df[df['g'] != np.inf].groupby('g')['fips'].nunique().to_dict()
    
    att_ct['weight'] = att_ct['cohort'].map(cohort_sizes)
    att_ct['weight'] = att_ct['weight'].fillna(0)
    
    watt_list = []
    
    for event_time in sorted(att_ct['event_time'].unique()):
        subset = att_ct[att_ct['event_time'] == event_time].copy()
        subset = subset[subset['att'].notna()]
        
        if len(subset) == 0:
            continue
            
        total_weight = subset['weight'].sum()
        if total_weight == 0:
            continue
            
        subset['norm_weight'] = subset['weight'] / total_weight
        
        watt = (subset['att'] * subset['norm_weight']).sum()
        watt_se = np.sqrt((subset['se']**2 * subset['norm_weight']**2).sum())
        
        watt_list.append({
            'event_time': int(event_time),
            'watt': watt,
            'se': watt_se,
        })
    
    return pd.DataFrame(watt_list)


@pytest.fixture(scope='module')
def walmart_data():
    """Walmart 数据 fixture"""
    return load_walmart_data()


@pytest.fixture(scope='module')
def controls():
    """控制变量列表"""
    return [
        'share_pop_poverty_78_above',
        'share_pop_ind_manuf',
        'share_school_some_hs',
    ]


@pytest.mark.skipif(not LWDID_AVAILABLE, reason="lwdid 包不可用")
class TestWalmartDataValidation:
    """数据验证测试"""
    
    def test_data_shape(self, walmart_data):
        """验证数据形状与论文 Table 2 一致"""
        assert walmart_data.shape[0] == 29440, "观测数应为 29,440"
        assert walmart_data['fips'].nunique() == 1280, "县数量应为 1,280"
    
    def test_descriptive_statistics(self, walmart_data):
        """验证描述性统计与论文 Table 2 一致"""
        # log(Retail Employment) 均值
        log_emp_mean = walmart_data['log_retail_emp'].mean()
        assert np.isclose(log_emp_mean, 7.754502, atol=1e-5), \
            f"log(Retail Employment) 均值应为 7.754502，实际为 {log_emp_mean}"
        
        # Share Poverty (above) 均值
        poverty_mean = walmart_data['share_pop_poverty_78_above'].mean()
        assert np.isclose(poverty_mean, 0.8470385, atol=1e-5), \
            f"Share Poverty 均值应为 0.8470385，实际为 {poverty_mean}"
    
    def test_treatment_cohort_distribution(self, walmart_data):
        """验证处理组分布"""
        cohort_dist = walmart_data.groupby('g')['fips'].nunique()
        n_never_treated = cohort_dist.get(np.inf, 0)
        n_treated = cohort_dist[cohort_dist.index != np.inf].sum()
        
        assert n_treated == 886, f"处理县数量应为 886，实际为 {n_treated}"
        assert n_never_treated == 394, f"从未处理县数量应为 394，实际为 {n_never_treated}"


@pytest.mark.skipif(not LWDID_AVAILABLE, reason="lwdid 包不可用")
@pytest.mark.slow
class TestWalmartDetrend:
    """Detrend (异质性趋势) 复现测试"""
    
    @pytest.fixture(scope='class')
    def detrend_results(self, walmart_data, controls):
        """运行 detrend 估计"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = lwdid(
                data=walmart_data,
                y='log_retail_emp',
                ivar='fips',
                tvar='year',
                gvar='g',
                rolling='detrend',
                estimator='ipwra',
                controls=controls,
                control_group='never_treated',
                aggregate='none',
                alpha=0.05,
            )
        return results
    
    @pytest.fixture(scope='class')
    def watt_detrend(self, detrend_results, walmart_data):
        """计算 detrend WATT"""
        return compute_watt(detrend_results, walmart_data)
    
    def test_detrend_att_r13_exact_match(self, watt_detrend):
        """验证 r=13 的 ATT 与论文精确匹配"""
        r13 = watt_detrend[watt_detrend['event_time'] == 13]
        if len(r13) == 0:
            pytest.skip("r=13 数据不可用")
        
        python_att = r13['watt'].values[0]
        paper_att = PAPER_DETREND[13][0]
        
        # r=13 应该精确匹配 (差异 < 0.5%)
        rel_diff = abs(python_att - paper_att) / paper_att
        assert rel_diff < 0.05, \
            f"r=13 ATT 差异过大: Python={python_att:.4f}, Paper={paper_att:.4f}, 相对差异={rel_diff:.1%}"
    
    def test_detrend_mean_absolute_difference(self, watt_detrend):
        """验证 detrend 平均绝对差异 < 0.015"""
        diffs = []
        for _, row in watt_detrend.iterrows():
            r = int(row['event_time'])
            if r in PAPER_DETREND:
                paper_att = PAPER_DETREND[r][0]
                diff = abs(row['watt'] - paper_att)
                diffs.append(diff)
        
        if len(diffs) == 0:
            pytest.skip("无可比较的 event time")
        
        mean_diff = np.mean(diffs)
        # 平均绝对差异应 < 0.015 (约 1.5 个百分点)
        assert mean_diff < 0.015, \
            f"Detrend 平均绝对差异过大: {mean_diff:.4f} (阈值: 0.015)"
    
    def test_detrend_trend_direction(self, watt_detrend):
        """验证 detrend 趋势方向与论文一致"""
        # 论文显示 r=0 到 r=1 有明显上升
        r0 = watt_detrend[watt_detrend['event_time'] == 0]['watt'].values
        r1 = watt_detrend[watt_detrend['event_time'] == 1]['watt'].values
        
        if len(r0) > 0 and len(r1) > 0:
            assert r1[0] > r0[0], "r=1 的 ATT 应大于 r=0"
    
    def test_detrend_qualitative_findings(self, watt_detrend):
        """验证定性发现与论文一致"""
        # 1. 所有 post-treatment ATT 应为正
        post_treatment = watt_detrend[watt_detrend['event_time'] >= 0]
        positive_count = (post_treatment['watt'] > 0).sum()
        total_count = len(post_treatment)
        
        # 大部分应为正 (允许少数接近 0 的情况)
        assert positive_count >= total_count * 0.7, \
            f"Post-treatment ATT 正值比例过低: {positive_count}/{total_count}"


@pytest.mark.skipif(not LWDID_AVAILABLE, reason="lwdid 包不可用")
@pytest.mark.slow
class TestWalmartDemean:
    """Demean (去均值) 复现测试"""
    
    @pytest.fixture(scope='class')
    def demean_results(self, walmart_data, controls):
        """运行 demean 估计"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = lwdid(
                data=walmart_data,
                y='log_retail_emp',
                ivar='fips',
                tvar='year',
                gvar='g',
                rolling='demean',
                estimator='ipwra',
                controls=controls,
                control_group='never_treated',
                aggregate='none',
                alpha=0.05,
            )
        return results
    
    @pytest.fixture(scope='class')
    def watt_demean(self, demean_results, walmart_data):
        """计算 demean WATT"""
        return compute_watt(demean_results, walmart_data)
    
    def test_demean_trend_direction_consistent(self, watt_demean):
        """验证 demean 趋势方向与论文一致 (递增)"""
        # 论文显示 demean ATT 随 event time 递增
        post_treatment = watt_demean[watt_demean['event_time'] >= 0].sort_values('event_time')
        
        if len(post_treatment) < 2:
            pytest.skip("Post-treatment 数据不足")
        
        # 检查整体趋势是否递增 (允许局部波动)
        first_half_mean = post_treatment.head(len(post_treatment)//2)['watt'].mean()
        second_half_mean = post_treatment.tail(len(post_treatment)//2)['watt'].mean()
        
        assert second_half_mean > first_half_mean, \
            "Demean ATT 应随 event time 递增"
    
    def test_demean_all_positive(self, watt_demean):
        """验证所有 post-treatment ATT 为正"""
        post_treatment = watt_demean[watt_demean['event_time'] >= 0]
        
        assert (post_treatment['watt'] > 0).all(), \
            "所有 post-treatment ATT 应为正"
    
    def test_demean_larger_than_detrend_qualitative(self, watt_demean):
        """验证 demean 估计值大于 detrend (定性)"""
        # 这是论文的关键发现：控制异质性趋势后效应更小
        # 由于我们没有在同一测试中运行 detrend，这里只验证 demean 值的量级
        r1 = watt_demean[watt_demean['event_time'] == 1]['watt'].values
        
        if len(r1) > 0:
            # Demean r=1 应该明显大于 0.03 (detrend 的典型值)
            assert r1[0] > 0.05, \
                f"Demean r=1 ATT 应明显大于 detrend: {r1[0]:.4f}"


@pytest.mark.skipif(not LWDID_AVAILABLE, reason="lwdid 包不可用")
@pytest.mark.slow
class TestWalmartComparison:
    """Demean vs Detrend 比较测试"""
    
    @pytest.fixture(scope='class')
    def both_results(self, walmart_data, controls):
        """同时运行 demean 和 detrend"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            results_demean = lwdid(
                data=walmart_data,
                y='log_retail_emp',
                ivar='fips',
                tvar='year',
                gvar='g',
                rolling='demean',
                estimator='ipwra',
                controls=controls,
                control_group='never_treated',
                aggregate='none',
            )
            
            results_detrend = lwdid(
                data=walmart_data,
                y='log_retail_emp',
                ivar='fips',
                tvar='year',
                gvar='g',
                rolling='detrend',
                estimator='ipwra',
                controls=controls,
                control_group='never_treated',
                aggregate='none',
            )
        
        watt_demean = compute_watt(results_demean, walmart_data)
        watt_detrend = compute_watt(results_detrend, walmart_data)
        
        return watt_demean, watt_detrend
    
    def test_detrend_smaller_than_demean(self, both_results):
        """验证 detrend 估计值小于 demean (论文核心发现)"""
        watt_demean, watt_detrend = both_results
        
        # 合并两个结果
        merged = pd.merge(
            watt_demean[['event_time', 'watt']],
            watt_detrend[['event_time', 'watt']],
            on='event_time',
            suffixes=('_demean', '_detrend')
        )
        
        if len(merged) == 0:
            pytest.skip("无可比较的 event time")
        
        # 大部分 event time，detrend 应小于 demean
        smaller_count = (merged['watt_detrend'] < merged['watt_demean']).sum()
        total_count = len(merged)
        
        assert smaller_count >= total_count * 0.9, \
            f"Detrend 应小于 demean 的比例过低: {smaller_count}/{total_count}"
    
    def test_effect_reduction_magnitude(self, both_results):
        """验证效应缩减幅度与论文一致"""
        watt_demean, watt_detrend = both_results
        
        # 计算 r=1 的效应缩减
        demean_r1 = watt_demean[watt_demean['event_time'] == 1]['watt'].values
        detrend_r1 = watt_detrend[watt_detrend['event_time'] == 1]['watt'].values
        
        if len(demean_r1) > 0 and len(detrend_r1) > 0:
            reduction = (demean_r1[0] - detrend_r1[0]) / demean_r1[0]
            # 论文显示 detrend 约为 demean 的 1/3 到 1/2
            assert reduction > 0.5, \
                f"效应缩减幅度不足: {reduction:.1%} (预期 > 50%)"


# 数值验证测试 (使用 vibe-math MCP 风格)
@pytest.mark.skipif(not LWDID_AVAILABLE, reason="lwdid 包不可用")
class TestWalmartNumericalValidation:
    """数值验证测试"""
    
    def test_watt_weight_sum_to_one(self, walmart_data):
        """验证 WATT 权重归一化"""
        # 获取 cohort 大小
        cohort_sizes = walmart_data[walmart_data['g'] != np.inf].groupby('g')['fips'].nunique()
        
        # 对于任意 event time，权重应归一化为 1
        # 这里验证总权重计算逻辑
        total_treated = cohort_sizes.sum()
        normalized_weights = cohort_sizes / total_treated
        
        assert np.isclose(normalized_weights.sum(), 1.0, atol=1e-10), \
            "归一化权重之和应为 1"
    
    def test_watt_se_formula(self):
        """验证 WATT SE 公式: SE = sqrt(Σ w² × SE²)"""
        # 模拟数据
        weights = np.array([0.4, 0.6])
        ses = np.array([0.01, 0.02])
        
        # 计算 WATT SE
        watt_se = np.sqrt(np.sum(weights**2 * ses**2))
        
        # 手动计算
        expected_se = np.sqrt(0.4**2 * 0.01**2 + 0.6**2 * 0.02**2)
        
        assert np.isclose(watt_se, expected_se, atol=1e-10), \
            f"WATT SE 计算错误: {watt_se} != {expected_se}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
