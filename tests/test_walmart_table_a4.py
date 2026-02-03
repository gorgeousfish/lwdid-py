"""
Walmart Table A4 复现测试：验证 Rolling IPWRA 估计与论文 Table A4 的匹配

基准来源：Lee & Wooldridge (2023v2) Table A4: Effects of Walmart Opening on log(Retail employment)
- paper/paper_v2/2023v2.md 第 971-988 行

Table A4 包含两列 Rolling IPWRA 结果：
1. Rolling IPWRA (demean)：无异质趋势，unit-specific demeaning
2. Rolling IPWRA (heterogeneous trend/detrend)：有异质线性趋势，unit-specific detrending

论文 SE 使用 bootstrap 100 次重复

验收标准：
- Detrend 列：r=0-9 相对误差 < 20%，r=10-13 放宽
- Demean 列：点估计与论文一致（关键：与论文一致的控制组口径为 all_others）
"""

import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
warnings.filterwarnings('ignore')


# ============================================================================
# Table A4 基准值：从 paper/paper_v2/2023v2.md 第 971-988 行提取
# ============================================================================

# Rolling IPWRA (demean, 无异质趋势) - Table A4 第 5-6 列
PAPER_WATT_IPWRA_DEMEAN = {
    0: {'att': 0.018, 'se': 0.004},
    1: {'att': 0.045, 'se': 0.004},
    2: {'att': 0.038, 'se': 0.004},
    3: {'att': 0.032, 'se': 0.004},
    4: {'att': 0.031, 'se': 0.004},
    5: {'att': 0.036, 'se': 0.005},
    6: {'att': 0.040, 'se': 0.005},
    7: {'att': 0.054, 'se': 0.006},
    8: {'att': 0.062, 'se': 0.008},
    9: {'att': 0.063, 'se': 0.010},
    10: {'att': 0.081, 'se': 0.013},
    11: {'att': 0.083, 'se': 0.018},
    12: {'att': 0.080, 'se': 0.026},
    13: {'att': 0.107, 'se': 0.039},
}

# Rolling IPWRA (heterogeneous trend/detrend) - Table A4 第 9-10 列
PAPER_WATT_IPWRA_DETREND = {
    0: {'att': 0.007, 'se': 0.004},
    1: {'att': 0.032, 'se': 0.005},
    2: {'att': 0.025, 'se': 0.006},
    3: {'att': 0.021, 'se': 0.007},
    4: {'att': 0.018, 'se': 0.009},
    5: {'att': 0.017, 'se': 0.010},
    6: {'att': 0.019, 'se': 0.012},
    7: {'att': 0.036, 'se': 0.013},
    8: {'att': 0.041, 'se': 0.016},
    9: {'att': 0.041, 'se': 0.019},
    10: {'att': 0.037, 'se': 0.023},
    11: {'att': 0.018, 'se': 0.030},
    12: {'att': 0.017, 'se': 0.036},
    13: {'att': 0.047, 'se': 0.053},
}


# ============================================================================
# 辅助函数
# ============================================================================

def load_walmart_data():
    """加载 Walmart 数据"""
    data_path = Path(__file__).parent.parent / 'data' / 'walmart.csv'
    return pd.read_csv(data_path)


def compute_watt_from_results(results, df):
    """
    从 lwdid 结果计算 WATT（与论文 Section 6.2 公式一致）
    
    WATT(r) = Σ_g w(g,r) × ATT(g, g+r)
    w(g,r) = N_g / N_Gr (cohort g 的处理单元数 / 所有贡献 r 期 cohort 的处理单元总数)
    """
    att_ct = results.att_by_cohort_time.copy()
    
    if att_ct is None or len(att_ct) == 0:
        return {}
    
    # 获取每个 cohort 的单元数
    cohort_sizes = df[df['g'] != np.inf].groupby('g')['fips'].nunique().to_dict()
    
    att_ct['weight'] = att_ct['cohort'].map(cohort_sizes)
    att_ct['weight'] = att_ct['weight'].fillna(0)
    
    watt_dict = {}
    
    for event_time in sorted(att_ct['event_time'].unique()):
        if event_time < 0:
            continue
            
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
        
        watt_dict[int(event_time)] = {'watt': watt, 'se': watt_se}
    
    return watt_dict


def estimate_and_compute_watt(df, rolling_method, controls, control_group='not_yet_treated'):
    """运行 lwdid 估计并计算 WATT"""
    from lwdid import lwdid
    
    results = lwdid(
        data=df,
        y='log_retail_emp',
        ivar='fips',
        tvar='year',
        gvar='g',
        rolling=rolling_method,
        estimator='ipwra',
        controls=controls,
        control_group=control_group,
        aggregate='none',
        alpha=0.05,
    )
    
    return compute_watt_from_results(results, df)


# ============================================================================
# 测试类
# ============================================================================

class TestWalmartTableA4Benchmarks:
    """验证 Rolling IPWRA 估计与论文 Table A4 的匹配"""
    
    @pytest.fixture(scope='class')
    def walmart_data(self):
        """加载数据"""
        return load_walmart_data()
    
    @pytest.fixture(scope='class')
    def controls(self):
        """控制变量（与论文 Table 2 一致）"""
        return [
            'share_pop_poverty_78_above',
            'share_pop_ind_manuf',
            'share_school_some_hs',
        ]
    
    @pytest.fixture(scope='class')
    def watt_detrend(self, walmart_data, controls):
        """计算 Detrend WATT"""
        return estimate_and_compute_watt(
            walmart_data, 'detrend', controls, control_group='not_yet_treated'
        )
    
    @pytest.fixture(scope='class')
    def watt_demean(self, walmart_data, controls):
        """计算 Demean WATT"""
        return estimate_and_compute_watt(
            walmart_data, 'demean', controls, control_group='all_others'
        )
    
    # ========================================================================
    # Detrend 测试（预期精确匹配）
    # ========================================================================
    
    @pytest.mark.parametrize("r", range(10))
    def test_detrend_event_time_0_to_9(self, watt_detrend, r):
        """验证 Detrend r=0-9 与论文匹配（严格：20% 容差）"""
        if r not in watt_detrend:
            pytest.skip(f"Event time {r} not in results")
        
        python_val = watt_detrend[r]['watt']
        paper_val = PAPER_WATT_IPWRA_DETREND[r]['att']
        
        if abs(paper_val) < 0.001:
            # 论文值接近零，使用绝对差异
            assert abs(python_val - paper_val) < 0.01, \
                f"Detrend r={r}: Python={python_val:.4f}, Paper={paper_val:.4f}"
        else:
            ratio = python_val / paper_val
            assert 0.8 <= ratio <= 1.2, \
                f"Detrend r={r}: Python={python_val:.4f}, Paper={paper_val:.4f}, ratio={ratio:.2f}"
    
    @pytest.mark.parametrize("r", [10, 11, 12, 13])
    def test_detrend_event_time_10_to_13(self, watt_detrend, r):
        """验证 Detrend r=10-13 与论文匹配（宽松：50% 容差）"""
        if r not in watt_detrend:
            pytest.skip(f"Event time {r} not in results")
        
        python_val = watt_detrend[r]['watt']
        paper_val = PAPER_WATT_IPWRA_DETREND[r]['att']
        
        if abs(paper_val) < 0.001:
            # 论文值接近零，使用绝对差异
            assert abs(python_val - paper_val) < 0.03, \
                f"Detrend r={r}: Python={python_val:.4f}, Paper={paper_val:.4f}"
        else:
            ratio = python_val / paper_val
            assert 0.5 <= ratio <= 1.5, \
                f"Detrend r={r}: Python={python_val:.4f}, Paper={paper_val:.4f}, ratio={ratio:.2f}"
    
    def test_detrend_average_ratio(self, watt_detrend):
        """验证 Detrend r=0-9 的平均比率接近 1.0"""
        ratios = []
        for r in range(10):
            if r not in watt_detrend:
                continue
            python_val = watt_detrend[r]['watt']
            paper_val = PAPER_WATT_IPWRA_DETREND[r]['att']
            if abs(paper_val) > 0.001:
                ratios.append(python_val / paper_val)
        
        if len(ratios) < 5:
            pytest.skip("Not enough valid event times")
        
        avg_ratio = np.mean(ratios)
        assert 0.85 <= avg_ratio <= 1.15, f"Detrend average ratio (r=0-9): {avg_ratio:.2f}"
    
    def test_detrend_correlation(self, watt_detrend):
        """验证 Detrend 趋势方向与论文一致"""
        python_vals = []
        paper_vals = []
        
        for r in range(10):
            if r not in watt_detrend:
                continue
            python_vals.append(watt_detrend[r]['watt'])
            paper_vals.append(PAPER_WATT_IPWRA_DETREND[r]['att'])
        
        if len(python_vals) < 5:
            pytest.skip("Not enough valid event times")
        
        correlation = np.corrcoef(python_vals, paper_vals)[0, 1]
        assert correlation > 0.8, f"Detrend correlation: {correlation:.3f}"
    
    # ========================================================================
    # Demean 测试（使用论文口径的 control_group='all_others'）
    # ========================================================================
    
    @pytest.mark.parametrize("r", range(14))
    def test_demean_event_time_0_to_9(self, watt_demean, r):
        """验证 Demean r=0-13 与论文匹配"""
        if r not in watt_demean:
            pytest.skip(f"Event time {r} not in results")
        
        python_val = watt_demean[r]['watt']
        paper_val = PAPER_WATT_IPWRA_DEMEAN[r]['att']
        
        if abs(paper_val) < 0.001:
            assert abs(python_val - paper_val) < 0.01, \
                f"Demean r={r}: Python={python_val:.4f}, Paper={paper_val:.4f}"
        else:
            ratio = python_val / paper_val
            assert 0.8 <= ratio <= 1.2, \
                f"Demean r={r}: Python={python_val:.4f}, Paper={paper_val:.4f}, ratio={ratio:.2f}"
    
    def test_demean_average_ratio(self, watt_demean):
        """验证 Demean r=0-13 的平均比率接近 1.0"""
        ratios = []
        for r in range(14):
            if r not in watt_demean:
                continue
            python_val = watt_demean[r]['watt']
            paper_val = PAPER_WATT_IPWRA_DEMEAN[r]['att']
            if abs(paper_val) > 0.001:
                ratios.append(python_val / paper_val)
        
        if len(ratios) < 5:
            pytest.skip("Not enough valid event times")
        
        avg_ratio = np.mean(ratios)
        assert 0.85 <= avg_ratio <= 1.15, \
            f"Demean average ratio (r=0-13): {avg_ratio:.2f} (expect 0.85-1.15)"


class TestWalmartTableA4Diagnostics:
    """诊断测试：输出详细对比信息"""
    
    @pytest.fixture(scope='class')
    def walmart_data(self):
        return load_walmart_data()
    
    @pytest.fixture(scope='class')
    def controls(self):
        return [
            'share_pop_poverty_78_above',
            'share_pop_ind_manuf',
            'share_school_some_hs',
        ]
    
    def test_print_comparison_table(self, walmart_data, controls):
        """打印 Python vs Paper 对比表"""
        watt_detrend = estimate_and_compute_watt(
            walmart_data, 'detrend', controls, control_group='not_yet_treated'
        )
        watt_demean = estimate_and_compute_watt(
            walmart_data, 'demean', controls, control_group='all_others'
        )
        
        print("\n" + "=" * 80)
        print("Table A4 复现对比")
        print("=" * 80)
        
        print("\nDetrend (Heterogeneous Trend):")
        print(f"{'r':>3} | {'Python':>10} | {'Paper':>10} | {'Ratio':>8} | {'Status':>10}")
        print("-" * 50)
        
        for r in range(14):
            paper_val = PAPER_WATT_IPWRA_DETREND[r]['att']
            python_val = watt_detrend.get(r, {}).get('watt', np.nan)
            
            if not np.isnan(python_val) and abs(paper_val) > 0.001:
                ratio = python_val / paper_val
                status = "PASS" if 0.8 <= ratio <= 1.2 else "FAIL"
            else:
                ratio = np.nan
                status = "N/A"
            
            print(f"{r:>3} | {python_val:>10.4f} | {paper_val:>10.4f} | {ratio:>8.2f} | {status:>10}")
        
        print("\nDemean:")
        print(f"{'r':>3} | {'Python':>10} | {'Paper':>10} | {'Ratio':>8} | {'Status':>10}")
        print("-" * 50)
        
        for r in range(14):
            paper_val = PAPER_WATT_IPWRA_DEMEAN[r]['att']
            python_val = watt_demean.get(r, {}).get('watt', np.nan)
            
            if not np.isnan(python_val) and abs(paper_val) > 0.001:
                ratio = python_val / paper_val
                status = "PASS" if 0.8 <= ratio <= 1.2 else "HIGH" if ratio > 1.2 else "LOW"
            else:
                ratio = np.nan
                status = "N/A"
            
            print(f"{r:>3} | {python_val:>10.4f} | {paper_val:>10.4f} | {ratio:>8.2f} | {status:>10}")
        
        # 不断言，只打印诊断信息
        assert True


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
