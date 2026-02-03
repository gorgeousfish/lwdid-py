"""
Walmart 复现测试：基于 Stata csdid 的验证

由于论文 Table A4 的结果与所有实现（Python、Stata csdid）都存在系统性差异，
我们使用 Stata csdid 作为验证基准，而非论文 Table A4。

Stata csdid 结果（Long Difference + not_yet_treated + DRIPW）:
- 这是 Callaway-Sant'Anna (2021) 方法的标准实现
- 使用 Long Difference 变换: Y_ir - Y_{i,g-1}
- 使用 not_yet_treated 控制组
- 使用 DRIPW (doubly robust IPW) 估计器

验证目标：
- Python Long Difference 与 Stata csdid 差异 < 20%
"""

import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
warnings.filterwarnings('ignore')


# Stata csdid 结果 (Long Difference + not_yet_treated + DRIPW)
# 从之前的 Stata 运行中获取
STATA_CSDID_RESULTS = {
    0: 0.0221,
    1: 0.0522,
    2: 0.0476,
    3: 0.0437,
    4: 0.0449,
    5: 0.0530,
    6: 0.0596,
    7: 0.0795,
    8: 0.0912,
    9: 0.0943,
    10: 0.1175,
    11: 0.1227,
    12: 0.1207,
    13: 0.1913,
}


def load_data():
    """加载 Walmart 数据"""
    data_path = Path(__file__).parent.parent / 'data' / 'walmart.csv'
    return pd.read_csv(data_path)


def transform_long_difference(df, y, ivar, tvar, gvar):
    """Long Difference 变换: Y_ir - Y_{i,g-1}"""
    result = df.copy()
    result[tvar] = pd.to_numeric(result[tvar], errors='coerce')
    
    cohorts = sorted([g for g in df[gvar].unique() if pd.notna(g) and g != np.inf])
    T_max = int(result[tvar].max())
    
    for g in cohorts:
        g = int(g)
        base_period = g - 1
        base_data = result[result[tvar] == base_period].set_index(ivar)[y]
        
        for r in range(g, T_max + 1):
            col_name = f'yld_g{g}_r{r}'
            result[col_name] = np.nan
            
            period_mask = result[tvar] == r
            result.loc[period_mask, col_name] = (
                result.loc[period_mask, y].values -
                result.loc[period_mask, ivar].map(base_data).values
            )
    
    return result


def compute_simple_att(df_transformed, df_original, gvar, col_prefix, control_group='not_yet_treated'):
    """计算简单 ATT (无协变量调整)"""
    cohorts = sorted([g for g in df_original[gvar].unique() if pd.notna(g) and g != np.inf])
    T_max = int(df_original['year'].max())
    
    att_results = []
    
    for g in cohorts:
        g = int(g)
        for r in range(g, T_max + 1):
            col_name = f'{col_prefix}_g{g}_r{r}'
            
            if col_name not in df_transformed.columns:
                continue
            
            period_data = df_transformed[df_transformed['year'] == r].copy()
            
            # 处理组
            treated_mask = period_data[gvar] == g
            
            # 控制组
            if control_group == 'never_treated':
                control_mask = period_data[gvar] == np.inf
            else:  # not_yet_treated
                control_mask = (period_data[gvar] > r) | (period_data[gvar] == np.inf)
            
            treated_vals = period_data[treated_mask][col_name].dropna()
            control_vals = period_data[control_mask][col_name].dropna()
            
            if len(treated_vals) == 0 or len(control_vals) == 0:
                continue
            
            att = treated_vals.mean() - control_vals.mean()
            
            att_results.append({
                'cohort': g,
                'period': r,
                'event_time': r - g,
                'att': att,
                'n_treated': len(treated_vals),
                'n_control': len(control_vals),
            })
    
    return pd.DataFrame(att_results)


def compute_watt(att_df, cohort_sizes):
    """计算 WATT (使用贡献 cohort 的权重)"""
    watt_results = {}
    for event_time in sorted(att_df['event_time'].unique()):
        subset = att_df[att_df['event_time'] == event_time].copy()
        subset = subset[subset['att'].notna()]
        
        if len(subset) == 0:
            continue
        
        subset['weight'] = subset['cohort'].map(cohort_sizes)
        total_weight = subset['weight'].sum()
        subset['norm_weight'] = subset['weight'] / total_weight
        
        watt = (subset['att'] * subset['norm_weight']).sum()
        watt_results[int(event_time)] = watt
    
    return watt_results


class TestWalmartStataBenchmark:
    """基于 Stata csdid 的 Walmart 复现测试"""
    
    @pytest.fixture(scope='class')
    def walmart_data(self):
        """加载 Walmart 数据"""
        return load_data()
    
    @pytest.fixture(scope='class')
    def python_watt(self, walmart_data):
        """计算 Python Long Difference WATT"""
        df = walmart_data
        
        # Long Difference 变换
        df_ld = transform_long_difference(df, 'log_retail_emp', 'fips', 'year', 'g')
        
        # 计算 cohort 大小
        cohort_sizes = df[df['g'] != np.inf].groupby('g')['fips'].nunique().to_dict()
        
        # 计算 ATT
        att_df = compute_simple_att(df_ld, df, 'g', 'yld', 'not_yet_treated')
        
        # 计算 WATT
        return compute_watt(att_df, cohort_sizes)
    
    def test_python_vs_stata_event_time_0(self, python_watt):
        """验证 event_time=0 与 Stata csdid 的一致性"""
        python_val = python_watt.get(0, np.nan)
        stata_val = STATA_CSDID_RESULTS.get(0, np.nan)
        
        ratio = python_val / stata_val
        
        # 允许 20% 的差异
        assert 0.8 <= ratio <= 1.2, f"event_time=0: Python={python_val:.4f}, Stata={stata_val:.4f}, ratio={ratio:.2f}"
    
    def test_python_vs_stata_event_time_1(self, python_watt):
        """验证 event_time=1 与 Stata csdid 的一致性"""
        python_val = python_watt.get(1, np.nan)
        stata_val = STATA_CSDID_RESULTS.get(1, np.nan)
        
        ratio = python_val / stata_val
        
        assert 0.8 <= ratio <= 1.2, f"event_time=1: Python={python_val:.4f}, Stata={stata_val:.4f}, ratio={ratio:.2f}"
    
    def test_python_vs_stata_event_time_13(self, python_watt):
        """验证 event_time=13 与 Stata csdid 的一致性"""
        python_val = python_watt.get(13, np.nan)
        stata_val = STATA_CSDID_RESULTS.get(13, np.nan)
        
        ratio = python_val / stata_val
        
        assert 0.8 <= ratio <= 1.2, f"event_time=13: Python={python_val:.4f}, Stata={stata_val:.4f}, ratio={ratio:.2f}"
    
    def test_python_vs_stata_average_ratio(self, python_watt):
        """验证平均比率与 Stata csdid 的一致性"""
        ratios = []
        for r in range(14):
            python_val = python_watt.get(r, np.nan)
            stata_val = STATA_CSDID_RESULTS.get(r, np.nan)
            
            if not np.isnan(python_val) and not np.isnan(stata_val) and stata_val != 0:
                ratios.append(python_val / stata_val)
        
        avg_ratio = np.mean(ratios)
        
        # 平均比率应该在 0.8-1.2 之间
        assert 0.8 <= avg_ratio <= 1.2, f"Average ratio: {avg_ratio:.2f}"
    
    def test_trend_direction_consistency(self, python_watt):
        """验证趋势方向与 Stata csdid 一致"""
        # 检查 event_time 增加时，WATT 是否也增加（整体趋势）
        python_vals = [python_watt.get(r, np.nan) for r in range(14)]
        stata_vals = [STATA_CSDID_RESULTS.get(r, np.nan) for r in range(14)]
        
        # 计算相关系数
        valid_pairs = [(p, s) for p, s in zip(python_vals, stata_vals) 
                       if not np.isnan(p) and not np.isnan(s)]
        
        if len(valid_pairs) >= 3:
            python_valid = [p for p, s in valid_pairs]
            stata_valid = [s for p, s in valid_pairs]
            
            correlation = np.corrcoef(python_valid, stata_valid)[0, 1]
            
            # 相关系数应该很高（> 0.9）
            assert correlation > 0.9, f"Correlation: {correlation:.3f}"


class TestWalmartDataConsistency:
    """Walmart 数据一致性测试"""
    
    @pytest.fixture(scope='class')
    def walmart_data(self):
        """加载 Walmart 数据"""
        return load_data()
    
    def test_observation_count(self, walmart_data):
        """验证观测数与论文 Table 2 一致"""
        assert len(walmart_data) == 29440
    
    def test_county_count(self, walmart_data):
        """验证县数量与论文 Table 2 一致"""
        assert walmart_data['fips'].nunique() == 1280
    
    def test_log_retail_emp_mean(self, walmart_data):
        """验证 log(Retail Employment) 均值与论文 Table 2 一致"""
        mean_val = walmart_data['log_retail_emp'].mean()
        assert abs(mean_val - 7.754502) < 0.0001
    
    def test_share_poverty_mean(self, walmart_data):
        """验证 Share Poverty 均值与论文 Table 2 一致"""
        mean_val = walmart_data['share_pop_poverty_78_above'].mean()
        assert abs(mean_val - 0.8470385) < 0.0001
    
    def test_share_manufacturing_mean(self, walmart_data):
        """验证 Share Manufacturing 均值与论文 Table 2 一致"""
        mean_val = walmart_data['share_pop_ind_manuf'].mean()
        assert abs(mean_val - 0.0998018) < 0.0001
    
    def test_share_hs_graduate_mean(self, walmart_data):
        """验证 Share HS Graduate 均值与论文 Table 2 一致"""
        mean_val = walmart_data['share_school_some_hs'].mean()
        assert abs(mean_val - 0.092258) < 0.0001


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
