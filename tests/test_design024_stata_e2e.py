"""
DESIGN-024 Stata E2E 验证测试

目的：验证 Python 实现与 Stata 参考实现（castle_lw.do）的数值一致性

Stata 参考值来源：
- castle_lw.do 运行结果
- 论文 Lee & Wooldridge (2025) Section 7.2

测试内容：
1. Overall ATT 数值与 Stata 一致
2. HC3 标准误与 Stata 一致
3. 样本量与 Stata 一致
4. Cohort 权重与 Stata 一致

创建日期: 2026-01-17
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lwdid import lwdid
from lwdid.staggered.aggregation import (
    aggregate_to_overall,
    aggregate_to_cohort,
    construct_aggregated_outcome,
)
from lwdid.staggered.transformations import (
    transform_staggered_demean,
    transform_staggered_detrend,
    get_cohorts,
)


# =============================================================================
# Stata Reference Values (from castle_lw.do execution)
# =============================================================================

STATA_REFERENCE = {
    'demean': {
        # From: reg ydot_bar d if year == 2007, vce(hc3)
        'att': 0.0917454,
        'se_hc3': 0.0611743,
        't_stat_hc3': 1.50,
        'p_value_hc3': 0.140,  # approximate
    },
    'detrend': {
        # From paper Section 7.2
        'att': 0.067,
        't_stat_hc3': 1.21,
    },
    'sample': {
        'n_treated': 21,
        'n_control': 29,
        'n_total': 50,
    },
    'cohort_weights': {
        # From: mat w = freqs / r(N)
        # w_g = N_g / N_treat where N_treat = 21
        2005: 1/21,   # 0.04761905
        2006: 13/21,  # 0.61904762
        2007: 4/21,   # 0.19047619
        2008: 2/21,   # 0.09523810
        2009: 1/21,   # 0.04761905
    },
    'cohort_sizes': {
        2005: 1,
        2006: 13,
        2007: 4,
        2008: 2,
        2009: 1,
    },
}


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def castle_data():
    """Load Castle Law data."""
    possible_paths = [
        os.environ.get('LWDID_TEST_DATA_PATH', ''),
        'lwdid-py_v0.1.0/data/castle.csv',
        '../lwdid-py_v0.1.0/data/castle.csv',
        str(Path(__file__).parent.parent / 'data' / 'castle.csv'),
    ]
    
    data_path = None
    for path in possible_paths:
        if path and Path(path).exists():
            data_path = path
            break
    
    if data_path is None:
        pytest.skip("Castle Law data not found")
    
    data = pd.read_csv(data_path)
    # Create gvar from effyear (0 = never treated)
    data['gvar'] = data['effyear'].fillna(0).astype(int)
    
    return data


# =============================================================================
# Overall ATT Validation Tests
# =============================================================================

class TestStataOverallATTValidation:
    """
    验证 Python 计算的 Overall ATT 与 Stata 参考值一致。
    
    Stata 命令：reg ydot_bar d if year == 2007, vce(hc3)
    Python 等价：aggregate_to_overall(..., vce='hc3')
    """
    
    def test_demean_overall_att_matches_stata(self, castle_data):
        """
        验证 demean 变换的 Overall ATT 与 Stata 一致。
        
        Stata 结果: ATT = 0.0917454
        允许误差: ±0.001 (考虑浮点精度)
        """
        transformed = transform_staggered_demean(
            castle_data, y='lhomicide', ivar='sid', tvar='year', gvar='gvar'
        )
        
        result = aggregate_to_overall(
            transformed, 'gvar', 'sid', 'year',
            never_treated_values=[0],
            vce='hc3'
        )
        
        stata_att = STATA_REFERENCE['demean']['att']
        python_att = result.att
        
        print(f"\n=== Demean Overall ATT 验证 ===")
        print(f"  Stata ATT:  {stata_att:.7f}")
        print(f"  Python ATT: {python_att:.7f}")
        print(f"  差异:       {abs(python_att - stata_att):.7f}")
        
        # 允许 0.001 的绝对误差
        assert np.isclose(python_att, stata_att, atol=0.001), \
            f"Python ATT {python_att:.6f} differs from Stata {stata_att:.6f}"
    
    def test_demean_hc3_se_matches_stata(self, castle_data):
        """
        验证 HC3 标准误与 Stata 一致。
        
        Stata 结果: SE (HC3) = 0.0611743
        允许误差: ±0.005 (SE 对数据微小差异更敏感)
        """
        transformed = transform_staggered_demean(
            castle_data, y='lhomicide', ivar='sid', tvar='year', gvar='gvar'
        )
        
        result = aggregate_to_overall(
            transformed, 'gvar', 'sid', 'year',
            never_treated_values=[0],
            vce='hc3'
        )
        
        stata_se = STATA_REFERENCE['demean']['se_hc3']
        python_se = result.se
        
        print(f"\n=== Demean HC3 SE 验证 ===")
        print(f"  Stata SE:  {stata_se:.7f}")
        print(f"  Python SE: {python_se:.7f}")
        print(f"  差异:      {abs(python_se - stata_se):.7f}")
        
        # 允许 0.005 的绝对误差
        assert np.isclose(python_se, stata_se, atol=0.005), \
            f"Python SE {python_se:.6f} differs from Stata {stata_se:.6f}"
    
    def test_demean_t_stat_matches_stata(self, castle_data):
        """
        验证 t 统计量与 Stata 一致。
        
        Stata 结果: t = 1.50
        允许误差: ±0.05
        """
        transformed = transform_staggered_demean(
            castle_data, y='lhomicide', ivar='sid', tvar='year', gvar='gvar'
        )
        
        result = aggregate_to_overall(
            transformed, 'gvar', 'sid', 'year',
            never_treated_values=[0],
            vce='hc3'
        )
        
        stata_t = STATA_REFERENCE['demean']['t_stat_hc3']
        python_t = result.t_stat
        
        print(f"\n=== Demean t-stat 验证 ===")
        print(f"  Stata t:  {stata_t:.2f}")
        print(f"  Python t: {python_t:.2f}")
        print(f"  差异:     {abs(python_t - stata_t):.2f}")
        
        # 允许 0.05 的绝对误差
        assert np.isclose(python_t, stata_t, atol=0.05), \
            f"Python t-stat {python_t:.2f} differs from Stata {stata_t:.2f}"
    
    def test_detrend_overall_att_matches_paper(self, castle_data):
        """
        验证 detrend 变换的 Overall ATT 与论文一致。
        
        论文结果: ATT ≈ 0.067
        允许误差: ±0.002
        """
        transformed = transform_staggered_detrend(
            castle_data, y='lhomicide', ivar='sid', tvar='year', gvar='gvar'
        )
        
        result = aggregate_to_overall(
            transformed, 'gvar', 'sid', 'year',
            never_treated_values=[0],
            transform_type='detrend',
            vce='hc3'
        )
        
        paper_att = STATA_REFERENCE['detrend']['att']
        python_att = result.att
        
        print(f"\n=== Detrend Overall ATT 验证 ===")
        print(f"  Paper ATT:  {paper_att:.3f}")
        print(f"  Python ATT: {python_att:.4f}")
        print(f"  差异:       {abs(python_att - paper_att):.4f}")
        
        # 允许 0.002 的绝对误差
        assert np.isclose(python_att, paper_att, atol=0.002), \
            f"Python ATT {python_att:.4f} differs from paper {paper_att:.3f}"


# =============================================================================
# Sample Size Validation Tests
# =============================================================================

class TestStataSampleSizeValidation:
    """
    验证样本量计算与 Stata 一致。
    
    Castle Law 数据:
    - n_treated = 21 (states with Castle Doctrine)
    - n_control = 29 (never-treated states)
    """
    
    def test_n_treated_matches_stata(self, castle_data):
        """验证 treated 样本量 = 21"""
        transformed = transform_staggered_demean(
            castle_data, y='lhomicide', ivar='sid', tvar='year', gvar='gvar'
        )
        
        result = aggregate_to_overall(
            transformed, 'gvar', 'sid', 'year',
            never_treated_values=[0]
        )
        
        stata_n_treated = STATA_REFERENCE['sample']['n_treated']
        python_n_treated = result.n_treated
        
        print(f"\n=== n_treated 验证 ===")
        print(f"  Stata:  {stata_n_treated}")
        print(f"  Python: {python_n_treated}")
        
        assert python_n_treated == stata_n_treated, \
            f"Python n_treated {python_n_treated} != Stata {stata_n_treated}"
    
    def test_n_control_matches_stata(self, castle_data):
        """验证 control 样本量 = 29"""
        transformed = transform_staggered_demean(
            castle_data, y='lhomicide', ivar='sid', tvar='year', gvar='gvar'
        )
        
        result = aggregate_to_overall(
            transformed, 'gvar', 'sid', 'year',
            never_treated_values=[0]
        )
        
        stata_n_control = STATA_REFERENCE['sample']['n_control']
        python_n_control = result.n_control
        
        print(f"\n=== n_control 验证 ===")
        print(f"  Stata:  {stata_n_control}")
        print(f"  Python: {python_n_control}")
        
        assert python_n_control == stata_n_control, \
            f"Python n_control {python_n_control} != Stata {stata_n_control}"


# =============================================================================
# Cohort Weight Validation Tests
# =============================================================================

class TestStataCohortWeightValidation:
    """
    验证 cohort 权重计算与 Stata 一致。
    
    公式: ω_g = N_g / N_treat
    
    Stata 权重 (from mat w = freqs / r(N)):
    - 2005: 1/21 = 0.04761905
    - 2006: 13/21 = 0.61904762
    - 2007: 4/21 = 0.19047619
    - 2008: 2/21 = 0.09523810
    - 2009: 1/21 = 0.04761905
    """
    
    def test_cohort_weights_match_stata(self, castle_data):
        """验证每个 cohort 的权重与 Stata 一致"""
        transformed = transform_staggered_demean(
            castle_data, y='lhomicide', ivar='sid', tvar='year', gvar='gvar'
        )
        
        result = aggregate_to_overall(
            transformed, 'gvar', 'sid', 'year',
            never_treated_values=[0]
        )
        
        print(f"\n=== Cohort 权重验证 ===")
        
        for g, stata_weight in STATA_REFERENCE['cohort_weights'].items():
            python_weight = result.cohort_weights.get(g, 0)
            
            print(f"  Cohort {g}: Stata={stata_weight:.6f}, Python={python_weight:.6f}")
            
            assert np.isclose(python_weight, stata_weight, atol=1e-6), \
                f"Cohort {g} weight differs: Python={python_weight:.6f}, Stata={stata_weight:.6f}"
    
    def test_weights_sum_to_one(self, castle_data):
        """验证权重总和 = 1.0"""
        transformed = transform_staggered_demean(
            castle_data, y='lhomicide', ivar='sid', tvar='year', gvar='gvar'
        )
        
        result = aggregate_to_overall(
            transformed, 'gvar', 'sid', 'year',
            never_treated_values=[0]
        )
        
        weights_sum = sum(result.cohort_weights.values())
        
        print(f"\n=== 权重总和验证 ===")
        print(f"  Sum: {weights_sum:.10f}")
        
        assert np.isclose(weights_sum, 1.0, atol=1e-10), \
            f"Weights sum to {weights_sum}, expected 1.0"


# =============================================================================
# Cohort Effect Validation Tests
# =============================================================================

class TestStataCohortEffectValidation:
    """
    验证 cohort-specific 效应估计与 Stata 一致。
    """
    
    def test_cohort_structure_correct(self, castle_data):
        """验证 cohort 结构（每个 cohort 的单位数）与 Stata 一致"""
        transformed = transform_staggered_demean(
            castle_data, y='lhomicide', ivar='sid', tvar='year', gvar='gvar'
        )
        
        cohort_effects = aggregate_to_cohort(
            transformed,
            gvar='gvar',
            ivar='sid',
            tvar='year',
            cohorts=[2005, 2006, 2007, 2008, 2009],
            T_max=2010,
            never_treated_values=[0]
        )
        
        print(f"\n=== Cohort 结构验证 ===")
        
        for effect in cohort_effects:
            g = effect.cohort
            stata_n = STATA_REFERENCE['cohort_sizes'][g]
            python_n = effect.n_units
            
            print(f"  Cohort {g}: Stata={stata_n}, Python={python_n}")
            
            assert python_n == stata_n, \
                f"Cohort {g} size differs: Python={python_n}, Stata={stata_n}"
    
    def test_all_cohorts_have_valid_effects(self, castle_data):
        """验证所有 cohort 都有有效的效应估计"""
        transformed = transform_staggered_demean(
            castle_data, y='lhomicide', ivar='sid', tvar='year', gvar='gvar'
        )
        
        cohort_effects = aggregate_to_cohort(
            transformed,
            gvar='gvar',
            ivar='sid',
            tvar='year',
            cohorts=[2005, 2006, 2007, 2008, 2009],
            T_max=2010,
            never_treated_values=[0]
        )
        
        print(f"\n=== Cohort 效应有效性验证 ===")
        
        assert len(cohort_effects) == 5, f"Expected 5 cohorts, got {len(cohort_effects)}"
        
        for effect in cohort_effects:
            print(f"  Cohort {effect.cohort}: ATT={effect.att:.4f}, SE={effect.se:.4f}")
            
            assert not np.isnan(effect.att), f"Cohort {effect.cohort} ATT is NaN"
            assert effect.se > 0, f"Cohort {effect.cohort} SE should be positive"


# =============================================================================
# End-to-End lwdid API Validation Tests
# =============================================================================

class TestLwdidAPIStataValidation:
    """
    使用 lwdid() 主 API 验证与 Stata 结果一致。
    """
    
    def test_lwdid_demean_overall_matches_stata(self, castle_data):
        """使用 lwdid API 验证 demean overall 效应"""
        results = lwdid(
            data=castle_data,
            y='lhomicide',
            ivar='sid',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            control_group='never_treated',
            aggregate='overall',
            vce='hc3'
        )
        
        stata_att = STATA_REFERENCE['demean']['att']
        python_att = results.att_overall
        
        print(f"\n=== lwdid API demean overall 验证 ===")
        print(f"  Stata ATT:  {stata_att:.6f}")
        print(f"  Python ATT: {python_att:.6f}")
        print(f"  差异:       {abs(python_att - stata_att):.6f}")
        
        assert np.isclose(python_att, stata_att, atol=0.001), \
            f"lwdid ATT {python_att:.6f} differs from Stata {stata_att:.6f}"
    
    def test_lwdid_detrend_overall_matches_paper(self, castle_data):
        """使用 lwdid API 验证 detrend overall 效应"""
        results = lwdid(
            data=castle_data,
            y='lhomicide',
            ivar='sid',
            tvar='year',
            gvar='gvar',
            rolling='detrend',
            control_group='never_treated',
            aggregate='overall',
            vce='hc3'
        )
        
        paper_att = STATA_REFERENCE['detrend']['att']
        python_att = results.att_overall
        
        print(f"\n=== lwdid API detrend overall 验证 ===")
        print(f"  Paper ATT:  {paper_att:.3f}")
        print(f"  Python ATT: {python_att:.4f}")
        print(f"  差异:       {abs(python_att - paper_att):.4f}")
        
        assert np.isclose(python_att, paper_att, atol=0.002), \
            f"lwdid ATT {python_att:.4f} differs from paper {paper_att:.3f}"


# =============================================================================
# DESIGN-024 Specific: Missing Cohort Behavior Tests
# =============================================================================

class TestDesign024MissingCohortBehavior:
    """
    验证 DESIGN-024 修复后的行为与 Stata 一致。
    
    关键点：
    - Stata: missing + number = missing
    - Python (修复后): 缺失 cohort → NaN
    """
    
    def test_castle_law_no_missing_cohort_for_nt(self, castle_data):
        """
        Castle Law 数据中所有 NT 单位都有完整的 cohort 数据。
        
        验证：所有 NT 单位的 Y_bar 都应该是有效值（非 NaN）。
        """
        transformed = transform_staggered_demean(
            castle_data, y='lhomicide', ivar='sid', tvar='year', gvar='gvar'
        )
        
        cohorts = get_cohorts(transformed, 'gvar', 'sid', never_treated_values=[0])
        T_max = int(transformed['year'].max())
        
        # Compute weights
        unit_gvar = transformed.groupby('sid')['gvar'].first()
        cohort_sizes = {g: int((unit_gvar == g).sum()) for g in cohorts}
        N_treat = sum(cohort_sizes.values())
        weights = {g: n / N_treat for g, n in cohort_sizes.items()}
        
        # No DESIGN-024 warning should be raised for Castle Law data
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            Y_bar = construct_aggregated_outcome(
                transformed, 'gvar', 'sid', 'year',
                weights, cohorts, T_max,
                transform_type='demean',
                never_treated_values=[0]
            )
            
            design024_warnings = [x for x in w if 'DESIGN-024' in str(x.message)]
            
            print(f"\n=== Castle Law NT 完整性验证 ===")
            print(f"  DESIGN-024 warnings: {len(design024_warnings)}")
            
            # Castle Law data should have complete cohort data for all NT units
            assert len(design024_warnings) == 0, \
                "Unexpected DESIGN-024 warning for Castle Law data"
        
        # Verify all NT units have valid Y_bar
        nt_mask = unit_gvar == 0
        nt_y_bar = Y_bar[nt_mask]
        n_valid_nt = nt_y_bar.notna().sum()
        n_total_nt = len(nt_y_bar)
        
        print(f"  NT units with valid Y_bar: {n_valid_nt}/{n_total_nt}")
        
        assert n_valid_nt == n_total_nt, \
            f"Expected all {n_total_nt} NT units to have valid Y_bar, got {n_valid_nt}"
    
    def test_simulated_missing_cohort_gives_nan(self):
        """
        模拟测试：当 NT 单位缺少某个 cohort 数据时，Y_bar 应为 NaN。
        
        这验证了 DESIGN-024 修复的核心行为：与 Stata 一致，不重新归一化。
        """
        # Create data where NT unit is missing some cohort data
        data = pd.DataFrame({
            'id': [1,1,1,1, 2,2,2,2, 3,3,3,3],
            'year': [1,2,3,4, 1,2,3,4, 1,2,3,4],
            'y': [
                10, 12, 20, 22,  # cohort 3
                15, 17, 27, 29,  # cohort 4
                5, 7, 9, np.nan,  # NT - missing at t=4 (will have NaN for cohort 4)
            ],
            'gvar': [3]*4 + [4]*4 + [0]*4
        })
        
        transformed = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar',
            never_treated_values=[0]
        )
        
        cohorts = [3, 4]
        T_max = 4
        weights = {3: 0.5, 4: 0.5}
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            Y_bar = construct_aggregated_outcome(
                transformed, 'gvar', 'id', 'year',
                weights, cohorts, T_max,
                transform_type='demean',
                never_treated_values=[0]
            )
            
            # Should have DESIGN-024 warning
            design024_warnings = [x for x in w if 'DESIGN-024' in str(x.message)]
        
        print(f"\n=== 模拟缺失 cohort 行为验证 ===")
        print(f"  DESIGN-024 warnings: {len(design024_warnings)}")
        print(f"  NT unit (id=3) Y_bar: {Y_bar[3]}")
        
        # NT unit should have NaN (not re-normalized value)
        assert np.isnan(Y_bar[3]), \
            f"Expected NaN for NT unit with missing cohort, got {Y_bar[3]}"
        
        # Treated units should have valid values
        assert not np.isnan(Y_bar[1]), "Cohort 3 unit should have valid Y_bar"
        assert not np.isnan(Y_bar[2]), "Cohort 4 unit should have valid Y_bar"


# =============================================================================
# Comprehensive Summary Test
# =============================================================================

class TestStataValidationSummary:
    """
    综合验证测试：一次性验证所有关键指标。
    """
    
    def test_comprehensive_stata_validation(self, castle_data):
        """
        综合验证：ATT、SE、样本量、权重一次性检查。
        """
        transformed = transform_staggered_demean(
            castle_data, y='lhomicide', ivar='sid', tvar='year', gvar='gvar'
        )
        
        result = aggregate_to_overall(
            transformed, 'gvar', 'sid', 'year',
            never_treated_values=[0],
            vce='hc3'
        )
        
        print(f"\n{'='*60}")
        print(f"Python vs Stata 综合验证报告")
        print(f"{'='*60}")
        
        # ATT
        stata_att = STATA_REFERENCE['demean']['att']
        att_ok = np.isclose(result.att, stata_att, atol=0.001)
        print(f"\n1. Overall ATT:")
        print(f"   Stata:  {stata_att:.7f}")
        print(f"   Python: {result.att:.7f}")
        print(f"   状态:   {'✓ 通过' if att_ok else '✗ 失败'}")
        
        # SE
        stata_se = STATA_REFERENCE['demean']['se_hc3']
        se_ok = np.isclose(result.se, stata_se, atol=0.005)
        print(f"\n2. HC3 SE:")
        print(f"   Stata:  {stata_se:.7f}")
        print(f"   Python: {result.se:.7f}")
        print(f"   状态:   {'✓ 通过' if se_ok else '✗ 失败'}")
        
        # t-stat
        stata_t = STATA_REFERENCE['demean']['t_stat_hc3']
        t_ok = np.isclose(result.t_stat, stata_t, atol=0.05)
        print(f"\n3. t-statistic:")
        print(f"   Stata:  {stata_t:.2f}")
        print(f"   Python: {result.t_stat:.2f}")
        print(f"   状态:   {'✓ 通过' if t_ok else '✗ 失败'}")
        
        # Sample sizes
        n_treat_ok = result.n_treated == STATA_REFERENCE['sample']['n_treated']
        n_ctrl_ok = result.n_control == STATA_REFERENCE['sample']['n_control']
        print(f"\n4. 样本量:")
        print(f"   n_treated: Stata={STATA_REFERENCE['sample']['n_treated']}, Python={result.n_treated} {'✓' if n_treat_ok else '✗'}")
        print(f"   n_control: Stata={STATA_REFERENCE['sample']['n_control']}, Python={result.n_control} {'✓' if n_ctrl_ok else '✗'}")
        
        # Weights
        weights_ok = all(
            np.isclose(result.cohort_weights.get(g, 0), w, atol=1e-6)
            for g, w in STATA_REFERENCE['cohort_weights'].items()
        )
        print(f"\n5. Cohort 权重:")
        print(f"   状态: {'✓ 全部匹配' if weights_ok else '✗ 存在差异'}")
        
        print(f"\n{'='*60}")
        print(f"总结: {'全部通过' if all([att_ok, se_ok, t_ok, n_treat_ok, n_ctrl_ok, weights_ok]) else '存在失败项'}")
        print(f"{'='*60}")
        
        # Assert all checks
        assert att_ok, "ATT mismatch"
        assert se_ok, "SE mismatch"
        assert t_ok, "t-stat mismatch"
        assert n_treat_ok, "n_treated mismatch"
        assert n_ctrl_ok, "n_control mismatch"
        assert weights_ok, "Weights mismatch"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
