"""
IPWRA端到端测试

使用Castle Law数据进行完整流程测试：
1. 数据加载和预处理
2. 数据变换
3. IPWRA估计
4. 结果验证

Reference: Story E3-S1
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# 确保可以导入lwdid模块
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from lwdid.staggered.estimators import estimate_ipwra
from lwdid.staggered.transformations import transform_staggered_demean
from lwdid.staggered.control_groups import get_valid_control_units, ControlGroupStrategy


class TestIPWRAEndToEnd:
    """IPWRA端到端测试"""
    
    @pytest.fixture
    def castle_data(self):
        """加载并准备Castle Law数据"""
        data_path = Path(__file__).parent.parent.parent / 'data' / 'castle.csv'
        if not data_path.exists():
            pytest.skip(f"Castle data not found at {data_path}")
        
        data = pd.read_csv(data_path)
        
        # 验证列名
        assert 'lhomicide' in data.columns, "结果变量列名应为'lhomicide'"
        assert 'sid' in data.columns, "单位ID列名应为'sid'"
        assert 'year' in data.columns, "时间变量列名应为'year'"
        assert 'effyear' in data.columns, "效果年份列名应为'effyear'"
        
        # 创建gvar列
        data['gvar'] = data['effyear'].fillna(0).astype(int)
        
        # 验证数据结构
        n_units = data['sid'].nunique()
        n_years = data['year'].nunique()
        assert n_units == 50, f"应有50个州，实际{n_units}"
        assert n_years == 11, f"应有11年数据，实际{n_years}"
        
        return data
    
    @pytest.fixture
    def transformed_data(self, castle_data):
        """变换后的数据"""
        return transform_staggered_demean(
            castle_data, 
            y='lhomicide', 
            ivar='sid', 
            tvar='year', 
            gvar='gvar'
        )
    
    def test_ipwra_single_cohort_period(self, transformed_data):
        """
        单个(cohort, period)对的IPWRA估计
        
        测试cohort=2006, period=2006的估计
        """
        g, r = 2006, 2006
        
        # 提取period=2006横截面
        period_data = transformed_data[transformed_data['year'] == r].copy()
        
        # 获取控制组掩码
        unit_control_mask = get_valid_control_units(
            period_data, 'gvar', 'sid',
            cohort=g, period=r,
            strategy=ControlGroupStrategy.NOT_YET_TREATED
        )
        
        # 构建估计样本
        control_mask = period_data['sid'].map(unit_control_mask).fillna(False).astype(bool)
        treat_mask = period_data['gvar'] == g
        sample_mask = treat_mask | control_mask
        
        sample_data = period_data[sample_mask].copy()
        sample_data['D_treat'] = (sample_data['gvar'] == g).astype(int)
        
        # 变换列
        ydot_col = f'ydot_g{g}_r{r}'
        
        if ydot_col not in sample_data.columns:
            pytest.skip(f"变换列 {ydot_col} 不存在")
        
        # 选择控制变量
        controls = ['population', 'income']
        missing_controls = [c for c in controls if c not in sample_data.columns]
        if missing_controls:
            pytest.skip(f"控制变量缺失: {missing_controls}")
        
        # IPWRA估计
        result = estimate_ipwra(
            sample_data,
            y=ydot_col,
            d='D_treat',
            controls=controls,
            se_method='bootstrap',
            n_bootstrap=50,
            seed=42
        )
        
        # 验证结果
        assert result.att is not None
        assert not np.isnan(result.att)
        assert result.se > 0
        assert result.n_treated > 0
        assert result.n_control > 0
        
        print(f"\n=== IPWRA估计结果 (cohort={g}, period={r}) ===")
        print(f"ATT = {result.att:.4f}")
        print(f"SE = {result.se:.4f}")
        print(f"95% CI = [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
        print(f"t-stat = {result.t_stat:.2f}")
        print(f"p-value = {result.pvalue:.4f}")
        print(f"N_treated = {result.n_treated}")
        print(f"N_control = {result.n_control}")
    
    def test_ipwra_vs_ra_castle(self, transformed_data):
        """
        比较IPWRA与RA在Castle数据上的结果
        
        在正确指定的情况下，两者应该接近。
        """
        from lwdid.staggered.estimation import run_ols_regression
        
        g, r = 2006, 2007  # 使用不同的(g, r)对
        
        period_data = transformed_data[transformed_data['year'] == r].copy()
        
        unit_control_mask = get_valid_control_units(
            period_data, 'gvar', 'sid',
            cohort=g, period=r,
            strategy=ControlGroupStrategy.NOT_YET_TREATED
        )
        
        control_mask = period_data['sid'].map(unit_control_mask).fillna(False).astype(bool)
        treat_mask = period_data['gvar'] == g
        sample_mask = treat_mask | control_mask
        
        sample_data = period_data[sample_mask].copy()
        sample_data['D_treat'] = (sample_data['gvar'] == g).astype(int)
        
        ydot_col = f'ydot_g{g}_r{r}'
        
        if ydot_col not in sample_data.columns:
            pytest.skip(f"变换列 {ydot_col} 不存在")
        
        controls = ['population', 'income']
        missing_controls = [c for c in controls if c not in sample_data.columns]
        if missing_controls:
            pytest.skip(f"控制变量缺失: {missing_controls}")
        
        # RA估计
        ra_result = run_ols_regression(
            sample_data, y=ydot_col, d='D_treat', controls=controls
        )
        
        # IPWRA估计
        ipwra_result = estimate_ipwra(
            sample_data, y=ydot_col, d='D_treat', controls=controls,
            se_method='analytical'
        )
        
        print(f"\n=== RA vs IPWRA比较 (cohort={g}, period={r}) ===")
        print(f"RA ATT = {ra_result['att']:.4f}, SE = {ra_result['se']:.4f}")
        print(f"IPWRA ATT = {ipwra_result.att:.4f}, SE = {ipwra_result.se:.4f}")
        
        # 在正确指定的情况下，差异应该较小
        diff = abs(ipwra_result.att - ra_result['att'])
        print(f"差异 = {diff:.4f}")
        
        # 允许一定差异（因为两种方法本质不同）
        assert diff < 0.5, f"IPWRA与RA差异过大: {diff}"
    
    def test_ipwra_multiple_cohort_periods(self, transformed_data):
        """测试多个(cohort, period)对"""
        cohorts_to_test = [2006, 2007]
        
        results = []
        
        for g in cohorts_to_test:
            # 测试instantaneous effect (r = g)
            r = g
            
            period_data = transformed_data[transformed_data['year'] == r].copy()
            
            try:
                unit_control_mask = get_valid_control_units(
                    period_data, 'gvar', 'sid',
                    cohort=g, period=r,
                    strategy=ControlGroupStrategy.NOT_YET_TREATED
                )
                
                control_mask = period_data['sid'].map(unit_control_mask).fillna(False).astype(bool)
                treat_mask = period_data['gvar'] == g
                sample_mask = treat_mask | control_mask
                
                sample_data = period_data[sample_mask].copy()
                sample_data['D_treat'] = (sample_data['gvar'] == g).astype(int)
                
                ydot_col = f'ydot_g{g}_r{r}'
                
                if ydot_col not in sample_data.columns:
                    continue
                
                controls = ['population', 'income']
                if not all(c in sample_data.columns for c in controls):
                    continue
                
                result = estimate_ipwra(
                    sample_data, y=ydot_col, d='D_treat', controls=controls,
                    se_method='analytical'
                )
                
                results.append({
                    'cohort': g,
                    'period': r,
                    'att': result.att,
                    'se': result.se,
                    'n_treated': result.n_treated,
                    'n_control': result.n_control
                })
            except Exception as e:
                print(f"跳过 (g={g}, r={r}): {e}")
                continue
        
        assert len(results) > 0, "至少应有一个成功的估计"
        
        print("\n=== 多个(cohort, period)对的IPWRA估计 ===")
        for r in results:
            print(f"({r['cohort']}, {r['period']}): ATT={r['att']:.4f}, SE={r['se']:.4f}, "
                  f"N_treat={r['n_treated']}, N_ctrl={r['n_control']}")


class TestIPWRADataValidation:
    """数据验证测试"""
    
    def test_castle_data_structure(self):
        """验证Castle数据结构"""
        data_path = Path(__file__).parent.parent.parent / 'data' / 'castle.csv'
        if not data_path.exists():
            pytest.skip(f"Castle data not found at {data_path}")
        
        data = pd.read_csv(data_path)
        
        # 验证关键列
        required_cols = ['sid', 'year', 'lhomicide', 'effyear', 'state']
        for col in required_cols:
            assert col in data.columns, f"缺少列: {col}"
        
        # 验证数据范围
        assert data['year'].min() == 2000
        assert data['year'].max() == 2010
        
        # 验证cohort结构
        data['gvar'] = data['effyear'].fillna(0).astype(int)
        unit_gvar = data.groupby('sid')['gvar'].first()
        
        n_never_treated = (unit_gvar == 0).sum()
        n_treated = (unit_gvar > 0).sum()
        
        print(f"\nCastle数据结构:")
        print(f"  总州数: {len(unit_gvar)}")
        print(f"  Never Treated: {n_never_treated}")
        print(f"  Treated: {n_treated}")
        
        # 验证预期值
        assert n_never_treated == 29, f"应有29个NT州，实际{n_never_treated}"
        assert n_treated == 21, f"应有21个treated州，实际{n_treated}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
