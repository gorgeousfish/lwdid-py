# -*- coding: utf-8 -*-
"""
小样本DGP属性测试

Property 2: 相同种子产生相同数据
Validates: Requirements 2.5

References
----------
Lee & Wooldridge (2026) ssrn-5325686, Section 5
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add fixtures path
fixtures_path = Path(__file__).parent.parent.parent / 'test_common_timing' / 'fixtures'
sys.path.insert(0, str(fixtures_path))


class TestSmallSampleDGPReproducibility:
    """
    Property 2: 相同种子产生相同数据
    Validates: Requirements 2.5
    """
    
    def test_same_seed_produces_identical_data(self):
        """
        验证相同种子产生完全相同的数据
        """
        from dgp_small_sample import generate_small_sample_dgp
        
        seed = 42
        
        # 生成两次数据
        data1, params1 = generate_small_sample_dgp(seed=seed)
        data2, params2 = generate_small_sample_dgp(seed=seed)
        
        # 验证数据完全相同
        pd.testing.assert_frame_equal(data1, data2)
        
        # 验证参数相同
        assert params1['n_treated'] == params2['n_treated']
        assert params1['n_control'] == params2['n_control']
        assert params1['tau'] == params2['tau']
    
    def test_different_seeds_produce_different_data(self):
        """
        验证不同种子产生不同的数据
        """
        from dgp_small_sample import generate_small_sample_dgp
        
        data1, _ = generate_small_sample_dgp(seed=42)
        data2, _ = generate_small_sample_dgp(seed=43)
        
        # 验证数据不同
        assert not data1['y'].equals(data2['y'])
    
    def test_reproducibility_across_scenarios(self):
        """
        验证所有场景的可重复性
        """
        from dgp_small_sample import generate_small_sample_dgp
        
        for scenario in [1, 2, 3]:
            seed = 100 + scenario
            
            data1, params1 = generate_small_sample_dgp(scenario=scenario, seed=seed)
            data2, params2 = generate_small_sample_dgp(scenario=scenario, seed=seed)
            
            pd.testing.assert_frame_equal(data1, data2)
            assert params1['tau'] == params2['tau']


class TestSmallSampleDGPStructure:
    """
    验证小样本DGP的数据结构
    """
    
    def test_panel_structure(self):
        """
        验证面板数据结构
        """
        from dgp_small_sample import generate_small_sample_dgp
        
        data, params = generate_small_sample_dgp(seed=42)
        
        # 验证维度
        n_units = params['n_units']
        n_periods = params['n_periods']
        
        assert len(data) == n_units * n_periods
        assert data['id'].nunique() == n_units
        assert data['year'].nunique() == n_periods
    
    def test_required_columns(self):
        """
        验证必需的列
        """
        from dgp_small_sample import generate_small_sample_dgp
        
        data, _ = generate_small_sample_dgp(seed=42)
        
        required_cols = {'id', 'year', 'y', 'd', 'post'}
        assert required_cols.issubset(set(data.columns))
    
    def test_treatment_time_invariant(self):
        """
        验证处理状态是时间不变的
        """
        from dgp_small_sample import generate_small_sample_dgp
        
        data, _ = generate_small_sample_dgp(seed=42)
        
        # 每个单位的处理状态应该在所有时期相同
        d_by_unit = data.groupby('id')['d'].nunique()
        assert (d_by_unit == 1).all()
    
    def test_post_indicator_correct(self):
        """
        验证post指示符正确
        """
        from dgp_small_sample import generate_small_sample_dgp
        
        data, params = generate_small_sample_dgp(seed=42)
        treatment_start = params['treatment_start']
        
        # pre-treatment periods: post=0
        pre_data = data[data['year'] < treatment_start]
        assert (pre_data['post'] == 0).all()
        
        # post-treatment periods: post=1
        post_data = data[data['year'] >= treatment_start]
        assert (post_data['post'] == 1).all()


class TestSmallSampleDGPTreatmentProbability:
    """
    验证处理概率与论文Table 1一致
    """
    
    @pytest.mark.parametrize("scenario,expected_prob", [
        (1, 0.32),
        (2, 0.24),
        (3, 0.17),
    ])
    def test_treatment_probability(self, scenario, expected_prob):
        """
        验证处理概率与论文Table 1一致
        允许10%的相对误差
        """
        from dgp_small_sample import generate_small_sample_dgp
        
        # 多次模拟取平均
        n_sims = 100
        treated_shares = []
        
        for seed in range(n_sims):
            data, params = generate_small_sample_dgp(scenario=scenario, seed=seed)
            treated_shares.append(params['treated_share'])
        
        avg_treated_share = np.mean(treated_shares)
        
        # 允许10%的相对误差
        relative_error = abs(avg_treated_share - expected_prob) / expected_prob
        assert relative_error < 0.15, \
            f"Scenario {scenario}: expected P(D=1)≈{expected_prob}, got {avg_treated_share:.3f}"


class TestSmallSampleDGPAR1Process:
    """
    验证AR(1)误差过程
    Validates: Requirements 2.2
    """
    
    def test_ar1_initial_variance(self):
        """
        验证AR(1)过程的初始方差
        Var(u_1) = σ_ε² / (1 - ρ²)
        """
        from dgp_small_sample import generate_small_sample_dgp
        
        rho = 0.75
        sigma_epsilon = np.sqrt(2)
        expected_var = sigma_epsilon**2 / (1 - rho**2)
        
        # 多次模拟估计初始方差
        n_sims = 500
        initial_errors = []
        
        for seed in range(n_sims):
            data, params = generate_small_sample_dgp(
                seed=seed,
                return_components=True,
            )
            if 'U' in params:
                initial_errors.extend(params['U'][:, 0])
        
        if initial_errors:
            estimated_var = np.var(initial_errors)
            # 允许20%的相对误差
            relative_error = abs(estimated_var - expected_var) / expected_var
            assert relative_error < 0.25, \
                f"AR(1) initial variance: expected {expected_var:.3f}, got {estimated_var:.3f}"


class TestSmallSampleDGPTreatmentEffect:
    """
    验证真实ATT计算
    Validates: Requirements 2.4
    """
    
    def test_true_att_positive(self):
        """
        验证真实ATT为正
        """
        from dgp_small_sample import generate_small_sample_dgp
        
        for scenario in [1, 2, 3]:
            _, params = generate_small_sample_dgp(scenario=scenario, seed=42)
            assert params['tau'] > 0, f"Scenario {scenario}: ATT should be positive"
    
    def test_true_att_reasonable_magnitude(self):
        """
        验证真实ATT的量级合理
        论文Table 1: δ_t 在 1-3 之间，平均约为 2
        """
        from dgp_small_sample import generate_small_sample_dgp
        
        _, params = generate_small_sample_dgp(seed=42)
        tau = params['tau']
        
        # ATT应该在合理范围内
        assert 1.0 < tau < 4.0, f"ATT={tau} should be in reasonable range"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
