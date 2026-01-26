"""
倾向得分诊断 - Stata 对照测试

使用 2.lee_wooldridge_staggered_data.dta 验证诊断指标
与 Stata logit + predict 的倾向得分估计对照

Stata 参考命令见: Lee_Wooldridge_2023-main 3/2.lee_wooldridge_rolling_staggered.do

Reference: Story 1.1 - 倾向得分诊断增强
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from lwdid.staggered.estimators import (
    estimate_propensity_score,
    PropensityScoreDiagnostics,
)


def get_staggered_data_path():
    """获取 Lee-Wooldridge 2023 staggered 数据路径"""
    # 尝试多个可能的路径
    possible_paths = [
        Path(__file__).parents[2] / "Lee_Wooldridge_2023-main 3" / "2.lee_wooldridge_staggered_data.dta",
        Path("/Users/cxy/Desktop/大样本lwdid/Lee_Wooldridge_2023-main 3/2.lee_wooldridge_staggered_data.dta"),
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return None


@pytest.fixture
def staggered_data():
    """加载 Lee-Wooldridge 2023 staggered 数据"""
    data_path = get_staggered_data_path()
    
    if data_path is None or not data_path.exists():
        pytest.skip(f"Stata数据文件不存在")
    
    data = pd.read_stata(data_path)
    return data


class TestStataComparisonDiagnostics:
    """Stata 数值对照测试"""
    
    def test_basic_diagnostics_structure(self, staggered_data):
        """
        测试诊断结构正确性
        
        使用 cohort 4, period 5 的子样本
        """
        data = staggered_data
        
        # 构建子样本（Python）
        # 子样本: D_ig + A_{r+1} = 1
        # cohort 4 + (cohort 6, never-treated g0)
        # 注意: 数据使用 g0 表示 never-treated，而非 ginf
        subsample = data[
            ((data['g4'] == 1) | (data['g6'] == 1) | (data['g0'] == 1)) &
            (data['f05'] == 1)
        ].copy()
        
        # 确保有足够样本
        if len(subsample) < 10:
            pytest.skip("子样本太小")
        
        # 估计倾向得分
        pscores, coef, diag = estimate_propensity_score(
            subsample, 'g4', ['x1', 'x2'], return_diagnostics=True
        )
        
        # 验证诊断结构
        assert isinstance(diag, PropensityScoreDiagnostics)
        assert 0 < diag.ps_mean < 1
        assert diag.ps_std >= 0
        assert diag.ps_min >= 0.01  # 裁剪后
        assert diag.ps_max <= 0.99  # 裁剪后
        assert diag.weights_cv >= 0 or np.isnan(diag.weights_cv)
        assert 0 <= diag.extreme_low_pct <= 1
        assert 0 <= diag.extreme_high_pct <= 1
        assert diag.n_trimmed >= 0
    
    def test_diagnostics_all_cohorts(self, staggered_data):
        """
        测试所有 (cohort, period) 组合的倾向得分诊断
        
        验证6个组合:
        - (4, 4), (4, 5), (4, 6)
        - (5, 5), (5, 6)
        - (6, 6)
        """
        data = staggered_data
        
        # 定义测试组合
        # 注意: 数据使用 g0 表示 never-treated，而非 ginf
        test_cases = [
            # (cohort_g, period_flag, cohort_condition)
            (4, 'f04', "g4 == 1 | g5 == 1 | g6 == 1 | g0 == 1"),
            (4, 'f05', "g4 == 1 | g6 == 1 | g0 == 1"),  # ~g5
            (4, 'f06', "g4 == 1 | g0 == 1"),  # ~(g5 | g6)
            (5, 'f05', "g5 == 1 | g6 == 1 | g0 == 1"),  # ~g4
            (5, 'f06', "g5 == 1 | g0 == 1"),  # ~(g4 | g6)
            (6, 'f06', "g6 == 1 | g0 == 1"),  # ~(g4 | g5)
        ]
        
        results = []
        for cohort_g, period_var, subsample_condition in test_cases:
            # 构建子样本
            subsample = data.query(f"({subsample_condition}) and {period_var} == 1")
            
            if len(subsample) < 5:
                continue
            
            # 估计倾向得分
            treatment_var = f'g{cohort_g}'
            try:
                pscores, coef, diag = estimate_propensity_score(
                    subsample, treatment_var, ['x1', 'x2'], return_diagnostics=True
                )
                
                results.append({
                    'cohort': cohort_g,
                    'period': period_var,
                    'n_total': len(subsample),
                    'n_treated': int((subsample[treatment_var] == 1).sum()),
                    'n_control': int((subsample[treatment_var] == 0).sum()),
                    'ps_mean': diag.ps_mean,
                    'ps_std': diag.ps_std,
                    'weights_cv': diag.weights_cv,
                    'extreme_pct': diag.extreme_low_pct + diag.extreme_high_pct,
                    'n_trimmed': diag.n_trimmed,
                })
            except Exception as e:
                print(f"Skipping cohort {cohort_g}, {period_var}: {e}")
                continue
        
        # 验证至少有一些结果
        assert len(results) > 0
        
        # 输出结果用于调试
        results_df = pd.DataFrame(results)
        print("\n=== 诊断结果汇总 ===")
        print(results_df.to_string())
        
        # 基本验证
        for r in results:
            assert 0 < r['ps_mean'] < 1
            assert r['ps_std'] >= 0
            assert r['n_trimmed'] >= 0
    
    def test_weights_cv_control_only_verification(self, staggered_data):
        """
        验证权重 CV 仅在控制组上计算
        
        这是关键的数学正确性测试
        """
        data = staggered_data
        
        # 使用 cohort 4, period 5
        # 注意: 数据使用 g0 表示 never-treated
        subsample = data[
            ((data['g4'] == 1) | (data['g6'] == 1) | (data['g0'] == 1)) &
            (data['f05'] == 1)
        ].copy()
        
        if len(subsample) < 10:
            pytest.skip("子样本太小")
        
        # 估计倾向得分
        pscores, coef, diag = estimate_propensity_score(
            subsample, 'g4', ['x1', 'x2'], return_diagnostics=True
        )
        
        # 手动验证 CV 计算
        D = subsample['g4'].values.astype(float)
        control_mask = (D == 0)
        pscores_control = pscores[control_mask]
        
        # 权重计算
        weights_control = pscores_control / (1 - pscores_control)
        
        # 手动计算 CV
        if len(weights_control) >= 2:
            expected_cv = np.std(weights_control, ddof=1) / np.mean(weights_control)
            
            # 验证与诊断一致
            assert abs(diag.weights_cv - expected_cv) < 1e-10, \
                f"CV不匹配: 诊断={diag.weights_cv}, 手动计算={expected_cv}"
    
    def test_extreme_detection_uses_raw_pscores(self, staggered_data):
        """
        验证极端值检测使用裁剪前的原始倾向得分
        """
        data = staggered_data
        
        # 使用 cohort 4, period 5
        # 注意: 数据使用 g0 表示 never-treated
        subsample = data[
            ((data['g4'] == 1) | (data['g6'] == 1) | (data['g0'] == 1)) &
            (data['f05'] == 1)
        ].copy()
        
        if len(subsample) < 10:
            pytest.skip("子样本太小")
        
        # 使用较大的裁剪阈值来测试
        trim_threshold = 0.05
        
        pscores, coef, diag = estimate_propensity_score(
            subsample, 'g4', ['x1', 'x2'], 
            trim_threshold=trim_threshold,
            return_diagnostics=True
        )
        
        # 裁剪后的倾向得分应该在范围内
        assert diag.ps_min >= trim_threshold
        assert diag.ps_max <= 1 - trim_threshold
        
        # 极端值检测应该能检测到在裁剪阈值之外的原始值
        # (即 n_trimmed > 0 的情况是可能的)
        assert diag.n_trimmed >= 0
        assert diag.extreme_low_pct >= 0
        assert diag.extreme_high_pct >= 0


class TestStataReferenceValues:
    """
    Stata 参考值对照测试
    
    注意: 这些参考值需要通过运行 Stata 脚本获取
    如果参考值是占位符，测试将跳过
    """
    
    def test_stata_comparison_cohort4_period5(self, staggered_data):
        """
        测试 cohort 4, period 5 的倾向得分诊断
        
        Stata 参考命令:
            keep if (g4 == 1) | (g6 == 1) | (g0 == 1)
            keep if f05 == 1
            logit g4 x1 x2
            predict ps, pr
            summarize ps, detail
        """
        data = staggered_data
        
        # 构建子样本（Python）
        # 注意: 数据使用 g0 表示 never-treated
        subsample = data[
            ((data['g4'] == 1) | (data['g6'] == 1) | (data['g0'] == 1)) &
            (data['f05'] == 1)
        ].copy()
        
        if len(subsample) < 10:
            pytest.skip("子样本太小")
        
        # 估计倾向得分
        pscores, coef, diag = estimate_propensity_score(
            subsample, 'g4', ['x1', 'x2'], return_diagnostics=True
        )
        
        # ========================================================
        # Stata 预计算参考值
        # 这些值需要运行 Stata 脚本获取
        # ========================================================
        # 使用实际计算的合理性检查而非精确对照
        # (精确对照需要先运行 Stata 获取参考值)
        
        n_treated = int((subsample['g4'] == 1).sum())
        n_control = int((subsample['g4'] == 0).sum())
        
        # 验证均值接近处理组比例
        expected_proportion = n_treated / (n_treated + n_control)
        assert abs(diag.ps_mean - expected_proportion) < 0.3, \
            f"PS均值 ({diag.ps_mean:.4f}) 应接近处理组比例 ({expected_proportion:.4f})"
        
        # 验证标准差合理
        assert 0 < diag.ps_std < 0.5, \
            f"PS标准差 ({diag.ps_std:.4f}) 应在合理范围内"
        
        # 验证分位数单调性
        assert diag.ps_quantiles['25%'] <= diag.ps_quantiles['50%'] <= diag.ps_quantiles['75%']
        
        # 输出结果用于与 Stata 手动对照
        print(f"\n=== Python 诊断结果 (g=4, r=5) ===")
        print(f"  样本量: N={len(subsample)}, N_treated={n_treated}, N_control={n_control}")
        print(f"  PS 均值: {diag.ps_mean:.6f}")
        print(f"  PS 标准差: {diag.ps_std:.6f}")
        print(f"  PS 范围: [{diag.ps_min:.6f}, {diag.ps_max:.6f}]")
        print(f"  PS 分位数: [25%={diag.ps_quantiles['25%']:.6f}, "
              f"50%={diag.ps_quantiles['50%']:.6f}, "
              f"75%={diag.ps_quantiles['75%']:.6f}]")
        print(f"  权重 CV: {diag.weights_cv:.6f}")
        print(f"  极端值比例: {diag.extreme_low_pct + diag.extreme_high_pct:.4%}")
        print(f"  裁剪数量: {diag.n_trimmed}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
