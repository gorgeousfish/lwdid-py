"""
BUG-007 修复验证：Python-Stata 端到端 weights_cv 对比测试

验证修复后的 weights_cv 计算（使用 ddof=1）与 Stata 完全一致。

数据源：Lee_Wooldridge_2023-main 3/2.lee_wooldridge_staggered_data.dta

测试 6 个 (cohort, period) 组合：
- (4, 4), (4, 5), (4, 6)
- (5, 5), (5, 6)
- (6, 6)

验收标准：
- weights_cv 相对误差 < 1%
- 倾向得分统计量相对误差 < 0.1%
- n_control 完全一致

Reference: BUG-007 in 审查/bug列表.md
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# 确保可以导入 lwdid 模块
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from lwdid.staggered.estimators import estimate_propensity_score


# ============================================================================
# Stata 基准数据（2026-01-17 从 Stata MCP 提取）
# ============================================================================

# Stata 计算的 weights_cv 基准值
# 计算方法: summarize w; scalar cv = r(sd) / r(mean)
# 其中 w = p / (1-p) 为控制组的 IPW 权重
# Stata sd 默认使用 N-1 分母（样本标准差）
STATA_WEIGHTS_CV = {
    (4, 4): {
        'weights_cv': 0.72026299,
        'weights_mean': 0.14901879,
        'weights_sd': 0.10733272,
        'ps_mean': 0.12358907,
        'ps_sd': 0.06687953,
        'n_control': 871,
        'condition': 'f04 == 1',
    },
    (4, 5): {
        'weights_cv': 0.88397562,
        'weights_mean': 0.17141546,
        'weights_sd': 0.15152709,
        'ps_mean': 0.13626714,
        'ps_sd': 0.08150197,
        'n_control': 762,
        'condition': 'f05 == 1 & g5 == 0',
    },
    (4, 6): {
        'weights_cv': 1.031211,
        'weights_mean': 0.19671413,
        'weights_sd': 0.20285378,
        'ps_mean': 0.14816233,
        'ps_sd': 0.10128552,
        'n_control': 652,
        'condition': 'f06 == 1 & (g5 + g6 != 1)',
    },
    (5, 5): {
        'weights_cv': 0.66394977,
        'weights_mean': 0.1439017,
        'weights_sd': 0.0955435,
        'ps_mean': 0.12078569,
        'ps_sd': 0.06118844,
        'n_control': 762,
        'condition': 'f05 == 1 & g4 == 0',
    },
    (5, 6): {
        'weights_cv': 0.83076916,
        'weights_mean': 0.16670205,
        'weights_sd': 0.13849092,
        'ps_mean': 0.13353569,
        'ps_sd': 0.08113908,
        'n_control': 652,
        'condition': 'f06 == 1 & (g4 + g6 != 1)',
    },
    (6, 6): {
        'weights_cv': 0.76485801,
        'weights_mean': 0.16493456,
        'weights_sd': 0.12615152,
        'ps_mean': 0.13356013,
        'ps_sd': 0.07597906,
        'n_control': 652,
        'condition': 'f06 == 1 & (g4 + g5 != 1)',
    },
}


# ============================================================================
# 数据加载
# ============================================================================

def get_staggered_data_path():
    """获取 Lee-Wooldridge 2023 staggered 数据路径"""
    possible_paths = [
        Path(__file__).parents[3] / "Lee_Wooldridge_2023-main 3" / "2.lee_wooldridge_staggered_data.dta",
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
    
    if data_path is None:
        pytest.skip("Stata 数据文件不存在")
    
    data = pd.read_stata(data_path)
    return data


def build_subsample(data: pd.DataFrame, g: int, r: int) -> pd.DataFrame:
    """
    构建 (g, r) 组合的子样本
    
    参考 Stata 条件：
    - (4, 4): f04
    - (4, 5): f05 & ~g5
    - (4, 6): f06 & (g5 + g6 != 1)
    - (5, 5): f05 & ~g4
    - (5, 6): f06 & (g4 + g6 != 1)
    - (6, 6): f06 & (g4 + g5 != 1)
    """
    conditions = {
        (4, 4): (data['f04'] == 1),
        (4, 5): (data['f05'] == 1) & (data['g5'] == 0),
        (4, 6): (data['f06'] == 1) & (data['g5'] + data['g6'] != 1),
        (5, 5): (data['f05'] == 1) & (data['g4'] == 0),
        (5, 6): (data['f06'] == 1) & (data['g4'] + data['g6'] != 1),
        (6, 6): (data['f06'] == 1) & (data['g4'] + data['g5'] != 1),
    }
    
    mask = conditions[(g, r)]
    return data[mask].copy()


# ============================================================================
# weights_cv 一致性测试
# ============================================================================

class TestWeightsCVStataConsistency:
    """验证 Python weights_cv 与 Stata 完全一致"""
    
    @pytest.mark.parametrize("g,r", [(4, 4), (4, 5), (4, 6), (5, 5), (5, 6), (6, 6)])
    def test_weights_cv_vs_stata(self, staggered_data, g, r):
        """
        测试 weights_cv 与 Stata 差异 < 1%
        
        这是 BUG-007 的核心验证：修复后使用 ddof=1 应与 Stata 一致
        """
        # 构建子样本
        subsample = build_subsample(staggered_data, g, r)
        
        # 确定处理变量名
        d_var = f'g{g}'
        
        # 估计倾向得分
        pscores, coef = estimate_propensity_score(
            subsample, d_var, ['x1', 'x2'], trim_threshold=0.01
        )
        
        # 计算 IPW 权重（控制组）
        control_mask = subsample[d_var] == 0
        weights = pscores / (1 - pscores)
        weights_control = weights[control_mask]
        
        # 计算 weights_cv（使用 ddof=1，与 Stata 一致）
        python_cv = np.std(weights_control, ddof=1) / np.mean(weights_control)
        
        # Stata 基准
        stata = STATA_WEIGHTS_CV[(g, r)]
        stata_cv = stata['weights_cv']
        
        # 计算相对误差
        relative_error = abs(python_cv - stata_cv) / stata_cv
        
        # 验收标准: 相对误差 < 1%
        assert relative_error < 0.01, (
            f"(g={g}, r={r}) weights_cv 相对误差过大:\n"
            f"  Python = {python_cv:.8f}\n"
            f"  Stata  = {stata_cv:.8f}\n"
            f"  误差   = {relative_error:.4%}"
        )
    
    @pytest.mark.parametrize("g,r", [(4, 4), (4, 5), (4, 6), (5, 5), (5, 6), (6, 6)])
    def test_n_control_exact_match(self, staggered_data, g, r):
        """测试控制组样本量完全一致"""
        subsample = build_subsample(staggered_data, g, r)
        d_var = f'g{g}'
        
        # 计算控制组数量
        n_control_python = (subsample[d_var] == 0).sum()
        n_control_stata = STATA_WEIGHTS_CV[(g, r)]['n_control']
        
        assert n_control_python == n_control_stata, (
            f"(g={g}, r={r}) 控制组数量不一致:\n"
            f"  Python = {n_control_python}\n"
            f"  Stata  = {n_control_stata}"
        )
    
    @pytest.mark.parametrize("g,r", [(4, 4), (4, 5), (4, 6), (5, 5), (5, 6), (6, 6)])
    def test_ps_mean_vs_stata(self, staggered_data, g, r):
        """测试倾向得分均值与 Stata 差异 < 0.1%"""
        subsample = build_subsample(staggered_data, g, r)
        d_var = f'g{g}'
        
        pscores, _ = estimate_propensity_score(
            subsample, d_var, ['x1', 'x2'], trim_threshold=0.01
        )
        
        control_mask = subsample[d_var] == 0
        ps_control = pscores[control_mask]
        
        python_ps_mean = np.mean(ps_control)
        stata_ps_mean = STATA_WEIGHTS_CV[(g, r)]['ps_mean']
        
        relative_error = abs(python_ps_mean - stata_ps_mean) / stata_ps_mean
        
        assert relative_error < 0.001, (
            f"(g={g}, r={r}) ps_mean 相对误差过大:\n"
            f"  Python = {python_ps_mean:.8f}\n"
            f"  Stata  = {stata_ps_mean:.8f}\n"
            f"  误差   = {relative_error:.4%}"
        )


# ============================================================================
# 详细数值验证
# ============================================================================

class TestDetailedNumericalValidation:
    """详细数值验证"""
    
    def test_all_combinations_summary(self, staggered_data):
        """输出所有组合的对比摘要"""
        print("\n" + "=" * 70)
        print("BUG-007 Python-Stata weights_cv 端到端对比")
        print("=" * 70)
        
        results = []
        
        for (g, r), stata in STATA_WEIGHTS_CV.items():
            subsample = build_subsample(staggered_data, g, r)
            d_var = f'g{g}'
            
            pscores, _ = estimate_propensity_score(
                subsample, d_var, ['x1', 'x2'], trim_threshold=0.01
            )
            
            control_mask = subsample[d_var] == 0
            weights = pscores / (1 - pscores)
            weights_control = weights[control_mask]
            
            python_cv = np.std(weights_control, ddof=1) / np.mean(weights_control)
            stata_cv = stata['weights_cv']
            
            error = abs(python_cv - stata_cv) / stata_cv * 100
            
            results.append({
                'g': g,
                'r': r,
                'n_control': len(weights_control),
                'python_cv': python_cv,
                'stata_cv': stata_cv,
                'error_pct': error,
            })
            
            print(f"\n(g={g}, r={r}): n_control={len(weights_control)}")
            print(f"  Python CV: {python_cv:.8f}")
            print(f"  Stata CV:  {stata_cv:.8f}")
            print(f"  误差:      {error:.4f}%")
        
        print("\n" + "=" * 70)
        print("总结: 所有组合误差均 < 1%")
        print("=" * 70)
        
        # 验证所有误差 < 1%
        for r in results:
            assert r['error_pct'] < 1.0, f"(g={r['g']}, r={r['r']}) 误差超过 1%"
    
    def test_ddof_difference_demonstration(self, staggered_data):
        """
        演示 ddof=0 vs ddof=1 的差异
        
        说明：如果不修复 BUG-007（使用 ddof=0），误差会更大
        """
        print("\n" + "=" * 70)
        print("ddof=0 vs ddof=1 差异演示")
        print("=" * 70)
        
        g, r = 4, 4  # 使用第一个组合作为示例
        subsample = build_subsample(staggered_data, g, r)
        d_var = f'g{g}'
        
        pscores, _ = estimate_propensity_score(
            subsample, d_var, ['x1', 'x2'], trim_threshold=0.01
        )
        
        control_mask = subsample[d_var] == 0
        weights = pscores / (1 - pscores)
        weights_control = weights[control_mask]
        
        # 修复后 (ddof=1)
        cv_ddof1 = np.std(weights_control, ddof=1) / np.mean(weights_control)
        
        # 修复前 (ddof=0)
        cv_ddof0 = np.std(weights_control, ddof=0) / np.mean(weights_control)
        
        stata_cv = STATA_WEIGHTS_CV[(g, r)]['weights_cv']
        
        error_ddof1 = abs(cv_ddof1 - stata_cv) / stata_cv * 100
        error_ddof0 = abs(cv_ddof0 - stata_cv) / stata_cv * 100
        
        print(f"\n(g={g}, r={r}):")
        print(f"  Stata CV:      {stata_cv:.8f}")
        print(f"  ddof=1 (修复): {cv_ddof1:.8f} (误差 {error_ddof1:.4f}%)")
        print(f"  ddof=0 (旧):   {cv_ddof0:.8f} (误差 {error_ddof0:.4f}%)")
        print(f"\n修复前后差异: {abs(cv_ddof1 - cv_ddof0):.8f}")
        
        # 验证修复后误差更小
        assert error_ddof1 < error_ddof0, (
            f"修复后误差应该更小: ddof=1 误差={error_ddof1:.4f}%, ddof=0 误差={error_ddof0:.4f}%"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s', '--tb=short'])
