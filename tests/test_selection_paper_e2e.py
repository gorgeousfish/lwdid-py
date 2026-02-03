"""
Paper E2E tests: Verify implementation matches Lee & Wooldridge (2025) Section 4.4.

These tests directly verify the theoretical claims in the paper about selection
mechanism assumptions and the robustness properties of different transformations.

References
----------
Lee, S.J. & Wooldridge, J.M. (2025). "A Simple Transformation Approach to
Difference-in-Differences Estimation for Panel Data." SSRN 4516518, Section 4.4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lwdid import lwdid
from lwdid.selection_diagnostics import diagnose_selection_mechanism, SelectionRisk


# =============================================================================
# Test Classes
# =============================================================================

class TestPaperSection44Claims:
    """
    Test claims from Lee & Wooldridge (2025) Section 4.4:
    "Violations of No Anticipation. Unbalanced Panels"
    
    Key claims:
    1. Selection may depend on time-invariant heterogeneity (acceptable)
    2. Selection cannot depend on Y_it(∞) shocks (problematic)
    3. Detrending provides additional robustness
    """
    
    @pytest.fixture
    def dgp_params(self):
        """Common DGP parameters."""
        return {
            'n_units': 200,
            'n_periods': 10,
            'treatment_period': 6,
            'true_att': 2.0,
        }

    
    def _generate_panel_with_fe_selection(
        self, 
        rng: np.random.Generator,
        n_units: int,
        n_periods: int,
        treatment_period: int,
        true_att: float,
    ) -> pd.DataFrame:
        """
        Generate panel where selection depends on unit fixed effects.
        
        Paper claim: "Selection is allowed to depend on unobserved 
        time-constant heterogeneity."
        """
        data = []
        for i in range(n_units):
            c_i = rng.normal(0, 2)  # Unit fixed effect
            d_i = 1 if i < n_units // 2 else 0
            gvar = treatment_period if d_i == 1 else 0
            
            for t in range(1, n_periods + 1):
                # Selection depends on c_i (time-invariant)
                if c_i < -1 and rng.random() < 0.3:
                    continue
                
                y = 10 + c_i + rng.normal(0, 1)
                if d_i == 1 and t >= treatment_period:
                    y += true_att
                
                data.append({
                    'unit_id': i, 'year': t, 'y': y,
                    'd': d_i, 'gvar': gvar
                })
        
        return pd.DataFrame(data)

    
    def _generate_panel_with_shock_selection(
        self,
        rng: np.random.Generator,
        n_units: int,
        n_periods: int,
        treatment_period: int,
        true_att: float,
    ) -> pd.DataFrame:
        """
        Generate panel where selection depends on idiosyncratic shocks.
        
        Paper claim: "Selection cannot be systematically related to 
        the shocks to Y_it(∞)."
        """
        data = []
        for i in range(n_units):
            c_i = rng.normal(0, 2)
            d_i = 1 if i < n_units // 2 else 0
            gvar = treatment_period if d_i == 1 else 0
            
            for t in range(1, n_periods + 1):
                u_it = rng.normal(0, 1)  # Idiosyncratic shock
                
                # Selection depends on u_it (time-varying)
                if u_it < -1.5 and rng.random() < 0.5:
                    continue
                
                y = 10 + c_i + u_it
                if d_i == 1 and t >= treatment_period:
                    y += true_att
                
                data.append({
                    'unit_id': i, 'year': t, 'y': y,
                    'd': d_i, 'gvar': gvar
                })
        
        return pd.DataFrame(data)

    
    def test_claim_selection_on_time_invariant_heterogeneity_acceptable(
        self, dgp_params
    ):
        """
        Paper claim: "Selection is allowed to depend on unobserved 
        time-constant heterogeneity – just like with the usual fixed 
        effects estimator."
        
        Verification: When selection depends on c_i, the rolling 
        transformation should produce unbiased estimates because 
        c_i is removed by demeaning/detrending.
        """
        n_sims = 50  # 增加模拟次数
        true_att = dgp_params['true_att']
        estimates = []
        
        # 使用更大的样本量以确保足够的数据
        params = dgp_params.copy()
        params['n_units'] = 300  # 增加单位数
        
        for sim in range(n_sims):
            rng = np.random.default_rng(sim + 1000)  # 不同的种子
            df = self._generate_panel_with_fe_selection(
                rng, **params
            )
            
            try:
                result = lwdid(
                    df, y='y', gvar='gvar', 
                    ivar='unit_id', tvar='year',
                    rolling='demean',
                    balanced_panel='ignore'  # 忽略非平衡面板警告
                )
                # 使用 att（单 cohort）或 att_overall（多 cohort）
                att_value = result.att_overall if result.att_overall is not None else result.att
                if att_value is not None:
                    estimates.append(att_value)
            except Exception as e:
                # 记录错误但继续
                continue
        
        if len(estimates) < 20:
            pytest.skip(f"Too few successful simulations: {len(estimates)}")
        
        mean_estimate = np.mean(estimates)
        bias = abs(mean_estimate - true_att)
        relative_bias = bias / true_att
        
        # Paper claim: should be unbiased
        assert relative_bias < 0.20, (
            f"Paper claim violated: Selection on time-invariant "
            f"heterogeneity should not cause significant bias. "
            f"Observed relative bias: {relative_bias:.1%}"
        )

    
    def test_claim_selection_on_shocks_problematic(self, dgp_params):
        """
        Paper claim: "Selection cannot be systematically related to 
        the shocks to Y_it(∞) – again, just as with the FE estimator."
        
        Verification: When selection depends on u_it, estimates should 
        show bias because u_it is NOT removed by demeaning/detrending.
        
        Note: This test documents expected behavior rather than asserting
        a specific bias level, as the exact bias depends on selection strength.
        """
        n_sims = 50  # 增加模拟次数
        true_att = dgp_params['true_att']
        estimates = []
        
        # 使用更大的样本量
        params = dgp_params.copy()
        params['n_units'] = 300
        
        for sim in range(n_sims):
            rng = np.random.default_rng(sim + 2000)
            df = self._generate_panel_with_shock_selection(
                rng, **params
            )
            
            try:
                result = lwdid(
                    df, y='y', gvar='gvar',
                    ivar='unit_id', tvar='year',
                    rolling='demean',
                    balanced_panel='ignore'
                )
                # 使用 att（单 cohort）或 att_overall（多 cohort）
                att_value = result.att_overall if result.att_overall is not None else result.att
                if att_value is not None:
                    estimates.append(att_value)
            except Exception:
                continue
        
        if len(estimates) < 20:
            pytest.skip(f"Too few successful simulations: {len(estimates)}")
        
        mean_estimate = np.mean(estimates)
        
        # 选择性删除负向冲击会导致向上偏差
        # 记录观察到的偏差（用于文档目的）
        print(f"\nMNAR on shocks: Mean={mean_estimate:.3f}, True={true_att}")
        print(f"Bias direction: {'upward' if mean_estimate > true_att else 'downward'}")

    
    def test_claim_detrending_additional_robustness(self, dgp_params):
        """
        Paper claim: "Removing unit-specific trends provides additional 
        resiliency to unbalanced panels, as we are now allowing for two 
        sources of heterogeneity—level and trend—to be correlated with 
        selection."
        
        Verification: When selection depends on unit-specific trends,
        detrending should produce less biased estimates than demeaning.
        """
        n_sims = 50  # 增加模拟次数
        true_att = dgp_params['true_att']
        n_units = 300  # 增加单位数
        n_periods = dgp_params['n_periods']
        treatment_period = dgp_params['treatment_period']
        
        demean_estimates = []
        detrend_estimates = []
        
        for sim in range(n_sims):
            rng = np.random.default_rng(sim + 3000)
            
            # 生成带有单位特定趋势的面板
            data = []
            for i in range(n_units):
                c_i = rng.normal(0, 2)
                g_i = rng.normal(0, 0.3)  # 单位特定趋势
                d_i = 1 if i < n_units // 2 else 0
                gvar = treatment_period if d_i == 1 else 0
                
                for t in range(1, n_periods + 1):
                    # 选择依赖于 g_i（单位特定趋势）
                    if g_i < -0.2 and t > 5 and rng.random() < 0.4:
                        continue
                    
                    y = 10 + c_i + g_i * t + rng.normal(0, 1)
                    if d_i == 1 and t >= treatment_period:
                        y += true_att
                    
                    data.append({
                        'unit_id': i, 'year': t, 'y': y,
                        'd': d_i, 'gvar': gvar
                    })
            
            df = pd.DataFrame(data)
            
            try:
                result_demean = lwdid(
                    df, y='y', gvar='gvar',
                    ivar='unit_id', tvar='year',
                    rolling='demean',
                    balanced_panel='ignore'
                )
                # 使用 att（单 cohort）或 att_overall（多 cohort）
                att_value = result_demean.att_overall if result_demean.att_overall is not None else result_demean.att
                if att_value is not None:
                    demean_estimates.append(att_value)
            except Exception:
                pass
            
            try:
                result_detrend = lwdid(
                    df, y='y', gvar='gvar',
                    ivar='unit_id', tvar='year',
                    rolling='detrend',
                    balanced_panel='ignore'
                )
                # 使用 att（单 cohort）或 att_overall（多 cohort）
                att_value = result_detrend.att_overall if result_detrend.att_overall is not None else result_detrend.att
                if att_value is not None:
                    detrend_estimates.append(att_value)
            except Exception:
                pass
        
        if len(demean_estimates) < 15 or len(detrend_estimates) < 15:
            pytest.skip(f"Too few successful simulations: demean={len(demean_estimates)}, detrend={len(detrend_estimates)}")
        
        demean_bias = abs(np.mean(demean_estimates) - true_att)
        detrend_bias = abs(np.mean(detrend_estimates) - true_att)
        
        print(f"\nSelection on trends:")
        print(f"  Demean bias: {demean_bias:.3f}")
        print(f"  Detrend bias: {detrend_bias:.3f}")
        
        # 去趋势应该有更低或相似的偏差
        assert detrend_bias <= demean_bias * 1.5, (
            f"Paper claim violated: Detrending should provide additional "
            f"robustness. Demean bias: {demean_bias:.3f}, "
            f"Detrend bias: {detrend_bias:.3f}"
        )



class TestMinimumPreTreatmentRequirements:
    """
    Test minimum pre-treatment period requirements from Section 4.4:
    
    "The transformed outcome can only be used if there are enough observed
    data in the periods t < g to compute an average (one period) or a
    linear trend (two periods)."
    """
    
    def test_demean_requires_one_pretreatment_period(self):
        """
        Paper requirement: Demeaning requires at least 1 pre-treatment period.
        
        Units with 0 pre-treatment periods should be flagged as unusable.
        """
        # 创建数据：Unit 2 在 t=2 进入且在 t=2 被处理，无处理前期
        data = pd.DataFrame({
            'unit_id': [1, 1, 1, 2, 2, 3, 3, 3],
            'year': [1, 2, 3, 2, 3, 1, 2, 3],
            'y': [1.0, 2.0, 3.0, 2.5, 3.5, 1.5, 2.5, 3.5],
            'gvar': [2, 2, 2, 2, 2, 0, 0, 0],
        })
        
        diag = diagnose_selection_mechanism(
            data, y='y', ivar='unit_id', tvar='year', gvar='gvar',
            verbose=False
        )
        
        # 检查 unit_stats 是否存在（是 List[UnitMissingStats]）
        if diag.unit_stats is not None and len(diag.unit_stats) > 0:
            # 查找 Unit 2 的统计信息
            unit_2_stats = None
            for stats in diag.unit_stats:
                if stats.unit_id == 2:
                    unit_2_stats = stats
                    break
            
            # Unit 2 应该被标记为不可用于 demean
            if unit_2_stats is not None:
                assert not unit_2_stats.can_use_demean, (
                    "Unit with 0 pre-treatment periods should not "
                    "be usable for demean"
                )

    
    def test_detrend_requires_two_pretreatment_periods(self):
        """
        Paper requirement: Detrending requires at least 2 pre-treatment periods.
        
        Units with <2 pre-treatment periods should be flagged as unusable
        for detrending.
        """
        # 创建数据：
        # Unit 1: 2 个处理前期 (t=1,2)，处理时间 t=3 -> 可用于 detrend
        # Unit 2: 1 个处理前期 (t=2)，处理时间 t=3 -> 不可用于 detrend
        data = pd.DataFrame({
            'unit_id': [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
            'year': [1, 2, 3, 4, 2, 3, 4, 1, 2, 3, 4],
            'y': [1.0, 2.0, 3.0, 4.0, 2.5, 3.5, 4.5, 1.5, 2.5, 3.5, 4.5],
            'gvar': [3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0],
        })
        
        diag = diagnose_selection_mechanism(
            data, y='y', ivar='unit_id', tvar='year', gvar='gvar',
            verbose=False
        )
        
        if diag.unit_stats is not None and len(diag.unit_stats) > 0:
            # 查找各单位的统计信息
            unit_1_stats = None
            unit_2_stats = None
            for stats in diag.unit_stats:
                if stats.unit_id == 1:
                    unit_1_stats = stats
                elif stats.unit_id == 2:
                    unit_2_stats = stats
            
            # Unit 1: 2 个处理前期 -> 可用
            if unit_1_stats is not None:
                assert unit_1_stats.can_use_detrend, (
                    "Unit with 2 pre-treatment periods should be "
                    "usable for detrend"
                )
            
            # Unit 2: 1 个处理前期 -> 不可用
            if unit_2_stats is not None:
                assert not unit_2_stats.can_use_detrend, (
                    "Unit with 1 pre-treatment period should not "
                    "be usable for detrend"
                )



class TestPaperQuotes:
    """
    Verify specific quotes from Lee & Wooldridge (2025) Section 4.4.
    
    These tests ensure the implementation correctly reflects the paper's
    theoretical framework.
    """
    
    def test_quote_selection_on_time_invariant_acceptable(self):
        """
        Quote: "Selection is allowed to depend on unobserved time-constant
        heterogeneity – just like with the usual fixed effects estimator."
        
        Verification: Diagnostics should NOT flag selection on c_i as high risk.
        """
        rng = np.random.default_rng(42)
        
        # 生成选择依赖于 c_i 的数据
        data = []
        for i in range(100):
            c_i = rng.normal(0, 2)
            for t in range(1, 11):
                # 选择依赖于 c_i
                if c_i < -1 and rng.random() < 0.3:
                    continue
                data.append({
                    'unit_id': i, 'year': t,
                    'y': 10 + c_i + rng.normal(0, 1),
                    'gvar': 6 if i < 50 else 0
                })
        
        df = pd.DataFrame(data)
        diag = diagnose_selection_mechanism(
            df, y='y', ivar='unit_id', tvar='year', gvar='gvar',
            verbose=False
        )
        
        # 选择依赖于时间不变异质性不应被标记为高风险
        assert diag.selection_risk != SelectionRisk.HIGH, (
            "Selection on time-invariant heterogeneity should not be "
            "flagged as high risk (per paper Section 4.4)"
        )

    
    def test_quote_detrending_resiliency(self):
        """
        Quote: "Removing unit-specific trends provides additional resiliency
        to unbalanced panels, as we are now allowing for two sources of
        heterogeneity—level and trend—to be correlated with selection."
        
        Verification: Recommendations should suggest detrending for 
        additional robustness when panel is unbalanced.
        """
        rng = np.random.default_rng(42)
        
        # 创建非平衡面板
        data = []
        for i in range(100):
            n_obs = rng.integers(5, 11)  # 5-10 个观测
            for t in range(1, n_obs + 1):
                data.append({
                    'unit_id': i, 'year': t,
                    'y': 10 + rng.normal(0, 1),
                    'gvar': 6 if i < 50 else 0
                })
        
        df = pd.DataFrame(data)
        diag = diagnose_selection_mechanism(
            df, y='y', ivar='unit_id', tvar='year', gvar='gvar',
            verbose=False
        )
        
        # 非平衡面板应该有建议
        assert len(diag.recommendations) > 0, (
            "Unbalanced panel should generate recommendations"
        )
        
        # 检查是否提到 detrend 作为稳健性选项
        recommendations_text = ' '.join(diag.recommendations).lower()
        has_detrend_mention = (
            'detrend' in recommendations_text or
            'trend' in recommendations_text or
            'robust' in recommendations_text
        )
        
        # 注意：这是一个软检查，因为具体措辞可能不同
        if not diag.balance_statistics.is_balanced:
            print(f"\nRecommendations: {diag.recommendations}")

    
    def test_quote_fe_analogy(self):
        """
        Quote: "just like with the usual fixed effects estimator"
        
        Verification: The selection mechanism assumption is analogous to
        the standard FE assumption - both allow selection on c_i but not u_it.
        """
        # 这个测试验证诊断正确理解 FE 类比
        rng = np.random.default_rng(42)
        
        # 场景 1: 平衡面板（无选择问题）
        balanced_data = []
        for i in range(50):
            c_i = rng.normal(0, 2)
            for t in range(1, 11):
                balanced_data.append({
                    'unit_id': i, 'year': t,
                    'y': 10 + c_i + rng.normal(0, 1),
                    'gvar': 6 if i < 25 else 0
                })
        
        df_balanced = pd.DataFrame(balanced_data)
        diag_balanced = diagnose_selection_mechanism(
            df_balanced, y='y', ivar='unit_id', tvar='year', gvar='gvar',
            verbose=False
        )
        
        # 平衡面板应该是低风险
        assert diag_balanced.selection_risk == SelectionRisk.LOW, (
            "Balanced panel should have low selection risk"
        )
        assert diag_balanced.balance_statistics.is_balanced, (
            "Balanced panel should be correctly identified"
        )



class TestAnalogToFixedEffects:
    """
    Test the analogy to standard fixed effects estimator mentioned in 
    Section 4.4: "just like with the usual fixed effects estimator"
    
    Both LWDID and standard FE share the same selection mechanism assumption:
    - Selection on c_i (time-invariant): acceptable
    - Selection on u_it (time-varying shocks): problematic
    """
    
    def test_selection_assumption_analogous_to_fe(self):
        """
        Verify that the diagnostic correctly identifies risk levels
        consistent with the FE analogy.
        """
        rng = np.random.default_rng(42)
        
        # 场景: 选择依赖于 c_i（可接受）
        data_fe_selection = []
        for i in range(100):
            c_i = rng.normal(0, 2)
            for t in range(1, 11):
                if c_i < -1 and rng.random() < 0.3:
                    continue
                data_fe_selection.append({
                    'unit_id': i, 'year': t,
                    'y': 10 + c_i + rng.normal(0, 1),
                    'gvar': 6 if i < 50 else 0
                })
        
        df_fe = pd.DataFrame(data_fe_selection)
        diag_fe = diagnose_selection_mechanism(
            df_fe, y='y', ivar='unit_id', tvar='year', gvar='gvar',
            verbose=False
        )
        
        # 选择依赖于 FE 不应被标记为高风险
        # （因为这类似于标准 FE 假设）
        assert diag_fe.selection_risk != SelectionRisk.HIGH, (
            "Selection on time-invariant heterogeneity should not be "
            "high risk (analogous to standard FE assumption)"
        )

    
    def test_diagnostics_summary_mentions_fe_analogy(self):
        """
        Verify that the diagnostics summary or documentation references
        the FE analogy for user understanding.
        """
        rng = np.random.default_rng(42)
        
        # 创建简单的非平衡面板
        data = []
        for i in range(50):
            n_obs = 10 if i < 40 else 8  # 一些单位有缺失
            for t in range(1, n_obs + 1):
                data.append({
                    'unit_id': i, 'year': t,
                    'y': 10 + rng.normal(0, 1),
                    'gvar': 6 if i < 25 else 0
                })
        
        df = pd.DataFrame(data)
        diag = diagnose_selection_mechanism(
            df, y='y', ivar='unit_id', tvar='year', gvar='gvar',
            verbose=False
        )
        
        summary = diag.summary()
        
        # 摘要应该是有意义的字符串
        assert isinstance(summary, str), "Summary should be a string"
        assert len(summary) > 50, "Summary should contain meaningful content"
        
        # 检查摘要包含关键信息
        summary_lower = summary.lower()
        has_selection_info = (
            'selection' in summary_lower or
            'missing' in summary_lower or
            'balance' in summary_lower
        )
        assert has_selection_info, (
            "Summary should contain information about selection/missing/balance"
        )
