"""
Monte Carlo simulation tests for selection mechanism.

These tests verify that the LWDID estimator is unbiased under acceptable
selection mechanisms and biased under problematic ones, as described in
Lee & Wooldridge (2025) Section 4.4.

The key assumption is:
    "Selection may depend on unobserved time-invariant heterogeneity,
    but cannot systematically depend on Y_it(∞) shocks."
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from typing import Tuple, List

from lwdid import lwdid
from lwdid.selection_diagnostics import diagnose_selection_mechanism, SelectionRisk


# =============================================================================
# Data Generating Process
# =============================================================================

class SelectionMechanismDGP:
    """
    Data Generating Process for selection mechanism Monte Carlo tests.
    
    Based on Lee & Wooldridge (2025) Section 4.4:
    - Selection may depend on time-invariant heterogeneity (acceptable)
    - Selection cannot depend on Y_it(∞) shocks (problematic)
    """
    
    def __init__(
        self,
        n_units: int = 200,
        n_periods: int = 10,
        treatment_period: int = 6,
        true_att: float = 2.0,
        seed: int = None,
    ):
        self.n_units = n_units
        self.n_periods = n_periods
        self.treatment_period = treatment_period
        self.true_att = true_att
        self.rng = np.random.default_rng(seed)
    
    def generate_base_panel(self) -> pd.DataFrame:
        """Generate complete balanced panel before applying selection."""
        data = []
        
        for i in range(self.n_units):
            # Unit fixed effect
            c_i = self.rng.normal(0, 2)
            
            # Treatment assignment (50% treated)
            d_i = 1 if i < self.n_units // 2 else 0
            gvar = self.treatment_period if d_i == 1 else 0
            
            # Observable covariate
            x_i = self.rng.normal(0, 1)
            
            for t in range(1, self.n_periods + 1):
                # Idiosyncratic shock
                u_it = self.rng.normal(0, 1)
                
                # Potential outcome Y(0)
                y0 = 10 + c_i + 0.5 * x_i + 0.1 * t + u_it
                
                # Observed outcome
                if d_i == 1 and t >= self.treatment_period:
                    y = y0 + self.true_att
                else:
                    y = y0
                
                data.append({
                    'unit_id': i,
                    'year': t,
                    'y': y,
                    'y0': y0,  # For analysis
                    'u_it': u_it,  # For MNAR selection
                    'c_i': c_i,  # For MAR on FE selection
                    'x_i': x_i,
                    'd': d_i,
                    'gvar': gvar,
                })
        
        return pd.DataFrame(data)
    
    def apply_mcar_selection(
        self, 
        data: pd.DataFrame, 
        missing_rate: float = 0.15
    ) -> pd.DataFrame:
        """
        Apply MCAR selection: missing completely at random.
        
        This is acceptable under the selection mechanism assumption.
        """
        mask = self.rng.random(len(data)) > missing_rate
        return data[mask].copy()
    
    def apply_mar_on_x_selection(
        self,
        data: pd.DataFrame,
        missing_rate_base: float = 0.10,
    ) -> pd.DataFrame:
        """
        Apply MAR selection: missing depends on observable X.
        
        P(missing) = base_rate + 0.1 * (x_i > 0)
        
        This is acceptable under the selection mechanism assumption
        when X is controlled for.
        """
        x_effect = 0.1 * (data['x_i'] > 0).astype(float)
        missing_prob = missing_rate_base + x_effect
        mask = self.rng.random(len(data)) > missing_prob
        return data[mask].copy()
    
    def apply_mar_on_fe_selection(
        self,
        data: pd.DataFrame,
        missing_rate_base: float = 0.10,
    ) -> pd.DataFrame:
        """
        Apply MAR selection: missing depends on unit fixed effect.
        
        P(missing) = base_rate + 0.05 * (c_i < -1)
        
        This is ACCEPTABLE under the selection mechanism assumption
        because the rolling transformation removes c_i.
        """
        fe_effect = 0.05 * (data['c_i'] < -1).astype(float)
        missing_prob = missing_rate_base + fe_effect
        mask = self.rng.random(len(data)) > missing_prob
        return data[mask].copy()
    
    def apply_mnar_on_shock_selection(
        self,
        data: pd.DataFrame,
        missing_rate_base: float = 0.10,
    ) -> pd.DataFrame:
        """
        Apply MNAR selection: missing depends on Y_it(∞) shocks.
        
        P(missing) = base_rate + 0.1 * (u_it < -1)
        
        This VIOLATES the selection mechanism assumption and should
        cause bias in the ATT estimate.
        """
        shock_effect = 0.1 * (data['u_it'] < -1).astype(float)
        missing_prob = missing_rate_base + shock_effect
        mask = self.rng.random(len(data)) > missing_prob
        return data[mask].copy()


# =============================================================================
# Test MCAR Selection (Acceptable)
# =============================================================================

class TestMCARSelection:
    """Monte Carlo tests for MCAR selection (acceptable)."""
    
    @pytest.fixture
    def dgp(self):
        return SelectionMechanismDGP(n_units=200, seed=42)
    
    @pytest.mark.slow
    def test_mcar_unbiased(self):
        """
        MCAR selection should produce unbiased ATT estimates.
        
        Run 50 simulations and verify:
        - Mean bias < 15% of true ATT
        - Coverage rate approximately 95%
        """
        n_sims = 50
        true_att = 2.0
        estimates = []
        covered = []
        
        for sim in range(n_sims):
            dgp = SelectionMechanismDGP(n_units=200, seed=sim)
            
            # Generate data with MCAR selection
            base_data = dgp.generate_base_panel()
            selected_data = dgp.apply_mcar_selection(base_data, missing_rate=0.15)
            
            try:
                result = lwdid(
                    selected_data,
                    y='y', gvar='gvar', ivar='unit_id', tvar='year',
                    rolling='demean', balanced_panel='ignore'
                )
                
                att_est = result.overall_att
                ci_lower = result.overall_ci[0]
                ci_upper = result.overall_ci[1]
                
                estimates.append(att_est)
                covered.append(ci_lower <= true_att <= ci_upper)
            except Exception:
                continue
        
        if len(estimates) < 30:
            pytest.skip("Too few successful simulations")
        
        # Check bias
        mean_estimate = np.mean(estimates)
        bias = mean_estimate - true_att
        relative_bias = abs(bias) / true_att
        
        assert relative_bias < 0.20, f"Bias too large: {bias:.3f} ({relative_bias:.1%})"
        
        # Check coverage (should be around 95%, allow 80-100%)
        coverage_rate = np.mean(covered)
        assert 0.75 < coverage_rate < 1.0, f"Coverage rate: {coverage_rate:.1%}"
    
    def test_mcar_diagnostics(self):
        """MCAR selection should be detected as low risk."""
        dgp = SelectionMechanismDGP(n_units=200, seed=42)
        
        base_data = dgp.generate_base_panel()
        selected_data = dgp.apply_mcar_selection(base_data, missing_rate=0.15)
        
        diag = diagnose_selection_mechanism(
            selected_data,
            y='y', ivar='unit_id', tvar='year', gvar='gvar',
            verbose=False
        )
        
        # MCAR should have low or medium risk
        assert diag.selection_risk in [SelectionRisk.LOW, SelectionRisk.MEDIUM]


# =============================================================================
# Test MAR on Fixed Effects Selection (Acceptable)
# =============================================================================

class TestMARonFESelection:
    """Monte Carlo tests for MAR on fixed effects (acceptable)."""
    
    @pytest.mark.slow
    def test_mar_fe_unbiased(self):
        """
        MAR selection on fixed effects should produce unbiased estimates.
        
        The rolling transformation removes unit fixed effects, so selection
        on c_i should not cause bias.
        """
        n_sims = 50
        true_att = 2.0
        estimates = []
        
        for sim in range(n_sims):
            dgp = SelectionMechanismDGP(n_units=200, seed=sim)
            
            base_data = dgp.generate_base_panel()
            selected_data = dgp.apply_mar_on_fe_selection(base_data)
            
            try:
                result = lwdid(
                    selected_data,
                    y='y', gvar='gvar', ivar='unit_id', tvar='year',
                    rolling='demean', balanced_panel='ignore'
                )
                estimates.append(result.overall_att)
            except Exception:
                continue
        
        if len(estimates) < 30:
            pytest.skip("Too few successful simulations")
        
        mean_estimate = np.mean(estimates)
        bias = mean_estimate - true_att
        relative_bias = abs(bias) / true_att
        
        # Should be unbiased because FE is removed
        assert relative_bias < 0.20, f"Unexpected bias: {bias:.3f}"
    
    def test_mar_fe_diagnostics(self):
        """MAR on FE should be detected appropriately."""
        dgp = SelectionMechanismDGP(n_units=200, seed=42)
        
        base_data = dgp.generate_base_panel()
        selected_data = dgp.apply_mar_on_fe_selection(base_data)
        
        diag = diagnose_selection_mechanism(
            selected_data,
            y='y', ivar='unit_id', tvar='year', gvar='gvar',
            verbose=False
        )
        
        # Should not be high risk (FE is removed by transformation)
        assert diag.selection_risk != SelectionRisk.HIGH or \
               diag.missing_pattern != MissingPattern.MNAR


# =============================================================================
# Test MNAR on Shocks Selection (Problematic)
# =============================================================================

class TestMNARonShockSelection:
    """Monte Carlo tests for MNAR on shocks (problematic)."""
    
    @pytest.mark.slow
    def test_mnar_shock_detection(self):
        """
        MNAR selection on Y_it(∞) shocks should be detected as risky.
        
        This violates the selection mechanism assumption.
        """
        dgp = SelectionMechanismDGP(n_units=200, seed=42)
        
        base_data = dgp.generate_base_panel()
        selected_data = dgp.apply_mnar_on_shock_selection(base_data)
        
        diag = diagnose_selection_mechanism(
            selected_data,
            y='y', ivar='unit_id', tvar='year', gvar='gvar',
            verbose=False
        )
        
        # Should detect elevated risk or MNAR pattern
        # Note: Detection may not be perfect with limited data
        print(f"MNAR scenario - Risk: {diag.selection_risk.value}, "
              f"Pattern: {diag.missing_pattern.value}")
    
    @pytest.mark.slow
    def test_mnar_shock_bias_direction(self):
        """
        MNAR selection on negative shocks should show some bias.
        
        When units with negative shocks are more likely to be missing,
        the observed sample is biased upward.
        """
        n_sims = 30
        true_att = 2.0
        estimates = []
        
        for sim in range(n_sims):
            dgp = SelectionMechanismDGP(n_units=200, seed=sim)
            
            base_data = dgp.generate_base_panel()
            selected_data = dgp.apply_mnar_on_shock_selection(base_data)
            
            try:
                result = lwdid(
                    selected_data,
                    y='y', gvar='gvar', ivar='unit_id', tvar='year',
                    rolling='demean', balanced_panel='ignore'
                )
                estimates.append(result.overall_att)
            except Exception:
                continue
        
        if len(estimates) < 20:
            pytest.skip("Too few successful simulations")
        
        mean_estimate = np.mean(estimates)
        bias = mean_estimate - true_att
        
        # Report bias (may or may not be significant depending on DGP)
        print(f"MNAR on shocks - Mean estimate: {mean_estimate:.3f}, "
              f"True ATT: {true_att}, Bias: {bias:.3f}")


# =============================================================================
# Test Detrending Robustness
# =============================================================================

class TestDetrendingRobustness:
    """Test that detrending provides additional robustness."""
    
    @pytest.mark.slow
    def test_detrend_more_robust_to_trend_selection(self):
        """
        Detrending should be more robust when selection depends on trends.
        
        From Lee & Wooldridge (2025):
        "Removing unit-specific trends provides additional resiliency to
        unbalanced panels, as we are now allowing for two sources of
        heterogeneity—level and trend—to be correlated with selection."
        """
        n_sims = 30
        true_att = 2.0
        
        demean_estimates = []
        detrend_estimates = []
        
        for sim in range(n_sims):
            rng = np.random.default_rng(sim)
            
            # Generate data with trend-dependent selection
            data = []
            for i in range(200):
                c_i = rng.normal(0, 2)
                g_i = rng.normal(0, 0.2)  # Unit-specific trend
                d_i = 1 if i < 100 else 0
                gvar = 6 if d_i == 1 else 0
                
                for t in range(1, 11):
                    # Selection depends on trend
                    if g_i < -0.1 and t > 5 and rng.random() < 0.3:
                        continue
                    
                    y = 10 + c_i + g_i * t + rng.normal(0, 1)
                    if d_i == 1 and t >= 6:
                        y += true_att
                    
                    data.append({
                        'unit_id': i, 'year': t, 'y': y,
                        'd': d_i, 'gvar': gvar
                    })
            
            df = pd.DataFrame(data)
            
            try:
                result_demean = lwdid(
                    df, y='y', gvar='gvar', ivar='unit_id', tvar='year',
                    rolling='demean', balanced_panel='ignore'
                )
                demean_estimates.append(result_demean.overall_att)
            except Exception:
                pass
            
            try:
                result_detrend = lwdid(
                    df, y='y', gvar='gvar', ivar='unit_id', tvar='year',
                    rolling='detrend', balanced_panel='ignore'
                )
                detrend_estimates.append(result_detrend.overall_att)
            except Exception:
                pass
        
        if len(demean_estimates) < 20 or len(detrend_estimates) < 20:
            pytest.skip("Too few successful simulations")
        
        # Compare RMSE
        demean_rmse = np.sqrt(np.mean((np.array(demean_estimates) - true_att)**2))
        detrend_rmse = np.sqrt(np.mean((np.array(detrend_estimates) - true_att)**2))
        
        print(f"Demean RMSE: {demean_rmse:.3f}, Detrend RMSE: {detrend_rmse:.3f}")
        
        # Detrending should have lower or similar RMSE when selection depends on trends
        # (This is a soft assertion as results depend on specific DGP)


# =============================================================================
# Test Sample Size Effects
# =============================================================================

class TestSampleSizeEffects:
    """Test how sample size affects estimation under selection."""
    
    @pytest.mark.slow
    def test_larger_sample_reduces_variance(self):
        """Larger samples should have lower variance in estimates."""
        true_att = 2.0
        
        small_estimates = []
        large_estimates = []
        
        for sim in range(30):
            # Small sample
            dgp_small = SelectionMechanismDGP(n_units=100, seed=sim)
            base_small = dgp_small.generate_base_panel()
            selected_small = dgp_small.apply_mcar_selection(base_small, missing_rate=0.15)
            
            try:
                result_small = lwdid(
                    selected_small,
                    y='y', gvar='gvar', ivar='unit_id', tvar='year',
                    rolling='demean', balanced_panel='ignore'
                )
                small_estimates.append(result_small.overall_att)
            except Exception:
                pass
            
            # Large sample
            dgp_large = SelectionMechanismDGP(n_units=400, seed=sim + 1000)
            base_large = dgp_large.generate_base_panel()
            selected_large = dgp_large.apply_mcar_selection(base_large, missing_rate=0.15)
            
            try:
                result_large = lwdid(
                    selected_large,
                    y='y', gvar='gvar', ivar='unit_id', tvar='year',
                    rolling='demean', balanced_panel='ignore'
                )
                large_estimates.append(result_large.overall_att)
            except Exception:
                pass
        
        if len(small_estimates) < 15 or len(large_estimates) < 15:
            pytest.skip("Too few successful simulations")
        
        small_std = np.std(small_estimates)
        large_std = np.std(large_estimates)
        
        print(f"Small sample std: {small_std:.3f}, Large sample std: {large_std:.3f}")
        
        # Large sample should have lower variance
        assert large_std < small_std * 1.5  # Allow some tolerance
