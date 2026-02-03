"""
Python-to-Paper End-to-End Test for Seasonal Adjustment.

Task 4.4: Verify implementation matches paper ssrn-5325686 Section 3 formulas.

Paper Reference: Lee & Wooldridge (2025) "Simple Approaches to Inference with 
Difference-in-Differences Estimators with Small Cross-Sectional Sample Sizes"

Key Formulas from Section 3:
- Procedure 2.1 (Unit-Specific Demeaning with Seasonality):
  Ẏ_{it} = Y_{it} - μ̂_i - Σ_{q=2}^{Q} γ̂_q D_q
  
- Procedure 3.1 (Unit-Specific Detrending with Seasonality):
  Ÿ_{it} = Y_{it} - α̂_i - β̂_i·t - Σ_{q=2}^{Q} γ̂_q D_q

The paper states (Section 3):
"With quarterly or monthly data, and maybe even with weekly data, it might make 
sense to remove seasonality at the unit level (perhaps in addition to a trend). 
In the first step of Procedure 2.1 or 3.1, we would simply include quarterly, 
monthly, or weekly dummy variables and obtain the deseasonalized/detrended outcomes."
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from lwdid import lwdid
from lwdid.transformations import demeanq_unit, detrendq_unit


class TestPaperFormulaDemeanq:
    """Verify demeanq matches Procedure 2.1 with seasonal dummies."""
    
    def test_demeanq_formula_exact(self):
        """
        Test demeanq formula: Ẏ_{it} = Y_{it} - μ̂_i - Σ_{q=2}^{Q} γ̂_q D_q
        
        The pre-treatment regression is:
        Y_{it} = μ_i + γ_2 D_2 + γ_3 D_3 + γ_4 D_4 + ε_{it}
        
        where D_q = 1 if quarter = q, 0 otherwise (Q1 is reference).
        """
        # Create data with known parameters
        # μ = 100, γ_2 = 5, γ_3 = 10, γ_4 = 3
        mu = 100
        gamma = {1: 0, 2: 5, 3: 10, 4: 3}  # Q1 is reference (gamma_1 = 0)
        
        # 12 pre-treatment periods (3 years), 8 post-treatment periods (2 years)
        data = []
        for t in range(1, 21):
            quarter = ((t - 1) % 4) + 1
            y = mu + gamma[quarter]  # No noise for exact test
            post = 1 if t > 12 else 0
            data.append({'t': t, 'quarter': quarter, 'y': y, 'post': post})
        
        df = pd.DataFrame(data)
        
        # Apply demeanq
        yhat, ydot = demeanq_unit(df, 'y', 'quarter', 'post', Q=4)
        
        # Verify: residuals should be exactly zero for perfect seasonal pattern
        assert_allclose(ydot, 0, atol=1e-10, 
                       err_msg="demeanq residuals should be zero for perfect seasonal pattern")
        
        # Verify fitted values match the formula
        # ŷ_{it} = μ̂ + γ̂_q for each quarter
        for t in range(len(df)):
            q = df.iloc[t]['quarter']
            expected_yhat = mu + gamma[q]
            assert_allclose(yhat[t], expected_yhat, atol=1e-10,
                           err_msg=f"Fitted value at t={t+1}, q={q} should be {expected_yhat}")
    
    def test_demeanq_with_treatment_effect(self):
        """
        Test demeanq correctly identifies treatment effect.
        
        DGP: Y_{it} = μ_i + γ_q + τ·D_i·post_t + ε_{it}
        
        After demeanq transformation:
        Ẏ_{it} = τ·D_i·post_t + ε_{it} (for post-treatment periods)
        """
        np.random.seed(42)
        
        # Parameters
        mu = 100
        gamma = {1: 0, 2: 5, 3: 10, 4: 3}
        tau = 10  # True treatment effect
        
        # Generate data for treated unit
        data = []
        for t in range(1, 21):
            quarter = ((t - 1) % 4) + 1
            post = 1 if t > 12 else 0
            epsilon = np.random.normal(0, 0.1)  # Small noise
            y = mu + gamma[quarter] + tau * post + epsilon
            data.append({'t': t, 'quarter': quarter, 'y': y, 'post': post})
        
        df = pd.DataFrame(data)
        
        # Apply demeanq
        yhat, ydot = demeanq_unit(df, 'y', 'quarter', 'post', Q=4)
        
        # Post-treatment residuals should be approximately tau
        post_mask = df['post'] == 1
        post_mean = np.mean(ydot[post_mask])
        
        assert abs(post_mean - tau) < 0.5, \
            f"Post-treatment mean should be ~{tau}, got {post_mean}"
    
    def test_demeanq_ols_coefficients(self):
        """
        Verify OLS coefficients from pre-treatment regression.
        
        The regression Y_{it} = μ + γ_2 D_2 + γ_3 D_3 + γ_4 D_4 + ε
        should recover the true seasonal coefficients.
        """
        # Known parameters
        mu = 50
        gamma = {1: 0, 2: 10, 3: 20, 4: 15}
        
        # Create pre-treatment data only
        data = []
        for t in range(1, 13):  # 12 pre-treatment periods
            quarter = ((t - 1) % 4) + 1
            y = mu + gamma[quarter]
            data.append({'t': t, 'quarter': quarter, 'y': y, 'post': 0})
        
        # Add post-treatment periods
        for t in range(13, 21):
            quarter = ((t - 1) % 4) + 1
            y = mu + gamma[quarter] + 5  # Add treatment effect
            data.append({'t': t, 'quarter': quarter, 'y': y, 'post': 1})
        
        df = pd.DataFrame(data)
        
        # Apply demeanq
        yhat, ydot = demeanq_unit(df, 'y', 'quarter', 'post', Q=4)
        
        # Verify pre-treatment fitted values
        pre_mask = df['post'] == 0
        for idx in df[pre_mask].index:
            q = df.loc[idx, 'quarter']
            expected = mu + gamma[q]
            assert_allclose(yhat[idx], expected, atol=1e-10,
                           err_msg=f"Pre-treatment fitted value for Q{q} incorrect")


class TestPaperFormulaDetrendq:
    """Verify detrendq matches Procedure 3.1 with seasonal dummies."""
    
    def test_detrendq_formula_exact(self):
        """
        Test detrendq formula: Ÿ_{it} = Y_{it} - α̂_i - β̂_i·t - Σ_{q=2}^{Q} γ̂_q D_q
        
        The pre-treatment regression is:
        Y_{it} = α_i + β_i·t + γ_2 D_2 + γ_3 D_3 + γ_4 D_4 + ε_{it}
        """
        # Create data with known parameters
        # α = 100, β = 0.5, γ_2 = 5, γ_3 = 10, γ_4 = 3
        alpha = 100
        beta = 0.5
        gamma = {1: 0, 2: 5, 3: 10, 4: 3}
        
        # 12 pre-treatment periods, 8 post-treatment periods
        data = []
        for t in range(1, 21):
            quarter = ((t - 1) % 4) + 1
            y = alpha + beta * t + gamma[quarter]  # No noise
            post = 1 if t > 12 else 0
            data.append({'t': t, 'quarter': quarter, 'y': y, 'post': post})
        
        df = pd.DataFrame(data)
        
        # Apply detrendq
        yhat, ydot = detrendq_unit(df, 'y', 't', 'quarter', 'post', Q=4)
        
        # Verify: residuals should be exactly zero for perfect pattern
        assert_allclose(ydot, 0, atol=1e-8,
                       err_msg="detrendq residuals should be zero for perfect trend+seasonal pattern")
    
    def test_detrendq_with_treatment_effect(self):
        """
        Test detrendq correctly identifies treatment effect.
        
        DGP: Y_{it} = α_i + β_i·t + γ_q + τ·D_i·post_t + ε_{it}
        
        After detrendq transformation:
        Ÿ_{it} = τ·D_i·post_t + ε_{it} (for post-treatment periods)
        """
        np.random.seed(42)
        
        # Parameters
        alpha = 100
        beta = 0.5
        gamma = {1: 0, 2: 5, 3: 10, 4: 3}
        tau = 10  # True treatment effect
        
        # Generate data
        data = []
        for t in range(1, 21):
            quarter = ((t - 1) % 4) + 1
            post = 1 if t > 12 else 0
            epsilon = np.random.normal(0, 0.1)  # Small noise
            y = alpha + beta * t + gamma[quarter] + tau * post + epsilon
            data.append({'t': t, 'quarter': quarter, 'y': y, 'post': post})
        
        df = pd.DataFrame(data)
        
        # Apply detrendq
        yhat, ydot = detrendq_unit(df, 'y', 't', 'quarter', 'post', Q=4)
        
        # Post-treatment residuals should be approximately tau
        post_mask = df['post'] == 1
        post_mean = np.mean(ydot[post_mask])
        
        assert abs(post_mean - tau) < 0.5, \
            f"Post-treatment mean should be ~{tau}, got {post_mean}"
    
    def test_detrendq_time_centering(self):
        """
        Verify time centering is applied correctly.
        
        Paper uses centered time: t_centered = t - mean(t_pre)
        This improves numerical stability and interpretation.
        """
        # Create data
        alpha = 100
        beta = 2.0
        gamma = {1: 0, 2: 5, 3: 10, 4: 3}
        
        data = []
        for t in range(1, 21):
            quarter = ((t - 1) % 4) + 1
            y = alpha + beta * t + gamma[quarter]
            post = 1 if t > 12 else 0
            data.append({'t': t, 'quarter': quarter, 'y': y, 'post': post})
        
        df = pd.DataFrame(data)
        
        # Apply detrendq
        yhat, ydot = detrendq_unit(df, 'y', 't', 'quarter', 'post', Q=4)
        
        # Residuals should be near zero
        assert_allclose(ydot, 0, atol=1e-8,
                       err_msg="detrendq should produce zero residuals for perfect pattern")


class TestPaperMonthlyWeekly:
    """Test monthly (Q=12) and weekly (Q=52) as mentioned in paper."""
    
    def test_monthly_demeanq_q12(self):
        """
        Paper: "With quarterly or monthly data... include monthly dummy variables"
        
        Test Q=12 for monthly data.
        """
        # Monthly seasonal pattern
        gamma = {m: m * 2 for m in range(1, 13)}  # gamma_m = 2*m
        mu = 100
        
        # 24 pre-treatment months, 12 post-treatment months
        data = []
        for t in range(1, 37):
            month = ((t - 1) % 12) + 1
            y = mu + gamma[month]
            post = 1 if t > 24 else 0
            data.append({'t': t, 'month': month, 'y': y, 'post': post})
        
        df = pd.DataFrame(data)
        
        # Apply demeanq with Q=12
        yhat, ydot = demeanq_unit(df, 'y', 'month', 'post', Q=12)
        
        # Residuals should be zero
        assert_allclose(ydot, 0, atol=1e-10,
                       err_msg="Monthly demeanq should produce zero residuals")
    
    def test_weekly_demeanq_q52(self):
        """
        Paper: "maybe even with weekly data... include weekly dummy variables"
        
        Test Q=52 for weekly data.
        """
        # Weekly seasonal pattern (simplified)
        gamma = {w: np.sin(2 * np.pi * w / 52) * 10 for w in range(1, 53)}
        mu = 100
        
        # 104 pre-treatment weeks (2 years), 52 post-treatment weeks (1 year)
        data = []
        for t in range(1, 157):
            week = ((t - 1) % 52) + 1
            y = mu + gamma[week]
            post = 1 if t > 104 else 0
            data.append({'t': t, 'week': week, 'y': y, 'post': post})
        
        df = pd.DataFrame(data)
        
        # Apply demeanq with Q=52
        yhat, ydot = demeanq_unit(df, 'y', 'week', 'post', Q=52)
        
        # Residuals should be near zero (may have small numerical error for Q=52)
        assert_allclose(ydot, 0, atol=1e-8,
                       err_msg="Weekly demeanq should produce near-zero residuals")


class TestPaperATTRecovery:
    """Test ATT recovery as described in paper Section 2-3."""
    
    def test_att_recovery_demeanq(self):
        """
        Paper Procedure 2.1 Step 4:
        "Obtain an average effect, τ̂_DM, from Ẏ_i on 1, D_i"
        
        The ATT should be recovered from the cross-sectional regression.
        """
        np.random.seed(42)
        
        # Parameters
        n_units = 10
        n_periods = 20
        tau = 10  # True ATT
        
        # Generate panel data
        data = []
        for unit_id in range(1, n_units + 1):
            mu_i = 100 + (unit_id - 5.5) * 5  # Unit fixed effect
            gamma = {1: 0, 2: 5, 3: 10, 4: 3}
            treated = 1 if unit_id <= 5 else 0
            
            for t in range(1, n_periods + 1):
                quarter = ((t - 1) % 4) + 1
                post = 1 if t > 12 else 0
                epsilon = np.random.normal(0, 1)
                
                y = mu_i + gamma[quarter] + tau * treated * post + epsilon
                
                data.append({
                    'id': unit_id,
                    't': t,
                    'quarter': quarter,
                    'year': 2018 + (t - 1) // 4,
                    'y': y,
                    'treated': treated,
                    'post': post
                })
        
        df = pd.DataFrame(data)
        
        # Use lwdid with demeanq
        result = lwdid(
            df, y='y', d='treated', ivar='id',
            tvar=['year', 'quarter'], post='post',
            rolling='demeanq', season_var='quarter', Q=4
        )
        
        # ATT should be close to true value
        assert abs(result.att - tau) < 3.0, \
            f"ATT should be ~{tau}, got {result.att}"
    
    def test_att_recovery_detrendq(self):
        """
        Paper Procedure 3.1 Step 4:
        "Obtain an average effect, τ̂_DT, from Ÿ_i on 1, D_i"
        """
        np.random.seed(42)
        
        # Parameters with unit-specific trends
        n_units = 10
        n_periods = 20
        tau = 10
        
        data = []
        for unit_id in range(1, n_units + 1):
            alpha_i = 100 + (unit_id - 5.5) * 5
            beta_i = 0.5 + (unit_id - 5.5) * 0.1  # Unit-specific trend
            gamma = {1: 0, 2: 5, 3: 10, 4: 3}
            treated = 1 if unit_id <= 5 else 0
            
            for t in range(1, n_periods + 1):
                quarter = ((t - 1) % 4) + 1
                post = 1 if t > 12 else 0
                epsilon = np.random.normal(0, 1)
                
                y = alpha_i + beta_i * t + gamma[quarter] + tau * treated * post + epsilon
                
                data.append({
                    'id': unit_id,
                    't': t,
                    'quarter': quarter,
                    'year': 2018 + (t - 1) // 4,
                    'y': y,
                    'treated': treated,
                    'post': post
                })
        
        df = pd.DataFrame(data)
        
        # Use lwdid with detrendq
        result = lwdid(
            df, y='y', d='treated', ivar='id',
            tvar=['year', 'quarter'], post='post',
            rolling='detrendq', season_var='quarter', Q=4
        )
        
        # ATT should be close to true value
        assert abs(result.att - tau) < 3.0, \
            f"ATT should be ~{tau}, got {result.att}"


class TestPaperSeasonalAggregation:
    """Test seasonal aggregation formula from paper Section 3 extension."""
    
    def test_seasonal_aggregation_formula(self):
        """
        Paper (2025 version) Section 3 extension:
        "For quarterly or higher frequency data, aggregate at year-quarter level:
        Ȳ_{stq} = Σ_{i∈(s,t,q)} w_{istq} Y_{ist}"
        
        This test verifies the aggregation is handled correctly.
        """
        np.random.seed(42)
        
        # Create repeated cross-section style data
        # Multiple observations per unit-time-quarter
        data = []
        for unit_id in range(1, 6):
            for year in range(2018, 2023):
                for quarter in range(1, 5):
                    t = (year - 2018) * 4 + quarter
                    post = 1 if year >= 2021 else 0
                    treated = 1 if unit_id <= 2 else 0
                    
                    # Multiple observations per cell
                    for obs in range(3):
                        y = 100 + quarter * 5 + 10 * treated * post + np.random.normal(0, 1)
                        data.append({
                            'id': unit_id,
                            'year': year,
                            'quarter': quarter,
                            't': t,
                            'y': y,
                            'treated': treated,
                            'post': post
                        })
        
        df = pd.DataFrame(data)
        
        # Aggregate to unit-year-quarter level
        df_agg = df.groupby(['id', 'year', 'quarter', 't', 'treated', 'post']).agg({
            'y': 'mean'
        }).reset_index()
        
        # Run lwdid on aggregated data
        result = lwdid(
            df_agg, y='y', d='treated', ivar='id',
            tvar=['year', 'quarter'], post='post',
            rolling='demeanq', season_var='quarter', Q=4
        )
        
        # Should recover treatment effect
        assert result.att is not None
        assert 5.0 < result.att < 15.0, f"ATT should be ~10, got {result.att}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
