"""
Unit tests for the IPWRA (Inverse Probability Weighted Regression Adjustment) estimator.

Validates basic functionality, boundary conditions, numerical correctness,
and consistency with the RA estimator for the IPWRA implementation.

Validates Section 7.1 (IPWRA estimator specification and doubly robust
property) of the Lee-Wooldridge Difference-in-Differences framework.

References
----------
Lee, S. & Wooldridge, J. M. (2025). A Simple Transformation Approach to
    Difference-in-Differences Estimation for Panel Data. SSRN 4516518.
Lee, S. & Wooldridge, J. M. (2026). Simple Approaches to Inference with
    DiD Estimators with Small Cross-Sectional Sample Sizes. SSRN 5325686.
"""

import numpy as np
import pandas as pd
import pytest

from lwdid.staggered.estimators import (
    estimate_ipwra,
    estimate_propensity_score,
    estimate_outcome_model,
    IPWRAResult,
)


class TestEstimatePropensityScore:
    """Tests for propensity score estimation."""
    
    def test_basic_propensity_score(self):
        """Test basic propensity score estimation."""
        np.random.seed(42)
        n = 200
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        # True propensity score: logit(p) = -0.5 + 0.5*x1 + 0.3*x2
        logit_p = -0.5 + 0.5 * x1 + 0.3 * x2
        p_true = 1 / (1 + np.exp(-logit_p))
        d = (np.random.uniform(0, 1, n) < p_true).astype(int)
        
        data = pd.DataFrame({'d': d, 'x1': x1, 'x2': x2})
        
        pscores, coef = estimate_propensity_score(
            data, 'd', ['x1', 'x2'], trim_threshold=0.01
        )
        
        # Verify propensity scores are in a reasonable range
        assert pscores.min() >= 0.01
        assert pscores.max() <= 0.99
        assert len(pscores) == n
        
        # Verify coefficient signs (x1 and x2 should be positive)
        assert coef['x1'] > 0, f"x1 coefficient should be positive: {coef['x1']}"
        assert coef['x2'] > 0, f"x2 coefficient should be positive: {coef['x2']}"
    
    def test_propensity_score_trimming(self):
        """Test propensity score trimming."""
        np.random.seed(42)
        n = 100
        # Construct extreme data: near-perfect separation
        x = np.concatenate([np.random.normal(-3, 0.5, 50),
                           np.random.normal(3, 0.5, 50)])
        d = np.concatenate([np.zeros(50), np.ones(50)])
        
        data = pd.DataFrame({'d': d, 'x': x})
        
        pscores, _ = estimate_propensity_score(
            data, 'd', ['x'], trim_threshold=0.05
        )
        
        assert pscores.min() >= 0.05, f"Min propensity score should be >= 0.05: {pscores.min()}"
        assert pscores.max() <= 0.95, f"Max propensity score should be <= 0.95: {pscores.max()}"


class TestEstimateOutcomeModel:
    """Tests for outcome model estimation."""
    
    def test_basic_outcome_model(self):
        """Test basic outcome model estimation."""
        np.random.seed(42)
        n = 200
        x = np.random.normal(0, 1, n)
        # True model: Y = 1 + 2*x + error
        y = 1 + 2 * x + np.random.normal(0, 0.5, n)
        d = np.random.binomial(1, 0.5, n)
        
        data = pd.DataFrame({'y': y, 'd': d, 'x': x})
        
        m0_hat, coef = estimate_outcome_model(data, 'y', 'd', ['x'])
        
        # Verify coefficients are close to true values
        assert abs(coef['_intercept'] - 1) < 0.5, f"Intercept should be close to 1: {coef['_intercept']}"
        assert abs(coef['x'] - 2) < 0.5, f"x coefficient should be close to 2: {coef['x']}"
        
        # Verify prediction length
        assert len(m0_hat) == n


class TestEstimateIPWRA:
    """Tests for IPWRA estimation."""
    
    @pytest.fixture
    def simple_data(self):
        """Construct simple test data with known ATT ≈ 2."""
        np.random.seed(42)
        n = 500
        
        # Covariates
        x = np.random.normal(0, 1, n)
        
        # Propensity score: P(D=1|X) = logit(-0.5 + 0.3*x)
        logit_p = -0.5 + 0.3 * x
        p_true = 1 / (1 + np.exp(-logit_p))
        d = (np.random.uniform(0, 1, n) < p_true).astype(int)
        
        # Potential outcomes
        # Y(0) = 1 + x + error
        # Y(1) = 3 + x + error  (ATT ≈ 2)
        y0 = 1 + x + np.random.normal(0, 0.5, n)
        y1 = 3 + x + np.random.normal(0, 0.5, n)
        y = d * y1 + (1 - d) * y0
        
        return pd.DataFrame({'y': y, 'd': d, 'x': x})
    
    def test_ipwra_basic(self, simple_data):
        """Test basic IPWRA estimation."""
        result = estimate_ipwra(
            simple_data, 'y', 'd', ['x'],
            se_method='analytical'
        )
        
        assert isinstance(result, IPWRAResult)
        # ATT should be close to 2
        assert abs(result.att - 2) < 0.5, f"ATT={result.att}, expected ≈ 2"
        assert result.se > 0, "SE should be positive"
        assert result.ci_lower < result.att < result.ci_upper, "CI should contain point estimate"
        assert result.n_treated > 0
        assert result.n_control > 0
    
    def test_ipwra_bootstrap_se(self, simple_data):
        """Test bootstrap standard errors."""
        result = estimate_ipwra(
            simple_data, 'y', 'd', ['x'],
            se_method='bootstrap',
            n_bootstrap=50,  # Reduced for faster testing
            seed=42
        )
        
        assert result.se > 0, "Bootstrap SE should be positive"
        assert result.ci_lower < result.att < result.ci_upper
    
    def test_ipwra_missing_controls_error(self, simple_data):
        """Missing control variables should raise error."""
        with pytest.raises(ValueError, match="Control variables not found"):
            estimate_ipwra(
                simple_data, 'y', 'd', ['nonexistent']
            )
    
    def test_ipwra_missing_y_error(self, simple_data):
        """Missing outcome variable should raise error."""
        with pytest.raises(ValueError, match="Outcome variable.*not found"):
            estimate_ipwra(
                simple_data, 'nonexistent', 'd', ['x']
            )
    
    def test_ipwra_missing_d_error(self, simple_data):
        """Missing treatment indicator should raise error."""
        with pytest.raises(ValueError, match="Treatment indicator.*not found"):
            estimate_ipwra(
                simple_data, 'y', 'nonexistent', ['x']
            )
    
    def test_ipwra_insufficient_treated(self):
        """Insufficient treated units should raise error."""
        data = pd.DataFrame({
            'y': [1, 2, 3, 4, 5],
            'd': [1, 0, 0, 0, 0],  # Only 1 treated
            'x': [1, 2, 3, 4, 5]
        })
        
        with pytest.raises(ValueError, match="Insufficient treated units"):
            estimate_ipwra(data, 'y', 'd', ['x'])
    
    def test_ipwra_insufficient_control(self):
        """Insufficient control units should raise error."""
        data = pd.DataFrame({
            'y': [1, 2, 3, 4, 5],
            'd': [1, 1, 1, 1, 0],  # Only 1 control
            'x': [1, 2, 3, 4, 5]
        })
        
        with pytest.raises(ValueError, match="Insufficient control units"):
            estimate_ipwra(data, 'y', 'd', ['x'])


class TestIPWRAEdgeCases:
    """IPWRA edge case tests."""
    
    def test_small_treated_warning(self):
        """Small treated sample should issue warning."""
        np.random.seed(42)
        data = pd.DataFrame({
            'y': np.random.normal(0, 1, 20),
            'd': np.array([1, 1, 1, 1] + [0] * 16),  # Only 4 treated
            'x': np.random.normal(0, 1, 20)
        })
        
        with pytest.warns(UserWarning, match="Small treated sample"):
            estimate_ipwra(data, 'y', 'd', ['x'])
    
    def test_small_control_warning(self):
        """Small control sample should issue warning."""
        np.random.seed(42)
        data = pd.DataFrame({
            'y': np.random.normal(0, 1, 20),
            'd': np.array([1] * 12 + [0] * 8),  # Only 8 control
            'x': np.random.normal(0, 1, 20)
        })
        
        with pytest.warns(UserWarning, match="Small control sample"):
            estimate_ipwra(data, 'y', 'd', ['x'])
    
    def test_unknown_se_method_error(self):
        """Unknown SE method should raise error."""
        np.random.seed(42)
        data = pd.DataFrame({
            'y': np.random.normal(0, 1, 100),
            'd': np.random.binomial(1, 0.5, 100),
            'x': np.random.normal(0, 1, 100)
        })
        
        with pytest.raises(ValueError, match="Unknown se_method"):
            estimate_ipwra(data, 'y', 'd', ['x'], se_method='invalid')
    
    def test_extreme_weights_warning(self):
        """Extreme weights should issue a warning."""
        np.random.seed(42)
        n = 100
        # Construct near-perfect separation data (leading to extreme propensity scores)
        x = np.concatenate([np.random.normal(-2, 0.3, 50),
                           np.random.normal(2, 0.3, 50)])
        d = np.concatenate([np.zeros(50), np.ones(50)])
        y = 1 + x + 2 * d + np.random.normal(0, 0.5, n)
        
        data = pd.DataFrame({'y': y, 'd': d, 'x': x})
        
        # Use a small trim_threshold to allow extreme values
        with pytest.warns(UserWarning, match="extreme|overlap|propensity"):
            estimate_ipwra(data, 'y', 'd', ['x'], trim_threshold=0.001)
    
    def test_ipwra_multiple_controls(self):
        """Test with multiple control variables."""
        np.random.seed(42)
        n = 300
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        x3 = np.random.normal(0, 1, n)
        
        logit_p = -0.3 + 0.2 * x1 + 0.1 * x2
        p = 1 / (1 + np.exp(-logit_p))
        d = (np.random.uniform(0, 1, n) < p).astype(int)
        
        y = 1 + 0.5 * x1 + 0.3 * x2 + 0.1 * x3 + 2 * d + np.random.normal(0, 0.5, n)
        
        data = pd.DataFrame({'y': y, 'd': d, 'x1': x1, 'x2': x2, 'x3': x3})
        
        result = estimate_ipwra(
            data, 'y', 'd', ['x1', 'x2', 'x3'],
            se_method='analytical'
        )
        
        # ATT should be close to 2
        assert abs(result.att - 2) < 0.5, f"ATT={result.att}, expected ≈ 2"
        assert result.se > 0


class TestIPWRADoublyRobust:
    """
    Tests for the doubly robust property of IPWRA.
    
    Verifies the core advantage of IPWRA: the estimator is consistent as long as
    either the propensity score model or the outcome model is correctly specified.
    
    Reference: Lee & Wooldridge (2023) Section 7.1, Tables 7.3-7.4
    """
    
    def test_doubly_robust_outcome_correct_propensity_wrong(self):
        """
        Doubly robust test 1: Correct outcome model + misspecified propensity score model.
        
        DGP:
        - True propensity score includes quadratic term: logit(p) = -1.2 + 0.5*(x1-4)/2 - x2 + 0.5*(x1-4)^2
        - Outcome model is linear: Y(0) = 1 + (x1-4)/3 + x2/2 + error
        - True ATT = 2.0
        
        Using a linear-only propensity score model (misspecified), ATT should still be close to the true value.
        """
        np.random.seed(42)
        n = 1000
        
        # Covariates
        x1 = np.random.gamma(2, 2, n)  # mean=4
        x2 = np.random.binomial(1, 0.6, n)
        
        # True propensity score (with quadratic term)
        logit_true = -1.2 + (x1-4)/2 - x2 + (x1-4)**2/4
        p_true = 1 / (1 + np.exp(-logit_true))
        d = (np.random.uniform(0, 1, n) < p_true).astype(int)
        
        # Outcome variable (linear)
        tau_true = 2.0
        y0 = 1 + (x1-4)/3 + x2/2 + np.random.normal(0, 1, n)
        y1 = y0 + tau_true
        y = d * y1 + (1 - d) * y0
        
        data = pd.DataFrame({'y': y, 'd': d, 'x1': x1, 'x2': x2})
        
        # IPWRA estimation (using linear propensity score model - misspecified)
        result = estimate_ipwra(
            data, 'y', 'd', 
            controls=['x1', 'x2'],  # Outcome model correct
            propensity_controls=['x1', 'x2'],  # Propensity score missing x1^2 term
            se_method='analytical'
        )
        
        # Since the outcome model is correct, ATT should be close to the true value
        bias = abs(result.att - tau_true)
        assert bias < 0.5, (
            f"Doubly robust property failed: ATT={result.att:.4f}, "
            f"expected ≈ {tau_true}, bias={bias:.4f}"
        )
    
    def test_doubly_robust_propensity_correct_outcome_wrong(self):
        """
        Doubly robust test 2: Correct propensity score model + misspecified outcome model.
        
        DGP:
        - Propensity score is linear: logit(p) = -1.2 + 0.5*(x1-4)/2 - x2
        - Outcome model includes quadratic term: Y(0) = 1 + (x1-4)/3 + x2/2 + (x1-4)^2/4 + error
        - True ATT = 2.0
        
        Using a linear-only outcome model (misspecified), ATT should still be close to the true value.
        """
        np.random.seed(123)
        n = 1000
        
        # Covariates
        x1 = np.random.gamma(2, 2, n)
        x2 = np.random.binomial(1, 0.6, n)
        
        # Propensity score (linear)
        logit = -1.2 + (x1-4)/2 - x2
        p = 1 / (1 + np.exp(-logit))
        d = (np.random.uniform(0, 1, n) < p).astype(int)
        
        # Outcome variable (with quadratic term)
        tau_true = 2.0
        y0 = 1 + (x1-4)/3 + x2/2 + (x1-4)**2/4 + np.random.normal(0, 1, n)
        y1 = y0 + tau_true
        y = d * y1 + (1 - d) * y0
        
        data = pd.DataFrame({'y': y, 'd': d, 'x1': x1, 'x2': x2})
        
        # IPWRA estimation (using linear outcome model - misspecified)
        result = estimate_ipwra(
            data, 'y', 'd',
            controls=['x1', 'x2'],  # Outcome model missing x1^2 term
            propensity_controls=['x1', 'x2'],  # Propensity score correct
            se_method='analytical'
        )
        
        # Since the propensity score is correct, ATT should be close to the true value
        bias = abs(result.att - tau_true)
        assert bias < 0.5, (
            f"Doubly robust property failed: ATT={result.att:.4f}, "
            f"expected ≈ {tau_true}, bias={bias:.4f}"
        )
    
    def test_both_models_correct_smallest_variance(self):
        """
        Verify that variance is smallest when both models are correctly specified (maximum efficiency).
        """
        np.random.seed(456)
        n = 500
        x = np.random.normal(0, 1, n)
        
        logit_p = -0.5 + 0.3 * x
        p = 1 / (1 + np.exp(-logit_p))
        d = (np.random.uniform(0, 1, n) < p).astype(int)
        
        tau_true = 2.0
        y0 = 1 + x + np.random.normal(0, 0.5, n)
        y1 = y0 + tau_true
        y = d * y1 + (1 - d) * y0
        
        data = pd.DataFrame({'y': y, 'd': d, 'x': x})
        
        result = estimate_ipwra(
            data, 'y', 'd', ['x'],
            se_method='analytical'
        )
        
        # Point estimate should be close to the true value
        assert abs(result.att - tau_true) < 0.3
        
        # SE should be reasonable (not too large)
        assert result.se < 0.3, f"SE too large: {result.se}"
        assert result.se > 0, f"SE should be positive: {result.se}"


class TestIPWRAvsRA:
    """Tests comparing IPWRA and RA estimators."""
    
    def test_ipwra_ra_similar_in_well_specified(self):
        """Under correct specification, IPWRA and RA should produce similar results."""
        np.random.seed(42)
        n = 500
        x = np.random.normal(0, 1, n)
        
        logit_p = -0.5 + 0.3 * x
        p = 1 / (1 + np.exp(-logit_p))
        d = (np.random.uniform(0, 1, n) < p).astype(int)
        
        y0 = 1 + x + np.random.normal(0, 0.5, n)
        y1 = 3 + x + np.random.normal(0, 0.5, n)
        y = d * y1 + (1 - d) * y0
        
        data = pd.DataFrame({'y': y, 'd': d, 'x': x})
        
        # IPWRA estimation
        ipwra_result = estimate_ipwra(
            data, 'y', 'd', ['x'],
            se_method='analytical'
        )
        
        # Simple RA estimation (using OLS regression)
        data['_D_treat'] = data['d']
        from lwdid.staggered.estimation import run_ols_regression
        ra_result = run_ols_regression(
            data, 'y', '_D_treat', controls=['x']
        )
        
        # The two should be close
        diff = abs(ipwra_result.att - ra_result['att'])
        assert diff < 0.3, f"IPWRA={ipwra_result.att}, RA={ra_result['att']}, diff={diff}"


class TestIPWRAResult:
    """Tests for the IPWRAResult dataclass."""
    
    def test_result_attributes(self):
        """Verify result object attributes."""
        np.random.seed(42)
        n = 200
        x = np.random.normal(0, 1, n)
        d = np.random.binomial(1, 0.5, n)
        y = 1 + x + 2 * d + np.random.normal(0, 0.5, n)
        
        data = pd.DataFrame({'y': y, 'd': d, 'x': x})
        
        result = estimate_ipwra(data, 'y', 'd', ['x'])
        
        # Verify all attributes exist
        assert hasattr(result, 'att')
        assert hasattr(result, 'se')
        assert hasattr(result, 'ci_lower')
        assert hasattr(result, 'ci_upper')
        assert hasattr(result, 't_stat')
        assert hasattr(result, 'pvalue')
        assert hasattr(result, 'propensity_scores')
        assert hasattr(result, 'weights')
        assert hasattr(result, 'outcome_model_coef')
        assert hasattr(result, 'propensity_model_coef')
        assert hasattr(result, 'n_treated')
        assert hasattr(result, 'n_control')
        
        # Verify types
        assert isinstance(result.att, float)
        assert isinstance(result.se, float)
        assert isinstance(result.propensity_scores, np.ndarray)
        assert isinstance(result.weights, np.ndarray)
        assert isinstance(result.outcome_model_coef, dict)
        assert isinstance(result.propensity_model_coef, dict)
        assert isinstance(result.n_treated, int)
        assert isinstance(result.n_control, int)