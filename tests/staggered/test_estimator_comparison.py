"""
Estimator comparison tests for RA, IPWRA, and PSM under staggered adoption.

Compares the consistency of Regression Adjustment (RA), Inverse Probability
Weighted Regression Adjustment (IPWRA), and Propensity Score Matching (PSM)
estimators on identical staggered DiD datasets.

Validates Section 7.1 (estimator comparison) of the Lee-Wooldridge
Difference-in-Differences framework.

References
----------
Lee, S. & Wooldridge, J. M. (2025). A Simple Transformation Approach to
    Difference-in-Differences Estimation for Panel Data. SSRN 4516518.
Lee, S. & Wooldridge, J. M. (2026). Simple Approaches to Inference with
    DiD Estimators with Small Cross-Sectional Sample Sizes. SSRN 5325686.
"""

import pytest
import numpy as np
import pandas as pd


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def dgp_data():
    """
    Generate test data with known DGP.
    
    DGP: Y = 1 + 0.5*x1 + 0.3*x2 + 2.0*D + ε
    True ATT = 2.0
    """
    np.random.seed(12345)
    n = 500
    
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    
    # Propensity score depends on covariates
    ps_true = 1 / (1 + np.exp(-0.3 * x1 - 0.2 * x2))
    D = (np.random.uniform(0, 1, n) < ps_true).astype(int)
    
    # Outcome variable
    Y = 1 + 0.5 * x1 + 0.3 * x2 + 2.0 * D + np.random.normal(0, 0.5, n)
    
    return pd.DataFrame({
        'Y': Y,
        'D': D,
        'x1': x1,
        'x2': x2,
    })


@pytest.fixture
def dgp_data_no_confounding():
    """
    DGP data without confounding (propensity score independent of covariates).
    
    DGP: Y = 1 + 0.5*x1 + 0.3*x2 + 1.5*D + ε
    True ATT = 1.5
    """
    np.random.seed(54321)
    n = 400
    
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    
    # Random treatment assignment (no confounding)
    D = (np.random.uniform(0, 1, n) < 0.5).astype(int)
    
    Y = 1 + 0.5 * x1 + 0.3 * x2 + 1.5 * D + np.random.normal(0, 0.5, n)
    
    return pd.DataFrame({
        'Y': Y,
        'D': D,
        'x1': x1,
        'x2': x2,
    })


# ============================================================================
# Test: All Estimators Consistent
# ============================================================================

class TestEstimatorConsistency:
    """
    Verify that all three estimators recover the true ATT.
    """
    
    def test_all_estimators_near_true_att(self, dgp_data):
        """
        Test that all three estimators are close to the true ATT = 2.0.
        """
        from lwdid.staggered.estimation import run_ols_regression
        from lwdid.staggered.estimators import estimate_ipwra, estimate_psm
        
        true_att = 2.0
        
        # RA estimation (without controls)
        ra_result = run_ols_regression(
            data=dgp_data,
            y='Y',
            d='D',
            controls=None,
        )
        
        # RA estimation (with controls)
        ra_result_ctrl = run_ols_regression(
            data=dgp_data,
            y='Y',
            d='D',
            controls=['x1', 'x2'],
        )
        
        # IPWRA estimation
        ipwra_result = estimate_ipwra(
            data=dgp_data,
            y='Y',
            d='D',
            controls=['x1', 'x2'],
            se_method='analytical',
        )
        
        # PSM estimation
        psm_result = estimate_psm(
            data=dgp_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            n_neighbors=1,
            se_method='abadie_imbens',
        )
        
        # RA without controls may be biased (due to confounding)
        # RA with controls should be close to the true value
        assert abs(ra_result_ctrl['att'] - true_att) < 0.5, \
            f"RA(ctrl) bias too large: {ra_result_ctrl['att']:.4f}"
        
        # IPWRA should be close to the true value
        assert abs(ipwra_result.att - true_att) < 0.5, \
            f"IPWRA bias too large: {ipwra_result.att:.4f}"
        
        # PSM should be close to the true value
        assert abs(psm_result.att - true_att) < 0.5, \
            f"PSM bias too large: {psm_result.att:.4f}"
    
    def test_no_confounding_all_estimators_similar(self, dgp_data_no_confounding):
        """
        Under no confounding, all estimators should be very close.
        """
        from lwdid.staggered.estimation import run_ols_regression
        from lwdid.staggered.estimators import estimate_ipwra, estimate_psm
        
        data = dgp_data_no_confounding
        true_att = 1.5
        
        # RA (without controls)
        ra_result = run_ols_regression(
            data=data,
            y='Y',
            d='D',
            controls=None,
        )
        
        # IPWRA
        ipwra_result = estimate_ipwra(
            data=data,
            y='Y',
            d='D',
            controls=['x1', 'x2'],
            se_method='analytical',
        )
        
        # PSM
        psm_result = estimate_psm(
            data=data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            n_neighbors=1,
        )
        
        # Without confounding, RA without controls should also be close to the true value
        assert abs(ra_result['att'] - true_att) < 0.3, \
            f"RA bias too large: {ra_result['att']:.4f}"
        
        assert abs(ipwra_result.att - true_att) < 0.3, \
            f"IPWRA bias too large: {ipwra_result.att:.4f}"
        
        assert abs(psm_result.att - true_att) < 0.3, \
            f"PSM bias too large: {psm_result.att:.4f}"


# ============================================================================
# Test: SE Comparison
# ============================================================================

class TestSEComparison:
    """
    Standard error comparison tests.
    """
    
    def test_se_all_positive(self, dgp_data):
        """
        All estimators should produce positive standard errors.
        """
        from lwdid.staggered.estimation import run_ols_regression
        from lwdid.staggered.estimators import estimate_ipwra, estimate_psm
        
        ra_result = run_ols_regression(
            data=dgp_data,
            y='Y',
            d='D',
            controls=['x1', 'x2'],
        )
        
        ipwra_result = estimate_ipwra(
            data=dgp_data,
            y='Y',
            d='D',
            controls=['x1', 'x2'],
        )
        
        psm_result = estimate_psm(
            data=dgp_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
        )
        
        assert ra_result['se'] > 0
        assert ipwra_result.se > 0
        assert psm_result.se > 0
    
    def test_se_same_order_of_magnitude(self, dgp_data):
        """
        Standard errors from different estimators should be of the same order of magnitude.
        """
        from lwdid.staggered.estimation import run_ols_regression
        from lwdid.staggered.estimators import estimate_ipwra, estimate_psm
        
        ra_result = run_ols_regression(
            data=dgp_data,
            y='Y',
            d='D',
            controls=['x1', 'x2'],
        )
        
        ipwra_result = estimate_ipwra(
            data=dgp_data,
            y='Y',
            d='D',
            controls=['x1', 'x2'],
        )
        
        psm_result = estimate_psm(
            data=dgp_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
        )
        
        # SEs should be of the same order of magnitude (within 0.3x to 3x range)
        se_ra = ra_result['se']
        se_ipwra = ipwra_result.se
        se_psm = psm_result.se
        
        assert 0.3 < se_ipwra / se_ra < 3, f"IPWRA vs RA SE ratio abnormal: {se_ipwra/se_ra}"
        assert 0.3 < se_psm / se_ra < 3, f"PSM vs RA SE ratio abnormal: {se_psm/se_ra}"


# ============================================================================
# Test: Coverage
# ============================================================================

class TestCoverage:
    """
    Confidence interval coverage tests.
    """
    
    def test_ci_contains_true_att(self, dgp_data):
        """
        Confidence intervals should contain the true ATT (in most cases).
        """
        from lwdid.staggered.estimation import run_ols_regression
        from lwdid.staggered.estimators import estimate_ipwra, estimate_psm
        
        true_att = 2.0
        
        ra_result = run_ols_regression(
            data=dgp_data,
            y='Y',
            d='D',
            controls=['x1', 'x2'],
        )
        
        ipwra_result = estimate_ipwra(
            data=dgp_data,
            y='Y',
            d='D',
            controls=['x1', 'x2'],
        )
        
        psm_result = estimate_psm(
            data=dgp_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
        )
        
        # Verify CI contains the true value
        # Due to random data, it may not always contain it, but should in most cases
        ra_covers = ra_result['ci_lower'] < true_att < ra_result['ci_upper']
        ipwra_covers = ipwra_result.ci_lower < true_att < ipwra_result.ci_upper
        psm_covers = psm_result.ci_lower < true_att < psm_result.ci_upper
        
        # At least some estimators' CIs should contain the true value
        n_covers = int(ra_covers) + int(ipwra_covers) + int(psm_covers)
        # Due to randomness, not all may cover, but at least 1 should
        assert n_covers >= 1, f"Too few CIs contain the true value: {n_covers}/3"


# ============================================================================
# Test: Robustness
# ============================================================================

class TestRobustness:
    """
    Robustness tests.
    """
    
    def test_small_sample(self):
        """
        Small sample test.
        """
        from lwdid.staggered.estimators import estimate_psm
        
        np.random.seed(999)
        n = 50
        
        x1 = np.random.normal(0, 1, n)
        D = (np.random.uniform(0, 1, n) < 0.5).astype(int)
        Y = 1 + D + np.random.normal(0, 0.5, n)
        
        data = pd.DataFrame({'Y': Y, 'D': D, 'x1': x1})
        
        result = estimate_psm(
            data=data,
            y='Y',
            d='D',
            propensity_controls=['x1'],
            n_neighbors=1,
        )
        
        # Should run successfully
        assert result.att is not None
        assert result.n_treated > 0
    
    def test_imbalanced_groups(self):
        """
        Imbalanced treatment group test.
        """
        from lwdid.staggered.estimators import estimate_psm
        
        np.random.seed(888)
        n = 100
        
        x1 = np.random.normal(0, 1, n)
        # Imbalanced: low treatment proportion
        D = (np.random.uniform(0, 1, n) < 0.2).astype(int)
        Y = 1 + D + np.random.normal(0, 0.5, n)
        
        data = pd.DataFrame({'Y': Y, 'D': D, 'x1': x1})
        
        result = estimate_psm(
            data=data,
            y='Y',
            d='D',
            propensity_controls=['x1'],
            n_neighbors=1,
        )
        
        assert result.att is not None
        # Treatment group should be small
        assert result.n_treated < result.n_control


# ============================================================================
# Test: Numerical Values Check
# ============================================================================

@pytest.mark.stata_alignment
class TestNumericalCheck:
    """
    Numerical validation tests.
    """
    
    def test_no_nan_in_results(self, dgp_data):
        """
        Results should not contain NaN (except in boundary cases).
        """
        from lwdid.staggered.estimators import estimate_psm
        
        result = estimate_psm(
            data=dgp_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
        )
        
        assert not np.isnan(result.att)
        assert not np.isnan(result.se)
        assert not np.isnan(result.t_stat)
        assert not np.isnan(result.pvalue)
        assert not np.isnan(result.ci_lower)
        assert not np.isnan(result.ci_upper)
    
    def test_pvalue_in_valid_range(self, dgp_data):
        """
        P-values should be in the [0, 1] range.
        """
        from lwdid.staggered.estimators import estimate_psm
        
        result = estimate_psm(
            data=dgp_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
        )
        
        assert 0 <= result.pvalue <= 1
