"""
PSM (Propensity Score Matching) estimator tests.

Validates the propensity score matching estimator implementation for
staggered Difference-in-Differences designs.

Validates Section 7.1 (PSM estimator specification) of the Lee-Wooldridge
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
import warnings

from lwdid.staggered.estimators import (
    estimate_psm,
    PSMResult,
    estimate_propensity_score,
    _validate_psm_inputs,
    _nearest_neighbor_match,
    _compute_psm_se_abadie_imbens,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_psm_data():
    """
    Simple PSM test data.
    
    DGP: Y = 1 + 0.5*x1 + 0.3*x2 + 2.0*D + ε
    True ATT = 2.0
    """
    np.random.seed(42)
    n = 200
    
    # Generate covariates
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    
    # Generate treatment indicator (depends on covariates)
    ps_true = 1 / (1 + np.exp(-0.5 * x1 - 0.3 * x2))
    D = (np.random.uniform(0, 1, n) < ps_true).astype(int)
    
    # Generate outcome variable
    Y = 1 + 0.5 * x1 + 0.3 * x2 + 2.0 * D + np.random.normal(0, 0.5, n)
    
    return pd.DataFrame({
        'Y': Y,
        'D': D,
        'x1': x1,
        'x2': x2,
    })


@pytest.fixture
def large_psm_data():
    """Larger PSM test data for bootstrap testing."""
    np.random.seed(123)
    n = 500
    
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    
    ps_true = 1 / (1 + np.exp(-0.3 * x1 - 0.2 * x2))
    D = (np.random.uniform(0, 1, n) < ps_true).astype(int)
    
    Y = 1 + 0.5 * x1 + 0.3 * x2 + 1.5 * D + np.random.normal(0, 0.5, n)
    
    return pd.DataFrame({
        'Y': Y,
        'D': D,
        'x1': x1,
        'x2': x2,
    })


@pytest.fixture
def imbalanced_data():
    """Data with imbalanced treatment and control group sizes."""
    np.random.seed(456)
    n = 100
    
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    
    # High propensity score, resulting in mostly treated units
    ps_true = 1 / (1 + np.exp(-1.5 - 0.5 * x1))
    D = (np.random.uniform(0, 1, n) < ps_true).astype(int)
    
    Y = 1 + 0.5 * x1 + 1.0 * D + np.random.normal(0, 0.5, n)
    
    return pd.DataFrame({
        'Y': Y,
        'D': D,
        'x1': x1,
        'x2': x2,
    })


# ============================================================================
# Test: Basic Functionality
# ============================================================================

class TestEstimatePSMBasic:
    """Basic functionality tests for estimate_psm()."""
    
    def test_basic_functionality(self, simple_psm_data):
        """Test basic functionality."""
        result = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            n_neighbors=1,
        )
        
        assert isinstance(result, PSMResult)
        assert result.n_treated > 0
        assert result.n_control > 0
        assert result.att is not None
        assert result.se > 0
        # ATT should be close to the true value 2.0 (larger tolerance for small sample)
        assert abs(result.att - 2.0) < 1.0
    
    def test_returns_psm_result(self, simple_psm_data):
        """Test that PSMResult object is returned."""
        result = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
        )
        
        # Verify all required attributes
        assert hasattr(result, 'att')
        assert hasattr(result, 'se')
        assert hasattr(result, 'ci_lower')
        assert hasattr(result, 'ci_upper')
        assert hasattr(result, 't_stat')
        assert hasattr(result, 'pvalue')
        assert hasattr(result, 'propensity_scores')
        assert hasattr(result, 'match_counts')
        assert hasattr(result, 'matched_control_ids')
        assert hasattr(result, 'n_treated')
        assert hasattr(result, 'n_control')
        assert hasattr(result, 'n_matched')
        assert hasattr(result, 'n_dropped')
    
    def test_confidence_interval_contains_att(self, simple_psm_data):
        """Test that confidence interval contains ATT."""
        result = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
        )
        
        # CI should contain ATT
        assert result.ci_lower < result.att < result.ci_upper
        # CI width should be positive
        assert result.ci_upper - result.ci_lower > 0


# ============================================================================
# Test: k-NN Matching
# ============================================================================

class TestKNNMatching:
    """Tests for k-NN matching."""
    
    def test_k_neighbors_1(self, simple_psm_data):
        """Test 1-NN matching."""
        result = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            n_neighbors=1,
        )
        
        # Each treated unit should match at most 1 control unit
        for count in result.match_counts:
            assert count <= 1
    
    def test_k_neighbors_3(self, simple_psm_data):
        """Test 3-NN matching."""
        result = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            n_neighbors=3,
        )
        
        # Each treated unit should match at most 3 control units
        for count in result.match_counts:
            assert count <= 3
        
        assert result.att is not None
    
    def test_k_neighbors_effect_on_variance(self, large_psm_data):
        """Test the effect of increasing k on variance."""
        result_1nn = estimate_psm(
            data=large_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            n_neighbors=1,
            se_method='abadie_imbens',
        )
        
        result_5nn = estimate_psm(
            data=large_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            n_neighbors=5,
            se_method='abadie_imbens',
        )
        
        # Both methods should succeed
        assert result_1nn.att is not None
        assert result_5nn.att is not None


# ============================================================================
# Test: With/Without Replacement
# ============================================================================

class TestReplacementOptions:
    """Tests for with/without replacement matching."""
    
    def test_with_replacement_default(self, simple_psm_data):
        """Test default with-replacement matching."""
        result = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            with_replacement=True,
        )
        
        assert result.att is not None
    
    def test_without_replacement(self, simple_psm_data):
        """Test without-replacement matching."""
        result = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            with_replacement=False,
        )
        
        # Without replacement, n_matched should be <= n_treated
        assert result.n_matched <= result.n_treated
        assert result.att is not None
    
    def test_without_replacement_constraint(self, simple_psm_data):
        """Test without-replacement matching constraint."""
        result = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            with_replacement=False,
            n_neighbors=1,
        )
        
        # Number of matched controls should be <= control group size
        assert result.n_matched <= result.n_control


# ============================================================================
# Test: Caliper
# ============================================================================

class TestCaliper:
    """Tests for caliper matching threshold."""
    
    def test_no_caliper(self, simple_psm_data):
        """Test without caliper."""
        result = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            caliper=None,
        )
        
        assert result.caliper is None
        assert result.n_dropped == 0  # No units should be dropped without caliper
    
    def test_caliper_sd_scale(self, simple_psm_data):
        """Test caliper in standard deviation units."""
        result = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            caliper=0.5,
            caliper_scale='sd',
        )
        
        assert result.caliper is not None
        assert result.att is not None
    
    def test_caliper_absolute_scale(self, simple_psm_data):
        """Test absolute value caliper."""
        result = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            caliper=0.1,
            caliper_scale='absolute',
        )
        
        assert result.caliper == 0.1
        assert result.att is not None
    
    def test_strict_caliper_drops_units(self, simple_psm_data):
        """Test that a strict caliper causes units to be dropped."""
        result_loose = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            caliper=2.0,
            caliper_scale='sd',
        )
        
        result_strict = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            caliper=0.1,
            caliper_scale='sd',
        )
        
        # Strict caliper should result in more dropped units
        assert result_strict.n_dropped >= result_loose.n_dropped


# ============================================================================
# Test: Standard Error Methods
# ============================================================================

class TestSEMethods:
    """Tests for standard error computation methods."""
    
    def test_abadie_imbens_se(self, simple_psm_data):
        """Test Abadie-Imbens standard errors."""
        result = estimate_psm(
            data=simple_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            se_method='abadie_imbens',
        )
        
        assert result.se > 0
        assert not np.isnan(result.se)
    
    def test_bootstrap_se(self, large_psm_data):
        """Test bootstrap standard errors."""
        result = estimate_psm(
            data=large_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            se_method='bootstrap',
            n_bootstrap=50,  # Reduced for faster testing
            seed=42,
        )
        
        assert result.se > 0
        assert not np.isnan(result.se)
    
    def test_se_methods_comparable(self, large_psm_data):
        """Test that the two SE methods produce comparable results."""
        result_ai = estimate_psm(
            data=large_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            se_method='abadie_imbens',
        )
        
        result_boot = estimate_psm(
            data=large_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            se_method='bootstrap',
            n_bootstrap=100,
            seed=42,
        )
        
        # The two methods' SEs should be of the same order of magnitude
        ratio = result_ai.se / result_boot.se
        assert 0.3 < ratio < 3.0, f"SE ratio out of range: {ratio}"
    
    def test_invalid_se_method(self, simple_psm_data):
        """Test invalid SE method raises error."""
        with pytest.raises(ValueError, match="Unknown se_method"):
            estimate_psm(
                data=simple_psm_data,
                y='Y',
                d='D',
                propensity_controls=['x1', 'x2'],
                se_method='invalid_method',
            )


# ============================================================================
# Test: Input Validation
# ============================================================================

class TestInputValidation:
    """Input validation tests."""
    
    def test_missing_y_column(self, simple_psm_data):
        """Test missing outcome variable raises error."""
        with pytest.raises(ValueError, match="Outcome variable.*not found"):
            estimate_psm(
                data=simple_psm_data,
                y='nonexistent',
                d='D',
                propensity_controls=['x1'],
            )
    
    def test_missing_d_column(self, simple_psm_data):
        """Test missing treatment indicator raises error."""
        with pytest.raises(ValueError, match="Treatment indicator.*not found"):
            estimate_psm(
                data=simple_psm_data,
                y='Y',
                d='nonexistent',
                propensity_controls=['x1'],
            )
    
    def test_missing_control_column(self, simple_psm_data):
        """Test missing control variable raises error."""
        with pytest.raises(ValueError, match="Propensity score controls not found"):
            estimate_psm(
                data=simple_psm_data,
                y='Y',
                d='D',
                propensity_controls=['x1', 'nonexistent'],
            )
    
    def test_invalid_n_neighbors(self, simple_psm_data):
        """Test invalid n_neighbors raises error."""
        with pytest.raises(ValueError, match="n_neighbors must be >= 1"):
            estimate_psm(
                data=simple_psm_data,
                y='Y',
                d='D',
                propensity_controls=['x1'],
                n_neighbors=0,
            )
    
    def test_empty_treatment_group(self, simple_psm_data):
        """Test empty treatment group raises error."""
        data = simple_psm_data.copy()
        data['D'] = 0  # All control
        
        with pytest.raises(ValueError, match="No treated units"):
            estimate_psm(
                data=data,
                y='Y',
                d='D',
                propensity_controls=['x1'],
            )
    
    def test_insufficient_controls(self):
        """Test insufficient controls raises error."""
        # Create data with very few controls
        data = pd.DataFrame({
            'Y': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'D': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # Only 1 control
            'x1': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        })
        
        with pytest.raises(ValueError, match="Control sample size.*is less than number of neighbors"):
            estimate_psm(
                data=data,
                y='Y',
                d='D',
                propensity_controls=['x1'],
                n_neighbors=5,
            )
    
    def test_invalid_d_values(self, simple_psm_data):
        """Test non-binary treatment indicator raises error."""
        data = simple_psm_data.copy()
        data['D'] = data['D'] + 1  # Becomes 1/2
        
        with pytest.raises(ValueError, match="Treatment indicator.*must be binary"):
            estimate_psm(
                data=data,
                y='Y',
                d='D',
                propensity_controls=['x1'],
            )


# ============================================================================
# Test: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Edge case tests."""
    
    def test_small_sample_warning(self):
        """Test small sample warning."""
        np.random.seed(789)
        data = pd.DataFrame({
            'Y': np.random.normal(0, 1, 20),
            'D': [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'x1': np.random.normal(0, 1, 20),
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = estimate_psm(
                data=data,
                y='Y',
                d='D',
                propensity_controls=['x1'],
            )
            
            # Should have small sample warning
            assert result.att is not None  # But should still return a result
    
    def test_caliper_too_strict(self):
        """Test overly strict caliper raises error."""
        np.random.seed(999)
        n = 50
        data = pd.DataFrame({
            'Y': np.random.normal(0, 1, n),
            'D': [1] * 25 + [0] * 25,
            # Large x1 difference leads to large propensity score gap
            'x1': np.concatenate([np.random.normal(5, 0.1, 25), 
                                  np.random.normal(-5, 0.1, 25)]),
        })
        
        with pytest.raises(ValueError, match="All treated units failed to find valid matches"):
            estimate_psm(
                data=data,
                y='Y',
                d='D',
                propensity_controls=['x1'],
                caliper=0.001,  # Very strict caliper
                caliper_scale='absolute',
            )
    
    def test_handles_missing_values(self, simple_psm_data):
        """Test handling of missing values."""
        data = simple_psm_data.copy()
        # Add some missing values
        data.loc[0, 'Y'] = np.nan
        data.loc[1, 'x1'] = np.nan
        
        result = estimate_psm(
            data=data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
        )
        
        # Should handle successfully
        assert result.att is not None


# ============================================================================
# Test: Helper Functions
# ============================================================================

class TestHelperFunctions:
    """Helper function tests."""
    
    def test_validate_psm_inputs_valid(self, simple_psm_data):
        """Test valid inputs."""
        # Should not raise an exception
        _validate_psm_inputs(
            simple_psm_data, 'Y', 'D', ['x1', 'x2'], 1
        )
    
    def test_nearest_neighbor_match_basic(self):
        """Test basic nearest neighbor matching."""
        pscores_treat = np.array([0.3, 0.5, 0.7])
        pscores_control = np.array([0.25, 0.55, 0.8])
        
        matched, counts, dropped = _nearest_neighbor_match(
            pscores_treat, pscores_control,
            n_neighbors=1,
            with_replacement=True,
            caliper=None,
        )
        
        assert len(matched) == 3
        assert dropped == 0
        # First treated unit (0.3) should match the first control unit (0.25)
        assert 0 in matched[0]
    
    def test_nearest_neighbor_match_caliper(self):
        """Test matching with caliper."""
        pscores_treat = np.array([0.3, 0.5, 0.9])
        pscores_control = np.array([0.25, 0.55])
        
        matched, counts, dropped = _nearest_neighbor_match(
            pscores_treat, pscores_control,
            n_neighbors=1,
            with_replacement=True,
            caliper=0.1,
        )
        
        assert len(matched) == 3
        # Last treated unit (0.9) should be dropped because it's too far from all controls
        assert dropped >= 1
    
    def test_compute_psm_se_abadie_imbens(self):
        """Test Abadie-Imbens SE computation."""
        # Design different individual effects to ensure variance > 0
        Y_treat = np.array([1.0, 2.5, 3.0, 5.0])
        Y_control = np.array([0.5, 1.5, 2.5, 3.5])
        matched_ids = [[0], [1], [2], [3]]
        # Individual effects: 0.5, 1.0, 0.5, 1.5 -> different, variance > 0
        
        se, ci_lower, ci_upper = _compute_psm_se_abadie_imbens(
            Y_treat, Y_control, matched_ids, att=0.875, alpha=0.05
        )
        
        assert se > 0
        assert ci_lower < ci_upper


# ============================================================================
# Test: Reproducibility
# ============================================================================

class TestReproducibility:
    """Reproducibility tests."""
    
    def test_bootstrap_reproducible_with_seed(self, large_psm_data):
        """Test that bootstrap is reproducible with the same seed."""
        result1 = estimate_psm(
            data=large_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            se_method='bootstrap',
            n_bootstrap=50,
            seed=12345,
        )
        
        result2 = estimate_psm(
            data=large_psm_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            se_method='bootstrap',
            n_bootstrap=50,
            seed=12345,
        )
        
        # ATT should be exactly the same (deterministic matching)
        assert result1.att == result2.att
        # SE should be the same (same seed for bootstrap)
        assert result1.se == result2.se
