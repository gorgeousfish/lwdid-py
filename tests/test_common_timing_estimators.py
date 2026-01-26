"""
Story 1.1: Common Timing IPW/IPWRA/PSM Estimators Unit Tests

This module tests the IPW, IPWRA, and PSM estimators for common timing
difference-in-differences scenarios.

Tests cover:
1. Estimator routing logic - correct estimator selection
2. Parameter validation - estimator, ps_controls, controls requirements
3. Result object structure - all expected fields present
4. Boundary conditions - small samples, extreme propensity scores
5. Numerical validation - results match expected values

References
----------
Lee & Wooldridge (2023), Section 3, Procedure 3.1
"""

import numpy as np
import pandas as pd
import pytest
import warnings

from lwdid import lwdid


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def common_timing_panel_data():
    """
    Create simulated panel data for common timing DiD.
    
    Design:
    - N = 100 units (40 treated, 60 control)
    - T = 6 periods (1-3 pre-treatment, 4-6 post-treatment)
    - Two control variables: x1 (continuous), x2 (binary)
    - True ATT = 3.0 for all post-treatment periods
    """
    np.random.seed(42)
    
    n_units = 100
    n_periods = 6
    n_treated = 40
    s = 4  # First treatment period
    
    # Unit-level data
    unit_ids = np.arange(1, n_units + 1)
    treated = np.zeros(n_units)
    treated[:n_treated] = 1
    
    # Control variables
    x1 = np.random.normal(4, 1, n_units)  # Continuous
    x2 = np.random.binomial(1, 0.6, n_units)  # Binary
    
    # Unit-specific effect (correlated with treatment for selection bias)
    c = 2 + 0.5 * treated + np.random.normal(0, 0.5, n_units)
    
    # Build panel
    records = []
    true_att = 3.0
    
    for i, unit_id in enumerate(unit_ids):
        for t in range(1, n_periods + 1):
            # Time effect
            delta_t = t
            
            # Covariate effect
            beta_t = 1.0 + 0.1 * t
            x_effect = beta_t * (x1[i] - 4) / 3 + beta_t * x2[i] / 2
            
            # Idiosyncratic error
            eps = np.random.normal(0, 1)
            
            # Potential outcome Y(0)
            y0 = delta_t + c[i] + x_effect + eps
            
            # Post-treatment indicator
            post = 1 if t >= s else 0
            
            # Treatment effect
            tau = true_att if (treated[i] == 1 and post == 1) else 0
            
            # Observed outcome
            y = y0 + tau
            
            records.append({
                'id': unit_id,
                'year': 2000 + t,
                'y': y,
                'd': treated[i],
                'post': post,
                'x1': x1[i],
                'x2': x2[i],
            })
    
    df = pd.DataFrame(records)
    df['d'] = df['d'].astype(int)
    df['post'] = df['post'].astype(int)
    
    return df


@pytest.fixture
def small_sample_data():
    """
    Create small sample data for boundary testing.
    
    Design:
    - N = 20 units (8 treated, 12 control)
    - T = 4 periods (2 pre, 2 post)
    """
    np.random.seed(123)
    
    n_units = 20
    n_periods = 4
    n_treated = 8
    s = 3
    
    unit_ids = np.arange(1, n_units + 1)
    treated = np.zeros(n_units)
    treated[:n_treated] = 1
    
    x1 = np.random.normal(4, 1, n_units)
    x2 = np.random.binomial(1, 0.5, n_units)
    
    records = []
    true_att = 2.0
    
    for i, unit_id in enumerate(unit_ids):
        for t in range(1, n_periods + 1):
            y0 = t + 0.5 * x1[i] + np.random.normal(0, 0.5)
            post = 1 if t >= s else 0
            tau = true_att if (treated[i] == 1 and post == 1) else 0
            y = y0 + tau
            
            records.append({
                'id': unit_id,
                'year': 2000 + t,
                'y': y,
                'd': treated[i],
                'post': post,
                'x1': x1[i],
                'x2': x2[i],
            })
    
    df = pd.DataFrame(records)
    df['d'] = df['d'].astype(int)
    df['post'] = df['post'].astype(int)
    
    return df


# =============================================================================
# Test: Estimator Routing Logic
# =============================================================================

class TestEstimatorRouting:
    """Test that the correct estimator is called based on the estimator parameter."""
    
    def test_ra_estimator_default(self, common_timing_panel_data):
        """Test that RA is the default estimator."""
        result = lwdid(
            data=common_timing_panel_data,
            y='y',
            d='d',
            ivar='id',
            tvar='year',
            post='post',
            rolling='demean',
        )
        
        assert result is not None
        assert hasattr(result, 'att')
        assert not np.isnan(result.att)
    
    def test_ipw_estimator(self, common_timing_panel_data):
        """Test IPW estimator in common timing mode."""
        result = lwdid(
            data=common_timing_panel_data,
            y='y',
            d='d',
            ivar='id',
            tvar='year',
            post='post',
            rolling='demean',
            estimator='ipw',
            controls=['x1', 'x2'],
        )
        
        assert result is not None
        assert hasattr(result, 'att')
        assert not np.isnan(result.att)
        # IPW should give reasonable ATT estimate
        assert abs(result.att - 3.0) < 2.0  # Within 2 of true ATT
    
    def test_ipwra_estimator(self, common_timing_panel_data):
        """Test IPWRA estimator in common timing mode."""
        result = lwdid(
            data=common_timing_panel_data,
            y='y',
            d='d',
            ivar='id',
            tvar='year',
            post='post',
            rolling='demean',
            estimator='ipwra',
            controls=['x1', 'x2'],
        )
        
        assert result is not None
        assert hasattr(result, 'att')
        assert not np.isnan(result.att)
        # IPWRA should give reasonable ATT estimate (doubly robust)
        assert abs(result.att - 3.0) < 1.5  # Closer to true ATT
    
    def test_psm_estimator(self, common_timing_panel_data):
        """Test PSM estimator in common timing mode."""
        result = lwdid(
            data=common_timing_panel_data,
            y='y',
            d='d',
            ivar='id',
            tvar='year',
            post='post',
            rolling='demean',
            estimator='psm',
            controls=['x1', 'x2'],
        )
        
        assert result is not None
        assert hasattr(result, 'att')
        assert not np.isnan(result.att)
        # PSM should give reasonable ATT estimate
        assert abs(result.att - 3.0) < 2.5  # More variance than IPWRA
    
    def test_estimator_case_insensitive(self, common_timing_panel_data):
        """Test that estimator parameter is case-insensitive."""
        for est_name in ['IPW', 'Ipw', 'ipw', 'IPWRA', 'IpwRa', 'PSM', 'Psm']:
            result = lwdid(
                data=common_timing_panel_data,
                y='y',
                d='d',
                ivar='id',
                tvar='year',
                post='post',
                rolling='demean',
                estimator=est_name,
                controls=['x1', 'x2'],
            )
            assert result is not None


# =============================================================================
# Test: Parameter Validation
# =============================================================================

class TestParameterValidation:
    """Test parameter validation for IPW/IPWRA/PSM in common timing mode."""
    
    def test_invalid_estimator_raises_error(self, common_timing_panel_data):
        """Test that invalid estimator raises ValueError."""
        with pytest.raises(ValueError, match="Invalid estimator"):
            lwdid(
                data=common_timing_panel_data,
                y='y',
                d='d',
                ivar='id',
                tvar='year',
                post='post',
                rolling='demean',
                estimator='invalid_estimator',
            )
    
    def test_ipw_requires_controls(self, common_timing_panel_data):
        """Test that IPW requires controls parameter."""
        with pytest.raises(ValueError, match="requires 'controls' or 'ps_controls' parameter"):
            lwdid(
                data=common_timing_panel_data,
                y='y',
                d='d',
                ivar='id',
                tvar='year',
                post='post',
                rolling='demean',
                estimator='ipw',
                controls=None,
            )
    
    def test_ipwra_requires_controls(self, common_timing_panel_data):
        """Test that IPWRA requires controls parameter."""
        with pytest.raises(ValueError, match="requires 'controls' parameter"):
            lwdid(
                data=common_timing_panel_data,
                y='y',
                d='d',
                ivar='id',
                tvar='year',
                post='post',
                rolling='demean',
                estimator='ipwra',
            )
    
    def test_psm_requires_controls(self, common_timing_panel_data):
        """Test that PSM requires controls parameter."""
        with pytest.raises(ValueError, match="requires 'controls' or 'ps_controls' parameter"):
            lwdid(
                data=common_timing_panel_data,
                y='y',
                d='d',
                ivar='id',
                tvar='year',
                post='post',
                rolling='demean',
                estimator='psm',
            )
    
    def test_ps_controls_separate_from_controls(self, common_timing_panel_data):
        """Test that ps_controls can be specified separately from controls."""
        result = lwdid(
            data=common_timing_panel_data,
            y='y',
            d='d',
            ivar='id',
            tvar='year',
            post='post',
            rolling='demean',
            estimator='ipwra',
            controls=['x1', 'x2'],
            ps_controls=['x1'],  # Only x1 for PS model
        )
        
        assert result is not None
        assert not np.isnan(result.att)
    
    def test_ra_ignores_ipw_params_with_warning(self, common_timing_panel_data):
        """Test that RA estimator ignores IPW parameters with warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = lwdid(
                data=common_timing_panel_data,
                y='y',
                d='d',
                ivar='id',
                tvar='year',
                post='post',
                rolling='demean',
                estimator='ra',
                controls=['x1', 'x2'],
                trim_threshold=0.05,  # Should be ignored for RA
            )
            
            # Should have warning about ignored parameters
            assert any("ignored" in str(warning.message).lower() for warning in w)
        
        assert result is not None


# =============================================================================
# Test: Result Object Structure
# =============================================================================

class TestResultStructure:
    """Test that result objects have correct structure for all estimators."""
    
    def test_ipw_result_has_required_fields(self, common_timing_panel_data):
        """Test IPW result has all required fields."""
        result = lwdid(
            data=common_timing_panel_data,
            y='y',
            d='d',
            ivar='id',
            tvar='year',
            post='post',
            rolling='demean',
            estimator='ipw',
            controls=['x1', 'x2'],
        )
        
        # Check main result attributes
        assert hasattr(result, 'att')
        assert hasattr(result, 'se_att')
        assert hasattr(result, 't_stat')
        assert hasattr(result, 'pvalue')
        assert hasattr(result, 'ci_lower')
        assert hasattr(result, 'ci_upper')
        assert hasattr(result, 'nobs')
        
        # Check period-specific effects
        assert hasattr(result, 'att_by_period')
        assert isinstance(result.att_by_period, pd.DataFrame)
        assert 'beta' in result.att_by_period.columns
        assert 'se' in result.att_by_period.columns
    
    def test_ipwra_result_has_required_fields(self, common_timing_panel_data):
        """Test IPWRA result has all required fields."""
        result = lwdid(
            data=common_timing_panel_data,
            y='y',
            d='d',
            ivar='id',
            tvar='year',
            post='post',
            rolling='demean',
            estimator='ipwra',
            controls=['x1', 'x2'],
        )
        
        assert hasattr(result, 'att')
        assert hasattr(result, 'se_att')
        assert hasattr(result, 'att_by_period')
    
    def test_psm_result_has_required_fields(self, common_timing_panel_data):
        """Test PSM result has all required fields."""
        result = lwdid(
            data=common_timing_panel_data,
            y='y',
            d='d',
            ivar='id',
            tvar='year',
            post='post',
            rolling='demean',
            estimator='psm',
            controls=['x1', 'x2'],
        )
        
        assert hasattr(result, 'att')
        assert hasattr(result, 'se_att')
        assert hasattr(result, 'att_by_period')
    
    def test_period_effects_structure(self, common_timing_panel_data):
        """Test that period effects DataFrame has correct structure."""
        result = lwdid(
            data=common_timing_panel_data,
            y='y',
            d='d',
            ivar='id',
            tvar='year',
            post='post',
            rolling='demean',
            estimator='ipw',
            controls=['x1', 'x2'],
        )
        
        period_df = result.att_by_period
        
        # Check required columns
        required_cols = ['period', 'tindex', 'beta', 'se', 'ci_lower', 'ci_upper', 'tstat', 'pval', 'N']
        for col in required_cols:
            assert col in period_df.columns, f"Missing column: {col}"
        
        # Check that we have average + period-specific effects
        # Average should be first row
        assert period_df.iloc[0]['period'] == 'average'
        
        # Post-treatment periods should follow
        post_periods = period_df[period_df['period'] != 'average']
        assert len(post_periods) >= 1


# =============================================================================
# Test: PSM-Specific Parameters
# =============================================================================

class TestPSMParameters:
    """Test PSM-specific parameters in common timing mode."""
    
    def test_psm_n_neighbors(self, common_timing_panel_data):
        """Test PSM with different n_neighbors values."""
        for k in [1, 3, 5]:
            result = lwdid(
                data=common_timing_panel_data,
                y='y',
                d='d',
                ivar='id',
                tvar='year',
                post='post',
                rolling='demean',
                estimator='psm',
                controls=['x1', 'x2'],
                n_neighbors=k,
            )
            assert result is not None
            assert not np.isnan(result.att)
    
    def test_psm_caliper(self, common_timing_panel_data):
        """Test PSM with caliper."""
        result = lwdid(
            data=common_timing_panel_data,
            y='y',
            d='d',
            ivar='id',
            tvar='year',
            post='post',
            rolling='demean',
            estimator='psm',
            controls=['x1', 'x2'],
            caliper=0.25,
        )
        assert result is not None
    
    def test_psm_without_replacement(self, common_timing_panel_data):
        """Test PSM without replacement."""
        result = lwdid(
            data=common_timing_panel_data,
            y='y',
            d='d',
            ivar='id',
            tvar='year',
            post='post',
            rolling='demean',
            estimator='psm',
            controls=['x1', 'x2'],
            with_replacement=False,
        )
        assert result is not None
    
    def test_psm_match_order(self, common_timing_panel_data):
        """Test PSM with different match orders."""
        for order in ['data', 'random', 'largest', 'smallest']:
            result = lwdid(
                data=common_timing_panel_data,
                y='y',
                d='d',
                ivar='id',
                tvar='year',
                post='post',
                rolling='demean',
                estimator='psm',
                controls=['x1', 'x2'],
                with_replacement=False,
                match_order=order,
            )
            assert result is not None


# =============================================================================
# Test: Trim Threshold
# =============================================================================

class TestTrimThreshold:
    """Test propensity score trimming in common timing mode."""
    
    def test_ipw_trim_threshold(self, common_timing_panel_data):
        """Test IPW with different trim thresholds."""
        for threshold in [0.01, 0.05, 0.1]:
            result = lwdid(
                data=common_timing_panel_data,
                y='y',
                d='d',
                ivar='id',
                tvar='year',
                post='post',
                rolling='demean',
                estimator='ipw',
                controls=['x1', 'x2'],
                trim_threshold=threshold,
            )
            assert result is not None
            assert not np.isnan(result.att)
    
    def test_ipwra_trim_threshold(self, common_timing_panel_data):
        """Test IPWRA with different trim thresholds."""
        result = lwdid(
            data=common_timing_panel_data,
            y='y',
            d='d',
            ivar='id',
            tvar='year',
            post='post',
            rolling='demean',
            estimator='ipwra',
            controls=['x1', 'x2'],
            trim_threshold=0.05,
        )
        assert result is not None


# =============================================================================
# Test: Small Sample Behavior
# =============================================================================

class TestSmallSample:
    """Test estimator behavior with small samples."""
    
    def test_ipw_small_sample(self, small_sample_data):
        """Test IPW with small sample."""
        result = lwdid(
            data=small_sample_data,
            y='y',
            d='d',
            ivar='id',
            tvar='year',
            post='post',
            rolling='demean',
            estimator='ipw',
            controls=['x1', 'x2'],
        )
        
        assert result is not None
        # Small sample may have larger SE
        assert result.se_att > 0
    
    def test_ipwra_small_sample(self, small_sample_data):
        """Test IPWRA with small sample."""
        result = lwdid(
            data=small_sample_data,
            y='y',
            d='d',
            ivar='id',
            tvar='year',
            post='post',
            rolling='demean',
            estimator='ipwra',
            controls=['x1', 'x2'],
        )
        
        assert result is not None
    
    def test_psm_small_sample(self, small_sample_data):
        """Test PSM with small sample."""
        result = lwdid(
            data=small_sample_data,
            y='y',
            d='d',
            ivar='id',
            tvar='year',
            post='post',
            rolling='demean',
            estimator='psm',
            controls=['x1', 'x2'],
        )
        
        assert result is not None


# =============================================================================
# Test: Diagnostics
# =============================================================================

class TestDiagnostics:
    """Test diagnostics return for IPW/IPWRA/PSM."""
    
    def test_ipw_diagnostics(self, common_timing_panel_data):
        """Test IPW returns diagnostics when requested."""
        result = lwdid(
            data=common_timing_panel_data,
            y='y',
            d='d',
            ivar='id',
            tvar='year',
            post='post',
            rolling='demean',
            estimator='ipw',
            controls=['x1', 'x2'],
            return_diagnostics=True,
        )
        
        assert result is not None
        # Diagnostics should be stored in metadata
        assert hasattr(result, '_metadata')
    
    def test_ipwra_diagnostics(self, common_timing_panel_data):
        """Test IPWRA returns diagnostics when requested."""
        result = lwdid(
            data=common_timing_panel_data,
            y='y',
            d='d',
            ivar='id',
            tvar='year',
            post='post',
            rolling='demean',
            estimator='ipwra',
            controls=['x1', 'x2'],
            return_diagnostics=True,
        )
        
        assert result is not None


# =============================================================================
# Test: Rolling Methods Compatibility
# =============================================================================

class TestRollingMethods:
    """Test IPW/IPWRA/PSM with different rolling transformation methods."""
    
    def test_ipw_demean(self, common_timing_panel_data):
        """Test IPW with demean transformation."""
        result = lwdid(
            data=common_timing_panel_data,
            y='y',
            d='d',
            ivar='id',
            tvar='year',
            post='post',
            rolling='demean',
            estimator='ipw',
            controls=['x1', 'x2'],
        )
        assert result is not None
    
    def test_ipw_detrend(self, common_timing_panel_data):
        """Test IPW with detrend transformation."""
        result = lwdid(
            data=common_timing_panel_data,
            y='y',
            d='d',
            ivar='id',
            tvar='year',
            post='post',
            rolling='detrend',
            estimator='ipw',
            controls=['x1', 'x2'],
        )
        assert result is not None
    
    def test_ipwra_demean(self, common_timing_panel_data):
        """Test IPWRA with demean transformation."""
        result = lwdid(
            data=common_timing_panel_data,
            y='y',
            d='d',
            ivar='id',
            tvar='year',
            post='post',
            rolling='demean',
            estimator='ipwra',
            controls=['x1', 'x2'],
        )
        assert result is not None
    
    def test_psm_detrend(self, common_timing_panel_data):
        """Test PSM with detrend transformation."""
        result = lwdid(
            data=common_timing_panel_data,
            y='y',
            d='d',
            ivar='id',
            tvar='year',
            post='post',
            rolling='detrend',
            estimator='psm',
            controls=['x1', 'x2'],
        )
        assert result is not None


# =============================================================================
# Test: Comparison Across Estimators
# =============================================================================

class TestEstimatorComparison:
    """Compare results across different estimators."""
    
    def test_all_estimators_similar_direction(self, common_timing_panel_data):
        """Test that all estimators produce ATT in the same direction."""
        results = {}
        
        # RA estimator
        results['ra'] = lwdid(
            data=common_timing_panel_data,
            y='y',
            d='d',
            ivar='id',
            tvar='year',
            post='post',
            rolling='demean',
            estimator='ra',
            controls=['x1', 'x2'],
        )
        
        # IPW estimator
        results['ipw'] = lwdid(
            data=common_timing_panel_data,
            y='y',
            d='d',
            ivar='id',
            tvar='year',
            post='post',
            rolling='demean',
            estimator='ipw',
            controls=['x1', 'x2'],
        )
        
        # IPWRA estimator
        results['ipwra'] = lwdid(
            data=common_timing_panel_data,
            y='y',
            d='d',
            ivar='id',
            tvar='year',
            post='post',
            rolling='demean',
            estimator='ipwra',
            controls=['x1', 'x2'],
        )
        
        # PSM estimator
        results['psm'] = lwdid(
            data=common_timing_panel_data,
            y='y',
            d='d',
            ivar='id',
            tvar='year',
            post='post',
            rolling='demean',
            estimator='psm',
            controls=['x1', 'x2'],
        )
        
        # All should be positive (true ATT = 3.0)
        for name, result in results.items():
            assert result.att > 0, f"{name} estimator should give positive ATT"
        
        # All should be within reasonable range of each other
        atts = [r.att for r in results.values()]
        att_range = max(atts) - min(atts)
        assert att_range < 3.0, "ATT estimates should be reasonably close"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
