"""
Unit tests for BUG-130, BUG-131, and BUG-132 fixes.

BUG-130: df_inference hardcoded to -2 without considering control variables
BUG-131: weights_cv returns 0 for single weight instead of NaN
BUG-132: PSM partial_delta_gamma incorrectly iterates over all observations
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


class TestBug130DfInferenceControls:
    """Tests for BUG-130: df_inference should consider control variables."""

    def test_ipwra_df_inference_no_controls(self):
        """IPWRA df_inference with no controls should be n - 2."""
        from lwdid.staggered.estimation import _estimate_single_effect_ipwra

        # Create minimal test data
        np.random.seed(42)
        n = 100
        data = pd.DataFrame({
            'y': np.random.randn(n),
            'd': np.array([1] * 50 + [0] * 50),
        })

        result = _estimate_single_effect_ipwra(
            data=data,
            y='y',
            d='d',
            controls=[],
            propensity_controls=[],
            trim_threshold=0.01,
            se_method='analytical',
            alpha=0.05,
            return_diagnostics=False,
        )

        # With no controls, df = n - 2 (intercept + treatment)
        expected_df = n - 2
        assert result['df_resid'] == expected_df
        assert result['df_inference'] == expected_df

    def test_ipwra_df_inference_with_controls(self):
        """IPWRA df_inference with controls should be n - 2 - K."""
        from lwdid.staggered.estimation import _estimate_single_effect_ipwra

        np.random.seed(42)
        n = 100
        data = pd.DataFrame({
            'y': np.random.randn(n),
            'd': np.array([1] * 50 + [0] * 50),
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
        })

        controls = ['x1', 'x2']
        result = _estimate_single_effect_ipwra(
            data=data,
            y='y',
            d='d',
            controls=controls,
            propensity_controls=controls,
            trim_threshold=0.01,
            se_method='analytical',
            alpha=0.05,
            return_diagnostics=False,
        )

        # With 2 controls, df = n - 2 - 2 = n - 4
        expected_df = n - 2 - len(controls)
        assert result['df_resid'] == expected_df
        assert result['df_inference'] == expected_df

    def test_ipw_df_inference_with_controls(self):
        """IPW df_inference should be n - 2 (matches Stata teffects ipw)."""
        from lwdid.staggered.estimation import _estimate_single_effect_ipw

        np.random.seed(42)
        n = 100
        data = pd.DataFrame({
            'y': np.random.randn(n),
            'd': np.array([1] * 50 + [0] * 50),
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
            'x3': np.random.randn(n),
        })

        propensity_controls = ['x1', 'x2', 'x3']
        result = _estimate_single_effect_ipw(
            data=data,
            y='y',
            d='d',
            propensity_controls=propensity_controls,
            trim_threshold=0.01,
            se_method='analytical',
            alpha=0.05,
            return_diagnostics=False,
        )

        # IPW uses influence function-based SE, df = n - 2 (matches Stata teffects ipw)
        # PS controls do not affect df for IPW ATT inference
        expected_df = n - 2
        assert result['df_resid'] == expected_df
        assert result['df_inference'] == expected_df

    def test_df_resid_equals_df_inference(self):
        """df_resid and df_inference should be equal for standard (non-cluster) SE."""
        from lwdid.staggered.estimation import _estimate_single_effect_ipwra

        np.random.seed(42)
        n = 80
        data = pd.DataFrame({
            'y': np.random.randn(n),
            'd': np.array([1] * 40 + [0] * 40),
            'x1': np.random.randn(n),
        })

        result = _estimate_single_effect_ipwra(
            data=data,
            y='y',
            d='d',
            controls=['x1'],
            propensity_controls=['x1'],
            trim_threshold=0.01,
            alpha=0.05,
        )

        # For non-clustered SE, df_resid should equal df_inference
        assert result['df_resid'] == result['df_inference']


class TestBug131WeightsCvSingleWeight:
    """Tests for BUG-131: weights_cv for single weight."""

    def test_single_weight_returns_zero(self):
        """Single weight should return 0.0 (no variation in a single value)."""
        from lwdid.core import _convert_ipwra_result_to_dict

        # Mock IPWRA result with single weight
        mock_result = MagicMock()
        mock_result.att = 1.5
        mock_result.se = 0.3
        mock_result.t_stat = 5.0
        mock_result.pvalue = 0.001
        mock_result.ci_lower = 0.9
        mock_result.ci_upper = 2.1
        mock_result.n_treated = 50
        mock_result.n_control = 1  # Only 1 control
        mock_result.weights = np.array([1.5])  # Single weight
        mock_result.propensity_scores = np.array([0.5])

        result = _convert_ipwra_result_to_dict(
            ipwra_result=mock_result,
            alpha=0.05,
            vce=None,
            cluster_var=None,
            controls=None,
        )

        # BUG-256 FIX: Single weight has no variation, so CV = 0.0
        # This is mathematically correct: CV = std/mean, and std of single value is 0
        assert result['weights_cv'] == 0.0, \
            f"weights_cv should be 0.0 for single weight, got {result['weights_cv']}"

    def test_no_weights_returns_nan(self):
        """No weights (None) should return NaN."""
        from lwdid.core import _convert_ipwra_result_to_dict

        mock_result = MagicMock()
        mock_result.att = 1.5
        mock_result.se = 0.3
        mock_result.t_stat = 5.0
        mock_result.pvalue = 0.001
        mock_result.ci_lower = 0.9
        mock_result.ci_upper = 2.1
        mock_result.n_treated = 50
        mock_result.n_control = 50
        mock_result.weights = None  # No weights
        mock_result.propensity_scores = np.array([0.5] * 100)

        result = _convert_ipwra_result_to_dict(
            ipwra_result=mock_result,
            alpha=0.05,
            vce=None,
            cluster_var=None,
            controls=None,
        )

        assert np.isnan(result['weights_cv']), \
            f"weights_cv should be NaN for no weights, got {result['weights_cv']}"

    def test_empty_weights_returns_nan(self):
        """Empty weights array should return NaN."""
        from lwdid.core import _convert_ipwra_result_to_dict

        mock_result = MagicMock()
        mock_result.att = 1.5
        mock_result.se = 0.3
        mock_result.t_stat = 5.0
        mock_result.pvalue = 0.001
        mock_result.ci_lower = 0.9
        mock_result.ci_upper = 2.1
        mock_result.n_treated = 50
        mock_result.n_control = 50
        mock_result.weights = np.array([])  # Empty array
        mock_result.propensity_scores = np.array([0.5] * 100)

        result = _convert_ipwra_result_to_dict(
            ipwra_result=mock_result,
            alpha=0.05,
            vce=None,
            cluster_var=None,
            controls=None,
        )

        assert np.isnan(result['weights_cv']), \
            f"weights_cv should be NaN for empty weights, got {result['weights_cv']}"

    def test_two_weights_returns_valid_cv(self):
        """Two weights should return valid CV."""
        from lwdid.core import _convert_ipwra_result_to_dict

        mock_result = MagicMock()
        mock_result.att = 1.5
        mock_result.se = 0.3
        mock_result.t_stat = 5.0
        mock_result.pvalue = 0.001
        mock_result.ci_lower = 0.9
        mock_result.ci_upper = 2.1
        mock_result.n_treated = 50
        mock_result.n_control = 2
        mock_result.weights = np.array([1.0, 2.0])  # Two weights
        mock_result.propensity_scores = np.array([0.5] * 52)

        result = _convert_ipwra_result_to_dict(
            ipwra_result=mock_result,
            alpha=0.05,
            vce=None,
            cluster_var=None,
            controls=None,
        )

        # weights_cv should be a valid number for 2+ weights
        assert not np.isnan(result['weights_cv']), \
            "weights_cv should be valid for 2+ weights"
        assert result['weights_cv'] >= 0, \
            f"weights_cv should be non-negative, got {result['weights_cv']}"

    def test_multiple_weights_returns_valid_cv(self):
        """Multiple weights should return correct CV."""
        from lwdid.core import _convert_ipwra_result_to_dict

        mock_result = MagicMock()
        mock_result.att = 1.5
        mock_result.se = 0.3
        mock_result.t_stat = 5.0
        mock_result.pvalue = 0.001
        mock_result.ci_lower = 0.9
        mock_result.ci_upper = 2.1
        mock_result.n_treated = 50
        mock_result.n_control = 5
        weights = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        mock_result.weights = weights
        mock_result.propensity_scores = np.array([0.5] * 55)

        result = _convert_ipwra_result_to_dict(
            ipwra_result=mock_result,
            alpha=0.05,
            vce=None,
            cluster_var=None,
            controls=None,
        )

        # Calculate expected CV
        expected_cv = np.std(weights, ddof=1) / np.mean(weights)
        assert result['weights_cv'] == pytest.approx(expected_cv, rel=1e-10), \
            f"weights_cv mismatch: expected {expected_cv}, got {result['weights_cv']}"


class TestBug132PartialDeltaGammaTreatedOnly:
    """Tests for BUG-132: partial_delta_gamma should only sum over treated units."""

    def test_partial_delta_gamma_treated_only(self):
        """Verify partial_delta_gamma only iterates over treated observations."""
        # This is a code verification test - we check the implementation directly
        import inspect
        from lwdid.staggered.estimators import _compute_psm_variance_adjustment

        source = inspect.getsource(_compute_psm_variance_adjustment)

        # Verify the fix is in place:
        # 1. Check for the fix comment
        assert "BUG-132 FIX" in source, \
            "BUG-132 fix comment should be present in _compute_psm_variance_adjustment"

        # 2. Check that iteration is over treat_idx, not range(n)
        assert "for i in treat_idx:" in source, \
            "partial_delta_gamma loop should iterate over treat_idx"

        # 3. Check that range(n) is NOT used in the partial_delta_gamma section
        # Find the section between "Compute d_delta/d_gamma" and "Step 5"
        step4_start = source.find("Step 4: Compute")
        step5_start = source.find("Step 5: Compute")

        if step4_start != -1 and step5_start != -1:
            step4_section = source[step4_start:step5_start]
            # There should be no "for i in range(n)" in this section
            assert "for i in range(n)" not in step4_section, \
                "partial_delta_gamma should NOT iterate over range(n)"

    def test_psm_variance_adjustment_returns_scalar(self):
        """PSM variance adjustment should return a scalar value.
        
        This test verifies the function signature and basic return type.
        The detailed numerical correctness is validated via Stata comparison tests.
        """
        from lwdid.staggered.estimators import _compute_psm_variance_adjustment

        np.random.seed(42)
        n_treat = 20
        n_control = 30
        n = n_treat + n_control
        p = 2  # intercept + 1 covariate

        # Create test data
        D = np.array([1] * n_treat + [0] * n_control)
        Y = np.random.randn(n)
        # Create Z with intercept column
        Z = np.column_stack([np.ones(n), np.random.randn(n)])
        pscores = np.clip(np.random.rand(n) * 0.8 + 0.1, 0.1, 0.9)

        # Create V_gamma (variance-covariance matrix of PS model coefficients)
        V_gamma = np.eye(p) * 0.01

        # Create matched_control_ids: indices into the control subset (0 to n_control-1)
        # For each treated unit, assign a matched control index
        matched_control_ids = [[i % n_control] for i in range(n_treat)]

        adjustment = _compute_psm_variance_adjustment(
            Y=Y,
            D=D,
            Z=Z,
            pscores=pscores,
            V_gamma=V_gamma,
            matched_control_ids=matched_control_ids,
            att=0.5,
        )

        # Adjustment should be a scalar
        assert np.isscalar(adjustment) or (isinstance(adjustment, np.ndarray) and adjustment.shape == ()), \
            f"Adjustment should be scalar, got {type(adjustment)}"

        # Adjustment should be finite (not NaN or inf for valid inputs)
        assert np.isfinite(adjustment), \
            f"Adjustment should be finite, got {adjustment}"


class TestBug130CorePyFixes:
    """Tests for BUG-130 fixes in core.py conversion functions."""

    def test_convert_ipw_result_with_controls(self):
        """_convert_ipw_result_to_dict uses df = n - 2 (matches Stata teffects ipw)."""
        from lwdid.core import _convert_ipw_result_to_dict

        mock_result = MagicMock()
        mock_result.att = 1.0
        mock_result.se = 0.2
        mock_result.t_stat = 5.0
        mock_result.pvalue = 0.001
        mock_result.ci_lower = 0.6
        mock_result.ci_upper = 1.4
        mock_result.n_treated = 50
        mock_result.n_control = 50
        mock_result.weights_cv = 0.5
        mock_result.propensity_scores = np.array([0.5] * 100)

        controls = ['x1', 'x2', 'x3']
        result = _convert_ipw_result_to_dict(
            ipw_result=mock_result,
            alpha=0.05,
            vce=None,
            cluster_var=None,
            controls=controls,
        )

        # IPW uses influence function-based SE, df = n - 2 (matches Stata teffects ipw)
        # Controls do not affect df for IPW ATT inference
        expected_df = 100 - 2
        assert result['df_resid'] == expected_df, \
            f"df_resid should be {expected_df}, got {result['df_resid']}"
        assert result['df_inference'] == expected_df, \
            f"df_inference should be {expected_df}, got {result['df_inference']}"

    def test_convert_ipwra_result_with_controls(self):
        """_convert_ipwra_result_to_dict should consider controls in df calculation."""
        from lwdid.core import _convert_ipwra_result_to_dict

        mock_result = MagicMock()
        mock_result.att = 1.0
        mock_result.se = 0.2
        mock_result.t_stat = 5.0
        mock_result.pvalue = 0.001
        mock_result.ci_lower = 0.6
        mock_result.ci_upper = 1.4
        mock_result.n_treated = 60
        mock_result.n_control = 40
        mock_result.weights = np.array([1.0, 1.5] * 20)
        mock_result.propensity_scores = np.array([0.5] * 100)

        controls = ['x1', 'x2']
        result = _convert_ipwra_result_to_dict(
            ipwra_result=mock_result,
            alpha=0.05,
            vce=None,
            cluster_var=None,
            controls=controls,
        )

        # df should be n - 2 - K = 100 - 2 - 2 = 96
        expected_df = 100 - 2 - len(controls)
        assert result['df_resid'] == expected_df, \
            f"df_resid should be {expected_df}, got {result['df_resid']}"
        assert result['df_inference'] == expected_df, \
            f"df_inference should be {expected_df}, got {result['df_inference']}"

    def test_convert_ipw_result_no_controls(self):
        """_convert_ipw_result_to_dict with no controls should use df = n - 2."""
        from lwdid.core import _convert_ipw_result_to_dict

        mock_result = MagicMock()
        mock_result.att = 1.0
        mock_result.se = 0.2
        mock_result.t_stat = 5.0
        mock_result.pvalue = 0.001
        mock_result.ci_lower = 0.6
        mock_result.ci_upper = 1.4
        mock_result.n_treated = 50
        mock_result.n_control = 50
        mock_result.weights_cv = 0.5
        mock_result.propensity_scores = np.array([0.5] * 100)

        result = _convert_ipw_result_to_dict(
            ipw_result=mock_result,
            alpha=0.05,
            vce=None,
            cluster_var=None,
            controls=None,
        )

        # df should be n - 2 = 100 - 2 = 98
        expected_df = 100 - 2
        assert result['df_resid'] == expected_df
        assert result['df_inference'] == expected_df


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
