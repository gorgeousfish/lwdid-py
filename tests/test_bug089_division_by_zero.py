"""
BUG-089 Regression Test: Division by Zero Risk in estimate_att()

This test validates that estimate_att() properly handles near-zero standard errors
by setting inference statistics (t_stat, pvalue, ci_lower, ci_upper) to NaN
instead of computing potentially misleading inf/0 values.

Test coverage:
1. SE near zero -> t_stat, pvalue, CI should be NaN
2. Warning should be issued when SE is too small
3. ATT and SE values should still be returned
4. Normal cases should work unchanged

References:
- BUG-089: estimation.py estimate_att() division by zero risk
"""

import warnings
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from lwdid.estimation import estimate_att


class TestBug089DivisionByZero:
    """Test BUG-089: Division by zero risk in estimate_att()"""

    def test_near_zero_se_returns_nan_inference(self):
        """When SE is near zero, inference statistics should be NaN.
        
        This test creates a perfect-fit scenario where all observations
        have identical outcomes within treatment groups, resulting in
        SE approaching zero.
        """
        # Create data with perfect fit (identical outcomes within groups)
        # This results in very small or zero residuals and SE
        data = pd.DataFrame({
            'unit_id': list(range(1, 21)),
            'd_': [1] * 10 + [0] * 10,
            # All treated units have exactly the same outcome
            # All control units have exactly the same outcome
            'ydot_postavg': [5.0] * 10 + [2.0] * 10,
            'firstpost': [True] * 20,
        })

        # Capture warning
        with pytest.warns(UserWarning, match="Extremely small standard error"):
            results = estimate_att(
                data=data,
                y_transformed='ydot_postavg',
                d='d_',
                ivar='unit_id',
                controls=None,
                vce=None,
                cluster_var=None,
                sample_filter=data['firstpost'],
            )

        # ATT should still be computed correctly (5.0 - 2.0 = 3.0)
        assert abs(results['att'] - 3.0) < 1e-10, \
            f"ATT should be 3.0, got {results['att']}"

        # SE should be returned (even if very small)
        assert results['se_att'] < 1e-10, \
            f"SE should be very small, got {results['se_att']}"

        # Inference statistics should be NaN
        assert np.isnan(results['t_stat']), \
            f"t_stat should be NaN when SE is near zero, got {results['t_stat']}"
        assert np.isnan(results['pvalue']), \
            f"pvalue should be NaN when SE is near zero, got {results['pvalue']}"
        assert np.isnan(results['ci_lower']), \
            f"ci_lower should be NaN when SE is near zero, got {results['ci_lower']}"
        assert np.isnan(results['ci_upper']), \
            f"ci_upper should be NaN when SE is near zero, got {results['ci_upper']}"

    def test_normal_se_returns_valid_inference(self):
        """Normal case: when SE is reasonable, inference stats should be valid."""
        # Create data with natural variation
        np.random.seed(42)
        data = pd.DataFrame({
            'unit_id': list(range(1, 21)),
            'd_': [1] * 10 + [0] * 10,
            'ydot_postavg': (
                [5.0 + np.random.normal(0, 0.5) for _ in range(10)] +
                [2.0 + np.random.normal(0, 0.5) for _ in range(10)]
            ),
            'firstpost': [True] * 20,
        })

        results = estimate_att(
            data=data,
            y_transformed='ydot_postavg',
            d='d_',
            ivar='unit_id',
            controls=None,
            vce=None,
            cluster_var=None,
            sample_filter=data['firstpost'],
        )

        # SE should be positive and reasonable
        assert results['se_att'] > 1e-10, "SE should be positive"
        assert results['se_att'] < 10, "SE should be reasonable"

        # Inference statistics should be finite
        assert np.isfinite(results['t_stat']), "t_stat should be finite"
        assert np.isfinite(results['pvalue']), "pvalue should be finite"
        assert np.isfinite(results['ci_lower']), "ci_lower should be finite"
        assert np.isfinite(results['ci_upper']), "ci_upper should be finite"

        # p-value should be in [0, 1]
        assert 0 <= results['pvalue'] <= 1, \
            f"pvalue should be in [0,1], got {results['pvalue']}"

        # CI should bracket ATT
        assert results['ci_lower'] < results['att'] < results['ci_upper'], \
            "CI should bracket ATT"

    def test_warning_message_content(self):
        """Warning message should indicate that inference stats are set to NaN."""
        data = pd.DataFrame({
            'unit_id': list(range(1, 11)),
            'd_': [1] * 5 + [0] * 5,
            'ydot_postavg': [3.0] * 5 + [1.0] * 5,  # Perfect fit
            'firstpost': [True] * 10,
        })

        with pytest.warns(UserWarning) as record:
            estimate_att(
                data=data,
                y_transformed='ydot_postavg',
                d='d_',
                ivar='unit_id',
                controls=None,
                vce=None,
                cluster_var=None,
                sample_filter=data['firstpost'],
            )

        # Check warning message contains useful information
        warning_msg = str(record[0].message)
        assert "small standard error" in warning_msg.lower() or "SE=" in warning_msg, \
            "Warning should mention small SE"
        assert "NaN" in warning_msg or "nan" in warning_msg.lower(), \
            "Warning should mention that values are set to NaN"

    def test_df_inference_still_computed(self):
        """Degrees of freedom should still be computed even when SE is near zero."""
        data = pd.DataFrame({
            'unit_id': list(range(1, 21)),
            'd_': [1] * 10 + [0] * 10,
            'ydot_postavg': [5.0] * 10 + [2.0] * 10,  # Perfect fit
            'firstpost': [True] * 20,
        })

        with pytest.warns(UserWarning):
            results = estimate_att(
                data=data,
                y_transformed='ydot_postavg',
                d='d_',
                ivar='unit_id',
                controls=None,
                vce=None,
                cluster_var=None,
                sample_filter=data['firstpost'],
            )

        # df_inference should still be computed (N - k = 20 - 2 = 18)
        assert results['df_inference'] == 18, \
            f"df_inference should be 18, got {results['df_inference']}"
        assert results['df_resid'] == 18, \
            f"df_resid should be 18, got {results['df_resid']}"

    def test_near_zero_se_with_hc3(self):
        """Test with HC3 variance estimator when SE is near zero."""
        data = pd.DataFrame({
            'unit_id': list(range(1, 21)),
            'd_': [1] * 10 + [0] * 10,
            'ydot_postavg': [5.0] * 10 + [2.0] * 10,  # Perfect fit
            'firstpost': [True] * 20,
        })

        with pytest.warns(UserWarning):
            results = estimate_att(
                data=data,
                y_transformed='ydot_postavg',
                d='d_',
                ivar='unit_id',
                controls=None,
                vce='hc3',
                cluster_var=None,
                sample_filter=data['firstpost'],
            )

        # Inference statistics should be NaN
        assert np.isnan(results['t_stat']), "t_stat should be NaN with HC3"
        assert np.isnan(results['pvalue']), "pvalue should be NaN with HC3"
        assert np.isnan(results['ci_lower']), "ci_lower should be NaN with HC3"
        assert np.isnan(results['ci_upper']), "ci_upper should be NaN with HC3"

    def test_edge_case_se_exactly_at_threshold(self):
        """Test behavior when SE is exactly at the threshold (1e-10)."""
        # This is a boundary test - SE = 1e-10 exactly should trigger NaN
        # We can't easily create data with SE exactly at threshold,
        # but we can verify the logic by checking that small SE triggers NaN
        data = pd.DataFrame({
            'unit_id': list(range(1, 101)),
            'd_': [1] * 50 + [0] * 50,
            'ydot_postavg': [5.0] * 50 + [2.0] * 50,  # Perfect fit
            'firstpost': [True] * 100,
        })

        with pytest.warns(UserWarning):
            results = estimate_att(
                data=data,
                y_transformed='ydot_postavg',
                d='d_',
                ivar='unit_id',
                controls=None,
                vce=None,
                cluster_var=None,
                sample_filter=data['firstpost'],
            )

        # With perfect fit, SE should be essentially zero
        assert results['se_att'] < 1e-10, "SE should be below threshold"
        assert np.isnan(results['t_stat']), "t_stat should be NaN"

    def test_result_dict_structure_unchanged(self):
        """Result dictionary structure should be unchanged after fix."""
        data = pd.DataFrame({
            'unit_id': list(range(1, 11)),
            'd_': [1] * 5 + [0] * 5,
            'ydot_postavg': [5.0] * 5 + [2.0] * 5,
            'firstpost': [True] * 10,
        })

        with pytest.warns(UserWarning):
            results = estimate_att(
                data=data,
                y_transformed='ydot_postavg',
                d='d_',
                ivar='unit_id',
                controls=None,
                vce=None,
                cluster_var=None,
                sample_filter=data['firstpost'],
            )

        # All expected keys should be present
        expected_keys = {
            'att', 'se_att', 't_stat', 'pvalue', 'ci_lower', 'ci_upper',
            'params', 'bse', 'vcov', 'resid', 'nobs', 'df_resid',
            'df_inference', 'vce_type', 'cluster_var', 'n_clusters',
            'controls_used', 'controls', 'controls_spec',
            'n_treated_sample', 'n_control_sample'
        }
        assert set(results.keys()) == expected_keys, \
            f"Result keys mismatch: {set(results.keys())} vs {expected_keys}"


class TestNumericalValidation:
    """Numerical validation of the fix."""

    def test_no_inf_in_results(self):
        """Ensure inf is never returned (replaced by NaN)."""
        # Create perfect fit data that would produce inf without fix
        data = pd.DataFrame({
            'unit_id': list(range(1, 51)),
            'd_': [1] * 25 + [0] * 25,
            'ydot_postavg': [10.0] * 25 + [5.0] * 25,  # Perfect fit
            'firstpost': [True] * 50,
        })

        with pytest.warns(UserWarning):
            results = estimate_att(
                data=data,
                y_transformed='ydot_postavg',
                d='d_',
                ivar='unit_id',
                controls=None,
                vce=None,
                cluster_var=None,
                sample_filter=data['firstpost'],
            )

        # No inf values should appear
        assert not np.isinf(results['t_stat']), "t_stat should not be inf"
        assert not np.isinf(results['pvalue']), "pvalue should not be inf"
        assert not np.isinf(results['ci_lower']), "ci_lower should not be inf"
        assert not np.isinf(results['ci_upper']), "ci_upper should not be inf"

    def test_att_value_preserved(self):
        """ATT point estimate should be preserved even when SE is near zero."""
        for att_true in [0.0, 1.5, -2.3, 100.0]:
            data = pd.DataFrame({
                'unit_id': list(range(1, 21)),
                'd_': [1] * 10 + [0] * 10,
                'ydot_postavg': [att_true] * 10 + [0.0] * 10,  # Perfect fit
                'firstpost': [True] * 20,
            })

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = estimate_att(
                    data=data,
                    y_transformed='ydot_postavg',
                    d='d_',
                    ivar='unit_id',
                    controls=None,
                    vce=None,
                    cluster_var=None,
                    sample_filter=data['firstpost'],
                )

            assert abs(results['att'] - att_true) < 1e-10, \
                f"ATT should be {att_true}, got {results['att']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
