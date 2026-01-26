"""
Unit tests for BUG-105 and BUG-112 fixes.

BUG-105: estimation.py cluster-robust SE n==k division by zero risk
BUG-112-a: transformations.py _demean_transform() missing column existence check  
BUG-112-b: core.py match_rate calculation may produce negative values
"""

import numpy as np
import pandas as pd
import pytest

from lwdid.exceptions import MissingRequiredColumnError
from lwdid.staggered.estimation import _compute_hc1_variance, run_ols_regression
from lwdid.transformations import (
    apply_rolling_transform,
    _demean_transform,
    _detrend_transform,
)
from lwdid.core import _convert_psm_result_to_dict


# =============================================================================
# BUG-105: n <= k Division by Zero Risk in Cluster-Robust SE
# =============================================================================

class TestBug105ClusterRobustSE:
    """Test cases for BUG-105: n <= k division by zero protection."""

    def test_hc1_variance_n_equals_k_raises_error(self):
        """HC1 variance should raise ValueError when n == k."""
        # 2x2 design matrix: n=2, k=2
        X = np.array([[1, 0], [1, 1]])
        residuals = np.array([0.1, -0.1])
        XtX_inv = np.linalg.inv(X.T @ X)

        with pytest.raises(ValueError) as exc_info:
            _compute_hc1_variance(X, residuals, XtX_inv)

        assert "n > k" in str(exc_info.value)
        assert "n=2" in str(exc_info.value)
        assert "k=2" in str(exc_info.value)

    def test_hc1_variance_n_less_than_k_raises_error(self):
        """HC1 variance should raise ValueError when n < k."""
        # 2x3 design matrix: n=2, k=3
        X = np.array([[1, 0, 1], [1, 1, 0]])
        residuals = np.array([0.1, -0.1])
        # X'X is 3x3 but rank-deficient, use pseudo-inverse
        XtX_inv = np.linalg.pinv(X.T @ X)

        with pytest.raises(ValueError) as exc_info:
            _compute_hc1_variance(X, residuals, XtX_inv)

        assert "n > k" in str(exc_info.value)

    def test_hc1_variance_n_greater_than_k_succeeds(self):
        """HC1 variance should work when n > k."""
        # 4x2 design matrix: n=4, k=2
        X = np.array([[1, 0], [1, 1], [1, 0], [1, 1]])
        residuals = np.array([0.1, -0.1, 0.05, -0.05])
        XtX_inv = np.linalg.inv(X.T @ X)

        var = _compute_hc1_variance(X, residuals, XtX_inv)

        assert var.shape == (2, 2)
        assert np.all(np.isfinite(var))

    def test_run_ols_regression_n_equals_k_returns_nan_se(self):
        """run_ols_regression should handle n==k gracefully (NaN SE)."""
        df = pd.DataFrame({
            'y': [1.0, 2.0],
            'd': [0, 1],
            'cluster': [1, 2]
        })

        # n=2, k=2 (intercept + d)
        result = run_ols_regression(
            data=df,
            y='y',
            d='d',
            controls=None,
            vce='cluster',
            cluster_var='cluster'
        )

        # Point estimate should be valid
        assert np.isfinite(result['att'])
        # SE should be NaN due to df_resid=0
        assert np.isnan(result['se'])
        assert result['df_resid'] == 0


# =============================================================================
# BUG-112-a: Missing Column Existence Check in Transformations
# =============================================================================

class TestBug112aColumnExistenceCheck:
    """Test cases for BUG-112-a: column existence validation."""

    @pytest.fixture
    def valid_panel_data(self):
        """Create valid panel data for transformation."""
        return pd.DataFrame({
            'y': [1.0, 2.0, 1.5, 2.5, 3.0, 4.0],
            'id': [1, 1, 2, 2, 3, 3],
            'time': [1, 2, 1, 2, 1, 2],
            'post': [0, 1, 0, 1, 0, 1],
            'quarter': [1, 2, 1, 2, 1, 2],
        })

    def test_demean_transform_missing_y_column(self, valid_panel_data):
        """_demean_transform should raise MissingRequiredColumnError for missing y."""
        df = valid_panel_data.drop(columns=['y'])

        with pytest.raises(MissingRequiredColumnError) as exc_info:
            _demean_transform(df, y='y', ivar='id', post='post')

        assert 'y' in str(exc_info.value)
        assert 'demean' in str(exc_info.value).lower()

    def test_demean_transform_missing_ivar_column(self, valid_panel_data):
        """_demean_transform should raise MissingRequiredColumnError for missing ivar."""
        df = valid_panel_data.drop(columns=['id'])

        with pytest.raises(MissingRequiredColumnError) as exc_info:
            _demean_transform(df, y='y', ivar='id', post='post')

        assert 'id' in str(exc_info.value)

    def test_demean_transform_missing_post_column(self, valid_panel_data):
        """_demean_transform should raise MissingRequiredColumnError for missing post."""
        df = valid_panel_data.drop(columns=['post'])

        with pytest.raises(MissingRequiredColumnError) as exc_info:
            _demean_transform(df, y='y', ivar='id', post='post')

        assert 'post' in str(exc_info.value)

    def test_demean_transform_valid_columns_succeeds(self, valid_panel_data):
        """_demean_transform should work with valid columns."""
        result = _demean_transform(
            valid_panel_data.copy(),
            y='y', ivar='id', post='post'
        )

        assert 'ydot' in result.columns
        assert result['ydot'].notna().any()

    def test_detrend_transform_missing_tindex_column(self, valid_panel_data):
        """_detrend_transform should raise MissingRequiredColumnError for missing tindex."""
        df = valid_panel_data.drop(columns=['time'])

        with pytest.raises(MissingRequiredColumnError) as exc_info:
            _detrend_transform(df, y='y', ivar='id', tindex='time', post='post')

        assert 'time' in str(exc_info.value)
        assert 'detrend' in str(exc_info.value).lower()

    def test_apply_rolling_transform_missing_columns(self, valid_panel_data):
        """apply_rolling_transform should validate columns before transformation."""
        df = valid_panel_data.drop(columns=['y'])

        with pytest.raises(MissingRequiredColumnError) as exc_info:
            apply_rolling_transform(
                df, y='y', ivar='id', tindex='time',
                post='post', rolling='demean', tpost1=2
            )

        assert 'y' in str(exc_info.value)

    def test_apply_rolling_transform_quarterly_missing_quarter(self, valid_panel_data):
        """apply_rolling_transform should require quarter for quarterly methods."""
        df = valid_panel_data.drop(columns=['quarter'])

        with pytest.raises(MissingRequiredColumnError) as exc_info:
            apply_rolling_transform(
                df, y='y', ivar='id', tindex='time',
                post='post', rolling='demeanq', tpost1=2,
                quarter='quarter'
            )

        assert 'quarter' in str(exc_info.value)

    def test_apply_rolling_transform_quarterly_none_quarter_param(self, valid_panel_data):
        """apply_rolling_transform should error if quarter=None for quarterly methods."""
        with pytest.raises(MissingRequiredColumnError) as exc_info:
            apply_rolling_transform(
                valid_panel_data, y='y', ivar='id', tindex='time',
                post='post', rolling='demeanq', tpost1=2,
                quarter=None
            )

        assert 'quarter' in str(exc_info.value).lower()


# =============================================================================
# BUG-112-b: match_rate Range Validation
# =============================================================================

class MockPSMResult:
    """Mock PSMResult for testing _convert_psm_result_to_dict."""

    def __init__(self, n_treated, n_dropped, n_matched, n_control):
        self.att = 0.5
        self.se = 0.1
        self.t_stat = 5.0
        self.pvalue = 0.001
        self.ci_lower = 0.3
        self.ci_upper = 0.7
        self.n_treated = n_treated
        self.n_control = n_control
        self.n_matched = n_matched
        self.n_dropped = n_dropped
        self.diagnostics = None


class TestBug112bMatchRateRange:
    """Test cases for BUG-112-b: match_rate range [0, 1]."""

    def test_match_rate_normal_case(self):
        """Normal case: match_rate = (n_treated - n_dropped) / n_treated."""
        mock = MockPSMResult(n_treated=10, n_dropped=2, n_matched=8, n_control=50)
        result = _convert_psm_result_to_dict(
            mock, alpha=0.05, vce=None, cluster_var=None, controls=None
        )

        assert result['match_rate'] == 0.8

    def test_match_rate_clamped_to_zero_when_negative(self):
        """match_rate should be clamped to 0 when n_dropped > n_treated."""
        mock = MockPSMResult(n_treated=10, n_dropped=15, n_matched=0, n_control=50)
        result = _convert_psm_result_to_dict(
            mock, alpha=0.05, vce=None, cluster_var=None, controls=None
        )

        # Raw rate would be (10-15)/10 = -0.5
        assert result['match_rate'] == 0.0

    def test_match_rate_all_matched(self):
        """match_rate should be 1.0 when all treated are matched."""
        mock = MockPSMResult(n_treated=10, n_dropped=0, n_matched=10, n_control=50)
        result = _convert_psm_result_to_dict(
            mock, alpha=0.05, vce=None, cluster_var=None, controls=None
        )

        assert result['match_rate'] == 1.0

    def test_match_rate_zero_treated(self):
        """match_rate should be 0.0 when n_treated = 0 (division guard)."""
        mock = MockPSMResult(n_treated=0, n_dropped=0, n_matched=0, n_control=50)
        result = _convert_psm_result_to_dict(
            mock, alpha=0.05, vce=None, cluster_var=None, controls=None
        )

        assert result['match_rate'] == 0.0

    def test_match_rate_clamped_to_one_when_exceeds(self):
        """match_rate should be clamped to 1.0 if raw rate > 1."""
        # Hypothetical case with negative n_dropped
        mock = MockPSMResult(n_treated=10, n_dropped=-5, n_matched=15, n_control=50)
        result = _convert_psm_result_to_dict(
            mock, alpha=0.05, vce=None, cluster_var=None, controls=None
        )

        # Raw rate would be (10-(-5))/10 = 1.5
        assert result['match_rate'] == 1.0

    def test_match_rate_always_in_valid_range(self):
        """match_rate should always be in [0, 1] for any inputs."""
        test_cases = [
            (100, 0, 100, 200),   # All matched
            (100, 50, 50, 200),   # Half matched
            (100, 100, 0, 200),   # None matched
            (100, 200, 0, 200),   # More dropped than treated (edge case)
            (1, 0, 1, 10),        # Single treated
            (1, 1, 0, 10),        # Single treated, dropped
        ]

        for n_treated, n_dropped, n_matched, n_control in test_cases:
            mock = MockPSMResult(n_treated, n_dropped, n_matched, n_control)
            result = _convert_psm_result_to_dict(
                mock, alpha=0.05, vce=None, cluster_var=None, controls=None
            )

            assert 0.0 <= result['match_rate'] <= 1.0, (
                f"match_rate={result['match_rate']} out of range for "
                f"n_treated={n_treated}, n_dropped={n_dropped}"
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
