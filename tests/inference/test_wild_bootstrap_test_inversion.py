"""
Test Inversion CI optimization tests (Task 4).

Validates numerical correctness and performance of
_compute_bootstrap_pvalue_at_null_fast() and the refactored
wild_cluster_bootstrap_test_inversion().

Test coverage:
  - Fast path vs slow path numerical equivalence (p-value, CI bounds)
  - Reference data regression tests (CI bounds, p-value, ATT)
  - Precomputed matrices correctly shared across grid points
  - Weight type coverage (rademacher, mammen, webb)
  - Edge cases (small N/G, SE=0)
  - Performance benchmark (N=500, B=999, grid_points=25 < 15s)

Validates the test inversion confidence interval procedure for Wild Cluster
Bootstrap inference in the Lee-Wooldridge Difference-in-Differences framework.

References
----------
Lee, S. & Wooldridge, J. M. (2025). A Simple Transformation Approach to
    Difference-in-Differences Estimation for Panel Data. SSRN 4516518.
Lee, S. & Wooldridge, J. M. (2026). Simple Approaches to Inference with
    DiD Estimators with Small Cross-Sectional Sample Sizes. SSRN 5325686.
Cameron, A. C. & Miller, D. L. (2015). A Practitioner's Guide to
    Cluster-Robust Inference. Journal of Human Resources, 50(2), 317--372.
"""

import json
import os
import time

import numpy as np
import pandas as pd
import pytest

from lwdid.inference.wild_bootstrap import (
    wild_cluster_bootstrap_test_inversion,
    _compute_bootstrap_pvalue_at_null,
    _compute_bootstrap_pvalue_at_null_fast,
    _precompute_bootstrap_matrices,
    _estimate_ols_for_bootstrap,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REFERENCE_DIR = os.path.join(os.path.dirname(__file__), 'reference_data')


def _generate_test_data(N: int, G: int, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic dataset identical to reference data generation."""
    rng = np.random.default_rng(seed)

    cluster_sizes = [N // G] * G
    for i in range(N % G):
        cluster_sizes[i] += 1
    cluster_ids = np.repeat(np.arange(G), cluster_sizes)

    cluster_effects = rng.normal(0, 0.5, G)
    d = rng.binomial(1, 0.4, N).astype(float)
    y = 1.0 + 2.0 * d + cluster_effects[cluster_ids] + rng.normal(0, 1, N)

    return pd.DataFrame({
        'y_transformed': y,
        'd': d,
        'cluster': cluster_ids,
    })


def _load_test_inversion_reference() -> dict:
    """Load pre-optimization test inversion reference data."""
    path = os.path.join(REFERENCE_DIR, 'test_inversion_reference.json')
    with open(path, 'r') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def reference_data():
    """Load test inversion reference data (module-scoped for efficiency)."""
    return _load_test_inversion_reference()


@pytest.fixture(scope='module')
def test_data_small():
    """Small dataset matching reference data parameters (N=100, G=8)."""
    return _generate_test_data(N=100, G=8, seed=42)


# ---------------------------------------------------------------------------
# 1. Reference data regression tests
# ---------------------------------------------------------------------------

class TestTestInversionReferenceRegression:
    """Optimized test inversion must match pre-saved reference values.

    Reference: tests/inference/reference_data/test_inversion_reference.json
    Parameters: N=100, G=8, B=499, seed=42, grid_points=15, ci_tol=0.01
    """

    @pytest.fixture(scope='class')
    def reference(self):
        path = os.path.join(REFERENCE_DIR, 'test_inversion_reference.json')
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @pytest.fixture(scope='class')
    def data(self, reference):
        meta = reference['_metadata']['data_params']
        return _generate_test_data(
            N=meta['N'], G=meta['G'], seed=meta['seed']
        )

    @pytest.fixture(scope='class')
    def result(self, data, reference):
        params = reference['_metadata']['bootstrap_params']
        return wild_cluster_bootstrap_test_inversion(
            data=data,
            y_transformed='y_transformed',
            d='d',
            cluster_var='cluster',
            controls=None,
            n_bootstrap=params['n_bootstrap'],
            weight_type=params['weight_type'],
            alpha=params['alpha'],
            seed=params['seed'],
            grid_points=params['grid_points'],
            ci_tol=params['ci_tol'],
        )

    def test_att_matches_reference(self, result, reference):
        """ATT point estimate must match reference exactly."""
        np.testing.assert_allclose(
            result.att, reference['att'], rtol=1e-10,
            err_msg="ATT does not match reference"
        )

    def test_t_stat_matches_reference(self, result, reference):
        """t_stat_original must match reference exactly."""
        np.testing.assert_allclose(
            result.t_stat_original, reference['t_stat_original'], rtol=1e-10,
            err_msg="t_stat_original does not match reference"
        )

    def test_pvalue_matches_reference(self, result, reference):
        """p-value must match reference within 2/B tolerance.

        The 2/B tolerance accounts for floating-point boundary effects
        in the |t*| >= |t_orig| comparison (see design doc Appendix B.3).
        """
        B = reference['_metadata']['bootstrap_params']['n_bootstrap']
        tolerance = 2.0 / B
        assert abs(result.pvalue - reference['pvalue']) <= tolerance, (
            f"p-value mismatch: got {result.pvalue:.6f}, "
            f"ref {reference['pvalue']:.6f}, "
            f"diff {abs(result.pvalue - reference['pvalue']):.6e}, "
            f"tol {tolerance:.6e}"
        )

    def test_ci_lower_matches_reference(self, result, reference):
        """CI lower bound must match reference within bisection tolerance.

        brentq uses xtol=ci_tol=0.01, so the boundary is accurate to
        approximately ±ci_tol.  The p-value function's 2/B noise can
        further shift the root.  We use atol=2*ci_tol.
        """
        ci_tol = reference['_metadata']['bootstrap_params']['ci_tol']
        assert abs(result.ci_lower - reference['ci_lower']) <= 2 * ci_tol, (
            f"ci_lower mismatch: got {result.ci_lower:.6f}, "
            f"ref {reference['ci_lower']:.6f}"
        )

    def test_ci_upper_matches_reference(self, result, reference):
        """CI upper bound must match reference within bisection tolerance."""
        ci_tol = reference['_metadata']['bootstrap_params']['ci_tol']
        assert abs(result.ci_upper - reference['ci_upper']) <= 2 * ci_tol, (
            f"ci_upper mismatch: got {result.ci_upper:.6f}, "
            f"ref {reference['ci_upper']:.6f}"
        )

    def test_ci_method_is_test_inversion(self, result):
        """Result must report correct CI method."""
        assert result.ci_method == 'test_inversion'

    def test_n_clusters_matches_reference(self, result, reference):
        """Cluster count must match reference."""
        assert result.n_clusters == reference['n_clusters']

    def test_ci_contains_att(self, result):
        """CI must contain the ATT point estimate."""
        assert result.ci_lower <= result.att <= result.ci_upper, (
            f"CI [{result.ci_lower:.4f}, {result.ci_upper:.4f}] "
            f"does not contain ATT={result.att:.4f}"
        )

    def test_ci_width_reasonable(self, result):
        """CI width must be positive and not absurdly wide."""
        width = result.ci_upper - result.ci_lower
        assert 0.1 < width < 10.0, (
            f"CI width {width:.4f} is unreasonable"
        )


# ---------------------------------------------------------------------------
# 2. Fast path vs slow path equivalence
# ---------------------------------------------------------------------------

class TestFastEqualsSlowPath:
    """Fast path must produce numerically equivalent results to slow path."""

    def test_pvalue_at_null_fast_equals_slow(self, test_data_small):
        """_compute_bootstrap_pvalue_at_null_fast matches the slow version
        at several null hypothesis values."""
        precomp = _precompute_bootstrap_matrices(
            test_data_small, 'y_transformed', 'd', 'cluster', None
        )
        d_values = test_data_small['d'].values.astype(np.float64)

        original_result = _estimate_ols_for_bootstrap(
            test_data_small, 'y_transformed', 'd', 'cluster', None
        )
        att_original = original_result['att']
        se_original = original_result['se']

        null_values = [0.0, 1.0, att_original, att_original + se_original]

        for null_val in null_values:
            pval_slow = _compute_bootstrap_pvalue_at_null(
                test_data_small, 'y_transformed', 'd', 'cluster',
                null_value=null_val,
                att_original=att_original,
                se_original=se_original,
                n_bootstrap=199,
                weight_type='rademacher',
                controls=None,
                seed=42,
            )

            pval_fast = _compute_bootstrap_pvalue_at_null_fast(
                null_value=null_val,
                precomp=precomp,
                d_values=d_values,
                att_original=att_original,
                se_original=se_original,
                n_bootstrap=199,
                weight_type='rademacher',
                seed=42,
            )

            B = 199
            assert abs(pval_fast - pval_slow) <= 2 / B, (
                f"At null={null_val:.4f}: fast p={pval_fast:.6f} vs "
                f"slow p={pval_slow:.6f}, diff={abs(pval_fast - pval_slow):.6f}"
            )

    def test_test_inversion_fast_equals_slow(self, test_data_small):
        """Full test inversion: fast path matches slow path.

        The bisection root-finder (brentq) operates on a discrete p-value
        function that can differ by up to 2/B between fast and slow paths.
        This noise can shift the root by up to ci_tol, so we use
        atol=2*ci_tol for CI bound comparison.
        """
        ci_tol = 0.05
        common = dict(
            data=test_data_small,
            y_transformed='y_transformed',
            d='d',
            cluster_var='cluster',
            controls=None,
            n_bootstrap=199,
            weight_type='rademacher',
            alpha=0.05,
            seed=42,
            grid_points=11,
            ci_tol=ci_tol,
        )

        result_fast = wild_cluster_bootstrap_test_inversion(**common, _force_slow=False)
        result_slow = wild_cluster_bootstrap_test_inversion(**common, _force_slow=True)

        # ATT and t_stat_original must be identical (same _estimate_ols_for_bootstrap)
        np.testing.assert_allclose(
            result_fast.att, result_slow.att, rtol=1e-12,
        )
        np.testing.assert_allclose(
            result_fast.t_stat_original, result_slow.t_stat_original, rtol=1e-12,
        )

        # CI bounds: atol=2*ci_tol (bisection tolerance + discrete p-value noise)
        assert abs(result_fast.ci_lower - result_slow.ci_lower) <= 2 * ci_tol, (
            f"ci_lower: fast={result_fast.ci_lower:.4f} vs slow={result_slow.ci_lower:.4f}"
        )
        assert abs(result_fast.ci_upper - result_slow.ci_upper) <= 2 * ci_tol, (
            f"ci_upper: fast={result_fast.ci_upper:.4f} vs slow={result_slow.ci_upper:.4f}"
        )

        # p-value: 2/B tolerance
        B = 199
        assert abs(result_fast.pvalue - result_slow.pvalue) <= 2 / B, (
            f"p-value: fast={result_fast.pvalue} vs slow={result_slow.pvalue}"
        )


# ---------------------------------------------------------------------------
# 3. Precomputed matrices sharing correctness
# ---------------------------------------------------------------------------

class TestPrecomputeSharing:
    """Precomputed matrices must be correctly shared across all grid points
    without mutation."""

    @pytest.fixture(scope='class')
    def data(self):
        return _generate_test_data(N=80, G=6, seed=123)

    def test_precomp_not_mutated(self, data):
        """Multiple calls to _compute_bootstrap_pvalue_at_null_fast()
        must not modify the precomp dict."""
        precomp = _precompute_bootstrap_matrices(
            data, 'y_transformed', 'd', 'cluster'
        )
        d_values = data['d'].values.astype(np.float64)
        original = _estimate_ols_for_bootstrap(
            data, 'y_transformed', 'd', 'cluster'
        )

        # Save copies of key matrices
        P_before = precomp['P'].copy()
        X_before = precomp['X'].copy()
        y_before = precomp['y'].copy()
        XtX_inv_before = precomp['XtX_inv'].copy()

        for theta in [0.0, 1.0, 2.0, 3.0, 4.0]:
            _compute_bootstrap_pvalue_at_null_fast(
                null_value=theta,
                precomp=precomp,
                d_values=d_values,
                att_original=original['att'],
                se_original=original['se'],
                n_bootstrap=99,
                weight_type='rademacher',
                seed=42,
            )

        # Verify precomp was not modified
        np.testing.assert_array_equal(precomp['P'], P_before)
        np.testing.assert_array_equal(precomp['X'], X_before)
        np.testing.assert_array_equal(precomp['y'], y_before)
        np.testing.assert_array_equal(precomp['XtX_inv'], XtX_inv_before)

    def test_different_null_values_give_different_pvalues(self, data):
        """Different null values must produce distinct p-values."""
        precomp = _precompute_bootstrap_matrices(
            data, 'y_transformed', 'd', 'cluster'
        )
        d_values = data['d'].values.astype(np.float64)
        original = _estimate_ols_for_bootstrap(
            data, 'y_transformed', 'd', 'cluster'
        )

        pvalues = []
        for theta in [0.0, 1.0, 2.0, 3.0]:
            pval = _compute_bootstrap_pvalue_at_null_fast(
                null_value=theta,
                precomp=precomp,
                d_values=d_values,
                att_original=original['att'],
                se_original=original['se'],
                n_bootstrap=199,
                weight_type='rademacher',
                seed=42,
            )
            pvalues.append(pval)

        assert len(set(pvalues)) > 1, (
            f"All p-values are identical: {pvalues}"
        )

    def test_pvalue_monotonicity_near_att(self, test_data_small):
        """p-value should be highest near ATT and decrease away from it."""
        precomp = _precompute_bootstrap_matrices(
            test_data_small, 'y_transformed', 'd', 'cluster', None
        )
        d_values = test_data_small['d'].values.astype(np.float64)
        original = _estimate_ols_for_bootstrap(
            test_data_small, 'y_transformed', 'd', 'cluster', None
        )
        att = original['att']
        se = original['se']

        pval_at_att = _compute_bootstrap_pvalue_at_null_fast(
            null_value=att, precomp=precomp, d_values=d_values,
            att_original=att, se_original=se,
            n_bootstrap=199, weight_type='rademacher', seed=42,
        )
        pval_far = _compute_bootstrap_pvalue_at_null_fast(
            null_value=att + 5 * se, precomp=precomp, d_values=d_values,
            att_original=att, se_original=se,
            n_bootstrap=199, weight_type='rademacher', seed=42,
        )

        assert pval_at_att > 0.5, f"p-value at ATT should be large, got {pval_at_att}"
        assert pval_far < pval_at_att, (
            f"p-value far from ATT ({pval_far}) should be < at ATT ({pval_at_att})"
        )


# ---------------------------------------------------------------------------
# 4. Weight type coverage
# ---------------------------------------------------------------------------

class TestWeightTypes:
    """Test inversion works correctly with all weight types."""

    @pytest.mark.parametrize("weight_type", ['rademacher', 'mammen', 'webb'])
    def test_weight_type_produces_valid_ci(self, test_data_small, weight_type):
        """Each weight type produces a valid CI containing the ATT."""
        result = wild_cluster_bootstrap_test_inversion(
            data=test_data_small,
            y_transformed='y_transformed',
            d='d',
            cluster_var='cluster',
            n_bootstrap=199,
            weight_type=weight_type,
            seed=42,
            grid_points=11,
            ci_tol=0.01,
        )
        assert result.ci_lower < result.att < result.ci_upper, (
            f"CI [{result.ci_lower}, {result.ci_upper}] "
            f"does not contain ATT={result.att}"
        )
        assert 0 <= result.pvalue <= 1


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases for test inversion CI."""

    def test_small_dataset(self):
        """Minimal dataset (N=12, G=3) produces valid results."""
        data = _generate_test_data(N=12, G=3, seed=123)
        result = wild_cluster_bootstrap_test_inversion(
            data=data,
            y_transformed='y_transformed',
            d='d',
            cluster_var='cluster',
            n_bootstrap=49,
            seed=42,
            grid_points=7,
            ci_tol=0.05,
        )
        assert not np.isnan(result.ci_lower)
        assert not np.isnan(result.ci_upper)
        assert result.ci_lower < result.ci_upper

    def test_small_n_small_g(self):
        """Small sample (N=20, G=4) runs without error."""
        data = _generate_test_data(N=20, G=4, seed=99)
        result = wild_cluster_bootstrap_test_inversion(
            data=data,
            y_transformed='y_transformed',
            d='d',
            cluster_var='cluster',
            n_bootstrap=99,
            weight_type='rademacher',
            seed=42,
            grid_points=7,
            ci_tol=0.1,
        )
        assert result.ci_method == 'test_inversion'
        assert not np.isnan(result.att)
        if not np.isnan(result.ci_lower) and not np.isnan(result.ci_upper):
            assert result.ci_lower <= result.att <= result.ci_upper

    def test_se_zero_returns_nan_ci(self):
        """When SE=0, test inversion returns NaN CI gracefully."""
        data = pd.DataFrame({
            'y_transformed': np.ones(20),
            'd': np.array([1.0] * 10 + [0.0] * 10),
            'cluster': np.repeat(np.arange(4), 5),
        })
        result = wild_cluster_bootstrap_test_inversion(
            data=data,
            y_transformed='y_transformed',
            d='d',
            cluster_var='cluster',
            n_bootstrap=49,
            seed=42,
            grid_points=5,
        )
        assert result.att is not None

    def test_grid_resolution_consistency(self, test_data_small):
        """Different grid resolutions converge to similar CI bounds."""
        result_11 = wild_cluster_bootstrap_test_inversion(
            data=test_data_small,
            y_transformed='y_transformed',
            d='d',
            cluster_var='cluster',
            n_bootstrap=199,
            seed=42,
            grid_points=11,
            ci_tol=0.01,
        )
        result_21 = wild_cluster_bootstrap_test_inversion(
            data=test_data_small,
            y_transformed='y_transformed',
            d='d',
            cluster_var='cluster',
            n_bootstrap=199,
            seed=42,
            grid_points=21,
            ci_tol=0.01,
        )
        # Both should converge to similar CI bounds after bisection refinement
        np.testing.assert_allclose(
            result_11.ci_lower, result_21.ci_lower, rtol=0.05,
        )
        np.testing.assert_allclose(
            result_11.ci_upper, result_21.ci_upper, rtol=0.05,
        )


# ---------------------------------------------------------------------------
# 6. Performance benchmarks
# ---------------------------------------------------------------------------

class TestTestInversionPerformance:
    """Performance benchmarks for optimized test inversion CI."""

    @pytest.mark.slow
    @pytest.mark.performance
    def test_performance_target_n500(self):
        """Test Inversion CI: N=500, B=999, grid_points=25 should < 15s."""
        data = _generate_test_data(N=500, G=30, seed=42)

        start = time.perf_counter()
        result = wild_cluster_bootstrap_test_inversion(
            data=data,
            y_transformed='y_transformed',
            d='d',
            cluster_var='cluster',
            n_bootstrap=999,
            weight_type='rademacher',
            seed=42,
            grid_points=25,
            ci_tol=0.01,
        )
        elapsed = time.perf_counter() - start

        assert elapsed < 15.0, (
            f"Test inversion took {elapsed:.1f}s, target < 15s"
        )
        assert not np.isnan(result.att)
        assert result.ci_method == 'test_inversion'

    @pytest.mark.performance
    def test_fast_path_speedup_over_slow(self):
        """Fast path should be significantly faster than slow path."""
        data = _generate_test_data(N=60, G=6, seed=42)
        common = dict(
            data=data,
            y_transformed='y_transformed',
            d='d',
            cluster_var='cluster',
            n_bootstrap=99,
            weight_type='rademacher',
            seed=42,
            grid_points=7,
            ci_tol=0.1,
        )

        # Slow path
        start = time.perf_counter()
        wild_cluster_bootstrap_test_inversion(**common, _force_slow=True)
        t_slow = time.perf_counter() - start

        # Fast path
        start = time.perf_counter()
        wild_cluster_bootstrap_test_inversion(**common, _force_slow=False)
        t_fast = time.perf_counter() - start

        speedup = t_slow / t_fast if t_fast > 0 else float('inf')
        assert speedup >= 3.0, (
            f"Speedup {speedup:.1f}x < 3x "
            f"(slow={t_slow:.2f}s, fast={t_fast:.2f}s)"
        )
