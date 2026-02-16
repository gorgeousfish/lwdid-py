"""
Unit tests for the estimation module.

Tests OLS regression, variance estimation (HC3, cluster), period-by-period
effects, and covariate centering logic in ``lwdid.estimation``.

References
----------
Lee, S. J. & Wooldridge, J. M. (2023). Table 3 (smoking data benchmarks).
Lee, S. J. & Wooldridge, J. M. (2025). Procedure 2.1, Section 4.12.
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from lwdid.estimation import estimate_att, estimate_period_effects, prepare_controls


# ============================================================================
# OLS point estimation
# ============================================================================

class TestOLSEstimation:
    """Basic OLS estimation tests using hand-computed expectations."""

    def test_ols_basic(self):
        """Verify ATT from a minimal three-unit cross-section.

        Hand computation:
            Treated (unit 1): ydot = 3.5
            Control (unit 2): ydot = 1.5
            Control (unit 3): ydot = -1.5
            Control mean = (1.5 + (-1.5)) / 2 = 0.0
            Expected ATT = 3.5 - 0.0 = 3.5
        """
        data = pd.DataFrame({
            'unit_id': [1, 2, 3],
            'd_': [1, 0, 0],
            'ydot_postavg': [3.5, 1.5, -1.5],
            'firstpost': [True, True, True],
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

        assert abs(results['att'] - 3.5) < 1e-7
        assert results['nobs'] == 3
        # df_resid = N - k = 3 - 2 = 1
        assert results['df_resid'] == 1

    def test_hc3_variance(self):
        """Verify HC3 variance matrix is symmetric and point estimate unchanged."""
        data = pd.DataFrame({
            'unit_id': [1, 2, 3],
            'd_': [1, 0, 0],
            'ydot_postavg': [3.5, 1.5, -1.5],
            'firstpost': [True, True, True],
        })

        results_ols = estimate_att(
            data, 'ydot_postavg', 'd_', 'unit_id', None, None, None,
            data['firstpost'],
        )
        results_hc3 = estimate_att(
            data, 'ydot_postavg', 'd_', 'unit_id', None, 'hc3', None,
            data['firstpost'],
        )

        # Point estimate is invariant to VCE choice
        assert abs(results_ols['att'] - results_hc3['att']) < 1e-10
        assert results_hc3['se_att'] > 0

        # HC3 covariance matrix must be symmetric
        V_hc3 = results_hc3['vcov']
        assert np.allclose(V_hc3, V_hc3.T, atol=1e-12)

    def test_inference_statistics(self):
        """Verify t-statistic, p-value, and confidence interval computation."""
        data = pd.DataFrame({
            'unit_id': [1, 2, 3, 4],
            'd_': [1, 0, 0, 0],
            'ydot_postavg': [2.0, 0.5, 0.3, 0.2],
            'firstpost': [True, True, True, True],
        })

        results = estimate_att(
            data, 'ydot_postavg', 'd_', 'unit_id', None, None, None,
            data['firstpost'],
        )

        # t = ATT / SE(ATT)
        expected_t = results['att'] / results['se_att']
        assert abs(results['t_stat'] - expected_t) < 1e-10

        # 95% CI = ATT +/- 1.96 * SE
        expected_ci_lower = results['att'] - 1.96 * results['se_att']
        expected_ci_upper = results['att'] + 1.96 * results['se_att']
        assert abs(results['ci_lower'] - expected_ci_lower) < 0.5
        assert abs(results['ci_upper'] - expected_ci_upper) < 0.5

        assert 0 <= results['pvalue'] <= 1

    def test_controls_warning_insufficient_sample(self):
        """Controls should be dropped with a warning when N_treated < K+1."""
        data = pd.DataFrame({
            'unit_id': [1, 2, 3, 4],
            'd_': [1, 0, 0, 0],
            'ydot_postavg': [1.0, 0.5, 0.3, 0.2],
            'firstpost': [True, True, True, True],
            'tindex': [1, 1, 1, 1],
            'x1': [1.0, 2.0, 3.0, 4.0],
        })

        with pytest.warns(UserWarning, match="Controls not applied"):
            results = estimate_att(
                data, 'ydot_postavg', 'd_', 'unit_id', ['x1'],
                vce=None, cluster_var=None, sample_filter=data['firstpost'],
            )

        assert results['controls_used'] is False
        assert len(results['params']) == 2  # intercept + d only


# ============================================================================
# VCE parameter mapping
# ============================================================================

class TestVCEMapping:
    """Tests for VCE parameter dispatch (None, hc3, cluster)."""

    def test_vce_none(self):
        """vce=None should produce OLS standard errors."""
        data = pd.DataFrame({
            'unit_id': [1, 2, 3],
            'd_': [1, 0, 0],
            'ydot_postavg': [3.5, 1.5, -1.5],
            'firstpost': [True, True, True],
        })
        results = estimate_att(
            data, 'ydot_postavg', 'd_', 'unit_id', None, None, None,
            data['firstpost'],
        )
        assert results['vce_type'] == 'ols'

    def test_vce_hc3(self):
        """vce='hc3' should produce HC3 heteroskedasticity-robust standard errors."""
        data = pd.DataFrame({
            'unit_id': [1, 2, 3, 4],
            'd_': [1, 0, 0, 0],
            'ydot_postavg': [2.0, 0.5, 0.3, 0.2],
            'firstpost': [True, True, True, True],
        })
        results = estimate_att(
            data, 'ydot_postavg', 'd_', 'unit_id', None, 'hc3', None,
            data['firstpost'],
        )
        assert results['vce_type'] == 'hc3'
        assert results['se_att'] > 0

    def test_vce_cluster(self):
        """vce='cluster' should produce cluster-robust standard errors."""
        from lwdid.exceptions import InvalidParameterError

        data = pd.DataFrame({
            'unit_id': [1, 2, 3],
            'd_': [1, 0, 0],
            'ydot_postavg': [3.5, 1.5, -1.5],
            'firstpost': [True, True, True],
            'state': [1, 2, 3],
        })

        result = estimate_att(
            data, 'ydot_postavg', 'd_', 'unit_id', None,
            vce='cluster', cluster_var='state',
            sample_filter=data['firstpost'],
        )
        assert result['vce_type'] == 'cluster'
        assert result['cluster_var'] == 'state'
        assert result['n_clusters'] == 3
        assert result['se_att'] > 0


# ============================================================================
# Period-by-period effect estimation
# ============================================================================

class TestPeriodEffects:
    """Tests for period-by-period ATT estimation (Paper Procedure 2.1)."""

    def test_estimate_period_effects_structure(self):
        """Verify the returned DataFrame has the expected columns and shape.

        Expected columns:
            period, tindex, beta, se, ci_lower, ci_upper, tstat, pval, N
        """
        data = pd.DataFrame({
            'id': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            'tindex': [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
            'year': [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
            'ydot': [0.5, 0.3, 1.0, 1.2, -0.2, 0.1, 0.3, 0.4, 0.0, -0.1, 0.2, 0.5],
            'd_': [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            'post_': [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        })

        period_labels = {3: "2003", 4: "2004"}
        df = estimate_period_effects(
            data=data, ydot='ydot', d='d_', tindex='tindex',
            tpost1=3, Tmax=4, controls_spec=None, vce=None,
            cluster_var=None, period_labels=period_labels,
        )

        expected_cols = [
            'period', 'tindex', 'beta', 'se',
            'ci_lower', 'ci_upper', 'tstat', 'pval', 'N',
        ]
        assert list(df.columns) == expected_cols
        assert len(df) == 2  # two post-treatment periods
        assert df['period'].dtype == object
        assert df.iloc[0]['period'] == "2003"
        assert df.iloc[1]['period'] == "2004"
        assert 'average' not in df['period'].values

    def test_period_effects_cross_section_sample_size(self):
        """Each period regression should use all N units (cross-sectional OLS)."""
        np.random.seed(42)
        post_pattern = [0, 0, 1, 1, 1]
        data = pd.DataFrame({
            'id': np.repeat([1, 2, 3, 4, 5], 5),
            'tindex': np.tile([1, 2, 3, 4, 5], 5),
            'year': np.tile([2001, 2002, 2003, 2004, 2005], 5),
            'ydot': np.random.randn(25),
            'd_': np.repeat([1, 1, 0, 0, 0], 5),
            'post_': np.tile(post_pattern, 5),
        })

        period_labels = {3: "2003", 4: "2004", 5: "2005"}
        df = estimate_period_effects(
            data, 'ydot', 'd_', 'tindex', tpost1=3, Tmax=5,
            controls_spec=None, vce=None, cluster_var=None,
            period_labels=period_labels,
        )

        # N = 5 units in every period
        assert all(df['N'] == 5), "Each period N must equal total number of units"
        assert len(df) == 3


# ============================================================================
# Covariate centering (Paper Section 4.12)
# ============================================================================

class TestPrepareControls:
    """Tests for the covariate centering and interaction construction."""

    def test_basic_centering(self):
        """Covariates should be centered at the treated-group mean.

        Setup: 3 treated units (x1 = 1, 2, 3) and 3 control units.
        Treated mean: (1 + 2 + 3) / 3 = 2.0
        Centered: x1_c = x1 - 2.0
        Interaction: d * x1_c
        """
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
            'x1': [1.0]*2 + [2.0]*2 + [3.0]*2 + [4.0]*2 + [5.0]*2 + [6.0]*2,
            'd_': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            'tindex': [1, 2] * 6,
        })

        spec = prepare_controls(data, 'd_', 'id', ['x1'], N_treated=3, N_control=3)

        assert spec['include'] is True
        assert abs(spec['X_mean_treated']['x1'] - 2.0) < 1e-10

        expected_xc = data['x1'] - 2.0
        assert np.allclose(spec['X_centered']['x1_c'], expected_xc, atol=1e-10)

        expected_dx = data['d_'] * expected_xc
        assert np.allclose(spec['interactions']['d_x1_c'], expected_dx, atol=1e-10)

    def test_inclusion_boundary(self):
        """Controls should be included when N > K+1 and dropped otherwise."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3, 4, 4],
            'd_': [1, 1, 1, 1, 0, 0, 0, 0],
            'x1': [1.0]*2 + [2.0]*2 + [3.0]*2 + [4.0]*2,
            'x2': [0.5]*2 + [1.5]*2 + [2.5]*2 + [3.5]*2,
        })

        # N=4, K=2 => K+1=3, so N=4 > 3 => include
        spec = prepare_controls(data, 'd_', 'id', ['x1', 'x2'], N_treated=4, N_control=4)
        assert spec['include'] is True

        # N=3, K=2 => K+1=3, so N=3 is NOT > 3 => exclude
        with pytest.warns(UserWarning, match="Controls not applied"):
            spec_warn = prepare_controls(data, 'd_', 'id', ['x1', 'x2'],
                                         N_treated=3, N_control=3)
        assert spec_warn['include'] is False


class TestPrepareControlsComprehensive:
    """Comprehensive tests for the prepare_controls inclusion logic."""

    def test_both_groups_satisfy(self):
        """Controls included when both N_1 > K+1 and N_0 > K+1."""
        data = pd.DataFrame({
            'id': list(range(1, 11)) * 2,
            'd_': [1]*10 + [0]*10,
            'x1': list(range(1, 21)),
            'x2': [0.1 * i for i in range(1, 21)],
        })
        # Assign repeated id values properly
        data = pd.DataFrame({
            'id': np.repeat(range(1, 11), 2),
            'd_': [1]*10 + [0]*10,
            'x1': [1.0]*2 + [2.0]*2 + [3.0]*2 + [4.0]*2 + [5.0]*2 + [6.0]*2 + [7.0]*2 + [8.0]*2 + [9.0]*2 + [10.0]*2,
            'x2': [0.1]*2 + [0.2]*2 + [0.3]*2 + [0.4]*2 + [0.5]*2 + [0.6]*2 + [0.7]*2 + [0.8]*2 + [0.9]*2 + [1.0]*2,
        })

        spec = prepare_controls(data, 'd_', 'id', ['x1', 'x2'], N_treated=5, N_control=5)

        assert spec['include'] is True
        assert isinstance(spec['X_centered'], pd.DataFrame)
        assert isinstance(spec['interactions'], pd.DataFrame)
        assert spec['RHS_varnames'] == ['x1', 'x2', 'd_x1_c', 'd_x2_c']

    def test_treated_group_too_small(self):
        """Warning issued when N_1 <= K+1."""
        data = pd.DataFrame({
            'id': np.repeat(range(1, 8), 2),
            'd_': [1]*4 + [0]*10,
            'x1': list(range(1, 15)),
            'x2': [0.1 * i for i in range(1, 15)],
        })

        with pytest.warns(UserWarning, match="Controls not applied") as w:
            spec = prepare_controls(data, 'd_', 'id', ['x1', 'x2'],
                                     N_treated=2, N_control=5)

        assert spec['include'] is False
        assert spec['X_centered'] is None

        msg = str(w[0].message)
        assert "K=2" in msg
        assert "N_1=2" in msg

    def test_control_group_too_small(self):
        """Warning issued when N_0 <= K+1."""
        data = pd.DataFrame({
            'id': np.repeat(range(1, 8), 2),
            'd_': [1]*10 + [0]*4,
            'x1': list(range(1, 15)),
            'x2': [0.1 * i for i in range(1, 15)],
        })

        with pytest.warns(UserWarning, match="Controls not applied"):
            spec = prepare_controls(data, 'd_', 'id', ['x1', 'x2'],
                                     N_treated=5, N_control=2)
        assert spec['include'] is False

    def test_warning_message_format(self):
        """Warning message should follow the Stata-aligned format."""
        data = pd.DataFrame({
            'id': np.repeat(range(1, 7), 2),
            'd_': [1]*6 + [0]*6,
            'x1': [1.0]*2 + [2.0]*2 + [3.0]*2 + [4.0]*2 + [5.0]*2 + [6.0]*2,
            'x2': [0.1]*2 + [0.2]*2 + [0.3]*2 + [0.4]*2 + [0.5]*2 + [0.6]*2,
        })

        with pytest.warns(UserWarning) as w:
            prepare_controls(data, 'd_', 'id', ['x1', 'x2'], N_treated=3, N_control=3)

        msg = str(w[0].message)
        assert "Controls not applied: sample does not satisfy N_1 > K+1 and N_0 > K+1" in msg
        assert "Controls will be ignored" in msg


# ============================================================================
# Cluster standard errors
# ============================================================================

class TestClusterSE:
    """Tests for cluster-robust standard error computation.

    Validates parameter validation, computation, and small-sample warnings
    following Cameron & Miller (2015) guidelines.
    """

    def test_cluster_vce_invalid_type(self, smoking_data):
        """Invalid vce type should raise InvalidVCETypeError."""
        from lwdid import lwdid
        from lwdid.exceptions import InvalidVCETypeError

        with pytest.raises(InvalidVCETypeError, match="Invalid vce type"):
            lwdid(
                smoking_data,
                y='lcigsale', d='d', ivar='state',
                tvar='year', post='post', rolling='demean',
                vce='unknown',
            )

    def test_cluster_var_required(self, smoking_data):
        """vce='cluster' without cluster_var should raise InvalidParameterError."""
        from lwdid import lwdid
        from lwdid.exceptions import InvalidParameterError

        with pytest.raises(InvalidParameterError, match="requires cluster_var"):
            lwdid(
                smoking_data,
                y='lcigsale', d='d', ivar='state',
                tvar='year', post='post', rolling='demean',
                vce='cluster', cluster_var=None,
            )

    def test_cluster_var_must_exist(self, smoking_data):
        """cluster_var column must exist in the data."""
        from lwdid import lwdid
        from lwdid.exceptions import InvalidParameterError

        with pytest.raises(InvalidParameterError, match="not found in data"):
            lwdid(
                smoking_data,
                y='lcigsale', d='d', ivar='state',
                tvar='year', post='post', rolling='demean',
                vce='cluster', cluster_var='nonexistent',
            )

    def test_cluster_minimum_clusters(self):
        """Cluster variable must have at least 2 unique values."""
        from lwdid import lwdid
        from lwdid.exceptions import InvalidParameterError

        data_single = pd.DataFrame({
            'unit': [1, 1, 2, 2, 3, 3, 4, 4],
            'year': [1, 2, 1, 2, 1, 2, 1, 2],
            'y': [5.0, 5.1, 4.5, 4.8, 5.2, 5.5, 4.9, 5.0],
            'd': [0, 0, 0, 0, 1, 1, 1, 1],
            'post': [0, 1, 0, 1, 0, 1, 0, 1],
            'cluster': ['A'] * 8,
        })

        with pytest.raises(InvalidParameterError, match="must have at least 2"):
            lwdid(
                data_single,
                y='y', d='d', ivar='unit',
                tvar='year', post='post', rolling='demean',
                vce='cluster', cluster_var='cluster',
            )

    def test_cluster_se_computation(self, smoking_data):
        """Cluster SE should be finite and positive on the smoking dataset."""
        from lwdid import lwdid

        result = lwdid(
            smoking_data,
            y='lcigsale', d='d', ivar='state',
            tvar='year', post='post', rolling='demean',
            vce='cluster', cluster_var='state',
        )
        assert result.vce_type == 'cluster'
        assert result.cluster_var == 'state'
        assert result.n_clusters == 39
        assert result.se_att > 0
        assert np.isfinite(result.se_att)

    def test_cluster_se_vs_robust(self, smoking_data):
        """On firstpost data (one obs per cluster), cluster SE ~ robust SE."""
        from lwdid import lwdid

        r_cluster = lwdid(
            smoking_data,
            y='lcigsale', d='d', ivar='state',
            tvar='year', post='post', rolling='demean',
            vce='cluster', cluster_var='state',
        )
        r_robust = lwdid(
            smoking_data,
            y='lcigsale', d='d', ivar='state',
            tvar='year', post='post', rolling='demean',
            vce='robust',
        )

        rel_diff = abs(r_cluster.se_att - r_robust.se_att) / r_robust.se_att
        assert rel_diff < 0.20, f"Relative difference too large: {rel_diff:.2%}"

    def test_cluster_se_att_by_period(self, smoking_data):
        """Period-by-period SEs under cluster VCE should be finite and positive."""
        from lwdid import lwdid

        result = lwdid(
            smoking_data,
            y='lcigsale', d='d', ivar='state',
            tvar='year', post='post', rolling='demean',
            vce='cluster', cluster_var='state',
        )
        assert result.att_by_period is not None

        period_rows = result.att_by_period[result.att_by_period['period'] != 'average']
        assert all(period_rows['se'] > 0)
        assert all(np.isfinite(period_rows['se']))

    def test_vce_options_compatibility(self, smoking_data):
        """All supported VCE options should produce valid results."""
        from lwdid import lwdid

        for vce_type in [None, 'robust', 'hc3']:
            result = lwdid(
                smoking_data,
                y='lcigsale', d='d', ivar='state',
                tvar='year', post='post', rolling='demean',
                vce=vce_type,
            )
            assert result is not None and result.att is not None

        result = lwdid(
            smoking_data,
            y='lcigsale', d='d', ivar='state',
            tvar='year', post='post', rolling='demean',
            vce='cluster', cluster_var='state',
        )
        assert result.vce_type == 'cluster'

    def test_cluster_se_small_sample_warning(self):
        """Fewer than 10 clusters should trigger a small-sample warning."""
        from lwdid import lwdid

        np.random.seed(123)
        simple_data = pd.DataFrame({
            'unit': [1, 1, 2, 2, 3, 3, 4, 4] * 5,
            'year': list(range(1, 11)) * 4,
            'y': np.random.normal(5.0, 1.0, 40),
            'd': [0, 0, 0, 0, 1, 1, 1, 1] * 5,
            'post': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 4,
        })

        with pytest.warns(UserWarning, match=r"\d+ clusters may be unreliable"):
            result = lwdid(
                simple_data,
                y='y', d='d', ivar='unit',
                tvar='year', post='post', rolling='demean',
                vce='cluster', cluster_var='unit',
            )
            assert result.n_clusters == 4
            assert result.se_att > 0

    def test_cluster_se_with_controls(self, smoking_data):
        """Cluster SE should work correctly when covariates are included."""
        from lwdid import lwdid

        df = smoking_data.copy()
        df['retprice_ti'] = df.groupby('state')['retprice'].transform('first')
        df['age15to24_ti'] = df.groupby('state')['age15to24'].transform('first')

        result = lwdid(
            df,
            y='lcigsale', d='d', ivar='state',
            tvar='year', post='post', rolling='demean',
            controls=['retprice_ti', 'age15to24_ti'],
            vce='cluster', cluster_var='state',
        )
        assert result.vce_type == 'cluster'
        assert result.se_att > 0
