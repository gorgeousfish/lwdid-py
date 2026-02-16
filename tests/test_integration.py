"""
End-to-end integration tests with Stata-aligned benchmarks.

Validates the full ``lwdid()`` pipeline against reference values from
Lee & Wooldridge (2023) Table 3 using the California Proposition 99
smoking dataset and hand-constructed MVE (Minimum Verifiable Example) data.

Test IDs follow the PRD test matrix (T001--T020, B001--B020).

References
----------
Lee, S. J. & Wooldridge, J. M. (2023). Table 3 — Smoking data benchmarks.
Lee, S. J. & Wooldridge, J. M. (2025). Procedure 2.1 and 3.1.

Stata command:
    lwdid lcigsale d, ivar(state) tvar(year) post(post) rolling(demean)
"""

import numpy as np
import pandas as pd
import pytest

from lwdid import lwdid
from lwdid.exceptions import (
    NoControlUnitsError,
    NoTreatedUnitsError,
    TimeDiscontinuityError,
)


# ============================================================================
# Smoking data — Stata alignment (Paper Table 3)
# ============================================================================

class TestSmokingDataIntegration:
    """End-to-end tests on the smoking dataset against Paper Table 3."""

    @pytest.mark.stata_alignment
    @pytest.mark.paper_validation
    def test_T001_smoking_demean_ols(self, smoking_data):
        """T001: demean + OLS — core Stata alignment test.

        Validates Procedure 2.1 from Lee & Wooldridge (2023), Table 3.

        Stata benchmark (Paper Table 3, Procedure 2.1):
            ATT = -0.422, SE = 0.121, t ~ -3.49, p ~ 0.001, N = 39
        """
        results = lwdid(
            smoking_data,
            y='lcigsale', d='d', ivar='state',
            tvar='year', post='post', rolling='demean', vce=None,
        )

        assert abs(results.att - (-0.422)) < 5e-4, \
            f"ATT mismatch: {results.att:.6f} vs -0.422"
        assert abs(results.se_att - 0.121) < 5e-4, \
            f"SE mismatch: {results.se_att:.6f} vs 0.121"
        assert abs(results.t_stat - (-3.49)) < 1e-2
        assert abs(results.pvalue - 0.001) < 1e-3

        # Exact sample counts
        assert results.nobs == 39
        assert results.K == 19
        assert results.tpost1 == 20
        assert results.n_treated == 1
        assert results.n_control == 38

        # Method metadata
        assert results.cmd == 'lwdid'
        assert results.rolling == 'demean'
        assert results.vce_type == 'ols'
        assert results.depvar == 'lcigsale'

    @pytest.mark.stata_alignment
    def test_T002_smoking_demean_hc3(self, smoking_data):
        """T002: demean + HC3 — point estimate invariance under VCE change.

        HC3 is unstable with N_1 = 1 (leverage h = 1), so only the point
        estimate (not SE) is compared to T001.
        """
        results = lwdid(
            smoking_data,
            y='lcigsale', d='d', ivar='state',
            tvar='year', post='post', rolling='demean', vce='hc3',
        )

        assert abs(results.att - (-0.422)) < 5e-4
        assert results.se_att > 0
        assert not np.isnan(results.se_att)
        assert results.vce_type == 'hc3'
        assert results.nobs == 39

    @pytest.mark.stata_alignment
    @pytest.mark.paper_validation
    def test_T003_smoking_detrend_ols(self, smoking_data):
        """T003: detrend + OLS — Procedure 3.1 alignment.

        Validates Procedure 3.1 from Lee & Wooldridge (2023), Table 3.

        Stata benchmark (exact values):
            ATT = -0.2269887, SE = 0.0940689, t = -2.413005, p = 0.0208919
        """
        results = lwdid(
            smoking_data,
            y='lcigsale', d='d', ivar='state',
            tvar='year', post='post', rolling='detrend', vce=None,
        )

        assert abs(results.att - (-0.2269887)) < 1e-7
        assert abs(results.se_att - 0.0940689) < 1e-7
        assert abs(results.t_stat - (-2.413005)) < 1e-4
        assert abs(results.pvalue - 0.0208919) < 1e-4

        assert results.nobs == 39
        assert results.K == 19
        assert results.tpost1 == 20

        # att_by_period: 1 average row + 12 period rows (1989--2000)
        assert results.att_by_period is not None
        assert isinstance(results.att_by_period, pd.DataFrame)
        assert len(results.att_by_period) == 13

        avg_row = results.att_by_period.iloc[0]
        assert avg_row['period'] == 'average'
        assert avg_row['tindex'] == '-'
        assert abs(avg_row['beta'] - (-0.227)) < 1e-4

        expected_cols = [
            'period', 'tindex', 'beta', 'se',
            'ci_lower', 'ci_upper', 'tstat', 'pval', 'N',
        ]
        assert list(results.att_by_period.columns) == expected_cols

    @pytest.mark.stata_alignment
    def test_T004_smoking_detrend_hc3(self, smoking_data):
        """T004: detrend + HC3 — point estimate invariance."""
        results = lwdid(
            smoking_data,
            y='lcigsale', d='d', ivar='state',
            tvar='year', post='post', rolling='detrend', vce='hc3',
        )
        assert abs(results.att - (-0.227)) < 1e-4
        assert results.se_att > 0 and not np.isnan(results.se_att)
        assert results.vce_type == 'hc3'
        assert results.att_by_period is not None
        assert len(results.att_by_period) == 13

    @pytest.mark.stata_alignment
    @pytest.mark.paper_validation
    def test_T010_period_effect_1989(self, smoking_data):
        """T010: period-specific effect for 1989 (first post-treatment year).

        Validates period-by-period effects from Lee & Wooldridge (2023), Table 3.

        Stata benchmark: beta = -0.04226831, se = 0.0592916,
                         t = -0.7128885, p = 0.4803869
        """
        results = lwdid(
            smoking_data,
            y='lcigsale', d='d', ivar='state',
            tvar='year', post='post', rolling='detrend', vce=None,
        )
        row = results.att_by_period[results.att_by_period['period'] == '1989']
        assert len(row) == 1
        assert abs(row['beta'].values[0] - (-0.04226831)) < 1e-7
        assert abs(row['se'].values[0] - 0.0592916) < 1e-7
        assert abs(row['tstat'].values[0] - (-0.7128885)) < 1e-4
        assert abs(row['pval'].values[0] - 0.4803869) < 1e-4

    @pytest.mark.stata_alignment
    @pytest.mark.paper_validation
    def test_T011_period_effect_1995(self, smoking_data):
        """T011: period-specific effect for 1995.

        Stata benchmark: beta = -0.28203907, se = 0.1121333
        """
        results = lwdid(
            smoking_data,
            y='lcigsale', d='d', ivar='state',
            tvar='year', post='post', rolling='detrend',
        )
        row = results.att_by_period[results.att_by_period['period'] == '1995']
        assert abs(row['beta'].values[0] - (-0.28203907)) < 1e-7
        assert abs(row['se'].values[0] - 0.1121333) < 1e-7

    @pytest.mark.stata_alignment
    @pytest.mark.paper_validation
    def test_T012_period_effect_2000(self, smoking_data):
        """T012: period-specific effect for 2000 (final period).

        Stata benchmark: beta = -0.40287678, se = 0.1524529
        """
        results = lwdid(
            smoking_data,
            y='lcigsale', d='d', ivar='state',
            tvar='year', post='post', rolling='detrend',
        )
        row = results.att_by_period[results.att_by_period['period'] == '2000']
        assert abs(row['beta'].values[0] - (-0.40287678)) < 1e-7
        assert abs(row['se'].values[0] - 0.1524529) < 1e-7


# ============================================================================
# MVE (Minimum Verifiable Example) data
# ============================================================================

class TestMVEIntegration:
    """End-to-end tests on hand-crafted MVE data with known true ATT."""

    def test_mve_demean_end_to_end(self):
        """MVE demean: true ATT = 3.5 by construction."""
        data = pd.read_csv('tests/data/mve_demean.csv')
        results = lwdid(data, 'y', 'd', 'id', 'year', 'post', 'demean')

        assert abs(results.att - 3.5) < 1e-7
        assert results.nobs == 3
        assert results.n_treated == 1
        assert results.n_control == 2
        assert results.K == 2
        assert results.tpost1 == 3

    def test_mve_detrend_end_to_end(self):
        """MVE detrend: true ATT = 5.0 (perfect linear trend).

        DGP: unit 1 (treated) y = 3 + 2t + 5*post;
             units 2-3 (control) follow perfect linear trends.
        """
        data = pd.read_csv('tests/data/mve_detrend.csv')
        if 'tindex' in data.columns:
            data = data.drop(columns=['tindex'])

        results = lwdid(data, 'y', 'd', 'id', 'year', 'post', 'detrend')

        assert abs(results.att - 5.0) < 1e-7
        assert results.nobs == 3
        assert results.K == 3
        assert results.tpost1 == 4
        assert results.att_by_period is not None
        assert len(results.att_by_period) == 3  # 1 average + 2 periods


# ============================================================================
# Boundary conditions
# ============================================================================

class TestBoundaryConditions:
    """Tests for edge cases and invalid inputs."""

    def test_B001_minimum_sample(self):
        """Minimum sample (N=3, N_1=1, N_0=2) should produce valid results."""
        data = pd.read_csv('tests/data/mve_demean.csv')
        results = lwdid(data, 'y', 'd', 'id', 'year', 'post', 'demean')

        assert results.nobs == 3
        assert not np.isnan(results.att)
        assert not np.isnan(results.se_att)

    def test_B005_no_control_units(self):
        """All-treated data should raise NoControlUnitsError."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3],
            'year': [1, 2, 1, 2, 1, 2],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'd': [1, 1, 1, 1, 1, 1],
            'post': [0, 1, 0, 1, 0, 1],
        })
        with pytest.raises(NoControlUnitsError):
            lwdid(data, 'y', 'd', 'id', 'year', 'post', 'demean')

    def test_B006_no_treated_units(self):
        """All-control data should raise NoTreatedUnitsError."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3],
            'year': [1, 2, 1, 2, 1, 2],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'd': [0, 0, 0, 0, 0, 0],
            'post': [0, 1, 0, 1, 0, 1],
        })
        with pytest.raises(NoTreatedUnitsError):
            lwdid(data, 'y', 'd', 'id', 'year', 'post', 'demean')

    def test_B008_string_id_conversion(self):
        """String unit identifiers should be handled automatically."""
        data = pd.DataFrame({
            'state': ['CA', 'CA', 'TX', 'TX', 'NY', 'NY'],
            'year': [1, 2, 1, 2, 1, 2],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'd': [1, 1, 0, 0, 0, 0],
            'post': [0, 1, 0, 1, 0, 1],
        })
        results = lwdid(data, 'y', 'd', 'state', 'year', 'post', 'demean')

        assert results.nobs == 3
        assert not np.isnan(results.att)

    def test_B009_missing_values_dropped(self):
        """Rows with missing outcome values should be dropped with a warning."""
        data = pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'year': [1, 2, 3, 1, 2, 3, 1, 2, 3],
            'y': [1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            'd': [1, 1, 1, 0, 0, 0, 0, 0, 0],
            'post': [0, 0, 1, 0, 0, 1, 0, 0, 1],
        })
        with pytest.warns(UserWarning, match="Dropped 1 observations"):
            results = lwdid(data, 'y', 'd', 'id', 'year', 'post', 'demean')
        assert not np.isnan(results.att)


# ============================================================================
# LWDIDResults object interface
# ============================================================================

class TestResultsObject:
    """Tests for the LWDIDResults public interface."""

    def test_results_attributes(self):
        """All documented attributes should be accessible."""
        data = pd.read_csv('tests/data/mve_demean.csv')
        results = lwdid(data, 'y', 'd', 'id', 'year', 'post', 'demean')

        for attr in ['att', 'se_att', 't_stat', 'pvalue', 'ci_lower', 'ci_upper',
                      'K', 'tpost1', 'nobs', 'n_treated', 'n_control', 'df_resid',
                      'cmd', 'depvar', 'rolling', 'vce_type',
                      'params', 'bse', 'vcov']:
            assert hasattr(results, attr), f"Missing attribute: {attr}"

    def test_summary_output(self):
        """summary() should return a formatted string with key statistics."""
        data = pd.read_csv('tests/data/mve_demean.csv')
        results = lwdid(data, 'y', 'd', 'id', 'year', 'post', 'demean')

        summary = results.summary()
        for keyword in ['lwdid Results', 'demean', 'ATT:', 'Std. Err.:', str(results.nobs)]:
            assert keyword in summary, f"Missing keyword in summary: {keyword}"

    def test_repr_methods(self):
        """__repr__ and __str__ should produce informative output."""
        data = pd.read_csv('tests/data/mve_demean.csv')
        results = lwdid(data, 'y', 'd', 'id', 'year', 'post', 'demean')

        assert 'LWDIDResults' in repr(results)
        assert 'att=' in repr(results)
        assert 'lwdid Results' in str(results)


# ============================================================================
# Quarterly data (Procedure 2.1 / 3.1 with quarterly frequency)
# ============================================================================

class TestQuarterlyData:
    """Tests for quarterly panel data support (demeanq / detrendq)."""

    def test_T014_smoking_quarterly_demeanq(self):
        """T014: quarterly demean — period labels should be '1989q1' format."""
        data = pd.read_csv('tests/data/smoking_quarterly.csv')
        results = lwdid(
            data, y='lcigsale', d='d', ivar='state',
            tvar=['year', 'quarter'], post='post',
            rolling='demeanq', vce=None,
        )

        assert results.att is not None
        assert results.rolling == 'demeanq'
        assert results.att_by_period.iloc[1]['period'] == '1989q1'
        assert results.att_by_period.iloc[-1]['period'] == '2000q4'
        # 1 average + 12 years * 4 quarters = 49 rows
        assert len(results.att_by_period) == 49

    def test_T015_smoking_quarterly_detrendq(self):
        """T015: quarterly detrend — same structure as T014."""
        data = pd.read_csv('tests/data/smoking_quarterly.csv')
        results = lwdid(
            data, y='lcigsale', d='d', ivar='state',
            tvar=['year', 'quarter'], post='post',
            rolling='detrendq', vce=None,
        )

        assert results.rolling == 'detrendq'
        assert results.att_by_period.iloc[1]['period'] == '1989q1'
        assert len(results.att_by_period) == 49
        assert results.se_att > 0


# ============================================================================
# Covariate adjustment (controls parameter)
# ============================================================================

class TestControlVariables:
    """Tests for time-invariant covariate adjustment."""

    def test_T017_controls_included(self):
        """T017: covariates included when both N_1 and N_0 exceed K+1."""
        data = pd.read_csv('tests/data/smoking_controls_large.csv')
        results = lwdid(
            data, y='lcigsale', d='d', ivar='state',
            tvar='year', post='post', rolling='detrend', vce=None,
            controls=['x1', 'x2'],
        )

        # Design matrix: [const, d, x1, x2, d*x1_c, d*x2_c]
        assert len(results.params) == 6
        assert not np.isnan(results.att)

    def test_T018_controls_warning(self):
        """T018: covariates dropped with warning when sample is too small."""
        data = pd.read_csv('tests/data/smoking_controls_small.csv')
        with pytest.warns(UserWarning, match="Controls not applied"):
            results = lwdid(
                data, y='lcigsale', d='d', ivar='state',
                tvar='year', post='post', rolling='detrend', vce=None,
                controls=['x1', 'x2'],
            )
        # Only [const, d] remain
        assert len(results.params) == 2
        assert not np.isnan(results.att)


class TestBoundaryConditionsQuarterly:
    """Boundary tests for quarterly and controls features."""

    def test_B007_invalid_quarter(self):
        """Quarter value outside {1,2,3,4} should raise InvalidParameterError."""
        from lwdid.exceptions import InvalidParameterError

        data = pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2],
            'year': [1, 1, 2, 1, 1, 2],
            'quarter': [1, 5, 1, 1, 2, 1],  # 5 is invalid
            'y': [10.0, 12.0, 15.0, 5.0, 6.0, 7.0],
            'd': [1, 1, 1, 0, 0, 0],
            'post': [0, 0, 1, 0, 0, 1],
        })

        with pytest.raises(InvalidParameterError,
                           match="Quarter variable.*invalid values"):
            lwdid(data, y='y', d='d', ivar='id',
                  tvar=['year', 'quarter'], post='post', rolling='demeanq')

    def test_B020_controls_boundary(self):
        """Controls boundary: N = K+2 should just satisfy the inclusion rule."""
        data = pd.read_csv('tests/data/smoking_controls_large.csv')

        # Subset to 4 treated + 4 control states => N_1=N_0=4, K=2, K+1=3 < 4
        treated_ids = data[data['d'] == 1]['state'].unique()[:4]
        control_ids = data[data['d'] == 0]['state'].unique()[:4]
        subset = data[data['state'].isin(list(treated_ids) + list(control_ids))].copy()

        results = lwdid(
            subset, y='lcigsale', d='d', ivar='state',
            tvar='year', post='post', rolling='detrend', vce=None,
            controls=['x1', 'x2'],
        )

        assert len(results.params) == 6  # controls included
        assert not np.isnan(results.att)
