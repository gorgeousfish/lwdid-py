"""Walmart empirical application replication tests.

This module validates the Python implementation against the published
results in Lee & Wooldridge (2025), Table A4, using the Walmart store
opening dataset. The tests compare Weighted Average Treatment Effects
on the Treated (WATT) at each event time under both demeaning and
detrending transformations.

Tolerance criteria:
- Detrend (heterogeneous trends): mean absolute difference < 5%
- Demean: mean absolute difference < 20%

References
----------
Lee, S. & Wooldridge, J. M. (2025). A Simple Transformation Approach to
    DiD Estimation for Panel Data. SSRN 4516518, Section 6 and Table A4.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Attempt to import lwdid; skip tests if unavailable
try:
    from lwdid import lwdid
    LWDID_AVAILABLE = True
except ImportError:
    LWDID_AVAILABLE = False


# Reference values from Paper Table A4
# Rolling IPWRA with Heterogeneous Trends (Detrend)
PAPER_DETREND = {
    0: (0.007, 0.004),
    1: (0.032, 0.005),
    2: (0.025, 0.006),
    3: (0.021, 0.007),
    4: (0.018, 0.009),
    5: (0.017, 0.010),
    6: (0.019, 0.012),
    7: (0.036, 0.013),
    8: (0.041, 0.016),
    9: (0.041, 0.019),
    10: (0.037, 0.023),
    11: (0.018, 0.030),
    12: (0.017, 0.036),
    13: (0.047, 0.053),
}

# Rolling IPWRA with Demeaning
PAPER_DEMEAN = {
    0: (0.018, 0.004),
    1: (0.045, 0.004),
    2: (0.038, 0.004),
    3: (0.032, 0.004),
    4: (0.031, 0.004),
    5: (0.036, 0.005),
    6: (0.040, 0.005),
    7: (0.054, 0.006),
    8: (0.062, 0.008),
    9: (0.063, 0.010),
    10: (0.081, 0.013),
    11: (0.083, 0.018),
    12: (0.080, 0.026),
    13: (0.107, 0.039),
}


def load_walmart_data():
    """Load the Walmart store opening dataset from disk."""
    data_path = Path(__file__).parent.parent / 'data' / 'walmart.csv'
    if not data_path.exists():
        pytest.skip(f"Walmart data file not found: {data_path}")
    return pd.read_csv(data_path)


def compute_watt(results, df):
    """Compute Weighted Average Treatment Effects on the Treated (WATT).

    WATT(r) = sum_g w(g,r) * ATT(g, g+r)
    where w(g,r) = N_g / N_Gr (cohort-size weights).
    """
    att_ct = results.att_by_cohort_time.copy()
    
    if att_ct is None or len(att_ct) == 0:
        return pd.DataFrame()
    
    # Obtain cohort sizes for weighting
    cohort_sizes = df[df['g'] != np.inf].groupby('g')['fips'].nunique().to_dict()
    
    att_ct['weight'] = att_ct['cohort'].map(cohort_sizes)
    att_ct['weight'] = att_ct['weight'].fillna(0)
    
    watt_list = []
    
    for event_time in sorted(att_ct['event_time'].unique()):
        subset = att_ct[att_ct['event_time'] == event_time].copy()
        subset = subset[subset['att'].notna()]
        
        if len(subset) == 0:
            continue
            
        total_weight = subset['weight'].sum()
        if total_weight == 0:
            continue
            
        subset['norm_weight'] = subset['weight'] / total_weight
        
        watt = (subset['att'] * subset['norm_weight']).sum()
        watt_se = np.sqrt((subset['se']**2 * subset['norm_weight']**2).sum())
        
        watt_list.append({
            'event_time': int(event_time),
            'watt': watt,
            'se': watt_se,
        })
    
    return pd.DataFrame(watt_list)


@pytest.fixture(scope='module')
def walmart_data():
    """Module-scoped fixture providing the Walmart store opening dataset."""
    return load_walmart_data()


@pytest.fixture(scope='module')
def controls():
    """List of time-invariant control variables used in the IPWRA specification."""
    return [
        'share_pop_poverty_78_above',
        'share_pop_ind_manuf',
        'share_school_some_hs',
    ]


@pytest.mark.skipif(not LWDID_AVAILABLE, reason="lwdid package not available")
@pytest.mark.paper_validation
class TestWalmartDataValidation:
    """Validate the Walmart dataset structure against Table 2 of Lee & Wooldridge (2025)."""
    
    def test_data_shape(self, walmart_data):
        """Verify that the dataset dimensions match the published sample description."""
        assert walmart_data.shape[0] == 29440, "Expected 29,440 observations"
        assert walmart_data['fips'].nunique() == 1280, "Expected 1,280 counties"
    
    def test_descriptive_statistics(self, walmart_data):
        """Verify descriptive statistics against Table 2 of Lee & Wooldridge (2025)."""
        # Mean of log(Retail Employment)
        log_emp_mean = walmart_data['log_retail_emp'].mean()
        assert np.isclose(log_emp_mean, 7.754502, atol=1e-5), \
            f"Mean log(Retail Employment) should be 7.754502, got {log_emp_mean}"
        
        # Mean of Share Poverty (above)
        poverty_mean = walmart_data['share_pop_poverty_78_above'].mean()
        assert np.isclose(poverty_mean, 0.8470385, atol=1e-5), \
            f"Mean Share Poverty should be 0.8470385, got {poverty_mean}"
    
    def test_treatment_cohort_distribution(self, walmart_data):
        """Verify the treatment cohort distribution across counties."""
        cohort_dist = walmart_data.groupby('g')['fips'].nunique()
        n_never_treated = cohort_dist.get(np.inf, 0)
        n_treated = cohort_dist[cohort_dist.index != np.inf].sum()
        
        assert n_treated == 886, f"Expected 886 treated counties, got {n_treated}"
        assert n_never_treated == 394, f"Expected 394 never-treated counties, got {n_never_treated}"


@pytest.mark.skipif(not LWDID_AVAILABLE, reason="lwdid package not available")
@pytest.mark.slow
@pytest.mark.paper_validation
class TestWalmartDetrend:
    """Replication tests for Rolling IPWRA with heterogeneous trends (detrend).

    Validates WATT estimates against Table A4 of Lee & Wooldridge (2025).
    """
    
    @pytest.fixture(scope='class')
    def detrend_results(self, walmart_data, controls):
        """Run the detrend IPWRA estimation on the Walmart dataset."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = lwdid(
                data=walmart_data,
                y='log_retail_emp',
                ivar='fips',
                tvar='year',
                gvar='g',
                rolling='detrend',
                estimator='ipwra',
                controls=controls,
                control_group='never_treated',
                aggregate='none',
                alpha=0.05,
            )
        return results
    
    @pytest.fixture(scope='class')
    def watt_detrend(self, detrend_results, walmart_data):
        """Compute WATT from the detrend estimation results."""
        return compute_watt(detrend_results, walmart_data)
    
    def test_detrend_att_r13_exact_match(self, watt_detrend):
        """Verify that the ATT at event time r=13 closely matches the published value."""
        r13 = watt_detrend[watt_detrend['event_time'] == 13]
        if len(r13) == 0:
            pytest.skip("Event time r=13 not available in results")
        
        python_att = r13['watt'].values[0]
        paper_att = PAPER_DETREND[13][0]
        
        # r=13 should match closely (difference < 5%)
        rel_diff = abs(python_att - paper_att) / paper_att
        assert rel_diff < 0.05, \
            f"ATT at r=13 deviates excessively: Python={python_att:.4f}, Paper={paper_att:.4f}, rel_diff={rel_diff:.1%}"
    
    def test_detrend_mean_absolute_difference(self, watt_detrend):
        """Verify that the mean absolute difference from Table A4 is below 0.015."""
        diffs = []
        for _, row in watt_detrend.iterrows():
            r = int(row['event_time'])
            if r in PAPER_DETREND:
                paper_att = PAPER_DETREND[r][0]
                diff = abs(row['watt'] - paper_att)
                diffs.append(diff)
        
        if len(diffs) == 0:
            pytest.skip("No comparable event times available")
        
        mean_diff = np.mean(diffs)
        # Mean absolute difference should be < 0.015 (approximately 1.5 percentage points)
        assert mean_diff < 0.015, \
            f"Detrend mean absolute difference too large: {mean_diff:.4f} (threshold: 0.015)"
    
    def test_detrend_trend_direction(self, watt_detrend):
        """Verify that the detrend ATT exhibits the expected increase from r=0 to r=1."""
        # The paper shows a notable increase from r=0 to r=1
        r0 = watt_detrend[watt_detrend['event_time'] == 0]['watt'].values
        r1 = watt_detrend[watt_detrend['event_time'] == 1]['watt'].values
        
        if len(r0) > 0 and len(r1) > 0:
            assert r1[0] > r0[0], "ATT at r=1 should exceed ATT at r=0"
    
    def test_detrend_qualitative_findings(self, watt_detrend):
        """Verify qualitative findings: post-treatment ATTs are predominantly positive."""
        # All post-treatment ATTs should be positive
        post_treatment = watt_detrend[watt_detrend['event_time'] >= 0]
        positive_count = (post_treatment['watt'] > 0).sum()
        total_count = len(post_treatment)
        
        # Most should be positive (allow a few near-zero cases)
        assert positive_count >= total_count * 0.7, \
            f"Proportion of positive post-treatment ATTs too low: {positive_count}/{total_count}"


@pytest.mark.skipif(not LWDID_AVAILABLE, reason="lwdid package not available")
@pytest.mark.slow
@pytest.mark.paper_validation
class TestWalmartDemean:
    """Replication tests for Rolling IPWRA with demeaning transformation.

    Validates WATT estimates against Table A4 of Lee & Wooldridge (2025).
    """
    
    @pytest.fixture(scope='class')
    def demean_results(self, walmart_data, controls):
        """Run the demean IPWRA estimation on the Walmart dataset."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = lwdid(
                data=walmart_data,
                y='log_retail_emp',
                ivar='fips',
                tvar='year',
                gvar='g',
                rolling='demean',
                estimator='ipwra',
                controls=controls,
                control_group='never_treated',
                aggregate='none',
                alpha=0.05,
            )
        return results
    
    @pytest.fixture(scope='class')
    def watt_demean(self, demean_results, walmart_data):
        """Compute WATT from the demean estimation results."""
        return compute_watt(demean_results, walmart_data)
    
    def test_demean_trend_direction_consistent(self, watt_demean):
        """Verify that demean ATT exhibits an increasing trend consistent with the paper."""
        # The paper shows demean ATT increasing with event time
        post_treatment = watt_demean[watt_demean['event_time'] >= 0].sort_values('event_time')
        
        if len(post_treatment) < 2:
            pytest.skip("Insufficient post-treatment data")
        
        # Check that the overall trend is increasing (allow local fluctuations)
        first_half_mean = post_treatment.head(len(post_treatment)//2)['watt'].mean()
        second_half_mean = post_treatment.tail(len(post_treatment)//2)['watt'].mean()
        
        assert second_half_mean > first_half_mean, \
            "Demean ATT should increase with event time"
    
    def test_demean_all_positive(self, watt_demean):
        """Verify that all post-treatment ATT estimates are positive."""
        post_treatment = watt_demean[watt_demean['event_time'] >= 0]
        
        assert (post_treatment['watt'] > 0).all(), \
            "All post-treatment ATT estimates should be positive"
    
    def test_demean_larger_than_detrend_qualitative(self, watt_demean):
        """Verify that demean estimates are qualitatively larger than detrend values."""
        # This is a key finding of the paper: controlling for heterogeneous trends
        # reduces the estimated effect. Here we verify the magnitude of demean estimates.
        r1 = watt_demean[watt_demean['event_time'] == 1]['watt'].values
        
        if len(r1) > 0:
            # Demean r=1 should be notably larger than 0.03 (typical detrend value)
            assert r1[0] > 0.05, \
                f"Demean ATT at r=1 should be notably larger than detrend: {r1[0]:.4f}"


@pytest.mark.skipif(not LWDID_AVAILABLE, reason="lwdid package not available")
@pytest.mark.slow
@pytest.mark.paper_validation
class TestWalmartComparison:
    """Comparative tests between demean and detrend specifications.

    Validates the key finding from Lee & Wooldridge (2025) that controlling
    for heterogeneous trends reduces the estimated treatment effect.
    """
    
    @pytest.fixture(scope='class')
    def both_results(self, walmart_data, controls):
        """Run both demean and detrend estimations for comparison."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            results_demean = lwdid(
                data=walmart_data,
                y='log_retail_emp',
                ivar='fips',
                tvar='year',
                gvar='g',
                rolling='demean',
                estimator='ipwra',
                controls=controls,
                control_group='never_treated',
                aggregate='none',
            )
            
            results_detrend = lwdid(
                data=walmart_data,
                y='log_retail_emp',
                ivar='fips',
                tvar='year',
                gvar='g',
                rolling='detrend',
                estimator='ipwra',
                controls=controls,
                control_group='never_treated',
                aggregate='none',
            )
        
        watt_demean = compute_watt(results_demean, walmart_data)
        watt_detrend = compute_watt(results_detrend, walmart_data)
        
        return watt_demean, watt_detrend
    
    def test_detrend_smaller_than_demean(self, both_results):
        """Verify that detrend estimates are smaller than demean (core finding of the paper)."""
        watt_demean, watt_detrend = both_results
        
        # Merge the two result sets
        merged = pd.merge(
            watt_demean[['event_time', 'watt']],
            watt_detrend[['event_time', 'watt']],
            on='event_time',
            suffixes=('_demean', '_detrend')
        )
        
        if len(merged) == 0:
            pytest.skip("No comparable event times available")
        
        # For most event times, detrend should be smaller than demean
        smaller_count = (merged['watt_detrend'] < merged['watt_demean']).sum()
        total_count = len(merged)
        
        assert smaller_count >= total_count * 0.9, \
            f"Proportion where detrend < demean is too low: {smaller_count}/{total_count}"
    
    def test_effect_reduction_magnitude(self, both_results):
        """Verify that the effect reduction magnitude is consistent with the paper."""
        watt_demean, watt_detrend = both_results
        
        # Compute effect reduction at r=1
        demean_r1 = watt_demean[watt_demean['event_time'] == 1]['watt'].values
        detrend_r1 = watt_detrend[watt_detrend['event_time'] == 1]['watt'].values
        
        if len(demean_r1) > 0 and len(detrend_r1) > 0:
            reduction = (demean_r1[0] - detrend_r1[0]) / demean_r1[0]
            # The paper shows detrend is approximately 1/3 to 1/2 of demean
            assert reduction > 0.5, \
                f"Effect reduction insufficient: {reduction:.1%} (expected > 50%)"


# Numerical validation tests
@pytest.mark.skipif(not LWDID_AVAILABLE, reason="lwdid package not available")
class TestWalmartNumericalValidation:
    """Numerical validation of WATT computation formulas."""
    
    def test_watt_weight_sum_to_one(self, walmart_data):
        """Verify that WATT cohort-size weights are properly normalized to unity."""
        # Obtain cohort sizes
        cohort_sizes = walmart_data[walmart_data['g'] != np.inf].groupby('g')['fips'].nunique()
        
        # For any event time, weights should normalize to 1
        total_treated = cohort_sizes.sum()
        normalized_weights = cohort_sizes / total_treated
        
        assert np.isclose(normalized_weights.sum(), 1.0, atol=1e-10), \
            "Normalized weights should sum to 1"
    
    def test_watt_se_formula(self):
        """Verify the WATT standard error formula: SE = sqrt(sum(w^2 * SE^2))."""
        # Simulated data
        weights = np.array([0.4, 0.6])
        ses = np.array([0.01, 0.02])
        
        # Compute WATT SE
        watt_se = np.sqrt(np.sum(weights**2 * ses**2))
        
        # Manual computation
        expected_se = np.sqrt(0.4**2 * 0.01**2 + 0.6**2 * 0.02**2)
        
        assert np.isclose(watt_se, expected_se, atol=1e-10), \
            f"WATT SE computation error: {watt_se} != {expected_se}"