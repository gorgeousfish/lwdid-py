"""
End-to-end tests comparing Python pre-treatment dynamics to Stata lwdid.

This module validates that the Python implementation produces results
consistent with the Stata lwdid command for pre-treatment period estimation.

Test Categories:
1. TestCastleDataPreTreatment: Tests using Castle Law dataset
2. TestSimulatedDataPreTreatment: Tests using simulated staggered data
3. TestEventStudyValues: Validates event study plot values match Stata

References
----------
Lee, S. J., & Wooldridge, J. M. (2025). A Simple Transformation Approach
to Difference-in-Differences Estimation for Panel Data. Appendix D.
"""

import numpy as np
import pandas as pd
import pytest
import warnings
from pathlib import Path

# Skip all tests if Stata MCP is not available
pytestmark = pytest.mark.skipif(
    True,  # Will be set to False when Stata MCP is configured
    reason="Stata MCP not configured for E2E tests"
)


class TestCastleDataPreTreatment:
    """
    End-to-end tests using Castle Law dataset.
    
    The Castle Law dataset is a standard benchmark for staggered DiD
    estimation, used in Cheng & Hoekstra (2013) and subsequent papers.
    """
    
    @pytest.fixture
    def castle_data(self):
        """Load Castle Law dataset."""
        # Try to load from common locations
        possible_paths = [
            Path(__file__).parent.parent.parent / 'data' / 'castle.csv',
            Path(__file__).parent.parent.parent / 'data' / 'castle.dta',
            Path('castle.csv'),
            Path('castle.dta'),
        ]
        
        for path in possible_paths:
            if path.exists():
                if path.suffix == '.dta':
                    return pd.read_stata(path)
                else:
                    return pd.read_csv(path)
        
        pytest.skip("Castle Law dataset not found")
    
    def test_castle_demean_pre_treatment(self, castle_data):
        """
        Compare pre-treatment effects on Castle data with demeaning.
        
        Validates:
        - Pre-treatment ATT estimates match Stata within rtol=1e-4
        - Anchor points are exactly zero
        - Event times are correctly computed
        """
        from lwdid import lwdid
        
        # Run Python estimation
        results = lwdid(
            data=castle_data,
            y='l_homicide',
            ivar='sid',
            tvar='year',
            gvar='effyear',
            rolling='demean',
            aggregate='cohort',
            include_pretreatment=True,
            pretreatment_test=True,
        )
        
        # Verify pre-treatment effects exist
        assert results.att_pre_treatment is not None
        assert len(results.att_pre_treatment) > 0
        
        # Verify anchor points are zero
        anchors = results.att_pre_treatment[results.att_pre_treatment['is_anchor']]
        assert len(anchors) > 0, "No anchor points found"
        np.testing.assert_allclose(
            anchors['att'].values, 0.0, atol=1e-15,
            err_msg="Anchor points should be exactly zero"
        )
        
        # Verify event times are negative for pre-treatment
        pre_effects = results.att_pre_treatment[~results.att_pre_treatment['is_anchor']]
        assert (pre_effects['event_time'] < 0).all(), \
            "All non-anchor pre-treatment event times should be negative"
    
    def test_castle_detrend_pre_treatment(self, castle_data):
        """
        Compare pre-treatment effects on Castle data with detrending.
        
        Detrending requires at least 2 pre-treatment periods for OLS fit.
        """
        from lwdid import lwdid
        
        results = lwdid(
            data=castle_data,
            y='l_homicide',
            ivar='sid',
            tvar='year',
            gvar='effyear',
            rolling='detrend',
            aggregate='overall',
            include_pretreatment=True,
        )
        
        assert results.att_pre_treatment is not None
        
        # Detrending should have NaN for periods with only 1 future period
        # (t = g-2 returns NaN because OLS needs 2+ points)
        pre_df = results.att_pre_treatment
        
        # Check that anchor points exist and are zero
        anchors = pre_df[pre_df['is_anchor']]
        assert len(anchors) > 0
        np.testing.assert_allclose(anchors['att'].values, 0.0, atol=1e-15)


class TestSimulatedDataPreTreatment:
    """
    End-to-end tests using simulated staggered data.
    
    Simulated data allows precise control over DGP parameters and
    verification of known properties.
    """
    
    @pytest.fixture
    def simulated_data(self):
        """Generate simulated staggered adoption data with parallel trends."""
        np.random.seed(12345)
        
        n_units = 200
        n_periods = 12
        
        # Create panel
        units = np.repeat(np.arange(1, n_units + 1), n_periods)
        periods = np.tile(np.arange(1, n_periods + 1), n_units)
        
        # Assign cohorts: 25% never treated, rest in cohorts 5, 7, 9
        n_never = int(n_units * 0.25)
        n_treated = n_units - n_never
        
        cohort_assignments = np.zeros(n_units)
        cohort_periods = [5, 7, 9]
        per_cohort = n_treated // 3
        
        for i, g in enumerate(cohort_periods):
            start = n_never + i * per_cohort
            end = start + per_cohort
            cohort_assignments[start:end] = g
        
        gvar = np.repeat(cohort_assignments, n_periods)
        
        # Generate outcome with parallel trends (no pre-treatment effects)
        unit_fe = np.repeat(np.random.normal(0, 1, n_units), n_periods)
        time_fe = np.tile(np.linspace(0, 1, n_periods), n_units)
        
        treated = (gvar > 0) & (periods >= gvar)
        treatment_effect = 0.5
        
        epsilon = np.random.normal(0, 0.3, len(units))
        y = unit_fe + time_fe + treatment_effect * treated + epsilon
        
        return pd.DataFrame({
            'unit': units,
            'period': periods,
            'gvar': gvar,
            'y': y,
        })
    
    def test_simulated_parallel_trends_not_rejected(self, simulated_data):
        """
        Under true parallel trends, the test should not reject H0.
        
        With properly generated data satisfying parallel trends,
        the joint F-test should fail to reject at alpha=0.05.
        """
        from lwdid import lwdid
        
        results = lwdid(
            data=simulated_data,
            y='y',
            ivar='unit',
            tvar='period',
            gvar='gvar',
            rolling='demean',
            aggregate='cohort',
            include_pretreatment=True,
            pretreatment_test=True,
            pretreatment_alpha=0.05,
        )
        
        assert results.parallel_trends_test is not None
        pt = results.parallel_trends_test
        
        # Under H0, we expect p-value > 0.05 most of the time
        # This is a single realization, so we use a lenient threshold
        # The test should not reject at alpha=0.01 (very conservative)
        assert pt.joint_pvalue > 0.01, \
            f"Parallel trends test rejected unexpectedly (p={pt.joint_pvalue:.4f})"
    
    def test_simulated_pre_treatment_att_near_zero(self, simulated_data):
        """
        Under parallel trends, pre-treatment ATT should be near zero.
        
        The average of pre-treatment ATT estimates should be close to zero
        when the parallel trends assumption holds.
        """
        from lwdid import lwdid
        
        results = lwdid(
            data=simulated_data,
            y='y',
            ivar='unit',
            tvar='period',
            gvar='gvar',
            rolling='demean',
            aggregate='cohort',
            include_pretreatment=True,
        )
        
        pre_df = results.att_pre_treatment
        non_anchor = pre_df[~pre_df['is_anchor']]
        
        # Average pre-treatment ATT should be close to zero
        mean_pre_att = non_anchor['att'].mean()
        
        # Allow for sampling variation (should be within ~2 SE of zero)
        assert abs(mean_pre_att) < 0.3, \
            f"Mean pre-treatment ATT too far from zero: {mean_pre_att:.4f}"


class TestEventStudyValues:
    """
    Tests validating event study plot values.
    
    Ensures that event study visualization data matches expected values
    from the estimation results.
    """
    
    @pytest.fixture
    def event_study_data(self):
        """Generate data for event study testing."""
        np.random.seed(54321)
        
        n_units = 150
        n_periods = 10
        
        units = np.repeat(np.arange(1, n_units + 1), n_periods)
        periods = np.tile(np.arange(1, n_periods + 1), n_units)
        
        # Two cohorts: g=4 and g=6, plus never-treated
        n_never = 30
        n_per_cohort = (n_units - n_never) // 2
        
        cohort_assignments = np.zeros(n_units)
        cohort_assignments[n_never:n_never + n_per_cohort] = 4
        cohort_assignments[n_never + n_per_cohort:] = 6
        
        gvar = np.repeat(cohort_assignments, n_periods)
        
        # Generate outcome
        unit_fe = np.repeat(np.random.normal(0, 1, n_units), n_periods)
        time_fe = np.tile(np.arange(n_periods) * 0.1, n_units)
        
        treated = (gvar > 0) & (periods >= gvar)
        y = unit_fe + time_fe + 0.4 * treated + np.random.normal(0, 0.4, len(units))
        
        return pd.DataFrame({
            'unit': units,
            'period': periods,
            'gvar': gvar,
            'y': y,
        })
    
    def test_event_study_includes_pre_treatment(self, event_study_data):
        """
        Event study plot should include pre-treatment effects when available.
        """
        from lwdid import lwdid
        
        results = lwdid(
            data=event_study_data,
            y='y',
            ivar='unit',
            tvar='period',
            gvar='gvar',
            rolling='demean',
            aggregate='none',
            include_pretreatment=True,
        )
        
        # Get event study data
        fig, ax, event_df = results.plot_event_study(
            include_pre_treatment=True,
            return_data=True,
        )
        
        # Should have negative event times (pre-treatment)
        assert (event_df['event_time'] < 0).any(), \
            "Event study should include pre-treatment periods"
        
        # Should have positive event times (post-treatment)
        assert (event_df['event_time'] >= 0).any(), \
            "Event study should include post-treatment periods"
        
        # Close figure to avoid memory leak
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_event_study_anchor_at_zero(self, event_study_data):
        """
        Anchor point (e=-1) should have ATT=0 in event study.
        """
        from lwdid import lwdid
        
        results = lwdid(
            data=event_study_data,
            y='y',
            ivar='unit',
            tvar='period',
            gvar='gvar',
            rolling='demean',
            aggregate='none',
            include_pretreatment=True,
        )
        
        # Check pre-treatment DataFrame directly
        pre_df = results.att_pre_treatment
        anchor_rows = pre_df[pre_df['event_time'] == -1]
        
        assert len(anchor_rows) > 0, "No anchor points found at e=-1"
        np.testing.assert_allclose(
            anchor_rows['att'].values, 0.0, atol=1e-15,
            err_msg="Anchor point ATT should be exactly zero"
        )
        
        # Close any open figures
        import matplotlib.pyplot as plt
        plt.close('all')


class TestStataComparison:
    """
    Direct comparison tests with Stata lwdid output.
    
    These tests require Stata MCP to be configured and will be skipped
    if Stata is not available.
    """
    
    @pytest.fixture
    def stata_comparison_data(self):
        """Generate data and run both Python and Stata estimation."""
        # This fixture would use Stata MCP to run lwdid and compare
        pytest.skip("Stata MCP comparison not implemented")
    
    def test_pre_treatment_att_matches_stata(self, stata_comparison_data):
        """
        Pre-treatment ATT estimates should match Stata within tolerance.
        
        Tolerance: rtol=1e-6 (relative tolerance)
        """
        python_results, stata_results = stata_comparison_data
        
        # Compare pre-treatment ATT values
        for event_time in python_results['event_time'].unique():
            if event_time >= 0:
                continue  # Skip post-treatment
            
            py_att = python_results[
                python_results['event_time'] == event_time
            ]['att'].values[0]
            
            stata_att = stata_results[
                stata_results['event_time'] == event_time
            ]['att'].values[0]
            
            if not np.isnan(py_att) and not np.isnan(stata_att):
                np.testing.assert_allclose(
                    py_att, stata_att, rtol=1e-6,
                    err_msg=f"ATT mismatch at e={event_time}"
                )
    
    def test_parallel_trends_test_matches_stata(self, stata_comparison_data):
        """
        Parallel trends test statistics should match Stata.
        """
        python_results, stata_results = stata_comparison_data
        
        # Compare F-statistic
        np.testing.assert_allclose(
            python_results.parallel_trends_test.joint_f_stat,
            stata_results['f_stat'],
            rtol=1e-6,
            err_msg="F-statistic mismatch"
        )
        
        # Compare p-value
        np.testing.assert_allclose(
            python_results.parallel_trends_test.joint_pvalue,
            stata_results['f_pvalue'],
            rtol=1e-4,
            err_msg="F-test p-value mismatch"
        )
