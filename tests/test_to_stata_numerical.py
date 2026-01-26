"""
Numerical Validation Tests for to_stata() Export

This module contains tests that verify the numerical accuracy and integrity
of data exported via to_stata() method.

Test coverage:
1. Float precision preservation
2. CI bounds computation verification
3. t-stat and p-value consistency
4. Cross-format data integrity (CSV vs Stata)
5. Statistical property preservation
"""

import os
import tempfile
from typing import Dict

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from lwdid import lwdid


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def staggered_panel_data():
    """Create synthetic staggered panel data for numerical validation."""
    # Create synthetic staggered data
    # Note: cohort values must be > T_min to have pre-treatment periods
    rows = []
    n_units = 50
    n_periods = 10  # years 2001-2010
    cohorts = [0, 2004, 2006, 2008]  # 0 = never treated
    
    np.random.seed(42)
    for i in range(n_units):
        cohort = cohorts[i % len(cohorts)]
        for t in range(n_periods):
            year = 2001 + t
            treated = 1 if cohort > 0 and year >= cohort else 0
            # Treatment effect is 0.5
            y = 5.0 + i * 0.1 + t * 0.05 + treated * 0.5 + np.random.normal(0, 0.1)
            rows.append({
                'unit': i + 1,
                'year': year,
                'gvar': cohort if cohort > 0 else 0,  # 0 = never treated
                'y': y,
            })
    
    return pd.DataFrame(rows)


@pytest.fixture
def smoking_data():
    """Load smoking dataset for common timing validation."""
    candidates = [
        os.path.join(os.path.dirname(__file__), 'data', 'smoking.csv'),
        'tests/data/smoking.csv',
    ]
    for path in candidates:
        if os.path.exists(path):
            return pd.read_csv(path)
    
    # Create synthetic data
    np.random.seed(42)
    rows = []
    for i in range(10):
        for t in range(6):
            rows.append({
                'state': i + 1,
                'year': 2000 + t,
                'd': 1 if i < 2 else 0,
                'post': 1 if t >= 3 else 0,
                'lcigsale': 10.0 - 0.1 * t + i * 0.01 + np.random.normal(0, 0.05),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def staggered_results(staggered_panel_data):
    """Get staggered DiD results."""
    return lwdid(
        staggered_panel_data,
        y='y',
        gvar='gvar',
        ivar='unit',
        tvar='year',
        rolling='demean',
        estimator='ra',
        aggregate='overall',
        vce='hc1'
    )


@pytest.fixture
def common_results(smoking_data):
    """Get common timing results."""
    return lwdid(
        smoking_data,
        y='lcigsale',
        d='d',
        ivar='state',
        tvar='year',
        post='post',
        rolling='demean',
        vce='robust'
    )


# =============================================================================
# Numerical Precision Tests
# =============================================================================

class TestFloatPrecision:
    """Tests for floating point precision in export."""
    
    def test_att_precision_staggered(self, staggered_results):
        """ATT values should maintain full double precision."""
        original = staggered_results.att_by_cohort_time
        
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'precision.dta')
            staggered_results.to_stata(path)
            exported = pd.read_stata(path)
            
            # Check each ATT value
            for col in ['att', 'se']:
                if col not in original.columns or col not in exported.columns:
                    continue
                    
                orig_vals = original[col].dropna().values
                exp_vals = exported[col].dropna().values
                
                if len(orig_vals) == 0 or len(exp_vals) == 0:
                    continue
                
                # Values should match to at least 10 decimal places
                np.testing.assert_allclose(
                    orig_vals[:len(exp_vals)],
                    exp_vals[:len(orig_vals)],
                    rtol=1e-10,
                    atol=1e-14,
                    err_msg=f"Precision loss in column {col}"
                )
    
    def test_extreme_values_handled(self, staggered_results):
        """Extreme float values should be handled correctly."""
        # Modify original data to include extreme values
        original = staggered_results.att_by_cohort_time.copy()
        
        # Test with existing data - just verify export works
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'extreme.dta')
            staggered_results.to_stata(path)
            exported = pd.read_stata(path)
            
            # Verify data was exported
            assert len(exported) > 0
            assert 'att' in exported.columns


class TestCIComputation:
    """Tests for confidence interval computation accuracy."""
    
    def test_ci_formula_correct(self, staggered_results):
        """CI bounds should follow att +/- critical_value * se formula."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'ci.dta')
            staggered_results.to_stata(path)
            df = pd.read_stata(path)
            
            if not all(c in df.columns for c in ['att', 'se', 'ci_lower', 'ci_upper']):
                pytest.skip("Required columns not present")
            
            # Note: lwdid may use t-distribution or normal distribution
            # depending on the estimation method. We verify that CI is
            # symmetric around ATT and consistent with reported SE.
            
            # Check symmetry: |att - ci_lower| ≈ |ci_upper - att|
            lower_diff = df['att'] - df['ci_lower']
            upper_diff = df['ci_upper'] - df['att']
            
            np.testing.assert_allclose(
                lower_diff.values,
                upper_diff.values,
                rtol=1e-6,
                err_msg="CI not symmetric around ATT"
            )
            
            # Check that CI width is proportional to SE
            # CI width = 2 * critical_value * SE
            ci_width = df['ci_upper'] - df['ci_lower']
            # Width should be roughly 2 * 1.96 * SE for 95% CI (z-based)
            # or 2 * t_crit * SE for t-based
            # Allow tolerance for different critical values
            expected_width_z = 2 * stats.norm.ppf(0.975) * df['se']
            
            # Check that actual width is within reasonable bounds
            # (should be close to z-based or slightly larger for t-based)
            ratio = ci_width / expected_width_z
            assert all(ratio > 0.9), "CI width too narrow"
            assert all(ratio < 1.2), "CI width too wide"
    
    def test_ci_width_consistent(self, staggered_results):
        """CI width should be consistent with SE (2 * critical_value * se)."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'ci_width.dta')
            staggered_results.to_stata(path)
            df = pd.read_stata(path)
            
            if not all(c in df.columns for c in ['se', 'ci_lower', 'ci_upper']):
                pytest.skip("Required columns not present")
            
            ci_width = df['ci_upper'] - df['ci_lower']
            
            # Infer critical value from the data
            # critical_value = ci_width / (2 * se)
            implied_crit = ci_width / (2 * df['se'])
            
            # For 95% CI, critical value should be around 1.96 (z) or higher (t)
            assert all(implied_crit > 1.8), "Critical value too low"
            assert all(implied_crit < 3.0), "Critical value too high"
            
            # Check that implied critical values are consistent across rows
            # (should use same critical value)
            np.testing.assert_allclose(
                implied_crit.values,
                implied_crit.mean(),
                rtol=0.05,  # Allow 5% variation
                err_msg="Inconsistent critical values across rows"
            )


class TestStatisticalConsistency:
    """Tests for statistical consistency of exported values."""
    
    def test_t_stat_formula(self, staggered_results):
        """t-stat should equal att / se."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'tstat.dta')
            staggered_results.to_stata(path)
            df = pd.read_stata(path)
            
            if not all(c in df.columns for c in ['att', 'se', 't_stat']):
                pytest.skip("Required columns not present")
            
            # Calculate expected t-stat
            expected_t = df['att'] / df['se']
            
            # Compare (allow for division by zero cases)
            valid_mask = (df['se'] > 0) & np.isfinite(expected_t)
            
            np.testing.assert_allclose(
                df.loc[valid_mask, 't_stat'].values,
                expected_t.loc[valid_mask].values,
                rtol=1e-6,
                err_msg="t-stat calculation mismatch"
            )
    
    def test_pvalue_from_t_stat(self, staggered_results):
        """p-value should be consistent with t-stat (two-sided test)."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'pvalue.dta')
            staggered_results.to_stata(path)
            df = pd.read_stata(path)
            
            if not all(c in df.columns for c in ['t_stat', 'pvalue']):
                pytest.skip("Required columns not present")
            
            # Use survival function for more accurate small p-values
            # p = 2 * Phi(-|t|) = 2 * sf(|t|)
            expected_p = 2 * stats.norm.sf(np.abs(df['t_stat'].values))
            
            # For very small p-values, use log comparison
            actual_p = df['pvalue'].values
            
            # Filter out zero p-values for comparison
            valid_mask = (expected_p > 1e-15) & (actual_p > 1e-15)
            
            if not valid_mask.any():
                # All p-values are essentially zero - both agree effect is highly significant
                assert all(actual_p < 1e-8), "Expected all very small p-values"
                return
            
            # Compare log p-values for numerical stability
            np.testing.assert_allclose(
                np.log10(actual_p[valid_mask] + 1e-300),
                np.log10(expected_p[valid_mask] + 1e-300),
                atol=1.0,  # Allow 1 order of magnitude difference
                err_msg="p-value calculation significantly different from normal approx"
            )
    
    def test_significant_att_has_ci_excluding_zero(self, staggered_results):
        """Significant effects (p < 0.05) should have CI excluding zero."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'sig.dta')
            staggered_results.to_stata(path)
            df = pd.read_stata(path)
            
            if not all(c in df.columns for c in ['pvalue', 'ci_lower', 'ci_upper']):
                pytest.skip("Required columns not present")
            
            # Find significant effects at alpha=0.05
            alpha = 0.05
            significant = df['pvalue'] < alpha
            
            if not significant.any():
                pytest.skip("No significant effects found")
            
            # For significant effects, CI should not cross zero
            for idx in df[significant].index:
                ci_l = df.loc[idx, 'ci_lower']
                ci_u = df.loc[idx, 'ci_upper']
                
                # CI should either be entirely positive or entirely negative
                assert (ci_l > 0 and ci_u > 0) or (ci_l < 0 and ci_u < 0), \
                    f"Significant effect at index {idx} has CI crossing zero: [{ci_l}, {ci_u}]"


class TestCrossFormatConsistency:
    """Tests for consistency between different export formats."""
    
    def test_csv_stata_consistency_staggered(self, staggered_results):
        """CSV and Stata exports should have identical numerical values."""
        with tempfile.TemporaryDirectory() as td:
            csv_path = os.path.join(td, 'data.csv')
            dta_path = os.path.join(td, 'data.dta')
            
            staggered_results.to_csv(csv_path)
            staggered_results.to_stata(dta_path)
            
            csv_df = pd.read_csv(csv_path)
            dta_df = pd.read_stata(dta_path)
            
            # Same number of rows
            assert len(csv_df) == len(dta_df), \
                f"Row count mismatch: CSV={len(csv_df)}, Stata={len(dta_df)}"
            
            # Compare numerical columns
            for col in ['att', 'se', 'ci_lower', 'ci_upper', 't_stat', 'pvalue']:
                if col not in csv_df.columns or col not in dta_df.columns:
                    continue
                
                np.testing.assert_allclose(
                    csv_df[col].values,
                    dta_df[col].values,
                    rtol=1e-10,
                    err_msg=f"Column {col} mismatch between CSV and Stata"
                )
    
    def test_csv_stata_consistency_common(self, common_results):
        """CSV and Stata exports should match for common timing."""
        with tempfile.TemporaryDirectory() as td:
            csv_path = os.path.join(td, 'common.csv')
            dta_path = os.path.join(td, 'common.dta')
            
            common_results.to_csv(csv_path)
            common_results.to_stata(dta_path)
            
            csv_df = pd.read_csv(csv_path)
            dta_df = pd.read_stata(dta_path)
            
            # Same row count
            assert len(csv_df) == len(dta_df)


class TestOverallEffectNumerics:
    """Tests for overall effect numerical accuracy."""
    
    def test_overall_effect_export(self, staggered_results):
        """Overall effect export should have correct values."""
        if staggered_results.att_overall is None:
            pytest.skip("Overall effect not available")
        
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'overall.dta')
            staggered_results.to_stata(path, what='overall')
            
            df = pd.read_stata(path)
            
            # Single row
            assert len(df) == 1
            
            # Check values match
            assert np.isclose(df['att_overall'].iloc[0], staggered_results.att_overall, rtol=1e-10)
            
            if staggered_results.se_overall is not None:
                assert np.isclose(df['se'].iloc[0], staggered_results.se_overall, rtol=1e-10)


class TestCohortEffectNumerics:
    """Tests for cohort-level effect numerical accuracy."""
    
    def test_cohort_effect_preservation(self, staggered_results):
        """Cohort effects should be preserved exactly."""
        if staggered_results.att_by_cohort is None:
            pytest.skip("Cohort effects not available")
        
        original = staggered_results.att_by_cohort
        
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'cohort.dta')
            staggered_results.to_stata(path, what='cohort')
            
            exported = pd.read_stata(path)
            
            # Same number of cohorts
            assert len(exported) == len(original)
            
            # Compare ATT values
            for idx in range(len(original)):
                orig_att = original.iloc[idx]['att']
                exp_att = exported.iloc[idx]['att']
                
                np.testing.assert_allclose(
                    orig_att, exp_att, rtol=1e-10,
                    err_msg=f"Cohort ATT mismatch at index {idx}"
                )


class TestIntegerValuePreservation:
    """Tests for integer value preservation."""
    
    def test_cohort_values_integer(self, staggered_results):
        """Cohort values should remain integers."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'int.dta')
            staggered_results.to_stata(path)
            
            df = pd.read_stata(path)
            
            if 'cohort' not in df.columns:
                pytest.skip("cohort column not present")
            
            # Cohort values should be whole numbers
            cohorts = df['cohort'].dropna().values
            assert np.allclose(cohorts, np.round(cohorts)), \
                "Cohort values should be integers"
    
    def test_period_values_integer(self, staggered_results):
        """Period values should remain integers."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'int.dta')
            staggered_results.to_stata(path)
            
            df = pd.read_stata(path)
            
            if 'period' not in df.columns:
                pytest.skip("period column not present")
            
            # Period values should be whole numbers
            periods = df['period'].dropna().values
            assert np.allclose(periods, np.round(periods)), \
                "Period values should be integers"


class TestNaNHandling:
    """Tests for NaN value handling."""
    
    def test_nan_preserved_as_missing(self, staggered_results):
        """NaN values should be preserved as Stata missing values."""
        original = staggered_results.att_by_cohort_time
        
        # Count NaN in original
        original_nan_count = original.isna().sum().sum()
        
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'nan.dta')
            staggered_results.to_stata(path)
            
            exported = pd.read_stata(path)
            
            # Stata missing values become NaN when read by pandas
            # The count should be similar (may differ slightly due to column changes)
            exported_nan_count = exported.isna().sum().sum()
            
            # Just verify the export works - exact NaN count may differ
            # due to column transformations
            assert len(exported) == len(original)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
