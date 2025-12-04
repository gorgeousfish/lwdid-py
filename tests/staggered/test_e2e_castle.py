"""
End-to-End Tests for Castle Law Data

This is the most critical test suite for verifying implementation correctness.
Uses hardcoded verified values from Florida (sid=10) and California (sid=5).
"""

import time
import pytest
import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from lwdid.staggered.transformations import (
    transform_staggered_demean,
    transform_staggered_detrend,
    get_cohorts,
)


# Data path
CASTLE_DATA_PATH = '/Users/cxy/Desktop/rebuildlwdid/lwdid-py_v0.1.0/data/castle.csv'


class TestCastleLawE2E:
    """Castle Law data end-to-end test class."""
    
    @pytest.fixture
    def castle_data(self):
        """Load and preprocess Castle Law data."""
        castle = pd.read_csv(CASTLE_DATA_PATH)
        castle['gvar'] = castle['effyear'].fillna(0).astype(int)
        return castle
    
    def test_data_structure(self, castle_data):
        """Verify data structure is correct."""
        # Must have these columns
        assert 'sid' in castle_data.columns
        assert 'year' in castle_data.columns
        assert 'lhomicide' in castle_data.columns
        assert 'gvar' in castle_data.columns
        
        # Year range
        assert castle_data['year'].min() == 2000
        assert castle_data['year'].max() == 2010
        
        # Cohort distribution
        cohorts = sorted([g for g in castle_data['gvar'].unique() if g > 0])
        assert 2005 in cohorts
        assert 2006 in cohorts
        
        # 50 states × 11 years = 550 rows
        assert len(castle_data) == 550
    
    def test_transform_columns_generated(self, castle_data):
        """Verify all necessary transformation columns are generated."""
        result = transform_staggered_demean(
            castle_data, 'lhomicide', 'sid', 'year', 'gvar'
        )
        
        cohorts = sorted([g for g in castle_data['gvar'].unique() if g > 0])
        T_max = castle_data['year'].max()  # 2010
        
        expected_cols = []
        for g in cohorts:
            for r in range(g, T_max + 1):
                expected_cols.append(f'ydot_g{g}_r{r}')
        
        for col in expected_cols:
            assert col in result.columns, f"Missing transformation column: {col}"
        
        print(f"✓ Generated {len(expected_cols)} transformation columns")
    
    def test_florida_transformation(self, castle_data):
        """⚠️ CRITICAL: Detailed verification of Florida (only 2005 cohort) values.
        
        Verified exact values:
        - Pre-treatment mean: 1.7212106
        - ydot_g2005_r2005: -0.0959351
        - ydot_g2005_r2006:  0.1099387
        - ydot_g2005_r2007:  0.1767458
        - ydot_g2005_r2008:  0.1396586
        - ydot_g2005_r2009: -0.0018359
        - ydot_g2005_r2010: -0.0562806
        """
        result = transform_staggered_demean(
            castle_data, 'lhomicide', 'sid', 'year', 'gvar'
        )
        
        florida = result[result['sid'] == 10].copy()
        assert florida['gvar'].iloc[0] == 2005, "Florida should be 2005 cohort"
        
        # Verify pre-treatment mean
        expected_pre_mean = 1.7212106
        pre_y = florida[florida['year'] < 2005]['lhomicide'].mean()
        assert np.isclose(pre_y, expected_pre_mean, atol=1e-6), \
            f"Florida pre_mean incorrect: expected={expected_pre_mean}, actual={pre_y}"
        
        # Verify each post period transformation (hardcoded verified values)
        expected_ydots = {
            2005: -0.0959351,
            2006:  0.1099387,
            2007:  0.1767458,
            2008:  0.1396586,
            2009: -0.0018359,
            2010: -0.0562806,
        }
        
        for r, expected_ydot in expected_ydots.items():
            col = f'ydot_g2005_r{r}'
            actual = florida[florida['year'] == r][col].iloc[0]
            
            assert np.isclose(actual, expected_ydot, atol=1e-6), \
                f"Florida ydot_g2005_r{r} error: expected={expected_ydot}, actual={actual}"
        
        print(f"✓ Florida pre-treatment mean = {pre_y:.7f}")
        print("✓ All Florida post-period transformations verified")
    
    def test_pre_treatment_mean_fixed(self, castle_data):
        """⚠️ CRITICAL: Verify pre-treatment mean is fixed across all post periods.
        
        For cohort g=2005 (Florida), the pre-treatment mean must be the same
        value (2000-2004 mean) regardless of calendar time r.
        """
        result = transform_staggered_demean(
            castle_data, 'lhomicide', 'sid', 'year', 'gvar'
        )
        
        florida = result[result['sid'] == 10]
        expected_pre_mean = florida[florida['year'] < 2005]['lhomicide'].mean()
        
        pre_means_from_transform = []
        for r in [2005, 2006, 2007, 2008, 2009, 2010]:
            col = f'ydot_g2005_r{r}'
            y_r = florida[florida['year'] == r]['lhomicide'].iloc[0]
            ydot_r = florida[florida['year'] == r][col].iloc[0]
            pre_mean_r = y_r - ydot_r
            pre_means_from_transform.append(pre_mean_r)
            
            assert np.isclose(pre_mean_r, expected_pre_mean, atol=1e-10), \
                f"Pre-mean not fixed! r={r}: {pre_mean_r} != {expected_pre_mean}"
        
        # All reversed pre_means must be identical
        assert np.allclose(pre_means_from_transform, expected_pre_mean, atol=1e-10)
        print("✓ Pre-treatment mean fixedness verified (precision: 1e-10)")
    
    def test_never_treated_also_transformed(self, castle_data):
        """Verify never treated states also get transformation values.
        
        Using California (sid=5) as verification example:
        - pre_mean_g2005 = 1.9041211
        - ydot_g2005_r2005 = 0.0552577
        """
        result = transform_staggered_demean(
            castle_data, 'lhomicide', 'sid', 'year', 'gvar'
        )
        
        # California (sid=5) as never treated verification
        california = result[result['sid'] == 5]
        assert california['gvar'].iloc[0] == 0, "California should be never treated (gvar=0)"
        
        # California should have transformation value for cohort 2005 at year 2005
        assert 'ydot_g2005_r2005' in result.columns
        assert not california[california['year'] == 2005]['ydot_g2005_r2005'].isna().all(), \
            "California should have ydot_g2005_r2005 value"
        
        # Verify California transformation calculation (hardcoded verified values)
        expected_pre_mean = 1.9041211
        expected_ydot = 0.0552577
        
        pre_y_ca = california[california['year'] < 2005]['lhomicide'].mean()
        assert np.isclose(pre_y_ca, expected_pre_mean, atol=1e-6), \
            f"California pre_mean error: expected={expected_pre_mean}, actual={pre_y_ca}"
        
        actual_ydot = california[california['year'] == 2005]['ydot_g2005_r2005'].iloc[0]
        assert np.isclose(actual_ydot, expected_ydot, atol=1e-6), \
            f"California ydot_g2005_r2005 error: expected={expected_ydot}, actual={actual_ydot}"
        
        print(f"✓ California (sid=5, never treated) transformation verified")
    
    def test_multiple_cohorts_cross_validation(self, castle_data):
        """Verify cross-cohort calculation correctness.
        
        Key: For cohort g=2006, all units (including 2005 cohort)
        should have ydot_g2006_* values.
        """
        result = transform_staggered_demean(
            castle_data, 'lhomicide', 'sid', 'year', 'gvar'
        )
        
        # Florida (2005 cohort) should also have cohort 2006 transformation
        florida = result[result['sid'] == 10]
        
        # ydot_g2006_r2006: uses 2005 and earlier (2000-2005) mean
        pre_2006_mean = florida[florida['year'] < 2006]['lhomicide'].mean()
        y_2006 = florida[florida['year'] == 2006]['lhomicide'].iloc[0]
        expected = y_2006 - pre_2006_mean
        actual = florida[florida['year'] == 2006]['ydot_g2006_r2006'].iloc[0]
        
        assert np.isclose(actual, expected, atol=1e-10), \
            f"Cross-cohort transformation error: expected={expected}, actual={actual}"
        
        print("✓ Multiple cohort cross-validation passed")
    
    def test_all_cohorts_have_transforms(self, castle_data):
        """Verify all cohorts have correct transformation columns."""
        result = transform_staggered_demean(
            castle_data, 'lhomicide', 'sid', 'year', 'gvar'
        )
        
        # Castle Law cohorts: 2005, 2006, 2007, 2008, 2009
        expected_cohorts = [2005, 2006, 2007, 2008, 2009]
        T_max = 2010
        
        for g in expected_cohorts:
            for r in range(g, T_max + 1):
                col_name = f'ydot_g{g}_r{r}'
                assert col_name in result.columns, \
                    f"Missing transformation column: {col_name}"
                # Verify column has values at corresponding period
                period_data = result[result['year'] == r]
                assert not period_data[col_name].isna().all(), \
                    f"{col_name} all NaN at year={r}"
        
        print("✓ All cohort transformation columns verified")
    
    def test_cohort_sizes(self, castle_data):
        """Verify cohort sizes match expected values."""
        # Count states per cohort
        cohort_counts = castle_data.groupby('gvar')['sid'].nunique()
        
        # Expected values (based on paper data)
        expected = {
            2005: 1,   # Florida
            2006: 13,
            2007: 4,
            2008: 2,
            2009: 1,
            0: 29      # Never treated (gvar=0 after fillna)
        }
        
        for cohort, expected_count in expected.items():
            if cohort in cohort_counts.index:
                actual_count = cohort_counts[cohort]
                assert actual_count == expected_count, \
                    f"Cohort {cohort}: expected {expected_count}, got {actual_count}"
        
        print("✓ Cohort sizes verified")
    
    def test_performance(self, castle_data):
        """Verify performance requirements (<0.5 seconds)."""
        start = time.time()
        result = transform_staggered_demean(
            castle_data, 'lhomicide', 'sid', 'year', 'gvar'
        )
        elapsed = time.time() - start
        
        assert elapsed < 0.5, f"Transformation took {elapsed:.3f}s, exceeds 0.5s limit"
        print(f"✓ Performance test passed: {elapsed:.3f}s")
    
    def test_transform_column_count(self, castle_data):
        """Verify transformation column count matches formula.
        
        Total columns = sum(T_max - g + 1) for each cohort g
        """
        result = transform_staggered_demean(
            castle_data, 'lhomicide', 'sid', 'year', 'gvar'
        )
        
        # Castle Law: cohorts={2005,2006,2007,2008,2009}, T_max=2010
        cohorts = [2005, 2006, 2007, 2008, 2009]
        T_max = 2010
        
        expected_count = sum(T_max - g + 1 for g in cohorts)
        # = (2010-2005+1) + (2010-2006+1) + (2010-2007+1) + (2010-2008+1) + (2010-2009+1)
        # = 6 + 5 + 4 + 3 + 2 = 20
        
        ydot_columns = [c for c in result.columns if c.startswith('ydot_')]
        actual_count = len(ydot_columns)
        
        assert actual_count == expected_count, \
            f"Expected {expected_count} ydot columns, got {actual_count}"
        
        print(f"✓ Column count verified: {actual_count} columns")


def test_e2e_castle_law_full():
    """Complete end-to-end test entry point."""
    castle = pd.read_csv(CASTLE_DATA_PATH)
    castle['gvar'] = castle['effyear'].fillna(0).astype(int)
    
    test_suite = TestCastleLawE2E()
    
    # P0 - Must-pass core tests
    test_suite.test_data_structure(castle)
    test_suite.test_transform_columns_generated(castle)
    test_suite.test_florida_transformation(castle)
    test_suite.test_pre_treatment_mean_fixed(castle)  # Critical!
    test_suite.test_never_treated_also_transformed(castle)
    test_suite.test_multiple_cohorts_cross_validation(castle)
    test_suite.test_all_cohorts_have_transforms(castle)
    test_suite.test_cohort_sizes(castle)
    test_suite.test_performance(castle)
    test_suite.test_transform_column_count(castle)
    
    print("\n" + "="*50)
    print("Castle Law E2E tests all passed!")
    print("="*50)


if __name__ == '__main__':
    test_e2e_castle_law_full()
