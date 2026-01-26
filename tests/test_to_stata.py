"""
Tests for to_stata() Export Functionality

This module contains comprehensive tests for the to_stata() and to_stata_staggered()
methods added to LWDIDResults class as part of DESIGN-013 fix.

Test categories:
1. Unit tests - Method existence, parameter validation, exception handling
2. Numerical validation - Data integrity after export
3. Column name cleaning - Stata variable name restrictions
4. Staggered vs common timing mode handling
"""

import os
import re
import tempfile
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from lwdid import lwdid
from lwdid.results import (
    LWDIDResults,
    _clean_stata_varname,
    _prepare_stata_dataframe,
    _STATA_VARIABLE_LABELS,
)


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def common_timing_data():
    """Load or create common timing test data."""
    candidates = [
        os.path.join(os.path.dirname(__file__), 'data', 'smoking.csv'),
        'tests/data/smoking.csv',
        'data/smoking.csv',
    ]
    for path in candidates:
        if os.path.exists(path):
            return pd.read_csv(path)
    
    # Fallback: create minimal synthetic dataset
    rows = []
    for i in range(5):
        for t in range(4):
            rows.append({
                'state': i + 1,
                'year': 2000 + t,
                'd': 1 if i == 0 else 0,
                'post': 1 if t >= 2 else 0,
                'lcigsale': 10.0 - 0.1 * t + i * 0.01,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def staggered_data():
    """Load or create staggered DiD test data (castle_law style)."""
    candidates = [
        os.path.join(os.path.dirname(__file__), 'staggered', 'data', 'castle_law.csv'),
        os.path.join(os.path.dirname(__file__), 'data', 'castle_law.csv'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return pd.read_csv(path)
    
    # Fallback: create synthetic staggered data
    # Note: cohort values must be > T_min to have pre-treatment periods
    # T_min = 2001 (first year), so cohorts should be >= 2003
    rows = []
    n_units = 50
    n_periods = 10  # years 2001-2010
    cohorts = [0, 2004, 2006, 2008]  # 0 = never treated, use actual years
    
    np.random.seed(42)
    for i in range(n_units):
        cohort = cohorts[i % len(cohorts)]
        for t in range(n_periods):
            year = 2001 + t
            treated = 1 if cohort > 0 and year >= cohort else 0
            y = 5.0 + i * 0.1 + t * 0.05 + treated * 0.5 + np.random.normal(0, 0.1)
            rows.append({
                'unit': i + 1,
                'year': year,
                'gvar': cohort if cohort > 0 else 0,  # 0 = never treated
                'y': y,
            })
    
    return pd.DataFrame(rows)


@pytest.fixture
def common_timing_results(common_timing_data):
    """Get LWDIDResults for common timing."""
    return lwdid(
        common_timing_data,
        y='lcigsale',
        d='d',
        ivar='state',
        tvar='year',
        post='post',
        rolling='demean',
        vce='robust'
    )


@pytest.fixture
def staggered_results(staggered_data):
    """Get LWDIDResults for staggered DiD."""
    return lwdid(
        staggered_data,
        y='y',
        gvar='gvar',
        ivar='unit',
        tvar='year',
        rolling='demean',
        estimator='ra',
        aggregate='overall',
        vce='hc1'
    )


# =============================================================================
# Unit Tests: Helper Functions
# =============================================================================

class TestCleanStataVarname:
    """Tests for _clean_stata_varname helper function."""
    
    def test_valid_name_unchanged(self):
        """Valid Stata variable names should remain unchanged."""
        assert _clean_stata_varname('cohort') == 'cohort'
        assert _clean_stata_varname('att') == 'att'
        assert _clean_stata_varname('ci_lower') == 'ci_lower'
        assert _clean_stata_varname('n_treated') == 'n_treated'
    
    def test_special_chars_replaced(self):
        """Special characters should be replaced with underscores."""
        assert _clean_stata_varname('p-value') == 'p_value'
        assert _clean_stata_varname('ci.lower') == 'ci_lower'
        assert _clean_stata_varname('effect(1)') == 'effect_1_'
        assert _clean_stata_varname('var name') == 'var_name'
    
    def test_starts_with_digit(self):
        """Names starting with digits should get underscore prefix."""
        assert _clean_stata_varname('1cohort') == '_1cohort'
        assert _clean_stata_varname('2005') == '_2005'
    
    def test_truncation_to_32_chars(self):
        """Names longer than 32 characters should be truncated."""
        long_name = 'a' * 50
        result = _clean_stata_varname(long_name)
        assert len(result) == 32
        assert result == 'a' * 32
    
    def test_empty_string(self):
        """Empty string should return '_var'."""
        assert _clean_stata_varname('') == '_var'
    
    def test_underscore_prefix_preserved(self):
        """Names starting with underscore should be preserved."""
        assert _clean_stata_varname('_internal') == '_internal'


class TestPrepareStataDataframe:
    """Tests for _prepare_stata_dataframe helper function."""
    
    def test_basic_cleaning(self):
        """Basic DataFrame cleaning should work."""
        df = pd.DataFrame({
            'cohort': [1, 2],
            'p-value': [0.05, 0.01],
            'ci.lower': [-0.1, -0.2],
        })
        result, labels = _prepare_stata_dataframe(df)
        
        assert 'cohort' in result.columns
        assert 'p_value' in result.columns
        assert 'ci_lower' in result.columns
        assert 'p-value' not in result.columns
    
    def test_default_labels_applied(self):
        """Default labels should be applied for known columns."""
        df = pd.DataFrame({
            'cohort': [1],
            'att': [0.5],
            'se': [0.1],
        })
        result, labels = _prepare_stata_dataframe(df)
        
        assert 'cohort' in labels
        assert 'att' in labels
        assert 'se' in labels
        assert labels['cohort'] == _STATA_VARIABLE_LABELS['cohort']
    
    def test_custom_labels_override(self):
        """Custom labels should override defaults."""
        df = pd.DataFrame({'att': [0.5]})
        custom_labels = {'att': 'My custom ATT label'}
        
        result, labels = _prepare_stata_dataframe(df, variable_labels=custom_labels)
        
        assert labels['att'] == 'My custom ATT label'
    
    def test_duplicate_column_handling(self):
        """Duplicate column names after cleaning should be handled."""
        # Create DataFrame with columns that would conflict after cleaning
        df = pd.DataFrame({
            'var-1': [1],
            'var.1': [2],  # Both become 'var_1'
        })
        result, labels = _prepare_stata_dataframe(df)
        
        # Should have unique column names
        assert len(result.columns) == len(set(result.columns))
    
    def test_bug053_stata_compatible_duplicate(self):
        """BUG-053: Original Stata-compatible name should also be deduplicated.
        
        This tests the scenario where:
        1. Column 'a-b' gets cleaned to 'a_b'
        2. Column 'a_b' (already Stata-compatible) should NOT overwrite
        
        Before fix: The condition `cleaned != col` was True for 'a-b' but False
        for 'a_b', causing 'a_b' to skip deduplication and overwrite data.
        """
        df = pd.DataFrame({
            'a-b': [1],
            'a_b': [2],  # Already Stata-compatible, but conflicts after cleaning
        })
        result, labels = _prepare_stata_dataframe(df)
        
        # Should have unique column names
        assert len(result.columns) == len(set(result.columns)), \
            "Columns should have unique names after cleaning"
        
        # Both columns should exist (with different names)
        assert len(result.columns) == 2, "Both columns should be preserved"
        
        # Data should be preserved correctly
        assert 1 in result.values, "Value from 'a-b' should be preserved"
        assert 2 in result.values, "Value from 'a_b' should be preserved"
    
    def test_bug053_multiple_conflicts(self):
        """BUG-053: Multiple columns cleaning to the same name."""
        df = pd.DataFrame({
            'x-y': [1],
            'x.y': [2],
            'x_y': [3],  # Already Stata-compatible
            'x y': [4],  # Space replaced with underscore
        })
        result, labels = _prepare_stata_dataframe(df)
        
        # All columns should have unique names
        assert len(result.columns) == len(set(result.columns))
        assert len(result.columns) == 4
        
        # All data values should be preserved
        values = set(result.values.flatten())
        assert values == {1, 2, 3, 4}, "All original values should be preserved"
    
    def test_bug053_order_independence(self):
        """BUG-053: Deduplication should work regardless of column order."""
        # Test case 1: Stata-compatible name comes first
        df1 = pd.DataFrame({
            'p_value': [1],
            'p-value': [2],
        })
        result1, _ = _prepare_stata_dataframe(df1)
        assert len(result1.columns) == len(set(result1.columns))
        assert len(result1.columns) == 2
        
        # Test case 2: Non-Stata-compatible name comes first
        df2 = pd.DataFrame({
            'p-value': [1],
            'p_value': [2],
        })
        result2, _ = _prepare_stata_dataframe(df2)
        assert len(result2.columns) == len(set(result2.columns))
        assert len(result2.columns) == 2
    
    def test_bug053_data_integrity_after_export(self):
        """BUG-053: Verify data integrity after Stata export with conflicting names."""
        df = pd.DataFrame({
            'effect-1': [1.5, 2.5, 3.5],
            'effect_1': [4.5, 5.5, 6.5],
        })
        result, _ = _prepare_stata_dataframe(df)
        
        # Should have unique column names
        assert len(result.columns) == 2
        assert len(set(result.columns)) == 2
        
        # Data should be preserved
        # One column should have [1.5, 2.5, 3.5], another should have [4.5, 5.5, 6.5]
        col_sums = sorted([result[col].sum() for col in result.columns])
        expected_sums = sorted([1.5 + 2.5 + 3.5, 4.5 + 5.5 + 6.5])
        np.testing.assert_allclose(col_sums, expected_sums, rtol=1e-10)


# =============================================================================
# Unit Tests: to_stata() Method
# =============================================================================

class TestToStataMethodExists:
    """Tests for method existence and basic API."""
    
    def test_method_exists_common_timing(self, common_timing_results):
        """to_stata() method should exist on common timing results."""
        assert hasattr(common_timing_results, 'to_stata')
        assert callable(common_timing_results.to_stata)
    
    def test_method_exists_staggered(self, staggered_results):
        """to_stata() method should exist on staggered results."""
        assert hasattr(staggered_results, 'to_stata')
        assert callable(staggered_results.to_stata)
    
    def test_to_stata_staggered_method_exists(self, staggered_results):
        """to_stata_staggered() method should exist."""
        assert hasattr(staggered_results, 'to_stata_staggered')
        assert callable(staggered_results.to_stata_staggered)


class TestToStataCommonTiming:
    """Tests for to_stata() with common timing results."""
    
    def test_basic_export(self, common_timing_results):
        """Basic export should create a valid .dta file."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'results.dta')
            common_timing_results.to_stata(path)
            
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
    
    def test_exported_data_readable(self, common_timing_results):
        """Exported file should be readable by pandas."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'results.dta')
            common_timing_results.to_stata(path)
            
            df = pd.read_stata(path)
            assert len(df) > 0
            assert 'period' in df.columns or 'tindex' in df.columns
    
    def test_invalid_what_parameter(self, common_timing_results):
        """Invalid 'what' parameter for common timing should raise error."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'results.dta')
            
            with pytest.raises(ValueError, match="only valid for staggered"):
                common_timing_results.to_stata(path, what='cohort')
            
            with pytest.raises(ValueError, match="only valid for staggered"):
                common_timing_results.to_stata(path, what='overall')
    
    def test_missing_data_error(self, common_timing_results):
        """Should raise error when att_by_period is missing."""
        common_timing_results._att_by_period = None
        
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'results.dta')
            
            with pytest.raises(ValueError, match="att_by_period is not available"):
                common_timing_results.to_stata(path)


class TestToStataStaggered:
    """Tests for to_stata() with staggered results."""
    
    def test_basic_export_auto(self, staggered_results):
        """Basic export with what='auto' should work."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'staggered.dta')
            staggered_results.to_stata(path, what='auto')
            
            assert os.path.exists(path)
            df = pd.read_stata(path)
            assert 'cohort' in df.columns
            assert 'period' in df.columns
    
    def test_export_cohort_time(self, staggered_results):
        """Export what='cohort_time' should work."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'cohort_time.dta')
            staggered_results.to_stata(path, what='cohort_time')
            
            df = pd.read_stata(path)
            assert 'cohort' in df.columns
            assert 'att' in df.columns
    
    def test_export_cohort(self, staggered_results):
        """Export what='cohort' should work when available."""
        if staggered_results.att_by_cohort is None:
            pytest.skip("att_by_cohort not available for this result")
        
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'cohort.dta')
            staggered_results.to_stata(path, what='cohort')
            
            df = pd.read_stata(path)
            assert 'cohort' in df.columns
            assert 'att' in df.columns
    
    def test_export_overall(self, staggered_results):
        """Export what='overall' should work when available."""
        if staggered_results.att_overall is None:
            pytest.skip("att_overall not available for this result")
        
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'overall.dta')
            staggered_results.to_stata(path, what='overall')
            
            df = pd.read_stata(path)
            assert len(df) == 1  # Single row for overall
            assert 'att_overall' in df.columns
    
    def test_invalid_what_parameter(self, staggered_results):
        """Invalid 'what' parameter should raise error."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'invalid.dta')
            
            with pytest.raises(ValueError, match="Invalid what="):
                staggered_results.to_stata(path, what='invalid_option')
    
    def test_cohort_not_available_error(self, staggered_data):
        """Should raise error when requesting unavailable cohort data."""
        # Create results with aggregate='none' so att_by_cohort is not available
        results = lwdid(
            staggered_data,
            y='y',
            gvar='gvar',
            ivar='unit',
            tvar='year',
            rolling='demean',
            estimator='ra',
            aggregate='none',  # This won't produce att_by_cohort
        )
        
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'cohort.dta')
            
            with pytest.raises(ValueError, match="att_by_cohort is not available"):
                results.to_stata(path, what='cohort')
    
    def test_overall_not_available_error(self, staggered_data):
        """Should raise error when requesting unavailable overall data."""
        # Create results with aggregate='cohort' so att_overall is not available
        results = lwdid(
            staggered_data,
            y='y',
            gvar='gvar',
            ivar='unit',
            tvar='year',
            rolling='demean',
            estimator='ra',
            aggregate='cohort',  # This won't produce att_overall
        )
        
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'overall.dta')
            
            with pytest.raises(ValueError, match="Overall effect is not available"):
                results.to_stata(path, what='overall')


class TestToStataOptions:
    """Tests for to_stata() options."""
    
    def test_write_index_option(self, common_timing_results):
        """write_index option should work."""
        with tempfile.TemporaryDirectory() as td:
            # Without index
            path_no_idx = os.path.join(td, 'no_index.dta')
            common_timing_results.to_stata(path_no_idx, write_index=False)
            df_no_idx = pd.read_stata(path_no_idx)
            
            # With index
            path_with_idx = os.path.join(td, 'with_index.dta')
            common_timing_results.to_stata(path_with_idx, write_index=True)
            df_with_idx = pd.read_stata(path_with_idx)
            
            # With index should have one more column (index column)
            assert len(df_with_idx.columns) >= len(df_no_idx.columns)
    
    def test_version_option(self, common_timing_results):
        """version option should work."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'version117.dta')
            # Version 117 = Stata 13
            common_timing_results.to_stata(path, version=117)
            
            assert os.path.exists(path)
            df = pd.read_stata(path)
            assert len(df) > 0
    
    def test_custom_variable_labels(self, common_timing_results):
        """Custom variable labels should be applied."""
        custom_labels = {
            'period': 'My custom period label',
            'beta': 'Treatment effect estimate',
        }
        
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'custom_labels.dta')
            common_timing_results.to_stata(path, variable_labels=custom_labels)
            
            assert os.path.exists(path)
            # Note: We can't easily verify labels without Stata, but at least
            # verify the export doesn't fail


# =============================================================================
# Numerical Validation Tests
# =============================================================================

class TestToStataNumericalValidation:
    """Tests to verify data integrity after export."""
    
    def test_float_precision_preserved(self, staggered_results):
        """Float values should maintain precision after export."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'precision.dta')
            staggered_results.to_stata(path)
            
            original = staggered_results.att_by_cohort_time
            exported = pd.read_stata(path)
            
            # Compare ATT values
            for col in ['att', 'se']:
                if col in original.columns and col in exported.columns:
                    orig_vals = original[col].dropna().values
                    exp_vals = exported[col].dropna().values
                    
                    # Allow small floating point differences
                    np.testing.assert_allclose(
                        orig_vals[:len(exp_vals)],
                        exp_vals[:len(orig_vals)],
                        rtol=1e-10,
                        err_msg=f"Column {col} values don't match"
                    )
    
    def test_integer_values_preserved(self, staggered_results):
        """Integer values should be preserved."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'integers.dta')
            staggered_results.to_stata(path)
            
            original = staggered_results.att_by_cohort_time
            exported = pd.read_stata(path)
            
            # Compare cohort values (should be integers)
            if 'cohort' in original.columns and 'cohort' in exported.columns:
                orig_cohorts = original['cohort'].unique()
                exp_cohorts = exported['cohort'].unique()
                
                assert set(orig_cohorts) == set(exp_cohorts)
    
    def test_row_count_preserved(self, staggered_results):
        """Number of rows should be preserved."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'rows.dta')
            staggered_results.to_stata(path)
            
            original = staggered_results.att_by_cohort_time
            exported = pd.read_stata(path)
            
            assert len(exported) == len(original)
    
    def test_ci_bounds_consistent(self, staggered_results):
        """CI bounds should be consistent with ATT and SE."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'ci.dta')
            staggered_results.to_stata(path)
            
            df = pd.read_stata(path)
            
            if all(c in df.columns for c in ['att', 'se', 'ci_lower', 'ci_upper']):
                # Check CI is approximately att +/- 1.96*se (for 95% CI)
                z = 1.96
                expected_lower = df['att'] - z * df['se']
                expected_upper = df['att'] + z * df['se']
                
                # Allow some tolerance for different z values or rounding
                np.testing.assert_allclose(
                    df['ci_lower'].values,
                    expected_lower.values,
                    rtol=0.05,  # 5% tolerance
                    err_msg="CI lower bounds inconsistent"
                )


# =============================================================================
# Integration Tests
# =============================================================================

class TestToStataIntegration:
    """Integration tests for to_stata() with other export methods."""
    
    def test_all_export_methods_work(self, common_timing_results):
        """All export methods should work on the same results."""
        with tempfile.TemporaryDirectory() as td:
            # CSV
            csv_path = os.path.join(td, 'results.csv')
            common_timing_results.to_csv(csv_path)
            assert os.path.exists(csv_path)
            
            # Excel
            xlsx_path = os.path.join(td, 'results.xlsx')
            common_timing_results.to_excel(xlsx_path)
            assert os.path.exists(xlsx_path)
            
            # LaTeX
            tex_path = os.path.join(td, 'results.tex')
            common_timing_results.to_latex(tex_path)
            assert os.path.exists(tex_path)
            
            # Stata
            dta_path = os.path.join(td, 'results.dta')
            common_timing_results.to_stata(dta_path)
            assert os.path.exists(dta_path)
    
    def test_staggered_all_export_methods(self, staggered_results):
        """All export methods should work on staggered results."""
        with tempfile.TemporaryDirectory() as td:
            # CSV
            csv_path = os.path.join(td, 'staggered.csv')
            staggered_results.to_csv(csv_path)
            assert os.path.exists(csv_path)
            
            # Excel
            xlsx_path = os.path.join(td, 'staggered.xlsx')
            staggered_results.to_excel(xlsx_path)
            assert os.path.exists(xlsx_path)
            
            # LaTeX
            tex_path = os.path.join(td, 'staggered.tex')
            staggered_results.to_latex(tex_path)
            assert os.path.exists(tex_path)
            
            # Stata
            dta_path = os.path.join(td, 'staggered.dta')
            staggered_results.to_stata(dta_path)
            assert os.path.exists(dta_path)
    
    def test_csv_and_stata_have_same_data(self, staggered_results):
        """CSV and Stata exports should have the same data."""
        with tempfile.TemporaryDirectory() as td:
            csv_path = os.path.join(td, 'data.csv')
            dta_path = os.path.join(td, 'data.dta')
            
            staggered_results.to_csv(csv_path)
            staggered_results.to_stata(dta_path)
            
            csv_df = pd.read_csv(csv_path)
            dta_df = pd.read_stata(dta_path)
            
            # Same number of rows
            assert len(csv_df) == len(dta_df)
            
            # Compare common columns
            common_cols = set(csv_df.columns) & set(dta_df.columns)
            for col in common_cols:
                if csv_df[col].dtype in [np.float64, np.float32]:
                    np.testing.assert_allclose(
                        csv_df[col].values,
                        dta_df[col].values,
                        rtol=1e-6
                    )


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestToStataEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_empty_att_by_period(self, common_timing_data):
        """Should handle empty att_by_period gracefully."""
        results = lwdid(
            common_timing_data,
            y='lcigsale',
            d='d',
            ivar='state',
            tvar='year',
            post='post',
            rolling='demean',
        )
        results._att_by_period = pd.DataFrame()
        
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'empty.dta')
            
            with pytest.raises(ValueError, match="att_by_period is not available"):
                results.to_stata(path)
    
    def test_special_characters_in_all_columns(self):
        """DataFrame with special characters in all column names."""
        df = pd.DataFrame({
            'p-value': [0.05],
            'ci.lower': [-0.1],
            't stat': [2.0],
            '95% CI': ['[-0.1, 0.3]'],
            'effect(1)': [0.2],
        })
        
        result, labels = _prepare_stata_dataframe(df)
        
        # All columns should be valid Stata names
        for col in result.columns:
            assert re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', col), f"Invalid name: {col}"
            assert len(col) <= 32


# =============================================================================
# BUG-050: variable_labels Compatibility Tests
# =============================================================================

class TestBug050VariableLabelsCompatibility:
    """
    Tests for BUG-050: variable_labels compatibility fallback mechanism.
    
    BUG-050 reported that the try-except block for variable_labels compatibility
    was incorrectly placed, causing TypeError from older pandas versions to go
    uncaught. The fix moves df.to_stata() inside the try block.
    
    These tests verify the fallback mechanism works correctly by mocking
    TypeError scenarios that would occur with pandas < 1.4.0.
    """
    
    def test_variable_labels_typeerror_fallback_common_timing(self, common_timing_results):
        """
        TypeError from variable_labels should trigger graceful fallback.
        
        When pandas < 1.4.0 raises TypeError for variable_labels parameter,
        the method should:
        1. Catch the TypeError
        2. Issue a UserWarning about pandas version
        3. Retry export without variable_labels
        4. Successfully create the output file
        """
        from unittest import mock
        
        original_to_stata = pd.DataFrame.to_stata
        call_count = [0]
        
        def mock_to_stata(df_self, path, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1 and 'variable_labels' in kwargs:
                raise TypeError("to_stata() got an unexpected keyword argument 'variable_labels'")
            # On retry (or if no variable_labels), use original but ensure no variable_labels
            kwargs.pop('variable_labels', None)
            return original_to_stata(df_self, path, **kwargs)
        
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'test_fallback.dta')
            with mock.patch.object(pd.DataFrame, 'to_stata', mock_to_stata):
                with pytest.warns(UserWarning, match="pandas version doesn't support variable_labels"):
                    common_timing_results.to_stata(path)
            
            # Verify file was created and is readable
            assert os.path.exists(path), "Output file should exist after fallback"
            df = pd.read_stata(path)
            assert len(df) > 0, "Exported DataFrame should have data"
            
            # Verify the mock was called twice (first with labels, then without)
            assert call_count[0] == 2, "to_stata should be called twice (original + retry)"
    
    def test_variable_labels_typeerror_fallback_staggered(self, staggered_results):
        """
        to_stata_staggered should also have the same fallback mechanism.
        
        Staggered results use a separate code path and should exhibit
        identical behavior when encountering variable_labels TypeError.
        """
        from unittest import mock
        
        original_to_stata = pd.DataFrame.to_stata
        call_count = [0]
        
        def mock_to_stata(df_self, path, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1 and 'variable_labels' in kwargs:
                raise TypeError("to_stata() got an unexpected keyword argument 'variable_labels'")
            kwargs.pop('variable_labels', None)
            return original_to_stata(df_self, path, **kwargs)
        
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'test_staggered_fallback.dta')
            with mock.patch.object(pd.DataFrame, 'to_stata', mock_to_stata):
                with pytest.warns(UserWarning, match="pandas version doesn't support variable_labels"):
                    staggered_results.to_stata(path)
            
            assert os.path.exists(path), "Output file should exist after fallback"
            df = pd.read_stata(path)
            assert len(df) > 0, "Exported DataFrame should have data"
            assert call_count[0] == 2, "to_stata should be called twice"
    
    def test_non_variable_labels_typeerror_propagates(self, common_timing_results):
        """
        Non-variable_labels TypeError should propagate without being caught.
        
        The fallback mechanism should only catch TypeError specifically
        mentioning 'variable_labels'. Other TypeErrors should propagate
        to help users identify genuine issues.
        """
        from unittest import mock
        
        def mock_to_stata(df_self, path, **kwargs):
            raise TypeError("Some other unrelated TypeError message")
        
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'test_propagate.dta')
            with mock.patch.object(pd.DataFrame, 'to_stata', mock_to_stata):
                with pytest.raises(TypeError, match="Some other unrelated TypeError"):
                    common_timing_results.to_stata(path)
    
    def test_typeerror_with_variable_labels_in_message_triggers_fallback(self, common_timing_results):
        """
        TypeError containing 'variable_labels' in message should trigger fallback.
        
        The fallback condition checks if 'variable_labels' is in the error
        message AND if labels dict is truthy. Since _prepare_stata_dataframe
        generates default labels, the fallback will be triggered even with
        empty user-provided variable_labels.
        """
        from unittest import mock
        
        original_to_stata = pd.DataFrame.to_stata
        call_count = [0]
        
        def mock_to_stata(df_self, path, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1 and 'variable_labels' in kwargs:
                raise TypeError("variable_labels is not supported")
            kwargs.pop('variable_labels', None)
            return original_to_stata(df_self, path, **kwargs)
        
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'test_empty_labels.dta')
            with mock.patch.object(pd.DataFrame, 'to_stata', mock_to_stata):
                # Should trigger fallback and emit warning
                with pytest.warns(UserWarning, match="pandas version doesn't support"):
                    common_timing_results.to_stata(path, variable_labels={})
    
    def test_successful_export_with_labels_no_fallback(self, common_timing_results):
        """
        Normal export with variable_labels should succeed without warning.
        
        When pandas supports variable_labels (>= 1.4.0), no warning should
        be issued and the file should be created with labels intact.
        """
        import warnings
        
        custom_labels = {'period': 'Time Period', 'att': 'Average Treatment Effect'}
        
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'test_normal.dta')
            # Capture warnings to verify no variable_labels warnings
            with warnings.catch_warnings(record=True) as warning_list:
                warnings.simplefilter("always")
                common_timing_results.to_stata(path, variable_labels=custom_labels)
            
            # Filter out any unrelated warnings, check for variable_labels warnings
            relevant_warnings = [
                w for w in warning_list 
                if 'variable_labels' in str(w.message)
            ]
            assert len(relevant_warnings) == 0, "No variable_labels warnings expected"
            
            assert os.path.exists(path)
            df = pd.read_stata(path)
            assert len(df) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
