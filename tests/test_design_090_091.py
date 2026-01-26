"""
Unit tests for DESIGN-090 and DESIGN-091 fixes.

DESIGN-090: to_stata_staggered string column Stata compatibility
DESIGN-091: LWDIDResults.__init__ parameter validation before assignment
"""

import os
import tempfile
import warnings
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from lwdid.results import LWDIDResults, _prepare_stata_dataframe


class TestDesign090StataStringCompatibility:
    """Tests for DESIGN-090: Stata string column compatibility."""

    def _create_mock_staggered_results(self, estimator='ra', aggregate='overall'):
        """Create mock staggered results for testing."""
        results_dict = {
            'att': 0.5,
            'se_att': 0.1,
            't_stat': 5.0,
            'pvalue': 0.001,
            'ci_lower': 0.3,
            'ci_upper': 0.7,
            'nobs': 1000,
            'df_resid': 998,
            'params': np.array([0.5]),
            'bse': np.array([0.1]),
            'vcov': np.array([[0.01]]),
            'resid': np.zeros(1000),
            'vce_type': 'robust',
            'is_staggered': True,
            'cohorts': [2004, 2005],
            'cohort_sizes': {2004: 50, 2005: 50},
            'att_by_cohort_time': pd.DataFrame({
                'cohort': [2004, 2004, 2005],
                'period': [2004, 2005, 2005],
                'att': [0.4, 0.5, 0.6],
                'se': [0.1, 0.1, 0.1],
            }),
            'att_by_cohort': pd.DataFrame({
                'cohort': [2004, 2005],
                'att': [0.45, 0.6],
                'se': [0.08, 0.1],
            }),
            'att_overall': 0.525,
            'se_overall': 0.07,
            'ci_overall_lower': 0.4,
            'ci_overall_upper': 0.65,
            't_stat_overall': 7.5,
            'pvalue_overall': 0.0001,
            'cohort_weights': {2004: 0.5, 2005: 0.5},
            'control_group': 'not_yet_treated',
            'control_group_used': 'not_yet_treated',
            'aggregate': aggregate,
            'estimator': estimator,
            'n_never_treated': 100,
            'n_treated_sample': 100,
            'n_control_sample': 100,
        }
        metadata = {
            'K': 3,
            'tpost1': 2004,
            'depvar': 'y',
            'N_treated': 100,
            'N_control': 100,
        }
        return LWDIDResults(results_dict, metadata)

    def test_to_stata_staggered_with_string_columns(self):
        """Test that to_stata_staggered handles string columns properly."""
        results = self._create_mock_staggered_results()
        
        with tempfile.NamedTemporaryFile(suffix='.dta', delete=False) as f:
            path = f.name
        
        try:
            # Export overall results which include string columns
            results.to_stata_staggered(path, what='overall')
            
            # Read back and verify
            df_read = pd.read_stata(path)
            assert 'estimator' in df_read.columns
            # Note: 'aggregate' is renamed to '_aggregate' by pandas because it's a Stata reserved word
            assert '_aggregate' in df_read.columns or 'aggregate' in df_read.columns
            assert df_read['estimator'].iloc[0] == 'ra'
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_to_stata_staggered_non_ascii_sanitization(self):
        """Test that non-ASCII characters are properly sanitized."""
        results = self._create_mock_staggered_results()
        
        # Manually add non-ASCII content to att_by_cohort_time
        results._att_by_cohort_time = pd.DataFrame({
            'cohort': [2004, 2005],
            'period': [2004, 2005],
            'att': [0.4, 0.5],
            'se': [0.1, 0.1],
            'note': ['中文测试', 'éàü特殊字符'],
        })
        
        with tempfile.NamedTemporaryFile(suffix='.dta', delete=False) as f:
            path = f.name
        
        try:
            # Should not raise an error
            results.to_stata_staggered(path, what='cohort_time')
            
            # Read back and verify ASCII-safe encoding
            df_read = pd.read_stata(path)
            # Non-ASCII chars should be replaced with '?'
            assert '?' in df_read['note'].iloc[0] or df_read['note'].iloc[0].isascii()
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_to_stata_staggered_long_string_warning(self):
        """Test that long strings trigger a warning and are truncated."""
        results = self._create_mock_staggered_results()
        
        # Create a very long string (> 2045 chars)
        long_string = 'A' * 3000
        results._att_by_cohort_time = pd.DataFrame({
            'cohort': [2004],
            'period': [2004],
            'att': [0.4],
            'se': [0.1],
            'long_note': [long_string],
        })
        
        with tempfile.NamedTemporaryFile(suffix='.dta', delete=False) as f:
            path = f.name
        
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                # Use version=117 to support strL for long strings (default version has 244 char limit)
                results.to_stata_staggered(path, what='cohort_time', version=117)
                
                # Check that a warning was raised about string length
                str_warnings = [x for x in w if 'Stata str limit' in str(x.message)]
                assert len(str_warnings) > 0
            
            # Read back and verify truncation
            df_read = pd.read_stata(path)
            assert len(df_read['long_note'].iloc[0]) <= 2045
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_to_stata_common_timing_string_handling(self):
        """Test that to_stata for common timing also handles strings."""
        results_dict = {
            'att': 0.5,
            'se_att': 0.1,
            't_stat': 5.0,
            'pvalue': 0.001,
            'ci_lower': 0.3,
            'ci_upper': 0.7,
            'nobs': 100,
            'df_resid': 98,
            'params': np.array([0.5]),
            'bse': np.array([0.1]),
            'vcov': np.array([[0.01]]),
            'resid': np.zeros(100),
            'vce_type': 'robust',
        }
        metadata = {
            'K': 3,
            'tpost1': 4,
            'depvar': 'y',
            'N_treated': 50,
            'N_control': 50,
        }
        att_by_period = pd.DataFrame({
            'tindex': [4, 5],
            'period': ['2004', '2005'],
            'beta': [0.4, 0.6],
            'se': [0.1, 0.1],
        })
        
        results = LWDIDResults(results_dict, metadata, att_by_period=att_by_period)
        
        with tempfile.NamedTemporaryFile(suffix='.dta', delete=False) as f:
            path = f.name
        
        try:
            results.to_stata(path)
            df_read = pd.read_stata(path)
            assert 'period' in df_read.columns
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestDesign091InitValidationOrder:
    """Tests for DESIGN-091: Parameter validation before assignment."""

    def test_n_treated_validation_before_assignment_negative(self):
        """Test that negative n_treated is corrected with warning."""
        results_dict = {
            'att': 0.5,
            'se_att': 0.1,
            't_stat': 5.0,
            'pvalue': 0.001,
            'ci_lower': 0.3,
            'ci_upper': 0.7,
            'nobs': 100,
            'df_resid': 98,
            'params': np.array([0.5]),
            'bse': np.array([0.1]),
            'vcov': np.array([[0.01]]),
            'resid': np.zeros(100),
            'vce_type': 'robust',
            'n_treated_sample': -10,  # Invalid negative value
        }
        metadata = {
            'K': 3,
            'tpost1': 4,
            'depvar': 'y',
            'N_treated': 50,
            'N_control': 50,
        }
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = LWDIDResults(results_dict, metadata)
            
            # Warning should be raised
            n_warnings = [x for x in w if 'n_treated should be a non-negative integer' in str(x.message)]
            assert len(n_warnings) > 0
        
        # Value should be corrected to 0
        assert results.n_treated == 0

    def test_n_control_validation_before_assignment_float(self):
        """Test that float n_control is corrected with warning."""
        results_dict = {
            'att': 0.5,
            'se_att': 0.1,
            't_stat': 5.0,
            'pvalue': 0.001,
            'ci_lower': 0.3,
            'ci_upper': 0.7,
            'nobs': 100,
            'df_resid': 98,
            'params': np.array([0.5]),
            'bse': np.array([0.1]),
            'vcov': np.array([[0.01]]),
            'resid': np.zeros(100),
            'vce_type': 'robust',
            'n_control_sample': 45.7,  # Invalid float value
        }
        metadata = {
            'K': 3,
            'tpost1': 4,
            'depvar': 'y',
            'N_treated': 50,
            'N_control': 50,
        }
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = LWDIDResults(results_dict, metadata)
            
            # Warning should be raised
            n_warnings = [x for x in w if 'n_control should be a non-negative integer' in str(x.message)]
            assert len(n_warnings) > 0
        
        # Value should be corrected to int(45.7) = 45
        assert results.n_control == 45

    def test_valid_values_no_warning(self):
        """Test that valid integer values produce no warnings."""
        results_dict = {
            'att': 0.5,
            'se_att': 0.1,
            't_stat': 5.0,
            'pvalue': 0.001,
            'ci_lower': 0.3,
            'ci_upper': 0.7,
            'nobs': 100,
            'df_resid': 98,
            'params': np.array([0.5]),
            'bse': np.array([0.1]),
            'vcov': np.array([[0.01]]),
            'resid': np.zeros(100),
            'vce_type': 'robust',
            'n_treated_sample': 50,
            'n_control_sample': 50,
        }
        metadata = {
            'K': 3,
            'tpost1': 4,
            'depvar': 'y',
            'N_treated': 50,
            'N_control': 50,
        }
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = LWDIDResults(results_dict, metadata)
            
            # No warnings about n_treated or n_control
            n_warnings = [x for x in w if 'n_treated' in str(x.message) or 'n_control' in str(x.message)]
            assert len(n_warnings) == 0
        
        assert results.n_treated == 50
        assert results.n_control == 50

    def test_numpy_integer_types_accepted(self):
        """Test that numpy integer types are accepted without warning."""
        results_dict = {
            'att': 0.5,
            'se_att': 0.1,
            't_stat': 5.0,
            'pvalue': 0.001,
            'ci_lower': 0.3,
            'ci_upper': 0.7,
            'nobs': 100,
            'df_resid': 98,
            'params': np.array([0.5]),
            'bse': np.array([0.1]),
            'vcov': np.array([[0.01]]),
            'resid': np.zeros(100),
            'vce_type': 'robust',
            'n_treated_sample': np.int64(50),
            'n_control_sample': np.int32(50),
        }
        metadata = {
            'K': 3,
            'tpost1': 4,
            'depvar': 'y',
            'N_treated': 50,
            'N_control': 50,
        }
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = LWDIDResults(results_dict, metadata)
            
            # No warnings for numpy integer types
            n_warnings = [x for x in w if 'n_treated' in str(x.message) or 'n_control' in str(x.message)]
            assert len(n_warnings) == 0
        
        assert results.n_treated == 50
        assert results.n_control == 50

    def test_fallback_to_metadata_values(self):
        """Test fallback to metadata values when sample values not provided."""
        results_dict = {
            'att': 0.5,
            'se_att': 0.1,
            't_stat': 5.0,
            'pvalue': 0.001,
            'ci_lower': 0.3,
            'ci_upper': 0.7,
            'nobs': 100,
            'df_resid': 98,
            'params': np.array([0.5]),
            'bse': np.array([0.1]),
            'vcov': np.array([[0.01]]),
            'resid': np.zeros(100),
            'vce_type': 'robust',
            # No n_treated_sample or n_control_sample
        }
        metadata = {
            'K': 3,
            'tpost1': 4,
            'depvar': 'y',
            'N_treated': 60,
            'N_control': 40,
        }
        
        results = LWDIDResults(results_dict, metadata)
        
        # Should use metadata values
        assert results.n_treated == 60
        assert results.n_control == 40


class TestPrepareStataDataframe:
    """Tests for _prepare_stata_dataframe helper function."""

    def test_column_name_cleaning(self):
        """Test that column names are cleaned for Stata compatibility."""
        df = pd.DataFrame({
            'valid_name': [1, 2],
            'Name With Spaces': [3, 4],
            '123_starts_with_number': [5, 6],
            'special@chars!': [7, 8],
        })
        
        cleaned_df, labels = _prepare_stata_dataframe(df)
        
        # All column names should be Stata-compatible
        for col in cleaned_df.columns:
            assert col.replace('_', '').isalnum() or col.startswith('_')
            assert len(col) <= 32
            assert not col[0].isdigit()

    def test_variable_labels_preserved(self):
        """Test that custom variable labels are preserved."""
        df = pd.DataFrame({
            'att': [0.5, 0.6],
            'se': [0.1, 0.1],
        })
        custom_labels = {
            'att': 'Average Treatment Effect',
            'se': 'Standard Error',
        }
        
        cleaned_df, labels = _prepare_stata_dataframe(df, variable_labels=custom_labels)
        
        assert labels['att'] == 'Average Treatment Effect'
        assert labels['se'] == 'Standard Error'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
