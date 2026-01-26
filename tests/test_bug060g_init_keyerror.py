"""
Unit tests for BUG-060-G: LWDIDResults.__init__() KeyError validation

Tests that missing required keys in results_dict or metadata raise
friendly ValueError messages instead of cryptic KeyError.

Test cases:
1. Missing key in results_dict raises ValueError with clear message
2. Missing key in metadata raises ValueError with clear message
3. Complete dicts work correctly (no error)
4. Error message includes list of missing keys
"""

import pytest
import numpy as np
import pandas as pd

from lwdid.results import LWDIDResults


def _create_minimal_results_dict():
    """Create a minimal complete results_dict for testing."""
    return {
        'att': 1.0,
        'se_att': 0.5,
        't_stat': 2.0,
        'pvalue': 0.046,
        'ci_lower': 0.02,
        'ci_upper': 1.98,
        'nobs': 100,
        'df_resid': 98,
        'params': np.array([0.5, 1.0]),
        'bse': np.array([0.1, 0.5]),
        'vcov': np.array([[0.01, 0], [0, 0.25]]),
        'resid': np.zeros(100),
        'vce_type': 'ols',
    }


def _create_minimal_metadata():
    """Create a minimal complete metadata dict for testing."""
    return {
        'K': 3,
        'tpost1': 4,
        'depvar': 'y',
        'N_treated': 50,
        'N_control': 50,
    }


class TestBug060GResultsDictValidation:
    """Test results_dict validation in LWDIDResults.__init__()"""
    
    def test_missing_att_raises_valueerror(self):
        """Missing 'att' key raises ValueError, not KeyError"""
        results_dict = _create_minimal_results_dict()
        del results_dict['att']
        metadata = _create_minimal_metadata()
        
        with pytest.raises(ValueError) as exc_info:
            LWDIDResults(results_dict, metadata)
        
        assert 'missing required keys in results_dict' in str(exc_info.value)
        assert 'att' in str(exc_info.value)
    
    def test_missing_df_resid_raises_valueerror(self):
        """Missing 'df_resid' key raises ValueError with friendly message"""
        results_dict = _create_minimal_results_dict()
        del results_dict['df_resid']
        metadata = _create_minimal_metadata()
        
        with pytest.raises(ValueError) as exc_info:
            LWDIDResults(results_dict, metadata)
        
        assert 'missing required keys in results_dict' in str(exc_info.value)
        assert 'df_resid' in str(exc_info.value)
    
    def test_missing_multiple_keys_lists_all(self):
        """Missing multiple keys shows all in error message"""
        results_dict = _create_minimal_results_dict()
        del results_dict['att']
        del results_dict['se_att']
        del results_dict['nobs']
        metadata = _create_minimal_metadata()
        
        with pytest.raises(ValueError) as exc_info:
            LWDIDResults(results_dict, metadata)
        
        error_msg = str(exc_info.value)
        assert 'att' in error_msg
        assert 'se_att' in error_msg
        assert 'nobs' in error_msg
    
    def test_missing_vce_type_raises_valueerror(self):
        """Missing 'vce_type' key raises ValueError"""
        results_dict = _create_minimal_results_dict()
        del results_dict['vce_type']
        metadata = _create_minimal_metadata()
        
        with pytest.raises(ValueError) as exc_info:
            LWDIDResults(results_dict, metadata)
        
        assert 'vce_type' in str(exc_info.value)


class TestBug060GMetadataValidation:
    """Test metadata validation in LWDIDResults.__init__()"""
    
    def test_missing_depvar_raises_valueerror(self):
        """Missing 'depvar' key raises ValueError, not KeyError"""
        results_dict = _create_minimal_results_dict()
        metadata = _create_minimal_metadata()
        del metadata['depvar']
        
        with pytest.raises(ValueError) as exc_info:
            LWDIDResults(results_dict, metadata)
        
        assert 'missing required keys in metadata' in str(exc_info.value)
        assert 'depvar' in str(exc_info.value)
    
    def test_missing_K_raises_valueerror(self):
        """Missing 'K' key raises ValueError"""
        results_dict = _create_minimal_results_dict()
        metadata = _create_minimal_metadata()
        del metadata['K']
        
        with pytest.raises(ValueError) as exc_info:
            LWDIDResults(results_dict, metadata)
        
        assert 'K' in str(exc_info.value)
    
    def test_missing_tpost1_raises_valueerror(self):
        """Missing 'tpost1' key raises ValueError"""
        results_dict = _create_minimal_results_dict()
        metadata = _create_minimal_metadata()
        del metadata['tpost1']
        
        with pytest.raises(ValueError) as exc_info:
            LWDIDResults(results_dict, metadata)
        
        assert 'tpost1' in str(exc_info.value)
    
    def test_missing_N_treated_raises_valueerror(self):
        """Missing 'N_treated' key raises ValueError"""
        results_dict = _create_minimal_results_dict()
        metadata = _create_minimal_metadata()
        del metadata['N_treated']
        
        with pytest.raises(ValueError) as exc_info:
            LWDIDResults(results_dict, metadata)
        
        assert 'N_treated' in str(exc_info.value)


class TestBug060GCompleteInputs:
    """Test that complete inputs work correctly"""
    
    def test_complete_inputs_no_error(self):
        """Complete results_dict and metadata do not raise errors"""
        results_dict = _create_minimal_results_dict()
        metadata = _create_minimal_metadata()
        
        # Should not raise any exception
        result = LWDIDResults(results_dict, metadata)
        
        # Verify core attributes are set correctly
        assert result.att == 1.0
        assert result.se_att == 0.5
        assert result.df_resid == 98
        assert result.depvar == 'y'
        assert result.K == 3
        assert result.tpost1 == 4
    
    def test_extra_keys_are_allowed(self):
        """Extra keys in dicts do not cause errors"""
        results_dict = _create_minimal_results_dict()
        results_dict['extra_key'] = 'extra_value'
        results_dict['another_extra'] = 123
        
        metadata = _create_minimal_metadata()
        metadata['extra_metadata'] = True
        
        # Should not raise any exception
        result = LWDIDResults(results_dict, metadata)
        assert result.att == 1.0


class TestBug060GErrorMessageQuality:
    """Test that error messages are helpful and informative"""
    
    def test_error_message_mentions_internal_error(self):
        """Error message suggests this is an internal error"""
        results_dict = _create_minimal_results_dict()
        del results_dict['att']
        metadata = _create_minimal_metadata()
        
        with pytest.raises(ValueError) as exc_info:
            LWDIDResults(results_dict, metadata)
        
        assert 'internal error' in str(exc_info.value).lower()
    
    def test_error_message_suggests_reporting(self):
        """Error message suggests reporting the issue"""
        results_dict = _create_minimal_results_dict()
        del results_dict['att']
        metadata = _create_minimal_metadata()
        
        with pytest.raises(ValueError) as exc_info:
            LWDIDResults(results_dict, metadata)
        
        assert 'report' in str(exc_info.value).lower()
    
    def test_results_dict_error_checked_first(self):
        """results_dict errors are checked before metadata errors"""
        # Both have missing keys
        results_dict = _create_minimal_results_dict()
        del results_dict['att']
        metadata = _create_minimal_metadata()
        del metadata['depvar']
        
        with pytest.raises(ValueError) as exc_info:
            LWDIDResults(results_dict, metadata)
        
        # Should mention results_dict error first
        assert 'results_dict' in str(exc_info.value)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
