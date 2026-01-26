"""
Test suite for bug fixes BUG-283 to BUG-285 (Round 85 code review).

This module tests the following bug fixes:
- BUG-283: staggered/randomization.py data completeness check for ivar and tvar
- BUG-284: staggered/transformations.py tvar NaN validation
- BUG-285: results.py plot_event_study Inf SE warning
"""

import warnings
import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lwdid.staggered.randomization import randomization_inference_staggered
from lwdid.staggered.transformations import (
    transform_staggered_demean,
    transform_staggered_detrend,
)
from lwdid.results import LWDIDResults
from lwdid.exceptions import RandomizationError


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def staggered_panel_data():
    """Create a staggered adoption panel dataset for testing."""
    np.random.seed(42)
    n_units = 30
    n_periods = 6
    
    data = []
    for i in range(1, n_units + 1):
        # Assign cohort: 10 units never treated, 10 units treated in 2003, 10 in 2005
        if i <= 10:
            gvar = 0  # never treated
        elif i <= 20:
            gvar = 2003
        else:
            gvar = 2005
            
        for t in range(2000, 2000 + n_periods):
            treated = 1 if gvar > 0 and t >= gvar else 0
            y = 10 + 2*treated + np.random.normal(0, 1)
            data.append({
                'unit': i,
                'year': t,
                'gvar': gvar,
                'outcome': y,
            })
    
    return pd.DataFrame(data)


# =============================================================================
# BUG-283: staggered/randomization.py data completeness check
# =============================================================================

class TestBug283IvarTvarValidation:
    """Test that randomization_inference_staggered validates ivar and tvar for NaN."""
    
    def test_ivar_missing_raises_error(self, staggered_panel_data):
        """Test that missing ivar values raise RandomizationError."""
        df = staggered_panel_data.copy()
        # Introduce NaN in ivar
        df.loc[0, 'unit'] = np.nan
        
        with pytest.raises(RandomizationError) as excinfo:
            randomization_inference_staggered(
                data=df,
                gvar='gvar',
                ivar='unit',
                tvar='year',
                y='outcome',
                observed_att=1.0,
                target='cohort_time',
                target_cohort=2003,
                target_period=2003,
                rireps=10,
                seed=42,
            )
        
        assert "unit identifier" in str(excinfo.value).lower()
        assert "missing" in str(excinfo.value).lower()
    
    def test_tvar_missing_raises_error(self, staggered_panel_data):
        """Test that missing tvar values raise RandomizationError."""
        df = staggered_panel_data.copy()
        # Introduce NaN in tvar
        df.loc[0, 'year'] = np.nan
        
        with pytest.raises(RandomizationError) as excinfo:
            randomization_inference_staggered(
                data=df,
                gvar='gvar',
                ivar='unit',
                tvar='year',
                y='outcome',
                observed_att=1.0,
                target='cohort_time',
                target_cohort=2003,
                target_period=2003,
                rireps=10,
                seed=42,
            )
        
        assert "time period" in str(excinfo.value).lower()
        assert "missing" in str(excinfo.value).lower()
    
    def test_complete_data_runs_without_error(self, staggered_panel_data):
        """Test that complete data passes validation."""
        df = staggered_panel_data.copy()
        
        # Should not raise
        result = randomization_inference_staggered(
            data=df,
            gvar='gvar',
            ivar='unit',
            tvar='year',
            y='outcome',
            observed_att=1.0,
            target='cohort_time',
            target_cohort=2003,
            target_period=2003,
            rireps=20,
            seed=42,
        )
        
        assert result is not None
        assert hasattr(result, 'p_value')


# =============================================================================
# BUG-284: staggered/transformations.py tvar NaN validation
# =============================================================================

class TestBug284TvarNaNValidation:
    """Test that transform functions validate tvar for NaN after conversion."""
    
    def test_demean_tvar_nan_raises_error(self, staggered_panel_data):
        """Test that transform_staggered_demean raises error for NaN tvar."""
        df = staggered_panel_data.copy()
        # Introduce NaN in tvar
        df.loc[0, 'year'] = np.nan
        
        with pytest.raises(ValueError) as excinfo:
            transform_staggered_demean(
                data=df,
                y='outcome',
                ivar='unit',
                tvar='year',
                gvar='gvar',
            )
        
        assert "missing value" in str(excinfo.value).lower()
        assert "year" in str(excinfo.value).lower()
    
    def test_detrend_tvar_nan_raises_error(self, staggered_panel_data):
        """Test that transform_staggered_detrend raises error for NaN tvar."""
        df = staggered_panel_data.copy()
        # Introduce NaN in tvar
        df.loc[0, 'year'] = np.nan
        
        with pytest.raises(ValueError) as excinfo:
            transform_staggered_detrend(
                data=df,
                y='outcome',
                ivar='unit',
                tvar='year',
                gvar='gvar',
            )
        
        assert "missing value" in str(excinfo.value).lower()
        assert "year" in str(excinfo.value).lower()
    
    def test_demean_complete_tvar_runs(self, staggered_panel_data):
        """Test that transform_staggered_demean works with complete tvar."""
        df = staggered_panel_data.copy()
        
        result = transform_staggered_demean(
            data=df,
            y='outcome',
            ivar='unit',
            tvar='year',
            gvar='gvar',
        )
        
        assert result is not None
        assert len(result) == len(df)
    
    def test_detrend_complete_tvar_runs(self, staggered_panel_data):
        """Test that transform_staggered_detrend works with complete tvar."""
        df = staggered_panel_data.copy()
        
        result = transform_staggered_detrend(
            data=df,
            y='outcome',
            ivar='unit',
            tvar='year',
            gvar='gvar',
        )
        
        assert result is not None
        assert len(result) == len(df)
    
    def test_demean_non_numeric_tvar_raises_error(self, staggered_panel_data):
        """Test that non-numeric tvar values (converted to NaN) raise error."""
        df = staggered_panel_data.copy()
        df['year'] = df['year'].astype(str)
        df.loc[0, 'year'] = 'invalid'
        
        with pytest.raises(ValueError) as excinfo:
            transform_staggered_demean(
                data=df,
                y='outcome',
                ivar='unit',
                tvar='year',
                gvar='gvar',
            )
        
        assert "missing value" in str(excinfo.value).lower()


# =============================================================================
# BUG-285: results.py plot_event_study Inf SE warning
# =============================================================================

class TestBug285InfSEWarning:
    """Test that plot_event_study warns about infinite standard errors."""
    
    @staticmethod
    def _create_base_results_dict():
        """Create base results_dict with all required keys for staggered results."""
        return {
            'att': 1.5,
            'se_att': 0.3,
            't_stat': 5.0,
            'pvalue': 0.001,
            'ci_lower': 0.9,
            'ci_upper': 2.1,
            'nobs': 100,
            'df_resid': 98,
            'df_inference': 98,
            'params': np.array([1.5]),
            'bse': np.array([0.3]),
            'vcov': np.array([[0.09]]),
            'resid': np.zeros(100),
            'vce_type': 'robust',
            'n_treated': 50,
            'n_control': 50,
            'is_staggered': True,  # Must be in results_dict, not metadata
            'cohorts': [2003],
            'cohort_sizes': {2003: 20},
        }
    
    @staticmethod
    def _create_base_metadata():
        """Create base metadata for staggered results."""
        return {
            'K': 2,
            'tpost1': 2003,
            'depvar': 'outcome',
            'N_treated': 50,
            'N_control': 50,
            'rolling': 'demean',
        }
    
    def test_inf_se_warning_issued(self):
        """Test that infinite SE values trigger a warning."""
        # Create mock LWDIDResults with inf SE
        results_dict = self._create_base_results_dict()
        results_dict['att_by_cohort_time'] = pd.DataFrame({
            'cohort': [2003, 2003, 2003],
            'period': [2003, 2004, 2005],
            'att': [1.0, 1.5, 2.0],
            'se': [0.3, np.inf, 0.4],  # One infinite SE
            'n_units': [20, 20, 20],
        })
        results_dict['att_by_cohort'] = pd.DataFrame({
            'cohort': [2003],
            'att': [1.5],
            'se': [0.3],
            'n_units': [20],
            'n_periods': [3],
        })
        
        metadata = self._create_base_metadata()
        
        result = LWDIDResults(results_dict, metadata)
        
        # Capture warnings during plot_event_study
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            try:
                fig, ax = result.plot_event_study(se_method='analytical')
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception:
                # Plot may fail for other reasons in test environment
                pass
            
            # Check if Inf SE warning was issued
            inf_warnings = [
                warning for warning in w 
                if "infinite standard errors" in str(warning.message).lower()
            ]
            
            # The warning should be issued
            assert len(inf_warnings) >= 1, (
                f"Expected warning about infinite SE, got warnings: "
                f"{[str(x.message) for x in w]}"
            )
    
    def test_warning_code_structure_exists(self):
        """Test that both NaN and Inf SE warning code paths exist in results.py.
        
        This is a code structure test rather than a runtime test. The NaN SE warning
        was pre-existing code; the Inf SE warning was added by BUG-285.
        Runtime testing of the Inf SE warning is covered by test_inf_se_warning_issued.
        """
        import inspect
        from lwdid.results import LWDIDResults
        
        # Get the source code of plot_event_study
        source = inspect.getsource(LWDIDResults.plot_event_study)
        
        # Verify NaN SE warning code exists (pre-existing)
        assert "event_df['se'].isna().any()" in source, "NaN SE check should exist"
        assert "NaN standard errors" in source or "nan standard errors" in source.lower(), \
            "NaN SE warning message should exist"
        
        # Verify Inf SE warning code exists (BUG-285 fix)
        assert "np.isinf(event_df['se']).any()" in source, "Inf SE check should exist"
        assert "infinite standard errors" in source, "Inf SE warning message should exist"
    
    def test_no_warning_for_valid_se(self):
        """Test that no SE warning is issued when all SE values are valid."""
        results_dict = self._create_base_results_dict()
        results_dict['att_by_cohort_time'] = pd.DataFrame({
            'cohort': [2003, 2003, 2003],
            'period': [2003, 2004, 2005],
            'att': [1.0, 1.5, 2.0],
            'se': [0.3, 0.35, 0.4],  # All valid SE
            'n_units': [20, 20, 20],
        })
        results_dict['att_by_cohort'] = pd.DataFrame({
            'cohort': [2003],
            'att': [1.5],
            'se': [0.35],
            'n_units': [20],
            'n_periods': [3],
        })
        
        metadata = self._create_base_metadata()
        
        result = LWDIDResults(results_dict, metadata)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            try:
                fig, ax = result.plot_event_study(se_method='analytical')
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception:
                pass
            
            # Check that no SE-related warnings were issued
            se_warnings = [
                warning for warning in w 
                if "standard errors" in str(warning.message).lower()
                and ("nan" in str(warning.message).lower() 
                     or "infinite" in str(warning.message).lower())
            ]
            
            assert len(se_warnings) == 0, (
                f"Expected no SE warnings, got: {[str(x.message) for x in se_warnings]}"
            )


# =============================================================================
# Run tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
