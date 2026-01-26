"""
Tests for bug fixes: BUG-276, BUG-350, BUG-351, BUG-355.

BUG-276: IPW degrees of freedom should be n - 2 (intercept + treatment only)
BUG-350: tvar type check should accept tuple (already fixed, verified here)
BUG-351: rireps parameter should accept numpy integer types
BUG-355: cluster NaN filtering should validate treatment/control group composition
"""

import numpy as np
import pandas as pd
import pytest
import warnings

from lwdid import lwdid
from lwdid.core import _convert_ipw_result_to_dict
from lwdid.staggered.estimation import run_ols_regression


# =============================================================================
# BUG-276: IPW degrees of freedom calculation
# =============================================================================

class TestBUG276IPWDegreesOfFreedom:
    """Test that IPW df is calculated as n - 2, not n - (2 + n_ps_controls)."""
    
    def test_ipw_df_without_ps_controls(self):
        """IPW df should be n - 2 regardless of ps_controls."""
        # Create mock IPWResult
        class MockIPWResult:
            att = 1.5
            se = 0.3
            t_stat = 5.0
            pvalue = 0.001
            ci_lower = 0.9
            ci_upper = 2.1
            n_treated = 50
            n_control = 100
            weights_cv = 0.5
            propensity_scores = np.array([0.3, 0.4, 0.5])
            diagnostics = {}
        
        result = _convert_ipw_result_to_dict(
            MockIPWResult(),
            alpha=0.05,
            vce='robust',
            cluster_var=None,
            controls=['x1', 'x2'],  # 2 controls
            ps_controls=None
        )
        
        # df should be n - 2 = 150 - 2 = 148
        # NOT n - (2 + n_ps_controls) = 150 - 4 = 146
        expected_df = 50 + 100 - 2
        assert result['df_resid'] == expected_df, \
            f"Expected df_resid={expected_df}, got {result['df_resid']}"
        assert result['df_inference'] == expected_df, \
            f"Expected df_inference={expected_df}, got {result['df_inference']}"
    
    def test_ipw_df_with_ps_controls(self):
        """IPW df should still be n - 2 even with ps_controls specified."""
        class MockIPWResult:
            att = 1.5
            se = 0.3
            t_stat = 5.0
            pvalue = 0.001
            ci_lower = 0.9
            ci_upper = 2.1
            n_treated = 50
            n_control = 100
            weights_cv = 0.5
            propensity_scores = np.array([0.3, 0.4, 0.5])
            diagnostics = {}
        
        result = _convert_ipw_result_to_dict(
            MockIPWResult(),
            alpha=0.05,
            vce='robust',
            cluster_var=None,
            controls=['x1', 'x2'],
            ps_controls=['x1', 'x2', 'x3', 'x4']  # 4 PS controls
        )
        
        # df should still be n - 2 = 148
        # NOT n - (2 + 4) = 144
        expected_df = 50 + 100 - 2
        assert result['df_resid'] == expected_df, \
            f"Expected df_resid={expected_df}, got {result['df_resid']}"
        assert result['df_inference'] == expected_df, \
            f"Expected df_inference={expected_df}, got {result['df_inference']}"


# =============================================================================
# BUG-351: rireps parameter should accept numpy integer types
# =============================================================================

class TestBUG351RirepsNumpyInteger:
    """Test that rireps accepts numpy integer types."""
    
    @pytest.fixture
    def simple_data(self):
        """Create simple test data for common timing."""
        np.random.seed(42)
        n_units = 50
        n_periods = 4
        
        data = pd.DataFrame({
            'unit': np.repeat(range(n_units), n_periods),
            'time': np.tile(range(n_periods), n_units),
            'y': np.random.randn(n_units * n_periods),
            'd': np.repeat(np.random.choice([0, 1], n_units), n_periods),
            'post': np.tile([0, 0, 1, 1], n_units),
        })
        return data
    
    def test_rireps_accepts_numpy_int64(self, simple_data):
        """Test that np.int64 is accepted for rireps."""
        # This should NOT raise ValueError
        rireps_value = np.int64(10)
        
        # Verify it's actually a numpy integer, not Python int
        assert isinstance(rireps_value, np.integer)
        assert not isinstance(rireps_value, int) or isinstance(rireps_value, np.integer)
        
        # Should work without error
        result = lwdid(
            data=simple_data,
            y='y',
            d='d',
            ivar='unit',
            tvar='time',
            post='post',
            rolling='demean',
            ri=True,
            rireps=rireps_value  # np.int64
        )
        assert result is not None
    
    def test_rireps_accepts_numpy_int32(self, simple_data):
        """Test that np.int32 is accepted for rireps."""
        rireps_value = np.int32(10)
        
        result = lwdid(
            data=simple_data,
            y='y',
            d='d',
            ivar='unit',
            tvar='time',
            post='post',
            rolling='demean',
            ri=True,
            rireps=rireps_value
        )
        assert result is not None
    
    def test_rireps_accepts_python_int(self, simple_data):
        """Test that Python int still works."""
        result = lwdid(
            data=simple_data,
            y='y',
            d='d',
            ivar='unit',
            tvar='time',
            post='post',
            rolling='demean',
            ri=True,
            rireps=10  # Python int
        )
        assert result is not None


# =============================================================================
# BUG-350: tvar type check should accept tuple
# =============================================================================

class TestBUG350TvarTupleAcceptance:
    """Test that tvar accepts both list and tuple for quarterly data."""
    
    @pytest.fixture
    def quarterly_data(self):
        """Create quarterly panel data."""
        np.random.seed(42)
        n_units = 20
        years = [2018, 2019, 2020, 2021]
        quarters = [1, 2, 3, 4]
        
        rows = []
        for unit in range(n_units):
            gvar = np.random.choice([0, 2020], p=[0.5, 0.5])  # Half treated in 2020
            for year in years:
                for quarter in quarters:
                    rows.append({
                        'unit': unit,
                        'year': year,
                        'quarter': quarter,
                        'y': np.random.randn(),
                        'gvar': gvar,
                    })
        
        return pd.DataFrame(rows)
    
    def test_tvar_accepts_tuple(self, quarterly_data):
        """Test that tvar=('year', 'quarter') is accepted."""
        # This should NOT raise TypeError
        result = lwdid(
            data=quarterly_data,
            y='y',
            gvar='gvar',
            ivar='unit',
            tvar=('year', 'quarter'),  # tuple
            rolling='demean',
        )
        assert result is not None
    
    def test_tvar_accepts_list(self, quarterly_data):
        """Test that tvar=['year', 'quarter'] is still accepted."""
        result = lwdid(
            data=quarterly_data,
            y='y',
            gvar='gvar',
            ivar='unit',
            tvar=['year', 'quarter'],  # list
            rolling='demean',
        )
        assert result is not None


# =============================================================================
# BUG-355: cluster NaN filtering validation
# =============================================================================

class TestBUG355ClusterNaNValidation:
    """Test that cluster NaN filtering validates group composition."""
    
    def test_cluster_nan_removes_all_treated(self):
        """Test error when NaN filtering removes all treated units."""
        # Create data where all treated units have NaN cluster
        data = pd.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0],
            'd': [1, 1, 0, 0, 0],  # 2 treated, 3 control
            'cluster': [np.nan, np.nan, 1, 2, 3],  # treated have NaN cluster
        })
        
        with pytest.raises(ValueError, match="No treated units remain"):
            run_ols_regression(
                data=data,
                y='y',
                d='d',
                controls=None,
                vce='cluster',
                cluster_var='cluster'
            )
    
    def test_cluster_nan_removes_all_control(self):
        """Test error when NaN filtering removes all control units."""
        data = pd.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0],
            'd': [1, 1, 0, 0, 0],
            'cluster': [1, 2, np.nan, np.nan, np.nan],  # control have NaN cluster
        })
        
        with pytest.raises(ValueError, match="No control units remain"):
            run_ols_regression(
                data=data,
                y='y',
                d='d',
                controls=None,
                vce='cluster',
                cluster_var='cluster'
            )
    
    def test_cluster_nan_partial_filtering_succeeds(self):
        """Test that partial NaN filtering still works."""
        np.random.seed(42)
        data = pd.DataFrame({
            'y': np.random.randn(100),
            'd': np.array([1] * 30 + [0] * 70),
            'cluster': np.array([np.nan] * 5 + list(range(1, 96))),  # Only 5 NaN
        })
        
        # Should succeed with warning about NaN values
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = run_ols_regression(
                data=data,
                y='y',
                d='d',
                controls=None,
                vce='cluster',
                cluster_var='cluster'
            )
            
            # Check warning was issued
            nan_warnings = [x for x in w if "missing values" in str(x.message)]
            assert len(nan_warnings) >= 1, "Expected warning about NaN cluster values"
        
        assert 'att' in result
        assert np.isfinite(result['att'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
