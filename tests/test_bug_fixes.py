"""
Unit tests for bug fixes: BUG-099, BUG-100, BUG-101.

Tests validate:
1. BUG-099: warnings module cleanup (no functional change, code quality)
2. BUG-100: controls/propensity_controls string type validation
3. BUG-101: (ivar, tvar) panel uniqueness validation
4. BUG-101: caliper parameter boundary validation
"""

import numpy as np
import pandas as pd
import pytest

from lwdid.staggered.estimators import estimate_ipwra, estimate_ra, estimate_ipw, estimate_psm
from lwdid.validation import validate_staggered_data
from lwdid.exceptions import InvalidParameterError


class TestBUG100ControlsStringValidation:
    """Test BUG-100: controls/propensity_controls string type validation."""
    
    def setup_method(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n = 100
        self.data = pd.DataFrame({
            'y': np.random.randn(n),
            'd': np.random.binomial(1, 0.5, n),
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
            'x3': np.random.randn(n),
        })
    
    def test_estimate_ra_rejects_string_controls(self):
        """estimate_ra() should reject string controls parameter."""
        with pytest.raises(TypeError, match="controls must be a list"):
            estimate_ra(
                data=self.data,
                y='y',
                d='d',
                controls='x1 x2 x3'  # Wrong: should be ['x1', 'x2', 'x3']
            )
    
    def test_estimate_ra_accepts_list_controls(self):
        """estimate_ra() should accept list controls parameter."""
        # This should not raise an error
        result = estimate_ra(
            data=self.data,
            y='y',
            d='d',
            controls=['x1', 'x2', 'x3']  # Correct
        )
        assert result is not None
        assert hasattr(result, 'att')
    
    def test_estimate_ipwra_rejects_string_controls(self):
        """estimate_ipwra() should reject string controls parameter."""
        with pytest.raises(TypeError, match="controls must be a list"):
            estimate_ipwra(
                data=self.data,
                y='y',
                d='d',
                controls='x1 x2'  # Wrong
            )
    
    def test_estimate_ipwra_rejects_string_propensity_controls(self):
        """estimate_ipwra() should reject string propensity_controls parameter."""
        with pytest.raises(TypeError, match="propensity_controls must be a list"):
            estimate_ipwra(
                data=self.data,
                y='y',
                d='d',
                controls=['x1', 'x2'],
                propensity_controls='x1 x2 x3'  # Wrong
            )
    
    def test_estimate_ipwra_accepts_list_parameters(self):
        """estimate_ipwra() should accept list parameters."""
        result = estimate_ipwra(
            data=self.data,
            y='y',
            d='d',
            controls=['x1', 'x2'],
            propensity_controls=['x1', 'x2', 'x3']  # Correct
        )
        assert result is not None
        assert hasattr(result, 'att')
    
    def test_estimate_ipw_rejects_string_propensity_controls(self):
        """estimate_ipw() should reject string propensity_controls parameter."""
        with pytest.raises(TypeError, match="propensity_controls must be a list"):
            estimate_ipw(
                data=self.data,
                y='y',
                d='d',
                propensity_controls='x1 x2'  # Wrong
            )
    
    def test_estimate_ipw_accepts_list_propensity_controls(self):
        """estimate_ipw() should accept list propensity_controls parameter."""
        result = estimate_ipw(
            data=self.data,
            y='y',
            d='d',
            propensity_controls=['x1', 'x2']  # Correct
        )
        assert result is not None
        assert hasattr(result, 'att')
    
    def test_estimate_psm_rejects_string_propensity_controls(self):
        """estimate_psm() should reject string propensity_controls parameter."""
        with pytest.raises(TypeError, match="propensity_controls must be a list"):
            estimate_psm(
                data=self.data,
                y='y',
                d='d',
                propensity_controls='x1 x2'  # Wrong
            )
    
    def test_estimate_psm_accepts_list_propensity_controls(self):
        """estimate_psm() should accept list propensity_controls parameter."""
        result = estimate_psm(
            data=self.data,
            y='y',
            d='d',
            propensity_controls=['x1', 'x2'],  # Correct
            n_neighbors=1
        )
        assert result is not None
        assert hasattr(result, 'att')


class TestBUG101CaliperValidation:
    """Test BUG-101: caliper parameter boundary validation."""
    
    def setup_method(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n = 100
        self.data = pd.DataFrame({
            'y': np.random.randn(n),
            'd': np.random.binomial(1, 0.5, n),
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
        })
    
    def test_estimate_psm_rejects_negative_caliper(self):
        """estimate_psm() should reject negative caliper."""
        with pytest.raises(ValueError, match="caliper must be positive"):
            estimate_psm(
                data=self.data,
                y='y',
                d='d',
                propensity_controls=['x1', 'x2'],
                caliper=-0.1  # Invalid: negative
            )
    
    def test_estimate_psm_rejects_zero_caliper(self):
        """estimate_psm() should reject zero caliper."""
        with pytest.raises(ValueError, match="caliper must be positive"):
            estimate_psm(
                data=self.data,
                y='y',
                d='d',
                propensity_controls=['x1', 'x2'],
                caliper=0.0  # Invalid: zero
            )
    
    def test_estimate_psm_rejects_non_numeric_caliper(self):
        """estimate_psm() should reject non-numeric caliper."""
        with pytest.raises(TypeError, match="caliper must be a number"):
            estimate_psm(
                data=self.data,
                y='y',
                d='d',
                propensity_controls=['x1', 'x2'],
                caliper='0.1'  # Invalid: string
            )
    
    def test_estimate_psm_accepts_positive_caliper(self):
        """estimate_psm() should accept positive caliper."""
        result = estimate_psm(
            data=self.data,
            y='y',
            d='d',
            propensity_controls=['x1', 'x2'],
            caliper=0.25  # Valid
        )
        assert result is not None
        assert hasattr(result, 'att')
    
    def test_estimate_psm_accepts_none_caliper(self):
        """estimate_psm() should accept None caliper (no caliper matching)."""
        result = estimate_psm(
            data=self.data,
            y='y',
            d='d',
            propensity_controls=['x1', 'x2'],
            caliper=None  # Valid: no caliper
        )
        assert result is not None
        assert hasattr(result, 'att')


class TestBUG101PanelUniquenessValidation:
    """Test BUG-101: (ivar, tvar) panel uniqueness validation."""
    
    def test_validate_staggered_data_rejects_duplicate_annual(self):
        """validate_staggered_data() should reject duplicate (id, year) combinations."""
        # Create data with duplicate (id, year) combinations
        data = pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2],  # Unit 1 appears twice in year 2000
            'year': [2000, 2000, 2001, 2000, 2001, 2002],
            'y': [1.0, 1.5, 2.0, 1.1, 2.1, 3.1],
            'gvar': [2001, 2001, 2001, 0, 0, 0],
        })
        
        with pytest.raises(InvalidParameterError, match="Duplicate.*observations found"):
            validate_staggered_data(
                data=data,
                gvar='gvar',
                ivar='id',
                tvar='year',
                y='y'
            )
    
    def test_validate_staggered_data_rejects_duplicate_quarterly(self):
        """validate_staggered_data() should reject duplicate (id, year, quarter) combinations."""
        # Create data with duplicate (id, year, quarter) combinations
        data = pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2],
            'year': [2000, 2000, 2000, 2000, 2000, 2001],
            'quarter': [1, 1, 2, 1, 2, 1],  # Unit 1 appears twice in 2000q1
            'y': [1.0, 1.5, 2.0, 1.1, 2.1, 3.1],
            'gvar': [2001, 2001, 2001, 0, 0, 0],
        })
        
        with pytest.raises(InvalidParameterError, match="Duplicate.*observations found"):
            validate_staggered_data(
                data=data,
                gvar='gvar',
                ivar='id',
                tvar=['year', 'quarter'],
                y='y'
            )
    
    def test_validate_staggered_data_accepts_unique_annual(self):
        """validate_staggered_data() should accept unique (id, year) combinations."""
        data = pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2],
            'year': [2000, 2001, 2002, 2000, 2001, 2002],  # All unique
            'y': [1.0, 2.0, 3.0, 1.1, 2.1, 3.1],
            'gvar': [2001, 2001, 2001, 0, 0, 0],
        })
        
        # Should not raise an error
        result = validate_staggered_data(
            data=data,
            gvar='gvar',
            ivar='id',
            tvar='year',
            y='y'
        )
        assert result is not None
        assert isinstance(result, dict)
        assert 'cohorts' in result
    
    def test_validate_staggered_data_accepts_unique_quarterly(self):
        """validate_staggered_data() should accept unique (id, year, quarter) combinations."""
        data = pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2],
            'year': [2000, 2000, 2000, 2000, 2000, 2001],
            'quarter': [1, 2, 3, 1, 2, 1],  # All unique
            'y': [1.0, 2.0, 3.0, 1.1, 2.1, 3.1],
            'gvar': [8001, 8001, 8001, 0, 0, 0],  # Quarterly format
        })
        
        # Should not raise an error
        result = validate_staggered_data(
            data=data,
            gvar='gvar',
            ivar='id',
            tvar=['year', 'quarter'],
            y='y'
        )
        assert result is not None
        assert isinstance(result, dict)


class TestBUG099WarningsImport:
    """Test BUG-099: warnings module is properly imported (code quality test)."""
    
    def test_warnings_imported_globally(self):
        """Verify warnings module is imported at module level."""
        import lwdid.results as results_module
        assert hasattr(results_module, 'warnings')
        assert results_module.warnings.__name__ == 'warnings'


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v'])
