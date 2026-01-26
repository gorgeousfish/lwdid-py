"""Tests for bug fixes BUG-244, BUG-263, BUG-264.

BUG-244: staggered/estimators.py AI Robust SE normalization denominator issue
BUG-263: core.py ri_failed type inconsistency (int/str mixed)
BUG-264: core.py rolling parameter missing string type validation
"""

import numpy as np
import pandas as pd
import pytest
import warnings

from lwdid import lwdid
from lwdid.results import LWDIDResults


class TestBug244AIRobustSENormalization:
    """Tests for BUG-244: AI Robust SE normalization when unmatched units exist.
    
    According to Abadie-Imbens (2016), the AI Robust SE adjustment requires
    all treated units to have valid matches. When caliper restrictions cause
    some treated units to be unmatched, the code should raise an error rather
    than compute a biased SE estimate.
    
    This is consistent with Stata teffects psmatch behavior.
    """

    @pytest.fixture
    def psm_staggered_data(self):
        """Create staggered data for PSM testing."""
        np.random.seed(42)
        n_units = 60
        n_periods = 6
        
        data = []
        for i in range(n_units):
            # Create groups: 20 treated in period 4, 40 never-treated
            if i < 20:
                gvar = 2021
                # Give treated units distinct x1 values
                x1 = np.random.randn() + 2 if i < 10 else np.random.randn()
            else:
                gvar = np.inf
                x1 = np.random.randn() - 0.5
            
            x2 = np.random.randn()
            
            for t in range(n_periods):
                year = 2018 + t
                treated = (year >= gvar) if gvar != np.inf else False
                y = np.random.randn() + x1 * 0.5 + (0.5 if treated else 0)
                data.append({
                    'id': i,
                    'year': year,
                    'gvar': gvar,
                    'y': y,
                    'x1': x1,
                    'x2': x2,
                })
        
        return pd.DataFrame(data)

    def test_ai_robust_se_with_valid_matches(self, psm_staggered_data):
        """Test that AI Robust SE works when all treated units have matches.
        
        With a reasonable caliper and good overlap, all treated units should
        find matches and the estimation should succeed.
        """
        data = psm_staggered_data
        
        # This should succeed with all matches
        result = lwdid(
            data=data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            estimator='psm',
            controls=['x1', 'x2'],  # Use controls for PSM
            caliper=0.5,  # Reasonable caliper
        )
        
        # Should succeed
        assert result is not None
        assert result.att is not None


class TestBug263RIFailedTypeConsistency:
    """Tests for BUG-263: ri_failed type consistency.
    
    ri_failed should always be an integer:
    - Normal case: number of failed replications (0 or positive)
    - Error case: -1 (with error message in ri_error attribute)
    
    This ensures type safety when users check isinstance(ri_failed, int).
    """

    @pytest.fixture
    def simple_data(self):
        """Create simple panel data for testing."""
        np.random.seed(42)
        n_units = 30
        n_periods = 5
        
        data = []
        for i in range(n_units):
            treatment = 1 if i < 15 else 0
            for t in range(n_periods):
                post = 1 if t >= 3 else 0
                y = np.random.randn() + treatment * post * 0.5
                data.append({
                    'id': i,
                    'year': 2018 + t,
                    'd': treatment,
                    'post': post,
                    'y': y,
                })
        
        return pd.DataFrame(data)

    def test_ri_failed_is_integer_on_success(self, simple_data):
        """Test that ri_failed is an integer when RI succeeds."""
        result = lwdid(
            data=simple_data,
            y='y',
            d='d',
            post='post',
            ivar='id',
            tvar='year',
            rolling='demean',
            ri=True,
            rireps=50,
            seed=42,
            ri_method='permutation'
        )
        
        assert isinstance(result.ri_failed, int)
        assert result.ri_failed >= 0
        assert result.ri_error is None

    def test_ri_error_attribute_exists(self, simple_data):
        """Test that ri_error attribute exists on LWDIDResults."""
        result = lwdid(
            data=simple_data,
            y='y',
            d='d',
            post='post',
            ivar='id',
            tvar='year',
            rolling='demean',
            ri=False,  # No RI
        )
        
        # ri_error should exist as an attribute (None when RI not requested)
        assert hasattr(result, 'ri_error')
        assert result.ri_error is None

    def test_ri_failed_type_consistency_in_results_class(self, simple_data):
        """Test that LWDIDResults handles ri_failed and ri_error correctly."""
        # Use actual lwdid call to get a valid results object
        result = lwdid(
            data=simple_data,
            y='y',
            d='d',
            post='post',
            ivar='id',
            tvar='year',
            rolling='demean',
            ri=False,
        )
        
        # Test setting integer ri_failed
        result.ri_failed = 5
        assert isinstance(result.ri_failed, int)
        assert result.ri_failed == 5
        
        # Test setting -1 for failure
        result.ri_failed = -1
        result.ri_error = "Test error message"
        assert result.ri_failed == -1
        assert result.ri_error == "Test error message"


class TestBug264RollingTypeValidation:
    """Tests for BUG-264: rolling parameter string type validation.
    
    The rolling parameter should only accept string values. Passing
    non-string types (int, float, etc.) should raise TypeError with
    a clear message.
    """

    @pytest.fixture
    def simple_panel_data(self):
        """Create simple panel data for testing."""
        np.random.seed(42)
        n_units = 40
        n_periods = 6
        
        data = []
        for i in range(n_units):
            treatment = 1 if i < 20 else 0
            for t in range(n_periods):
                post = 1 if t >= 3 else 0
                y = np.random.randn() + treatment * post * 0.5
                data.append({
                    'id': i,
                    'year': 2017 + t,
                    'd': treatment,
                    'post': post,
                    'y': y,
                })
        
        return pd.DataFrame(data)

    @pytest.fixture
    def staggered_data(self):
        """Create staggered adoption data for testing."""
        np.random.seed(42)
        n_units = 30
        n_periods = 6
        
        data = []
        for i in range(n_units):
            # Assign gvar: 10 treated in period 3, 10 in period 4, 10 never-treated
            if i < 10:
                gvar = 2020
            elif i < 20:
                gvar = 2021
            else:
                gvar = np.inf
            
            for t in range(n_periods):
                year = 2018 + t
                treated = (year >= gvar) if gvar != np.inf else False
                y = np.random.randn() + (0.5 if treated else 0)
                data.append({
                    'id': i,
                    'year': year,
                    'gvar': gvar,
                    'y': y,
                })
        
        return pd.DataFrame(data)

    def test_rolling_integer_raises_typeerror_common_timing(self, simple_panel_data):
        """Test that passing integer to rolling raises TypeError in common timing mode."""
        with pytest.raises(TypeError) as excinfo:
            lwdid(
                data=simple_panel_data,
                y='y',
                d='d',
                post='post',
                ivar='id',
                tvar='year',
                rolling=1,  # Integer instead of string
            )
        
        assert "rolling" in str(excinfo.value).lower()
        assert "string" in str(excinfo.value).lower()
        assert "int" in str(excinfo.value).lower()

    def test_rolling_float_raises_typeerror_common_timing(self, simple_panel_data):
        """Test that passing float to rolling raises TypeError in common timing mode."""
        with pytest.raises(TypeError) as excinfo:
            lwdid(
                data=simple_panel_data,
                y='y',
                d='d',
                post='post',
                ivar='id',
                tvar='year',
                rolling=1.5,  # Float instead of string
            )
        
        assert "rolling" in str(excinfo.value).lower()
        assert "string" in str(excinfo.value).lower()

    def test_rolling_list_raises_typeerror_common_timing(self, simple_panel_data):
        """Test that passing list to rolling raises TypeError in common timing mode."""
        with pytest.raises(TypeError) as excinfo:
            lwdid(
                data=simple_panel_data,
                y='y',
                d='d',
                post='post',
                ivar='id',
                tvar='year',
                rolling=['demean'],  # List instead of string
            )
        
        assert "rolling" in str(excinfo.value).lower()
        assert "string" in str(excinfo.value).lower()

    def test_rolling_integer_raises_typeerror_staggered(self, staggered_data):
        """Test that passing integer to rolling raises TypeError in staggered mode."""
        with pytest.raises(TypeError) as excinfo:
            lwdid(
                data=staggered_data,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                rolling=2,  # Integer instead of string
            )
        
        assert "rolling" in str(excinfo.value).lower()
        assert "string" in str(excinfo.value).lower()

    def test_rolling_valid_string_works_common_timing(self, simple_panel_data):
        """Test that valid string rolling values work correctly in common timing mode."""
        # Test 'demean'
        result = lwdid(
            data=simple_panel_data,
            y='y',
            d='d',
            post='post',
            ivar='id',
            tvar='year',
            rolling='demean',
        )
        assert result.att is not None
        
        # Test 'detrend'
        result = lwdid(
            data=simple_panel_data,
            y='y',
            d='d',
            post='post',
            ivar='id',
            tvar='year',
            rolling='detrend',
        )
        assert result.att is not None

    def test_rolling_valid_string_works_staggered(self, staggered_data):
        """Test that valid string rolling values work correctly in staggered mode."""
        # Test 'demean'
        result = lwdid(
            data=staggered_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
        )
        assert result.att is not None
        
        # Test 'detrend'
        result = lwdid(
            data=staggered_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='detrend',
        )
        assert result.att is not None

    def test_rolling_case_insensitive(self, simple_panel_data):
        """Test that rolling parameter is case-insensitive."""
        # Test uppercase
        result = lwdid(
            data=simple_panel_data,
            y='y',
            d='d',
            post='post',
            ivar='id',
            tvar='year',
            rolling='DEMEAN',
        )
        assert result.att is not None
        
        # Test mixed case
        result = lwdid(
            data=simple_panel_data,
            y='y',
            d='d',
            post='post',
            ivar='id',
            tvar='year',
            rolling='DeMeAn',
        )
        assert result.att is not None


class TestRIErrorAttributeIntegration:
    """Integration tests for the new ri_error attribute."""

    @pytest.fixture
    def valid_panel_data(self):
        """Create valid panel data for RI testing."""
        np.random.seed(42)
        n_units = 30
        n_periods = 5
        
        data = []
        for i in range(n_units):
            treatment = 1 if i < 15 else 0
            for t in range(n_periods):
                post = 1 if t >= 3 else 0
                y = np.random.randn() + treatment * post * 0.5
                data.append({
                    'id': i,
                    'year': 2017 + t,
                    'd': treatment,
                    'post': post,
                    'y': y,
                })
        return pd.DataFrame(data)

    def test_ri_attributes_type_consistency(self, valid_panel_data):
        """Test that RI attributes have consistent types."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = lwdid(
                data=valid_panel_data,
                y='y',
                d='d',
                post='post',
                ivar='id',
                tvar='year',
                rolling='demean',
                ri=True,
                rireps=50,
                seed=42,
                ri_method='permutation'
            )
        
        # ri_failed should be an integer (whether success or failure)
        if result.ri_failed is not None:
            assert isinstance(result.ri_failed, int), \
                f"ri_failed should be int, got {type(result.ri_failed)}"
            
            # If ri_failed == -1, ri_error should have the error message
            if result.ri_failed == -1:
                assert result.ri_error is not None
                assert isinstance(result.ri_error, str)
            else:
                # Normal case: ri_error should be None
                assert result.ri_error is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
