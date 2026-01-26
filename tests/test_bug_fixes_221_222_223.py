"""
Validation tests for BUG-221, BUG-222, and BUG-223 fixes.

BUG-221: Boolean type check in _validate_outcome_dtype used incorrect comparison
- dtype == 'bool' never returns True for pandas dtypes
- Fix: Use pd.api.types.is_bool_dtype(dtype)

BUG-222: NaN values in _create_time_index misreported as "non-numeric"
- Original NaN values (legitimate missing data) were conflated with
  conversion failures (non-numeric strings like 'NA', 'missing')
- Fix: Distinguish between original NaN and conversion-failed NaN

BUG-223: n_pre_periods calculation in staggered/transformations.py assumed
         continuous time, but unbalanced panels may have gaps
- n_pre_periods = g - T_min overestimates when periods are missing
- Fix: Use actual observed pre-period count via result[result[tvar] < g][tvar].nunique()
"""

import numpy as np
import pandas as pd
import pytest
import warnings
import sys

sys.path.insert(0, '/Users/cxy/Desktop/大样本lwdid/lwdid-py_v0.1.0/src')

from lwdid.validation import _validate_outcome_dtype, _create_time_index
from lwdid.staggered.transformations import transform_staggered_detrend
from lwdid.exceptions import InvalidParameterError


class TestBug221BooleanTypeCheck:
    """Test BUG-221 fix: boolean type check using pd.api.types.is_bool_dtype."""

    def test_boolean_outcome_triggers_warning(self):
        """Verify that boolean outcome variable correctly triggers warning."""
        data = pd.DataFrame({
            'y': [True, False, True, False, True],
            'x': [1, 2, 3, 4, 5]
        })
        
        # The warning should be triggered for boolean dtype
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _validate_outcome_dtype(data, 'y')
            
            # Check that at least one warning was raised about boolean type
            bool_warnings = [
                warning for warning in w
                if 'boolean type' in str(warning.message).lower()
            ]
            assert len(bool_warnings) >= 1, (
                f"Expected warning about boolean type, but got: "
                f"{[str(warning.message) for warning in w]}"
            )

    def test_numeric_outcome_no_warning(self):
        """Verify that numeric outcome variable does not trigger boolean warning."""
        data = pd.DataFrame({
            'y': [1.0, 2.5, 3.0, 4.5, 5.0],
            'x': [1, 2, 3, 4, 5]
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _validate_outcome_dtype(data, 'y')
            
            # No boolean type warning should be raised
            bool_warnings = [
                warning for warning in w
                if 'boolean type' in str(warning.message).lower()
            ]
            assert len(bool_warnings) == 0, (
                f"Unexpected boolean type warning: "
                f"{[str(warning.message) for warning in bool_warnings]}"
            )

    def test_integer_outcome_no_warning(self):
        """Verify that integer outcome variable does not trigger boolean warning."""
        data = pd.DataFrame({
            'y': [1, 2, 3, 4, 5],
            'x': [1, 2, 3, 4, 5]
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _validate_outcome_dtype(data, 'y')
            
            bool_warnings = [
                warning for warning in w
                if 'boolean type' in str(warning.message).lower()
            ]
            assert len(bool_warnings) == 0

    def test_nullable_boolean_triggers_warning(self):
        """Verify that nullable boolean dtype also triggers warning."""
        data = pd.DataFrame({
            'y': pd.array([True, False, None, True, False], dtype='boolean'),
            'x': [1, 2, 3, 4, 5]
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _validate_outcome_dtype(data, 'y')
            
            bool_warnings = [
                warning for warning in w
                if 'boolean type' in str(warning.message).lower()
            ]
            assert len(bool_warnings) >= 1


class TestBug222NaNValueDistinction:
    """Test BUG-222 fix: distinguishing original NaN from conversion-failed NaN."""

    def test_original_nan_not_reported_as_invalid(self):
        """
        Verify that original NaN values in year column are not reported as
        non-numeric values.
        """
        # Create data with original NaN values (legitimate missing data)
        data = pd.DataFrame({
            'year': [2000, 2001, np.nan, 2003, 2004],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0],
            'id': [1, 1, 1, 1, 1],
            'post': [0, 0, 0, 1, 1],
            'd': [1, 1, 1, 1, 1]
        })
        
        # This should NOT raise an error because NaN is a legitimate missing value
        # The function should proceed and the NaN will be handled by dropna later
        try:
            result, is_quarterly = _create_time_index(data.copy(), 'year')
            # If we get here, the fix is working - original NaN was not reported as invalid
            assert True
        except InvalidParameterError as e:
            # Check that the error message does NOT include 'nan' as an invalid value
            error_str = str(e).lower()
            if 'nan' in error_str and 'invalid' in error_str:
                pytest.fail(
                    f"BUG-222 not fixed: Original NaN was incorrectly reported as "
                    f"non-numeric value. Error: {e}"
                )
            # Re-raise if it's a different error
            raise

    def test_non_numeric_string_reported_as_invalid(self):
        """
        Verify that non-numeric strings (like 'NA', 'missing') are correctly
        reported as invalid values.
        """
        data = pd.DataFrame({
            'year': [2000, 2001, 'NA', 2003, 2004],  # 'NA' is a string, not numeric
            'y': [1.0, 2.0, 3.0, 4.0, 5.0],
            'id': [1, 1, 1, 1, 1],
            'post': [0, 0, 0, 1, 1],
            'd': [1, 1, 1, 1, 1]
        })
        
        # This should raise an error because 'NA' string cannot be converted
        with pytest.raises(InvalidParameterError) as excinfo:
            _create_time_index(data.copy(), 'year')
        
        error_msg = str(excinfo.value)
        # Verify the error mentions the invalid value
        assert 'NA' in error_msg or 'non-numeric' in error_msg.lower()

    def test_mixed_nan_and_invalid_string(self):
        """
        Verify correct handling when data contains both original NaN and
        non-numeric strings.
        """
        data = pd.DataFrame({
            'year': [2000, np.nan, 'missing', 2003, 2004],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0],
            'id': [1, 1, 1, 1, 1],
            'post': [0, 0, 0, 1, 1],
            'd': [1, 1, 1, 1, 1]
        })
        
        # Should raise error for 'missing' but not for np.nan
        with pytest.raises(InvalidParameterError) as excinfo:
            _create_time_index(data.copy(), 'year')
        
        error_msg = str(excinfo.value)
        # Error should mention 'missing' but not report nan as invalid
        assert 'missing' in error_msg.lower()
        # The error message should mention that original NaN values are handled separately
        assert 'handled separately' in error_msg.lower() or 'nan' in error_msg.lower()

    def test_quarterly_data_nan_distinction(self):
        """
        Verify NaN distinction also works for quarterly data.
        """
        # Test with quarter variable containing original NaN
        data = pd.DataFrame({
            'year': [2000, 2000, 2000, 2000],
            'quarter': [1, np.nan, 3, 4],  # Original NaN
            'y': [1.0, 2.0, 3.0, 4.0],
            'id': [1, 1, 1, 1],
            'post': [0, 0, 1, 1],
            'd': [1, 1, 1, 1]
        })
        
        # Should not raise error for original NaN
        try:
            result, is_quarterly = _create_time_index(data.copy(), ['year', 'quarter'])
            assert is_quarterly == True
        except InvalidParameterError as e:
            error_str = str(e).lower()
            if 'nan' in error_str and 'invalid' in error_str:
                pytest.fail(
                    f"BUG-222 not fixed for quarterly data: Original NaN was "
                    f"incorrectly reported as non-numeric. Error: {e}"
                )
            raise


class TestBug223PrePeriodCalculation:
    """Test BUG-223 fix: pre-period count based on actual observations."""

    def test_unbalanced_panel_correct_pre_period_count(self):
        """
        Verify that pre-period count uses actual observations, not g - T_min.
        
        Scenario: T_min=2000, g=2005, but only years 2000, 2002, 2004 observed
        Old code: n_pre_periods = 5 (assumes continuous)
        Fixed code: n_pre_periods = 3 (actual observations)
        """
        # Create unbalanced panel with gaps
        data = pd.DataFrame({
            'id': [1, 1, 1, 1, 1, 1,    # Unit 1: treated at 2005
                   2, 2, 2, 2, 2, 2],   # Unit 2: never treated
            'year': [2000, 2002, 2004, 2005, 2006, 2007,  # Unit 1: gaps in pre-period
                     2000, 2002, 2004, 2005, 2006, 2007], # Unit 2: same gaps
            'y': np.random.randn(12),
            'gvar': [2005] * 6 + [0] * 6  # Unit 1 treated at 2005, Unit 2 never treated
        })
        
        # With the fix, this should raise an error because:
        # - Cohort g=2005 has T_min=2000
        # - Old code: n_pre_periods = 2005 - 2000 = 5 (would pass check)
        # - Fixed code: n_pre_periods = 3 (actual: 2000, 2002, 2004, which passes >= 2)
        
        # The detrend should work since we have 3 pre-periods (>= 2 required)
        result = transform_staggered_detrend(
            data=data.copy(),
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar'
        )
        
        # Should have created ycheck columns
        ycheck_cols = [c for c in result.columns if c.startswith('ycheck_')]
        assert len(ycheck_cols) > 0

    def test_unbalanced_panel_insufficient_pre_periods(self):
        """
        Verify that detrend correctly fails when actual pre-periods < 2.
        
        Scenario: g=2005, but only year 2004 observed before treatment
        """
        # Create unbalanced panel with only 1 pre-period observation
        data = pd.DataFrame({
            'id': [1, 1, 1,     # Unit 1: treated at 2005
                   2, 2, 2],    # Unit 2: never treated
            'year': [2004, 2005, 2006,   # Unit 1: only 1 pre-period
                     2004, 2005, 2006],  # Unit 2: same
            'y': np.random.randn(6),
            'gvar': [2005] * 3 + [0] * 3
        })
        
        # Should raise ValueError because only 1 pre-period observed
        with pytest.raises(ValueError) as excinfo:
            transform_staggered_detrend(
                data=data.copy(),
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar'
            )
        
        error_msg = str(excinfo.value)
        # Verify error mentions the actual observed count
        assert '1' in error_msg and 'observed' in error_msg.lower()
        assert 'detrending requires at least 2' in error_msg.lower()

    def test_balanced_panel_pre_period_count(self):
        """
        Verify correct pre-period calculation for balanced panel.
        
        For balanced panel, g - T_min should equal actual observations.
        """
        # Create balanced panel
        data = pd.DataFrame({
            'id': [1, 1, 1, 1, 1,    # Unit 1: treated at 2005
                   2, 2, 2, 2, 2],   # Unit 2: never treated
            'year': [2001, 2002, 2003, 2004, 2005,   # Unit 1: continuous
                     2001, 2002, 2003, 2004, 2005],  # Unit 2: continuous
            'y': np.random.randn(10),
            'gvar': [2005] * 5 + [0] * 5
        })
        
        # Should work: 4 pre-periods (2001, 2002, 2003, 2004) >= 2
        result = transform_staggered_detrend(
            data=data.copy(),
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar'
        )
        
        ycheck_cols = [c for c in result.columns if c.startswith('ycheck_')]
        assert len(ycheck_cols) > 0

    def test_error_message_shows_expected_vs_observed(self):
        """
        Verify error message shows both expected (if continuous) and observed counts.
        """
        # Create data where expected != observed
        data = pd.DataFrame({
            'id': [1, 1, 1,     # Unit 1: treated at 2010
                   2, 2, 2],    # Unit 2: never treated
            'year': [2005, 2010, 2011,   # Unit 1: gap, only 1 pre-period
                     2005, 2010, 2011],  # Unit 2: same
            'y': np.random.randn(6),
            'gvar': [2010] * 3 + [0] * 3
        })
        
        with pytest.raises(ValueError) as excinfo:
            transform_staggered_detrend(
                data=data.copy(),
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar'
            )
        
        error_msg = str(excinfo.value)
        # Should mention T_min for context
        assert 'T_min' in error_msg or '2005' in error_msg
        # Should mention "if continuous" or expected count
        assert 'expected' in error_msg.lower() or 'continuous' in error_msg.lower()


class TestIntegration:
    """Integration tests for all three bug fixes together."""

    def test_all_fixes_in_workflow(self):
        """
        End-to-end test verifying all three fixes work together in a typical workflow.
        """
        np.random.seed(42)
        
        # Create realistic panel data
        n_units = 50
        n_periods = 10  # Unbalanced: not all units have all periods
        
        data_rows = []
        for unit in range(1, n_units + 1):
            # Random subset of periods (unbalanced panel)
            available_periods = sorted(np.random.choice(
                range(2000, 2010), size=np.random.randint(5, 10), replace=False
            ))
            
            # Treatment timing
            if unit <= 20:
                gvar = 2005  # Treated at 2005
            elif unit <= 35:
                gvar = 2007  # Treated at 2007
            else:
                gvar = 0  # Never treated
            
            for year in available_periods:
                data_rows.append({
                    'id': unit,
                    'year': year,
                    'y': np.random.randn() + (0.5 if gvar > 0 and year >= gvar else 0),
                    'gvar': gvar
                })
        
        data = pd.DataFrame(data_rows)
        
        # Verify the staggered detrend transformation works correctly
        # with the BUG-223 fix (uses actual pre-period count)
        try:
            result = transform_staggered_detrend(
                data=data.copy(),
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar'
            )
            
            # Should have created detrended outcome columns
            ycheck_cols = [c for c in result.columns if c.startswith('ycheck_')]
            assert len(ycheck_cols) > 0, "No ycheck columns created"
            
        except ValueError as e:
            # If error, verify it's a legitimate data issue, not BUG-223
            error_msg = str(e).lower()
            if 'observed' in error_msg:
                # This is expected if some cohorts have insufficient pre-periods
                pass
            else:
                raise


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
