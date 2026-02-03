"""Tests for error message consistency and detail level (DESIGN-025).

This module tests that all error messages in validation.py follow the
standardized template with:
- Problem description
- Why this matters section
- How to fix section
- Example (where applicable)

These tests verify the error message quality improvements for DESIGN-025.
"""

import numpy as np
import pandas as pd
import pytest

from lwdid.validation import (
    validate_and_prepare_data,
    validate_quarter_diversity,
    validate_quarter_coverage,
    validate_staggered_data,
    _validate_no_reserved_columns,
    _validate_required_columns,
    _validate_outcome_dtype,
    _validate_controls_dtype,
    _validate_treatment_time_invariance,
    _validate_time_invariant_controls,
    _validate_rolling_parameter,
    _create_time_index,
    _validate_time_continuity,
)
from lwdid.exceptions import (
    InvalidParameterError,
    InvalidRollingMethodError,
    InsufficientDataError,
    InsufficientQuarterDiversityError,
    TimeDiscontinuityError,
    MissingRequiredColumnError,
    NoTreatedUnitsError,
    NoControlUnitsError,
    InvalidStaggeredDataError,
)


class TestErrorMessageFormat:
    """Verify all error messages contain required sections."""
    
    def _verify_message_sections(self, error_msg: str, check_why: bool = True, 
                                  check_how: bool = True):
        """Helper to verify error message contains required sections."""
        if check_why:
            assert "Why this matters:" in error_msg or "why this matters" in error_msg.lower(), \
                f"Error message should contain 'Why this matters' section:\n{error_msg}"
        if check_how:
            assert "How to fix:" in error_msg or "how to fix" in error_msg.lower(), \
                f"Error message should contain 'How to fix' section:\n{error_msg}"


class TestTypeErrorMessages(TestErrorMessageFormat):
    """Test TypeError messages."""
    
    def test_dataframe_type_error_message(self):
        """TypeError for non-DataFrame input should have Why/How sections."""
        with pytest.raises(TypeError) as exc_info:
            validate_and_prepare_data(
                data=[1, 2, 3],
                y='y', d='d', ivar='id', tvar='year', post='post',
                rolling='demean'
            )
        
        error_msg = str(exc_info.value)
        self._verify_message_sections(error_msg)
        assert "DataFrame" in error_msg
        assert "pd.DataFrame" in error_msg or "pandas" in error_msg.lower()


class TestMissingColumnErrorMessages(TestErrorMessageFormat):
    """Test MissingRequiredColumnError messages."""
    
    def test_missing_column_error_message(self):
        """Missing column error should have Why/How sections."""
        data = pd.DataFrame({'id': [1, 2], 'year': [1, 2]})
        
        with pytest.raises(MissingRequiredColumnError) as exc_info:
            validate_and_prepare_data(
                data, y='missing_y', d='d', ivar='id',
                tvar='year', post='post', rolling='demean'
            )
        
        error_msg = str(exc_info.value)
        self._verify_message_sections(error_msg)
        assert "missing_y" in error_msg
        assert "Available columns:" in error_msg


class TestInvalidRollingMethodMessages(TestErrorMessageFormat):
    """Test InvalidRollingMethodError messages."""
    
    def test_invalid_rolling_method_message(self):
        """Invalid rolling method error should have clear error message."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3],
            'year': [1, 2, 1, 2, 1, 2],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'd': [1, 1, 0, 0, 0, 0],
            'post': [0, 1, 0, 1, 0, 1],
        })
        
        with pytest.raises(InvalidRollingMethodError) as exc_info:
            validate_and_prepare_data(
                data, y='y', d='d', ivar='id', tvar='year',
                post='post', rolling='invalid_method'
            )
        
        error_msg = str(exc_info.value)
        # InvalidRollingMethodError uses a concise format without Why/How sections
        assert "invalid_method" in error_msg
        assert "demean" in error_msg
        assert "detrend" in error_msg

    def test_quarterly_method_tvar_error_message(self):
        """Quarterly method without tvar list should have clear error message."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3],
            'year': [1, 2, 1, 2, 1, 2],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'd': [1, 1, 0, 0, 0, 0],
            'post': [0, 1, 0, 1, 0, 1],
        })
        
        with pytest.raises(InvalidRollingMethodError) as exc_info:
            validate_and_prepare_data(
                data, y='y', d='d', ivar='id', tvar='year',
                post='post', rolling='demeanq'
            )
        
        error_msg = str(exc_info.value)
        # InvalidRollingMethodError uses a concise format with example usage
        assert "demeanq" in error_msg
        # Should mention either tvar list format or season_var parameter
        assert "tvar" in error_msg.lower() or "season_var" in error_msg.lower()


class TestInsufficientDataErrorMessages(TestErrorMessageFormat):
    """Test InsufficientDataError messages."""
    
    def test_insufficient_sample_size_message(self):
        """N < 3 error should have Why/How sections."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'year': [1, 2, 1, 2],
            'y': [1.0, 2.0, 3.0, 4.0],
            'd': [1, 1, 0, 0],
            'post': [0, 1, 0, 1],
        })
        
        with pytest.raises(InsufficientDataError) as exc_info:
            validate_and_prepare_data(
                data, y='y', d='d', ivar='id', tvar='year',
                post='post', rolling='demean'
            )
        
        error_msg = str(exc_info.value)
        self._verify_message_sections(error_msg)
        assert "N=2" in error_msg or "2 units" in error_msg

    def test_no_pre_period_message(self):
        """No pre-treatment observations error should have Why/How sections."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3],
            'year': [1, 2, 1, 2, 1, 2],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'd': [1, 1, 0, 0, 0, 0],
            'post': [1, 1, 1, 1, 1, 1],  # All post
        })
        
        with pytest.raises(InsufficientDataError) as exc_info:
            validate_and_prepare_data(
                data, y='y', d='d', ivar='id', tvar='year',
                post='post', rolling='demean'
            )
        
        error_msg = str(exc_info.value)
        self._verify_message_sections(error_msg)
        assert "pre" in error_msg.lower()

    def test_no_post_period_message(self):
        """No post-treatment observations error should have Why/How sections."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3],
            'year': [1, 2, 1, 2, 1, 2],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'd': [1, 1, 0, 0, 0, 0],
            'post': [0, 0, 0, 0, 0, 0],  # All pre
        })
        
        with pytest.raises(InsufficientDataError) as exc_info:
            validate_and_prepare_data(
                data, y='y', d='d', ivar='id', tvar='year',
                post='post', rolling='demean'
            )
        
        error_msg = str(exc_info.value)
        self._verify_message_sections(error_msg)
        assert "post" in error_msg.lower()


class TestNoTreatedControlUnitMessages(TestErrorMessageFormat):
    """Test NoTreatedUnitsError and NoControlUnitsError messages."""
    
    def test_no_treated_units_message(self):
        """No treated units error should have Why/How sections."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3],
            'year': [1, 2, 1, 2, 1, 2],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'd': [0, 0, 0, 0, 0, 0],  # All control
            'post': [0, 1, 0, 1, 0, 1],
        })
        
        with pytest.raises(NoTreatedUnitsError) as exc_info:
            validate_and_prepare_data(
                data, y='y', d='d', ivar='id', tvar='year',
                post='post', rolling='demean'
            )
        
        error_msg = str(exc_info.value)
        self._verify_message_sections(error_msg)
        assert "treated" in error_msg.lower()

    def test_no_control_units_message(self):
        """No control units error should have Why/How sections."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3],
            'year': [1, 2, 1, 2, 1, 2],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'd': [1, 1, 1, 1, 1, 1],  # All treated
            'post': [0, 1, 0, 1, 0, 1],
        })
        
        with pytest.raises(NoControlUnitsError) as exc_info:
            validate_and_prepare_data(
                data, y='y', d='d', ivar='id', tvar='year',
                post='post', rolling='demean'
            )
        
        error_msg = str(exc_info.value)
        self._verify_message_sections(error_msg)
        assert "control" in error_msg.lower()


class TestDuplicateObservationMessages(TestErrorMessageFormat):
    """Test duplicate observation error messages."""
    
    def test_duplicate_annual_observations_message(self):
        """Duplicate (unit, time) error should have Why/How sections."""
        data = pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 3, 3],  # unit 1 has duplicate year=1
            'year': [1, 1, 2, 1, 2, 1, 2],  # year 1 appears twice for unit 1
            'y': [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0],
            'd': [1, 1, 1, 0, 0, 0, 0],
            'post': [0, 0, 1, 0, 1, 0, 1],
        })
        
        with pytest.raises(InvalidParameterError) as exc_info:
            validate_and_prepare_data(
                data, y='y', d='d', ivar='id', tvar='year',
                post='post', rolling='demean'
            )
        
        error_msg = str(exc_info.value)
        self._verify_message_sections(error_msg)
        assert "duplicate" in error_msg.lower() or "Duplicate" in error_msg


class TestReservedColumnMessages(TestErrorMessageFormat):
    """Test reserved column name error messages."""
    
    def test_reserved_column_message(self):
        """Reserved column error should have Why/How sections."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3],
            'year': [1, 2, 1, 2, 1, 2],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'd': [1, 1, 0, 0, 0, 0],
            'post': [0, 1, 0, 1, 0, 1],
            'd_': [1, 1, 0, 0, 0, 0],  # Reserved column
        })
        
        with pytest.raises(InvalidParameterError) as exc_info:
            validate_and_prepare_data(
                data, y='y', d='d', ivar='id', tvar='year',
                post='post', rolling='demean'
            )
        
        error_msg = str(exc_info.value)
        self._verify_message_sections(error_msg)
        assert "d_" in error_msg
        assert "reserved" in error_msg.lower() or "Reserved" in error_msg


class TestControlsValidationMessages(TestErrorMessageFormat):
    """Test control variable validation error messages."""
    
    def test_non_numeric_controls_message(self):
        """Non-numeric control error should have Why/How sections."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3],
            'year': [1, 2, 1, 2, 1, 2],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'd': [1, 1, 0, 0, 0, 0],
            'post': [0, 1, 0, 1, 0, 1],
            'region': ['A', 'A', 'B', 'B', 'C', 'C'],  # Non-numeric control
        })
        
        with pytest.raises(InvalidParameterError) as exc_info:
            validate_and_prepare_data(
                data, y='y', d='d', ivar='id', tvar='year',
                post='post', rolling='demean', controls=['region']
            )
        
        error_msg = str(exc_info.value)
        self._verify_message_sections(error_msg)
        assert "region" in error_msg
        assert "numeric" in error_msg.lower()

    def test_single_observation_unit_warning(self):
        """BUG-060-R: Units with single observation should produce warning."""
        # Create data where unit 1 has only 1 observation (cannot verify time-invariance)
        data = pd.DataFrame({
            'id': [1, 2, 2, 3, 3],  # Unit 1 has only 1 observation
            'year': [1, 1, 2, 1, 2],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0],
            'd': [1, 0, 0, 0, 0],
            'post': [0, 0, 1, 0, 1],
            'x': [1.0, 2.0, 2.0, 3.0, 3.0],  # Time-invariant control
        })
        
        with pytest.warns(UserWarning, match=r"1 unit\(s\) have only 1 observation"):
            _validate_time_invariant_controls(data, 'id', ['x'])
    
    def test_multiple_single_observation_units_warning(self):
        """BUG-060-R: Multiple units with single observation should report count."""
        # Create data where units 1 and 2 have only 1 observation each
        data = pd.DataFrame({
            'id': [1, 2, 3, 3],  # Units 1 and 2 have only 1 observation
            'year': [1, 1, 1, 2],
            'y': [1.0, 2.0, 3.0, 4.0],
            'd': [1, 0, 0, 0],
            'post': [0, 0, 0, 1],
            'x': [1.0, 2.0, 3.0, 3.0],  # Time-invariant control
        })
        
        with pytest.warns(UserWarning, match=r"2 unit\(s\) have only 1 observation"):
            _validate_time_invariant_controls(data, 'id', ['x'])
    
    def test_no_single_observation_unit_no_warning(self):
        """BUG-060-R: Units with multiple observations should not produce warning."""
        # All units have at least 2 observations
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3],
            'year': [1, 2, 1, 2, 1, 2],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'd': [1, 1, 0, 0, 0, 0],
            'post': [0, 1, 0, 1, 0, 1],
            'x': [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],  # Time-invariant control
        })
        
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _validate_time_invariant_controls(data, 'id', ['x'])
            # Filter for our specific warning
            single_obs_warnings = [
                warning for warning in w 
                if "only 1 observation" in str(warning.message)
            ]
            assert len(single_obs_warnings) == 0, "Should not warn when all units have multiple observations"

    def test_single_observation_with_time_varying_control_both_errors(self):
        """BUG-060-R: Should warn about single-obs units AND error on time-varying controls."""
        # Unit 1 has single observation (warning), unit 2 has time-varying control (error)
        data = pd.DataFrame({
            'id': [1, 2, 2, 3, 3],  # Unit 1 has only 1 observation
            'year': [1, 1, 2, 1, 2],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0],
            'd': [1, 0, 0, 0, 0],
            'post': [0, 0, 1, 0, 1],
            'x': [1.0, 2.0, 3.0, 4.0, 4.0],  # Time-varying for unit 2!
        })
        
        # Should warn about single observation AND raise error for time-varying
        with pytest.warns(UserWarning, match=r"1 unit\(s\) have only 1 observation"):
            with pytest.raises(InvalidParameterError, match="time-varying"):
                _validate_time_invariant_controls(data, 'id', ['x'])


class TestQuarterValidationMessages(TestErrorMessageFormat):
    """Test quarter validation error messages."""
    
    def test_invalid_quarter_values_message(self):
        """Invalid quarter values error should have Why/How sections."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3],
            'year': [2000, 2000, 2000, 2000, 2000, 2000],
            'quarter': [1, 5, 1, 5, 1, 5],  # Invalid quarter 5
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'd': [1, 1, 0, 0, 0, 0],
            'post': [0, 1, 0, 1, 0, 1],
        })
        
        with pytest.raises(InvalidParameterError) as exc_info:
            validate_and_prepare_data(
                data, y='y', d='d', ivar='id', tvar=['year', 'quarter'],
                post='post', rolling='demean'
            )
        
        error_msg = str(exc_info.value)
        self._verify_message_sections(error_msg)
        assert "quarter" in error_msg.lower()
        assert "5" in error_msg or "invalid" in error_msg.lower()

    def test_quarter_diversity_message(self):
        """Insufficient quarter diversity error should have Why/How sections."""
        data = pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2],
            'quarter': [1, 1, 1, 1, 2, 3],  # Unit 1 has only Q1
            'post': [0, 0, 1, 0, 0, 1],
        })
        
        with pytest.raises(InsufficientQuarterDiversityError) as exc_info:
            validate_quarter_diversity(data, 'id', 'quarter', 'post')
        
        error_msg = str(exc_info.value)
        self._verify_message_sections(error_msg)
        assert "Unit 1" in error_msg
        assert "quarter" in error_msg.lower()


class TestCommonTimingMessages(TestErrorMessageFormat):
    """Test common timing assumption error messages."""
    
    def test_common_timing_violation_message(self):
        """Common timing violation error should have Why/How sections."""
        # Create data where post varies across units at the same time
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3],
            'year': [1, 2, 1, 2, 1, 2],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'd': [1, 1, 0, 0, 0, 0],
            'post': [0, 1, 0, 0, 0, 1],  # Inconsistent post at year=2
        })
        
        with pytest.raises(InvalidParameterError) as exc_info:
            validate_and_prepare_data(
                data, y='y', d='d', ivar='id', tvar='year',
                post='post', rolling='demean'
            )
        
        error_msg = str(exc_info.value)
        self._verify_message_sections(error_msg)
        assert "common" in error_msg.lower() or "timing" in error_msg.lower()


class TestStaggeredValidationMessages(TestErrorMessageFormat):
    """Test staggered DiD validation error messages."""
    
    def test_staggered_empty_data_message(self):
        """Empty data error should have Why/How sections."""
        data = pd.DataFrame(columns=['id', 'year', 'y', 'gvar'])
        
        with pytest.raises(InvalidStaggeredDataError) as exc_info:
            validate_staggered_data(data, 'gvar', 'id', 'year', 'y')
        
        error_msg = str(exc_info.value)
        self._verify_message_sections(error_msg)
        assert "empty" in error_msg.lower()

    def test_staggered_non_numeric_gvar_message(self):
        """Non-numeric gvar error should have Why/How sections."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'year': [2000, 2001, 2000, 2001],
            'y': [1.0, 2.0, 3.0, 4.0],
            'gvar': ['2001', '2001', 'never', 'never'],  # String values
        })
        
        with pytest.raises(InvalidStaggeredDataError) as exc_info:
            validate_staggered_data(data, 'gvar', 'id', 'year', 'y')
        
        error_msg = str(exc_info.value)
        self._verify_message_sections(error_msg)
        assert "numeric" in error_msg.lower()

    def test_staggered_negative_gvar_message(self):
        """Negative gvar error should have Why/How sections."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'year': [2000, 2001, 2000, 2001],
            'y': [1.0, 2.0, 3.0, 4.0],
            'gvar': [-1, -1, 0, 0],  # Negative value
        })
        
        with pytest.raises(InvalidStaggeredDataError) as exc_info:
            validate_staggered_data(data, 'gvar', 'id', 'year', 'y')
        
        error_msg = str(exc_info.value)
        self._verify_message_sections(error_msg)
        assert "negative" in error_msg.lower()

    def test_staggered_no_cohorts_message(self):
        """No cohorts error should have Why/How sections."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'year': [2000, 2001, 2000, 2001],
            'y': [1.0, 2.0, 3.0, 4.0],
            'gvar': [0, 0, 0, 0],  # All never-treated
        })
        
        with pytest.raises(InvalidStaggeredDataError) as exc_info:
            validate_staggered_data(data, 'gvar', 'id', 'year', 'y')
        
        error_msg = str(exc_info.value)
        self._verify_message_sections(error_msg)
        assert "cohort" in error_msg.lower() or "treated" in error_msg.lower()

    def test_staggered_time_varying_gvar_message(self):
        """Time-varying gvar error should have Why/How sections."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'year': [2000, 2001, 2000, 2001],
            'y': [1.0, 2.0, 3.0, 4.0],
            'gvar': [2001, 2002, 0, 0],  # Unit 1 has varying gvar
        })
        
        with pytest.raises(InvalidStaggeredDataError) as exc_info:
            validate_staggered_data(data, 'gvar', 'id', 'year', 'y')
        
        error_msg = str(exc_info.value)
        self._verify_message_sections(error_msg)
        assert "time-invariant" in error_msg.lower() or "varying" in error_msg.lower()


class TestTreatmentTimeInvarianceMessages(TestErrorMessageFormat):
    """Test treatment time-invariance validation error messages."""
    
    def test_treatment_time_varying_message(self):
        """Time-varying treatment error should have Why/How sections."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3],
            'year': [1, 2, 1, 2, 1, 2],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'd': [0, 1, 0, 0, 0, 0],  # Unit 1 has time-varying d
            'post': [0, 1, 0, 1, 0, 1],
        })
        
        with pytest.raises(InvalidParameterError) as exc_info:
            validate_and_prepare_data(
                data, y='y', d='d', ivar='id', tvar='year',
                post='post', rolling='demean'
            )
        
        error_msg = str(exc_info.value)
        self._verify_message_sections(error_msg)
        assert "time-invariant" in error_msg.lower() or "time invariant" in error_msg.lower()


class TestOutcomeDtypeMessages(TestErrorMessageFormat):
    """Test outcome variable dtype validation error messages."""
    
    def test_non_numeric_outcome_message(self):
        """Non-numeric outcome error should have Why/How sections."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3],
            'year': [1, 2, 1, 2, 1, 2],
            'y': ['a', 'b', 'c', 'd', 'e', 'f'],  # String outcome
            'd': [1, 1, 0, 0, 0, 0],
            'post': [0, 1, 0, 1, 0, 1],
        })
        
        with pytest.raises(InvalidParameterError) as exc_info:
            validate_and_prepare_data(
                data, y='y', d='d', ivar='id', tvar='year',
                post='post', rolling='demean'
            )
        
        error_msg = str(exc_info.value)
        self._verify_message_sections(error_msg)
        assert "numeric" in error_msg.lower()
        assert "y" in error_msg


class TestTimeDiscontinuityMessages(TestErrorMessageFormat):
    """Test time discontinuity error messages."""
    
    def test_time_gap_message(self):
        """Time gap error should have detailed explanation."""
        from lwdid import lwdid
        
        # Data with a gap in years (missing 1972 and 1973)
        data = pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'year': [1970, 1971, 1974, 1970, 1971, 1974, 1970, 1971, 1974],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            'd': [1, 1, 1, 0, 0, 0, 0, 0, 0],
            'post': [0, 0, 1, 0, 0, 1, 0, 0, 1],
        })
        
        with pytest.raises(TimeDiscontinuityError) as exc_info:
            lwdid(data, 'y', 'd', 'id', 'year', 'post', 'demean')
        
        error_msg = str(exc_info.value)
        # TimeDiscontinuityError has detailed explanation but may not follow strict template
        assert "discontinuous" in error_msg.lower() or "gap" in error_msg.lower()
        assert "missing" in error_msg.lower() or "Missing" in error_msg


class TestBUG060RSingleObservationWarning:
    """Test BUG-060-R: Single observation units warning for time-invariance validation.
    
    When a unit has only 1 observation, std(ddof=1) returns NaN, so time-invariance
    cannot be verified for that unit. The fix adds a warning to inform users.
    """
    
    def test_single_observation_unit_warning_triggered(self):
        """Single-observation units should trigger a warning."""
        # Create data where unit 4 has only 1 observation
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3, 4],  # Unit 4 has only 1 obs
            'year': [1, 2, 1, 2, 1, 2, 1],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            'd': [1, 1, 0, 0, 0, 0, 0],
            'post': [0, 1, 0, 1, 0, 1, 0],
            'x': [10.0, 10.0, 20.0, 20.0, 30.0, 30.0, 40.0],  # Time-invariant control
        })
        
        with pytest.warns(UserWarning, match=r"1 unit.*only 1 observation"):
            validate_and_prepare_data(
                data, y='y', d='d', ivar='id', tvar='year',
                post='post', rolling='demean', controls=['x']
            )
    
    def test_multiple_single_observation_units_warning(self):
        """Multiple single-observation units should be counted in warning."""
        # Create data where units 4 and 5 each have only 1 observation
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3, 4, 5],  # Units 4 and 5 have only 1 obs each
            'year': [1, 2, 1, 2, 1, 2, 1, 1],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            'd': [1, 1, 0, 0, 0, 0, 0, 0],
            'post': [0, 1, 0, 1, 0, 1, 0, 0],
            'x': [10.0, 10.0, 20.0, 20.0, 30.0, 30.0, 40.0, 50.0],
        })
        
        with pytest.warns(UserWarning, match=r"2 unit.*only 1 observation"):
            validate_and_prepare_data(
                data, y='y', d='d', ivar='id', tvar='year',
                post='post', rolling='demean', controls=['x']
            )
    
    def test_no_warning_when_all_units_have_multiple_obs(self):
        """No warning should be raised when all units have multiple observations."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3],
            'year': [1, 2, 1, 2, 1, 2],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'd': [1, 1, 0, 0, 0, 0],
            'post': [0, 1, 0, 1, 0, 1],
            'x': [10.0, 10.0, 20.0, 20.0, 30.0, 30.0],
        })
        
        # Should not raise any warnings about single observation units
        import warnings as warn_module
        with warn_module.catch_warnings(record=True) as w:
            warn_module.simplefilter("always")
            validate_and_prepare_data(
                data, y='y', d='d', ivar='id', tvar='year',
                post='post', rolling='demean', controls=['x']
            )
            
            # Filter for our specific warning
            single_obs_warnings = [
                warning for warning in w 
                if "only 1 observation" in str(warning.message)
            ]
            assert len(single_obs_warnings) == 0, \
                "Should not warn about single observations when all units have multiple obs"
    
    def test_std_ddof1_returns_nan_for_single_obs(self):
        """Verify that std(ddof=1) returns NaN for single observations (root cause)."""
        import pandas as pd
        
        # Create data with one single-observation unit
        data = pd.DataFrame({
            'id': [1, 1, 2],  # Unit 2 has only 1 obs
            'x': [10.0, 10.0, 20.0],
        })
        
        within_unit_std = data.groupby('id')['x'].std()  # ddof=1 by default
        
        # Unit 1 with 2 identical values should have std=0
        assert within_unit_std[1] == 0.0, "Unit with 2 identical values should have std=0"
        
        # Unit 2 with 1 value should have std=NaN (ddof=1 requires n>1)
        assert pd.isna(within_unit_std[2]), \
            "Unit with 1 observation should have NaN std (ddof=1 requires n>1)"
    
    def test_warning_message_format(self):
        """Warning message should clearly describe the limitation."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3],  # Unit 3 has only 1 obs
            'year': [1, 2, 1, 2, 1],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0],
            'd': [1, 1, 0, 0, 0],
            'post': [0, 1, 0, 1, 0],
            'my_control': [10.0, 10.0, 20.0, 20.0, 30.0],
        })
        
        with pytest.warns(UserWarning) as record:
            validate_and_prepare_data(
                data, y='y', d='d', ivar='id', tvar='year',
                post='post', rolling='demean', controls=['my_control']
            )
        
        # Find our specific warning
        warning_messages = [str(w.message) for w in record]
        single_obs_warnings = [
            msg for msg in warning_messages 
            if "only 1 observation" in msg
        ]
        
        assert len(single_obs_warnings) >= 1, "Should have at least one single-obs warning"
        
        warning_msg = single_obs_warnings[0]
        assert "my_control" in warning_msg, "Warning should mention the control variable name"
        assert "Time-invariance cannot be verified" in warning_msg, \
            "Warning should explain the limitation"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
