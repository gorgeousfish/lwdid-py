"""Tests for the exception hierarchy and error messages.

This module verifies the inheritance structure, message formatting, and
catching behavior of all exception classes defined in ``exceptions.py``.
"""

import pytest

from lwdid.exceptions import (
    InsufficientDataError,
    InsufficientPrePeriodsError,
    InsufficientQuarterDiversityError,
    InvalidParameterError,
    InvalidRollingMethodError,
    InvalidVCETypeError,
    LWDIDError,
    MissingRequiredColumnError,
    NoControlUnitsError,
    NoTreatedUnitsError,
    TimeDiscontinuityError,
)


class TestExceptionHierarchy:
    """Tests for the inheritance structure of exception classes."""

    def test_base_exception(self):
        """Check that ``LWDIDError`` is the base class for all lwdid exceptions."""
        # LWDIDError should inherit from Exception
        assert issubclass(LWDIDError, Exception)

        # It should be possible to raise and catch LWDIDError directly
        with pytest.raises(LWDIDError):
            raise LWDIDError("Test error")

    def test_invalid_parameter_error_hierarchy(self):
        """Check the inheritance of ``InvalidParameterError`` and its subclasses."""
        # InvalidParameterError → LWDIDError → Exception
        assert issubclass(InvalidParameterError, LWDIDError)
        
        # InvalidRollingMethodError → InvalidParameterError
        assert issubclass(InvalidRollingMethodError, InvalidParameterError)
        assert issubclass(InvalidRollingMethodError, LWDIDError)
        
        # InvalidVCETypeError → InvalidParameterError
        assert issubclass(InvalidVCETypeError, InvalidParameterError)
        assert issubclass(InvalidVCETypeError, LWDIDError)
    
    def test_insufficient_data_error_hierarchy(self):
        """Check the hierarchy for ``InsufficientDataError`` and its subclasses."""
        # InsufficientDataError → LWDIDError
        assert issubclass(InsufficientDataError, LWDIDError)

        # NoTreatedUnitsError → InsufficientDataError
        assert issubclass(NoTreatedUnitsError, InsufficientDataError)
        assert issubclass(NoTreatedUnitsError, LWDIDError)

        # NoControlUnitsError → InsufficientDataError
        assert issubclass(NoControlUnitsError, InsufficientDataError)
        assert issubclass(NoControlUnitsError, LWDIDError)

        # InsufficientPrePeriodsError → InsufficientDataError
        assert issubclass(InsufficientPrePeriodsError, InsufficientDataError)
        assert issubclass(InsufficientPrePeriodsError, LWDIDError)

        # InsufficientQuarterDiversityError → InsufficientDataError
        assert issubclass(InsufficientQuarterDiversityError, InsufficientDataError)
        assert issubclass(InsufficientQuarterDiversityError, LWDIDError)

    def test_time_discontinuity_error_hierarchy(self):
        """Verify the inheritance of ``TimeDiscontinuityError``."""
        # TimeDiscontinuityError → LWDIDError
        assert issubclass(TimeDiscontinuityError, LWDIDError)
    
    def test_missing_required_column_error_hierarchy(self):
        """Verify the inheritance of ``MissingRequiredColumnError``."""
        # MissingRequiredColumnError → LWDIDError
        assert issubclass(MissingRequiredColumnError, LWDIDError)


class TestExceptionCatching:
    """Tests that exceptions can be caught at different levels of the hierarchy."""

    def test_catch_specific_exception(self):
        """A specific exception class should be caught by its own type."""
        with pytest.raises(InvalidRollingMethodError):
            raise InvalidRollingMethodError("Invalid rolling method")

    def test_catch_base_exception(self):
        """The base class should catch all derived exceptions."""
        # Use LWDIDError to catch any subclass
        with pytest.raises(LWDIDError):
            raise NoTreatedUnitsError("No treated units")

        with pytest.raises(LWDIDError):
            raise TimeDiscontinuityError("Time discontinuity")

    def test_catch_intermediate_class(self):
        """Intermediate classes should catch their own subclasses."""
        # Catch subclasses via InsufficientDataError
        with pytest.raises(InsufficientDataError):
            raise NoControlUnitsError("No control units")
        
        with pytest.raises(InsufficientDataError):
            raise InsufficientPrePeriodsError("Insufficient pre-periods")
        
        with pytest.raises(InsufficientDataError):
            raise InsufficientQuarterDiversityError("Insufficient quarter diversity")


class TestExceptionMessages:
    """Tests for exception message formatting and content."""
    
    def test_insufficient_quarter_diversity_message(self):
        """Verify the message format of ``InsufficientQuarterDiversityError``.

        The message should contain: unit ID, actual quarter count, the
        requirement, and the list of observed quarters.
        """
        # Construct an exception with a detailed message
        unit_id = 5
        unique_quarters = 1
        found_quarters = [3]
        
        error = InsufficientQuarterDiversityError(
            f"Unit {unit_id} has only {unique_quarters} quarter(s) in pre-period. "
            f"demeanq/detrendq requires ≥2 different quarters per unit to identify seasonal effects. "
            f"Found quarters: {sorted(found_quarters)}"
        )
        
        error_msg = str(error)
        
        # Verify all key information is present
        assert "Unit 5" in error_msg, "Should contain the unit ID"
        assert "only 1 quarter" in error_msg, "Should contain the actual quarter count"
        assert "demeanq/detrendq requires" in error_msg, "Should state the requirement"
        assert "[3]" in error_msg, "Should contain the observed quarter values"
    
    def test_time_discontinuity_message(self):
        """Verify the message format of ``TimeDiscontinuityError``.

        The message should include the values of K, tpost1, and the expected
        relationship between them.
        """
        K = 19
        tpost1 = 22
        
        error = TimeDiscontinuityError(
            f"Time discontinuity: last pre-period K={K}, "
            f"first post-period tpost1={tpost1}. "
            f"Expected tpost1 = K+1 = {K+1}. "
            f"Check 'post' variable definition."
        )
        
        error_msg = str(error)
        
        assert "K=19" in error_msg, "Should contain the K value"
        assert "tpost1=22" in error_msg, "Should contain the tpost1 value"
        assert "Expected tpost1 = K+1 = 20" in error_msg, "Should contain the expected value"
        assert "Check 'post'" in error_msg, "Should suggest checking the post variable"
    
    def test_no_treated_units_message(self):
        """Verify the message format of ``NoTreatedUnitsError``."""
        error = NoTreatedUnitsError("No treated units found (d==1).")
        
        error_msg = str(error)
        assert "No treated units found" in error_msg
        assert "d==1" in error_msg
    
    def test_no_control_units_message(self):
        """Verify the message format of ``NoControlUnitsError``."""
        error = NoControlUnitsError("No control units found (d==0).")
        
        error_msg = str(error)
        assert "No control units found" in error_msg
        assert "d==0" in error_msg
    
    def test_insufficient_pre_periods_message(self):
        """InsufficientPrePeriodsError message should explain the T0 condition."""
        error = InsufficientPrePeriodsError(
            "rolling('detrend') requires at least 2 pre-treatment periods. Found: T0=1"
        )
        
        error_msg = str(error)
        assert "detrend" in error_msg
        assert "at least 2 pre-treatment periods" in error_msg
        assert "T0=1" in error_msg


class TestExceptionUsageInCode:
    """Tests for raising and catching exceptions in representative code paths."""

    def test_exception_can_be_raised_and_caught(self):
        """Exceptions should be raisable and catchable in representative code paths."""
        from lwdid.exceptions import InsufficientQuarterDiversityError
        
        # Simulate real-world usage in application code
        def check_quarter_diversity(unique_quarters: int):
            if unique_quarters < 2:
                raise InsufficientQuarterDiversityError(
                    f"Unit has only {unique_quarters} quarter(s) in pre-period."
                )
        
        # Normal cases: should not raise
        check_quarter_diversity(2)
        check_quarter_diversity(4)
        
        # Error cases: should raise
        with pytest.raises(InsufficientQuarterDiversityError):
            check_quarter_diversity(1)
        
        with pytest.raises(InsufficientQuarterDiversityError):
            check_quarter_diversity(0)

