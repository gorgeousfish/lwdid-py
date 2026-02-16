"""Unit tests for the input validation and data preparation pipeline.

This module verifies the validation logic implemented in ``validation.py``,
which enforces the structural requirements of the Lee-Wooldridge DiD
estimator: DataFrame type checking, required column existence, rolling
method validation, sample size constraints, time index construction
(annual and quarterly), missing value handling, and string-to-numeric
ID conversion.

References
----------
Lee, S. & Wooldridge, J. M. (2026). Simple Approaches to Inference with
    DiD Estimators with Small Cross-Sectional Sample Sizes. SSRN 5325686.
Lee, S. & Wooldridge, J. M. (2025). A Simple Transformation Approach to
    DiD Estimation for Panel Data. SSRN 4516518.
"""

import numpy as np
import pandas as pd
import pytest

from lwdid import lwdid
from lwdid.validation import validate_and_prepare_data
from lwdid.exceptions import (
    InvalidParameterError,
    InvalidRollingMethodError,
    InsufficientDataError,
    TimeDiscontinuityError,
    MissingRequiredColumnError,
    NoTreatedUnitsError,
    NoControlUnitsError,
)


class TestBasicValidation:
    """Basic validation tests (V1–V3)."""

    def test_dataframe_type_check(self):
        """V1: DataFrame type check."""
        # Non-DataFrame input should raise TypeError
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            validate_and_prepare_data(
                data=[1, 2, 3],  # A plain Python list, not a DataFrame
                y='y', d='d', ivar='id', tvar='year', post='post',
                rolling='demean'
            )

    def test_required_columns(self):
        """V2: Required column existence check."""
        data = pd.DataFrame({'id': [1, 2], 'year': [1, 2]})

        # Missing outcome column
        with pytest.raises(MissingRequiredColumnError, match="not found"):
            validate_and_prepare_data(
                data, y='missing_y', d='d', ivar='id',
                tvar='year', post='post', rolling='demean'
            )

    def test_rolling_parameter_valid(self):
        """V3: Valid rolling parameter values (four allowed options)."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'year': [1, 2, 1, 2],
            'y': [1.0, 2.0, 3.0, 4.0],
            'd': [1, 1, 0, 0],
            'post': [0, 1, 0, 1],
        })

        # "demean" should pass validation and is handled in ``transformations``.
        # The other three methods should pass validation here but raise
        # NotImplementedError in ``transformations`` when not implemented.
        for method in ['demean', 'detrend', 'demeanq', 'detrendq']:
            if method in ['demeanq', 'detrendq']:
                # Quarterly methods require ``tvar`` to be a list
                with pytest.raises(InvalidRollingMethodError, match="requires either"):
                    validate_and_prepare_data(
                        data, y='y', d='d', ivar='id',
                        tvar='year',  # should be a list here
                        post='post', rolling=method
                    )
            # Other rolling-specific checks are handled in transformations.py

    def test_rolling_parameter_invalid(self):
        """V3: Invalid rolling value should raise an exception."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'year': [1, 2, 1, 2],
            'y': [1.0, 2.0, 3.0, 4.0],
            'd': [1, 1, 0, 0],
            'post': [0, 1, 0, 1],
        })
        
        with pytest.raises(InvalidRollingMethodError, match="must be one of"):
            validate_and_prepare_data(
                data, y='y', d='d', ivar='id', tvar='year',
                post='post', rolling='invalid_method'
            )


class TestSampleSizeValidation:
    """Sample size validation tests (V4)."""

    def test_minimum_sample_size(self):
        """V4: Minimum sample size N=3 should be accepted."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3],
            'year': [1, 2, 1, 2, 1, 2],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'd': [1, 1, 0, 0, 0, 0],
            'post': [0, 1, 0, 1, 0, 1],
        })
        
        data_clean, metadata = validate_and_prepare_data(
            data, y='y', d='d', ivar='id', tvar='year',
            post='post', rolling='demean'
        )
        
        assert metadata['N'] == 3
        assert metadata['N_treated'] == 1
        assert metadata['N_control'] == 2
    
    def test_insufficient_total_units(self):
        """V4: N < 3 should raise an InsufficientDataError."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'year': [1, 2, 1, 2],
            'y': [1.0, 2.0, 3.0, 4.0],
            'd': [1, 1, 0, 0],
            'post': [0, 1, 0, 1],
        })

        with pytest.raises(InsufficientDataError, match="N=2.*need N >= 3"):
            validate_and_prepare_data(
                data, y='y', d='d', ivar='id', tvar='year',
                post='post', rolling='demean'
            )

    def test_no_treated_units(self):
        """B006: No treated units should raise ``NoTreatedUnitsError``."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3],
            'year': [1, 2, 1, 2, 1, 2],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'd': [0, 0, 0, 0, 0, 0],  # all control units
            'post': [0, 1, 0, 1, 0, 1],
        })
        
        with pytest.raises(NoTreatedUnitsError, match="No treated units found"):
            validate_and_prepare_data(
                data, y='y', d='d', ivar='id', tvar='year',
                post='post', rolling='demean'
            )
    
    def test_no_control_units(self):
        """B005: No control units should raise ``NoControlUnitsError``."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3],
            'year': [1, 2, 1, 2, 1, 2],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'd': [1, 1, 1, 1, 1, 1],  # all treated units
            'post': [0, 1, 0, 1, 0, 1],
        })

        with pytest.raises(NoControlUnitsError, match="No control units found"):
            validate_and_prepare_data(
                data, y='y', d='d', ivar='id', tvar='year',
                post='post', rolling='demean'
            )


class TestTimeValidation:
    """Time index validation tests (V5–V6)."""

    def test_time_continuity_valid(self):
        """V6: Time index continuity (tpost1 = K+1) should be accepted."""
        data = pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'year': [1, 2, 3, 1, 2, 3, 1, 2, 3],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            'd': [1, 1, 1, 0, 0, 0, 0, 0, 0],
            'post': [0, 0, 1, 0, 0, 1, 0, 0, 1],
        })
        
        data_clean, metadata = validate_and_prepare_data(
            data, y='y', d='d', ivar='id', tvar='year',
            post='post', rolling='demean'
        )
        
        # K=2 (last pre-treatment period), tpost1=3 (first post-treatment period), tpost1 = K+1
        assert metadata['K'] == 2
        assert metadata['tpost1'] == 3
        assert metadata['T'] == 3
    
    def test_time_discontinuity_error(self):
        """B010: Time discontinuity test (updated after fixing the continuity check).

        The time series must be continuous with no gaps. If there is a gap in the
        observed years, ``TimeDiscontinuityError`` should be raised. This is
        important because the ``detrend`` method fits unit-specific linear trends
        :math:`y_{it} = \\alpha_i + \\beta_i t + \\varepsilon_{it}`, which rely on a
        properly ordered and continuous time index to avoid distorting trend
        estimates.
        """
        # Construct data with a gap in years (missing 1972 and 1973)
        data = pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'year': [1970, 1971, 1974, 1970, 1971, 1974, 1970, 1971, 1974],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            'd': [1, 1, 1, 0, 0, 0, 0, 0, 0],
            'post': [0, 0, 1, 0, 0, 1, 0, 0, 1],  # 1974 is post-treatment
        })

        # After BUG-017 fix: should raise TimeDiscontinuityError
        with pytest.raises(TimeDiscontinuityError, match="Time series is discontinuous"):
            lwdid(data, 'y', 'd', 'id', 'year', 'post', 'demean')
    
    def test_no_pre_period(self):
        """V5: No pre-treatment observations should raise an error."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3],
            'year': [1, 2, 1, 2, 1, 2],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'd': [1, 1, 0, 0, 0, 0],
            'post': [1, 1, 1, 1, 1, 1],  # all post-treatment
        })
        
        with pytest.raises(InsufficientDataError, match="No pre-treatment observations"):
            validate_and_prepare_data(
                data, y='y', d='d', ivar='id', tvar='year',
                post='post', rolling='demean'
            )
    
    def test_no_post_period(self):
        """V5: No post-treatment observations should raise an error."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3],
            'year': [1, 2, 1, 2, 1, 2],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'd': [1, 1, 0, 0, 0, 0],
            'post': [0, 0, 0, 0, 0, 0],  # all pre-treatment
        })
        
        with pytest.raises(InsufficientDataError, match="No post-treatment observations"):
            validate_and_prepare_data(
                data, y='y', d='d', ivar='id', tvar='year',
                post='post', rolling='demean'
            )


class TestStringIDConversion:
    """String ID conversion tests (B008)."""

    def test_string_id_conversion(self):
        """B008: String-valued IDs should be converted to numeric codes."""
        data = pd.DataFrame({
            'state': ['CA', 'CA', 'TX', 'TX', 'NY', 'NY'],
            'year': [1, 2, 1, 2, 1, 2],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'd': [1, 1, 0, 0, 0, 0],
            'post': [0, 1, 0, 1, 0, 1],
        })
        
        data_clean, metadata = validate_and_prepare_data(
            data, y='y', d='d', ivar='state', tvar='year',
            post='post', rolling='demean'
        )
        
        # ID should be converted to numeric codes (using nullable integer dtype Int64)
        assert pd.api.types.is_integer_dtype(data_clean['state'])

        # Verify that the ID mapping is stored
        assert metadata['id_mapping'] is not None
        assert 'original_to_numeric' in metadata['id_mapping']
        assert 'numeric_to_original' in metadata['id_mapping']
    
    def test_numeric_id_unchanged(self):
        """Numeric IDs should remain unchanged (no conversion applied)."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3],
            'year': [1, 2, 1, 2, 1, 2],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'd': [1, 1, 0, 0, 0, 0],
            'post': [0, 1, 0, 1, 0, 1],
        })

        data_clean, metadata = validate_and_prepare_data(
            data, y='y', d='d', ivar='id', tvar='year',
            post='post', rolling='demean'
        )

        # Numeric IDs should not be converted
        assert metadata['id_mapping'] is None
        assert data_clean['id'].dtype in [np.int32, np.int64]

    def test_missing_string_id_removed(self):
        """BUG-069: Rows with missing string IDs should be dropped, not encoded as unit 0."""
        data = pd.DataFrame({
            'state': ['CA', 'CA', np.nan, 'TX', 'TX', 'TX', 'NY', 'NY', 'NY'],
            'year': [1, 2, 3, 1, 2, 3, 1, 2, 3],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            'd': [1, 1, 1, 0, 0, 0, 0, 0, 0],
            'post': [0, 1, 1, 0, 1, 1, 0, 1, 1],
        })

        data_clean, metadata = validate_and_prepare_data(
            data, y='y', d='d', ivar='state', tvar='year',
            post='post', rolling='demean'
        )

        # Verify that rows with missing IDs are dropped
        assert len(data_clean) == 8  # original 9 rows, 1 row with missing ID dropped

        # There should be no artificial "unit 0" created
        assert 0 not in data_clean['state'].values

        # Only valid IDs (1, 2, 3 corresponding to CA, TX, NY) should remain
        unique_ids = sorted([x for x in data_clean['state'].unique() if pd.notna(x)])
        assert unique_ids == [1, 2, 3]

        # The ID mapping should not contain missing values
        assert metadata['id_mapping'] is not None
        assert len(metadata['id_mapping']['original_to_numeric']) == 3
        assert 'CA' in metadata['id_mapping']['original_to_numeric']
        assert 'TX' in metadata['id_mapping']['original_to_numeric']
        assert 'NY' in metadata['id_mapping']['original_to_numeric']


class TestTimeIndexCreation:
    """Tests for constructing the internal time index (annual and quarterly)."""

    def test_annual_time_index(self):
        """Annual data time index creation."""
        # Use continuous years to satisfy the time continuity requirement
        data = pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'year': [1970, 1971, 1972, 1970, 1971, 1972, 1970, 1971, 1972],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            'd': [1, 1, 1, 0, 0, 0, 0, 0, 0],
            'post': [0, 0, 1, 0, 0, 1, 0, 0, 1],
        })

        data_clean, metadata = validate_and_prepare_data(
            data, y='y', d='d', ivar='id', tvar='year',
            post='post', rolling='demean'
        )

        # tindex = year - 1970 + 1 = [1, 2, 3]
        assert metadata['K'] == 2  # year 1971 corresponds to tindex = 2
        assert metadata['tpost1'] == 3  # year 1972 corresponds to tindex = 3
        assert not metadata['is_quarterly']
    
    def test_quarterly_time_index(self):
        """Quarterly data time index creation."""
        # Construct quarterly data (2 years × 4 quarters)
        data = pd.DataFrame({
            'id': [1]*8 + [2]*8 + [3]*8,
            'year': [1970]*4 + [1971]*4 + [1970]*4 + [1971]*4 + [1970]*4 + [1971]*4,
            'quarter': [1, 2, 3, 4, 1, 2, 3, 4] * 3,
            'y': np.random.randn(24),
            'd': [1]*8 + [0]*8 + [0]*8,
            'post': [0]*4 + [1]*4 + [0]*4 + [1]*4 + [0]*4 + [1]*4,
        })
        
        data_clean, metadata = validate_and_prepare_data(
            data, y='y', d='d', ivar='id', tvar=['year', 'quarter'],
            post='post', rolling='demean'
        )
        
        # tq = (1970-1960)*4 + quarter = 40 + quarter
        # 1970q1: tq = 41, tindex = 1
        # 1971q4: tq = 48, tindex = 8
        assert metadata['is_quarterly']
        assert metadata['K'] == 4  # 1970q4 corresponds to tindex = 4
        assert metadata['tpost1'] == 5  # 1971q1 corresponds to tindex = 5


class TestMissingValueHandling:
    """Missing-value handling tests (B009)."""

    def test_missing_values_dropped(self):
        """B009: Observations with missing outcomes should be dropped."""
        data = pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'year': [1, 2, 3, 1, 2, 3, 1, 2, 3],
            'y': [1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],  # unit 1 has missing value
            'd': [1, 1, 1, 0, 0, 0, 0, 0, 0],
            'post': [0, 0, 1, 0, 0, 1, 0, 0, 1],
        })
        
        with pytest.warns(UserWarning, match="Dropped 1 observations"):
            data_clean, metadata = validate_and_prepare_data(
                data, y='y', d='d', ivar='id', tvar='year',
                post='post', rolling='demean'
            )

        # After dropping the missing observation there should be 8 rows left
        assert len(data_clean) == 8
        # Unit 1 should now have only 2 rows (the missing row removed)
        assert len(data_clean[data_clean['id'] == 1]) == 2


class TestQuarterDiversityValidation:
    """Quarter diversity validation tests (seasonal identification)."""

    def test_quarter_diversity_valid(self):
        """All units have at least two distinct pre-treatment quarters.

        This test verifies that ``validate_quarter_diversity`` passes silently
        when each unit satisfies the requirement of at least two different
        quarters in the pre-treatment period.
        """
        from lwdid.validation import validate_quarter_diversity

        # Construct data where all units have sufficient quarter diversity
        # Unit 1 pre-period: {Q1, Q2, Q3}; Unit 2 pre-period: {Q1, Q2}
        data = pd.DataFrame({
            'id': [1,1,1,1, 2,2,2,2],
            'quarter': [1,2,3,1, 1,2,1,2],
            'post': [0,0,0,1, 0,0,1,1],
        })
        
        # Should pass without raising an exception
        # Unit 1 pre-period: {1, 2, 3} → 3 distinct quarters ≥ 2 ✓
        # Unit 2 pre-period: {1, 2} → 2 distinct quarters ≥ 2 ✓
        validate_quarter_diversity(data, 'id', 'quarter', 'post')
    
    def test_quarter_diversity_boundary(self):
        """Boundary case: exactly two distinct quarters should be accepted.

        This test verifies the boundary condition where ``nunique(quarter) == 2``
        in the pre-treatment period for each unit.
        """
        from lwdid.validation import validate_quarter_diversity

        # Construct boundary data: each unit has exactly two distinct quarters
        data = pd.DataFrame({
            'id': [1,1,1, 2,2,2],
            'quarter': [1,2,1, 3,4,4],  # Unit 1 pre: Q1, Q2; Unit 2 pre: Q3, Q4
            'post': [0,0,1, 0,0,1],
        })
        
        # Check that the constructed data satisfy the intended structure
        # Unit 1 pre-period (post==0): quarter=[1,2] → 2 distinct quarters
        # Unit 2 pre-period (post==0): quarter=[3,4] → 2 distinct quarters

        # The check should pass (exactly two quarters meets the ≥ 2 requirement)
        validate_quarter_diversity(data, 'id', 'quarter', 'post')
    
    def test_quarter_diversity_insufficient(self):
        """Insufficient diversity: a unit with only one quarter should fail.

        This test verifies that ``InsufficientQuarterDiversityError`` is raised
        and that the error message follows the expected format.
        """
        from lwdid.validation import validate_quarter_diversity
        from lwdid.exceptions import InsufficientQuarterDiversityError
        
        # Unit 1 pre-period has only Q1; Unit 2 pre-period has Q1 and Q2
        data = pd.DataFrame({
            'id': [1,1,1, 2,2,2],
            'quarter': [1,1,1, 1,2,3],  # Unit 1 has only Q1
            'post': [0,0,1, 0,0,1],
        })
        
        # The helper should raise InsufficientQuarterDiversityError
        with pytest.raises(InsufficientQuarterDiversityError) as exc_info:
            validate_quarter_diversity(data, 'id', 'quarter', 'post')

        # Check that the error message contains the key pieces of information
        error_msg = str(exc_info.value)
        assert "Unit 1" in error_msg, "Error message should contain the unit ID"
        assert "only 1 quarter" in error_msg, "Error message should contain the actual number of quarters"
        assert "demeanq/detrendq requires" in error_msg or "≥2" in error_msg, \
            "Error message should explain the requirement"

    
    def test_quarter_diversity_error_message_details(self):
        """Error message should include the list of observed quarters.

        The error message is expected to contain a "Found quarters: [...]" field
        with the sorted list of quarters observed in the pre-treatment period.
        """
        from lwdid.validation import validate_quarter_diversity
        from lwdid.exceptions import InsufficientQuarterDiversityError
        
        # Construct data: unit 1 has only Q3 in the pre-period
        data = pd.DataFrame({
            'id': [1,1, 2,2,2],
            'quarter': [3,3, 1,2,3],
            'post': [0,1, 0,0,1],
        })
        
        with pytest.raises(InsufficientQuarterDiversityError) as exc_info:
            validate_quarter_diversity(data, 'id', 'quarter', 'post')
        
        error_msg = str(exc_info.value)
        # Verify that the error message contains the observed quarter values
        assert "[3]" in error_msg or "3" in error_msg, \
            "Error message should contain the observed quarter values"

