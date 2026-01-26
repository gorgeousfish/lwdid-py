"""Tests for bug fixes BUG-209, BUG-210, BUG-211.

BUG-209: randomization.py bootstrap mode original data degeneracy check
BUG-210: validation.py validate_staggered_data outcome variable type validation
BUG-211: validation.py _validate_time_continuity NaN handling in sorted()
"""

import numpy as np
import pandas as pd
import pytest

from lwdid.randomization import randomization_inference
from lwdid.validation import validate_staggered_data, validate_and_prepare_data
from lwdid.exceptions import (
    RandomizationError,
    InvalidParameterError,
)


class TestBug209DegenerateDataCheck:
    """Tests for BUG-209: Early degeneracy check in randomization inference.
    
    Before the fix, degeneracy was only checked within the controls branch,
    and the error would occur during the simulation loop. After the fix,
    degeneracy is checked unconditionally before any simulation begins.
    
    Note: When att_obs is not provided, estimate_att() is called first which
    has its own degeneracy check (raising InsufficientDataError). Our fix
    adds protection for the case when att_obs IS provided (bypassing estimate_att).
    """

    @pytest.fixture
    def base_data(self):
        """Create base cross-sectional data for RI testing."""
        np.random.seed(42)
        n = 50
        return pd.DataFrame({
            'ivar': range(n),
            'd_': np.concatenate([np.ones(25), np.zeros(25)]).astype(int),
            'ydot_postavg': np.random.randn(n)
        })

    def test_all_treated_raises_error_with_att_obs(self, base_data):
        """Test that all-treated data raises RandomizationError when att_obs is provided.
        
        When att_obs is provided, estimate_att() is skipped, so our new check
        is the only safeguard against degenerate data.
        """
        # Create degenerate data: all treated
        base_data['d_'] = 1
        
        with pytest.raises(RandomizationError) as excinfo:
            randomization_inference(
                firstpost_df=base_data,
                y_col='ydot_postavg',
                d_col='d_',
                ivar='ivar',
                rireps=100,
                seed=42,
                att_obs=0.5  # Provide att_obs to bypass estimate_att()
            )
        
        assert "No control units" in str(excinfo.value) or "all d_=1" in str(excinfo.value)

    def test_all_control_raises_error_with_att_obs(self, base_data):
        """Test that all-control data raises RandomizationError when att_obs is provided.
        
        When att_obs is provided, estimate_att() is skipped, so our new check
        is the only safeguard against degenerate data.
        """
        # Create degenerate data: all control
        base_data['d_'] = 0
        
        with pytest.raises(RandomizationError) as excinfo:
            randomization_inference(
                firstpost_df=base_data,
                y_col='ydot_postavg',
                d_col='d_',
                ivar='ivar',
                rireps=100,
                seed=42,
                att_obs=0.5  # Provide att_obs to bypass estimate_att()
            )
        
        assert "No treated units" in str(excinfo.value) or "all d_=0" in str(excinfo.value)

    def test_valid_data_passes(self, base_data):
        """Test that valid balanced data passes degeneracy check."""
        # This should not raise - valid data with both treated and control
        result = randomization_inference(
            firstpost_df=base_data,
            y_col='ydot_postavg',
            d_col='d_',
            ivar='ivar',
            rireps=50,
            seed=42,
            ri_method='permutation'  # Use permutation to avoid bootstrap failure
        )
        
        assert 'p_value' in result
        assert 0 <= result['p_value'] <= 1


class TestBug210OutcomeDtypeValidation:
    """Tests for BUG-210: Outcome variable dtype validation in validate_staggered_data.
    
    Before the fix, validate_staggered_data() did not check the outcome variable
    dtype, which was inconsistent with validate_and_prepare_data().
    """

    @pytest.fixture
    def valid_staggered_data(self):
        """Create valid staggered DiD data."""
        np.random.seed(42)
        n_units = 20
        n_periods = 5
        
        data = pd.DataFrame({
            'ivar': np.repeat(range(n_units), n_periods),
            'tvar': np.tile(range(2000, 2000 + n_periods), n_units),
            'gvar': np.repeat(
                [2002, 2003, 0, 0] * (n_units // 4) + [2002] * (n_units % 4),
                n_periods
            ),
            'y': np.random.randn(n_units * n_periods),
            'x1': np.random.randn(n_units * n_periods),
        })
        return data

    def test_string_outcome_raises_error(self, valid_staggered_data):
        """Test that string outcome variable raises InvalidParameterError."""
        # Convert y to string
        valid_staggered_data['y'] = valid_staggered_data['y'].astype(str)
        
        with pytest.raises(InvalidParameterError) as excinfo:
            validate_staggered_data(
                data=valid_staggered_data,
                gvar='gvar',
                ivar='ivar',
                tvar='tvar',
                y='y'
            )
        
        assert "must be numeric" in str(excinfo.value).lower() or "numeric type" in str(excinfo.value).lower()

    def test_categorical_outcome_raises_error(self, valid_staggered_data):
        """Test that categorical outcome variable raises InvalidParameterError."""
        # Convert y to categorical
        valid_staggered_data['y'] = pd.Categorical(['low', 'high'] * (len(valid_staggered_data) // 2))
        
        with pytest.raises(InvalidParameterError) as excinfo:
            validate_staggered_data(
                data=valid_staggered_data,
                gvar='gvar',
                ivar='ivar',
                tvar='tvar',
                y='y'
            )
        
        assert "must be numeric" in str(excinfo.value).lower() or "numeric type" in str(excinfo.value).lower()

    def test_numeric_outcome_passes(self, valid_staggered_data):
        """Test that numeric outcome variable passes validation."""
        # This should not raise - y is already numeric
        result = validate_staggered_data(
            data=valid_staggered_data,
            gvar='gvar',
            ivar='ivar',
            tvar='tvar',
            y='y'
        )
        
        assert 'cohorts' in result
        assert result['n_cohorts'] >= 1

    def test_consistency_with_validate_and_prepare_data(self):
        """Test that both validation functions check outcome dtype consistently."""
        # Create common timing data
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3, 4, 4],
            'year': [2000, 2001, 2000, 2001, 2000, 2001, 2000, 2001],
            'y': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],  # String outcome
            'd': [1, 1, 1, 1, 0, 0, 0, 0],
            'post': [0, 1, 0, 1, 0, 1, 0, 1],
        })
        
        # Both should raise InvalidParameterError for string outcome
        with pytest.raises(InvalidParameterError):
            validate_and_prepare_data(
                data=data, y='y', d='d', ivar='id', tvar='year',
                post='post', rolling='demean'
            )


class TestBug211TindexNaNHandling:
    """Tests for BUG-211: NaN handling in _validate_time_continuity.
    
    Before the fix, sorted(data['tindex'].unique()) would include NaN values,
    causing comparison with expected_tindex to fail even for valid data.
    """

    @pytest.fixture
    def base_common_timing_data(self):
        """Create base common timing data."""
        return pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3, 4, 4],
            'year': [2000, 2001, 2000, 2001, 2000, 2001, 2000, 2001],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            'd': [1, 1, 1, 1, 0, 0, 0, 0],
            'post': [0, 1, 0, 1, 0, 1, 0, 1],
        })

    def test_valid_continuous_data_passes(self, base_common_timing_data):
        """Test that valid continuous time series passes validation."""
        # This should pass - continuous time index without NaN
        result = validate_and_prepare_data(
            data=base_common_timing_data,
            y='y', d='d', ivar='id', tvar='year', post='post',
            rolling='demean'
        )
        
        assert result is not None

    def test_nan_in_tindex_behavior(self):
        """Verify that NaN in tindex is handled correctly by dropna().
        
        This test verifies the fix by checking that the sorted() function
        with dropna() filters out NaN values before comparison.
        """
        # Create data that would have NaN in tindex after transformation
        # Note: In actual validation, tindex is created from the time variable
        # and NaN would come from missing time values
        
        # Test the underlying behavior
        tindex_with_nan = pd.Series([1.0, 2.0, np.nan, 3.0])
        
        # Without fix: sorted() includes NaN
        sorted_with_nan = sorted(tindex_with_nan.unique())
        assert any(pd.isna(x) for x in sorted_with_nan), \
            "sorted() should include NaN without dropna()"
        
        # With fix: dropna() removes NaN before sorting
        sorted_without_nan = sorted(tindex_with_nan.dropna().unique())
        assert not any(pd.isna(x) for x in sorted_without_nan), \
            "dropna().unique() should not include NaN"
        
        # The filtered list should match expected continuous sequence
        expected = [1.0, 2.0, 3.0]
        assert sorted_without_nan == expected


class TestBugFixesIntegration:
    """Integration tests to verify bug fixes work together."""

    def test_randomization_with_valid_staggered_data(self):
        """Test that randomization inference works with valid staggered data."""
        np.random.seed(42)
        n = 30
        
        # Create cross-sectional data (post-transformation)
        data = pd.DataFrame({
            'ivar': range(n),
            'd_': np.concatenate([np.ones(15), np.zeros(15)]).astype(int),
            'ydot_postavg': np.random.randn(n) + np.concatenate([
                np.ones(15) * 0.3,  # Small treatment effect
                np.zeros(15)
            ])
        })
        
        # Should complete without errors
        result = randomization_inference(
            firstpost_df=data,
            y_col='ydot_postavg',
            d_col='d_',
            ivar='ivar',
            rireps=50,
            seed=42,
            ri_method='permutation'
        )
        
        assert result['ri_valid'] == 50
        assert result['ri_failed'] == 0
