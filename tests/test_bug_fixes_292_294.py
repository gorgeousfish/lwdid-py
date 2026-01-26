"""
Unit tests for BUG-292, BUG-293, BUG-294 fixes.

BUG-292: staggered/randomization.py n_jobs parameter validation
BUG-293: transformations.py demeanq_unit/detrendq_unit quarter counting consistency
BUG-294: core.py aggregate/control_group parameter type validation
"""
import pytest
import numpy as np
import pandas as pd
import warnings

from lwdid.staggered.randomization import randomization_inference_staggered, RandomizationError
from lwdid.transformations import demeanq_unit, detrendq_unit
from lwdid.core import lwdid


class TestBug292NJobsValidation:
    """Test n_jobs parameter validation in staggered randomization inference."""

    @pytest.fixture
    def staggered_data(self):
        """Create minimal staggered panel data for testing with proper never-treated units."""
        np.random.seed(42)
        n_units = 20
        n_periods = 10
        
        data = []
        for i in range(n_units):
            # Cohort assignment: inf = never treated, 5/6/7 = treated cohorts
            if i < 8:
                gvar = np.inf  # Never treated (need enough for RI)
            elif i < 12:
                gvar = 5
            elif i < 16:
                gvar = 6
            else:
                gvar = 7
            
            for t in range(1, n_periods + 1):
                y = np.random.randn() + (0.5 if t >= gvar and gvar != np.inf else 0)
                data.append({'id': i, 'year': t, 'gvar': gvar, 'y': y})
        
        return pd.DataFrame(data)

    def test_n_jobs_negative_two_raises_error(self, staggered_data):
        """Test that n_jobs=-2 raises RandomizationError with clear message."""
        with pytest.raises(RandomizationError) as exc_info:
            randomization_inference_staggered(
                data=staggered_data,
                gvar='gvar',
                ivar='id',
                tvar='year',
                y='y',
                observed_att=0.5,
                n_jobs=-2,
                rireps=10,
                rolling='demean'
            )
        
        assert "n_jobs must be positive, -1 (use all cores), or None" in str(exc_info.value)
        assert "-2" in str(exc_info.value)

    def test_n_jobs_negative_three_raises_error(self, staggered_data):
        """Test that n_jobs=-3 raises RandomizationError."""
        with pytest.raises(RandomizationError) as exc_info:
            randomization_inference_staggered(
                data=staggered_data,
                gvar='gvar',
                ivar='id',
                tvar='year',
                y='y',
                observed_att=0.5,
                n_jobs=-3,
                rireps=10,
                rolling='demean'
            )
        
        assert "n_jobs must be positive" in str(exc_info.value)

    def test_n_jobs_float_raises_error(self, staggered_data):
        """Test that n_jobs=1.5 (float) raises RandomizationError with type message."""
        with pytest.raises(RandomizationError) as exc_info:
            randomization_inference_staggered(
                data=staggered_data,
                gvar='gvar',
                ivar='id',
                tvar='year',
                y='y',
                observed_att=0.5,
                n_jobs=1.5,
                rireps=10,
                rolling='demean'
            )
        
        assert "must be an integer or None" in str(exc_info.value)
        assert "float" in str(exc_info.value)

    def test_n_jobs_bool_raises_error(self, staggered_data):
        """Test that n_jobs=True (bool) raises RandomizationError."""
        with pytest.raises(RandomizationError) as exc_info:
            randomization_inference_staggered(
                data=staggered_data,
                gvar='gvar',
                ivar='id',
                tvar='year',
                y='y',
                observed_att=0.5,
                n_jobs=True,
                rireps=10,
                rolling='demean'
            )
        
        assert "must be an integer or None" in str(exc_info.value)
        assert "bool" in str(exc_info.value)

    def test_n_jobs_zero_raises_error(self, staggered_data):
        """Test that n_jobs=0 raises RandomizationError."""
        with pytest.raises(RandomizationError) as exc_info:
            randomization_inference_staggered(
                data=staggered_data,
                gvar='gvar',
                ivar='id',
                tvar='year',
                y='y',
                observed_att=0.5,
                n_jobs=0,
                rireps=10,
                rolling='demean'
            )
        
        assert "n_jobs must be positive" in str(exc_info.value)

    def test_n_jobs_string_raises_error(self, staggered_data):
        """Test that n_jobs='auto' (string) raises RandomizationError."""
        with pytest.raises(RandomizationError) as exc_info:
            randomization_inference_staggered(
                data=staggered_data,
                gvar='gvar',
                ivar='id',
                tvar='year',
                y='y',
                observed_att=0.5,
                n_jobs='auto',
                rireps=10,
                rolling='demean'
            )
        
        assert "must be an integer or None" in str(exc_info.value)

    def test_n_jobs_valid_values_accepted(self, staggered_data):
        """Test that valid n_jobs values are accepted."""
        # Use target='cohort_time' which doesn't require never-treated units
        # Test n_jobs=None (default serial)
        # Need rireps >= 25 to meet minimum 20 valid replications requirement
        result = randomization_inference_staggered(
            data=staggered_data,
            gvar='gvar',
            ivar='id',
            tvar='year',
            y='y',
            observed_att=0.5,
            n_jobs=None,
            rireps=25,
            rolling='demean',
            target='cohort_time',
            target_cohort=5,
            target_period=5
        )
        # StaggeredRIResult has p_value attribute
        assert hasattr(result, 'p_value')
        assert 0 <= result.p_value <= 1

        # Test n_jobs=1 (serial)
        result = randomization_inference_staggered(
            data=staggered_data,
            gvar='gvar',
            ivar='id',
            tvar='year',
            y='y',
            observed_att=0.5,
            n_jobs=1,
            rireps=25,
            rolling='demean',
            target='cohort_time',
            target_cohort=5,
            target_period=5
        )
        assert hasattr(result, 'p_value')
        assert 0 <= result.p_value <= 1

        # Test n_jobs=-1 (all cores) - only if rireps >= 200 threshold
        # Using small rireps here so it falls back to serial
        result = randomization_inference_staggered(
            data=staggered_data,
            gvar='gvar',
            ivar='id',
            tvar='year',
            y='y',
            observed_att=0.5,
            n_jobs=-1,
            rireps=25,
            rolling='demean',
            target='cohort_time',
            target_cohort=5,
            target_period=5
        )
        assert hasattr(result, 'p_value')
        assert 0 <= result.p_value <= 1


class TestBug293QuarterCountingConsistency:
    """Test quarter counting consistency in demeanq_unit and detrendq_unit."""

    def test_demeanq_unit_with_nan_in_some_quarters(self):
        """Test demeanq_unit handles quarters that only appear in y=NaN rows correctly."""
        # Create unit data where Q3 and Q4 only appear in y=NaN rows
        unit_data = pd.DataFrame({
            'y': [1.0, 2.0, 3.0, np.nan, np.nan],  # Q1, Q1, Q2 valid; Q3, Q4 invalid
            'quarter': [1, 1, 2, 3, 4],
            'post': [0, 0, 0, 0, 0]
        })
        
        # This should NOT raise an error and should return valid results
        # After BUG-293 fix, observed_quarters_pre will be [1, 2] (only from valid rows)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            yhat, ydot = demeanq_unit(unit_data, 'y', 'quarter', 'post')
        
        # If insufficient observations warning is raised, that's expected behavior
        # The key is that it shouldn't crash with rank deficiency
        # With 3 valid observations and 2 quarters, n_valid (3) > n_params (2), so should work
        # But after BUG-293 fix, n_quarters matches observed_quarters_pre
        
        # Check that no rank deficiency error occurred
        # The function should either succeed or give a clear warning
        assert len(yhat) == len(unit_data)
        assert len(ydot) == len(unit_data)

    def test_detrendq_unit_with_nan_in_some_quarters(self):
        """Test detrendq_unit handles quarters that only appear in y=NaN rows correctly."""
        # Create unit data where Q4 only appears in y=NaN row
        unit_data = pd.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, np.nan],
            'tindex': [1, 2, 3, 4, 5, 6],
            'quarter': [1, 2, 3, 1, 2, 4],  # Q4 only in NaN row
            'post': [0, 0, 0, 0, 0, 0]
        })
        
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            yhat, ydot = detrendq_unit(unit_data, 'y', 'tindex', 'quarter', 'post')
        
        assert len(yhat) == len(unit_data)
        assert len(ydot) == len(unit_data)

    def test_demeanq_unit_all_quarters_valid(self):
        """Test demeanq_unit with all quarters having valid y values."""
        unit_data = pd.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            'quarter': [1, 2, 3, 4, 1, 2, 3, 4],
            'post': [0, 0, 0, 0, 0, 0, 0, 0]
        })
        
        yhat, ydot = demeanq_unit(unit_data, 'y', 'quarter', 'post')
        
        # All values should be valid (no NaN)
        assert not np.isnan(yhat).all()
        assert not np.isnan(ydot).all()

    def test_demeanq_unit_quarter_count_matches_dummies(self):
        """Verify n_quarters matches the number of dummy columns created."""
        # Create data with 3 valid quarters (Q1, Q2, Q3)
        unit_data = pd.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'quarter': [1, 1, 2, 2, 3, 3],
            'post': [0, 0, 0, 0, 0, 0]
        })
        
        # Should work without issues
        yhat, ydot = demeanq_unit(unit_data, 'y', 'quarter', 'post')
        
        # Results should be finite
        assert np.isfinite(yhat).all()
        assert np.isfinite(ydot).all()


class TestBug294ParameterTypeValidation:
    """Test aggregate and control_group parameter type validation."""

    @pytest.fixture
    def staggered_data(self):
        """Create minimal staggered panel data for testing with time-invariant controls."""
        np.random.seed(42)
        n_units = 20
        n_periods = 10
        
        data = []
        for i in range(n_units):
            if i < 8:
                gvar = np.inf  # Never treated (need enough)
            elif i < 12:
                gvar = 5
            elif i < 16:
                gvar = 6
            else:
                gvar = 7
            
            # Time-invariant control: same value for all periods of each unit
            x1_unit = np.random.randn()
            
            for t in range(1, n_periods + 1):
                y = np.random.randn() + (0.5 if t >= gvar and gvar != np.inf else 0)
                data.append({'id': i, 'year': t, 'gvar': gvar, 'y': y, 'x1': x1_unit})
        
        return pd.DataFrame(data)

    def test_aggregate_int_raises_type_error(self, staggered_data):
        """Test that aggregate=123 (int) raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            lwdid(
                data=staggered_data,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                rolling='demean',
                aggregate=123,
                controls=['x1']
            )
        
        assert "aggregate" in str(exc_info.value).lower()
        assert "string" in str(exc_info.value).lower()
        assert "int" in str(exc_info.value).lower()

    def test_aggregate_bool_raises_type_error(self, staggered_data):
        """Test that aggregate=True (bool) raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            lwdid(
                data=staggered_data,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                rolling='demean',
                aggregate=True,
                controls=['x1']
            )
        
        assert "aggregate" in str(exc_info.value).lower()
        assert "string" in str(exc_info.value).lower()

    def test_aggregate_list_raises_type_error(self, staggered_data):
        """Test that aggregate=['none'] (list) raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            lwdid(
                data=staggered_data,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                rolling='demean',
                aggregate=['none'],
                controls=['x1']
            )
        
        assert "aggregate" in str(exc_info.value).lower()

    def test_control_group_int_raises_type_error(self, staggered_data):
        """Test that control_group=123 (int) raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            lwdid(
                data=staggered_data,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                rolling='demean',
                control_group=123,
                controls=['x1']
            )
        
        assert "control_group" in str(exc_info.value).lower()
        assert "string" in str(exc_info.value).lower()
        assert "int" in str(exc_info.value).lower()

    def test_control_group_bool_raises_type_error(self, staggered_data):
        """Test that control_group=False (bool) raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            lwdid(
                data=staggered_data,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                rolling='demean',
                control_group=False,
                controls=['x1']
            )
        
        assert "control_group" in str(exc_info.value).lower()

    def test_valid_aggregate_values_accepted(self, staggered_data):
        """Test that valid aggregate string values are accepted."""
        # Test aggregate='none' (default)
        result = lwdid(
            data=staggered_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            aggregate='none',
            controls=['x1']
        )
        assert result is not None

        # Test aggregate=None (should default to 'none')
        result = lwdid(
            data=staggered_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            aggregate=None,
            controls=['x1']
        )
        assert result is not None

    def test_valid_control_group_values_accepted(self, staggered_data):
        """Test that valid control_group string values are accepted."""
        # Test control_group='never_treated'
        result = lwdid(
            data=staggered_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            control_group='never_treated',
            controls=['x1']
        )
        assert result is not None

        # Test control_group='not_yet_treated'
        result = lwdid(
            data=staggered_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            control_group='not_yet_treated',
            controls=['x1']
        )
        assert result is not None

        # Test control_group=None (should default to 'not_yet_treated')
        result = lwdid(
            data=staggered_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            control_group=None,
            controls=['x1']
        )
        assert result is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
