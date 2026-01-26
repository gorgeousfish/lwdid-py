"""Tests for DESIGN-017: NoNeverTreatedError exception usage.

Validates that NoNeverTreatedError is correctly raised when:
1. aggregate='cohort' or 'overall' but no NT units exist (core.py)
2. Cohort effects estimated without NT units (aggregation.py)
3. Overall effects estimated without NT units (aggregation.py)
4. control_group='never_treated' but no NT units (estimators.py)
"""

import numpy as np
import pandas as pd
import pytest

from lwdid import lwdid
from lwdid.exceptions import (
    InsufficientDataError,
    LWDIDError,
    NoNeverTreatedError,
)
from lwdid.staggered.aggregation import (
    aggregate_to_cohort,
    aggregate_to_overall,
)
from lwdid.staggered.estimators import build_subsample_for_ps_estimation


class TestNoNeverTreatedErrorHierarchy:
    """Tests for NoNeverTreatedError inheritance structure."""

    def test_inherits_from_insufficient_data_error(self):
        """NoNeverTreatedError should inherit from InsufficientDataError."""
        assert issubclass(NoNeverTreatedError, InsufficientDataError)

    def test_inherits_from_lwdid_error(self):
        """NoNeverTreatedError should ultimately inherit from LWDIDError."""
        assert issubclass(NoNeverTreatedError, LWDIDError)

    def test_inherits_from_exception(self):
        """NoNeverTreatedError should be a valid Exception."""
        assert issubclass(NoNeverTreatedError, Exception)

    def test_can_raise_and_catch_directly(self):
        """NoNeverTreatedError should be raisable and catchable."""
        with pytest.raises(NoNeverTreatedError):
            raise NoNeverTreatedError("Test message")

    def test_catch_via_insufficient_data_error(self):
        """NoNeverTreatedError should be catchable via InsufficientDataError."""
        with pytest.raises(InsufficientDataError):
            raise NoNeverTreatedError("No NT units")

    def test_catch_via_lwdid_error(self):
        """NoNeverTreatedError should be catchable via LWDIDError."""
        with pytest.raises(LWDIDError):
            raise NoNeverTreatedError("No NT units")


class TestNoNeverTreatedErrorMessage:
    """Tests for NoNeverTreatedError message content."""

    def test_message_preserved(self):
        """Exception message should be preserved correctly."""
        message = "数据中没有never treated单位"
        error = NoNeverTreatedError(message)
        assert message in str(error)

    def test_multiline_message(self):
        """Exception should handle multiline messages."""
        message = """无法估计cohort效应: 数据中没有never treated单位。
原因: cohort效应需要NT单位作为统一参照基准。
建议: 使用aggregate='none'。"""
        error = NoNeverTreatedError(message)
        assert "cohort效应" in str(error)
        assert "NT单位" in str(error)


class TestNoNeverTreatedErrorInCore:
    """Tests that core.py raises NoNeverTreatedError correctly."""

    @pytest.fixture
    def data_no_nt(self):
        """Create staggered data with no never-treated units."""
        np.random.seed(42)
        n_units = 30
        n_periods = 10
        
        # All units are eventually treated (cohorts: 5, 6, 7)
        unit_cohorts = np.random.choice([5, 6, 7], size=n_units)
        
        data = []
        for i in range(n_units):
            cohort = unit_cohorts[i]
            for t in range(1, n_periods + 1):
                treated = 1 if t >= cohort else 0
                y = 1.0 + 0.5 * treated + np.random.normal(0, 0.1)
                data.append({
                    'id': i,
                    'year': t,
                    'y': y,
                    'gvar': cohort
                })
        
        return pd.DataFrame(data)

    def test_lwdid_raises_no_never_treated_error_for_cohort(self, data_no_nt):
        """lwdid should raise NoNeverTreatedError when aggregate='cohort' without NT."""
        with pytest.raises(NoNeverTreatedError) as exc_info:
            lwdid(
                data_no_nt,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                rolling='demean',
                aggregate='cohort'
            )
        
        # Check error message mentions cohort
        assert "cohort" in str(exc_info.value).lower() or "never" in str(exc_info.value).lower()

    def test_lwdid_raises_no_never_treated_error_for_overall(self, data_no_nt):
        """lwdid should raise NoNeverTreatedError when aggregate='overall' without NT."""
        with pytest.raises(NoNeverTreatedError) as exc_info:
            lwdid(
                data_no_nt,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                rolling='demean',
                aggregate='overall'
            )
        
        # Check error message mentions overall or never treated
        error_msg = str(exc_info.value).lower()
        assert "overall" in error_msg or "never" in error_msg

    def test_lwdid_aggregate_none_works_without_nt(self, data_no_nt):
        """lwdid with aggregate='none' should work without NT units (using not_yet_treated)."""
        # This should NOT raise NoNeverTreatedError
        result = lwdid(
            data_no_nt,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            aggregate='none',
            control_group='not_yet_treated'
        )
        assert result is not None


class TestNoNeverTreatedErrorInAggregation:
    """Tests that aggregation.py raises NoNeverTreatedError correctly."""

    @pytest.fixture
    def transformed_data_no_nt(self):
        """Create transformed staggered data without NT units."""
        np.random.seed(42)
        n_units = 20
        n_periods = 7
        
        # All units are treated (cohorts: 3, 4)
        unit_cohorts = np.random.choice([3, 4], size=n_units)
        
        data = []
        for i in range(n_units):
            cohort = unit_cohorts[i]
            for t in range(1, n_periods + 1):
                row = {
                    'id': i,
                    'year': t,
                    'gvar': cohort,
                }
                # Add transformed columns for post periods
                for g in [3, 4]:
                    for r in range(g, n_periods + 1):
                        col = f'ydot_g{g}_r{r}'
                        if t == r and i < 10 and cohort == g:
                            row[col] = np.random.normal(0, 1)
                        else:
                            row[col] = np.nan
                data.append(row)
        
        df = pd.DataFrame(data)
        return df

    def test_aggregate_to_cohort_raises_error(self, transformed_data_no_nt):
        """aggregate_to_cohort should raise NoNeverTreatedError without NT."""
        with pytest.raises(NoNeverTreatedError):
            aggregate_to_cohort(
                data_transformed=transformed_data_no_nt,
                gvar='gvar',
                ivar='id',
                tvar='year',
                cohorts=[3, 4],
                T_max=6,
                transform_type='demean'
            )

    def test_aggregate_to_overall_raises_error(self, transformed_data_no_nt):
        """aggregate_to_overall should raise NoNeverTreatedError without NT."""
        with pytest.raises(NoNeverTreatedError):
            aggregate_to_overall(
                data_transformed=transformed_data_no_nt,
                gvar='gvar',
                ivar='id',
                tvar='year',
                transform_type='demean'
            )


class TestNoNeverTreatedErrorInEstimators:
    """Tests that estimators.py raises NoNeverTreatedError correctly."""

    @pytest.fixture
    def staggered_data_no_nt(self):
        """Create staggered panel data without NT units for estimator testing."""
        np.random.seed(42)
        n_units = 40
        n_periods = 8
        
        # All units eventually treated
        unit_cohorts = np.random.choice([4, 5, 6], size=n_units)
        
        data = []
        for i in range(n_units):
            cohort = unit_cohorts[i]
            x1 = np.random.normal(0, 1)
            for t in range(1, n_periods + 1):
                treated = 1 if t >= cohort else 0
                y = 1.0 + 0.5 * treated + 0.3 * x1 + np.random.normal(0, 0.2)
                data.append({
                    'id': i,
                    'year': t,
                    'y': y,
                    'gvar': cohort,
                    'x1': x1
                })
        
        return pd.DataFrame(data)

    def test_build_subsample_raises_error(self, staggered_data_no_nt):
        """build_subsample_for_ps_estimation should raise NoNeverTreatedError with never_treated strategy but no NT."""
        with pytest.raises(NoNeverTreatedError) as exc_info:
            build_subsample_for_ps_estimation(
                data=staggered_data_no_nt,
                gvar_col='gvar',
                ivar_col='id',
                cohort_g=5,
                period_r=6,
                control_group='never_treated'  # Explicitly request NT control
            )
        
        # Should mention never treated
        assert "never" in str(exc_info.value).lower()


class TestNoNeverTreatedErrorNotRaisedWithNT:
    """Tests that NoNeverTreatedError is NOT raised when NT units exist."""

    @pytest.fixture
    def data_with_nt(self):
        """Create staggered data WITH never-treated units."""
        np.random.seed(42)
        n_units = 40
        n_periods = 8
        
        # Some units never treated (gvar = 0), others treated
        cohorts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # 10 NT units
                   5, 5, 5, 5, 5, 5, 5, 5, 5, 5,  # 10 cohort 5
                   6, 6, 6, 6, 6, 6, 6, 6, 6, 6,  # 10 cohort 6
                   7, 7, 7, 7, 7, 7, 7, 7, 7, 7]  # 10 cohort 7
        
        data = []
        for i in range(n_units):
            cohort = cohorts[i]
            for t in range(1, n_periods + 1):
                if cohort == 0:
                    treated = 0  # Never treated
                else:
                    treated = 1 if t >= cohort else 0
                y = 1.0 + 0.5 * treated + np.random.normal(0, 0.1)
                data.append({
                    'id': i,
                    'year': t,
                    'y': y,
                    'gvar': cohort
                })
        
        return pd.DataFrame(data)

    def test_lwdid_cohort_works_with_nt(self, data_with_nt):
        """lwdid with aggregate='cohort' should work when NT units exist."""
        result = lwdid(
            data_with_nt,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            aggregate='cohort'
        )
        assert result is not None
        assert result.att is not None

    def test_lwdid_overall_works_with_nt(self, data_with_nt):
        """lwdid with aggregate='overall' should work when NT units exist."""
        result = lwdid(
            data_with_nt,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            aggregate='overall'
        )
        assert result is not None
        assert result.att is not None


class TestExceptionTypePrecision:
    """Tests that the correct exception type is raised (not ValueError)."""

    def test_not_value_error_for_no_nt(self):
        """The exception should be NoNeverTreatedError, not ValueError."""
        # Create data without NT
        np.random.seed(42)
        data = pd.DataFrame({
            'id': [0] * 5 + [1] * 5,
            'year': list(range(1, 6)) * 2,
            'y': np.random.randn(10),
            'gvar': [3] * 5 + [4] * 5  # All treated, no NT
        })
        
        # Should NOT be caught by ValueError-only handler
        caught_as_value_error = False
        caught_as_no_nt_error = False
        
        try:
            lwdid(
                data,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                rolling='demean',
                aggregate='cohort'
            )
        except NoNeverTreatedError:
            caught_as_no_nt_error = True
        except ValueError:
            caught_as_value_error = True
        
        assert caught_as_no_nt_error, "Should raise NoNeverTreatedError"
        assert not caught_as_value_error, "Should NOT raise generic ValueError"


class TestUserCanCatchSpecifically:
    """Tests that users can catch NoNeverTreatedError specifically."""

    def test_user_workflow_catch_specific(self):
        """Demonstrate user workflow: catch NoNeverTreatedError specifically."""
        np.random.seed(42)
        data = pd.DataFrame({
            'id': [0] * 5 + [1] * 5,
            'year': list(range(1, 6)) * 2,
            'y': np.random.randn(10),
            'gvar': [3] * 5 + [4] * 5
        })
        
        # User code pattern: catch specific exception
        fallback_used = False
        try:
            result = lwdid(
                data,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                rolling='demean',
                aggregate='overall'
            )
        except NoNeverTreatedError:
            # User's fallback: use aggregate='none' instead
            fallback_used = True
            result = lwdid(
                data,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                rolling='demean',
                aggregate='none',
                control_group='not_yet_treated'
            )
        
        assert fallback_used, "NoNeverTreatedError should trigger fallback"
        assert result is not None, "Fallback should succeed"


class TestNumericalValidation:
    """Numerical validation tests for NoNeverTreatedError scenarios."""

    def test_boundary_nt_count(self):
        """Test boundary case: exactly 1 NT unit should not raise (but may warn)."""
        np.random.seed(42)
        n_periods = 8
        
        # 1 NT unit, 9 treated units
        data = []
        # NT unit
        for t in range(1, n_periods + 1):
            data.append({'id': 0, 'year': t, 'y': np.random.randn(), 'gvar': 0})
        # Treated units
        for i in range(1, 10):
            cohort = 5
            for t in range(1, n_periods + 1):
                treated = 1 if t >= cohort else 0
                data.append({
                    'id': i, 'year': t, 
                    'y': 1 + 0.5 * treated + np.random.randn() * 0.1,
                    'gvar': cohort
                })
        
        df = pd.DataFrame(data)
        
        # Should NOT raise NoNeverTreatedError (1 NT unit exists)
        # May issue warning about few NT units
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                df,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                rolling='demean',
                aggregate='cohort'
            )
        
        assert result is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
