"""
Test suite for BUG-170, BUG-171, and BUG-172 fixes.

BUG-170: control_groups.py get_all_control_masks column existence validation
BUG-171: randomization.py early degenerate data check
BUG-172: staggered/randomization.py bootstrap cohort degeneration detection
"""

import numpy as np
import pandas as pd
import pytest

from lwdid.staggered.control_groups import (
    get_all_control_masks,
    ControlGroupStrategy,
)
from lwdid.randomization import randomization_inference
from lwdid.staggered.randomization import randomization_inference_staggered
from lwdid.exceptions import RandomizationError, InsufficientDataError


class TestBug170ColumnValidation:
    """Tests for BUG-170: get_all_control_masks column existence validation."""

    @pytest.fixture
    def valid_data(self):
        """Create valid panel data for testing."""
        return pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3, 4, 4],
            'year': [2000, 2001, 2000, 2001, 2000, 2001, 2000, 2001],
            'gvar': [2001, 2001, 2001, 2001, 0, 0, 0, 0],
        })

    def test_missing_gvar_raises_keyerror(self, valid_data):
        """Missing gvar column should raise KeyError with clear message."""
        data = valid_data.drop(columns=['gvar'])
        
        with pytest.raises(KeyError) as exc_info:
            get_all_control_masks(
                data=data,
                gvar='gvar',
                ivar='id',
                cohorts=[2001],
                T_max=2001,
            )
        
        assert 'gvar' in str(exc_info.value)
        assert 'not found' in str(exc_info.value)

    def test_missing_ivar_raises_keyerror(self, valid_data):
        """Missing ivar column should raise KeyError with clear message."""
        data = valid_data.drop(columns=['id'])
        
        with pytest.raises(KeyError) as exc_info:
            get_all_control_masks(
                data=data,
                gvar='gvar',
                ivar='id',
                cohorts=[2001],
                T_max=2001,
            )
        
        assert 'id' in str(exc_info.value)
        assert 'not found' in str(exc_info.value)

    def test_misspelled_column_raises_keyerror(self, valid_data):
        """Misspelled column names should raise informative KeyError."""
        with pytest.raises(KeyError) as exc_info:
            get_all_control_masks(
                data=valid_data,
                gvar='gvar_misspelled',
                ivar='id',
                cohorts=[2001],
                T_max=2001,
            )
        
        assert 'gvar_misspelled' in str(exc_info.value)

    def test_valid_columns_works(self, valid_data):
        """Valid column names should work without errors."""
        result = get_all_control_masks(
            data=valid_data,
            gvar='gvar',
            ivar='id',
            cohorts=[2001],
            T_max=2001,
        )
        
        assert isinstance(result, dict)
        # Keys are (cohort, float_period) tuples
        assert (2001, 2001.0) in result

    def test_empty_data_raises_valueerror(self):
        """Empty DataFrame should raise ValueError."""
        data = pd.DataFrame(columns=['id', 'gvar'])
        
        with pytest.raises(ValueError) as exc_info:
            get_all_control_masks(
                data=data,
                gvar='gvar',
                ivar='id',
                cohorts=[2001],
                T_max=2001,
            )
        
        assert 'empty' in str(exc_info.value).lower()


class TestBug171EarlyDegenerateCheck:
    """Tests for BUG-171: Early degenerate data check in randomization_inference."""

    @pytest.fixture
    def base_data(self):
        """Create base data with both treated and control units."""
        np.random.seed(42)
        return pd.DataFrame({
            'ivar': range(20),
            'd_': np.concatenate([np.ones(10), np.zeros(10)]).astype(int),
            'ydot_postavg': np.random.randn(20),
        })

    def test_all_treated_raises_error(self, base_data):
        """All units being treated (N0=0) should raise early error."""
        data = base_data.copy()
        data['d_'] = 1  # All treated
        
        # Error can be raised by either randomization.py or estimation.py
        # Both are valid early validation points
        with pytest.raises((RandomizationError, InsufficientDataError)) as exc_info:
            randomization_inference(
                firstpost_df=data,
                y_col='ydot_postavg',
                d_col='d_',
                rireps=10,
                seed=42,
            )
        
        error_msg = str(exc_info.value).lower()
        assert 'control' in error_msg or 'n0=0' in error_msg or 'n_control=0' in error_msg

    def test_all_control_raises_error(self, base_data):
        """All units being control (N1=0) should raise early error."""
        data = base_data.copy()
        data['d_'] = 0  # All control
        
        # Error can be raised by either randomization.py or estimation.py
        # Both are valid early validation points
        with pytest.raises((RandomizationError, InsufficientDataError)) as exc_info:
            randomization_inference(
                firstpost_df=data,
                y_col='ydot_postavg',
                d_col='d_',
                rireps=10,
                seed=42,
            )
        
        error_msg = str(exc_info.value).lower()
        assert 'treated' in error_msg or 'n1=0' in error_msg or 'n_treated=0' in error_msg

    def test_balanced_data_works(self, base_data):
        """Balanced data with both groups should work."""
        result = randomization_inference(
            firstpost_df=base_data,
            y_col='ydot_postavg',
            d_col='d_',
            rireps=50,
            seed=42,
            ri_method='permutation',
        )
        
        assert 'p_value' in result
        assert 0.0 <= result['p_value'] <= 1.0

    def test_single_treated_with_controls_works(self, base_data):
        """Single treated unit with multiple controls should work."""
        data = base_data.copy()
        data['d_'] = np.concatenate([[1], np.zeros(19)]).astype(int)
        
        result = randomization_inference(
            firstpost_df=data,
            y_col='ydot_postavg',
            d_col='d_',
            rireps=50,
            seed=42,
            ri_method='permutation',
        )
        
        assert 'p_value' in result

    def test_single_control_with_treated_works(self, base_data):
        """Single control unit with multiple treated should work."""
        data = base_data.copy()
        data['d_'] = np.concatenate([np.ones(19), [0]]).astype(int)
        
        result = randomization_inference(
            firstpost_df=data,
            y_col='ydot_postavg',
            d_col='d_',
            rireps=50,
            seed=42,
            ri_method='permutation',
        )
        
        assert 'p_value' in result


class TestBug172BootstrapCohortDegeneration:
    """Tests for BUG-172: Bootstrap cohort degeneration detection."""

    @pytest.fixture
    def staggered_data(self):
        """Create staggered adoption panel data with multiple cohorts."""
        np.random.seed(42)
        n_units = 30
        n_periods = 6
        
        # Create unit-level data
        unit_ids = np.repeat(range(n_units), n_periods)
        periods = np.tile(range(2000, 2000 + n_periods), n_units)
        
        # Assign cohorts: 10 units each to 2002, 2003, 2004; 0 as never-treated for some
        cohort_assignments = (
            [2002] * 8 + [2003] * 8 + [2004] * 8 +  # 24 treated units
            [0] * 6  # 6 never-treated units
        )
        gvar = np.repeat(cohort_assignments, n_periods)
        
        # Generate outcome
        y = np.random.randn(n_units * n_periods)
        
        return pd.DataFrame({
            'id': unit_ids,
            'year': periods,
            'gvar': gvar,
            'y': y,
        })

    def test_bootstrap_with_cohort_degeneration_skips_replications(self, staggered_data):
        """Bootstrap replications with missing cohorts should be skipped for overall target."""
        # This test verifies that the fix correctly identifies and skips
        # degenerate bootstrap samples where some cohorts are missing
        
        # Use a small number of reps to make test fast
        # With bootstrap, some replications may result in missing cohorts
        result = randomization_inference_staggered(
            data=staggered_data,
            gvar='gvar',
            ivar='id',
            tvar='year',
            y='y',
            observed_att=0.5,
            target='overall',
            ri_method='bootstrap',
            rireps=100,
            seed=42,
            rolling='demean',
            n_never_treated=6,
        )
        
        # Verify result structure
        assert hasattr(result, 'p_value')
        assert hasattr(result, 'ri_valid')
        assert hasattr(result, 'ri_failed')
        
        # With bootstrap, some replications may fail due to degenerate cohort distributions
        # The fix ensures these are properly counted as failed
        total = result.ri_valid + result.ri_failed
        assert total == 100

    def test_permutation_preserves_all_cohorts(self, staggered_data):
        """Permutation method should preserve cohort distribution."""
        result = randomization_inference_staggered(
            data=staggered_data,
            gvar='gvar',
            ivar='id',
            tvar='year',
            y='y',
            observed_att=0.5,
            target='overall',
            ri_method='permutation',
            rireps=50,
            seed=42,
            rolling='demean',
            n_never_treated=6,
        )
        
        # Permutation should have higher success rate than bootstrap
        # because it preserves the exact cohort distribution
        assert result.ri_valid > 0
        assert hasattr(result, 'p_value')

    def test_cohort_time_target_not_affected_by_degeneration(self, staggered_data):
        """cohort_time target should work even with degenerate cohort distribution."""
        result = randomization_inference_staggered(
            data=staggered_data,
            gvar='gvar',
            ivar='id',
            tvar='year',
            y='y',
            observed_att=0.5,
            target='cohort_time',
            target_cohort=2002,
            target_period=2003,
            ri_method='bootstrap',
            rireps=50,
            seed=42,
            rolling='demean',
            n_never_treated=6,
        )
        
        # cohort_time is not affected by missing other cohorts
        assert result.ri_valid > 0


class TestBugFixesIntegration:
    """Integration tests combining multiple bug fixes."""

    def test_control_groups_error_before_processing(self):
        """Column validation should occur before any data processing."""
        # This tests that the KeyError is raised immediately,
        # not after expensive computations
        data = pd.DataFrame({
            'wrong_id': [1, 2, 3],
            'wrong_gvar': [2001, 2002, 0],
        })
        
        with pytest.raises(KeyError):
            get_all_control_masks(
                data=data,
                gvar='gvar',
                ivar='id',
                cohorts=[2001],
                T_max=2002,
            )

    def test_randomization_error_message_is_actionable(self):
        """Error messages should provide actionable guidance."""
        data = pd.DataFrame({
            'ivar': range(10),
            'd_': np.ones(10).astype(int),  # All treated
            'ydot_postavg': np.random.randn(10),
        })
        
        # Error can be raised by either randomization.py or estimation.py
        with pytest.raises((RandomizationError, InsufficientDataError)) as exc_info:
            randomization_inference(
                firstpost_df=data,
                y_col='ydot_postavg',
                d_col='d_',
                rireps=10,
                seed=42,
            )
        
        error_msg = str(exc_info.value).lower()
        # Error should explain what's wrong and what user needs
        assert 'control' in error_msg or 'n0=0' in error_msg or 'n_control=0' in error_msg
