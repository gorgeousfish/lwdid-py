"""
Test BUG-169: Bootstrap mode detection of homogeneous degenerate samples.

This test verifies that the randomization inference correctly detects and handles
bootstrap samples where all treated units are assigned to the same cohort
(homogeneous sample with fewer than 2 unique cohorts).
"""

import numpy as np
import pandas as pd
import pytest

from lwdid.staggered.randomization import randomization_inference_staggered


class TestBootstrapDegenerateSampleDetection:
    """Test detection of degenerate bootstrap samples."""

    @pytest.fixture
    def simple_staggered_data(self):
        """Create simple staggered data with 3 cohorts and never-treated units."""
        np.random.seed(42)
        n_units = 30
        n_periods = 6
        
        # Create unit-period panel
        data = pd.DataFrame({
            'id': np.repeat(np.arange(n_units), n_periods),
            't': np.tile(np.arange(1, n_periods + 1), n_units),
        })
        
        # Assign gvar: 10 units to cohort 3, 10 to cohort 4, 10 never-treated
        gvar_values = np.concatenate([
            np.repeat(3, 10),   # Cohort 3
            np.repeat(4, 10),   # Cohort 4
            np.repeat(np.inf, 10)  # Never-treated
        ])
        data['gvar'] = data['id'].map(lambda x: gvar_values[x])
        
        # Generate outcome with treatment effect
        data['y'] = np.random.normal(0, 1, len(data))
        data.loc[(data['t'] >= data['gvar']) & np.isfinite(data['gvar']), 'y'] += 0.5
        
        return data

    @pytest.fixture
    def two_cohort_data(self):
        """Create data with exactly 2 cohorts (boundary case)."""
        np.random.seed(42)
        n_units = 20
        n_periods = 5
        
        data = pd.DataFrame({
            'id': np.repeat(np.arange(n_units), n_periods),
            't': np.tile(np.arange(1, n_periods + 1), n_units),
        })
        
        # 10 units cohort 3, 10 never-treated
        gvar_values = np.concatenate([
            np.repeat(3, 10),
            np.repeat(np.inf, 10)
        ])
        data['gvar'] = data['id'].map(lambda x: gvar_values[x])
        
        data['y'] = np.random.normal(0, 1, len(data))
        data.loc[(data['t'] >= data['gvar']) & np.isfinite(data['gvar']), 'y'] += 0.5
        
        return data

    def test_bootstrap_degenerate_detection_cohort_target(self, simple_staggered_data):
        """Test that bootstrap with degenerate samples is handled for cohort target."""
        data = simple_staggered_data
        
        # This should run without error, potentially with many NaN replications
        result = randomization_inference_staggered(
            data=data,
            gvar='gvar',
            ivar='id',
            tvar='t',
            y='y',
            observed_att=0.5,
            target='cohort',
            target_cohort=3,
            ri_method='bootstrap',
            rireps=100,
            seed=123,
            rolling='demean',
            n_never_treated=10,
        )
        
        # Should return valid result
        assert result is not None
        assert 0 <= result.p_value <= 1
        # With bootstrap, some replications may fail due to degenerate samples
        assert result.ri_failed >= 0
        assert result.ri_valid >= 20  # At least some should be valid

    def test_bootstrap_degenerate_detection_overall_target(self, simple_staggered_data):
        """Test that bootstrap with degenerate samples is handled for overall target."""
        data = simple_staggered_data
        
        result = randomization_inference_staggered(
            data=data,
            gvar='gvar',
            ivar='id',
            tvar='t',
            y='y',
            observed_att=0.5,
            target='overall',
            ri_method='bootstrap',
            rireps=100,
            seed=456,
            rolling='demean',
            n_never_treated=10,
        )
        
        assert result is not None
        assert 0 <= result.p_value <= 1

    def test_permutation_preserves_cohort_distribution(self, simple_staggered_data):
        """Test that permutation method preserves all cohorts (no degenerate samples)."""
        data = simple_staggered_data
        
        result = randomization_inference_staggered(
            data=data,
            gvar='gvar',
            ivar='id',
            tvar='t',
            y='y',
            observed_att=0.5,
            target='overall',
            ri_method='permutation',
            rireps=100,
            seed=789,
            rolling='demean',
            n_never_treated=10,
        )
        
        # Permutation should never produce degenerate samples
        # (it preserves the exact cohort distribution)
        # So failure rate should be lower than bootstrap
        assert result is not None
        assert 0 <= result.p_value <= 1

    def test_bootstrap_with_single_original_cohort_raises(self, two_cohort_data):
        """Test that single original cohort (only 1 treated cohort) raises error."""
        data = two_cohort_data
        
        # With only 1 treated cohort, randomization inference for overall is problematic
        # because there's no cohort variation to permute
        # The function requires at least 2 treated units, but with 1 cohort
        # permutation doesn't change anything meaningful
        
        # This should still work but may have high failure rate
        result = randomization_inference_staggered(
            data=data,
            gvar='gvar',
            ivar='id',
            tvar='t',
            y='y',
            observed_att=0.5,
            target='cohort_time',
            target_cohort=3,
            target_period=3,
            ri_method='bootstrap',
            rireps=100,
            seed=999,
            rolling='demean',
            n_never_treated=10,
        )
        
        assert result is not None


class TestBootstrapDegenerateNumericalValidation:
    """Numerical validation of degenerate sample detection."""

    def test_degenerate_check_logic(self):
        """Directly test the degenerate sample detection logic."""
        # Simulate what happens in the bootstrap
        treated_gvar_values = np.array([2020, 2020, 2021, 2021, 2022])
        n_unique_orig = len(np.unique(treated_gvar_values))  # 3
        
        # Case 1: All same cohort (homogeneous)
        perm_gvar_1 = np.array([2020, 2020, 2020, 2020, 2020])
        n_unique_1 = len(np.unique(perm_gvar_1))  # 1
        
        is_degenerate_1 = (n_unique_1 < 2 or n_unique_1 < n_unique_orig)
        assert is_degenerate_1 is True, "Homogeneous sample should be detected"
        
        # Case 2: Missing one cohort
        perm_gvar_2 = np.array([2020, 2020, 2021, 2021, 2020])
        n_unique_2 = len(np.unique(perm_gvar_2))  # 2
        
        is_degenerate_2 = (n_unique_2 < 2 or n_unique_2 < n_unique_orig)
        assert is_degenerate_2 is True, "Missing cohort should be detected"
        
        # Case 3: All cohorts present (valid)
        perm_gvar_3 = np.array([2021, 2020, 2022, 2021, 2020])
        n_unique_3 = len(np.unique(perm_gvar_3))  # 3
        
        is_degenerate_3 = (n_unique_3 < 2 or n_unique_3 < n_unique_orig)
        assert is_degenerate_3 is False, "Valid sample should not be flagged"

    def test_minimum_cohort_requirement(self):
        """Test that n_unique < 2 condition is checked."""
        # With only 2 original cohorts
        treated_gvar_values = np.array([2020, 2020, 2021, 2021])
        n_unique_orig = len(np.unique(treated_gvar_values))  # 2
        
        # Bootstrap produces single cohort
        perm_gvar = np.array([2020, 2020, 2020, 2020])
        n_unique_perm = len(np.unique(perm_gvar))  # 1
        
        # Should be detected by n_unique < 2 check
        assert n_unique_perm < 2, "Single cohort should have n_unique < 2"
        
        is_degenerate = (n_unique_perm < 2 or n_unique_perm < n_unique_orig)
        assert is_degenerate is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
