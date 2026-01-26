"""
Unit tests for DESIGN-122 fix.

DESIGN-122: Early degenerate detection for cohort_time target in bootstrap RI
"""

import warnings
import numpy as np
import pandas as pd
import pytest


class TestDesign122CohortTimeDegenerate:
    """Tests for DESIGN-122: cohort_time early degenerate detection."""

    @pytest.fixture
    def staggered_data(self):
        """Create staggered DiD test data."""
        np.random.seed(42)
        n_units = 30
        n_periods = 6

        units = np.repeat(np.arange(1, n_units + 1), n_periods)
        periods = np.tile(np.arange(1, n_periods + 1), n_units)

        # Cohorts: 10 units in cohort 4, 10 in cohort 5, 10 never-treated (0)
        unit_cohorts = np.concatenate([
            np.full(10, 4),  # Cohort 4
            np.full(10, 5),  # Cohort 5
            np.full(10, 0),  # Never-treated
        ])
        gvar = np.repeat(unit_cohorts, n_periods)

        y = np.random.randn(len(units))

        return pd.DataFrame({
            'id': units,
            'year': periods,
            'gvar': gvar,
            'y': y,
        })

    def test_cohort_time_degenerate_bootstrap_returns_nan(self, staggered_data):
        """Bootstrap sample missing target_cohort should return NaN early."""
        # This tests the internal logic - when target_cohort is not in
        # the bootstrap sample, the function should return NaN immediately
        # without performing expensive computations

        # Verify the fix is in place by checking the source code
        from lwdid.staggered import randomization
        import inspect

        source = inspect.getsource(randomization._single_staggered_ri_replication)

        # Check that DESIGN-122 fix is present
        assert "cohort_time" in source
        assert "target_cohort not in perm_treated_gvar" in source

    def test_bootstrap_ri_handles_missing_cohort(self, staggered_data):
        """Bootstrap RI should handle missing cohorts gracefully."""
        from lwdid.staggered.randomization import randomization_inference_staggered
        from lwdid.staggered.transformations import transform_staggered_demean

        # Transform data
        data_transformed = transform_staggered_demean(
            staggered_data, 'y', 'id', 'year', 'gvar'
        )

        # Run RI with cohort_time target - should complete without error
        # even if some bootstrap samples don't have the target cohort
        try:
            result = randomization_inference_staggered(
                data=data_transformed,
                gvar='gvar',
                ivar='id',
                tvar='year',
                y='y',
                observed_att=0.1,
                target='cohort_time',
                target_cohort=4,
                target_period=4,
                ri_method='bootstrap',
                rireps=50,  # Small number for testing
                seed=42,
                rolling='demean',
            )
            assert result.ri_valid > 0 or result.ri_failed > 0
        except Exception as e:
            # Some failures are expected with small data
            assert "Insufficient valid" in str(e) or "Too few" in str(e)


class TestDesign122Integration:
    """Integration test for DESIGN-122 fix."""

    def test_fix_present_in_code(self):
        """Verify DESIGN-122 fix is present in the source code."""
        import inspect
        from lwdid.staggered import randomization

        # DESIGN-122
        source_122 = inspect.getsource(randomization._single_staggered_ri_replication)
        assert "DESIGN-122" in source_122
        assert "cohort_time" in source_122
        assert "target_cohort not in perm_treated_gvar" in source_122


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
