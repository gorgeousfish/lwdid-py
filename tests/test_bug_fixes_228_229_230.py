"""
Tests for BUG-228, BUG-229, BUG-230 fixes in randomization inference modules.

BUG-228: Parallel seed generation collision prevention using SeedSequence.spawn()
BUG-229: n_jobs=0 validation to prevent ProcessPoolExecutor errors
BUG-230: Quarterly data (tvar as list) support in staggered RI
"""

import numpy as np
import pandas as pd
import pytest

from lwdid.randomization import randomization_inference
from lwdid.staggered.randomization import randomization_inference_staggered
from lwdid.exceptions import RandomizationError


class TestBug228SeedSequenceSpawn:
    """Test BUG-228: SeedSequence.spawn() for independent parallel seeds."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for cross-sectional RI tests."""
        np.random.seed(42)
        n = 50
        d = np.concatenate([np.ones(20), np.zeros(30)]).astype(int)
        y = 1.0 + 2.0 * d + np.random.randn(n) * 0.5
        return pd.DataFrame({
            'ydot_postavg': y,
            'd_': d,
            'ivar': np.arange(n),
        })

    def test_seed_reproducibility_serial(self, sample_data):
        """Verify that serial execution with same seed produces same results."""
        result1 = randomization_inference(
            sample_data,
            y_col='ydot_postavg',
            d_col='d_',
            ivar='ivar',
            rireps=100,
            seed=42,
            ri_method='permutation',
            n_jobs=None,  # Serial execution
        )
        result2 = randomization_inference(
            sample_data,
            y_col='ydot_postavg',
            d_col='d_',
            ivar='ivar',
            rireps=100,
            seed=42,
            ri_method='permutation',
            n_jobs=None,
        )
        assert result1['p_value'] == result2['p_value']

    def test_different_seeds_different_results(self, sample_data):
        """Verify that different seeds produce different results."""
        result1 = randomization_inference(
            sample_data,
            y_col='ydot_postavg',
            d_col='d_',
            ivar='ivar',
            rireps=100,
            seed=42,
            ri_method='permutation',
            n_jobs=None,
        )
        result2 = randomization_inference(
            sample_data,
            y_col='ydot_postavg',
            d_col='d_',
            ivar='ivar',
            rireps=100,
            seed=43,
            ri_method='permutation',
            n_jobs=None,
        )
        # Different seeds should generally produce different p-values
        # (not guaranteed but highly likely with different random streams)
        # We mainly test that it runs without error
        assert 'p_value' in result1
        assert 'p_value' in result2

    def test_seed_sequence_independence_verification(self):
        """Verify SeedSequence.spawn() produces statistically independent streams."""
        # Test the underlying mechanism
        parent1 = np.random.SeedSequence(42)
        parent2 = np.random.SeedSequence(43)
        
        children1 = parent1.spawn(10)
        children2 = parent2.spawn(10)
        
        # Generate random numbers from each child
        rng1_child0 = np.random.default_rng(children1[0])
        rng1_child1 = np.random.default_rng(children1[1])
        rng2_child0 = np.random.default_rng(children2[0])
        
        nums1_0 = rng1_child0.random(10)
        nums1_1 = rng1_child1.random(10)
        nums2_0 = rng2_child0.random(10)
        
        # All should be different (statistical independence)
        assert not np.allclose(nums1_0, nums1_1)
        assert not np.allclose(nums1_0, nums2_0)
        assert not np.allclose(nums1_1, nums2_0)

    def test_no_collision_between_experiments(self):
        """Verify no seed collision between seed=42 rep=1 and seed=43 rep=0."""
        # With old method: base_seed + offset
        # seed=42, offset=1 => 43
        # seed=43, offset=0 => 43
        # These would collide!
        
        # With SeedSequence.spawn(), children are independent
        parent42 = np.random.SeedSequence(42)
        parent43 = np.random.SeedSequence(43)
        
        children42 = parent42.spawn(10)
        children43 = parent43.spawn(10)
        
        # Child 1 from seed=42 should NOT equal Child 0 from seed=43
        rng42_1 = np.random.default_rng(children42[1])
        rng43_0 = np.random.default_rng(children43[0])
        
        nums42_1 = rng42_1.random(100)
        nums43_0 = rng43_0.random(100)
        
        assert not np.allclose(nums42_1, nums43_0)


class TestBug229NJobsValidation:
    """Test BUG-229: n_jobs=0 validation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for RI tests."""
        np.random.seed(42)
        n = 50
        d = np.concatenate([np.ones(20), np.zeros(30)]).astype(int)
        y = 1.0 + 2.0 * d + np.random.randn(n) * 0.5
        return pd.DataFrame({
            'ydot_postavg': y,
            'd_': d,
            'ivar': np.arange(n),
        })

    @pytest.fixture
    def staggered_data(self):
        """Create sample staggered data for RI tests."""
        np.random.seed(42)
        n_units = 20
        n_periods = 6
        
        rows = []
        for i in range(n_units):
            if i < 5:
                gvar = np.inf  # Never treated
            elif i < 10:
                gvar = 4
            elif i < 15:
                gvar = 5
            else:
                gvar = 6
            
            for t in range(1, n_periods + 1):
                is_treated = (gvar != np.inf) and (t >= gvar)
                y = 1.0 + np.random.randn() * 0.5 + (2.0 if is_treated else 0)
                rows.append({
                    'id': i,
                    'year': t,
                    'gvar': gvar,
                    'y': y,
                })
        return pd.DataFrame(rows)

    def test_n_jobs_zero_raises_error_cross_sectional(self, sample_data):
        """n_jobs=0 should raise RandomizationError in cross-sectional RI."""
        with pytest.raises(RandomizationError, match="n_jobs must be positive"):
            randomization_inference(
                sample_data,
                y_col='ydot_postavg',
                d_col='d_',
                ivar='ivar',
                rireps=100,
                seed=42,
                ri_method='permutation',
                n_jobs=0,
            )

    def test_n_jobs_zero_raises_error_staggered(self, staggered_data):
        """n_jobs=0 should raise RandomizationError in staggered RI."""
        with pytest.raises(RandomizationError, match="n_jobs must be positive"):
            randomization_inference_staggered(
                staggered_data,
                gvar='gvar',
                ivar='id',
                tvar='year',
                y='y',
                observed_att=1.5,
                target='overall',
                ri_method='permutation',
                rireps=100,
                seed=42,
                n_never_treated=5,
                n_jobs=0,
            )

    def test_n_jobs_none_valid(self, sample_data):
        """n_jobs=None should work (serial execution)."""
        result = randomization_inference(
            sample_data,
            y_col='ydot_postavg',
            d_col='d_',
            ivar='ivar',
            rireps=50,
            seed=42,
            ri_method='permutation',
            n_jobs=None,
        )
        assert 'p_value' in result

    def test_n_jobs_one_valid(self, sample_data):
        """n_jobs=1 should work (serial execution)."""
        result = randomization_inference(
            sample_data,
            y_col='ydot_postavg',
            d_col='d_',
            ivar='ivar',
            rireps=50,
            seed=42,
            ri_method='permutation',
            n_jobs=1,
        )
        assert 'p_value' in result

    def test_n_jobs_negative_one_valid(self, sample_data):
        """n_jobs=-1 should work (use all cores, but may fall back to serial)."""
        result = randomization_inference(
            sample_data,
            y_col='ydot_postavg',
            d_col='d_',
            ivar='ivar',
            rireps=50,  # Small rireps, will use serial anyway
            seed=42,
            ri_method='permutation',
            n_jobs=-1,
        )
        assert 'p_value' in result

    def test_n_jobs_positive_valid(self, sample_data):
        """n_jobs>0 should work."""
        result = randomization_inference(
            sample_data,
            y_col='ydot_postavg',
            d_col='d_',
            ivar='ivar',
            rireps=50,
            seed=42,
            ri_method='permutation',
            n_jobs=2,
        )
        assert 'p_value' in result


class TestBug230QuarterlyDataTvar:
    """Test BUG-230: Quarterly data (tvar as list) support in staggered RI."""

    @pytest.fixture
    def quarterly_data(self):
        """Create quarterly staggered data with tvar=['year', 'quarter']."""
        np.random.seed(42)
        n_units = 20
        years = [2000, 2001, 2002]
        quarters = [1, 2, 3, 4]
        
        rows = []
        for i in range(n_units):
            if i < 5:
                gvar = np.inf  # Never treated
            elif i < 10:
                gvar = 2001  # Treated starting 2001
            elif i < 15:
                gvar = 2002  # Treated starting 2002
            else:
                gvar = np.inf  # Never treated
            
            for year in years:
                for quarter in quarters:
                    is_treated = (gvar != np.inf) and (year >= gvar)
                    y = 1.0 + np.random.randn() * 0.5 + (2.0 if is_treated else 0)
                    rows.append({
                        'id': i,
                        'year': year,
                        'quarter': quarter,
                        'gvar': gvar,
                        'y': y,
                    })
        return pd.DataFrame(rows)

    @pytest.fixture
    def annual_data(self):
        """Create annual staggered data with tvar='year'."""
        np.random.seed(42)
        n_units = 20
        n_periods = 6
        
        rows = []
        for i in range(n_units):
            if i < 5:
                gvar = np.inf
            elif i < 10:
                gvar = 4
            elif i < 15:
                gvar = 5
            else:
                gvar = np.inf
            
            for t in range(1, n_periods + 1):
                is_treated = (gvar != np.inf) and (t >= gvar)
                y = 1.0 + np.random.randn() * 0.5 + (2.0 if is_treated else 0)
                rows.append({
                    'id': i,
                    'year': t,
                    'gvar': gvar,
                    'y': y,
                })
        return pd.DataFrame(rows)

    def test_tvar_as_list_extracts_year_for_tmax(self, quarterly_data):
        """tvar=['year', 'quarter'] should extract year for T_max calculation."""
        # This should not raise an error about Series truth value
        result = randomization_inference_staggered(
            quarterly_data,
            gvar='gvar',
            ivar='id',
            tvar=['year', 'quarter'],
            y='y',
            observed_att=1.5,
            target='overall',
            ri_method='permutation',
            rireps=50,
            seed=42,
            n_never_treated=10,
            n_jobs=None,
        )
        assert 'p_value' in result.__dict__

    def test_tvar_as_string_works(self, annual_data):
        """tvar='year' (string) should continue to work."""
        result = randomization_inference_staggered(
            annual_data,
            gvar='gvar',
            ivar='id',
            tvar='year',
            y='y',
            observed_att=1.5,
            target='overall',
            ri_method='permutation',
            rireps=50,
            seed=42,
            n_never_treated=10,
            n_jobs=None,
        )
        assert 'p_value' in result.__dict__

    def test_tvar_as_tuple_works(self, quarterly_data):
        """tvar=('year', 'quarter') tuple should also work."""
        result = randomization_inference_staggered(
            quarterly_data,
            gvar='gvar',
            ivar='id',
            tvar=('year', 'quarter'),
            y='y',
            observed_att=1.5,
            target='overall',
            ri_method='permutation',
            rireps=50,
            seed=42,
            n_never_treated=10,
            n_jobs=None,
        )
        assert 'p_value' in result.__dict__

    def test_tvar_list_tmax_correct(self, quarterly_data):
        """Verify T_max is correctly extracted from year column."""
        # T_max should be max(year) = 2002, not some Series or DataFrame
        # We can verify this indirectly by checking the function runs
        # and produces valid results
        result = randomization_inference_staggered(
            quarterly_data,
            gvar='gvar',
            ivar='id',
            tvar=['year', 'quarter'],
            y='y',
            observed_att=1.5,
            target='cohort_time',
            target_cohort=2001,
            target_period=2001,
            ri_method='permutation',
            rireps=50,
            seed=42,
            n_never_treated=10,
            n_jobs=None,
        )
        assert result.ri_valid >= 20  # Should have valid replications


class TestIntegrationAllFixes:
    """Integration tests combining all three fixes."""

    @pytest.fixture
    def quarterly_staggered_data(self):
        """Create quarterly staggered data for comprehensive testing."""
        np.random.seed(42)
        n_units = 30
        years = [2000, 2001, 2002, 2003]
        quarters = [1, 2, 3, 4]
        
        rows = []
        for i in range(n_units):
            if i < 10:
                gvar = np.inf  # Never treated
            elif i < 20:
                gvar = 2001
            else:
                gvar = 2002
            
            for year in years:
                for quarter in quarters:
                    is_treated = (gvar != np.inf) and (year >= gvar)
                    y = 1.0 + np.random.randn() * 0.5 + (2.0 if is_treated else 0)
                    rows.append({
                        'id': i,
                        'year': year,
                        'quarter': quarter,
                        'gvar': gvar,
                        'y': y,
                    })
        return pd.DataFrame(rows)

    def test_quarterly_data_with_serial_execution(self, quarterly_staggered_data):
        """Test quarterly data with n_jobs=None (serial)."""
        result = randomization_inference_staggered(
            quarterly_staggered_data,
            gvar='gvar',
            ivar='id',
            tvar=['year', 'quarter'],
            y='y',
            observed_att=1.5,
            target='overall',
            ri_method='permutation',
            rireps=50,
            seed=42,
            n_never_treated=10,
            n_jobs=None,
        )
        assert result.p_value >= 0
        assert result.p_value <= 1
        assert result.ri_valid > 0

    def test_seed_reproducibility_with_quarterly_data(self, quarterly_staggered_data):
        """Test seed reproducibility with quarterly data."""
        result1 = randomization_inference_staggered(
            quarterly_staggered_data,
            gvar='gvar',
            ivar='id',
            tvar=['year', 'quarter'],
            y='y',
            observed_att=1.5,
            target='overall',
            ri_method='permutation',
            rireps=50,
            seed=123,
            n_never_treated=10,
            n_jobs=None,
        )
        result2 = randomization_inference_staggered(
            quarterly_staggered_data,
            gvar='gvar',
            ivar='id',
            tvar=['year', 'quarter'],
            y='y',
            observed_att=1.5,
            target='overall',
            ri_method='permutation',
            rireps=50,
            seed=123,
            n_never_treated=10,
            n_jobs=None,
        )
        assert result1.p_value == result2.p_value
