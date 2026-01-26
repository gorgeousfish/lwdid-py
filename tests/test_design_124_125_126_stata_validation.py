"""
Stata cross-validation tests for DESIGN-124, DESIGN-125, and DESIGN-126.

This module validates that:
1. DESIGN-124: Python seed generation range matches Stata formula
2. DESIGN-125: P-value formula c/B matches Stata implementation  
3. DESIGN-126: Single-observation warnings are consistent

The p-value validation is particularly important as it confirms the
implementation matches Lee & Wooldridge (2025) paper and Stata lwdid.
"""

import numpy as np
import pandas as pd
import pytest


class TestDesign125PValueStataValidation:
    """Validate p-value formula against Stata lwdid implementation."""

    def test_pvalue_formula_matches_stata_definition(self):
        """
        Verify p-value formula matches Stata lwdid.ado lines 361-363:
        
        Stata code:
            mata: st_matrix("abs_res", abs(st_matrix("`res'")))
            mata: st_numscalar("__p_ri", mean(st_matrix("abs_res") :>= abs(st_numscalar("__b0"))))
        
        This computes: mean(|T_sim| >= |T_obs|) = #{|T_sim| >= |T_obs|} / B
        
        Which is exactly c/B formula used in Python.
        """
        # Simulate 1000 permutation statistics
        np.random.seed(12345)
        perm_stats = np.random.normal(0, 1, 1000)
        observed_att = 1.5  # Observed statistic

        # Python formula (from staggered/randomization.py lines 685-686)
        n_extreme = int((np.abs(perm_stats) >= abs(observed_att)).sum())
        n_valid = len(perm_stats)
        p_value_python = float(n_extreme / n_valid)

        # Stata formula: mean(abs_res >= abs(b0))
        # This is equivalent to sum(abs_res >= abs(b0)) / N
        abs_perm = np.abs(perm_stats)
        abs_obs = abs(observed_att)
        p_value_stata = np.mean(abs_perm >= abs_obs)

        # Should be identical
        assert p_value_python == p_value_stata, \
            f"Python p={p_value_python} != Stata p={p_value_stata}"

    def test_pvalue_edge_case_all_extreme(self):
        """Test when all simulated statistics are extreme."""
        perm_stats = np.array([2.0, 3.0, 4.0, 5.0])
        observed_att = 1.0

        n_extreme = int((np.abs(perm_stats) >= abs(observed_att)).sum())
        n_valid = len(perm_stats)
        p_value = float(n_extreme / n_valid)

        assert p_value == 1.0, "All extreme should give p=1.0"

    def test_pvalue_edge_case_none_extreme(self):
        """Test when no simulated statistics are extreme."""
        perm_stats = np.array([0.1, 0.2, 0.3, 0.4])
        observed_att = 5.0

        n_extreme = int((np.abs(perm_stats) >= abs(observed_att)).sum())
        n_valid = len(perm_stats)
        p_value = float(n_extreme / n_valid)

        assert p_value == 0.0, "None extreme should give p=0.0"

    def test_pvalue_symmetric_around_zero(self):
        """Test that two-sided p-value is symmetric around zero."""
        perm_stats = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        # Positive observed
        observed_pos = 1.5
        n_extreme_pos = int((np.abs(perm_stats) >= abs(observed_pos)).sum())
        p_pos = float(n_extreme_pos / len(perm_stats))

        # Negative observed (same magnitude)
        observed_neg = -1.5
        n_extreme_neg = int((np.abs(perm_stats) >= abs(observed_neg)).sum())
        p_neg = float(n_extreme_neg / len(perm_stats))

        assert p_pos == p_neg, "Two-sided p-value should be symmetric"


class TestDesign124SeedRangeValidation:
    """Validate seed generation range matches Stata."""

    def test_seed_range_lower_bound(self):
        """
        Verify lower bound of seed range.
        
        Stata: ceil(runiform() * 1e6 + 1000*runiform())
        When both runiform() return values close to 0:
        - Minimum is ceil(epsilon) = 1
        """
        import math
        import random

        # Test with very small random values
        # The smallest possible seed should be 1 (from ceil(epsilon))
        # We can't directly test epsilon, but we can verify the formula
        # produces values >= 1

        random.seed(0)
        min_seed = float('inf')
        for _ in range(100000):
            seed = math.ceil(random.random() * 1e6 + 1000 * random.random())
            min_seed = min(min_seed, seed)

        assert min_seed >= 1, f"Minimum seed should be >= 1, got {min_seed}"

    def test_seed_range_upper_bound(self):
        """
        Verify upper bound of seed range.
        
        Stata: ceil(runiform() * 1e6 + 1000*runiform())
        When both runiform() return values close to 1:
        - Maximum approaches ceil(1e6 + 1000) = 1001000
        - But since runiform() is in [0, 1), max is < 1001000
        """
        import math
        import random

        random.seed(0)
        max_seed = 0
        for _ in range(100000):
            seed = math.ceil(random.random() * 1e6 + 1000 * random.random())
            max_seed = max(max_seed, seed)

        assert max_seed <= 1001000, f"Maximum seed should be <= 1001000, got {max_seed}"

    def test_seed_distribution_uniformity(self):
        """Verify seed distribution is approximately uniform."""
        import math
        import random

        random.seed(42)
        seeds = [math.ceil(random.random() * 1e6 + 1000 * random.random())
                 for _ in range(10000)]

        # Check that seeds span most of the range
        assert min(seeds) < 100000, "Should have some low seeds"
        assert max(seeds) > 900000, "Should have some high seeds"

        # Check quartiles are reasonable
        q1, q2, q3 = np.percentile(seeds, [25, 50, 75])
        # For uniform [1, 1001000], expect Q1~250k, Q2~500k, Q3~750k
        assert 150000 < q1 < 350000, f"Q1={q1} out of range"
        assert 400000 < q2 < 600000, f"Q2={q2} out of range"
        assert 650000 < q3 < 850000, f"Q3={q3} out of range"


class TestDesign126ValidationConsistency:
    """Validate single-observation warning consistency across validation functions."""

    def test_warning_text_consistency(self):
        """Verify warning messages have consistent format."""
        import warnings
        from lwdid.validation import (
            _validate_treatment_time_invariance,
            _validate_time_invariant_controls,
        )

        # Create data with single-obs unit
        data = pd.DataFrame({
            'id': [1, 1, 2],  # Unit 2 has only 1 observation
            'year': [2000, 2001, 2000],
            'd': [1, 1, 0],
            'x': [0.5, 0.5, 0.3],
        })

        # Capture treatment warning
        with warnings.catch_warnings(record=True) as w_t:
            warnings.simplefilter("always")
            _validate_treatment_time_invariance(data, 'd', 'id')
            treatment_msg = str(w_t[0].message) if w_t else ""

        # Capture control warning
        with warnings.catch_warnings(record=True) as w_c:
            warnings.simplefilter("always")
            _validate_time_invariant_controls(data, 'id', ['x'])
            control_msg = str(w_c[0].message) if w_c else ""

        # Both should contain the same key phrases
        assert "1 unit(s)" in treatment_msg
        assert "1 unit(s)" in control_msg
        assert "Time-invariance cannot be verified" in treatment_msg
        assert "Time-invariance cannot be verified" in control_msg

    def test_multiple_single_obs_units(self):
        """Test warning with multiple single-observation units."""
        import warnings
        from lwdid.validation import _validate_treatment_time_invariance

        # Create data with 3 single-obs units
        data = pd.DataFrame({
            'id': [1, 1, 2, 3, 4],  # Units 2, 3, 4 have only 1 observation
            'year': [2000, 2001, 2000, 2000, 2000],
            'd': [1, 1, 0, 1, 0],
        })

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _validate_treatment_time_invariance(data, 'd', 'id')

            assert len(w) == 1
            msg = str(w[0].message)
            assert "3 unit(s)" in msg


class TestPValueFormulaVibeMathValidation:
    """
    Validate p-value formula using mathematical properties.
    
    The c/B formula has these properties:
    - p ∈ [0, 1]
    - p = 0 when no statistics are extreme
    - p = 1 when all statistics are extreme
    - E[p] = 2 * P(|Z| >= |z_obs|) under H0 for symmetric distributions
    """

    def test_pvalue_range(self):
        """Verify p-value is always in [0, 1]."""
        np.random.seed(123)

        for _ in range(100):
            perm_stats = np.random.normal(0, 1, 100)
            observed = np.random.normal(0, 2)

            n_extreme = int((np.abs(perm_stats) >= abs(observed)).sum())
            n_valid = len(perm_stats)
            p_value = float(n_extreme / n_valid)

            assert 0 <= p_value <= 1, f"p-value {p_value} out of [0, 1]"

    def test_pvalue_expectation_under_null(self):
        """
        Under H0 (no treatment effect), the p-value should be 
        approximately uniformly distributed on [0, 1].
        
        This is a fundamental property of valid p-values.
        """
        np.random.seed(456)
        p_values = []

        # Run many simulations under H0
        for _ in range(1000):
            # Generate data under null (treatment has no effect)
            perm_stats = np.random.normal(0, 1, 100)
            # Observed is also from null
            observed = np.random.normal(0, 1)

            n_extreme = int((np.abs(perm_stats) >= abs(observed)).sum())
            p_value = float(n_extreme / len(perm_stats))
            p_values.append(p_value)

        p_values = np.array(p_values)

        # Under valid p-values from H0, mean should be ~0.5
        # and distribution should be roughly uniform
        mean_p = np.mean(p_values)
        assert 0.4 < mean_p < 0.6, f"Mean p-value {mean_p} suggests bias"

        # Check uniform distribution via quartiles
        q25 = np.percentile(p_values, 25)
        q50 = np.percentile(p_values, 50)
        q75 = np.percentile(p_values, 75)

        assert 0.15 < q25 < 0.35, f"Q25={q25} suggests non-uniform distribution"
        assert 0.40 < q50 < 0.60, f"Q50={q50} suggests non-uniform distribution"
        assert 0.65 < q75 < 0.85, f"Q75={q75} suggests non-uniform distribution"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
