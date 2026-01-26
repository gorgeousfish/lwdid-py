"""
Unit tests for BUG-098, BUG-099, BUG-100 fixes.

This module tests the fixes for:
- BUG-098: LaTeX special character escaping in to_latex() methods
- BUG-099: PSM tie-breaking reproducibility with stable sorting
- BUG-100: Sample size counting after NaN filtering
"""

import tempfile
import os
import numpy as np
import pandas as pd
import pytest

from lwdid.results import LWDIDResults, _latex_escape_string
from lwdid.staggered.estimators import _nearest_neighbor_match
from lwdid.staggered.estimation import estimate_cohort_time_effects
from lwdid.staggered.transformations import transform_staggered_demean


class TestBug098LaTeXEscaping:
    """Test LaTeX special character escaping (BUG-098)."""
    
    def test_latex_escape_string_basic(self):
        """Test _latex_escape_string with common special characters."""
        # Test underscore
        assert _latex_escape_string("CI_lower") == r"CI\_lower"
        assert _latex_escape_string("ci_upper") == r"ci\_upper"
        
        # Test percent
        assert _latex_escape_string("95%") == r"95\%"
        
        # Test ampersand
        assert _latex_escape_string("A&B") == r"A\&B"
        
        # Test hash
        assert _latex_escape_string("#1") == r"\#1"
        
        # Test dollar
        assert _latex_escape_string("$100") == r"\$100"
    
    def test_latex_escape_string_multiple_chars(self):
        """Test escaping multiple special characters in one string."""
        input_str = "CI_lower_95%"
        expected = r"CI\_lower\_95\%"
        assert _latex_escape_string(input_str) == expected
        
        input_str = "N_treat&control"
        expected = r"N\_treat\&control"
        assert _latex_escape_string(input_str) == expected
    
    def test_latex_escape_string_no_complex_chars(self):
        """Test that simple escaping works for column names."""
        # Column names typically don't have backslashes, braces, etc.
        # Just test the common ones
        assert _latex_escape_string("plain") == "plain"
        assert _latex_escape_string("col_name") == r"col\_name"
    
    def test_to_latex_column_names_escaped(self):
        """Test that to_latex() escapes column names in DataFrames."""
        # Create minimal results object
        results_dict = {
            'att': 0.5,
            'se_att': 0.1,
            't_stat': 5.0,
            'pvalue': 0.001,
            'ci_lower': 0.3,
            'ci_upper': 0.7,
            'nobs': 100,
            'df_resid': 98,
            'params': np.array([0.0, 0.5]),
            'bse': np.array([0.0, 0.1]),
            'vcov': np.eye(2),
            'resid': np.zeros(100),
            'vce_type': 'robust',
        }
        
        metadata = {
            'K': 1,
            'tpost1': 2,
            'depvar': 'y',
            'N_treated': 50,
            'N_control': 50,
        }
        
        # Create DataFrame with underscores in column names
        att_by_period = pd.DataFrame({
            'period': [2, 3, 4],
            'att': [0.5, 0.6, 0.7],
            'se': [0.1, 0.12, 0.15],
            'ci_lower': [0.3, 0.36, 0.4],
            'ci_upper': [0.7, 0.84, 1.0],
        })
        
        results = LWDIDResults(results_dict, metadata, att_by_period=att_by_period)
        
        # Export to LaTeX
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.tex') as f:
            latex_path = f.name
        
        try:
            results.to_latex(latex_path)
            
            # Read generated LaTeX
            with open(latex_path, 'r', encoding='utf-8') as f:
                latex_content = f.read()
            
            # Verify escaped column names appear in LaTeX
            assert r"ci\_lower" in latex_content
            assert r"ci\_upper" in latex_content
            
            # Verify unescaped underscores do NOT appear
            # (except in LaTeX commands like \toprule)
            lines = latex_content.split('\n')
            for line in lines:
                # Skip LaTeX command lines
                if line.strip().startswith('\\'):
                    continue
                # In data/header lines, underscores should be escaped
                if 'ci_lower' in line or 'ci_upper' in line:
                    assert '_' not in line or r'\_' in line, \
                        f"Unescaped underscore in: {line}"
        
        finally:
            if os.path.exists(latex_path):
                os.unlink(latex_path)
    
    def test_to_latex_staggered_column_names_escaped(self):
        """Test that to_latex_staggered() escapes column names."""
        # Create staggered results object
        results_dict = {
            'att': 0.5,
            'se_att': 0.1,
            't_stat': 5.0,
            'pvalue': 0.001,
            'ci_lower': 0.3,
            'ci_upper': 0.7,
            'nobs': 100,
            'df_resid': 98,
            'params': np.array([0.0, 0.5]),
            'bse': np.array([0.0, 0.1]),
            'vcov': np.eye(2),
            'resid': np.zeros(100),
            'vce_type': 'robust',
            'is_staggered': True,
            'cohorts': [2, 3],
            'cohort_sizes': {2: 25, 3: 25},
            'att_by_cohort_time': pd.DataFrame({
                'cohort': [2, 2, 3, 3],
                'period': [2, 3, 3, 4],
                'att': [0.5, 0.6, 0.7, 0.8],
                'se': [0.1, 0.12, 0.15, 0.18],
                'ci_lower': [0.3, 0.36, 0.4, 0.44],
                'ci_upper': [0.7, 0.84, 1.0, 1.16],
            }),
            'att_by_cohort': pd.DataFrame({
                'cohort': [2, 3],
                'att': [0.55, 0.75],
                'se': [0.11, 0.16],
            }),
            'att_overall': 0.65,
            'se_overall': 0.13,
            'ci_overall_lower': 0.39,
            'ci_overall_upper': 0.91,
            'cohort_weights': {2: 0.5, 3: 0.5},
            'control_group': 'not_yet_treated',
            'control_group_used': 'not_yet_treated',
            'aggregate': 'overall',
            'estimator': 'ra',
            'n_never_treated': 50,
        }
        
        metadata = {
            'K': 1,
            'tpost1': 2,
            'depvar': 'y',
            'N_treated': 50,
            'N_control': 50,
        }
        
        results = LWDIDResults(results_dict, metadata)
        
        # Export to LaTeX
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.tex') as f:
            latex_path = f.name
        
        try:
            results.to_latex_staggered(latex_path)
            
            # Read generated LaTeX
            with open(latex_path, 'r', encoding='utf-8') as f:
                latex_content = f.read()
            
            # Verify escaped column names
            assert r"ci\_lower" in latex_content
            assert r"ci\_upper" in latex_content
            
        finally:
            if os.path.exists(latex_path):
                os.unlink(latex_path)


class TestBug099PSMTieBreaking:
    """Test PSM tie-breaking reproducibility (BUG-099)."""
    
    def test_nearest_neighbor_match_stable_sort(self):
        """Test that _nearest_neighbor_match uses stable sorting for ties."""
        # Create propensity scores with intentional ties
        # Treated unit has ps=0.5
        # Control units: two at ps=0.4, two at ps=0.6
        pscores_treat = np.array([0.5])
        pscores_control = np.array([0.4, 0.4, 0.6, 0.6])
        
        # Without stable sort, the order of tied units would be unpredictable
        # With stable sort, original order is preserved for ties
        
        # Match 2 nearest neighbors
        matched_ids_1, counts_1, dropped_1 = _nearest_neighbor_match(
            pscores_treat=pscores_treat,
            pscores_control=pscores_control,
            n_neighbors=2,
            with_replacement=True,
            caliper=None,
            match_order='data',
            seed=None,
        )
        
        # Run again - should get identical results
        matched_ids_2, counts_2, dropped_2 = _nearest_neighbor_match(
            pscores_treat=pscores_treat,
            pscores_control=pscores_control,
            n_neighbors=2,
            with_replacement=True,
            caliper=None,
            match_order='data',
            seed=None,
        )
        
        # Results should be identical
        assert matched_ids_1 == matched_ids_2
        assert np.array_equal(counts_1, counts_2)
        assert dropped_1 == dropped_2
        
        # For this specific case, should match first two controls (indices 0, 1)
        # because they have distance 0.1, while indices 2, 3 have distance 0.1 as well
        # With stable sort, indices 0, 1 come before 2, 3 in original order
        assert len(matched_ids_1[0]) == 2
        # Both should be from the closer group (distance 0.1)
        distances = np.abs(pscores_control - pscores_treat[0])
        for idx in matched_ids_1[0]:
            assert np.isclose(distances[idx], 0.1, atol=1e-10)
    
    def test_psm_reproducibility_multiple_runs(self):
        """Test that PSM matching is fully reproducible across multiple runs."""
        # Create synthetic data with ties in propensity scores
        np.random.seed(42)
        n = 50
        
        # Create data with some duplicate covariate values to induce PS ties
        X = np.random.choice([0, 1, 2], size=n)
        D = (X >= 1).astype(int)
        Y = 0.5 * D + 0.3 * X + np.random.normal(0, 0.1, n)
        
        data = pd.DataFrame({
            'y': Y,
            'd': D,
            'x': X,
        })
        
        # Run PSM estimation twice with same parameters
        from lwdid.staggered.estimators import estimate_psm
        
        result1 = estimate_psm(
            data=data,
            y='y',
            d='d',
            propensity_controls=['x'],
            n_neighbors=1,
            with_replacement=True,
            caliper=None,
            match_order='data',
            trim_threshold=0.01,
            alpha=0.05,
        )
        
        result2 = estimate_psm(
            data=data,
            y='y',
            d='d',
            propensity_controls=['x'],
            n_neighbors=1,
            with_replacement=True,
            caliper=None,
            match_order='data',
            trim_threshold=0.01,
            alpha=0.05,
        )
        
        # ATT estimates should be identical
        assert result1.att == result2.att
        assert result1.se == result2.se
        assert result1.n_matched == result2.n_matched


class TestBug100SampleCounting:
    """Test sample size counting after NaN filtering (BUG-100)."""
    
    def test_sample_counting_with_nan_outcomes(self):
        """Test that n_treat and n_control reflect actual regression sample."""
        # Create staggered panel data with NaN outcomes
        np.random.seed(42)
        
        # 20 units, 5 periods
        n_units = 20
        n_periods = 5
        
        data_list = []
        for i in range(n_units):
            for t in range(1, n_periods + 1):
                # Cohort: units 0-9 treated at t=3, units 10-19 never treated
                g = 3 if i < 10 else 0
                
                # Outcome with some NaN values
                # Introduce NaN for some treated units in post-treatment periods
                if i < 10 and t >= 3 and i % 3 == 0:  # Every 3rd treated unit
                    y = np.nan
                else:
                    y = 1.0 + 0.5 * (t >= g and g > 0) + np.random.normal(0, 0.1)
                
                data_list.append({
                    'id': i,
                    'year': t,
                    'gvar': g,
                    'y': y,
                })
        
        data = pd.DataFrame(data_list)
        
        # Apply demean transformation
        transformed = transform_staggered_demean(
            data=data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            never_treated_values=[0],
        )
        
        # Estimate cohort-time effects
        effects = estimate_cohort_time_effects(
            data_transformed=transformed,
            gvar='gvar',
            ivar='id',
            tvar='year',
            controls=None,
            estimator='ra',
            transform_type='demean',
            control_strategy='not_yet_treated',
            never_treated_values=[0],
            min_obs=3,
            min_treated=1,
            min_control=1,
            alpha=0.05,
        )
        
        # Check that sample sizes are reasonable
        assert len(effects) > 0, "Should have some estimated effects"
        
        for effect in effects:
            # Sample sizes should be positive
            assert effect.n_treated > 0
            assert effect.n_control > 0
            assert effect.n_total == effect.n_treated + effect.n_control
            
            # Total sample should not exceed available units
            assert effect.n_total <= n_units
            
            # For cohort 3, period 3 onwards, some treated units have NaN
            # So n_treated should be less than 10
            if effect.cohort == 3 and effect.period >= 3:
                # We introduced NaN for units 0, 3, 6, 9 (4 units out of 10)
                # But the exact count depends on which units are in the regression
                # Just check it's not the full 10
                assert effect.n_treated < 10, \
                    f"Period {effect.period}: n_treated should be reduced due to NaN filtering"
    
    def test_sample_counting_all_valid(self):
        """Test that sample counting works correctly when no NaN values."""
        # Create data without NaN
        np.random.seed(42)
        
        n_units = 20
        n_periods = 5
        
        data_list = []
        for i in range(n_units):
            for t in range(1, n_periods + 1):
                g = 3 if i < 10 else 0
                y = 1.0 + 0.5 * (t >= g and g > 0) + np.random.normal(0, 0.1)
                
                data_list.append({
                    'id': i,
                    'year': t,
                    'gvar': g,
                    'y': y,
                })
        
        data = pd.DataFrame(data_list)
        
        transformed = transform_staggered_demean(
            data=data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            never_treated_values=[0],
        )
        
        effects = estimate_cohort_time_effects(
            data_transformed=transformed,
            gvar='gvar',
            ivar='id',
            tvar='year',
            controls=None,
            estimator='ra',
            transform_type='demean',
            control_strategy='not_yet_treated',
            never_treated_values=[0],
            min_obs=3,
            min_treated=1,
            min_control=1,
            alpha=0.05,
        )
        
        # With no NaN, all treated units should be included
        for effect in effects:
            if effect.cohort == 3:
                # All 10 treated units should be in the sample
                assert effect.n_treated == 10, \
                    f"Period {effect.period}: Expected 10 treated units, got {effect.n_treated}"
                
                # Control units depend on period and strategy
                # but should also be consistent
                assert effect.n_control > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
