"""
Test suite for bug fixes BUG-270 to BUG-273 (Round 80-81 code review).

This module tests the following bug fixes:
- BUG-270: construct_aggregated_outcome tvar parameter validation
- BUG-271: cohort_effects_to_dataframe None input handling
- BUG-273: Parallel mode memory optimization (verified via generator usage)
"""

import warnings
import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lwdid.staggered.aggregation import (
    construct_aggregated_outcome,
    cohort_effects_to_dataframe,
    aggregate_to_cohort,
    aggregate_to_overall,
    CohortEffect,
)
from lwdid.staggered.transformations import transform_staggered_demean


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def staggered_panel_data():
    """Create staggered DiD panel data with multiple cohorts."""
    np.random.seed(42)
    n_units = 30
    T = 6
    
    # Create cohort assignments: g=3, g=4, g=5, and never-treated (0)
    cohort_probs = [0.2, 0.25, 0.25, 0.3]  # [NT, g=3, g=4, g=5]
    cohort_values = [0, 3, 4, 5]
    
    unit_cohorts = np.random.choice(
        cohort_values, size=n_units, p=cohort_probs
    )
    
    data_rows = []
    for i, g in enumerate(unit_cohorts, start=1):
        for t in range(1, T + 1):
            treated = int(g > 0 and t >= g)
            y = 10 + 2 * t + np.random.normal(0, 1)
            if treated:
                y += 5  # Treatment effect
            data_rows.append({
                'id': i,
                'year': t,
                'gvar': g if g > 0 else 0,
                'y': y,
                'x1': np.random.normal(0, 1),
            })
    
    return pd.DataFrame(data_rows)


@pytest.fixture
def transformed_data(staggered_panel_data):
    """Apply demean transformation to staggered data."""
    return transform_staggered_demean(
        staggered_panel_data, 'y', 'id', 'year', 'gvar'
    )


@pytest.fixture
def sample_cohort_effects():
    """Create sample CohortEffect objects for testing."""
    return [
        CohortEffect(
            cohort=3, att=5.0, se=0.5, ci_lower=4.0, ci_upper=6.0,
            t_stat=10.0, pvalue=0.001, n_periods=3, n_units=5, n_control=10,
            df_resid=13, df_inference=13
        ),
        CohortEffect(
            cohort=4, att=4.5, se=0.6, ci_lower=3.3, ci_upper=5.7,
            t_stat=7.5, pvalue=0.002, n_periods=2, n_units=6, n_control=10,
            df_resid=14, df_inference=14
        ),
    ]


# =============================================================================
# BUG-270: construct_aggregated_outcome tvar validation
# =============================================================================

class TestBUG270TvarValidation:
    """Tests for BUG-270: tvar parameter validation in construct_aggregated_outcome."""

    def test_tvar_T_max_consistency_warning(self, transformed_data):
        """
        BUG-270: Verify warning when T_max differs from data's max time.
        
        When T_max parameter doesn't match the actual maximum time in data[tvar],
        a warning should be issued to alert the user of potential issues.
        """
        cohorts = [3, 4, 5]
        weights = {3: 0.3, 4: 0.4, 5: 0.3}
        
        # Data has T_max=6, but we pass T_max=10 (inconsistent)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            construct_aggregated_outcome(
                transformed_data, 'gvar', 'id', 'year',
                weights, cohorts, T_max=10, transform_type='demean'
            )
            
            # Check that a warning was issued about T_max inconsistency
            t_max_warnings = [
                warning for warning in w
                if 'T_max parameter' in str(warning.message)
                and 'differs from data' in str(warning.message)
            ]
            assert len(t_max_warnings) >= 1, (
                "Expected warning about T_max inconsistency"
            )

    def test_tvar_T_max_consistency_no_warning_when_match(self, transformed_data):
        """
        BUG-270: No warning when T_max matches data's max time.
        """
        cohorts = [3, 4, 5]
        weights = {3: 0.3, 4: 0.4, 5: 0.3}
        
        # Data has T_max=6, we pass T_max=6 (consistent)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            construct_aggregated_outcome(
                transformed_data, 'gvar', 'id', 'year',
                weights, cohorts, T_max=6, transform_type='demean'
            )
            
            # Filter for T_max related warnings only
            t_max_warnings = [
                warning for warning in w
                if 'T_max parameter' in str(warning.message)
                and 'differs from data' in str(warning.message)
            ]
            assert len(t_max_warnings) == 0, (
                f"Unexpected T_max warning when T_max matches: {t_max_warnings}"
            )

    def test_tvar_not_in_data_no_error(self, transformed_data):
        """
        BUG-270: No error when tvar column is not in data.
        
        This can happen when the transformed data doesn't include the original
        time column. The function should still work.
        """
        # Create data without 'year' column
        data_no_tvar = transformed_data.drop(columns=['year'])
        cohorts = [3, 4, 5]
        weights = {3: 0.3, 4: 0.4, 5: 0.3}
        
        # Should not raise an error
        result = construct_aggregated_outcome(
            data_no_tvar, 'gvar', 'id', 'nonexistent_tvar',
            weights, cohorts, T_max=6, transform_type='demean'
        )
        assert isinstance(result, pd.Series)


# =============================================================================
# BUG-271: cohort_effects_to_dataframe None handling
# =============================================================================

class TestBUG271NoneHandling:
    """Tests for BUG-271: cohort_effects_to_dataframe None input handling."""

    def test_none_input_raises_typeerror(self):
        """
        BUG-271: None input should raise TypeError.
        
        None indicates a programming error (forgot to return a list),
        so it should fail loudly with a clear error message.
        """
        with pytest.raises(TypeError) as exc_info:
            cohort_effects_to_dataframe(None)
        
        error_msg = str(exc_info.value)
        assert 'cannot be None' in error_msg
        assert 'empty list' in error_msg.lower()

    def test_empty_list_returns_empty_dataframe(self):
        """
        BUG-271: Empty list should return empty DataFrame with correct schema.
        
        Empty list is a valid case (no cohorts estimated), not an error.
        """
        df = cohort_effects_to_dataframe([])
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        # Check schema has expected columns
        expected_cols = [
            'cohort', 'att', 'se', 'ci_lower', 'ci_upper',
            't_stat', 'pvalue', 'n_periods', 'n_units', 'n_control',
            'df_resid', 'df_inference'
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_valid_list_returns_dataframe(self, sample_cohort_effects):
        """
        BUG-271: Valid list of CohortEffect objects should return DataFrame.
        """
        df = cohort_effects_to_dataframe(sample_cohort_effects)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert df['cohort'].tolist() == [3, 4]
        assert df['att'].tolist() == [5.0, 4.5]


# =============================================================================
# BUG-273: Parallel mode memory optimization verification
# =============================================================================

class TestBUG273ParallelMemory:
    """Tests for BUG-273: Parallel mode memory optimization."""

    def test_generator_pattern_in_parallel_mode(self):
        """
        BUG-273: Verify generator expression is used in parallel mode.
        
        This is a code inspection test - the fix uses a generator function
        instead of list comprehension to avoid creating all data copies upfront.
        """
        import inspect
        from lwdid.staggered.randomization import randomization_inference_staggered
        
        source = inspect.getsource(randomization_inference_staggered)
        
        # Verify the generator function pattern is present
        assert 'def worker_args_generator()' in source, (
            "Expected generator function 'worker_args_generator' in parallel mode"
        )
        
        # Verify chunksize=1 is used for memory efficiency
        assert 'chunksize=1' in source, (
            "Expected chunksize=1 in executor.map for memory efficiency"
        )
        
        # Verify the fix comment is present
        assert 'BUG-322 FIX' in source or 'BUG-273' in source, (
            "Expected fix comment documenting the memory optimization"
        )

    def test_serial_mode_memory_optimization(self):
        """
        BUG-273: Verify serial mode also has memory optimization.
        
        Serial mode should save/restore gvar column instead of copying
        entire DataFrame each iteration.
        """
        import inspect
        from lwdid.staggered.randomization import randomization_inference_staggered
        
        source = inspect.getsource(randomization_inference_staggered)
        
        # Verify serial mode saves original gvar
        assert 'original_gvar = data[gvar].copy()' in source, (
            "Expected gvar column save in serial mode"
        )
        
        # Verify restore pattern exists
        assert 'data[gvar] = original_gvar' in source, (
            "Expected gvar column restore in serial mode"
        )


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the bug fixes."""

    def test_aggregate_to_overall_uses_construct_aggregated_outcome(
        self, transformed_data
    ):
        """
        Integration: aggregate_to_overall internally calls construct_aggregated_outcome.
        
        Verify the function works correctly with the fixed tvar validation.
        """
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = aggregate_to_overall(
                transformed_data, 'gvar', 'id', 'year',
                transform_type='demean'
            )
        
        assert hasattr(result, 'att')
        assert hasattr(result, 'se')
        assert isinstance(result.cohort_weights, dict)

    def test_aggregate_to_cohort_to_dataframe_pipeline(self, transformed_data):
        """
        Integration: Full pipeline from aggregate_to_cohort to cohort_effects_to_dataframe.
        """
        cohort_effects = aggregate_to_cohort(
            transformed_data, 'gvar', 'id', 'year',
            cohorts=[3, 4, 5], T_max=6, transform_type='demean'
        )
        
        df = cohort_effects_to_dataframe(cohort_effects)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'cohort' in df.columns
        assert 'att' in df.columns


# =============================================================================
# Run tests if executed directly
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
