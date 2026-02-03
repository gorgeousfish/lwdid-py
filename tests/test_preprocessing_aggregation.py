"""
Tests for repeated cross-sectional data aggregation to panel format.

This module provides comprehensive tests for the preprocessing.aggregation module,
which implements the aggregation methodology from Lee & Wooldridge (2026), Section 3.

Test Categories:
1. Property-based tests (Hypothesis) - Universal correctness properties
2. Unit tests - Specific examples and edge cases
3. Numerical verification tests - Formula correctness with vibe-math MCP
4. Integration tests - lwdid pipeline compatibility

Formula Reference:
    Y_bar_st = sum_{i in (s,t)} w_ist * Y_ist,  where sum_{i in (s,t)} w_ist = 1
    ESS = (sum(w_i))^2 / sum(w_i^2)
"""

from __future__ import annotations

import warnings
from math import fsum

import numpy as np
import pandas as pd
import pytest

# Import from lwdid package
from lwdid.preprocessing import aggregate_to_panel, AggregationResult, CellStatistics
from lwdid.exceptions import (
    AggregationError,
    InvalidAggregationError,
    InsufficientCellSizeError,
    MissingRequiredColumnError,
)

# Try to import hypothesis for property-based tests
try:
    from hypothesis import given, settings, assume, HealthCheck
    from hypothesis import strategies as st
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    st = None  # Placeholder
    HealthCheck = None
    # Create dummy decorators
    def given(*args, **kwargs):
        def decorator(f):
            return pytest.mark.skip(reason="hypothesis not installed")(f)
        return decorator
    def settings(*args, **kwargs):
        return lambda f: f
    def assume(x):
        pass


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================

@pytest.fixture
def simple_rcs_data():
    """Simple repeated cross-section data for basic tests."""
    return pd.DataFrame({
        'state': ['CA', 'CA', 'CA', 'CA', 'TX', 'TX', 'TX', 'TX'],
        'year': [2000, 2000, 2001, 2001, 2000, 2000, 2001, 2001],
        'income': [50000, 55000, 60000, 65000, 45000, 48000, 52000, 55000],
        'weight': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        'treated': [1, 1, 1, 1, 0, 0, 0, 0],
        'gvar': [2001, 2001, 2001, 2001, 0, 0, 0, 0],
    })


@pytest.fixture
def weighted_rcs_data():
    """Repeated cross-section data with survey weights."""
    return pd.DataFrame({
        'state': ['CA', 'CA', 'CA', 'TX', 'TX', 'TX'],
        'year': [2000, 2000, 2000, 2000, 2000, 2000],
        'income': [50000, 60000, 70000, 40000, 50000, 60000],
        'weight': [1.0, 2.0, 3.0, 0.5, 1.5, 2.0],
    })


@pytest.fixture
def quarterly_rcs_data():
    """Repeated cross-section data with quarterly frequency."""
    data = []
    for state in ['CA', 'TX', 'NY']:
        for year in [2020, 2021]:
            for quarter in [1, 2, 3, 4]:
                for i in range(5):  # 5 obs per cell
                    data.append({
                        'state': state,
                        'year': year,
                        'quarter': quarter,
                        'income': 50000 + np.random.randn() * 5000,
                        'treated': 1 if state == 'CA' and year >= 2021 else 0,
                        'gvar': 2021 if state == 'CA' else 0,
                    })
    return pd.DataFrame(data)


def create_rcs_data(
    n_units: int = 5,
    n_periods: int = 3,
    n_obs_per_cell: int = 10,
    treatment_effect: float = 5.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create synthetic repeated cross-section data.
    
    Parameters
    ----------
    n_units : int
        Number of units (e.g., states).
    n_periods : int
        Number of time periods.
    n_obs_per_cell : int
        Number of observations per (unit, period) cell.
    treatment_effect : float
        Treatment effect for treated units.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    pd.DataFrame
        Synthetic RCS data.
    """
    np.random.seed(seed)
    data = []
    
    for unit in range(1, n_units + 1):
        # First half of units are treated starting at period 2
        is_treated_unit = unit <= n_units // 2
        gvar = 2 if is_treated_unit else 0
        
        for period in range(1, n_periods + 1):
            is_post = period >= 2 and is_treated_unit
            
            for _ in range(n_obs_per_cell):
                base_y = 100 + unit * 10 + period * 5
                y = base_y + np.random.randn() * 10
                if is_post:
                    y += treatment_effect
                
                data.append({
                    'unit': unit,
                    'period': period,
                    'y': y,
                    'treated': 1 if is_post else 0,
                    'gvar': gvar,
                    'weight': np.random.uniform(0.5, 2.0),
                })
    
    return pd.DataFrame(data)


# =============================================================================
# Property-Based Tests (Hypothesis)
# =============================================================================

@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestPropertyBasedWeightedMean:
    """Property tests for weighted mean formula correctness (Property 1)."""
    
    @given(
        values=st.lists(
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            min_size=1, max_size=100
        ) if HYPOTHESIS_AVAILABLE else st,
        weights=st.lists(
            st.floats(min_value=0.01, max_value=100, allow_nan=False, allow_infinity=False),
            min_size=1, max_size=100
        ) if HYPOTHESIS_AVAILABLE else st,
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow] if HealthCheck else [])
    def test_weighted_mean_formula(self, values, weights):
        """Property 1: Weighted mean equals sum(w_i * Y_i) / sum(w_i)."""
        # Ensure same length
        n = min(len(values), len(weights))
        assume(n >= 1)
        values = values[:n]
        weights = weights[:n]
        
        # Create test data
        data = pd.DataFrame({
            'unit': [1] * n,
            'period': [1] * n,
            'y': values,
            'weight': weights,
        })
        
        # Aggregate
        result = aggregate_to_panel(data, 'unit', 'period', 'y', weight_var='weight')
        
        # Manual calculation
        w_sum = fsum(weights)
        expected_mean = fsum(w * v for w, v in zip(weights, values)) / w_sum
        
        # Verify
        actual_mean = result.panel_data['y'].iloc[0]
        assert np.isclose(actual_mean, expected_mean, rtol=1e-10), \
            f"Expected {expected_mean}, got {actual_mean}"


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestPropertyBasedEqualWeights:
    """Property tests for equal weights default (Property 2)."""
    
    @given(
        values=st.lists(
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            min_size=1, max_size=100
        ) if HYPOTHESIS_AVAILABLE else st,
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow] if HealthCheck else [])
    def test_equal_weights_equals_simple_mean(self, values):
        """Property 2: Without weights, result equals simple arithmetic mean."""
        assume(len(values) >= 1)
        
        data = pd.DataFrame({
            'unit': [1] * len(values),
            'period': [1] * len(values),
            'y': values,
        })
        
        result = aggregate_to_panel(data, 'unit', 'period', 'y')
        
        expected_mean = np.mean(values)
        actual_mean = result.panel_data['y'].iloc[0]
        
        assert np.isclose(actual_mean, expected_mean, rtol=1e-10), \
            f"Expected {expected_mean}, got {actual_mean}"


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestPropertyBasedWeightNormalization:
    """Property tests for weight normalization (Property 3)."""
    
    @given(
        scale=st.floats(min_value=0.001, max_value=1000, allow_nan=False, allow_infinity=False) if HYPOTHESIS_AVAILABLE else st,
    )
    @settings(max_examples=50)
    def test_weight_scale_invariance(self, scale):
        """Property 3: Scaling all weights by constant doesn't change result."""
        values = [10.0, 20.0, 30.0]
        base_weights = [1.0, 2.0, 3.0]
        scaled_weights = [w * scale for w in base_weights]
        
        # Base weights
        data1 = pd.DataFrame({
            'unit': [1, 1, 1],
            'period': [1, 1, 1],
            'y': values,
            'weight': base_weights,
        })
        result1 = aggregate_to_panel(data1, 'unit', 'period', 'y', weight_var='weight')
        
        # Scaled weights
        data2 = pd.DataFrame({
            'unit': [1, 1, 1],
            'period': [1, 1, 1],
            'y': values,
            'weight': scaled_weights,
        })
        result2 = aggregate_to_panel(data2, 'unit', 'period', 'y', weight_var='weight')
        
        assert np.isclose(
            result1.panel_data['y'].iloc[0],
            result2.panel_data['y'].iloc[0],
            rtol=1e-10
        )


class TestPropertyBasedPanelFormat:
    """Property tests for panel format output (Property 4)."""
    
    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
    @given(
        n_units=st.integers(min_value=1, max_value=10) if HYPOTHESIS_AVAILABLE else st,
        n_periods=st.integers(min_value=1, max_value=10) if HYPOTHESIS_AVAILABLE else st,
        n_obs_per_cell=st.integers(min_value=1, max_value=20) if HYPOTHESIS_AVAILABLE else st,
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow] if HealthCheck else [])
    def test_output_is_balanced_panel(self, n_units, n_periods, n_obs_per_cell):
        """Property 4: Output has exactly one row per (unit, period) combination."""
        data = create_rcs_data(n_units, n_periods, n_obs_per_cell)
        result = aggregate_to_panel(data, 'unit', 'period', 'y')
        
        # Check unique combinations
        expected_rows = n_units * n_periods
        assert len(result.panel_data) == expected_rows, \
            f"Expected {expected_rows} rows, got {len(result.panel_data)}"
        
        # Check no duplicates
        duplicates = result.panel_data.duplicated(subset=['unit', 'period'])
        assert not duplicates.any(), "Found duplicate (unit, period) combinations"


class TestPropertyBasedTreatmentConsistency:
    """Property tests for treatment consistency validation (Property 5)."""
    
    def test_treatment_varies_within_cell_raises_error(self):
        """Property 5: Varying treatment within cell raises InvalidAggregationError."""
        data = pd.DataFrame({
            'unit': [1, 1, 1],
            'period': [1, 1, 1],
            'y': [10, 20, 30],
            'treated': [0, 1, 0],  # Varies within cell!
        })
        
        with pytest.raises(InvalidAggregationError) as excinfo:
            aggregate_to_panel(data, 'unit', 'period', 'y', treatment_var='treated')
        
        assert "varies within" in str(excinfo.value).lower()
    
    def test_treatment_constant_within_cell_succeeds(self):
        """Property 5: Constant treatment within cell succeeds."""
        data = pd.DataFrame({
            'unit': [1, 1, 2, 2],
            'period': [1, 1, 1, 1],
            'y': [10, 20, 30, 40],
            'treated': [1, 1, 0, 0],  # Constant within each cell
        })
        
        result = aggregate_to_panel(data, 'unit', 'period', 'y', treatment_var='treated')
        assert len(result.panel_data) == 2


class TestPropertyBasedGvarConsistency:
    """Property tests for gvar consistency validation (Property 6)."""
    
    def test_gvar_varies_within_unit_raises_error(self):
        """Property 6: Varying gvar within unit raises InvalidAggregationError."""
        data = pd.DataFrame({
            'unit': [1, 1, 1, 1],
            'period': [1, 1, 2, 2],
            'y': [10, 20, 30, 40],
            'gvar': [2020, 2020, 2021, 2021],  # Varies within unit!
        })
        
        with pytest.raises(InvalidAggregationError) as excinfo:
            aggregate_to_panel(data, 'unit', 'period', 'y', gvar='gvar')
        
        assert "varies within" in str(excinfo.value).lower()
    
    def test_gvar_constant_within_unit_succeeds(self):
        """Property 6: Constant gvar within unit succeeds."""
        data = pd.DataFrame({
            'unit': [1, 1, 2, 2],
            'period': [1, 2, 1, 2],
            'y': [10, 20, 30, 40],
            'gvar': [2020, 2020, 0, 0],  # Constant within each unit
        })
        
        result = aggregate_to_panel(data, 'unit', 'period', 'y', gvar='gvar')
        assert len(result.panel_data) == 4


class TestPropertyBasedQuarterlyAggregation:
    """Property tests for quarterly aggregation formula (Property 7)."""
    
    def test_quarterly_aggregation_creates_correct_cells(self):
        """Property 7: Quarterly aggregation creates (unit, year, quarter) cells."""
        data = pd.DataFrame({
            'state': ['CA'] * 8 + ['TX'] * 8,
            'year': [2020] * 4 + [2021] * 4 + [2020] * 4 + [2021] * 4,
            'quarter': [1, 2, 3, 4] * 4,
            'income': np.random.randn(16) * 1000 + 50000,
        })
        
        result = aggregate_to_panel(
            data, 'state', ['year', 'quarter'], 'income',
            frequency='quarterly'
        )
        
        # Should have 2 states * 2 years * 4 quarters = 16 cells
        assert len(result.panel_data) == 16
        assert 'year' in result.panel_data.columns
        assert 'quarter' in result.panel_data.columns


class TestPropertyBasedCellSizeFiltering:
    """Property tests for cell size filtering (Property 8)."""
    
    def test_cells_below_threshold_excluded(self):
        """Property 8: Cells below min_cell_size are excluded."""
        data = pd.DataFrame({
            'unit': [1, 1, 1, 2, 2],  # Unit 1: 3 obs, Unit 2: 2 obs
            'period': [1, 1, 1, 1, 1],
            'y': [10, 20, 30, 40, 50],
        })
        
        # min_cell_size=3 should exclude unit 2
        result = aggregate_to_panel(data, 'unit', 'period', 'y', min_cell_size=3)
        
        assert len(result.panel_data) == 1
        assert result.panel_data['unit'].iloc[0] == 1
        assert result.n_excluded_cells == 1
    
    def test_all_cells_below_threshold_raises_error(self):
        """Property 8: All cells below threshold raises InsufficientCellSizeError."""
        data = pd.DataFrame({
            'unit': [1, 2],
            'period': [1, 1],
            'y': [10, 20],
        })
        
        with pytest.raises(InsufficientCellSizeError):
            aggregate_to_panel(data, 'unit', 'period', 'y', min_cell_size=5)


class TestPropertyBasedNonNegativeWeights:
    """Property tests for non-negative weight validation (Property 10)."""
    
    def test_negative_weights_raise_error(self):
        """Property 10: Negative weights raise ValueError."""
        data = pd.DataFrame({
            'unit': [1, 1, 1],
            'period': [1, 1, 1],
            'y': [10, 20, 30],
            'weight': [1.0, -0.5, 2.0],  # Negative weight!
        })
        
        with pytest.raises(ValueError) as excinfo:
            aggregate_to_panel(data, 'unit', 'period', 'y', weight_var='weight')
        
        assert "negative" in str(excinfo.value).lower()
    
    def test_zero_weights_allowed(self):
        """Property 10: Zero weights are allowed."""
        data = pd.DataFrame({
            'unit': [1, 1, 1],
            'period': [1, 1, 1],
            'y': [10, 20, 30],
            'weight': [1.0, 0.0, 2.0],  # Zero weight is OK
        })
        
        result = aggregate_to_panel(data, 'unit', 'period', 'y', weight_var='weight')
        # With weights [1, 0, 2], normalized = [1/3, 0, 2/3]
        # Mean = 10*1/3 + 20*0 + 30*2/3 = 10/3 + 20 = 23.33...
        expected = (10 * 1 + 30 * 2) / 3
        assert np.isclose(result.panel_data['y'].iloc[0], expected, rtol=1e-10)


class TestPropertyBasedAllNaNExclusion:
    """Property tests for all-NaN cell exclusion (Property 12)."""
    
    def test_all_nan_cells_excluded(self):
        """Property 12: Cells with all-NaN outcomes are excluded."""
        data = pd.DataFrame({
            'unit': [1, 1, 2, 2],
            'period': [1, 1, 1, 1],
            'y': [10.0, 20.0, np.nan, np.nan],  # Unit 2 is all NaN
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = aggregate_to_panel(data, 'unit', 'period', 'y')
        
        assert len(result.panel_data) == 1
        assert result.panel_data['unit'].iloc[0] == 1
        assert result.n_excluded_cells == 1
    
    def test_partial_nan_cells_included(self):
        """Property 12: Cells with some NaN values are included (NaN excluded from mean)."""
        data = pd.DataFrame({
            'unit': [1, 1, 1],
            'period': [1, 1, 1],
            'y': [10.0, np.nan, 30.0],  # One NaN
        })
        
        result = aggregate_to_panel(data, 'unit', 'period', 'y')
        
        # Mean of [10, 30] = 20
        assert np.isclose(result.panel_data['y'].iloc[0], 20.0, rtol=1e-10)


# =============================================================================
# Unit Tests - Input Validation
# =============================================================================

class TestInputValidation:
    """Unit tests for input validation (Task 12.1)."""
    
    def test_non_dataframe_raises_typeerror(self):
        """Test that non-DataFrame input raises TypeError."""
        with pytest.raises(TypeError) as excinfo:
            aggregate_to_panel([1, 2, 3], 'unit', 'period', 'y')
        assert "DataFrame" in str(excinfo.value)
    
    def test_empty_dataframe_raises_valueerror(self):
        """Test that empty DataFrame raises ValueError."""
        data = pd.DataFrame(columns=['unit', 'period', 'y'])
        with pytest.raises(ValueError) as excinfo:
            aggregate_to_panel(data, 'unit', 'period', 'y')
        assert "empty" in str(excinfo.value).lower()
    
    def test_missing_unit_column_raises_error(self):
        """Test that missing unit column raises MissingRequiredColumnError."""
        data = pd.DataFrame({'period': [1], 'y': [10]})
        with pytest.raises(MissingRequiredColumnError):
            aggregate_to_panel(data, 'unit', 'period', 'y')
    
    def test_missing_time_column_raises_error(self):
        """Test that missing time column raises MissingRequiredColumnError."""
        data = pd.DataFrame({'unit': [1], 'y': [10]})
        with pytest.raises(MissingRequiredColumnError):
            aggregate_to_panel(data, 'unit', 'period', 'y')
    
    def test_missing_outcome_column_raises_error(self):
        """Test that missing outcome column raises MissingRequiredColumnError."""
        data = pd.DataFrame({'unit': [1], 'period': [1]})
        with pytest.raises(MissingRequiredColumnError):
            aggregate_to_panel(data, 'unit', 'period', 'y')
    
    def test_non_numeric_outcome_raises_valueerror(self):
        """Test that non-numeric outcome raises ValueError."""
        data = pd.DataFrame({
            'unit': [1, 1],
            'period': [1, 1],
            'y': ['a', 'b'],  # String outcome
        })
        with pytest.raises(ValueError) as excinfo:
            aggregate_to_panel(data, 'unit', 'period', 'y')
        assert "numeric" in str(excinfo.value).lower()


# =============================================================================
# Unit Tests - Weight Handling
# =============================================================================

class TestWeightHandling:
    """Unit tests for weight handling (Task 12.2)."""
    
    def test_equal_weights_default(self, simple_rcs_data):
        """Test that equal weights are used by default."""
        result = aggregate_to_panel(
            simple_rcs_data, 'state', 'year', 'income'
        )
        
        # CA 2000: mean of [50000, 55000] = 52500
        ca_2000 = result.panel_data[
            (result.panel_data['state'] == 'CA') & 
            (result.panel_data['year'] == 2000)
        ]['income'].iloc[0]
        
        assert np.isclose(ca_2000, 52500.0, rtol=1e-10)
    
    def test_survey_weights_applied(self, weighted_rcs_data):
        """Test that survey weights are correctly applied."""
        result = aggregate_to_panel(
            weighted_rcs_data, 'state', 'year', 'income', weight_var='weight'
        )
        
        # CA: weights [1, 2, 3], values [50000, 60000, 70000]
        # Weighted mean = (1*50000 + 2*60000 + 3*70000) / (1+2+3)
        #               = (50000 + 120000 + 210000) / 6 = 380000 / 6 = 63333.33...
        ca_mean = result.panel_data[
            result.panel_data['state'] == 'CA'
        ]['income'].iloc[0]
        
        expected = (1*50000 + 2*60000 + 3*70000) / 6
        assert np.isclose(ca_mean, expected, rtol=1e-10)
    
    def test_missing_weights_warning(self):
        """Test that missing weights issue a warning."""
        data = pd.DataFrame({
            'unit': [1, 1, 1],
            'period': [1, 1, 1],
            'y': [10, 20, 30],
            'weight': [1.0, np.nan, 2.0],  # Missing weight
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = aggregate_to_panel(data, 'unit', 'period', 'y', weight_var='weight')
            
            # Check warning was issued
            weight_warnings = [x for x in w if "missing weights" in str(x.message).lower()]
            assert len(weight_warnings) >= 1
        
        # Mean should be computed from non-missing weights
        # weights [1, 2], values [10, 30] -> (1*10 + 2*30) / 3 = 70/3 = 23.33...
        expected = (1*10 + 2*30) / 3
        assert np.isclose(result.panel_data['y'].iloc[0], expected, rtol=1e-10)


# =============================================================================
# Unit Tests - Treatment Validation
# =============================================================================

class TestTreatmentValidation:
    """Unit tests for treatment validation (Task 12.3)."""
    
    def test_constant_treatment_succeeds(self, simple_rcs_data):
        """Test that constant treatment within cells succeeds."""
        result = aggregate_to_panel(
            simple_rcs_data, 'state', 'year', 'income',
            treatment_var='treated'
        )
        
        assert 'treated' in result.panel_data.columns
        
        # CA should be treated=1, TX should be treated=0
        ca_treated = result.panel_data[
            result.panel_data['state'] == 'CA'
        ]['treated'].unique()
        tx_treated = result.panel_data[
            result.panel_data['state'] == 'TX'
        ]['treated'].unique()
        
        assert list(ca_treated) == [1]
        assert list(tx_treated) == [0]
    
    def test_gvar_preserved(self, simple_rcs_data):
        """Test that gvar is preserved in output."""
        result = aggregate_to_panel(
            simple_rcs_data, 'state', 'year', 'income',
            gvar='gvar'
        )
        
        assert 'gvar' in result.panel_data.columns
        
        # CA should have gvar=2001, TX should have gvar=0
        ca_gvar = result.panel_data[
            result.panel_data['state'] == 'CA'
        ]['gvar'].unique()
        tx_gvar = result.panel_data[
            result.panel_data['state'] == 'TX'
        ]['gvar'].unique()
        
        assert list(ca_gvar) == [2001]
        assert list(tx_gvar) == [0]


# =============================================================================
# Unit Tests - High-Frequency Aggregation
# =============================================================================

class TestHighFrequencyAggregation:
    """Unit tests for high-frequency aggregation (Task 12.4)."""
    
    def test_quarterly_aggregation(self, quarterly_rcs_data):
        """Test quarterly aggregation creates correct structure."""
        result = aggregate_to_panel(
            quarterly_rcs_data, 'state', ['year', 'quarter'], 'income',
            frequency='quarterly'
        )
        
        # 3 states * 2 years * 4 quarters = 24 cells
        assert len(result.panel_data) == 24
        assert result.frequency == 'quarterly'
        assert 'year' in result.panel_data.columns
        assert 'quarter' in result.panel_data.columns
    
    def test_monthly_aggregation(self):
        """Test monthly aggregation."""
        data = pd.DataFrame({
            'state': ['CA'] * 12,
            'year': [2020] * 12,
            'month': list(range(1, 13)),
            'income': np.random.randn(12) * 1000 + 50000,
        })
        
        result = aggregate_to_panel(
            data, 'state', ['year', 'month'], 'income',
            frequency='monthly'
        )
        
        assert len(result.panel_data) == 12
        assert result.frequency == 'monthly'
    
    def test_weekly_aggregation(self):
        """Test weekly aggregation."""
        data = pd.DataFrame({
            'state': ['CA'] * 4,
            'year': [2020] * 4,
            'week': [1, 2, 3, 4],
            'income': [50000, 51000, 52000, 53000],
        })
        
        result = aggregate_to_panel(
            data, 'state', ['year', 'week'], 'income',
            frequency='weekly'
        )
        
        assert len(result.panel_data) == 4
        assert result.frequency == 'weekly'


# =============================================================================
# Unit Tests - Edge Cases
# =============================================================================

class TestEdgeCases:
    """Unit tests for edge cases (Task 12.5)."""
    
    def test_single_observation_cell(self):
        """Test cell with single observation."""
        data = pd.DataFrame({
            'unit': [1],
            'period': [1],
            'y': [100.0],
        })
        
        result = aggregate_to_panel(data, 'unit', 'period', 'y')
        
        assert len(result.panel_data) == 1
        assert result.panel_data['y'].iloc[0] == 100.0
        assert result.panel_data['_n_obs'].iloc[0] == 1
    
    def test_single_unit(self):
        """Test data with single unit."""
        data = pd.DataFrame({
            'unit': [1, 1, 1],
            'period': [1, 2, 3],
            'y': [10, 20, 30],
        })
        
        result = aggregate_to_panel(data, 'unit', 'period', 'y')
        
        assert result.n_units == 1
        assert result.n_periods == 3
        assert len(result.panel_data) == 3
    
    def test_single_period(self):
        """Test data with single period."""
        data = pd.DataFrame({
            'unit': [1, 2, 3],
            'period': [1, 1, 1],
            'y': [10, 20, 30],
        })
        
        result = aggregate_to_panel(data, 'unit', 'period', 'y')
        
        assert result.n_units == 3
        assert result.n_periods == 1
        assert len(result.panel_data) == 3
    
    def test_variance_single_obs_is_nan(self):
        """Test that variance is NaN for single-observation cells."""
        data = pd.DataFrame({
            'unit': [1],
            'period': [1],
            'y': [100.0],
        })
        
        result = aggregate_to_panel(
            data, 'unit', 'period', 'y', compute_variance=True
        )
        
        # Variance should be NaN for single observation
        var_col = 'y_var'
        if var_col in result.panel_data.columns:
            assert pd.isna(result.panel_data[var_col].iloc[0])


# =============================================================================
# Numerical Verification Tests (Task 13)
# =============================================================================

class TestNumericalVerification:
    """Numerical verification tests using manual calculations and vibe-math MCP."""
    
    def test_weighted_mean_formula_manual(self):
        """Verify weighted mean formula: Y_bar = sum(w_i * Y_i) / sum(w_i)."""
        # Test data
        values = [100.0, 200.0, 300.0, 400.0]
        weights = [1.0, 2.0, 3.0, 4.0]
        
        data = pd.DataFrame({
            'unit': [1] * 4,
            'period': [1] * 4,
            'y': values,
            'weight': weights,
        })
        
        result = aggregate_to_panel(data, 'unit', 'period', 'y', weight_var='weight')
        
        # Manual calculation using fsum for numerical stability
        w_sum = fsum(weights)  # 10.0
        weighted_sum = fsum(w * v for w, v in zip(weights, values))
        # = 1*100 + 2*200 + 3*300 + 4*400 = 100 + 400 + 900 + 1600 = 3000
        expected = weighted_sum / w_sum  # 3000 / 10 = 300
        
        assert np.isclose(result.panel_data['y'].iloc[0], expected, rtol=1e-12)
        assert np.isclose(result.panel_data['y'].iloc[0], 300.0, rtol=1e-12)
    
    def test_ess_formula_manual(self):
        """Verify ESS formula: ESS = (sum(w_i))^2 / sum(w_i^2)."""
        # Test data with known ESS
        weights = [1.0, 2.0, 3.0, 4.0]
        
        data = pd.DataFrame({
            'unit': [1] * 4,
            'period': [1] * 4,
            'y': [100.0, 200.0, 300.0, 400.0],
            'weight': weights,
        })
        
        result = aggregate_to_panel(data, 'unit', 'period', 'y', weight_var='weight')
        
        # Manual ESS calculation
        w_sum = fsum(weights)  # 10.0
        w_sq_sum = fsum(w**2 for w in weights)  # 1 + 4 + 9 + 16 = 30
        expected_ess = (w_sum ** 2) / w_sq_sum  # 100 / 30 = 3.333...
        
        assert '_ess' in result.panel_data.columns
        assert np.isclose(result.panel_data['_ess'].iloc[0], expected_ess, rtol=1e-12)
        assert np.isclose(result.panel_data['_ess'].iloc[0], 100/30, rtol=1e-12)
    
    def test_equal_weights_ess_equals_n(self):
        """Verify ESS = n when all weights are equal."""
        n = 10
        data = pd.DataFrame({
            'unit': [1] * n,
            'period': [1] * n,
            'y': list(range(n)),
            'weight': [1.0] * n,  # Equal weights
        })
        
        result = aggregate_to_panel(data, 'unit', 'period', 'y', weight_var='weight')
        
        # With equal weights, ESS = n
        assert np.isclose(result.panel_data['_ess'].iloc[0], n, rtol=1e-12)
    
    def test_variance_formula_manual(self):
        """Verify weighted variance formula: Var = sum(w_i * (Y_i - Y_bar)^2)."""
        values = [10.0, 20.0, 30.0]
        weights = [1.0, 1.0, 1.0]  # Equal weights for simplicity
        
        data = pd.DataFrame({
            'unit': [1] * 3,
            'period': [1] * 3,
            'y': values,
            'weight': weights,
        })
        
        result = aggregate_to_panel(
            data, 'unit', 'period', 'y', 
            weight_var='weight', compute_variance=True
        )
        
        # Manual calculation
        # Mean = (10 + 20 + 30) / 3 = 20
        # Normalized weights = [1/3, 1/3, 1/3]
        # Variance = (1/3)*(10-20)^2 + (1/3)*(20-20)^2 + (1/3)*(30-20)^2
        #          = (1/3)*100 + 0 + (1/3)*100 = 200/3 = 66.67
        expected_var = (100 + 0 + 100) / 3
        
        assert 'y_var' in result.panel_data.columns
        assert np.isclose(result.panel_data['y_var'].iloc[0], expected_var, rtol=1e-10)
    
    def test_numerical_stability_large_weights(self):
        """Test numerical stability with large weight ratios (1e6 : 1)."""
        data = pd.DataFrame({
            'unit': [1, 1, 1],
            'period': [1, 1, 1],
            'y': [100.0, 200.0, 300.0],
            'weight': [1e6, 1.0, 1.0],  # Large weight ratio
        })
        
        result = aggregate_to_panel(data, 'unit', 'period', 'y', weight_var='weight')
        
        # With weight 1e6 on first value, mean should be very close to 100
        # Normalized weights ≈ [1, 1e-6, 1e-6]
        # Mean ≈ 100 * 1 + 200 * 1e-6 + 300 * 1e-6 ≈ 100.0005
        assert result.panel_data['y'].iloc[0] < 101  # Should be close to 100
        assert result.panel_data['y'].iloc[0] > 99
    
    def test_numerical_stability_small_weights(self):
        """Test numerical stability with very small weights (1e-10)."""
        data = pd.DataFrame({
            'unit': [1, 1, 1],
            'period': [1, 1, 1],
            'y': [100.0, 200.0, 300.0],
            'weight': [1e-10, 1e-10, 1e-10],  # Very small but equal
        })
        
        result = aggregate_to_panel(data, 'unit', 'period', 'y', weight_var='weight')
        
        # Equal weights should give simple mean = 200
        assert np.isclose(result.panel_data['y'].iloc[0], 200.0, rtol=1e-10)


# =============================================================================
# Integration Tests - lwdid Compatibility (Task 8)
# =============================================================================

class TestLwdidIntegration:
    """Integration tests for lwdid pipeline compatibility."""
    
    def test_output_columns_compatible(self):
        """Test that output columns are compatible with lwdid."""
        data = create_rcs_data(n_units=5, n_periods=4, n_obs_per_cell=10)
        
        result = aggregate_to_panel(
            data, 'unit', 'period', 'y',
            treatment_var='treated', gvar='gvar'
        )
        
        panel = result.panel_data
        
        # Required columns for lwdid
        assert 'unit' in panel.columns  # ivar
        assert 'period' in panel.columns  # tvar
        assert 'y' in panel.columns  # outcome
        assert 'treated' in panel.columns  # treatment indicator
        assert 'gvar' in panel.columns  # treatment timing
    
    def test_output_dtypes_compatible(self):
        """Test that output dtypes are compatible with lwdid."""
        data = create_rcs_data(n_units=5, n_periods=4, n_obs_per_cell=10)
        
        result = aggregate_to_panel(
            data, 'unit', 'period', 'y',
            treatment_var='treated', gvar='gvar'
        )
        
        panel = result.panel_data
        
        # Outcome should be numeric
        assert pd.api.types.is_numeric_dtype(panel['y'])
        
        # Treatment should be numeric (0/1)
        assert pd.api.types.is_numeric_dtype(panel['treated'])
        
        # gvar should be numeric
        assert pd.api.types.is_numeric_dtype(panel['gvar'])
    
    def test_control_variables_aggregated(self):
        """Test that control variables are properly aggregated."""
        data = pd.DataFrame({
            'unit': [1, 1, 2, 2],
            'period': [1, 1, 1, 1],
            'y': [100, 200, 300, 400],
            'x1': [10, 20, 30, 40],
            'x2': [1.0, 2.0, 3.0, 4.0],
        })
        
        result = aggregate_to_panel(
            data, 'unit', 'period', 'y',
            controls=['x1', 'x2']
        )
        
        panel = result.panel_data
        
        # Control variables should be in output
        assert 'x1' in panel.columns
        assert 'x2' in panel.columns
        
        # Unit 1: mean of [10, 20] = 15
        assert np.isclose(
            panel[panel['unit'] == 1]['x1'].iloc[0], 15.0, rtol=1e-10
        )
        
        # Unit 2: mean of [30, 40] = 35
        assert np.isclose(
            panel[panel['unit'] == 2]['x1'].iloc[0], 35.0, rtol=1e-10
        )


# =============================================================================
# AggregationResult Tests
# =============================================================================

class TestAggregationResult:
    """Tests for AggregationResult class methods."""
    
    def test_summary_method(self, simple_rcs_data):
        """Test summary() method returns formatted string."""
        result = aggregate_to_panel(simple_rcs_data, 'state', 'year', 'income')
        
        summary = result.summary()
        
        assert isinstance(summary, str)
        assert "Aggregation Summary" in summary
        assert "Original observations" in summary
        assert "Output cells" in summary
        assert "Units" in summary
        assert "Periods" in summary
    
    def test_to_dict_method(self, simple_rcs_data):
        """Test to_dict() method returns dictionary."""
        result = aggregate_to_panel(simple_rcs_data, 'state', 'year', 'income')
        
        d = result.to_dict()
        
        assert isinstance(d, dict)
        assert 'n_original_obs' in d
        assert 'n_cells' in d
        assert 'n_units' in d
        assert 'n_periods' in d
        assert 'unit_var' in d
        assert 'time_var' in d
        assert 'outcome_var' in d
    
    def test_to_csv_method(self, simple_rcs_data, tmp_path):
        """Test to_csv() method exports data correctly."""
        result = aggregate_to_panel(simple_rcs_data, 'state', 'year', 'income')
        
        csv_path = tmp_path / "test_output.csv"
        result.to_csv(str(csv_path), include_metadata=True)
        
        assert csv_path.exists()
        
        # Read back and verify
        with open(csv_path, 'r') as f:
            content = f.read()
        
        assert "# Aggregation Metadata" in content
        assert "state" in content
        assert "year" in content
        assert "income" in content
    
    def test_cell_stats_dataframe(self, simple_rcs_data):
        """Test cell_stats DataFrame is populated correctly."""
        result = aggregate_to_panel(simple_rcs_data, 'state', 'year', 'income')
        
        assert isinstance(result.cell_stats, pd.DataFrame)
        assert len(result.cell_stats) == result.n_cells
        assert 'unit' in result.cell_stats.columns
        assert 'n_obs' in result.cell_stats.columns
        assert 'outcome_mean' in result.cell_stats.columns


# =============================================================================
# Monte Carlo Simulation Tests (Task 15)
# =============================================================================

class TestMonteCarloSimulation:
    """Monte Carlo simulation tests for statistical properties."""
    
    def test_aggregation_preserves_mean(self):
        """Test that aggregation preserves population mean across replications."""
        np.random.seed(42)
        n_reps = 100
        true_mean = 100.0
        
        aggregated_means = []
        
        for _ in range(n_reps):
            # Generate data with known mean
            data = pd.DataFrame({
                'unit': [1] * 50 + [2] * 50,
                'period': [1] * 100,
                'y': np.random.normal(true_mean, 10, 100),
            })
            
            result = aggregate_to_panel(data, 'unit', 'period', 'y')
            overall_mean = result.panel_data['y'].mean()
            aggregated_means.append(overall_mean)
        
        # Mean of aggregated means should be close to true mean
        mean_of_means = np.mean(aggregated_means)
        assert np.isclose(mean_of_means, true_mean, atol=2.0), \
            f"Mean of means {mean_of_means} not close to true mean {true_mean}"
    
    def test_weighted_aggregation_unbiased(self):
        """Test that weighted aggregation is unbiased."""
        np.random.seed(123)
        n_reps = 100
        true_mean = 50.0
        
        weighted_means = []
        
        for _ in range(n_reps):
            # Generate data with random weights
            n = 100
            data = pd.DataFrame({
                'unit': [1] * n,
                'period': [1] * n,
                'y': np.random.normal(true_mean, 5, n),
                'weight': np.random.uniform(0.5, 2.0, n),
            })
            
            result = aggregate_to_panel(
                data, 'unit', 'period', 'y', weight_var='weight'
            )
            weighted_means.append(result.panel_data['y'].iloc[0])
        
        # Mean of weighted means should be close to true mean
        mean_of_weighted = np.mean(weighted_means)
        assert np.isclose(mean_of_weighted, true_mean, atol=1.0), \
            f"Mean of weighted means {mean_of_weighted} not close to true mean {true_mean}"


# =============================================================================
# Empirical Data Tests (Task 16)
# =============================================================================

class TestEmpiricalData:
    """Tests with simulated empirical-like data."""
    
    def test_realistic_survey_data(self):
        """Test with realistic survey-like data structure."""
        np.random.seed(42)
        
        # Simulate CPS-like data: states, years, individuals
        data = []
        for state in range(1, 11):  # 10 states
            gvar = 2015 if state <= 3 else 0  # 3 treated states
            
            for year in range(2010, 2020):  # 10 years
                n_obs = np.random.randint(50, 200)  # Variable cell sizes
                
                for _ in range(n_obs):
                    base_income = 40000 + state * 2000 + (year - 2010) * 1000
                    income = base_income + np.random.normal(0, 10000)
                    
                    # Treatment effect after 2015 for treated states
                    if state <= 3 and year >= 2015:
                        income += 5000
                    
                    data.append({
                        'state': state,
                        'year': year,
                        'income': income,
                        'weight': np.random.uniform(0.5, 2.0),
                        'treated': 1 if (state <= 3 and year >= 2015) else 0,
                        'gvar': gvar,
                    })
        
        df = pd.DataFrame(data)
        
        # Aggregate
        result = aggregate_to_panel(
            df, 'state', 'year', 'income',
            weight_var='weight',
            treatment_var='treated',
            gvar='gvar',
        )
        
        # Verify structure
        assert result.n_units == 10
        assert result.n_periods == 10
        assert result.n_cells == 100
        
        # Verify treatment assignment
        treated_states = result.panel_data[
            result.panel_data['gvar'] == 2015
        ]['state'].unique()
        assert len(treated_states) == 3
    
    def test_full_pipeline_integration(self):
        """Test full pipeline from RCS to aggregated panel."""
        np.random.seed(42)
        
        # Create RCS data
        data = create_rcs_data(
            n_units=10, n_periods=5, n_obs_per_cell=20,
            treatment_effect=10.0
        )
        
        # Aggregate
        result = aggregate_to_panel(
            data, 'unit', 'period', 'y',
            weight_var='weight',
            treatment_var='treated',
            gvar='gvar',
            compute_variance=True,
        )
        
        # Verify all expected outputs
        assert result.n_units == 10
        assert result.n_periods == 5
        assert result.n_cells == 50
        assert result.n_original_obs == 10 * 5 * 20
        
        # Verify metadata
        assert result.min_cell_size >= 1
        assert result.max_cell_size <= 20
        assert result.mean_cell_size == 20.0
        
        # Verify panel structure
        panel = result.panel_data
        assert len(panel) == 50
        assert '_n_obs' in panel.columns
        assert 'y_var' in panel.columns


# =============================================================================
# Vibe-Math MCP Numerical Verification Tests (Task 13)
# =============================================================================

class TestVibeMathVerification:
    """
    Numerical verification tests designed for vibe-math MCP validation.
    
    These tests provide explicit expected values that can be verified
    using the vibe-math MCP tools for formula correctness.
    """
    
    def test_weighted_mean_exact_values(self):
        """
        Test weighted mean with exact values for vibe-math verification.
        
        Formula: Y_bar = sum(w_i * Y_i) / sum(w_i)
        
        Values: [10, 20, 30, 40]
        Weights: [1, 2, 3, 4]
        
        Calculation:
        - sum(w_i) = 1 + 2 + 3 + 4 = 10
        - sum(w_i * Y_i) = 1*10 + 2*20 + 3*30 + 4*40 = 10 + 40 + 90 + 160 = 300
        - Y_bar = 300 / 10 = 30
        """
        data = pd.DataFrame({
            'unit': [1, 1, 1, 1],
            'period': [1, 1, 1, 1],
            'y': [10.0, 20.0, 30.0, 40.0],
            'weight': [1.0, 2.0, 3.0, 4.0],
        })
        
        result = aggregate_to_panel(data, 'unit', 'period', 'y', weight_var='weight')
        
        # Expected: 30.0
        assert result.panel_data['y'].iloc[0] == 30.0
    
    def test_ess_exact_values(self):
        """
        Test ESS with exact values for vibe-math verification.
        
        Formula: ESS = (sum(w_i))^2 / sum(w_i^2)
        
        Weights: [1, 2, 3, 4]
        
        Calculation:
        - sum(w_i) = 10
        - sum(w_i^2) = 1 + 4 + 9 + 16 = 30
        - ESS = 100 / 30 = 10/3 ≈ 3.333...
        """
        data = pd.DataFrame({
            'unit': [1, 1, 1, 1],
            'period': [1, 1, 1, 1],
            'y': [10.0, 20.0, 30.0, 40.0],
            'weight': [1.0, 2.0, 3.0, 4.0],
        })
        
        result = aggregate_to_panel(data, 'unit', 'period', 'y', weight_var='weight')
        
        # Expected: 10/3 = 3.333...
        expected_ess = 10.0 / 3.0
        assert np.isclose(result.panel_data['_ess'].iloc[0], expected_ess, rtol=1e-12)
    
    def test_weighted_variance_exact_values(self):
        """
        Test weighted variance with exact values for vibe-math verification.
        
        Formula: Var = sum(w_i * (Y_i - Y_bar)^2) where w_i are normalized
        
        Values: [0, 10, 20]
        Weights: [1, 1, 1] (equal)
        
        Calculation:
        - Y_bar = (0 + 10 + 20) / 3 = 10
        - Normalized weights = [1/3, 1/3, 1/3]
        - Var = (1/3)*(0-10)^2 + (1/3)*(10-10)^2 + (1/3)*(20-10)^2
        -     = (1/3)*100 + 0 + (1/3)*100
        -     = 200/3 ≈ 66.667
        """
        data = pd.DataFrame({
            'unit': [1, 1, 1],
            'period': [1, 1, 1],
            'y': [0.0, 10.0, 20.0],
            'weight': [1.0, 1.0, 1.0],
        })
        
        result = aggregate_to_panel(
            data, 'unit', 'period', 'y',
            weight_var='weight', compute_variance=True
        )
        
        # Expected: 200/3 ≈ 66.667
        expected_var = 200.0 / 3.0
        assert np.isclose(result.panel_data['y_var'].iloc[0], expected_var, rtol=1e-10)
    
    def test_multi_cell_aggregation_exact(self):
        """
        Test multi-cell aggregation with exact values.
        
        Cell (1, 1): values [10, 20], weights [1, 1]
            Y_bar = 15
        
        Cell (1, 2): values [30, 40], weights [1, 3]
            Y_bar = (1*30 + 3*40) / 4 = 150/4 = 37.5
        
        Cell (2, 1): values [50, 60, 70], weights [1, 2, 3]
            Y_bar = (1*50 + 2*60 + 3*70) / 6 = 380/6 ≈ 63.333
        """
        data = pd.DataFrame({
            'unit': [1, 1, 1, 1, 2, 2, 2],
            'period': [1, 1, 2, 2, 1, 1, 1],
            'y': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
            'weight': [1.0, 1.0, 1.0, 3.0, 1.0, 2.0, 3.0],
        })
        
        result = aggregate_to_panel(data, 'unit', 'period', 'y', weight_var='weight')
        panel = result.panel_data.sort_values(['unit', 'period'])
        
        # Cell (1, 1): expected 15.0
        assert np.isclose(
            panel[(panel['unit'] == 1) & (panel['period'] == 1)]['y'].iloc[0],
            15.0, rtol=1e-12
        )
        
        # Cell (1, 2): expected 37.5
        assert np.isclose(
            panel[(panel['unit'] == 1) & (panel['period'] == 2)]['y'].iloc[0],
            37.5, rtol=1e-12
        )
        
        # Cell (2, 1): expected 380/6 = 63.333...
        expected_21 = 380.0 / 6.0
        assert np.isclose(
            panel[(panel['unit'] == 2) & (panel['period'] == 1)]['y'].iloc[0],
            expected_21, rtol=1e-12
        )


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
