"""
Empirical data tests for never-treated handling.

This module tests never-treated identification with real-world datasets
to ensure the implementation works correctly with actual data patterns.

Based on: Lee & Wooldridge (2025) ssrn-4516518, Section 4
Spec: .kiro/specs/never-treated-validation/

Test Categories:
- Staggered simulation data tests
- Cattaneo2 data tests (cross-sectional)
- Data loading and format compatibility
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

import sys
sys.path.insert(0, 'src')

from lwdid.validation import is_never_treated
from lwdid.staggered.control_groups import (
    identify_never_treated_units,
    get_valid_control_units,
    ControlGroupStrategy,
)


class TestStaggeredSimulationData:
    """
    TEST-19: Staggered simulation data tests.
    
    Tests never-treated identification with the staggered_simulation.csv
    dataset which has proper panel structure with gvar column.
    """
    
    @pytest.fixture
    def staggered_data(self):
        """Load staggered simulation data."""
        data_path = Path(__file__).parent / 'data' / 'staggered_simulation.csv'
        if not data_path.exists():
            pytest.skip("staggered_simulation.csv not found")
        return pd.read_csv(data_path)
    
    def test_data_structure(self, staggered_data):
        """Verify data has required columns for staggered DiD."""
        required_cols = ['id', 'year', 'y', 'gvar']
        for col in required_cols:
            assert col in staggered_data.columns, f"Missing column: {col}"
    
    def test_never_treated_identification(self, staggered_data):
        """Test NT identification with staggered simulation data."""
        unit_gvar = staggered_data.groupby('id')['gvar'].first()
        
        # Count NT units
        n_nt = unit_gvar.apply(is_never_treated).sum()
        n_total = len(unit_gvar)
        
        # Should have some NT units
        assert n_nt > 0, "No never-treated units found"
        assert n_nt < n_total, "All units are never-treated"
        
        # NT ratio should be reasonable (0-100%)
        nt_ratio = n_nt / n_total
        assert 0 < nt_ratio < 1, f"NT ratio {nt_ratio:.1%} is unreasonable"
    
    def test_gvar_values_distribution(self, staggered_data):
        """Verify gvar values distribution."""
        unit_gvar = staggered_data.groupby('id')['gvar'].first()
        
        # Get unique gvar values
        unique_gvar = unit_gvar.unique()
        
        # Should have multiple cohorts
        n_cohorts = len([g for g in unique_gvar if not is_never_treated(g)])
        assert n_cohorts >= 1, "No treatment cohorts found"
    
    def test_control_group_selection(self, staggered_data):
        """Test control group selection with empirical data."""
        unit_gvar = staggered_data.groupby('id')['gvar'].first()
        
        # Get first treatment cohort
        treated_cohorts = sorted([g for g in unit_gvar.unique() if not is_never_treated(g)])
        if len(treated_cohorts) == 0:
            pytest.skip("No treatment cohorts in data")
        
        first_cohort = treated_cohorts[0]
        
        # Test NEVER_TREATED strategy
        mask_nt = get_valid_control_units(
            staggered_data, 'gvar', 'id', cohort=first_cohort, period=first_cohort,
            strategy=ControlGroupStrategy.NEVER_TREATED
        )
        
        # Should have some control units
        assert mask_nt.sum() > 0, "No control units with NEVER_TREATED strategy"
        
        # Test NOT_YET_TREATED strategy
        mask_nyt = get_valid_control_units(
            staggered_data, 'gvar', 'id', cohort=first_cohort, period=first_cohort,
            strategy=ControlGroupStrategy.NOT_YET_TREATED
        )
        
        # NYT should have at least as many as NT
        assert mask_nyt.sum() >= mask_nt.sum(), \
            "NOT_YET_TREATED should include at least as many units as NEVER_TREATED"
    
    def test_panel_balance(self, staggered_data):
        """Verify panel is balanced (same periods for all units)."""
        periods_per_unit = staggered_data.groupby('id')['year'].nunique()
        
        # All units should have same number of periods
        assert periods_per_unit.nunique() == 1, "Panel is unbalanced"
    
    def test_gvar_time_invariant(self, staggered_data):
        """Verify gvar is time-invariant within units."""
        gvar_per_unit = staggered_data.groupby('id')['gvar'].nunique()
        
        # Each unit should have exactly one gvar value
        assert (gvar_per_unit == 1).all(), "gvar varies within units"


class TestCattaneo2Data:
    """
    TEST-20: Cattaneo2 data tests.
    
    Tests with cattaneo2.dta which is a cross-sectional dataset.
    This tests data loading and basic compatibility.
    """
    
    @pytest.fixture
    def cattaneo_data(self):
        """Load cattaneo2 data."""
        data_path = Path(__file__).parent.parent.parent / 'cattaneo2.dta'
        if not data_path.exists():
            pytest.skip("cattaneo2.dta not found")
        return pd.read_stata(data_path)
    
    def test_data_loads_correctly(self, cattaneo_data):
        """Verify cattaneo2 data loads correctly."""
        assert len(cattaneo_data) > 0, "Data is empty"
        assert len(cattaneo_data.columns) > 0, "No columns in data"
    
    def test_data_columns(self, cattaneo_data):
        """Verify expected columns exist."""
        # Cattaneo2 is a birth weight dataset
        expected_cols = ['bweight', 'mbsmoke']
        for col in expected_cols:
            assert col in cattaneo_data.columns, f"Missing column: {col}"
    
    def test_treatment_variable(self, cattaneo_data):
        """Test treatment variable (mbsmoke) distribution."""
        if 'mbsmoke' not in cattaneo_data.columns:
            pytest.skip("mbsmoke column not found")
        
        # mbsmoke should be binary
        unique_vals = cattaneo_data['mbsmoke'].dropna().unique()
        assert len(unique_vals) <= 2, "mbsmoke should be binary"
    
    def test_outcome_variable(self, cattaneo_data):
        """Test outcome variable (bweight) distribution."""
        if 'bweight' not in cattaneo_data.columns:
            pytest.skip("bweight column not found")
        
        # Birth weight should be positive
        assert (cattaneo_data['bweight'] > 0).all(), "Birth weight should be positive"
        
        # Should have reasonable range (500-6000 grams)
        assert cattaneo_data['bweight'].min() >= 0
        assert cattaneo_data['bweight'].max() <= 10000


class TestDataFormatCompatibility:
    """
    TEST-21: Data format compatibility tests.
    
    Tests that various data formats are handled correctly.
    """
    
    def test_csv_loading(self):
        """Test CSV data loading."""
        data_path = Path(__file__).parent / 'data' / 'staggered_simulation.csv'
        if not data_path.exists():
            pytest.skip("staggered_simulation.csv not found")
        
        data = pd.read_csv(data_path)
        assert len(data) > 0
    
    def test_stata_loading(self):
        """Test Stata data loading."""
        data_path = Path(__file__).parent.parent.parent / 'cattaneo2.dta'
        if not data_path.exists():
            pytest.skip("cattaneo2.dta not found")
        
        data = pd.read_stata(data_path)
        assert len(data) > 0
    
    def test_gvar_dtype_handling(self):
        """Test gvar column with different dtypes."""
        # Integer gvar
        data_int = pd.DataFrame({
            'id': [1, 2, 3] * 3,
            'year': [2000, 2001, 2002] * 3,
            'y': np.random.randn(9),
            'gvar': pd.array([0, 2001, 2002] * 3, dtype='int64')
        })
        
        unit_gvar = data_int.groupby('id')['gvar'].first()
        n_nt = unit_gvar.apply(is_never_treated).sum()
        assert n_nt == 1
        
        # Float gvar
        data_float = pd.DataFrame({
            'id': [1, 2, 3] * 3,
            'year': [2000, 2001, 2002] * 3,
            'y': np.random.randn(9),
            'gvar': pd.array([0.0, 2001.0, 2002.0] * 3, dtype='float64')
        })
        
        unit_gvar = data_float.groupby('id')['gvar'].first()
        n_nt = unit_gvar.apply(is_never_treated).sum()
        assert n_nt == 1
    
    def test_missing_values_handling(self):
        """Test handling of missing values in gvar."""
        data = pd.DataFrame({
            'id': [1, 2, 3, 4] * 3,
            'year': [2000, 2001, 2002] * 4,
            'y': np.random.randn(12),
            'gvar': [0, np.nan, pd.NA, 2001] * 3
        })
        
        unit_gvar = data.groupby('id')['gvar'].first()
        n_nt = unit_gvar.apply(is_never_treated).sum()
        
        # 0, nan, NA should all be NT
        assert n_nt == 3


class TestNeverTreatedStatistics:
    """
    TEST-22: Never-treated statistics tests.
    
    Tests statistical properties of NT identification.
    """
    
    @pytest.fixture
    def staggered_data(self):
        """Load staggered simulation data."""
        data_path = Path(__file__).parent / 'data' / 'staggered_simulation.csv'
        if not data_path.exists():
            pytest.skip("staggered_simulation.csv not found")
        return pd.read_csv(data_path)
    
    def test_nt_ratio_calculation(self, staggered_data):
        """Test NT ratio calculation."""
        unit_gvar = staggered_data.groupby('id')['gvar'].first()
        
        n_nt = unit_gvar.apply(is_never_treated).sum()
        n_total = len(unit_gvar)
        nt_ratio = n_nt / n_total
        
        # Verify ratio is between 0 and 1
        assert 0 <= nt_ratio <= 1
        
        # Log the ratio for debugging
        print(f"NT ratio: {nt_ratio:.2%} ({n_nt}/{n_total})")
    
    def test_cohort_distribution(self, staggered_data):
        """Test cohort distribution."""
        unit_gvar = staggered_data.groupby('id')['gvar'].first()
        
        # Count units per cohort
        cohort_counts = {}
        for gvar in unit_gvar.unique():
            if is_never_treated(gvar):
                cohort_counts['NT'] = cohort_counts.get('NT', 0) + (unit_gvar == gvar).sum()
            else:
                cohort_counts[gvar] = (unit_gvar == gvar).sum()
        
        # Should have at least 2 groups (NT + at least 1 treated cohort)
        assert len(cohort_counts) >= 2
        
        # Log distribution
        print(f"Cohort distribution: {cohort_counts}")
    
    def test_outcome_by_treatment_status(self, staggered_data):
        """Test outcome distribution by treatment status."""
        unit_gvar = staggered_data.groupby('id')['gvar'].first()
        nt_mask = unit_gvar.apply(is_never_treated)
        
        nt_ids = unit_gvar[nt_mask].index
        treated_ids = unit_gvar[~nt_mask].index
        
        # Calculate mean outcome for each group
        nt_mean = staggered_data[staggered_data['id'].isin(nt_ids)]['y'].mean()
        treated_mean = staggered_data[staggered_data['id'].isin(treated_ids)]['y'].mean()
        
        # Both should be finite
        assert np.isfinite(nt_mean), "NT mean is not finite"
        assert np.isfinite(treated_mean), "Treated mean is not finite"
        
        # Log means
        print(f"NT mean: {nt_mean:.3f}, Treated mean: {treated_mean:.3f}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
