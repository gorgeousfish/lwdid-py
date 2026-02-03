"""
Simulated data tests for never-treated handling.

This module tests never-treated identification with various encoding
schemes and data configurations.

Based on: Lee & Wooldridge (2025) ssrn-4516518, Section 4
Spec: .kiro/specs/never-treated-validation/

Test Categories:
- Various NT encoding tests (0, inf, nan)
- Mixed encoding tests
- Edge case configurations
"""

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, 'src')

from lwdid.validation import is_never_treated
from lwdid.staggered.control_groups import (
    identify_never_treated_units,
    get_valid_control_units,
    ControlGroupStrategy,
)


class TestVariousNeverTreatedEncodings:
    """
    TEST-16: Various never-treated encoding tests.
    
    Tests that all common NT encodings (0, inf, nan) are correctly
    recognized and produce consistent results.
    """
    
    @pytest.mark.parametrize("nt_encoding,nt_value", [
        ("zero", 0),
        ("infinity", np.inf),
        ("nan", np.nan),
    ])
    def test_encoding_recognized_correctly(self, nt_encoding, nt_value):
        """Test that different NT encodings are recognized."""
        np.random.seed(123)
        
        # Create data with specific encoding
        data = self._create_simulated_data(nt_value=nt_value, n_nt=20, n_treated=40)
        
        # Verify NT units are recognized
        unit_gvar = data.groupby('id')['gvar'].first()
        n_nt = unit_gvar.apply(is_never_treated).sum()
        
        assert n_nt == 20, f"Expected 20 NT units with {nt_encoding}, got {n_nt}"
    
    @pytest.mark.parametrize("nt_encoding,nt_value", [
        ("zero", 0),
        ("infinity", np.inf),
        ("nan", np.nan),
    ])
    def test_encoding_produces_similar_estimates(self, nt_encoding, nt_value):
        """Test that different NT encodings produce similar ATT estimates."""
        np.random.seed(456)
        true_att = 5.0
        
        # Create data with treatment effect
        data = self._create_simulated_data_with_effect(
            nt_value=nt_value, n_nt=30, n_treated=60, true_att=true_att
        )
        
        # Simple ATT estimation
        att_est = self._simple_att_estimate(data)
        
        # Should be close to true ATT (within 2.0)
        assert abs(att_est - true_att) < 2.0, \
            f"ATT estimate {att_est:.2f} too far from {true_att} with {nt_encoding}"
    
    def test_mixed_encodings_in_same_dataset(self):
        """Test dataset with mixed NT encodings (0, inf, nan)."""
        np.random.seed(789)
        
        data_rows = []
        uid = 0
        
        # 10 units with gvar=0
        for _ in range(10):
            for t in range(1, 6):
                data_rows.append({
                    'id': uid, 'year': t, 'y': np.random.randn(), 'gvar': 0
                })
            uid += 1
        
        # 10 units with gvar=inf
        for _ in range(10):
            for t in range(1, 6):
                data_rows.append({
                    'id': uid, 'year': t, 'y': np.random.randn(), 'gvar': np.inf
                })
            uid += 1
        
        # 10 units with gvar=nan
        for _ in range(10):
            for t in range(1, 6):
                data_rows.append({
                    'id': uid, 'year': t, 'y': np.random.randn(), 'gvar': np.nan
                })
            uid += 1
        
        # 30 treated units
        for _ in range(30):
            gvar = np.random.choice([3, 4])
            for t in range(1, 6):
                data_rows.append({
                    'id': uid, 'year': t, 'y': np.random.randn(), 'gvar': gvar
                })
            uid += 1
        
        data = pd.DataFrame(data_rows)
        
        # Verify all encodings are recognized
        unit_gvar = data.groupby('id')['gvar'].first()
        n_nt = unit_gvar.apply(is_never_treated).sum()
        
        assert n_nt == 30, f"Expected 30 NT units (10+10+10), got {n_nt}"
    
    def test_all_encodings_produce_same_control_group(self):
        """Test that all NT encodings produce same control group selection."""
        np.random.seed(101)
        
        # Create three datasets with different NT encodings
        datasets = {}
        for nt_value, name in [(0, 'zero'), (np.inf, 'inf'), (np.nan, 'nan')]:
            data = self._create_simulated_data(nt_value=nt_value, n_nt=20, n_treated=40)
            datasets[name] = data
        
        # Get control group sizes for each
        control_sizes = {}
        for name, data in datasets.items():
            mask = get_valid_control_units(
                data, 'gvar', 'id', cohort=3, period=3,
                strategy=ControlGroupStrategy.NEVER_TREATED
            )
            control_sizes[name] = mask.sum()
        
        # All should have same control group size (20 NT units)
        assert control_sizes['zero'] == control_sizes['inf'] == control_sizes['nan'] == 20
    
    def _create_simulated_data(
        self, nt_value, n_nt: int = 20, n_treated: int = 40
    ) -> pd.DataFrame:
        """Create simulated data with specified NT encoding."""
        data_rows = []
        uid = 0
        
        # Never-treated units
        for _ in range(n_nt):
            for t in range(1, 6):
                data_rows.append({
                    'id': uid, 'year': t, 'y': np.random.randn(), 'gvar': nt_value
                })
            uid += 1
        
        # Treated units
        for _ in range(n_treated):
            gvar = np.random.choice([3, 4])
            for t in range(1, 6):
                data_rows.append({
                    'id': uid, 'year': t, 'y': np.random.randn(), 'gvar': gvar
                })
            uid += 1
        
        return pd.DataFrame(data_rows)
    
    def _create_simulated_data_with_effect(
        self, nt_value, n_nt: int, n_treated: int, true_att: float
    ) -> pd.DataFrame:
        """Create simulated data with treatment effect."""
        data_rows = []
        uid = 0
        
        # Never-treated units
        for _ in range(n_nt):
            alpha_i = np.random.randn() * 2
            for t in range(1, 6):
                y = alpha_i + 0.5*t + np.random.randn()
                data_rows.append({
                    'id': uid, 'year': t, 'y': y, 'gvar': nt_value
                })
            uid += 1
        
        # Treated units (cohort 3)
        for _ in range(n_treated):
            alpha_i = np.random.randn() * 2
            for t in range(1, 6):
                treated = 1 if t >= 3 else 0
                y = alpha_i + 0.5*t + true_att*treated + np.random.randn()
                data_rows.append({
                    'id': uid, 'year': t, 'y': y, 'gvar': 3
                })
            uid += 1
        
        return pd.DataFrame(data_rows)
    
    def _simple_att_estimate(self, data: pd.DataFrame) -> float:
        """Simple ATT estimation."""
        pre_means = data[data['year'] < 3].groupby('id')['y'].mean()
        post_means = data[data['year'] >= 3].groupby('id')['y'].mean()
        delta_y = post_means - pre_means
        
        unit_gvar = data.groupby('id')['gvar'].first()
        treated_mask = unit_gvar == 3
        control_mask = unit_gvar.apply(is_never_treated)
        
        att = delta_y[treated_mask].mean() - delta_y[control_mask].mean()
        return att


class TestEdgeCaseConfigurations:
    """
    TEST-17: Edge case configuration tests.
    """
    
    def test_single_nt_unit(self):
        """Test with single never-treated unit."""
        data = pd.DataFrame({
            'id': [1, 2, 3] * 3,
            'year': [2000, 2001, 2002] * 3,
            'y': np.random.randn(9),
            'gvar': [0, 2001, 2002] * 3  # Only 1 NT unit
        })
        
        unit_gvar = data.groupby('id')['gvar'].first()
        n_nt = unit_gvar.apply(is_never_treated).sum()
        
        assert n_nt == 1
    
    def test_all_nt_units(self):
        """Test with all never-treated units."""
        data = pd.DataFrame({
            'id': [1, 2, 3] * 3,
            'year': [2000, 2001, 2002] * 3,
            'y': np.random.randn(9),
            'gvar': [0, np.inf, np.nan] * 3  # All NT
        })
        
        unit_gvar = data.groupby('id')['gvar'].first()
        n_nt = unit_gvar.apply(is_never_treated).sum()
        
        assert n_nt == 3
    
    def test_no_nt_units(self):
        """Test with no never-treated units."""
        data = pd.DataFrame({
            'id': [1, 2, 3] * 3,
            'year': [2000, 2001, 2002] * 3,
            'y': np.random.randn(9),
            'gvar': [2001, 2002, 2003] * 3  # No NT
        })
        
        unit_gvar = data.groupby('id')['gvar'].first()
        n_nt = unit_gvar.apply(is_never_treated).sum()
        
        assert n_nt == 0
    
    def test_high_nt_ratio(self):
        """Test with high NT ratio (90%)."""
        np.random.seed(42)
        
        data_rows = []
        uid = 0
        
        # 90 NT units
        for _ in range(90):
            for t in range(1, 6):
                data_rows.append({
                    'id': uid, 'year': t, 'y': np.random.randn(), 'gvar': 0
                })
            uid += 1
        
        # 10 treated units
        for _ in range(10):
            for t in range(1, 6):
                data_rows.append({
                    'id': uid, 'year': t, 'y': np.random.randn(), 'gvar': 3
                })
            uid += 1
        
        data = pd.DataFrame(data_rows)
        
        unit_gvar = data.groupby('id')['gvar'].first()
        nt_ratio = unit_gvar.apply(is_never_treated).mean()
        
        assert np.isclose(nt_ratio, 0.9, rtol=0.01)
    
    def test_low_nt_ratio(self):
        """Test with low NT ratio (10%)."""
        np.random.seed(42)
        
        data_rows = []
        uid = 0
        
        # 10 NT units
        for _ in range(10):
            for t in range(1, 6):
                data_rows.append({
                    'id': uid, 'year': t, 'y': np.random.randn(), 'gvar': 0
                })
            uid += 1
        
        # 90 treated units
        for _ in range(90):
            for t in range(1, 6):
                data_rows.append({
                    'id': uid, 'year': t, 'y': np.random.randn(), 'gvar': 3
                })
            uid += 1
        
        data = pd.DataFrame(data_rows)
        
        unit_gvar = data.groupby('id')['gvar'].first()
        nt_ratio = unit_gvar.apply(is_never_treated).mean()
        
        assert np.isclose(nt_ratio, 0.1, rtol=0.01)
    
    def test_large_cohort_values(self):
        """Test with large cohort values (e.g., year 2050)."""
        data = pd.DataFrame({
            'id': [1, 2, 3] * 3,
            'year': [2040, 2045, 2050] * 3,
            'y': np.random.randn(9),
            'gvar': [0, 2045, 2050] * 3
        })
        
        unit_gvar = data.groupby('id')['gvar'].first()
        n_nt = unit_gvar.apply(is_never_treated).sum()
        
        assert n_nt == 1  # Only gvar=0 is NT
    
    def test_many_cohorts(self):
        """Test with many treatment cohorts."""
        np.random.seed(42)
        
        data_rows = []
        uid = 0
        
        # 10 cohorts (years 2001-2010)
        for cohort in range(2001, 2011):
            for _ in range(10):
                for t in range(2000, 2015):
                    data_rows.append({
                        'id': uid, 'year': t, 'y': np.random.randn(), 'gvar': cohort
                    })
                uid += 1
        
        # 20 NT units
        for _ in range(20):
            for t in range(2000, 2015):
                data_rows.append({
                    'id': uid, 'year': t, 'y': np.random.randn(), 'gvar': 0
                })
            uid += 1
        
        data = pd.DataFrame(data_rows)
        
        unit_gvar = data.groupby('id')['gvar'].first()
        n_nt = unit_gvar.apply(is_never_treated).sum()
        n_cohorts = unit_gvar[~unit_gvar.apply(is_never_treated)].nunique()
        
        assert n_nt == 20
        assert n_cohorts == 10


class TestDataTypeVariations:
    """
    TEST-18: Data type variation tests.
    """
    
    def test_integer_gvar(self):
        """Test with integer gvar column."""
        data = pd.DataFrame({
            'id': [1, 2, 3] * 3,
            'year': [2000, 2001, 2002] * 3,
            'y': np.random.randn(9),
            'gvar': pd.array([0, 2001, 2002] * 3, dtype='int64')
        })
        
        unit_gvar = data.groupby('id')['gvar'].first()
        n_nt = unit_gvar.apply(is_never_treated).sum()
        
        assert n_nt == 1  # Only gvar=0
    
    def test_float_gvar(self):
        """Test with float gvar column."""
        data = pd.DataFrame({
            'id': [1, 2, 3] * 3,
            'year': [2000, 2001, 2002] * 3,
            'y': np.random.randn(9),
            'gvar': pd.array([0.0, 2001.0, 2002.0] * 3, dtype='float64')
        })
        
        unit_gvar = data.groupby('id')['gvar'].first()
        n_nt = unit_gvar.apply(is_never_treated).sum()
        
        assert n_nt == 1  # Only gvar=0.0
    
    def test_nullable_integer_gvar(self):
        """Test with nullable integer gvar column (Int64)."""
        data = pd.DataFrame({
            'id': [1, 2, 3] * 3,
            'year': [2000, 2001, 2002] * 3,
            'y': np.random.randn(9),
            'gvar': pd.array([0, pd.NA, 2002] * 3, dtype='Int64')
        })
        
        unit_gvar = data.groupby('id')['gvar'].first()
        n_nt = unit_gvar.apply(is_never_treated).sum()
        
        assert n_nt == 2  # gvar=0 and gvar=NA


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
