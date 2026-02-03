"""
Comprehensive tests for never-treated value validation.

This module tests the complete never-treated identification logic across
all modules in the lwdid package, ensuring consistency and correctness.

Based on: Lee & Wooldridge (2025) ssrn-4516518, Section 4
Spec: .kiro/specs/never-treated-validation/

Test Categories:
- Unit tests for is_never_treated()
- Cross-module consistency tests
- never_treated_values parameter tests
- Data type compatibility tests
- Error handling tests
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import sys
sys.path.insert(0, 'src')

from lwdid.validation import is_never_treated, validate_staggered_data
from lwdid.staggered.control_groups import (
    identify_never_treated_units,
    get_valid_control_units,
    has_never_treated_units,
    ControlGroupStrategy,
)
from lwdid.staggered.aggregation import _identify_nt_mask
from lwdid.exceptions import InvalidStaggeredDataError


# =============================================================================
# Phase 2: Unit Tests - is_never_treated()
# =============================================================================

class TestIsNeverTreatedBasic:
    """TEST-1: Basic functionality tests for is_never_treated()."""
    
    def test_zero_is_never_treated(self):
        """Test that zero values are recognized as never-treated."""
        assert is_never_treated(0) is True
        assert is_never_treated(0.0) is True
    
    def test_positive_infinity_is_never_treated(self):
        """Test that positive infinity is recognized as never-treated."""
        assert is_never_treated(np.inf) is True
        assert is_never_treated(float('inf')) is True
    
    def test_nan_is_never_treated(self):
        """Test that NaN values are recognized as never-treated."""
        assert is_never_treated(np.nan) is True
        assert is_never_treated(float('nan')) is True
    
    def test_positive_integers_are_treated(self):
        """Test that positive integers are recognized as treated cohorts."""
        assert is_never_treated(1) is False
        assert is_never_treated(2005) is False
        assert is_never_treated(2010) is False
        assert is_never_treated(9999) is False
    
    def test_positive_floats_are_treated(self):
        """Test that positive floats are recognized as treated cohorts."""
        assert is_never_treated(2005.0) is False
        assert is_never_treated(1.5) is False


class TestIsNeverTreatedEdgeCases:
    """TEST-2: Edge case tests for is_never_treated()."""
    
    def test_near_zero_values(self):
        """Test floating point tolerance for near-zero values."""
        # Very close to zero should be treated as zero
        assert is_never_treated(1e-12) is True
        assert is_never_treated(1e-15) is True
        assert is_never_treated(-1e-15) is True  # Negative near-zero
        
        # Not close enough to zero
        assert is_never_treated(0.001) is False
        assert is_never_treated(1e-6) is False
    
    def test_negative_infinity_raises_error(self):
        """Test that negative infinity raises InvalidStaggeredDataError."""
        with pytest.raises(InvalidStaggeredDataError) as exc_info:
            is_never_treated(-np.inf)
        
        error_msg = str(exc_info.value).lower()
        assert "negative infinity" in error_msg
        assert "not a valid gvar value" in error_msg
    
    def test_none_value(self):
        """Test that None is recognized as never-treated."""
        assert is_never_treated(None) is True
    
    def test_pandas_na(self):
        """Test that pandas NA is recognized as never-treated."""
        assert is_never_treated(pd.NA) is True
    
    def test_large_positive_integer(self):
        """Test that large positive integers are treated cohorts."""
        assert is_never_treated(99999) is False
        assert is_never_treated(1e10) is False


class TestCrossModuleConsistency:
    """TEST-3: Cross-module consistency tests."""
    
    @pytest.fixture
    def test_data(self):
        """Create test data with various gvar values."""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5] * 3,
            'year': [2000, 2001, 2002] * 5,
            'y': np.random.randn(15),
            'gvar': [0, np.inf, np.nan, 2001, 2002] * 3
        })
    
    def test_is_never_treated_vs_identify_never_treated_units(self, test_data):
        """Verify is_never_treated matches identify_never_treated_units."""
        # Using is_never_treated directly
        unit_gvar = test_data.groupby('id')['gvar'].first()
        expected = unit_gvar.apply(is_never_treated)
        
        # Using identify_never_treated_units with default values
        actual = identify_never_treated_units(test_data, 'gvar', 'id')
        
        # Compare results
        pd.testing.assert_series_equal(
            expected.sort_index(), 
            actual.sort_index(),
            check_names=False
        )
    
    def test_is_never_treated_vs_identify_nt_mask(self, test_data):
        """Verify is_never_treated matches _identify_nt_mask."""
        unit_gvar = test_data.groupby('id')['gvar'].first()
        
        # Using is_never_treated directly
        expected = unit_gvar.apply(is_never_treated)
        
        # Using _identify_nt_mask
        actual = _identify_nt_mask(unit_gvar)
        
        pd.testing.assert_series_equal(expected, actual)
    
    def test_all_modules_agree_on_mixed_data(self):
        """Verify all modules produce identical results on mixed data."""
        data = pd.DataFrame({
            'id': list(range(1, 11)) * 3,
            'year': [2000, 2001, 2002] * 10,
            'y': np.random.randn(30),
            'gvar': [0, np.inf, np.nan, 2001, 2002, 0, np.inf, np.nan, 2001, 2002] * 3
        })
        
        unit_gvar = data.groupby('id')['gvar'].first()
        
        # Method 1: is_never_treated
        result1 = unit_gvar.apply(is_never_treated)
        
        # Method 2: identify_never_treated_units (default)
        result2 = identify_never_treated_units(data, 'gvar', 'id')
        
        # Method 3: _identify_nt_mask
        result3 = _identify_nt_mask(unit_gvar)
        
        # All should agree
        assert result1.sum() == result2.sum() == result3.sum()
        pd.testing.assert_series_equal(result1.sort_index(), result2.sort_index(), check_names=False)
        pd.testing.assert_series_equal(result1, result3)


class TestNeverTreatedValuesParameter:
    """TEST-4: Tests for never_treated_values parameter."""
    
    def test_default_values(self):
        """Test default never_treated_values = [0, np.inf]."""
        data = pd.DataFrame({
            'id': [1, 2, 3, 4] * 3,
            'year': [2000, 2001, 2002] * 4,
            'gvar': [0, np.inf, 999, 2005] * 3
        })
        
        mask = identify_never_treated_units(data, 'gvar', 'id')
        
        assert mask.loc[1] == True   # 0 in default list
        assert mask.loc[2] == True   # inf in default list
        assert mask.loc[3] == False  # 999 not in default list
        assert mask.loc[4] == False  # 2005 is treated
    
    def test_custom_values(self):
        """Test custom never_treated_values."""
        data = pd.DataFrame({
            'id': [1, 2, 3] * 3,
            'year': [2000, 2001, 2002] * 3,
            'gvar': [0, 999, 2005] * 3
        })
        
        mask = identify_never_treated_units(
            data, 'gvar', 'id', never_treated_values=[999]
        )
        
        assert mask.loc[1] == False  # 0 not in custom list
        assert mask.loc[2] == True   # 999 in custom list
        assert mask.loc[3] == False  # 2005 is treated
    
    def test_empty_list_only_nan_recognized(self):
        """Test empty never_treated_values list - only NaN recognized."""
        data = pd.DataFrame({
            'id': [1, 2, 3] * 3,
            'year': [2000, 2001, 2002] * 3,
            'gvar': [0, np.nan, 2005] * 3
        })
        
        mask = identify_never_treated_units(
            data, 'gvar', 'id', never_treated_values=[]
        )
        
        assert mask.loc[1] == False  # 0 not recognized with empty list
        assert mask.loc[2] == True   # NaN always recognized
        assert mask.loc[3] == False  # 2005 is treated
    
    def test_multiple_custom_values(self):
        """Test multiple custom never_treated_values."""
        data = pd.DataFrame({
            'id': [1, 2, 3, 4] * 3,
            'year': [2000, 2001, 2002] * 4,
            'gvar': [999, -1, 0, 2005] * 3
        })
        
        mask = identify_never_treated_units(
            data, 'gvar', 'id', never_treated_values=[999, -1]
        )
        
        assert mask.loc[1] == True   # 999 in custom list
        assert mask.loc[2] == True   # -1 in custom list
        assert mask.loc[3] == False  # 0 not in custom list
        assert mask.loc[4] == False  # 2005 is treated


class TestErrorHandling:
    """TEST-5: Error handling tests."""
    
    def test_negative_infinity_error_message(self):
        """Test that negative infinity error message is helpful."""
        with pytest.raises(InvalidStaggeredDataError) as exc_info:
            is_never_treated(-np.inf)
        
        error_msg = str(exc_info.value)
        
        # Check error message contains helpful information
        assert "negative infinity" in error_msg.lower()
        assert "valid gvar values" in error_msg.lower()
        assert "how to fix" in error_msg.lower()
    
    def test_missing_gvar_column(self):
        """Test error when gvar column is missing."""
        data = pd.DataFrame({
            'id': [1, 2],
            'year': [2000, 2001]
        })
        
        with pytest.raises(KeyError) as exc_info:
            identify_never_treated_units(data, 'gvar', 'id')
        
        assert 'gvar' in str(exc_info.value)
    
    def test_missing_ivar_column(self):
        """Test error when ivar column is missing."""
        data = pd.DataFrame({
            'gvar': [0, 2005],
            'year': [2000, 2001]
        })
        
        with pytest.raises(KeyError) as exc_info:
            identify_never_treated_units(data, 'gvar', 'id')
        
        assert 'id' in str(exc_info.value)
    
    def test_empty_data(self):
        """Test error when data is empty."""
        data = pd.DataFrame(columns=['id', 'gvar', 'year'])
        
        with pytest.raises(ValueError) as exc_info:
            identify_never_treated_units(data, 'gvar', 'id')
        
        assert "empty" in str(exc_info.value).lower()


# =============================================================================
# Phase 3: Numerical Validation Tests
# =============================================================================

class TestDataTypeCompatibility:
    """TEST-4 (Numerical): Data type compatibility tests."""
    
    @pytest.mark.parametrize("dtype,value,expected", [
        (int, 0, True),
        (int, 2005, False),
        (float, 0.0, True),
        (float, 2005.0, False),
        (np.int64, 0, True),
        (np.int64, 2005, False),
        (np.int32, 0, True),
        (np.int32, 2005, False),
        (np.float64, 0.0, True),
        (np.float64, 2005.0, False),
        (np.float32, 0.0, True),
        (np.float32, 2005.0, False),
    ])
    def test_various_dtypes(self, dtype, value, expected):
        """Test is_never_treated with various data types."""
        result = is_never_treated(dtype(value))
        assert result is expected, f"Failed for dtype={dtype}, value={value}"
    
    def test_mixed_dtype_dataframe(self):
        """Test with DataFrame containing mixed dtypes."""
        data = pd.DataFrame({
            'id': [1, 2, 3],
            'gvar': pd.array([0, np.inf, 2005], dtype='float64')
        })
        # Expand to panel
        data = pd.concat([data.assign(year=y) for y in [2000, 2001, 2002]])
        
        mask = identify_never_treated_units(data, 'gvar', 'id')
        
        assert mask.loc[1] == True   # 0
        assert mask.loc[2] == True   # inf
        assert mask.loc[3] == False  # 2005


class TestControlGroupSelection:
    """TEST-5 (Numerical): Control group selection tests."""
    
    @pytest.fixture
    def panel_data(self):
        """Create panel data for control group tests."""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5] * 3,
            'year': [2000, 2001, 2002] * 5,
            'y': np.random.randn(15),
            'gvar': [0, np.inf, np.nan, 2001, 2002] * 3
        })
    
    def test_never_treated_strategy(self, panel_data):
        """Verify NEVER_TREATED strategy selects only NT units."""
        mask = get_valid_control_units(
            panel_data, 'gvar', 'id', cohort=2001, period=2001,
            strategy=ControlGroupStrategy.NEVER_TREATED
        )
        
        # Only id 1, 2, 3 should be in control group (NT units)
        assert mask.loc[1] == True   # gvar=0
        assert mask.loc[2] == True   # gvar=inf
        assert mask.loc[3] == True   # gvar=nan
        assert mask.loc[4] == False  # gvar=2001 (treated cohort)
        assert mask.loc[5] == False  # gvar=2002 (not NT)
    
    def test_not_yet_treated_strategy(self, panel_data):
        """Verify NOT_YET_TREATED strategy includes NT + NYT."""
        mask = get_valid_control_units(
            panel_data, 'gvar', 'id', cohort=2001, period=2001,
            strategy=ControlGroupStrategy.NOT_YET_TREATED
        )
        
        # id 1, 2, 3 (NT) + id 5 (gvar=2002 > 2001)
        assert mask.loc[1] == True   # NT
        assert mask.loc[2] == True   # NT
        assert mask.loc[3] == True   # NT
        assert mask.loc[4] == False  # treated cohort
        assert mask.loc[5] == True   # not yet treated (2002 > 2001)
    
    def test_control_group_formula_43(self):
        """Verify control group matches paper formula (4.3)."""
        # Paper formula (4.3): A_{r+1} = {i : g_i > r} ∪ {i : g_i = ∞}
        data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5] * 6,
            'year': list(range(1, 7)) * 5,
            'y': np.random.randn(30),
            'gvar': [3, 4, 5, 0, np.inf] * 6  # cohort 3, 4, 5, NT(0), NT(inf)
        })
        
        # For cohort=3, period=3:
        # A_4 = {g > 3} ∪ {g ∈ NT} = {4, 5} ∪ {0, inf}
        mask = get_valid_control_units(
            data, 'gvar', 'id', cohort=3, period=3,
            strategy=ControlGroupStrategy.NOT_YET_TREATED
        )
        
        # Verify: id=2 (gvar=4), id=3 (gvar=5), id=4 (gvar=0), id=5 (gvar=inf)
        assert mask.loc[1] == False  # gvar=3 (treated cohort)
        assert mask.loc[2] == True   # gvar=4 > 3
        assert mask.loc[3] == True   # gvar=5 > 3
        assert mask.loc[4] == True   # gvar=0 (never-treated)
        assert mask.loc[5] == True   # gvar=inf (never-treated)


class TestFloatingPointPrecision:
    """TEST-6 (Numerical): Floating point precision tests."""
    
    def test_near_zero_tolerance(self):
        """Test floating point tolerance for near-zero values."""
        # Very close to zero should be treated as zero
        assert is_never_treated(1e-15) is True
        assert is_never_treated(-1e-15) is True
        
        # Slightly larger values should not be zero
        assert is_never_treated(0.001) is False
    
    def test_infinity_comparison(self):
        """Test infinity comparison edge cases."""
        assert is_never_treated(float('inf')) is True
        assert is_never_treated(np.float64('inf')) is True
        
        # Very large numbers are not infinity
        assert is_never_treated(1e308) is False
        assert is_never_treated(1e100) is False


# =============================================================================
# Phase 5: Paper End-to-End Tests
# =============================================================================

class TestPaperSection4:
    """TEST-7: Paper Section 4 example verification."""
    
    def test_paper_setup_t6_cohorts_456_inf(self):
        """Verify setup matches paper Section 4: T=6, cohorts={4,5,6,∞}."""
        np.random.seed(42)
        n_per_cohort = 50
        
        # Create paper Section 4 data
        data = self._create_paper_section4_data(n_per_cohort)
        
        # Verify cohort structure
        unit_gvar = data.groupby('id')['gvar'].first()
        treated_cohorts = sorted(unit_gvar[unit_gvar > 0].unique())
        assert treated_cohorts == [4, 5, 6]
        
        # Verify never-treated count
        n_nt = unit_gvar.apply(is_never_treated).sum()
        assert n_nt == n_per_cohort  # 50 never-treated units
    
    def test_control_group_for_cohort4_period4(self):
        """Verify control group for cohort=4, period=4."""
        np.random.seed(42)
        data = self._create_paper_section4_data(n_per_cohort=50)
        
        # For cohort=4, period=4, control group should include:
        # - Never-treated units
        # - Cohorts 5 and 6 (not yet treated at period 4)
        mask = get_valid_control_units(
            data, 'gvar', 'id', cohort=4, period=4,
            strategy=ControlGroupStrategy.NOT_YET_TREATED
        )
        
        unit_gvar = data.groupby('id')['gvar'].first()
        
        for uid, gvar in unit_gvar.items():
            if is_never_treated(gvar) or gvar > 4:
                assert mask.loc[uid] == True, f"Unit {uid} with gvar={gvar} should be in control"
            else:
                assert mask.loc[uid] == False, f"Unit {uid} with gvar={gvar} should not be in control"
    
    def _create_paper_section4_data(self, n_per_cohort: int = 50) -> pd.DataFrame:
        """Create data matching paper Section 4 setup."""
        data_rows = []
        uid = 0
        
        # Cohorts 4, 5, 6
        for cohort in [4, 5, 6]:
            for _ in range(n_per_cohort):
                for t in range(1, 7):
                    treated = 1 if t >= cohort else 0
                    y = 10 + 0.5*t + 5*treated + np.random.randn()
                    data_rows.append({
                        'id': uid, 'year': t, 'y': y, 'gvar': cohort
                    })
                uid += 1
        
        # Never-treated (gvar = 0)
        for _ in range(n_per_cohort):
            for t in range(1, 7):
                y = 10 + 0.5*t + np.random.randn()
                data_rows.append({
                    'id': uid, 'year': t, 'y': y, 'gvar': 0
                })
            uid += 1
        
        return pd.DataFrame(data_rows)


# =============================================================================
# Phase 7: Monte Carlo Tests
# =============================================================================

class TestMonteCarloNeverTreatedControl:
    """TEST-10: Monte Carlo validation of never-treated control group."""
    
    @pytest.mark.slow
    def test_coverage_rate_with_never_treated(self):
        """Monte Carlo: verify CI coverage rate with NT control."""
        n_simulations = 100  # Reduced for faster testing
        true_att = 5.0
        coverage_count = 0
        att_estimates = []
        
        for sim in range(n_simulations):
            np.random.seed(sim)
            
            # Generate DGP
            data = self._generate_dgp_with_never_treated(
                n_units=100, n_periods=6, true_att=true_att, nt_ratio=0.3
            )
            
            # Verify never-treated identification
            unit_gvar = data.groupby('id')['gvar'].first()
            n_nt = unit_gvar.apply(is_never_treated).sum()
            
            # Should have approximately 30% never-treated (allow wider range)
            assert 15 <= n_nt <= 50, f"Unexpected NT count: {n_nt}"
            
            # Simple ATT estimation (difference in means of transformed outcomes)
            att_est, se_est = self._simple_att_estimate(data, true_att)
            att_estimates.append(att_est)
            
            # Check coverage (using normal approximation for simplicity)
            ci_lower = att_est - 1.96 * se_est
            ci_upper = att_est + 1.96 * se_est
            if ci_lower <= true_att <= ci_upper:
                coverage_count += 1
        
        # Verify bias
        mean_att = np.mean(att_estimates)
        bias = mean_att - true_att
        assert abs(bias) < 1.0, f"Bias too large: {bias}"
        
        # Verify coverage rate (allow wider range due to small n_simulations)
        coverage_rate = coverage_count / n_simulations
        assert 0.80 <= coverage_rate <= 0.99, f"Coverage rate: {coverage_rate}"
    
    def _generate_dgp_with_never_treated(
        self, n_units: int, n_periods: int, true_att: float, nt_ratio: float
    ) -> pd.DataFrame:
        """Generate DGP with never-treated units."""
        data_rows = []
        
        for i in range(n_units):
            # Assign treatment status
            if np.random.rand() < nt_ratio:
                gvar = 0  # never-treated
            else:
                gvar = 4  # treated at period 4
            
            # Unit fixed effect
            alpha_i = np.random.randn()
            
            for t in range(1, n_periods + 1):
                # Time trend
                delta_t = 0.5 * t
                
                # Treatment effect
                treated = 1 if (gvar > 0 and t >= gvar) else 0
                
                # Error
                epsilon = np.random.randn()
                
                # Outcome
                y = alpha_i + delta_t + true_att * treated + epsilon
                
                data_rows.append({
                    'id': i, 'year': t, 'y': y, 'gvar': gvar
                })
        
        return pd.DataFrame(data_rows)
    
    def _simple_att_estimate(self, data: pd.DataFrame, true_att: float) -> tuple:
        """Simple ATT estimation using demeaning."""
        # Demean outcomes
        pre_means = data[data['year'] < 4].groupby('id')['y'].mean()
        post_means = data[data['year'] >= 4].groupby('id')['y'].mean()
        
        unit_gvar = data.groupby('id')['gvar'].first()
        
        # Transformed outcome
        delta_y = post_means - pre_means
        
        # Separate treated and control
        treated_mask = unit_gvar == 4
        control_mask = unit_gvar.apply(is_never_treated)
        
        treated_delta = delta_y[treated_mask]
        control_delta = delta_y[control_mask]
        
        # ATT estimate
        att = treated_delta.mean() - control_delta.mean()
        
        # SE estimate
        n_t = len(treated_delta)
        n_c = len(control_delta)
        se = np.sqrt(treated_delta.var()/n_t + control_delta.var()/n_c)
        
        return att, se


# =============================================================================
# Phase 8: Simulated Data Tests
# =============================================================================

class TestSimulatedDataVariousEncodings:
    """TEST-11: Various never-treated encoding tests."""
    
    @pytest.mark.parametrize("nt_encoding,nt_value", [
        ("zero", 0),
        ("infinity", np.inf),
        ("nan", np.nan),
    ])
    def test_encoding_recognized_correctly(self, nt_encoding, nt_value):
        """Test that different NT encodings are recognized."""
        np.random.seed(123)
        
        # Create data with specific encoding
        data = self._create_simulated_data(nt_value=nt_value)
        
        # Verify NT units are recognized
        unit_gvar = data.groupby('id')['gvar'].first()
        n_nt = unit_gvar.apply(is_never_treated).sum()
        
        # Should have 20 NT units
        assert n_nt == 20, f"Expected 20 NT units with {nt_encoding}, got {n_nt}"
    
    def test_mixed_encodings_in_same_dataset(self):
        """Test dataset with mixed NT encodings (0, inf, nan)."""
        np.random.seed(456)
        
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
    
    def _create_simulated_data(self, nt_value, n_nt: int = 20, n_treated: int = 40) -> pd.DataFrame:
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


# =============================================================================
# Additional Integration Tests
# =============================================================================

class TestHasNeverTreatedUnits:
    """Test has_never_treated_units helper function."""
    
    def test_returns_true_when_nt_exists(self):
        """Test returns True when never-treated units exist."""
        data = pd.DataFrame({
            'id': [1, 2, 3] * 3,
            'year': [2000, 2001, 2002] * 3,
            'gvar': [0, 2001, 2002] * 3
        })
        
        assert has_never_treated_units(data, 'gvar', 'id') == True
    
    def test_returns_false_when_no_nt(self):
        """Test returns False when no never-treated units."""
        data = pd.DataFrame({
            'id': [1, 2, 3] * 3,
            'year': [2000, 2001, 2002] * 3,
            'gvar': [2001, 2002, 2003] * 3
        })
        
        assert has_never_treated_units(data, 'gvar', 'id') == False
    
    def test_recognizes_all_nt_encodings(self):
        """Test recognizes all NT encodings."""
        # Test with 0
        data1 = pd.DataFrame({
            'id': [1, 2] * 3, 'year': [2000, 2001, 2002] * 2,
            'gvar': [0, 2001] * 3
        })
        assert has_never_treated_units(data1, 'gvar', 'id') == True
        
        # Test with inf
        data2 = pd.DataFrame({
            'id': [1, 2] * 3, 'year': [2000, 2001, 2002] * 2,
            'gvar': [np.inf, 2001] * 3
        })
        assert has_never_treated_units(data2, 'gvar', 'id') == True
        
        # Test with nan
        data3 = pd.DataFrame({
            'id': [1, 2] * 3, 'year': [2000, 2001, 2002] * 2,
            'gvar': [np.nan, 2001] * 3
        })
        assert has_never_treated_units(data3, 'gvar', 'id') == True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
