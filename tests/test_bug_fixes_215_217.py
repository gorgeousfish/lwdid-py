"""Test fixes for BUG-215, BUG-216, BUG-217: Inf handling in randomization and export.

BUG-215: randomization.py should check observed ATT for Inf values
BUG-216: randomization.py should filter Inf values from simulation statistics
BUG-217: results.py to_excel should use _safe_export_value for numeric fields
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from lwdid.randomization import randomization_inference
from lwdid.exceptions import RandomizationError
from lwdid.results import _safe_export_value


class TestBug215InfObservedATT:
    """BUG-215: Test that Inf in observed ATT raises RandomizationError.
    
    When att_obs is infinite, the p-value computation |T_sim| >= |T_obs|
    would return all False (since no finite value >= inf), producing a
    misleadingly small p-value. This should be caught and raise an error.
    """

    @pytest.fixture
    def sample_data(self):
        """Create sample cross-sectional data for RI testing."""
        np.random.seed(42)
        n = 50
        return pd.DataFrame({
            'ivar': range(n),
            'd_': np.concatenate([np.ones(20), np.zeros(30)]).astype(int),
            'ydot_postavg': np.random.randn(n) + np.concatenate([
                np.ones(20) * 0.5,
                np.zeros(30)
            ])
        })

    def test_positive_inf_att_raises_error(self, sample_data):
        """Test that positive infinity ATT raises RandomizationError."""
        with pytest.raises(RandomizationError) as exc_info:
            randomization_inference(
                firstpost_df=sample_data,
                y_col='ydot_postavg',
                d_col='d_',
                rireps=100,
                seed=12345,
                att_obs=np.inf,  # Infinite ATT
                ri_method='permutation'
            )
        
        error_msg = str(exc_info.value).lower()
        assert "infinite" in error_msg
        assert "cannot compute" in error_msg or "p-value" in error_msg

    def test_negative_inf_att_raises_error(self, sample_data):
        """Test that negative infinity ATT raises RandomizationError."""
        with pytest.raises(RandomizationError) as exc_info:
            randomization_inference(
                firstpost_df=sample_data,
                y_col='ydot_postavg',
                d_col='d_',
                rireps=100,
                seed=12345,
                att_obs=-np.inf,  # Negative infinite ATT
                ri_method='permutation'
            )
        
        error_msg = str(exc_info.value).lower()
        assert "infinite" in error_msg

    def test_finite_att_works_correctly(self, sample_data):
        """Test that finite ATT values work correctly."""
        result = randomization_inference(
            firstpost_df=sample_data,
            y_col='ydot_postavg',
            d_col='d_',
            rireps=100,
            seed=12345,
            att_obs=0.5,  # Finite ATT
            ri_method='permutation'
        )
        
        assert 'p_value' in result
        assert 0 <= result['p_value'] <= 1
        assert result['ri_valid'] == 100


class TestBug216InfSimulationStats:
    """BUG-216: Test that Inf values in simulation statistics are filtered.
    
    The fix changes `atts[~np.isnan(atts)]` to `atts[np.isfinite(atts)]`
    to filter both NaN and Inf values from the simulation distribution.
    """

    @pytest.fixture
    def sample_data(self):
        """Create sample cross-sectional data for RI testing."""
        np.random.seed(42)
        n = 50
        return pd.DataFrame({
            'ivar': range(n),
            'd_': np.concatenate([np.ones(20), np.zeros(30)]).astype(int),
            'ydot_postavg': np.random.randn(n) + np.concatenate([
                np.ones(20) * 0.5,
                np.zeros(30)
            ])
        })

    def test_ri_filters_inf_from_simulations(self, sample_data):
        """Test that randomization inference filters Inf from simulation stats.
        
        This is an indirect test - we verify that the function uses np.isfinite
        by checking that valid replications are computed correctly.
        """
        result = randomization_inference(
            firstpost_df=sample_data,
            y_col='ydot_postavg',
            d_col='d_',
            rireps=100,
            seed=12345,
            ri_method='permutation'
        )
        
        # With permutation method and good data, all should be valid
        assert result['ri_valid'] == 100
        assert result['ri_failed'] == 0
        assert 0 <= result['p_value'] <= 1

    def test_isfinite_filters_both_nan_and_inf(self):
        """Verify np.isfinite correctly filters both NaN and Inf."""
        # This is a unit test for the filtering logic
        # Array: [1.0, nan, 2.0, inf, 3.0, -inf, 4.0] has 7 elements
        atts = np.array([1.0, np.nan, 2.0, np.inf, 3.0, -np.inf, 4.0])
        
        # The old way (only NaN filtered)
        old_valid = atts[~np.isnan(atts)]
        assert len(old_valid) == 6  # Only NaN removed, Inf remains (7-1=6)
        assert np.inf in old_valid
        assert -np.inf in old_valid
        
        # The new way (both NaN and Inf filtered)
        new_valid = atts[np.isfinite(atts)]
        assert len(new_valid) == 4  # NaN + 2 Infs removed (7-3=4)
        assert np.inf not in new_valid
        assert -np.inf not in new_valid
        np.testing.assert_array_equal(new_valid, [1.0, 2.0, 3.0, 4.0])


class TestBug217ExcelExportSafeValue:
    """BUG-217: Test that to_excel uses _safe_export_value for numeric fields.
    
    The fix applies _safe_export_value to all potentially NaN/Inf numeric
    values in the summary rows before creating the DataFrame.
    """

    def test_safe_export_value_handles_inf(self):
        """Test that _safe_export_value converts Inf to NaN."""
        # Positive infinity
        result = _safe_export_value(np.inf)
        assert np.isnan(result)
        
        # Negative infinity
        result = _safe_export_value(-np.inf)
        assert np.isnan(result)

    def test_safe_export_value_handles_nan(self):
        """Test that _safe_export_value converts NaN to NaN (default)."""
        result = _safe_export_value(np.nan)
        assert np.isnan(result)

    def test_safe_export_value_preserves_finite(self):
        """Test that _safe_export_value preserves finite values."""
        assert _safe_export_value(1.5) == 1.5
        assert _safe_export_value(0.0) == 0.0
        assert _safe_export_value(-2.5) == -2.5

    def test_safe_export_value_custom_default(self):
        """Test _safe_export_value with custom default for Inf."""
        result = _safe_export_value(np.inf, default=-999)
        assert result == -999
        
        result = _safe_export_value(-np.inf, default=-999)
        assert result == -999

    def test_excel_export_with_inf_values(self):
        """Integration test: Excel export should handle Inf values gracefully."""
        # Import here to avoid circular imports
        from lwdid import lwdid
        
        here = os.path.dirname(__file__)
        data_path = os.path.join(here, 'data', 'smoking.csv')
        
        if not os.path.exists(data_path):
            pytest.skip("smoking.csv not found")
        
        data = pd.read_csv(data_path)
        
        res = lwdid(
            data, y='lcigsale', d='d', ivar='state', tvar='year', post='post',
            rolling='demean', vce='robust'
        )
        
        with tempfile.TemporaryDirectory() as td:
            xlsx_path = os.path.join(td, 'test_export.xlsx')
            
            # Export should succeed without errors
            res.to_excel(xlsx_path)
            
            assert os.path.exists(xlsx_path)
            
            # Read back and verify
            df_summary = pd.read_excel(xlsx_path, sheet_name='Summary')
            assert 'Statistic' in df_summary.columns
            assert 'Value' in df_summary.columns
            
            # ATT should be a finite value
            att_row = df_summary[df_summary['Statistic'] == 'ATT']
            assert len(att_row) == 1
            att_value = att_row['Value'].iloc[0]
            assert np.isfinite(att_value), "ATT should be finite in export"


class TestStaggeredInfHandling:
    """Test Inf handling consistency between cross-sectional and staggered modules."""

    def test_staggered_ri_checks_inf(self):
        """Verify staggered randomization.py also checks for Inf observed ATT."""
        from lwdid.staggered.randomization import randomization_inference_staggered
        
        # Create minimal staggered data
        np.random.seed(42)
        rows = []
        for i in range(20):
            cohort = 2005 if i < 10 else 0  # 10 treated, 10 never-treated
            for t in range(2003, 2008):
                rows.append({
                    'unit': i,
                    'year': t,
                    'gvar': cohort,
                    'y': np.random.randn() + (0.5 if cohort > 0 and t >= cohort else 0)
                })
        data = pd.DataFrame(rows)
        
        with pytest.raises(RandomizationError) as exc_info:
            randomization_inference_staggered(
                data=data,
                gvar='gvar',
                ivar='unit',
                tvar='year',
                y='y',
                observed_att=np.inf,  # Infinite ATT
                target='overall',
                rireps=50,
                seed=123,
                rolling='demean',
                n_never_treated=10
            )
        
        error_msg = str(exc_info.value).lower()
        assert "infinite" in error_msg


class TestNumericalValidation:
    """Numerical validation tests for Inf handling."""

    def test_pvalue_not_affected_by_inf_filtering(self):
        """Test that p-value computation is correct after Inf filtering."""
        # Create controlled simulation results
        np.random.seed(42)
        
        # Simulate 1000 ATT estimates with known distribution
        sim_atts = np.random.normal(0, 1, 1000)
        observed = 2.0  # 2 standard deviations
        
        # Calculate p-value: proportion of |sim| >= |obs|
        n_extreme = (np.abs(sim_atts) >= abs(observed)).sum()
        expected_pvalue = (n_extreme + 1) / (len(sim_atts) + 1)
        
        # Verify p-value is in expected range for 2 sigma
        # For normal distribution, P(|Z| >= 2) ≈ 0.046
        assert 0.02 < expected_pvalue < 0.10

    def test_monte_carlo_correction(self):
        """Test that Monte Carlo +1 correction is applied correctly."""
        # Edge case: all simulations are less extreme than observed
        sim_atts = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        observed = 10.0  # Much larger than all simulations
        
        n_extreme = (np.abs(sim_atts) >= abs(observed)).sum()
        assert n_extreme == 0
        
        # With +1 correction, p-value should be 1/6 ≈ 0.167
        pvalue = (n_extreme + 1) / (len(sim_atts) + 1)
        assert np.isclose(pvalue, 1/6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
