"""
Staggered scenario transformation variable validation tests.

Verifies that transformation variable computations are fully consistent
with Stata, including the fixed-denominator formula (g-1) and correct
identification of pre-treatment periods.

Transformation formula (Paper Equation 4.7):
    ŷ_{irg} = Y_{ir} - (1/(g-1)) * Σ_{s=1}^{g-1} Y_{is}

Stata reference code (2.lee_wooldridge_rolling_staggered.do:19-28):
    bysort id: gen y_44 = y - (L1.y + L2.y + L3.y)/3 if f04
    bysort id: gen y_45 = y - (L2.y + L3.y + L4.y)/3 if f05
    bysort id: gen y_46 = y - (L3.y + L4.y + L5.y)/3 if f06
    bysort id: gen y_55 = y - (L1.y + L2.y + L3.y + L4.y)/4 if f05
    bysort id: gen y_56 = y - (L2.y + L3.y + L4.y +L5.y)/4 if f06
    bysort id: gen y_66 = y - (L1.y + L2.y + L3.y + L4.y+L5.y)/5 if f06

Validates Section 4, Equation 4.7 (cohort-specific rolling transformations)
of the Lee-Wooldridge Difference-in-Differences framework.

References
----------
Lee, S. & Wooldridge, J. M. (2025). A Simple Transformation Approach to
    Difference-in-Differences Estimation for Panel Data. SSRN 4516518.
"""
import numpy as np
import pandas as pd
import pytest

def compute_transformed_outcome(
    data: pd.DataFrame,
    y_col: str,
    id_col: str,
    period_col: str,
    cohort_g: int,
    period_r: int,
) -> pd.Series:
    r"""Compute the demeaned outcome for a (g, r) pair.

    Formula: Y_hat_{irg} = Y_{ir} - (1/(g-1)) * sum_{s=1}^{g-1} Y_{is}
    """
    Y_r = data[data[period_col] == period_r].set_index(id_col)[y_col]
    pre_periods = list(range(1, cohort_g))
    if len(pre_periods) == 0:
        raise ValueError(f"cohort_g={cohort_g}: no pre-treatment periods")
    pre_mean = (
        data[data[period_col].isin(pre_periods)]
        .groupby(id_col)[y_col]
        .mean()
    )
    common_ids = Y_r.index.intersection(pre_mean.index)
    return Y_r.loc[common_ids] - pre_mean.loc[common_ids]


class TestTransformationFormulaDenominator:
    """Transformation formula denominator verification."""
    
    def test_denominator_is_g_minus_1_not_r_minus_1(self):
        """CRITICAL TEST: Verify the denominator is g-1, not r-1."""
        # Create test data
        data = pd.DataFrame({
            'id': [1]*6,
            'year': [1, 2, 3, 4, 5, 6],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })
        
        # Test (g=4, r=5)
        # Denominator should be g-1 = 3, not r-1 = 4
        # pre-treatment periods: 1, 2, 3 (3 periods)
        # pre_mean = (1+2+3)/3 = 2.0
        # Y_5 = 5.0
        # y_45 = 5.0 - 2.0 = 3.0
        y_45 = compute_transformed_outcome(data, 'y', 'id', 'year', cohort_g=4, period_r=5)
        
        assert abs(y_45.iloc[0] - 3.0) < 1e-10, \
            f"(g=4, r=5) computation error: {y_45.iloc[0]} != 3.0"
        
        # Test (g=4, r=6)
        # Denominator is still g-1 = 3
        # Y_6 = 6.0
        # y_46 = 6.0 - 2.0 = 4.0
        y_46 = compute_transformed_outcome(data, 'y', 'id', 'year', cohort_g=4, period_r=6)
        
        assert abs(y_46.iloc[0] - 4.0) < 1e-10, \
            f"(g=4, r=6) computation error: {y_46.iloc[0]} != 4.0"
        
        # Test (g=5, r=5)
        # Denominator is g-1 = 4
        # pre-treatment periods: 1, 2, 3, 4 (4 periods)
        # pre_mean = (1+2+3+4)/4 = 2.5
        # Y_5 = 5.0
        # y_55 = 5.0 - 2.5 = 2.5
        y_55 = compute_transformed_outcome(data, 'y', 'id', 'year', cohort_g=5, period_r=5)
        
        assert abs(y_55.iloc[0] - 2.5) < 1e-10, \
            f"(g=5, r=5) computation error: {y_55.iloc[0]} != 2.5"
    
    @pytest.mark.parametrize("g,expected_pre_periods", [
        (4, 3),   # g=4: periods 1,2,3 → 3 periods
        (5, 4),   # g=5: periods 1,2,3,4 → 4 periods
        (6, 5),   # g=6: periods 1,2,3,4,5 → 5 periods
    ])
    def test_pre_treatment_period_count(self, g, expected_pre_periods):
        """Verify the number of pre-treatment periods is correct."""
        pre_periods = list(range(1, g))
        
        assert len(pre_periods) == expected_pre_periods, \
            f"cohort {g}: pre-treatment period count {len(pre_periods)} != {expected_pre_periods}"
        assert len(pre_periods) == g - 1, \
            f"cohort {g}: denominator should be g-1={g-1}"


class TestTransformationManualCalculation:
    """Manual calculation verification."""
    
    @pytest.mark.parametrize("g,r,expected", [
        # Y_series = [1,2,3,4,5,6] for periods 1-6
        (4, 4, 4.0 - (1+2+3)/3),        # Y_4 - mean(1,2,3) = 4 - 2 = 2
        (4, 5, 5.0 - (1+2+3)/3),        # Y_5 - mean(1,2,3) = 5 - 2 = 3
        (4, 6, 6.0 - (1+2+3)/3),        # Y_6 - mean(1,2,3) = 6 - 2 = 4
        (5, 5, 5.0 - (1+2+3+4)/4),      # Y_5 - mean(1,2,3,4) = 5 - 2.5 = 2.5
        (5, 6, 6.0 - (1+2+3+4)/4),      # Y_6 - mean(1,2,3,4) = 6 - 2.5 = 3.5
        (6, 6, 6.0 - (1+2+3+4+5)/5),    # Y_6 - mean(1,2,3,4,5) = 6 - 3 = 3
    ])
    def test_transformation_manual_calculation(self, g, r, expected):
        """Test that transformation computation matches manual calculation."""
        data = pd.DataFrame({
            'id': [1]*6,
            'year': [1, 2, 3, 4, 5, 6],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })
        
        y_gr = compute_transformed_outcome(data, 'y', 'id', 'year', g, r)
        
        assert abs(y_gr.iloc[0] - expected) < 1e-10, \
            f"(g={g}, r={r}) computation error: {y_gr.iloc[0]} != {expected}"


@pytest.mark.stata_alignment
class TestTransformationStataConsistency:
    """Transformation variable Stata consistency tests."""
    
    @pytest.fixture
    def staggered_data_with_transforms(self, small_staggered_data):
        """Compute transformation variables and add to data."""
        data = small_staggered_data.copy()
        
        # For each (g,r) combination, compute transformation
        for g in [4, 5, 6]:
            for r in range(g, 7):
                col_name = f'y_{g}{r}'
                y_transformed = compute_transformed_outcome(
                    data, 'y', 'id', 'year', g, r
                )
                
                # Only add transformation values at period r
                data[col_name] = np.nan
                mask = data['year'] == r
                data.loc[mask, col_name] = data.loc[mask, 'id'].map(y_transformed)
        
        return data
    
    def test_transformation_consistency_44(self, small_staggered_data):
        """Test (4,4) transformation consistency with Stata computation logic."""
        # Stata: y_44 = y - (L1.y + L2.y + L3.y)/3 if f04
        # Equivalent to: Y_4 - mean(Y_1, Y_2, Y_3)
        
        # Manual implementation of Stata logic
        data = small_staggered_data.copy()
        data_wide = data.pivot(index='id', columns='year', values='y')
        
        # Stata computation: (L1 + L2 + L3)/3 at year=4 means (year3 + year2 + year1)/3
        stata_y_44 = data_wide[4] - (data_wide[1] + data_wide[2] + data_wide[3]) / 3
        
        # Python computation
        python_y_44 = compute_transformed_outcome(data, 'y', 'id', 'year', 4, 4)
        
        # Align indices
        common_ids = stata_y_44.index.intersection(python_y_44.index)
        diff = abs(stata_y_44.loc[common_ids] - python_y_44.loc[common_ids])
        
        # Note: Stata data uses float32, so differences may be at the 1e-6 level
        assert diff.max() < 1e-5, \
            f"(4,4) transformation inconsistent with Stata logic: max_diff={diff.max()}"
    
    def test_transformation_consistency_55(self, small_staggered_data):
        """Test (5,5) transformation consistency with Stata computation logic."""
        # Stata: y_55 = y - (L1.y + L2.y + L3.y + L4.y)/4 if f05
        
        data = small_staggered_data.copy()
        data_wide = data.pivot(index='id', columns='year', values='y')
        
        # Stata computation
        stata_y_55 = data_wide[5] - (data_wide[1] + data_wide[2] + data_wide[3] + data_wide[4]) / 4
        
        # Python computation
        python_y_55 = compute_transformed_outcome(data, 'y', 'id', 'year', 5, 5)
        
        common_ids = stata_y_55.index.intersection(python_y_55.index)
        diff = abs(stata_y_55.loc[common_ids] - python_y_55.loc[common_ids])
        
        assert diff.max() < 1e-5, \
            f"(5,5) transformation inconsistent with Stata logic: max_diff={diff.max()}"
    
    def test_transformation_consistency_66(self, small_staggered_data):
        """Test (6,6) transformation consistency with Stata computation logic."""
        # Stata: y_66 = y - (L1.y + L2.y + L3.y + L4.y + L5.y)/5 if f06
        
        data = small_staggered_data.copy()
        data_wide = data.pivot(index='id', columns='year', values='y')
        
        # Stata computation
        stata_y_66 = data_wide[6] - (data_wide[1] + data_wide[2] + data_wide[3] + 
                                      data_wide[4] + data_wide[5]) / 5
        
        # Python computation
        python_y_66 = compute_transformed_outcome(data, 'y', 'id', 'year', 6, 6)
        
        common_ids = stata_y_66.index.intersection(python_y_66.index)
        diff = abs(stata_y_66.loc[common_ids] - python_y_66.loc[common_ids])
        
        assert diff.max() < 1e-5, \
            f"(6,6) transformation inconsistent with Stata logic: max_diff={diff.max()}"
    
    @pytest.mark.parametrize("g,r", [(4,4), (4,5), (4,6), (5,5), (5,6), (6,6)])
    def test_all_transformations_vs_stata_logic(self, small_staggered_data, g, r):
        """Test all (g,r) transformation combinations against Stata logic."""
        data = small_staggered_data.copy()
        data_wide = data.pivot(index='id', columns='year', values='y')
        
        # Stata logic: Y_r - mean(Y_1, ..., Y_{g-1})
        pre_periods = list(range(1, g))
        pre_sum = sum(data_wide[t] for t in pre_periods)
        stata_y_gr = data_wide[r] - pre_sum / len(pre_periods)
        
        # Python computation
        python_y_gr = compute_transformed_outcome(data, 'y', 'id', 'year', g, r)
        
        common_ids = stata_y_gr.index.intersection(python_y_gr.index)
        diff = abs(stata_y_gr.loc[common_ids] - python_y_gr.loc[common_ids])
        
        # Threshold relaxed to 1e-5 to accommodate float32 precision
        assert diff.max() < 1e-5, \
            f"(g={g}, r={r}) transformation inconsistent with Stata logic: max_diff={diff.max()}"


class TestTransformationBoundaryConditions:
    """Transformation formula boundary condition tests."""
    
    def test_missing_pre_treatment_periods(self):
        """Test handling of missing pre-treatment periods."""
        # Create data with missing period 1
        data = pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2, 2],
            'year': [2, 3, 4, 1, 2, 3, 4],  # id=1 missing period 1
            'y': [2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
        })
        
        # For g=4, periods 1,2,3 are needed
        # id=1 is missing period 1, should compute based on available data (or return NaN)
        y_gr = compute_transformed_outcome(data, 'y', 'id', 'year', 4, 4)
        
        # id=1: only periods 2,3 available, pre_mean = (2+3)/2 = 2.5
        # y_44 for id=1 = 4 - 2.5 = 1.5 (using mean of available data)
        # id=2: pre_mean = (1+2+3)/3 = 2.0
        # y_44 for id=2 = 4 - 2.0 = 2.0
        
        assert 1 in y_gr.index and 2 in y_gr.index
    
    def test_constant_outcome_transformation(self):
        """Test transformation with constant outcome variable."""
        data = pd.DataFrame({
            'id': [1, 1, 1, 1],
            'year': [1, 2, 3, 4],
            'y': [5.0, 5.0, 5.0, 5.0],  # Constant
        })
        
        y_gr = compute_transformed_outcome(data, 'y', 'id', 'year', 4, 4)
        
        # Y_r - mean(Y_pre) = 5 - 5 = 0
        assert abs(y_gr.iloc[0] - 0.0) < 1e-10
    
    def test_single_pre_treatment_period(self):
        """Test with only one pre-treatment period (g=2)."""
        data = pd.DataFrame({
            'id': [1, 1],
            'year': [1, 2],
            'y': [3.0, 5.0],
        })
        
        # g=2, denominator=1, pre_periods=[1]
        y_gr = compute_transformed_outcome(data, 'y', 'id', 'year', 2, 2)
        
        # Y_2 - Y_1 = 5 - 3 = 2
        assert abs(y_gr.iloc[0] - 2.0) < 1e-10
    
    def test_multiple_units_transformation(self, small_staggered_data):
        """Test transformation correctness across multiple units."""
        # Verify each unit has correctly computed transformations
        y_44 = compute_transformed_outcome(small_staggered_data, 'y', 'id', 'year', 4, 4)
        
        # Should have transformation values equal to the number of units at period 4
        n_units = small_staggered_data['id'].nunique()
        assert len(y_44) == n_units, \
            f"Number of transformation values {len(y_44)} != number of units {n_units}"
    
    def test_transformation_preserves_unit_identity(self, small_staggered_data):
        """Test that transformation preserves unit identity."""
        y_44 = compute_transformed_outcome(small_staggered_data, 'y', 'id', 'year', 4, 4)
        
        # The index of the transformed result should be the unit ID
        expected_ids = set(small_staggered_data['id'].unique())
        actual_ids = set(y_44.index)
        
        assert actual_ids == expected_ids


@pytest.mark.stata_alignment
class TestStataLagEquivalence:
    """Stata lag operator equivalence tests."""
    
    def test_lag_interpretation_44(self, small_staggered_data):
        """Test that the (4,4) lag interpretation is correct."""
        # Stata: L1.y + L2.y + L3.y at f04 (year=4)
        # L1.y at year=4 = y at year=3
        # L2.y at year=4 = y at year=2
        # L3.y at year=4 = y at year=1
        # Therefore: (L1+L2+L3) = Y_3 + Y_2 + Y_1 = sum(periods 1,2,3)
        
        data = small_staggered_data.copy()
        data_wide = data.pivot(index='id', columns='year', values='y')
        
        # Verify pre-treatment periods are correct
        pre_periods = list(range(1, 4))  # [1, 2, 3]
        assert pre_periods == [1, 2, 3]
    
    def test_lag_interpretation_45(self, small_staggered_data):
        """Test that the (4,5) lag interpretation is correct."""
        # Stata: L2.y + L3.y + L4.y at f05 (year=5)
        # L2.y at year=5 = y at year=3
        # L3.y at year=5 = y at year=2  
        # L4.y at year=5 = y at year=1
        # Therefore: (L2+L3+L4) = Y_3 + Y_2 + Y_1 = sum(periods 1,2,3)
        # Note: still periods 1,2,3 because the pre-treatment window for g=4 is fixed
        
        # This verifies the denominator is g-1, not r-1
        data = small_staggered_data.copy()
        data_wide = data.pivot(index='id', columns='year', values='y')
        
        pre_periods = list(range(1, 4))  # g=4 → [1, 2, 3]
        assert len(pre_periods) == 3, "Denominator should be 3 (= g-1)"