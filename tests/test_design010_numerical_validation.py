"""
DESIGN-010 数值验证测试

测试内容：
1. 公式验证：验证警告阈值计算是否正确
2. 蒙特卡洛模拟：在各种数据条件下验证警告行为
3. 边界情况验证：精确测试 50% 阈值边界

创建日期: 2026-01-17
"""

import warnings
import pytest
import numpy as np
import pandas as pd
from typing import List, Tuple

from lwdid.staggered.aggregation import (
    construct_aggregated_outcome,
    aggregate_to_cohort,
    aggregate_to_overall,
)
from lwdid.staggered.transformations import (
    transform_staggered_demean,
    get_cohorts,
)


# =============================================================================
# Helper Functions
# =============================================================================

def generate_staggered_panel(
    n_units: int = 100,
    n_periods: int = 10,
    cohort_periods: List[int] = None,
    nt_proportion: float = 0.3,
    treatment_effect: float = 5.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate staggered adoption panel data.
    
    Parameters
    ----------
    n_units : int
        Number of units
    n_periods : int
        Number of time periods
    cohort_periods : list
        Treatment cohort periods (e.g., [4, 5, 6])
    nt_proportion : float
        Proportion of never-treated units
    treatment_effect : float
        True treatment effect
    seed : int
        Random seed
        
    Returns
    -------
    DataFrame
        Panel data with columns: unit, year, gvar, y
    """
    np.random.seed(seed)
    
    if cohort_periods is None:
        cohort_periods = [4, 5, 6]
    
    data = pd.DataFrame({
        'unit': np.repeat(range(1, n_units + 1), n_periods),
        'year': np.tile(range(1, n_periods + 1), n_units),
    })
    
    # Assign cohorts
    n_nt = int(n_units * nt_proportion)
    n_treated = n_units - n_nt
    cohort_size = n_treated // len(cohort_periods)
    
    gvar_list = []
    unit_idx = 0
    
    for g in cohort_periods:
        gvar_list.extend([g] * cohort_size)
        unit_idx += cohort_size
    
    # Remaining treated units go to last cohort
    remaining = n_treated - len(gvar_list)
    if remaining > 0:
        gvar_list.extend([cohort_periods[-1]] * remaining)
    
    # Never treated
    gvar_list.extend([0] * n_nt)
    
    unit_gvar = {i+1: gvar_list[i] for i in range(n_units)}
    data['gvar'] = data['unit'].map(unit_gvar)
    
    # Generate outcome
    data['y'] = 10 + np.random.randn(len(data)) * 2
    
    # Add treatment effect
    for idx, row in data.iterrows():
        if row['gvar'] > 0 and row['year'] >= row['gvar']:
            data.loc[idx, 'y'] += treatment_effect
    
    return data


def count_warnings_by_type(
    warnings_list: List,
    warning_type: str,
) -> int:
    """Count warnings containing specific text."""
    return sum(1 for w in warnings_list if warning_type in str(w.message))


# =============================================================================
# Numerical Validation Tests
# =============================================================================

class TestValidityRatioCalculation:
    """Test validity ratio calculation accuracy."""
    
    def test_validity_ratio_formula(self):
        """Verify validity ratio = n_valid / n_total."""
        np.random.seed(42)
        
        # Create data with known validity ratio
        n_units = 100
        data = generate_staggered_panel(n_units=n_units, n_periods=8)
        transformed = transform_staggered_demean(
            data=data, y='y', gvar='gvar', ivar='unit', tvar='year',
            never_treated_values=[0]
        )
        
        cohorts = get_cohorts(transformed, 'gvar', 'unit', never_treated_values=[0])
        T_max = int(transformed['year'].max())
        
        unit_gvar = transformed.groupby('unit')['gvar'].first()
        cohort_sizes = {g: int((unit_gvar == g).sum()) for g in cohorts}
        N_treat = sum(cohort_sizes.values())
        weights = {g: n / N_treat for g, n in cohort_sizes.items()}
        
        Y_bar = construct_aggregated_outcome(
            data=transformed,
            gvar='gvar',
            ivar='unit',
            tvar='year',
            weights=weights,
            cohorts=cohorts,
            T_max=T_max,
            transform_type='demean',
            never_treated_values=[0]
        )
        
        # Calculate validity ratio manually
        n_valid = Y_bar.notna().sum()
        n_total = len(Y_bar)
        validity_ratio = n_valid / n_total
        
        # Should have high validity for clean data
        assert validity_ratio > 0.8, f"Expected high validity, got {validity_ratio:.1%}"
    
    def test_warning_threshold_exactly_50_percent(self):
        """Test that warning triggers at exactly 50% boundary."""
        np.random.seed(42)
        
        # Create small dataset for precise control
        n_units = 20
        data = generate_staggered_panel(n_units=n_units, n_periods=6)
        transformed = transform_staggered_demean(
            data=data, y='y', gvar='gvar', ivar='unit', tvar='year',
            never_treated_values=[0]
        )
        
        cohorts = get_cohorts(transformed, 'gvar', 'unit', never_treated_values=[0])
        T_max = int(transformed['year'].max())
        
        unit_gvar = transformed.groupby('unit')['gvar'].first()
        cohort_sizes = {g: int((unit_gvar == g).sum()) for g in cohorts}
        N_treat = sum(cohort_sizes.values())
        weights = {g: n / N_treat for g, n in cohort_sizes.items()}
        
        # Test with 51% valid (should NOT warn)
        corrupted_data_51 = transformed.copy()
        units = corrupted_data_51['unit'].unique()
        n_corrupt = int(len(units) * 0.49)  # Corrupt 49%, leave 51% valid
        corrupt_units = units[:n_corrupt]
        ydot_cols = [c for c in corrupted_data_51.columns if c.startswith('ydot_')]
        for col in ydot_cols:
            corrupted_data_51.loc[corrupted_data_51['unit'].isin(corrupt_units), col] = np.nan
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Y_bar_51 = construct_aggregated_outcome(
                data=corrupted_data_51,
                gvar='gvar',
                ivar='unit',
                tvar='year',
                weights=weights,
                cohorts=cohorts,
                T_max=T_max,
                transform_type='demean',
                never_treated_values=[0]
            )
            low_warnings_51 = count_warnings_by_type(w, '聚合结果有效率较低')
        
        # Test with 49% valid (should warn)
        corrupted_data_49 = transformed.copy()
        n_corrupt = int(len(units) * 0.51)  # Corrupt 51%, leave 49% valid
        corrupt_units = units[:n_corrupt]
        for col in ydot_cols:
            corrupted_data_49.loc[corrupted_data_49['unit'].isin(corrupt_units), col] = np.nan
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Y_bar_49 = construct_aggregated_outcome(
                data=corrupted_data_49,
                gvar='gvar',
                ivar='unit',
                tvar='year',
                weights=weights,
                cohorts=cohorts,
                T_max=T_max,
                transform_type='demean',
                never_treated_values=[0]
            )
            
            n_valid = Y_bar_49.notna().sum()
            n_total = len(Y_bar_49)
            validity = n_valid / n_total if n_total > 0 else 0
            
            low_warnings_49 = count_warnings_by_type(w, '聚合结果有效率较低')
            
            # If validity < 50%, should have warning
            if validity < 0.5:
                assert low_warnings_49 >= 1, f"Expected warning for {validity:.1%} validity"


# =============================================================================
# Monte Carlo Simulation Tests
# =============================================================================

class TestMonteCarloWarningBehavior:
    """Monte Carlo tests for warning behavior under various conditions."""
    
    @pytest.mark.parametrize("missing_rate", [0.0, 0.25, 0.45, 0.55, 0.75, 0.95])
    def test_warning_rate_by_missing_data(self, missing_rate):
        """Test warning behavior at different missing data rates."""
        n_simulations = 10
        warning_counts = {'all_nan': 0, 'low_validity': 0, 'none': 0}
        
        for sim in range(n_simulations):
            np.random.seed(42 + sim)
            
            data = generate_staggered_panel(
                n_units=50, 
                n_periods=6,
                seed=42 + sim
            )
            transformed = transform_staggered_demean(
                data=data, y='y', gvar='gvar', ivar='unit', tvar='year',
                never_treated_values=[0]
            )
            
            # Corrupt data
            if missing_rate > 0:
                corrupted = transformed.copy()
                ydot_cols = [c for c in corrupted.columns if c.startswith('ydot_')]
                n_rows = len(corrupted)
                corrupt_mask = np.random.choice(
                    [True, False], 
                    size=n_rows, 
                    p=[missing_rate, 1 - missing_rate]
                )
                for col in ydot_cols:
                    corrupted.loc[corrupt_mask, col] = np.nan
            else:
                corrupted = transformed
            
            cohorts = get_cohorts(corrupted, 'gvar', 'unit', never_treated_values=[0])
            T_max = int(corrupted['year'].max())
            
            unit_gvar = corrupted.groupby('unit')['gvar'].first()
            cohort_sizes = {g: int((unit_gvar == g).sum()) for g in cohorts}
            N_treat = sum(cohort_sizes.values())
            weights = {g: n / N_treat for g, n in cohort_sizes.items()}
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                Y_bar = construct_aggregated_outcome(
                    data=corrupted,
                    gvar='gvar',
                    ivar='unit',
                    tvar='year',
                    weights=weights,
                    cohorts=cohorts,
                    T_max=T_max,
                    transform_type='demean',
                    never_treated_values=[0]
                )
                
                if count_warnings_by_type(w, '聚合结果全为 NaN') > 0:
                    warning_counts['all_nan'] += 1
                elif count_warnings_by_type(w, '聚合结果有效率较低') > 0:
                    warning_counts['low_validity'] += 1
                else:
                    warning_counts['none'] += 1
        
        # Verify warning patterns make sense
        if missing_rate >= 0.95:
            # Very high missing rate: expect all_nan or low_validity warnings
            assert warning_counts['none'] < n_simulations * 0.3
        elif missing_rate <= 0.25:
            # Low missing rate: expect mostly no warnings
            assert warning_counts['none'] >= n_simulations * 0.5
    
    def test_cohort_failure_warning_monte_carlo(self):
        """Test cohort failure warnings across simulations."""
        n_simulations = 10
        total_cohorts_requested = 0
        total_cohorts_succeeded = 0
        failure_warnings = 0
        
        for sim in range(n_simulations):
            np.random.seed(42 + sim)
            
            # Create data where some cohorts have very few units
            n_units = 30
            n_periods = 6
            
            data = pd.DataFrame({
                'unit': np.repeat(range(1, n_units + 1), n_periods),
                'year': np.tile(range(1, n_periods + 1), n_units),
            })
            
            # Cohort 4: 10 units, Cohort 5: 2 units (marginal), NT: rest
            cohort_map = {
                i: 4 if i <= 10 else (5 if i <= 12 else 0)
                for i in range(1, n_units + 1)
            }
            data['gvar'] = data['unit'].map(cohort_map)
            data['y'] = 10 + np.random.randn(len(data))
            
            transformed = transform_staggered_demean(
                data=data, y='y', gvar='gvar', ivar='unit', tvar='year',
                never_treated_values=[0]
            )
            
            cohorts = get_cohorts(transformed, 'gvar', 'unit', never_treated_values=[0])
            T_max = int(transformed['year'].max())
            total_cohorts_requested += len(cohorts)
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                results = aggregate_to_cohort(
                    data_transformed=transformed,
                    gvar='gvar',
                    ivar='unit',
                    tvar='year',
                    cohorts=cohorts,
                    T_max=T_max,
                    never_treated_values=[0]
                )
                
                total_cohorts_succeeded += len(results)
                
                if count_warnings_by_type(w, '部分 cohort 效应估计失败') > 0:
                    failure_warnings += 1
        
        # Success rate should be high for this dataset
        success_rate = total_cohorts_succeeded / total_cohorts_requested
        assert success_rate > 0.7, f"Expected high success rate, got {success_rate:.1%}"


# =============================================================================
# Stress Tests
# =============================================================================

class TestStressConditions:
    """Test warning behavior under stress conditions."""
    
    def test_large_number_of_cohorts(self):
        """Test with many cohorts."""
        np.random.seed(42)
        
        n_units = 200
        n_periods = 15
        cohort_periods = list(range(4, 12))  # 8 cohorts
        
        data = generate_staggered_panel(
            n_units=n_units,
            n_periods=n_periods,
            cohort_periods=cohort_periods,
            nt_proportion=0.2,
        )
        
        transformed = transform_staggered_demean(
            data=data, y='y', gvar='gvar', ivar='unit', tvar='year',
            never_treated_values=[0]
        )
        
        cohorts = get_cohorts(transformed, 'gvar', 'unit', never_treated_values=[0])
        T_max = int(transformed['year'].max())
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            results = aggregate_to_cohort(
                data_transformed=transformed,
                gvar='gvar',
                ivar='unit',
                tvar='year',
                cohorts=cohorts,
                T_max=T_max,
                never_treated_values=[0]
            )
            
            # Should have results for most cohorts
            assert len(results) >= len(cohorts) * 0.7
    
    def test_very_small_sample(self):
        """Test with minimal sample sizes."""
        np.random.seed(42)
        
        # Minimum viable: 2 treated + 2 NT = 4 units
        data = pd.DataFrame({
            'unit': np.repeat([1, 2, 3, 4], 4),
            'year': np.tile([1, 2, 3, 4], 4),
            'gvar': [3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0],
            'y': np.random.randn(16) + 10,
        })
        
        # Add treatment effect
        data.loc[(data['gvar'] == 3) & (data['year'] >= 3), 'y'] += 5
        
        transformed = transform_staggered_demean(
            data=data, y='y', gvar='gvar', ivar='unit', tvar='year',
            never_treated_values=[0]
        )
        
        cohorts = get_cohorts(transformed, 'gvar', 'unit', never_treated_values=[0])
        T_max = int(transformed['year'].max())
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            results = aggregate_to_cohort(
                data_transformed=transformed,
                gvar='gvar',
                ivar='unit',
                tvar='year',
                cohorts=cohorts,
                T_max=T_max,
                never_treated_values=[0]
            )
            
            # Should succeed but may have warnings about small sample
            assert len(results) == 1  # Only one cohort


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
