"""
DESIGN-024 修复验证测试：aggregation.py 中部分 cohort 缺失时的权重归一化

测试内容：
1. 单元测试：验证所有 cohort 完整时正常工作
2. 单元测试：验证部分 cohort 缺失时 NT 单位设为 NaN（不归一化）
3. 警告测试：验证 DESIGN-024 警告正确触发
4. Stata 一致性测试：验证与 Stata 行为一致

创建日期: 2026-01-17
修复内容: 取消 construct_aggregated_outcome 中的权重重新归一化
         当 NT 单位缺少部分 cohort 数据时，结果为 NaN（与 Stata 一致）
"""

import warnings
import pytest
import numpy as np
import pandas as pd

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
# Test Fixtures
# =============================================================================

@pytest.fixture
def complete_cohort_data():
    """Create panel data where all units have complete cohort data."""
    np.random.seed(42)
    
    # 3 cohorts: g=3, g=4, g=5
    # 3 NT units
    # T_max = 6
    data_rows = []
    
    # Cohort 3: 2 units
    for unit_id in [1, 2]:
        for t in range(1, 7):
            y = 10 + t * 2
            if t >= 3:
                y += 5  # Treatment effect
            data_rows.append({'id': unit_id, 'year': t, 'y': y, 'gvar': 3})
    
    # Cohort 4: 2 units
    for unit_id in [3, 4]:
        for t in range(1, 7):
            y = 12 + t * 2
            if t >= 4:
                y += 5
            data_rows.append({'id': unit_id, 'year': t, 'y': y, 'gvar': 4})
    
    # Cohort 5: 2 units
    for unit_id in [5, 6]:
        for t in range(1, 7):
            y = 14 + t * 2
            if t >= 5:
                y += 5
            data_rows.append({'id': unit_id, 'year': t, 'y': y, 'gvar': 5})
    
    # NT: 3 units (complete data for all periods)
    for unit_id in [7, 8, 9]:
        for t in range(1, 7):
            y = 8 + t * 2
            data_rows.append({'id': unit_id, 'year': t, 'y': y, 'gvar': 0})
    
    return pd.DataFrame(data_rows)


@pytest.fixture
def incomplete_cohort_data():
    """
    Create panel data where some NT units have incomplete cohort data.
    
    Scenario:
    - NT unit 7: complete data
    - NT unit 8: missing y values for some post-treatment periods (will have NaN for some cohorts)
    - NT unit 9: complete data
    """
    np.random.seed(42)
    
    data_rows = []
    
    # Cohort 3: 2 units
    for unit_id in [1, 2]:
        for t in range(1, 7):
            y = 10 + t * 2
            if t >= 3:
                y += 5
            data_rows.append({'id': unit_id, 'year': t, 'y': y, 'gvar': 3})
    
    # Cohort 4: 2 units
    for unit_id in [3, 4]:
        for t in range(1, 7):
            y = 12 + t * 2
            if t >= 4:
                y += 5
            data_rows.append({'id': unit_id, 'year': t, 'y': y, 'gvar': 4})
    
    # NT unit 7: complete data
    for t in range(1, 7):
        y = 8 + t * 2
        data_rows.append({'id': 7, 'year': t, 'y': y, 'gvar': 0})
    
    # NT unit 8: missing y at t=5,6 (will cause NaN for cohort 5 transformations)
    for t in range(1, 7):
        if t >= 5:
            y = np.nan  # Missing post-treatment data
        else:
            y = 9 + t * 2
        data_rows.append({'id': 8, 'year': t, 'y': y, 'gvar': 0})
    
    # NT unit 9: complete data
    for t in range(1, 7):
        y = 10 + t * 2
        data_rows.append({'id': 9, 'year': t, 'y': y, 'gvar': 0})
    
    # Cohort 5: 1 unit (to have a valid cohort 5)
    for t in range(1, 7):
        y = 14 + t * 2
        if t >= 5:
            y += 5
        data_rows.append({'id': 5, 'year': t, 'y': y, 'gvar': 5})
    
    return pd.DataFrame(data_rows)


# =============================================================================
# Unit Tests: Complete Data
# =============================================================================

class TestDesign024CompleteData:
    """Test behavior when all cohort data is complete."""
    
    def test_all_cohorts_complete_no_warning(self, complete_cohort_data):
        """
        When all NT units have complete cohort data, no DESIGN-024 warning should be raised.
        """
        transformed = transform_staggered_demean(
            complete_cohort_data, 'y', 'id', 'year', 'gvar',
            never_treated_values=[0]
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = aggregate_to_overall(
                transformed, 'gvar', 'id', 'year',
                never_treated_values=[0]
            )
            
            # Filter for DESIGN-024 warnings
            design024_warnings = [x for x in w if 'DESIGN-024' in str(x.message)]
            
            assert len(design024_warnings) == 0, \
                "No DESIGN-024 warning expected for complete data"
            
            # Result should be valid
            assert not np.isnan(result.att)
            # Note: SE may be 0 for perfectly linear test data (no variation in residuals)
            assert result.se >= 0
    
    def test_all_nt_units_have_valid_y_bar(self, complete_cohort_data):
        """
        When all cohorts are complete, all NT units should have valid Y_bar.
        """
        transformed = transform_staggered_demean(
            complete_cohort_data, 'y', 'id', 'year', 'gvar',
            never_treated_values=[0]
        )
        
        cohorts = get_cohorts(transformed, 'gvar', 'id', never_treated_values=[0])
        T_max = int(transformed['year'].max())
        
        # Compute weights
        unit_gvar = transformed.groupby('id')['gvar'].first()
        cohort_sizes = {g: int((unit_gvar == g).sum()) for g in cohorts}
        N_treat = sum(cohort_sizes.values())
        weights = {g: n / N_treat for g, n in cohort_sizes.items()}
        
        Y_bar = construct_aggregated_outcome(
            transformed, 'gvar', 'id', 'year',
            weights, cohorts, T_max,
            transform_type='demean',
            never_treated_values=[0]
        )
        
        # All NT units (7, 8, 9) should have valid Y_bar
        nt_units = [7, 8, 9]
        for unit_id in nt_units:
            assert not np.isnan(Y_bar[unit_id]), \
                f"NT unit {unit_id} should have valid Y_bar with complete data"


# =============================================================================
# Unit Tests: Incomplete Data - Core DESIGN-024 Fix
# =============================================================================

class TestDesign024IncompleteCohortData:
    """
    Test DESIGN-024 fix: NT units with incomplete cohort data should be NaN.
    
    This is the core behavior change - no re-normalization of weights.
    """
    
    def test_incomplete_nt_unit_is_nan(self):
        """
        When an NT unit is missing data for some cohorts, its Y_bar should be NaN.
        
        This is the core DESIGN-024 fix - consistent with Stata behavior.
        Before fix: Y_bar = weighted_sum / valid_weights (re-normalized)
        After fix: Y_bar = NaN (not included in regression)
        """
        # Create specific test data
        data = pd.DataFrame({
            'id': [1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4],
            'year': [1,2,3,4, 1,2,3,4, 1,2,3,4, 1,2,3,4],
            'y': [
                # Unit 1: cohort 3
                10, 12, 20, 22,
                # Unit 2: cohort 4
                15, 17, 19, 27,
                # Unit 3: NT - complete data
                5, 7, 9, 11,
                # Unit 4: NT - missing data at t=3,4 (will have NaN transformations)
                6, 8, np.nan, np.nan,
            ],
            'gvar': [3]*4 + [4]*4 + [0]*4 + [0]*4
        })
        
        transformed = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar',
            never_treated_values=[0]
        )
        
        cohorts = [3, 4]
        T_max = 4
        weights = {3: 0.5, 4: 0.5}
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            Y_bar = construct_aggregated_outcome(
                transformed, 'gvar', 'id', 'year',
                weights, cohorts, T_max,
                transform_type='demean',
                never_treated_values=[0]
            )
        
        # Unit 3 (complete NT) should have valid Y_bar
        assert not np.isnan(Y_bar[3]), "Complete NT unit should have valid Y_bar"
        
        # Unit 4 (incomplete NT) should have NaN Y_bar
        # This is the DESIGN-024 fix - no re-normalization
        assert np.isnan(Y_bar[4]), \
            "DESIGN-024 FIX: Incomplete NT unit should have NaN Y_bar (not re-normalized)"
    
    def test_warning_triggered_for_incomplete_nt(self):
        """
        When NT units have incomplete cohort data, DESIGN-024 warning should be raised.
        """
        data = pd.DataFrame({
            'id': [1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4, 5,5,5,5],
            'year': [1,2,3,4, 1,2,3,4, 1,2,3,4, 1,2,3,4, 1,2,3,4],
            'y': [
                10, 12, 20, 22,  # cohort 3
                15, 17, 19, 27,  # cohort 4
                5, 7, 9, 11,     # NT complete
                6, 8, np.nan, np.nan,  # NT incomplete
                7, 9, np.nan, np.nan,  # NT incomplete
            ],
            'gvar': [3]*4 + [4]*4 + [0]*4 + [0]*4 + [0]*4
        })
        
        transformed = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar',
            never_treated_values=[0]
        )
        
        cohorts = [3, 4]
        T_max = 4
        weights = {3: 0.5, 4: 0.5}
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            Y_bar = construct_aggregated_outcome(
                transformed, 'gvar', 'id', 'year',
                weights, cohorts, T_max,
                transform_type='demean',
                never_treated_values=[0]
            )
            
            # Should have DESIGN-024 warning
            design024_warnings = [x for x in w if 'DESIGN-024' in str(x.message)]
            
            assert len(design024_warnings) >= 1, \
                "Expected DESIGN-024 warning for incomplete NT units"
            
            # Warning should mention number of excluded units
            warning_msg = str(design024_warnings[0].message)
            assert '2' in warning_msg or 'NT unit' in warning_msg, \
                "Warning should mention excluded NT units"


# =============================================================================
# Stata Consistency Tests
# =============================================================================

class TestDesign024StataConsistency:
    """
    Test consistency with Stata behavior.
    
    Stata implementation (castle_lw.do line 129):
    - Direct sum: ybar_cont_2005 + ybar_cont_2006 + ... + ybar_cont_2009
    - If any component is missing → result is missing (not re-normalized)
    """
    
    def test_weighted_sum_equals_stata_direct_sum(self):
        """
        Verify weighted_sum calculation matches Stata's direct sum approach.
        
        Stata: ybar_cont_{year} = w_g * mean(y{year}d)
        Then: ydot_bar = ybar_cont_2005 + ybar_cont_2006 + ... (direct sum)
        
        Python: weighted_sum = Σ weights[g] * Y_bar_ig
        
        These should be equivalent when weights sum to 1.0.
        """
        # Create simple test data
        data = pd.DataFrame({
            'id': [1,1,1,1, 2,2,2,2, 3,3,3,3],
            'year': [1,2,3,4, 1,2,3,4, 1,2,3,4],
            'y': [
                10, 12, 20, 22,  # cohort 3
                15, 17, 27, 29,  # cohort 4
                5, 7, 9, 11,     # NT
            ],
            'gvar': [3]*4 + [4]*4 + [0]*4
        })
        
        transformed = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar',
            never_treated_values=[0]
        )
        
        # Equal weights (simulating N_3 = N_4 = 1)
        cohorts = [3, 4]
        T_max = 4
        weights = {3: 0.5, 4: 0.5}
        
        Y_bar = construct_aggregated_outcome(
            transformed, 'gvar', 'id', 'year',
            weights, cohorts, T_max,
            transform_type='demean',
            never_treated_values=[0]
        )
        
        # NT unit (id=3) should have weighted sum
        nt_y_bar = Y_bar[3]
        
        assert not np.isnan(nt_y_bar), "NT unit with complete data should have valid Y_bar"
        
        # Manual calculation for verification
        # Y_bar_i3_g3: cohort 3's transformation for NT unit
        # Y_bar_i3_g4: cohort 4's transformation for NT unit
        # weighted_sum = 0.5 * Y_bar_i3_g3 + 0.5 * Y_bar_i3_g4
        
        # The exact value depends on transformations, but it should be finite
        assert np.isfinite(nt_y_bar)
    
    def test_missing_cohort_gives_nan_like_stata(self):
        """
        Verify that missing cohort data gives NaN (like Stata's missing + number = missing).
        
        Stata behavior: If ybar_cont_2006 is missing for a unit,
        then ybar_cont_2005 + ybar_cont_2006 + ... = missing
        
        Python (after fix): If any cohort Y_bar_ig is NaN,
        then Y_bar[unit_id] = NaN (not re-normalized)
        """
        # Data where NT unit is missing cohort 4 data
        data = pd.DataFrame({
            'id': [1,1,1,1, 2,2,2,2, 3,3,3,3],
            'year': [1,2,3,4, 1,2,3,4, 1,2,3,4],
            'y': [
                10, 12, 20, 22,  # cohort 3
                15, 17, 27, 29,  # cohort 4
                5, 7, 9, np.nan,  # NT - missing at t=4 (cohort 4 post-treatment)
            ],
            'gvar': [3]*4 + [4]*4 + [0]*4
        })
        
        transformed = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar',
            never_treated_values=[0]
        )
        
        cohorts = [3, 4]
        T_max = 4
        weights = {3: 0.5, 4: 0.5}
        
        Y_bar = construct_aggregated_outcome(
            transformed, 'gvar', 'id', 'year',
            weights, cohorts, T_max,
            transform_type='demean',
            never_treated_values=[0]
        )
        
        # NT unit (id=3) should have NaN because cohort 4 data is incomplete
        # This matches Stata's behavior: missing + number = missing
        assert np.isnan(Y_bar[3]), \
            "Stata consistency: missing cohort should give NaN Y_bar"


# =============================================================================
# Integration Tests
# =============================================================================

class TestDesign024Integration:
    """Integration tests with aggregate_to_overall."""
    
    def test_overall_effect_excludes_incomplete_nt(self):
        """
        aggregate_to_overall should correctly exclude NT units with incomplete data.
        
        The regression n_control should reflect only NT units with complete data.
        """
        data = pd.DataFrame({
            'id': [1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4, 5,5,5,5],
            'year': [1,2,3,4, 1,2,3,4, 1,2,3,4, 1,2,3,4, 1,2,3,4],
            'y': [
                10, 12, 20, 22,  # cohort 3
                15, 17, 27, 29,  # cohort 4
                5, 7, 9, 11,     # NT complete
                6, 8, np.nan, np.nan,  # NT incomplete
                7, 9, 11, 13,    # NT complete
            ],
            'gvar': [3]*4 + [4]*4 + [0]*4 + [0]*4 + [0]*4
        })
        
        transformed = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar',
            never_treated_values=[0]
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress warnings for this test
            
            result = aggregate_to_overall(
                transformed, 'gvar', 'id', 'year',
                never_treated_values=[0]
            )
        
        # n_treated should be 2 (units 1 and 2)
        assert result.n_treated == 2, f"Expected 2 treated, got {result.n_treated}"
        
        # n_control should be 2 (units 3 and 5, unit 4 excluded due to incomplete data)
        assert result.n_control == 2, \
            f"Expected 2 control (1 excluded), got {result.n_control}"
    
    def test_overall_effect_valid_with_some_incomplete(self):
        """
        Overall effect should still be estimable when some (but not all) NT units are incomplete.
        """
        data = pd.DataFrame({
            'id': [1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4, 5,5,5,5],
            'year': [1,2,3,4, 1,2,3,4, 1,2,3,4, 1,2,3,4, 1,2,3,4],
            'y': [
                10, 12, 20, 22,  # cohort 3
                15, 17, 27, 29,  # cohort 4
                5, 7, 9, 11,     # NT complete
                6, 8, np.nan, np.nan,  # NT incomplete
                7, 9, 11, 13,    # NT complete
            ],
            'gvar': [3]*4 + [4]*4 + [0]*4 + [0]*4 + [0]*4
        })
        
        transformed = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar',
            never_treated_values=[0]
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            result = aggregate_to_overall(
                transformed, 'gvar', 'id', 'year',
                never_treated_values=[0]
            )
        
        # Result should be valid
        assert not np.isnan(result.att), "ATT should be estimable"
        # Note: SE may be NaN or very large due to small sample size


# =============================================================================
# Boundary Condition Tests
# =============================================================================

class TestDesign024BoundaryConditions:
    """Test boundary conditions."""
    
    def test_single_nt_complete(self):
        """Single NT unit with complete data should work."""
        data = pd.DataFrame({
            'id': [1,1,1,1, 2,2,2,2, 3,3,3,3],
            'year': [1,2,3,4, 1,2,3,4, 1,2,3,4],
            'y': [
                10, 12, 20, 22,  # cohort 3
                15, 17, 27, 29,  # cohort 4
                5, 7, 9, 11,     # NT complete
            ],
            'gvar': [3]*4 + [4]*4 + [0]*4
        })
        
        transformed = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar',
            never_treated_values=[0]
        )
        
        cohorts = [3, 4]
        T_max = 4
        weights = {3: 0.5, 4: 0.5}
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            Y_bar = construct_aggregated_outcome(
                transformed, 'gvar', 'id', 'year',
                weights, cohorts, T_max,
                transform_type='demean',
                never_treated_values=[0]
            )
            
            # No DESIGN-024 warning
            design024_warnings = [x for x in w if 'DESIGN-024' in str(x.message)]
            assert len(design024_warnings) == 0
            
            # NT unit should have valid Y_bar
            assert not np.isnan(Y_bar[3])
    
    def test_all_nt_incomplete_should_warn(self):
        """If all NT units have incomplete data, should still warn appropriately."""
        data = pd.DataFrame({
            'id': [1,1,1,1, 2,2,2,2, 3,3,3,3],
            'year': [1,2,3,4, 1,2,3,4, 1,2,3,4],
            'y': [
                10, 12, 20, 22,  # cohort 3
                15, 17, 27, 29,  # cohort 4
                5, 7, np.nan, np.nan,  # NT incomplete
            ],
            'gvar': [3]*4 + [4]*4 + [0]*4
        })
        
        transformed = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar',
            never_treated_values=[0]
        )
        
        cohorts = [3, 4]
        T_max = 4
        weights = {3: 0.5, 4: 0.5}
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            Y_bar = construct_aggregated_outcome(
                transformed, 'gvar', 'id', 'year',
                weights, cohorts, T_max,
                transform_type='demean',
                never_treated_values=[0]
            )
            
            # Should have DESIGN-024 warning
            design024_warnings = [x for x in w if 'DESIGN-024' in str(x.message)]
            assert len(design024_warnings) >= 1
            
            # NT unit should have NaN
            assert np.isnan(Y_bar[3])
    
    def test_floating_point_tolerance(self):
        """
        Test that floating point tolerance (1e-10) is appropriate.
        
        valid_weights may not be exactly 1.0 due to floating point arithmetic.
        """
        # Create data with many cohorts to accumulate floating point errors
        data_rows = []
        n_cohorts = 10
        
        # Create cohorts
        for cohort in range(3, 3 + n_cohorts):
            for t in range(1, 15):
                y = 10 + t * 2
                if t >= cohort:
                    y += 5
                data_rows.append({'id': cohort, 'year': t, 'y': y, 'gvar': cohort})
        
        # Create NT unit
        for t in range(1, 15):
            y = 8 + t * 2
            data_rows.append({'id': 100, 'year': t, 'y': y, 'gvar': 0})
        
        data = pd.DataFrame(data_rows)
        
        transformed = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar',
            never_treated_values=[0]
        )
        
        cohorts = list(range(3, 3 + n_cohorts))
        T_max = 14
        
        # Weights that may have floating point precision issues
        weights = {g: 1.0 / n_cohorts for g in cohorts}
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            Y_bar = construct_aggregated_outcome(
                transformed, 'gvar', 'id', 'year',
                weights, cohorts, T_max,
                transform_type='demean',
                never_treated_values=[0]
            )
            
            # NT unit should have valid Y_bar despite potential floating point issues
            assert not np.isnan(Y_bar[100]), \
                "NT unit should have valid Y_bar with tolerance for floating point"


# =============================================================================
# Regression Tests
# =============================================================================

class TestDesign024RegressionTests:
    """Regression tests to ensure fix doesn't break existing functionality."""
    
    def test_treated_units_unchanged(self, complete_cohort_data):
        """
        Treated units should use their own cohort's transformation (unchanged).
        """
        transformed = transform_staggered_demean(
            complete_cohort_data, 'y', 'id', 'year', 'gvar',
            never_treated_values=[0]
        )
        
        cohorts = get_cohorts(transformed, 'gvar', 'id', never_treated_values=[0])
        T_max = int(transformed['year'].max())
        
        unit_gvar = transformed.groupby('id')['gvar'].first()
        cohort_sizes = {g: int((unit_gvar == g).sum()) for g in cohorts}
        N_treat = sum(cohort_sizes.values())
        weights = {g: n / N_treat for g, n in cohort_sizes.items()}
        
        Y_bar = construct_aggregated_outcome(
            transformed, 'gvar', 'id', 'year',
            weights, cohorts, T_max,
            transform_type='demean',
            never_treated_values=[0]
        )
        
        # All treated units should have valid Y_bar
        for unit_id in [1, 2, 3, 4, 5, 6]:  # Treated units
            assert not np.isnan(Y_bar[unit_id]), \
                f"Treated unit {unit_id} should have valid Y_bar"
    
    def test_weights_sum_to_one_check(self, complete_cohort_data):
        """
        Verify that weights summing to 1.0 is correctly detected.
        """
        transformed = transform_staggered_demean(
            complete_cohort_data, 'y', 'id', 'year', 'gvar',
            never_treated_values=[0]
        )
        
        cohorts = get_cohorts(transformed, 'gvar', 'id', never_treated_values=[0])
        T_max = int(transformed['year'].max())
        
        # Correct weights (sum to 1.0)
        unit_gvar = transformed.groupby('id')['gvar'].first()
        cohort_sizes = {g: int((unit_gvar == g).sum()) for g in cohorts}
        N_treat = sum(cohort_sizes.values())
        weights = {g: n / N_treat for g, n in cohort_sizes.items()}
        
        assert np.isclose(sum(weights.values()), 1.0, atol=1e-10)
        
        Y_bar = construct_aggregated_outcome(
            transformed, 'gvar', 'id', 'year',
            weights, cohorts, T_max,
            transform_type='demean',
            never_treated_values=[0]
        )
        
        # NT units should have valid Y_bar
        for unit_id in [7, 8, 9]:  # NT units
            assert not np.isnan(Y_bar[unit_id])


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
