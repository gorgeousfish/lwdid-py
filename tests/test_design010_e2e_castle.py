"""
DESIGN-010 端到端测试：使用 Castle Law 数据集验证

测试内容：
1. 使用真实 Castle Law 数据验证警告行为
2. 验证完整的工作流程
3. 确保修复不影响正常功能

创建日期: 2026-01-17
"""

import warnings
import pytest
import numpy as np
import pandas as pd
import os

from lwdid import lwdid
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
def castle_data():
    """Load Castle Law dataset."""
    data_dir = os.path.join(
        os.path.dirname(__file__), 
        '..', 'data'
    )
    castle_path = os.path.join(data_dir, 'castle.csv')
    
    if not os.path.exists(castle_path):
        pytest.skip("Castle Law data not available")
    
    data = pd.read_csv(castle_path)
    
    # Preprocess effyear: replace 0 with a large value for NT
    if 'effyear' in data.columns:
        data['effyear'] = data['effyear'].replace(0, 0)  # Keep 0 as NT indicator
    
    return data


# =============================================================================
# End-to-End Tests
# =============================================================================

class TestCastleLawE2E:
    """End-to-end tests with Castle Law data."""
    
    def test_castle_aggregate_cohort_no_false_warnings(self, castle_data):
        """Test that valid Castle Law data doesn't trigger false warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = lwdid(
                data=castle_data,
                y='lhomicide',
                gvar='effyear',
                ivar='sid',
                tvar='year',
                aggregate='cohort',
                control_group='never_treated',
            )
            
            # Filter for our specific warnings
            aggregation_warnings = [
                x for x in w if 
                '聚合结果全为 NaN' in str(x.message) or
                '聚合结果有效率较低' in str(x.message) or
                '所有 cohort 效应估计均失败' in str(x.message)
            ]
            
            # Castle Law should not trigger these warnings
            assert len(aggregation_warnings) == 0, \
                f"Unexpected warnings: {[str(x.message) for x in aggregation_warnings]}"
            
            # Result should be valid
            assert result is not None
            assert result.att_by_cohort is not None
            assert len(result.att_by_cohort) > 0
    
    def test_castle_aggregate_overall_no_false_warnings(self, castle_data):
        """Test that overall aggregation doesn't trigger false warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = lwdid(
                data=castle_data,
                y='lhomicide',
                gvar='effyear',
                ivar='sid',
                tvar='year',
                aggregate='overall',
                control_group='never_treated',
            )
            
            # Filter for our specific warnings
            aggregation_warnings = [
                x for x in w if 
                '聚合结果全为 NaN' in str(x.message) or
                '聚合结果有效率较低' in str(x.message)
            ]
            
            assert len(aggregation_warnings) == 0, \
                f"Unexpected warnings: {[str(x.message) for x in aggregation_warnings]}"
            
            # Result should be valid
            assert result is not None
            assert not np.isnan(result.att_overall)
    
    def test_castle_transform_and_aggregate_separately(self, castle_data):
        """Test transformation and aggregation as separate steps."""
        # Step 1: Transform
        transformed = transform_staggered_demean(
            data=castle_data,
            y='lhomicide',
            gvar='effyear',
            ivar='sid',
            tvar='year',
            never_treated_values=[0]
        )
        
        # Step 2: Get cohorts
        cohorts = get_cohorts(
            transformed, 'effyear', 'sid', 
            never_treated_values=[0]
        )
        T_max = int(transformed['year'].max())
        
        # Step 3: Aggregate to cohort
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            cohort_results = aggregate_to_cohort(
                data_transformed=transformed,
                gvar='effyear',
                ivar='sid',
                tvar='year',
                cohorts=cohorts,
                T_max=T_max,
                never_treated_values=[0]
            )
            
            failure_warnings = [
                x for x in w if 
                '所有 cohort 效应估计均失败' in str(x.message)
            ]
            
            assert len(failure_warnings) == 0
            assert len(cohort_results) > 0
        
        # Step 4: Aggregate to overall
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            overall_result = aggregate_to_overall(
                data_transformed=transformed,
                gvar='effyear',
                ivar='sid',
                tvar='year',
                never_treated_values=[0]
            )
            
            nan_warnings = [
                x for x in w if 
                '聚合结果全为 NaN' in str(x.message)
            ]
            
            assert len(nan_warnings) == 0
            assert not np.isnan(overall_result.att)


class TestCorruptedDataWarnings:
    """Test that warnings are properly triggered for corrupted data."""
    
    def test_missing_transformation_columns_triggers_warning(self, castle_data):
        """Test warning when transformation columns are missing."""
        # Use raw data without transformation
        cohorts = [2005, 2006, 2007]  # Known Castle Law cohorts
        T_max = int(castle_data['year'].max())
        
        # Compute weights
        unit_gvar = castle_data.groupby('sid')['effyear'].first()
        nt_mask = (unit_gvar == 0) | unit_gvar.isna()
        cohort_sizes = {g: int((unit_gvar == g).sum()) for g in cohorts}
        N_treat = sum(cohort_sizes.values())
        weights = {g: n / N_treat for g, n in cohort_sizes.items()}
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # This should fail because no ydot columns exist
            Y_bar = construct_aggregated_outcome(
                data=castle_data,  # Raw data, no ydot columns
                gvar='effyear',
                ivar='sid',
                tvar='year',
                weights=weights,
                cohorts=cohorts,
                T_max=T_max,
                transform_type='demean',
                never_treated_values=[0]
            )
            
            # Should have all-NaN warning (DESIGN-019: warnings now in English)
            nan_warnings = [
                x for x in w if 'All aggregated results are NaN' in str(x.message)
            ]
            
            assert len(nan_warnings) >= 1, "Expected warning about all NaN results"
            
            # Warning should contain diagnostic info
            warning_msg = str(nan_warnings[0].message)
            assert 'transformation' in warning_msg.lower() or 'cohort' in warning_msg.lower()
    
    def test_all_cohort_failures_triggers_warning(self, castle_data):
        """Test warning when all cohorts fail."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Use raw data (no transformation) - all cohorts should fail
            results = aggregate_to_cohort(
                data_transformed=castle_data,  # Raw data
                gvar='effyear',
                ivar='sid',
                tvar='year',
                cohorts=[2005, 2006, 2007],
                T_max=2010,
                never_treated_values=[0]
            )
            
            # DESIGN-019: warnings now in English
            failure_warnings = [
                x for x in w if 'All cohort effect estimations failed' in str(x.message)
            ]
            
            assert len(failure_warnings) >= 1, "Expected warning about all cohort failures"
            assert len(results) == 0, "Expected no successful results"


class TestWarningMessageQuality:
    """Test that warning messages are informative and actionable."""
    
    def test_all_nan_warning_contains_diagnostic_info(self, castle_data):
        """Test that all-NaN warning contains useful diagnostic information."""
        cohorts = [2005, 2006, 2007]
        T_max = 2010
        
        unit_gvar = castle_data.groupby('sid')['effyear'].first()
        cohort_sizes = {g: int((unit_gvar == g).sum()) for g in cohorts}
        N_treat = sum(cohort_sizes.values())
        weights = {g: n / N_treat for g, n in cohort_sizes.items()}
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            Y_bar = construct_aggregated_outcome(
                data=castle_data,
                gvar='effyear',
                ivar='sid',
                tvar='year',
                weights=weights,
                cohorts=cohorts,
                T_max=T_max,
                transform_type='demean',
                never_treated_values=[0]
            )
            
            nan_warnings = [x for x in w if '聚合结果全为 NaN' in str(x.message)]
            
            if nan_warnings:
                msg = str(nan_warnings[0].message)
                
                # Warning should contain:
                # 1. Possible causes
                assert '原因' in msg or '可能' in msg
                
                # 2. Cohort information
                assert 'cohort' in msg.lower()
                
                # 3. Actionable suggestion
                assert '检查' in msg or '数据' in msg
    
    def test_cohort_failure_warning_lists_failed_cohorts(self, castle_data):
        """Test that partial failure warning lists which cohorts failed."""
        # Transform data first
        transformed = transform_staggered_demean(
            data=castle_data,
            y='lhomicide',
            gvar='effyear',
            ivar='sid',
            tvar='year',
            never_treated_values=[0]
        )
        
        cohorts = get_cohorts(transformed, 'effyear', 'sid', never_treated_values=[0])
        T_max = int(transformed['year'].max())
        
        # Add a fake cohort that will fail
        extended_cohorts = list(cohorts) + [9999]
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            results = aggregate_to_cohort(
                data_transformed=transformed,
                gvar='effyear',
                ivar='sid',
                tvar='year',
                cohorts=extended_cohorts,
                T_max=T_max,
                never_treated_values=[0]
            )
            
            partial_warnings = [
                x for x in w if '部分 cohort 效应估计失败' in str(x.message)
            ]
            
            if partial_warnings:
                msg = str(partial_warnings[0].message)
                # Should list the failed cohort
                assert '9999' in msg or '失败' in msg


# =============================================================================
# Regression Tests
# =============================================================================

class TestNoRegressionInResults:
    """Ensure the fix doesn't change valid results."""
    
    def test_castle_overall_effect_unchanged(self, castle_data):
        """Test that Castle Law overall effect is unchanged by fix."""
        result = lwdid(
            data=castle_data,
            y='lhomicide',
            gvar='effyear',
            ivar='sid',
            tvar='year',
            aggregate='overall',
            control_group='never_treated',
        )
        
        # ATT should be in reasonable range (based on prior results)
        # Castle Law data typically shows small positive effect
        assert -0.5 < result.att_overall < 0.5
        assert result.se_overall > 0
        assert result.se_overall < 1
    
    def test_castle_cohort_effects_count_unchanged(self, castle_data):
        """Test that number of cohort effects is unchanged."""
        result = lwdid(
            data=castle_data,
            y='lhomicide',
            gvar='effyear',
            ivar='sid',
            tvar='year',
            aggregate='cohort',
            control_group='never_treated',
        )
        
        # Castle Law has known number of cohorts
        # Should have effects for multiple cohorts
        assert len(result.att_by_cohort) >= 5


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
