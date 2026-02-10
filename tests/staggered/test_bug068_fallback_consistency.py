"""
BUG-068: core.py CI/t_stat/pvalue fallback 与 att/se_att 不一致

测试修复：当 aggregate != 'overall' 时，确保 t_stat, pvalue, ci_lower, ci_upper
的 fallback 值为 np.nan（而非 None），与 se_att = np.nan 保持一致。

状态不一致问题：
- se_att = np.nan 表示"统计上不可计算"
- t_stat = None（修复前）表示"未计算" —— 语义不同

修复后语义一致性：
- None = "未请求/未尝试"（如 ri_pvalue = None 表示未开启 RI）
- np.nan = "已尝试但统计上无效/不可计算"

相关设计问题: DESIGN-044
"""

import math
import numpy as np
import pandas as pd
import pytest

from lwdid import lwdid


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_staggered_data():
    """Create simple synthetic staggered data for testing.
    
    Structure:
    - Units 1-3: Cohort 3 (treated starting period 3)
    - Units 4-6: Cohort 4 (treated starting period 4)
    - Units 7-10: Never treated (gvar=0)
    - Periods: 1 to 5
    """
    np.random.seed(42)
    
    records = []
    for i in range(10):
        unit_id = i + 1
        
        # Units 1-3 -> cohort 3, units 4-6 -> cohort 4, units 7-10 -> never treated
        if i < 3:
            gvar = 3
        elif i < 6:
            gvar = 4
        else:
            gvar = 0  # never treated
        
        for t in range(1, 6):
            base = 10 + i * 0.5
            trend = t * 0.2
            effect = 2.0 if (gvar > 0 and t >= gvar) else 0
            noise = np.random.normal(0, 0.5)
            y = base + trend + effect + noise
            
            records.append({
                'unit': unit_id,
                'time': t,
                'y': y,
                'gvar': gvar,
            })
    
    return pd.DataFrame(records)


# =============================================================================
# Test: aggregate='none' should return np.nan for fallback values
# =============================================================================

class TestAggregatNoneFallback:
    """Test that aggregate='none' returns np.nan (not None) for inference stats."""
    
    def test_t_stat_is_nan_not_none(self, simple_staggered_data):
        """t_stat should be np.nan when aggregate='none'."""
        results = lwdid(
            data=simple_staggered_data,
            y='y',
            ivar='unit',
            tvar='time',
            gvar='gvar',
            rolling='demean',
            aggregate='none',
        )
        
        # Should be np.nan (float), not None
        assert results.t_stat is not None, "t_stat should be np.nan, not None"
        assert isinstance(results.t_stat, (float, np.floating)), \
            f"t_stat should be float type, got {type(results.t_stat)}"
        assert math.isnan(results.t_stat), \
            f"t_stat should be nan when aggregate='none', got {results.t_stat}"
    
    def test_pvalue_is_nan_not_none(self, simple_staggered_data):
        """pvalue should be np.nan when aggregate='none'."""
        results = lwdid(
            data=simple_staggered_data,
            y='y',
            ivar='unit',
            tvar='time',
            gvar='gvar',
            rolling='demean',
            aggregate='none',
        )
        
        assert results.pvalue is not None, "pvalue should be np.nan, not None"
        assert isinstance(results.pvalue, (float, np.floating)), \
            f"pvalue should be float type, got {type(results.pvalue)}"
        assert math.isnan(results.pvalue), \
            f"pvalue should be nan when aggregate='none', got {results.pvalue}"
    
    def test_ci_lower_is_nan_not_none(self, simple_staggered_data):
        """ci_lower should be np.nan when aggregate='none'."""
        results = lwdid(
            data=simple_staggered_data,
            y='y',
            ivar='unit',
            tvar='time',
            gvar='gvar',
            rolling='demean',
            aggregate='none',
        )
        
        assert results.ci_lower is not None, "ci_lower should be np.nan, not None"
        assert isinstance(results.ci_lower, (float, np.floating)), \
            f"ci_lower should be float type, got {type(results.ci_lower)}"
        assert math.isnan(results.ci_lower), \
            f"ci_lower should be nan when aggregate='none', got {results.ci_lower}"
    
    def test_ci_upper_is_nan_not_none(self, simple_staggered_data):
        """ci_upper should be np.nan when aggregate='none'."""
        results = lwdid(
            data=simple_staggered_data,
            y='y',
            ivar='unit',
            tvar='time',
            gvar='gvar',
            rolling='demean',
            aggregate='none',
        )
        
        assert results.ci_upper is not None, "ci_upper should be np.nan, not None"
        assert isinstance(results.ci_upper, (float, np.floating)), \
            f"ci_upper should be float type, got {type(results.ci_upper)}"
        assert math.isnan(results.ci_upper), \
            f"ci_upper should be nan when aggregate='none', got {results.ci_upper}"
    
    def test_se_att_consistency(self, simple_staggered_data):
        """se_att should also be np.nan when aggregate='none' (DESIGN-044)."""
        results = lwdid(
            data=simple_staggered_data,
            y='y',
            ivar='unit',
            tvar='time',
            gvar='gvar',
            rolling='demean',
            aggregate='none',
        )
        
        # se_att should be nan (from DESIGN-044)
        assert math.isnan(results.se_att), \
            f"se_att should be nan when aggregate='none', got {results.se_att}"
    
    def test_all_fallback_values_consistent(self, simple_staggered_data):
        """All fallback values should be np.nan, maintaining consistency."""
        results = lwdid(
            data=simple_staggered_data,
            y='y',
            ivar='unit',
            tvar='time',
            gvar='gvar',
            rolling='demean',
            aggregate='none',
        )
        
        # All should be nan
        fallback_attrs = ['se_att', 't_stat', 'pvalue', 'ci_lower', 'ci_upper']
        for attr in fallback_attrs:
            value = getattr(results, attr)
            assert value is not None, \
                f"{attr} should be np.nan, not None"
            assert isinstance(value, (float, np.floating)), \
                f"{attr} should be float type, got {type(value)}"
            assert math.isnan(value), \
                f"{attr} should be nan when aggregate='none', got {value}"
        
        # But att should have a valid value (the mean of cohort-time effects)
        assert results.att is not None, "att should have fallback value"
        assert not math.isnan(results.att), "att should have valid value (mean)"


# =============================================================================
# Test: aggregate='cohort' should return np.nan for fallback values
# =============================================================================

class TestAggregateCohortFallback:
    """Test that aggregate='cohort' populates top-level stats from cohort-weighted average.
    
    Note: aggregate='cohort' computes cohort-specific effects and then derives
    top-level statistics (att, se_att, etc.) as the n_units-weighted average
    across cohort effects. This is more informative than returning nan.
    """
    
    def test_cohort_aggregate_has_valid_top_level_stats(self, simple_staggered_data):
        """aggregate='cohort' should have valid top-level stats from weighted average."""
        results = lwdid(
            data=simple_staggered_data,
            y='y',
            ivar='unit',
            tvar='time',
            gvar='gvar',
            rolling='demean',
            aggregate='cohort',
        )
        
        # Top-level stats should be valid floats (weighted average of cohort effects)
        top_level_attrs = ['se_att', 't_stat', 'pvalue', 'ci_lower', 'ci_upper']
        for attr in top_level_attrs:
            value = getattr(results, attr)
            assert value is not None, \
                f"{attr} should not be None for aggregate='cohort'"
            assert isinstance(value, (float, np.floating)), \
                f"{attr} should be float type, got {type(value)}"
            assert np.isfinite(value), \
                f"{attr} should be finite when aggregate='cohort', got {value}"
        
        # att should also be valid
        assert results.att is not None
        assert np.isfinite(results.att)
        
        # Cohort-level effects should have valid values
        assert results.att_by_cohort is not None
        assert len(results.att_by_cohort) > 0
        for _, row in results.att_by_cohort.iterrows():
            assert not math.isnan(row['att']), "Cohort ATT should have valid value"
            assert not math.isnan(row['se']), "Cohort SE should have valid value"


# =============================================================================
# Test: aggregate='overall' should return valid numeric values
# =============================================================================

class TestAggregateOverallValid:
    """Test that aggregate='overall' returns valid numeric values (not nan/None)."""
    
    def test_overall_aggregate_has_valid_values(self, simple_staggered_data):
        """aggregate='overall' should have valid numeric inference stats."""
        results = lwdid(
            data=simple_staggered_data,
            y='y',
            ivar='unit',
            tvar='time',
            gvar='gvar',
            rolling='demean',
            aggregate='overall',
        )
        
        # All should have valid (non-nan) values
        assert not math.isnan(results.att), f"att should be valid, got {results.att}"
        assert not math.isnan(results.se_att), f"se_att should be valid, got {results.se_att}"
        assert not math.isnan(results.t_stat), f"t_stat should be valid, got {results.t_stat}"
        assert not math.isnan(results.pvalue), f"pvalue should be valid, got {results.pvalue}"
        assert not math.isnan(results.ci_lower), f"ci_lower should be valid, got {results.ci_lower}"
        assert not math.isnan(results.ci_upper), f"ci_upper should be valid, got {results.ci_upper}"
    
    def test_overall_t_stat_computation(self, simple_staggered_data):
        """t_stat should be computed as att / se_att for aggregate='overall'."""
        results = lwdid(
            data=simple_staggered_data,
            y='y',
            ivar='unit',
            tvar='time',
            gvar='gvar',
            rolling='demean',
            aggregate='overall',
        )
        
        expected_t = results.att / results.se_att
        # Allow small numerical tolerance
        assert abs(results.t_stat - expected_t) < 1e-10, \
            f"t_stat={results.t_stat} should equal att/se_att={expected_t}"
    
    def test_overall_ci_bounds(self, simple_staggered_data):
        """CI bounds should be reasonable for aggregate='overall'."""
        results = lwdid(
            data=simple_staggered_data,
            y='y',
            ivar='unit',
            tvar='time',
            gvar='gvar',
            rolling='demean',
            aggregate='overall',
        )
        
        # CI lower should be less than ATT, which should be less than CI upper
        assert results.ci_lower < results.att < results.ci_upper, \
            f"CI bounds should satisfy: {results.ci_lower} < {results.att} < {results.ci_upper}"


# =============================================================================
# Test: Type consistency
# =============================================================================

class TestTypeConsistency:
    """Test that return types are consistent across modes."""
    
    def test_fallback_types_are_float(self, simple_staggered_data):
        """Fallback values should be float (np.nan), not NoneType."""
        results = lwdid(
            data=simple_staggered_data,
            y='y',
            ivar='unit',
            tvar='time',
            gvar='gvar',
            rolling='demean',
            aggregate='none',
        )
        
        # Check that all are float type (not NoneType)
        for attr in ['se_att', 't_stat', 'pvalue', 'ci_lower', 'ci_upper']:
            value = getattr(results, attr)
            assert isinstance(value, (float, np.floating)), \
                f"{attr} should be float, got {type(value).__name__}"
    
    def test_valid_types_are_float(self, simple_staggered_data):
        """Valid values should also be float type."""
        results = lwdid(
            data=simple_staggered_data,
            y='y',
            ivar='unit',
            tvar='time',
            gvar='gvar',
            rolling='demean',
            aggregate='overall',
        )
        
        # Check that all are float type (not NoneType)
        for attr in ['att', 'se_att', 't_stat', 'pvalue', 'ci_lower', 'ci_upper']:
            value = getattr(results, attr)
            assert isinstance(value, (float, np.floating, int, np.integer)), \
                f"{attr} should be numeric, got {type(value).__name__}"


# =============================================================================
# Test: Semantic distinction from None
# =============================================================================

class TestNoneVsNanSemantics:
    """Test that np.nan is semantically different from None.
    
    - None = "not requested/not attempted" (e.g., ri_pvalue when ri=False)
    - np.nan = "attempted but statistically invalid/not computable"
    """
    
    def test_ri_pvalue_none_vs_inference_nan(self, simple_staggered_data):
        """ri_pvalue should be None (not nan) when RI not requested."""
        results = lwdid(
            data=simple_staggered_data,
            y='y',
            ivar='unit',
            tvar='time',
            gvar='gvar',
            rolling='demean',
            aggregate='none',
            ri=False,  # RI not requested
        )
        
        # ri_pvalue should be None (not requested)
        assert results.ri_pvalue is None, \
            f"ri_pvalue should be None when ri=False, got {results.ri_pvalue}"
        
        # But inference stats should be nan (attempted but invalid)
        assert results.t_stat is not None and math.isnan(results.t_stat), \
            "t_stat should be nan (not None) when aggregate='none'"


# =============================================================================
# Test: Edge cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases for fallback value handling."""
    
    def test_detrend_transformation_fallback(self, simple_staggered_data):
        """Fallback should work with detrend transformation too."""
        results = lwdid(
            data=simple_staggered_data,
            y='y',
            ivar='unit',
            tvar='time',
            gvar='gvar',
            rolling='detrend',  # Different transformation
            aggregate='none',
        )
        
        # Should still have nan fallbacks
        assert math.isnan(results.t_stat), "t_stat should be nan"
        assert math.isnan(results.pvalue), "pvalue should be nan"
        assert math.isnan(results.ci_lower), "ci_lower should be nan"
        assert math.isnan(results.ci_upper), "ci_upper should be nan"


# =============================================================================
# Test: Pandas NaN handling
# =============================================================================

class TestPandasNaNHandling:
    """Test that np.nan works well with pandas operations."""
    
    def test_nan_in_dataframe_export(self, simple_staggered_data):
        """np.nan values should export correctly to DataFrame."""
        results = lwdid(
            data=simple_staggered_data,
            y='y',
            ivar='unit',
            tvar='time',
            gvar='gvar',
            rolling='demean',
            aggregate='none',
        )
        
        # Create a summary dict
        summary_dict = {
            'att': results.att,
            'se_att': results.se_att,
            't_stat': results.t_stat,
            'pvalue': results.pvalue,
            'ci_lower': results.ci_lower,
            'ci_upper': results.ci_upper,
        }
        
        df = pd.DataFrame([summary_dict])
        
        # Should have nan values that pandas can detect
        assert pd.isna(df['se_att'].iloc[0])
        assert pd.isna(df['t_stat'].iloc[0])
        assert pd.isna(df['pvalue'].iloc[0])
        assert pd.isna(df['ci_lower'].iloc[0])
        assert pd.isna(df['ci_upper'].iloc[0])


# =============================================================================
# Run tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
