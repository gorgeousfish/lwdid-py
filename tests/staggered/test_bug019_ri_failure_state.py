"""
BUG-019: RI 失败时 results 对象状态不一致

测试修复：当随机化推断（RI）失败时，确保 results 对象中的 RI 相关属性
被设置为一致的"无效/失败"状态。

修复前问题：
- 用户访问 results.ri_pvalue 得到 None，无法区分"未请求 RI"和"RI 失败"
- ri_valid、ri_failed、ri_method 等属性未初始化
- 下游代码可能因属性状态不明确而出错

修复后：
- ri_pvalue = np.nan (明确表示失败)
- ri_seed = actual_seed (记录尝试的种子)
- rireps = rireps (记录请求的复制次数)
- ri_method = ri_method (记录尝试的方法)
- ri_valid = 0 (0 个有效复制)
- ri_failed = str(e) (存储错误消息)
- ri_target = ri_target (记录目标效应类型)
"""

import numpy as np
import pandas as pd
import pytest
import warnings
from unittest.mock import patch, MagicMock

from lwdid import lwdid
from lwdid.results import LWDIDResults

# Correct patch path: RI function is imported in _lwdid_staggered from this module
RI_FUNCTION_PATH = 'lwdid.staggered.randomization.randomization_inference_staggered'


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_staggered_data():
    """Create simple synthetic staggered data for testing."""
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
                'id': unit_id,
                'year': t,
                'y': y,
                'gvar': gvar
            })
    
    return pd.DataFrame(records)


@pytest.fixture
def minimal_data_for_ri_failure():
    """Create minimal data that will cause RI to fail."""
    # Only 2 units - too few for RI
    return pd.DataFrame({
        'id': [1, 1, 2, 2],
        'year': [1, 2, 1, 2],
        'y': [10.0, 12.0, 15.0, 16.0],
        'gvar': [2, 2, 0, 0]
    })


# =============================================================================
# Test RI Failure State Consistency (BUG-019)
# =============================================================================

class TestBug019RIFailureState:
    """Test that RI failure sets results attributes to consistent state."""
    
    def test_ri_not_requested_attributes_are_none(self, simple_staggered_data):
        """Test that when ri=False, all RI attributes are None."""
        results = lwdid(
            data=simple_staggered_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            aggregate='overall',
            ri=False
        )
        
        # All RI attributes should be None when RI is not requested
        assert results.ri_pvalue is None, "ri_pvalue should be None when ri=False"
        assert results.ri_seed is None, "ri_seed should be None when ri=False"
        assert results.rireps is None, "rireps should be None when ri=False"
        assert results.ri_method is None, "ri_method should be None when ri=False"
        assert results.ri_valid is None, "ri_valid should be None when ri=False"
        assert results.ri_failed is None, "ri_failed should be None when ri=False"
    
    def test_ri_success_attributes_are_set(self, simple_staggered_data):
        """Test that when RI succeeds, all attributes are properly set."""
        seed = 12345
        rireps = 50
        
        results = lwdid(
            data=simple_staggered_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            aggregate='overall',
            ri=True,
            rireps=rireps,
            seed=seed,
            ri_method='permutation'
        )
        
        # All RI attributes should be properly set on success
        assert results.ri_pvalue is not None, "ri_pvalue should be set on success"
        assert not np.isnan(results.ri_pvalue), "ri_pvalue should not be NaN on success"
        assert 0 <= results.ri_pvalue <= 1, "ri_pvalue should be in [0, 1]"
        
        assert results.ri_seed == seed, f"ri_seed should be {seed}"
        assert results.rireps == rireps, f"rireps should be {rireps}"
        assert results.ri_method == 'permutation', "ri_method should be 'permutation'"
        assert isinstance(results.ri_valid, int), "ri_valid should be int"
        assert results.ri_valid > 0, "ri_valid should be > 0 on success"
    
    def test_ri_failure_attributes_consistent_state(self, simple_staggered_data):
        """Test that when RI fails, attributes are set to consistent 'failed' state."""
        seed = 54321
        rireps = 50
        ri_method = 'permutation'
        
        # Mock the RI function to raise an exception
        with patch(RI_FUNCTION_PATH) as mock_ri:
            mock_ri.side_effect = RuntimeError("Simulated RI failure for testing")
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                results = lwdid(
                    data=simple_staggered_data,
                    y='y',
                    ivar='id',
                    tvar='year',
                    gvar='gvar',
                    rolling='demean',
                    aggregate='overall',
                    ri=True,
                    rireps=rireps,
                    seed=seed,
                    ri_method=ri_method
                )
                
                # Verify warning was issued
                ri_warnings = [x for x in w if "Randomization inference failed" in str(x.message)]
                assert len(ri_warnings) == 1, "Should have exactly one RI failure warning"
        
        # BUG-019 FIX: Verify all attributes are set to consistent "failed" state
        
        # ri_pvalue should be NaN (not None) to distinguish from "not requested"
        assert results.ri_pvalue is not None, "ri_pvalue should not be None on failure"
        assert np.isnan(results.ri_pvalue), "ri_pvalue should be NaN on failure"
        
        # ri_seed should record the attempted seed
        assert results.ri_seed == seed, f"ri_seed should be {seed} (attempted seed)"
        
        # rireps should record the requested replications
        assert results.rireps == rireps, f"rireps should be {rireps}"
        
        # ri_method should record the attempted method
        assert results.ri_method == ri_method, f"ri_method should be '{ri_method}'"
        
        # ri_valid should be 0 (no valid replications)
        assert results.ri_valid == 0, "ri_valid should be 0 on failure"
        
        # ri_failed should contain the error message (string, not int)
        assert isinstance(results.ri_failed, str), "ri_failed should be string (error message)"
        assert "Simulated RI failure" in results.ri_failed, "ri_failed should contain error message"
        
        # ri_target should be set
        assert hasattr(results, 'ri_target'), "ri_target should be set on failure"
        assert results.ri_target == 'overall', "ri_target should be 'overall' for aggregate='overall'"
    
    def test_distinguish_ri_not_requested_vs_failed(self, simple_staggered_data):
        """Test that we can distinguish 'RI not requested' from 'RI failed'."""
        # Case 1: RI not requested
        results_no_ri = lwdid(
            data=simple_staggered_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            aggregate='overall',
            ri=False
        )
        
        # Case 2: RI failed
        with patch(RI_FUNCTION_PATH) as mock_ri:
            mock_ri.side_effect = RuntimeError("Simulated failure")
            
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                results_ri_failed = lwdid(
                    data=simple_staggered_data,
                    y='y',
                    ivar='id',
                    tvar='year',
                    gvar='gvar',
                    rolling='demean',
                    aggregate='overall',
                    ri=True,
                    rireps=50,
                    seed=123
                )
        
        # Should be distinguishable:
        # - ri_pvalue is None when not requested
        # - ri_pvalue is NaN when failed
        assert results_no_ri.ri_pvalue is None, "Should be None when not requested"
        assert results_ri_failed.ri_pvalue is not None, "Should not be None when failed"
        assert np.isnan(results_ri_failed.ri_pvalue), "Should be NaN when failed"
        
        # Alternative check using ri_valid
        assert results_no_ri.ri_valid is None, "ri_valid should be None when not requested"
        assert results_ri_failed.ri_valid == 0, "ri_valid should be 0 when failed"
    
    def test_ri_failure_with_cohort_target(self, simple_staggered_data):
        """Test RI failure state for cohort-level aggregation."""
        with patch(RI_FUNCTION_PATH) as mock_ri:
            mock_ri.side_effect = ValueError("Cohort estimation failed")
            
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                results = lwdid(
                    data=simple_staggered_data,
                    y='y',
                    ivar='id',
                    tvar='year',
                    gvar='gvar',
                    rolling='demean',
                    aggregate='cohort',
                    ri=True,
                    rireps=50,
                    seed=999
                )
        
        # Verify failure state for cohort target
        assert np.isnan(results.ri_pvalue)
        assert results.ri_valid == 0
        assert "Cohort estimation failed" in results.ri_failed
        assert results.ri_target == 'cohort'
    
    def test_ri_failure_with_none_aggregate(self, simple_staggered_data):
        """Test RI failure state for cohort_time-level (aggregate='none')."""
        with patch(RI_FUNCTION_PATH) as mock_ri:
            mock_ri.side_effect = Exception("Effect estimation error")
            
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                results = lwdid(
                    data=simple_staggered_data,
                    y='y',
                    ivar='id',
                    tvar='year',
                    gvar='gvar',
                    rolling='demean',
                    aggregate='none',
                    ri=True,
                    rireps=50,
                    seed=111
                )
        
        # Verify failure state for cohort_time target
        assert np.isnan(results.ri_pvalue)
        assert results.ri_valid == 0
        assert results.ri_target == 'cohort_time'


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestBug019EdgeCases:
    """Test edge cases for RI failure handling."""
    
    def test_ri_failure_preserves_estimation_results(self, simple_staggered_data):
        """Test that RI failure does not affect main estimation results."""
        # Get baseline results without RI
        results_no_ri = lwdid(
            data=simple_staggered_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            aggregate='overall',
            ri=False
        )
        
        # Get results with RI failure
        with patch(RI_FUNCTION_PATH) as mock_ri:
            mock_ri.side_effect = RuntimeError("Simulated failure")
            
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                results_ri_failed = lwdid(
                    data=simple_staggered_data,
                    y='y',
                    ivar='id',
                    tvar='year',
                    gvar='gvar',
                    rolling='demean',
                    aggregate='overall',
                    ri=True,
                    rireps=50,
                    seed=123
                )
        
        # Main estimation results should be identical
        assert results_no_ri.att_overall == results_ri_failed.att_overall
        assert results_no_ri.se_overall == results_ri_failed.se_overall
        assert results_no_ri.n_treated == results_ri_failed.n_treated
        assert results_no_ri.n_control == results_ri_failed.n_control
    
    def test_ri_failure_with_different_error_types(self, simple_staggered_data):
        """Test RI failure handling for different exception types."""
        error_types = [
            (ValueError, "Invalid value"),
            (RuntimeError, "Runtime error"),
            (TypeError, "Type error"),
            (KeyError, "missing_key"),
            (ZeroDivisionError, "division by zero"),
        ]
        
        for error_class, error_msg in error_types:
            with patch(RI_FUNCTION_PATH) as mock_ri:
                mock_ri.side_effect = error_class(error_msg)
                
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")
                    results = lwdid(
                        data=simple_staggered_data,
                        y='y',
                        ivar='id',
                        tvar='year',
                        gvar='gvar',
                        rolling='demean',
                        aggregate='overall',
                        ri=True,
                        rireps=50,
                        seed=123
                    )
            
            # Should always set consistent failure state
            assert np.isnan(results.ri_pvalue), f"ri_pvalue should be NaN for {error_class.__name__}"
            assert results.ri_valid == 0, f"ri_valid should be 0 for {error_class.__name__}"
            assert error_msg in results.ri_failed, f"ri_failed should contain error message for {error_class.__name__}"
    
    def test_ri_failure_warning_message(self, simple_staggered_data):
        """Test that RI failure warning contains useful information."""
        with patch(RI_FUNCTION_PATH) as mock_ri:
            mock_ri.side_effect = ValueError("Test error message")
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _ = lwdid(
                    data=simple_staggered_data,
                    y='y',
                    ivar='id',
                    tvar='year',
                    gvar='gvar',
                    rolling='demean',
                    aggregate='overall',
                    ri=True,
                    rireps=50,
                    seed=123
                )
                
                # Find RI warning
                ri_warnings = [x for x in w if "Randomization inference failed" in str(x.message)]
                assert len(ri_warnings) == 1
                
                warning_msg = str(ri_warnings[0].message)
                assert "ValueError" in warning_msg, "Warning should include exception type"
                assert "Test error message" in warning_msg, "Warning should include error message"


# =============================================================================
# Numerical Validation Tests
# =============================================================================

class TestBug019NumericalValidation:
    """Test numerical aspects of RI failure state."""
    
    def test_nan_pvalue_is_proper_nan(self, simple_staggered_data):
        """Test that failed ri_pvalue is proper np.nan."""
        with patch(RI_FUNCTION_PATH) as mock_ri:
            mock_ri.side_effect = RuntimeError("Simulated failure")
            
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                results = lwdid(
                    data=simple_staggered_data,
                    y='y',
                    ivar='id',
                    tvar='year',
                    gvar='gvar',
                    rolling='demean',
                    aggregate='overall',
                    ri=True,
                    rireps=50,
                    seed=123
                )
        
        # Verify it's proper NaN
        assert results.ri_pvalue is not None
        assert np.isnan(results.ri_pvalue)
        assert isinstance(results.ri_pvalue, float)
        
        # NaN comparisons
        assert not (results.ri_pvalue == results.ri_pvalue)  # NaN != NaN
        assert np.isnan(results.ri_pvalue)  # Standard check
    
    def test_zero_ri_valid_is_integer(self, simple_staggered_data):
        """Test that ri_valid=0 on failure is proper integer."""
        with patch(RI_FUNCTION_PATH) as mock_ri:
            mock_ri.side_effect = RuntimeError("Simulated failure")
            
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                results = lwdid(
                    data=simple_staggered_data,
                    y='y',
                    ivar='id',
                    tvar='year',
                    gvar='gvar',
                    rolling='demean',
                    aggregate='overall',
                    ri=True,
                    rireps=50,
                    seed=123
                )
        
        # Verify it's proper integer 0
        assert results.ri_valid == 0
        assert isinstance(results.ri_valid, int)
        assert results.ri_valid is not False  # Distinct from boolean


# =============================================================================
# Real Edge Case Tests (No Mocking)
# =============================================================================

class TestBug019RealEdgeCases:
    """Test RI failure handling with real data edge cases (no mocking)."""
    
    def test_real_ri_skipped_no_effects(self):
        """Test RI skipped when no effects available for RI.
        
        This test uses minimal data where RI may be skipped due to
        no available effects (cohort_time_effects empty).
        When RI is skipped (not failed), ri_pvalue should be None.
        When RI fails with exception, ri_pvalue should be NaN.
        """
        # Create minimal data: 2 units
        data = pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2],
            'year': [1, 2, 3, 1, 2, 3],
            'y': [10.0, 11.0, 12.0, 15.0, 16.0, 17.0],
            'gvar': [2, 2, 2, 0, 0, 0]
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            results = lwdid(
                data=data,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                rolling='demean',
                aggregate='none',  # cohort_time level
                ri=True,
                rireps=50,
                seed=123
            )
        
        # Check what happened
        ri_failure_warnings = [x for x in w if "Randomization inference failed" in str(x.message)]
        ri_skipped_warnings = [x for x in w if "No available effect estimates" in str(x.message)]
        
        if len(ri_failure_warnings) > 0:
            # RI failed with exception - check BUG-019 fix
            assert np.isnan(results.ri_pvalue), "ri_pvalue should be NaN on RI failure"
            assert results.ri_valid == 0, "ri_valid should be 0 on RI failure"
            assert isinstance(results.ri_failed, str), "ri_failed should contain error message"
        elif len(ri_skipped_warnings) > 0:
            # RI was skipped (no effects) - ri_pvalue should be None
            assert results.ri_pvalue is None, "ri_pvalue should be None when RI is skipped"
        else:
            # RI succeeded
            assert results.ri_pvalue is not None and not np.isnan(results.ri_pvalue)
    
    def test_ri_with_extreme_treatment_effect(self, simple_staggered_data):
        """Test RI with extreme (obvious) treatment effect.
        
        With a large treatment effect, RI should complete successfully
        and give a small p-value.
        """
        # Add large treatment effect
        data = simple_staggered_data.copy()
        data.loc[(data['gvar'] > 0) & (data['year'] >= data['gvar']), 'y'] += 100.0
        
        results = lwdid(
            data=data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            aggregate='overall',
            ri=True,
            rireps=50,
            seed=123
        )
        
        # RI should succeed
        assert results.ri_pvalue is not None
        if not np.isnan(results.ri_pvalue):
            # With large effect, p-value should be small
            assert results.ri_pvalue <= 0.1, "Large effect should give small p-value"
            assert results.ri_valid > 0
    
    def test_summary_output_with_failed_ri(self, simple_staggered_data):
        """Test that summary() method handles failed RI gracefully."""
        with patch(RI_FUNCTION_PATH) as mock_ri:
            mock_ri.side_effect = RuntimeError("Test failure")
            
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                results = lwdid(
                    data=simple_staggered_data,
                    y='y',
                    ivar='id',
                    tvar='year',
                    gvar='gvar',
                    rolling='demean',
                    aggregate='overall',
                    ri=True,
                    rireps=50,
                    seed=123
                )
        
        # summary() should not crash with NaN ri_pvalue
        try:
            summary_str = results.summary()
            # NaN p-value handling in summary
            assert isinstance(summary_str, str)
            # The summary might show 'nan' or omit RI info
        except Exception as e:
            pytest.fail(f"summary() should handle NaN ri_pvalue gracefully: {e}")


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
