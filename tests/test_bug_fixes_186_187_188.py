"""
Test suite for BUG-186, BUG-187, BUG-188 fixes.

BUG-186: int(np.nan) ValueError in staggered/estimation.py error path
BUG-187: Empty DataFrame division by zero in validation.py
BUG-188: NaN observed_att produces misleading p-value in staggered/randomization.py
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from lwdid.validation import validate_and_prepare_data
from lwdid.exceptions import InsufficientDataError, RandomizationError
from lwdid.staggered.randomization import randomization_inference_staggered


class TestBUG186NaNDfResidHandling:
    """
    BUG-186: When IPW/IPWRA/PSM estimation fails, df_resid and df_inference
    may be NaN. Converting NaN to int raises ValueError.
    
    The fix handles NaN values gracefully by using a fallback value.
    """
    
    def test_nan_df_values_no_valueerror(self):
        """Verify that NaN df_resid/df_inference values don't crash int() conversion."""
        # Test the conversion logic directly
        import pandas as pd
        
        n_total = 100
        
        # Simulate est_result with NaN df values (as returned by failed estimation)
        est_result_with_nan = {
            'att': np.nan,
            'se': np.nan,
            'df_resid': np.nan,
            'df_inference': np.nan,
        }
        
        # This is the fixed logic
        df_resid_val = est_result_with_nan.get('df_resid', n_total - 2)
        df_resid_safe = int(df_resid_val) if pd.notna(df_resid_val) else n_total - 2
        
        df_inference_val = est_result_with_nan.get('df_inference', n_total - 2)
        df_inference_safe = int(df_inference_val) if pd.notna(df_inference_val) else n_total - 2
        
        # Should not raise ValueError and should use fallback
        assert df_resid_safe == n_total - 2
        assert df_inference_safe == n_total - 2
    
    def test_valid_df_values_preserved(self):
        """Verify that valid df_resid/df_inference values are correctly converted."""
        import pandas as pd
        
        n_total = 100
        
        # Simulate est_result with valid df values
        est_result_valid = {
            'att': 0.5,
            'se': 0.1,
            'df_resid': 45.0,  # float that should convert to int
            'df_inference': 50.0,
        }
        
        df_resid_val = est_result_valid.get('df_resid', n_total - 2)
        df_resid_safe = int(df_resid_val) if pd.notna(df_resid_val) else n_total - 2
        
        df_inference_val = est_result_valid.get('df_inference', n_total - 2)
        df_inference_safe = int(df_inference_val) if pd.notna(df_inference_val) else n_total - 2
        
        # Should preserve the actual values
        assert df_resid_safe == 45
        assert df_inference_safe == 50
    
    def test_missing_df_keys_use_fallback(self):
        """Verify that missing df keys use the n_total - 2 fallback."""
        import pandas as pd
        
        n_total = 100
        
        # Simulate est_result without df keys
        est_result_no_df = {
            'att': 0.5,
            'se': 0.1,
        }
        
        df_resid_val = est_result_no_df.get('df_resid', n_total - 2)
        df_resid_safe = int(df_resid_val) if pd.notna(df_resid_val) else n_total - 2
        
        # Should use fallback
        assert df_resid_safe == n_total - 2


class TestBUG187EmptyDataFrameValidation:
    """
    BUG-187: Empty DataFrame (len(data) == 0) causes division by zero
    in percentage calculations within validation functions.
    
    The fix adds early detection of empty DataFrames with clear error message.
    """
    
    def test_empty_dataframe_raises_insufficient_data_error(self):
        """Empty DataFrame should raise InsufficientDataError early."""
        # Create an empty DataFrame with the required columns
        empty_df = pd.DataFrame({
            'y': pd.Series([], dtype=float),
            'd': pd.Series([], dtype=int),
            'id': pd.Series([], dtype=int),
            'year': pd.Series([], dtype=int),
            'post': pd.Series([], dtype=int),
        })
        
        with pytest.raises(InsufficientDataError, match="no rows"):
            validate_and_prepare_data(
                data=empty_df,
                y='y',
                d='d',
                ivar='id',
                tvar='year',
                post='post',
                rolling='demean'
            )
    
    def test_empty_dataframe_error_message_informative(self):
        """Error message should guide users on how to diagnose the issue."""
        empty_df = pd.DataFrame({
            'y': pd.Series([], dtype=float),
            'd': pd.Series([], dtype=int),
            'id': pd.Series([], dtype=int),
            'year': pd.Series([], dtype=int),
            'post': pd.Series([], dtype=int),
        })
        
        with pytest.raises(InsufficientDataError) as exc_info:
            validate_and_prepare_data(
                data=empty_df,
                y='y',
                d='d',
                ivar='id',
                tvar='year',
                post='post',
                rolling='demean'
            )
        
        # Check that error message contains helpful guidance
        error_msg = str(exc_info.value)
        assert "no rows" in error_msg.lower()
        assert "How to fix" in error_msg
    
    def test_single_row_dataframe_not_blocked_by_empty_check(self):
        """Single-row DataFrame should pass empty check but may fail later validation."""
        single_row_df = pd.DataFrame({
            'y': [1.0],
            'd': [1],
            'id': [1],
            'year': [2000],
            'post': [1],
        })
        
        # Should NOT raise InsufficientDataError from empty check
        # (will fail later on other validation like no pre-treatment obs or insufficient units)
        with pytest.raises(InsufficientDataError):
            validate_and_prepare_data(
                data=single_row_df,
                y='y',
                d='d',
                ivar='id',
                tvar='year',
                post='post',
                rolling='demean'
            )


class TestBUG188NaNObservedATTValidation:
    """
    BUG-188: When observed_att is NaN, the comparison 
    np.abs(valid_stats) >= abs(np.nan) returns all False,
    producing misleadingly small p-values (close to 0).
    
    The fix validates observed_att before p-value computation.
    """
    
    def test_nan_observed_att_raises_randomization_error(self):
        """NaN observed_att should raise RandomizationError with clear message."""
        # Create minimal valid staggered data with proper structure
        np.random.seed(42)
        n_treated = 10
        n_control = 10
        n_periods = 6
        
        # Build data with explicit treated and never-treated units
        ids = []
        ts = []
        ys = []
        gvars = []
        
        # Treated units (gvar=3, treated at period 3)
        for i in range(1, n_treated + 1):
            for t in range(1, n_periods + 1):
                ids.append(i)
                ts.append(t)
                ys.append(np.random.randn())
                gvars.append(3)  # Treated at t=3
        
        # Never-treated units (gvar=0)
        for i in range(n_treated + 1, n_treated + n_control + 1):
            for t in range(1, n_periods + 1):
                ids.append(i)
                ts.append(t)
                ys.append(np.random.randn())
                gvars.append(0)  # Never treated
        
        data = pd.DataFrame({
            'id': ids,
            't': ts,
            'y': ys,
            'gvar': gvars
        })
        
        # Call randomization inference with NaN observed_att
        with pytest.raises(RandomizationError, match="NaN"):
            randomization_inference_staggered(
                data=data,
                gvar='gvar',
                ivar='id',
                tvar='t',
                y='y',
                observed_att=np.nan,  # Explicitly pass NaN
                target='overall',
                rolling='demean',
                ri_method='permutation',
                rireps=100,
                seed=42,
                n_never_treated=n_control
            )
    
    def test_nan_observed_att_error_message_informative(self):
        """Error message should explain the issue and suggest fixes."""
        np.random.seed(42)
        n_treated = 10
        n_control = 10
        n_periods = 6
        
        # Build proper staggered data
        ids = []
        ts = []
        ys = []
        gvars = []
        
        for i in range(1, n_treated + 1):
            for t in range(1, n_periods + 1):
                ids.append(i)
                ts.append(t)
                ys.append(np.random.randn())
                gvars.append(3)
        
        for i in range(n_treated + 1, n_treated + n_control + 1):
            for t in range(1, n_periods + 1):
                ids.append(i)
                ts.append(t)
                ys.append(np.random.randn())
                gvars.append(0)
        
        data = pd.DataFrame({
            'id': ids,
            't': ts,
            'y': ys,
            'gvar': gvars
        })
        
        with pytest.raises(RandomizationError) as exc_info:
            randomization_inference_staggered(
                data=data,
                gvar='gvar',
                ivar='id',
                tvar='t',
                y='y',
                observed_att=np.nan,
                target='overall',
                rolling='demean',
                ri_method='permutation',
                rireps=100,
                seed=42,
                n_never_treated=n_control
            )
        
        error_msg = str(exc_info.value)
        assert "NaN" in error_msg
        assert "How to fix" in error_msg
    
    def test_valid_observed_att_computes_pvalue(self):
        """Valid observed_att should produce valid p-value."""
        np.random.seed(42)
        n_treated = 10
        n_control = 10
        n_periods = 6
        
        # Build proper staggered data
        ids = []
        ts = []
        ys = []
        gvars = []
        
        for i in range(1, n_treated + 1):
            for t in range(1, n_periods + 1):
                ids.append(i)
                ts.append(t)
                ys.append(np.random.randn())
                gvars.append(3)
        
        for i in range(n_treated + 1, n_treated + n_control + 1):
            for t in range(1, n_periods + 1):
                ids.append(i)
                ts.append(t)
                ys.append(np.random.randn())
                gvars.append(0)
        
        data = pd.DataFrame({
            'id': ids,
            't': ts,
            'y': ys,
            'gvar': gvars
        })
        
        # Should not raise with valid observed_att
        result = randomization_inference_staggered(
            data=data,
            gvar='gvar',
            ivar='id',
            tvar='t',
            y='y',
            observed_att=0.5,  # Valid numeric value
            target='overall',
            rolling='demean',
            ri_method='permutation',
            rireps=100,
            seed=42,
            n_never_treated=n_control
        )
        
        # p-value should be between 0 and 1
        assert 0 <= result.p_value <= 1
        assert not np.isnan(result.p_value)


class TestBUG186187188Integration:
    """Integration tests verifying all three fixes work together."""
    
    def test_normal_data_still_works(self):
        """Verify that normal data processing is unaffected by the fixes."""
        # Create valid panel data with sufficient units and periods
        np.random.seed(123)
        data = pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            'year': [2000, 2001, 2002] * 4,
            'y': np.random.randn(12),
            'd': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            'post': [0, 1, 1] * 4,
        })
        
        # Should pass validation without errors
        # validate_and_prepare_data returns (data_work, metadata) tuple
        data_work, metadata = validate_and_prepare_data(
            data=data,
            y='y',
            d='d',
            ivar='id',
            tvar='year',
            post='post',
            rolling='demean'
        )
        
        assert data_work is not None
        assert len(data_work) > 0
        assert metadata is not None
        assert 'N' in metadata
        assert metadata['N'] == 4  # 4 units


class TestBUG186NaNComparisonBehavior:
    """Verify the numerical behavior that caused the original bug."""
    
    def test_nan_comparison_always_false(self):
        """Document that NaN comparisons return False (root cause of BUG-188)."""
        # This demonstrates why BUG-188 was producing misleading p-values
        valid_stats = np.array([1.0, 2.0, 3.0])
        nan_att = np.nan
        
        # All comparisons with NaN return False
        comparison = np.abs(valid_stats) >= abs(nan_att)
        assert not comparison.any()  # All False
        
        # This would have given n_extreme = 0
        n_extreme = comparison.sum()
        assert n_extreme == 0
        
        # Leading to p ≈ 0 (or 1/(n+1) with +1 correction)
        # which is misleadingly small for an undefined statistic
    
    def test_int_nan_raises_valueerror(self):
        """Document that int(np.nan) raises ValueError (root cause of BUG-186)."""
        with pytest.raises(ValueError, match="cannot convert"):
            int(np.nan)
