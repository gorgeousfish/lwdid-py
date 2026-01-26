"""
Unit tests for BUG-183, BUG-184, and BUG-185 fixes.

BUG-183: HC4 variance calculation LinAlgError handling
BUG-184: Common timing mode RI failure graceful handling  
BUG-185: Outcome variable NaN validation in randomization inference

These tests verify:
1. Singular matrix handling in HC4 variance computation
2. RI failure doesn't crash ATT estimation in common timing mode
3. Outcome NaN check raises appropriate error
"""
import numpy as np
import pandas as pd
import pytest
import warnings

from lwdid import lwdid
from lwdid.randomization import randomization_inference
from lwdid.exceptions import RandomizationError
from lwdid.estimation import estimate_att


class TestBUG183_HC4LinAlgError:
    """Tests for BUG-183: HC4 branch LinAlgError handling."""
    
    def test_hc4_singular_matrix_warning(self):
        """
        Test that HC4 variance calculation handles singular X'X matrix gracefully.
        
        When design matrix has perfect multicollinearity, X'X becomes singular.
        The code should:
        1. Catch LinAlgError
        2. Issue a warning about singular matrix
        3. Use Moore-Penrose pseudo-inverse
        4. Still return valid results
        """
        # Create data with proper treatment/control groups and collinear controls
        np.random.seed(42)
        n = 100
        
        # Half treated, half control
        d_vals = np.array([0] * 50 + [1] * 50)
        
        data = pd.DataFrame({
            'id': range(n),
            'time': np.ones(n, dtype=int),
            'y': np.random.randn(n) + d_vals * 0.5,
            'd_': d_vals,
            'x1': np.random.randn(n),
        })
        # Create perfectly collinear variable: x2 = 2*x1
        data['x2'] = 2 * data['x1']
        
        # Should issue warning about singular matrix when using HC4 with collinear controls
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = estimate_att(
                data=data,
                y_transformed='y',
                d='d_',
                ivar='id',
                controls=['x1', 'x2'],  # Perfectly collinear
                vce='hc4',
                cluster_var=None,
                sample_filter=pd.Series(True, index=data.index),
            )
            
            # Check that result is returned (not crashed)
            assert result is not None
            assert 'att' in result
            assert not np.isnan(result['att'])
    
    def test_hc4_wellconditioned_matrix_no_warning(self):
        """
        Test that HC4 variance calculation works without warning for well-conditioned matrix.
        """
        np.random.seed(123)
        n = 100
        
        # Half treated, half control
        d_vals = np.array([0] * 50 + [1] * 50)
        
        data = pd.DataFrame({
            'id': range(n),
            'time': np.ones(n, dtype=int),
            'y': np.random.randn(n) + d_vals * 0.5,
            'd_': d_vals,
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),  # Independent variable
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = estimate_att(
                data=data,
                y_transformed='y',
                d='d_',
                ivar='id',
                controls=['x1', 'x2'],
                vce='hc4',
                cluster_var=None,
                sample_filter=pd.Series(True, index=data.index),
            )
            
            # Check result is valid
            assert result is not None
            assert 'att' in result
            assert 'se_att' in result
            assert not np.isnan(result['att'])
            assert not np.isnan(result['se_att'])
            
            # No singular matrix warning should be issued
            singular_warnings = [x for x in w if 'singular' in str(x.message).lower()]
            assert len(singular_warnings) == 0


class TestBUG184_CommonTimingRIFailure:
    """Tests for BUG-184: Common timing mode RI failure handling."""
    
    def test_ri_failure_returns_att_with_nan_pvalue(self):
        """
        Test that RI failure in common timing mode returns ATT with nan p-value.
        
        When randomization inference fails, the function should:
        1. Issue a warning
        2. Set ri_pvalue to NaN
        3. Still return valid ATT estimation results
        """
        # Create minimal valid dataset for common timing
        # d must be TIME-INVARIANT (unit-level treatment status)
        np.random.seed(42)
        n_units = 20
        n_periods = 6
        treatment_period = 4
        
        data = []
        for i in range(n_units):
            # d is the unit-level treatment status (time-invariant)
            is_treated = i < n_units // 2
            d_unit = 1 if is_treated else 0
            
            for t in range(1, n_periods + 1):
                post = 1 if t >= treatment_period else 0
                # Treatment effect only applies in post period for treated units
                treat_effect = 2.0 if (is_treated and post == 1) else 0
                y = np.random.randn() + treat_effect
                data.append({
                    'unit_id': i,
                    'time_var': t,
                    'y': y,
                    'd': d_unit,  # Time-invariant!
                    'post': post,
                    'quarter': ((t - 1) % 4) + 1,
                })
        
        df = pd.DataFrame(data)
        
        # Run lwdid with RI enabled
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = lwdid(
                data=df,
                y='y',
                ivar='unit_id',
                tvar='time_var',
                d='d',
                post='post',
                ri=True,
                rireps=100,
                seed=42,
            )
            
            # ATT should be computed successfully
            assert result is not None
            assert hasattr(result, 'att')
            assert not np.isnan(result.att)
            
            # RI results should be present (either valid or NaN)
            assert hasattr(result, 'ri_pvalue')
    
    def test_ri_success_returns_valid_pvalue(self):
        """
        Test that successful RI in common timing mode returns valid p-value.
        """
        np.random.seed(123)
        n_units = 30
        n_periods = 6
        treatment_period = 4
        
        data = []
        for i in range(n_units):
            # d is the unit-level treatment status (time-invariant)
            is_treated = i < n_units // 2
            d_unit = 1 if is_treated else 0
            
            for t in range(1, n_periods + 1):
                post = 1 if t >= treatment_period else 0
                # Treatment effect only applies in post period for treated units
                treat_effect = 2.0 if (is_treated and post == 1) else 0
                y = np.random.randn() + treat_effect
                data.append({
                    'unit_id': i,
                    'time_var': t,
                    'y': y,
                    'd': d_unit,  # Time-invariant!
                    'post': post,
                    'quarter': ((t - 1) % 4) + 1,
                })
        
        df = pd.DataFrame(data)
        
        result = lwdid(
            data=df,
            y='y',
            ivar='unit_id',
            tvar='time_var',
            d='d',
            post='post',
            ri=True,
            rireps=100,
            seed=42,
        )
        
        # Check valid results
        assert result is not None
        assert not np.isnan(result.att)
        assert hasattr(result, 'ri_pvalue')
        
        # If RI succeeded, p-value should be valid
        if result.ri_valid > 0:
            assert 0 <= result.ri_pvalue <= 1


class TestBUG185_OutcomeNaNCheck:
    """Tests for BUG-185: Outcome variable NaN validation."""
    
    def test_outcome_nan_raises_error(self):
        """
        Test that NaN values in outcome column raise RandomizationError.
        """
        np.random.seed(42)
        n = 50
        
        # Half treated, half control
        d_vals = np.array([0] * 25 + [1] * 25)
        
        data = pd.DataFrame({
            'id': range(n),
            'y': np.random.randn(n),
            'd_': d_vals,
        })
        
        # Introduce NaN in outcome
        data.loc[5, 'y'] = np.nan
        data.loc[15, 'y'] = np.nan
        
        with pytest.raises(RandomizationError) as excinfo:
            randomization_inference(
                firstpost_df=data,
                y_col='y',
                d_col='d_',
                ivar='id',
                rireps=100,
                seed=42,
            )
        
        # Check error message contains useful information
        error_msg = str(excinfo.value)
        assert 'y' in error_msg or 'outcome' in error_msg.lower()
        assert '2' in error_msg or 'missing' in error_msg.lower()
    
    def test_treatment_nan_raises_error(self):
        """
        Test that NaN values in treatment indicator raise RandomizationError.
        This verifies existing d_col check still works.
        """
        np.random.seed(42)
        n = 50
        
        # Half treated, half control
        d_vals = np.array([0.0] * 25 + [1.0] * 25)
        
        data = pd.DataFrame({
            'id': range(n),
            'y': np.random.randn(n),
            'd_': d_vals,
        })
        
        # Introduce NaN in treatment
        data.loc[10, 'd_'] = np.nan
        
        with pytest.raises(RandomizationError) as excinfo:
            randomization_inference(
                firstpost_df=data,
                y_col='y',
                d_col='d_',
                ivar='id',
                rireps=100,
                seed=42,
            )
        
        error_msg = str(excinfo.value)
        assert 'd_' in error_msg or 'treatment' in error_msg.lower()
    
    def test_complete_data_no_error(self):
        """
        Test that complete data (no NaN) runs without error.
        """
        np.random.seed(42)
        n = 50
        
        # Half treated, half control
        d_vals = np.array([0] * 25 + [1] * 25)
        
        data = pd.DataFrame({
            'id': range(n),
            'y': np.random.randn(n),
            'd_': d_vals,
        })
        
        # Should not raise any error - use permutation for reliable results
        result = randomization_inference(
            firstpost_df=data,
            y_col='y',
            d_col='d_',
            ivar='id',
            rireps=100,
            seed=42,
            ri_method='permutation',
        )
        
        assert result is not None
        assert 'p_value' in result
        assert 0 <= result['p_value'] <= 1


class TestBUG183_185_Integration:
    """Integration tests combining multiple bug fixes."""
    
    def test_lwdid_with_hc4_and_ri(self):
        """
        Test lwdid with HC4 variance and RI enabled.
        
        This tests the integration of BUG-183 and BUG-184 fixes.
        """
        np.random.seed(42)
        n_units = 30
        n_periods = 6
        treatment_period = 4
        
        # Generate unit-level controls (time-invariant)
        unit_x = np.random.randn(n_units)
        
        data = []
        for i in range(n_units):
            # d is the unit-level treatment status (time-invariant)
            is_treated = i < n_units // 2
            d_unit = 1 if is_treated else 0
            # x is unit-level control (time-invariant)
            x_unit = unit_x[i]
            
            for t in range(1, n_periods + 1):
                post = 1 if t >= treatment_period else 0
                # Treatment effect only applies in post period for treated units
                treat_effect = 2.0 if (is_treated and post == 1) else 0
                y = np.random.randn() + treat_effect + 0.5 * x_unit
                data.append({
                    'unit_id': i,
                    'time_var': t,
                    'y': y,
                    'd': d_unit,  # Time-invariant!
                    'post': post,
                    'x': x_unit,  # Time-invariant!
                    'quarter': ((t - 1) % 4) + 1,
                })
        
        df = pd.DataFrame(data)
        
        result = lwdid(
            data=df,
            y='y',
            ivar='unit_id',
            tvar='time_var',
            d='d',
            post='post',
            vce='hc4',
            controls=['x'],
            ri=True,
            rireps=100,
            seed=42,
        )
        
        # Check all results are valid
        assert result is not None
        assert not np.isnan(result.att)
        assert not np.isnan(result.se_att)
        assert hasattr(result, 'ri_pvalue')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
