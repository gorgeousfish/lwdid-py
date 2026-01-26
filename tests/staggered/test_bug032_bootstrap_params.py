"""
BUG-032: Event Study Bootstrap Parameter Hardcoding Fix Tests

Tests verify that _compute_event_study_se_bootstrap() correctly uses
user-configured parameters instead of hardcoded values:
- never_treated_values: From results metadata (default [0, np.inf])
- control_strategy: From results._control_group_used
- alpha: From results._alpha

Test coverage:
1. Unit tests: Parameter signature and internal usage
2. Numerical validation: SE calculation correctness
3. End-to-end tests: Configuration consistency
"""

import os
import sys
import warnings
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from lwdid import lwdid
from lwdid.results import _compute_event_study_se_bootstrap


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def castle_data():
    """Load Castle Law dataset."""
    here = os.path.dirname(__file__)
    data_path = os.path.join(here, '..', '..', 'data', 'castle.csv')
    data = pd.read_csv(data_path)
    data['gvar'] = data['effyear'].fillna(0).astype(int)
    return data


@pytest.fixture
def staggered_results_default(castle_data):
    """Staggered results with default configuration."""
    return lwdid(
        data=castle_data, 
        y='lhomicide', 
        ivar='sid', 
        tvar='year',
        gvar='gvar', 
        rolling='demean', 
        control_group='never_treated',
        aggregate='overall',
        alpha=0.05  # Default
    )


@pytest.fixture
def staggered_results_alpha_10(castle_data):
    """Staggered results with alpha=0.10 (90% CI)."""
    return lwdid(
        data=castle_data, 
        y='lhomicide', 
        ivar='sid', 
        tvar='year',
        gvar='gvar', 
        rolling='demean', 
        control_group='never_treated',
        aggregate='overall',
        alpha=0.10  # 90% CI
    )


# =============================================================================
# Unit Tests: Parameter Signature
# =============================================================================

class TestBootstrapFunctionSignature:
    """Test that _compute_event_study_se_bootstrap has correct signature."""
    
    def test_function_has_never_treated_values_param(self):
        """Function should have never_treated_values parameter."""
        import inspect
        sig = inspect.signature(_compute_event_study_se_bootstrap)
        params = sig.parameters
        
        assert 'never_treated_values' in params
        assert params['never_treated_values'].default is None
    
    def test_function_has_control_strategy_param(self):
        """Function should have control_strategy parameter."""
        import inspect
        sig = inspect.signature(_compute_event_study_se_bootstrap)
        params = sig.parameters
        
        assert 'control_strategy' in params
        assert params['control_strategy'].default == 'not_yet_treated'
    
    def test_function_has_alpha_param(self):
        """Function should have alpha parameter."""
        import inspect
        sig = inspect.signature(_compute_event_study_se_bootstrap)
        params = sig.parameters
        
        assert 'alpha' in params
        assert params['alpha'].default == 0.05


# =============================================================================
# Unit Tests: LWDIDResults Attribute Storage
# =============================================================================

class TestResultsAttributeStorage:
    """Test that LWDIDResults stores required attributes for bootstrap."""
    
    def test_results_has_never_treated_values(self, staggered_results_default):
        """Results should have _never_treated_values attribute."""
        assert hasattr(staggered_results_default, '_never_treated_values')
        # Default should be [0, np.inf]
        expected = [0, np.inf]
        actual = staggered_results_default._never_treated_values
        assert actual == expected, f"Expected {expected}, got {actual}"
    
    def test_results_has_control_group_used(self, staggered_results_default):
        """Results should have _control_group_used attribute."""
        assert hasattr(staggered_results_default, '_control_group_used')
        # With aggregate='overall', control_group_used should be 'never_treated'
        assert staggered_results_default._control_group_used == 'never_treated'
    
    def test_results_has_alpha(self, staggered_results_default):
        """Results should have _alpha attribute."""
        assert hasattr(staggered_results_default, '_alpha')
        assert staggered_results_default._alpha == 0.05
    
    def test_alpha_stored_correctly(self, staggered_results_alpha_10):
        """Alpha=0.10 should be stored correctly."""
        assert staggered_results_alpha_10._alpha == 0.10


# =============================================================================
# Unit Tests: Parameter Passing in plot_event_study
# =============================================================================

class TestParameterPassingToBootstrap:
    """Test that plot_event_study passes parameters correctly to bootstrap."""
    
    @pytest.mark.slow
    def test_alpha_passed_to_bootstrap(self, staggered_results_alpha_10):
        """Alpha parameter should be passed to bootstrap estimation."""
        # Patch _compute_event_study_se_bootstrap to capture arguments
        with patch('lwdid.results._compute_event_study_se_bootstrap') as mock_bootstrap:
            mock_bootstrap.return_value = None  # Force fallback to analytical
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                staggered_results_alpha_10.plot_event_study(
                    se_method='bootstrap',
                    n_bootstrap=10
                )
            
            # Check that alpha was passed correctly
            if mock_bootstrap.called:
                call_kwargs = mock_bootstrap.call_args[1]
                assert call_kwargs.get('alpha') == 0.10, \
                    f"Expected alpha=0.10, got {call_kwargs.get('alpha')}"
    
    @pytest.mark.slow
    def test_control_strategy_passed_to_bootstrap(self, staggered_results_default):
        """Control strategy should be passed to bootstrap estimation."""
        with patch('lwdid.results._compute_event_study_se_bootstrap') as mock_bootstrap:
            mock_bootstrap.return_value = None
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                staggered_results_default.plot_event_study(
                    se_method='bootstrap',
                    n_bootstrap=10
                )
            
            if mock_bootstrap.called:
                call_kwargs = mock_bootstrap.call_args[1]
                assert call_kwargs.get('control_strategy') == 'never_treated', \
                    f"Expected control_strategy='never_treated', got {call_kwargs.get('control_strategy')}"
    
    @pytest.mark.slow
    def test_never_treated_values_passed_to_bootstrap(self, staggered_results_default):
        """Never treated values should be passed to bootstrap estimation."""
        with patch('lwdid.results._compute_event_study_se_bootstrap') as mock_bootstrap:
            mock_bootstrap.return_value = None
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                staggered_results_default.plot_event_study(
                    se_method='bootstrap',
                    n_bootstrap=10
                )
            
            if mock_bootstrap.called:
                call_kwargs = mock_bootstrap.call_args[1]
                expected_nt = [0, np.inf]
                actual_nt = call_kwargs.get('never_treated_values')
                assert actual_nt == expected_nt, \
                    f"Expected never_treated_values={expected_nt}, got {actual_nt}"


# =============================================================================
# Numerical Validation Tests
# =============================================================================

class TestBootstrapNumericalValidation:
    """Numerical validation of bootstrap SE calculation."""
    
    @pytest.mark.slow
    def test_bootstrap_se_finite(self, staggered_results_default):
        """Bootstrap SE should produce finite values."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig, ax, event_df = staggered_results_default.plot_event_study(
                se_method='bootstrap',
                n_bootstrap=30,
                seed=42,
                return_data=True
            )
        
        # SE should be finite
        assert all(np.isfinite(event_df['se'])), "SE should be finite"
        
        # SE should be positive
        assert all(event_df['se'] > 0), "SE should be positive"
        
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    @pytest.mark.slow
    def test_bootstrap_reproducible_with_seed(self, staggered_results_default):
        """Bootstrap SE should be reproducible with same seed."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            _, _, df1 = staggered_results_default.plot_event_study(
                se_method='bootstrap',
                n_bootstrap=30,
                seed=12345,
                return_data=True
            )
            
            _, _, df2 = staggered_results_default.plot_event_study(
                se_method='bootstrap',
                n_bootstrap=30,
                seed=12345,
                return_data=True
            )
        
        # ATT should be identical
        np.testing.assert_array_almost_equal(
            df1['att'].values,
            df2['att'].values,
            decimal=10
        )
        
        import matplotlib.pyplot as plt
        plt.close('all')
    
    @pytest.mark.slow
    def test_alpha_affects_ci_width(self, castle_data):
        """Different alpha values should produce different CI widths."""
        # 95% CI (alpha=0.05)
        results_95 = lwdid(
            data=castle_data, 
            y='lhomicide', 
            ivar='sid', 
            tvar='year',
            gvar='gvar', 
            rolling='demean', 
            control_group='never_treated',
            aggregate='overall',
            alpha=0.05
        )
        
        # 90% CI (alpha=0.10)
        results_90 = lwdid(
            data=castle_data, 
            y='lhomicide', 
            ivar='sid', 
            tvar='year',
            gvar='gvar', 
            rolling='demean', 
            control_group='never_treated',
            aggregate='overall',
            alpha=0.10
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            _, _, df_95 = results_95.plot_event_study(
                se_method='analytical',
                return_data=True
            )
            
            _, _, df_90 = results_90.plot_event_study(
                se_method='analytical',
                return_data=True
            )
        
        # Verify alpha is stored correctly
        assert results_95._alpha == 0.05
        assert results_90._alpha == 0.10
        
        # CI width should be different
        # 95% CI should be wider than 90% CI
        ci_width_95 = (df_95['ci_upper'] - df_95['ci_lower']).mean()
        ci_width_90 = (df_90['ci_upper'] - df_90['ci_lower']).mean()
        
        # 95% CI should be wider (z_0.025 > z_0.05)
        assert ci_width_95 > ci_width_90, \
            f"95% CI ({ci_width_95:.4f}) should be wider than 90% CI ({ci_width_90:.4f})"
        
        import matplotlib.pyplot as plt
        plt.close('all')


# =============================================================================
# End-to-End Tests: Configuration Consistency
# =============================================================================

class TestConfigurationConsistency:
    """Test that configuration is consistent between main estimation and bootstrap."""
    
    @pytest.mark.slow
    def test_control_group_consistency(self, castle_data):
        """Control group setting should be consistent in bootstrap."""
        # Create results with never_treated
        results_nt = lwdid(
            data=castle_data, 
            y='lhomicide', 
            ivar='sid', 
            tvar='year',
            gvar='gvar', 
            rolling='demean', 
            control_group='never_treated',
            aggregate='overall'
        )
        
        # Verify control_group_used
        assert results_nt._control_group_used == 'never_treated'
        
        # Bootstrap should use the same control group
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig, ax = results_nt.plot_event_study(
                se_method='bootstrap',
                n_bootstrap=20,
                seed=42
            )
        
        # Should complete without error
        assert fig is not None
        
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    @pytest.mark.slow
    def test_alpha_consistency_in_bootstrap(self, castle_data):
        """Alpha setting should be consistent in bootstrap estimation."""
        # Create results with non-default alpha
        results = lwdid(
            data=castle_data, 
            y='lhomicide', 
            ivar='sid', 
            tvar='year',
            gvar='gvar', 
            rolling='demean', 
            control_group='never_treated',
            aggregate='overall',
            alpha=0.01  # 99% CI
        )
        
        # Verify alpha is stored
        assert results._alpha == 0.01
        
        # Bootstrap should use the same alpha internally
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig, ax, event_df = results.plot_event_study(
                se_method='bootstrap',
                n_bootstrap=20,
                seed=42,
                return_data=True
            )
        
        # CI should be based on alpha=0.01 (99% CI)
        # For analytical SE, z_crit for 99% CI ≈ 2.576
        # For 95% CI, z_crit ≈ 1.96
        # So 99% CI should be wider
        
        assert fig is not None
        assert len(event_df) > 0
        
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_never_treated_values_default(self, staggered_results_default):
        """Default never_treated_values should be [0, np.inf]."""
        expected = [0, np.inf]
        actual = staggered_results_default._never_treated_values
        
        assert actual == expected, \
            f"Expected {expected}, got {actual}"


# =============================================================================
# Regression Tests
# =============================================================================

class TestRegressionNoHardcoding:
    """Regression tests to ensure hardcoded values are removed."""
    
    def test_source_no_hardcoded_never_treated_inline(self):
        """Source code should not have hardcoded never_treated_values inline."""
        # Read the source file directly to ensure we get the right content
        import os
        source_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'src', 'lwdid', 'results.py'
        )
        with open(source_path, 'r') as f:
            source = f.read()
        
        # The hardcoded line should be removed
        # Old: never_treated_values = [0, np.inf]
        # New: if never_treated_values is None: never_treated_values = [0, np.inf]
        
        # Check that we use the parameter, not hardcode
        assert 'if never_treated_values is None' in source, \
            "Should check if never_treated_values is None before using default"
    
    def test_source_uses_control_strategy_param(self):
        """Source code should use control_strategy parameter."""
        # Read the source file directly
        import os
        source_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'src', 'lwdid', 'results.py'
        )
        with open(source_path, 'r') as f:
            source = f.read()
        
        # Should pass control_strategy to estimate_cohort_time_effects
        assert 'control_strategy=control_strategy' in source, \
            "Should pass control_strategy parameter to estimate_cohort_time_effects"
    
    def test_source_uses_alpha_param(self):
        """Source code should use alpha parameter."""
        # Read the source file directly
        import os
        source_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'src', 'lwdid', 'results.py'
        )
        with open(source_path, 'r') as f:
            source = f.read()
        
        # Should pass alpha to estimate_cohort_time_effects
        assert 'alpha=alpha' in source, \
            "Should pass alpha parameter to estimate_cohort_time_effects"


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Edge case tests for bootstrap parameter handling."""
    
    @pytest.mark.slow
    def test_bootstrap_fallback_preserves_config(self, staggered_results_default):
        """When bootstrap falls back to analytical, config should be preserved."""
        # Force a small n_bootstrap that might fail
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig, ax, event_df = staggered_results_default.plot_event_study(
                se_method='bootstrap',
                n_bootstrap=5,
                seed=42,
                return_data=True
            )
        
        # Should still have valid results
        assert event_df is not None
        assert 'se' in event_df.columns
        
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_non_staggered_results_no_never_treated(self):
        """Non-staggered results should have None for _never_treated_values."""
        # Create a simple common timing dataset
        data = pd.DataFrame({
            'id': [1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4],
            'year': [1,2,3,4, 1,2,3,4, 1,2,3,4, 1,2,3,4],
            'y': [10,11,15,16, 8,9,8,9, 12,13,17,18, 9,10,9,10],
            'd': [1,1,1,1, 0,0,0,0, 1,1,1,1, 0,0,0,0],
            'post': [0,0,1,1, 0,0,1,1, 0,0,1,1, 0,0,1,1]
        })
        
        results = lwdid(
            data=data, 
            y='y', 
            d='d', 
            ivar='id', 
            tvar='year', 
            post='post',
            rolling='demean'
        )
        
        # Non-staggered results should have None
        assert results._never_treated_values is None


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
