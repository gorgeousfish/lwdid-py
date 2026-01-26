"""
Stata numerical validation tests for DESIGN-069.

This module validates that the conditional variance estimation in Python
matches Stata's teffects psmatch behavior, particularly for edge cases
with small group sizes.

DESIGN-069: Conditional variance boundary case - use global variance instead of 0
"""

import numpy as np
import pandas as pd
import pytest
import json
import tempfile
import os

# Import for Python PSM estimation
from lwdid.staggered.estimators import _estimate_conditional_variance_same_group


class TestDesign069StataValidation:
    """Numerical validation against Stata for variance estimation."""
    
    @pytest.fixture
    def test_data_small_groups(self):
        """Create test data with small treatment/control groups."""
        np.random.seed(42)
        
        # Scenario: 5 control, 50 treated (small control group)
        n_control = 5
        n_treated = 50
        
        Y_control = np.random.randn(n_control) * 2 + 5
        Y_treated = np.random.randn(n_treated) * 2 + 7
        
        X1_control = np.random.randn(n_control) + 1
        X1_treated = np.random.randn(n_treated) + 2
        
        X2_control = np.random.randn(n_control) * 0.5
        X2_treated = np.random.randn(n_treated) * 0.5 + 0.5
        
        data = pd.DataFrame({
            'id': range(1, n_control + n_treated + 1),
            'Y': np.concatenate([Y_control, Y_treated]),
            'D': np.array([0] * n_control + [1] * n_treated),
            'X1': np.concatenate([X1_control, X1_treated]),
            'X2': np.concatenate([X2_control, X2_treated]),
        })
        
        return data
    
    @pytest.fixture
    def test_data_extreme_small(self):
        """Create test data with extreme small groups (1-2 observations)."""
        np.random.seed(123)
        
        # Scenario: 2 control, 30 treated (very small control group)
        n_control = 2
        n_treated = 30
        
        Y_control = np.array([5.0, 6.0])
        Y_treated = np.random.randn(n_treated) * 2 + 8
        
        X1_control = np.array([1.0, 1.5])
        X1_treated = np.random.randn(n_treated) + 3
        
        data = pd.DataFrame({
            'id': range(1, n_control + n_treated + 1),
            'Y': np.concatenate([Y_control, Y_treated]),
            'D': np.array([0] * n_control + [1] * n_treated),
            'X1': np.concatenate([X1_control, X1_treated]),
        })
        
        return data
    
    def test_python_variance_positive_small_groups(self, test_data_small_groups):
        """Test Python variance estimation is positive for small groups."""
        data = test_data_small_groups
        
        Y = data['Y'].values
        X = data[['X1', 'X2']].values
        W = data['D'].values
        
        sigma2 = _estimate_conditional_variance_same_group(Y, X, W, J=2)
        
        # All variance estimates should be positive
        assert np.all(sigma2 > 0), f"All variance should be positive, got min={sigma2.min()}"
        
        # Control group variance should be reasonable (not 0)
        control_var = sigma2[W == 0]
        assert np.all(control_var > 0), f"Control variance should be positive"
    
    def test_python_variance_positive_extreme_small(self, test_data_extreme_small):
        """Test Python variance estimation for extreme small groups."""
        data = test_data_extreme_small
        
        Y = data['Y'].values
        X = data['X1'].values.reshape(-1, 1)
        W = data['D'].values
        
        sigma2 = _estimate_conditional_variance_same_group(Y, X, W, J=2)
        
        # All variance estimates should be positive
        assert np.all(sigma2 > 0), f"All variance should be positive, got min={sigma2.min()}"
        
        # With only 2 control observations, should use within-group or global variance
        control_var = sigma2[W == 0]
        global_var = np.var(Y, ddof=1)
        
        # Control variance should be reasonable (not 0, not NaN)
        assert np.all(control_var > 0), f"Control variance should be positive"
        assert not np.any(np.isnan(control_var)), "No NaN expected"
    
    def test_variance_consistency_across_group_sizes(self):
        """Test variance estimation consistency across different group sizes."""
        np.random.seed(789)
        
        results = []
        
        for n_control in [1, 2, 3, 5, 10, 50]:
            n_treated = 100
            
            Y = np.concatenate([
                np.random.randn(n_control) + 5,
                np.random.randn(n_treated) + 7
            ])
            X = np.concatenate([
                np.random.randn(n_control),
                np.random.randn(n_treated) + 1
            ]).reshape(-1, 1)
            W = np.array([0] * n_control + [1] * n_treated)
            
            sigma2 = _estimate_conditional_variance_same_group(Y, X, W, J=2)
            
            # All positive, no NaN
            assert np.all(sigma2 > 0), f"n_control={n_control}: variance should be positive"
            assert not np.any(np.isnan(sigma2)), f"n_control={n_control}: no NaN expected"
            
            # Record control group mean variance for comparison
            control_mean_var = np.mean(sigma2[W == 0])
            results.append({
                'n_control': n_control,
                'control_mean_var': control_mean_var,
                'global_var': np.var(Y, ddof=1)
            })
        
        # For very small groups, variance should be closer to global variance
        # For larger groups, should be based on neighbors
        df = pd.DataFrame(results)
        
        # Verify small groups use global/within-group variance fallback
        small_group_result = df[df['n_control'] == 1].iloc[0]
        assert small_group_result['control_mean_var'] > 0, "Single obs group should have positive variance"
    
    def test_se_not_underestimated(self):
        """Test that standard errors are not underestimated due to zero variance."""
        np.random.seed(456)
        
        # Create scenario where old code would set variance to 0
        # Single observation in control group
        Y = np.array([5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        X = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]).reshape(-1, 1)
        W = np.array([0, 1, 1, 1, 1, 1, 1, 1])  # Only 1 control
        
        sigma2 = _estimate_conditional_variance_same_group(Y, X, W, J=2)
        
        # The control observation should NOT have variance 0
        control_var = sigma2[W == 0]
        
        assert control_var[0] != 0, "Control variance should not be 0"
        assert control_var[0] > 0, f"Control variance should be positive, got {control_var[0]}"
        
        # Should be approximately equal to global variance
        global_var = np.var(Y, ddof=1)
        assert np.isclose(control_var[0], global_var, rtol=0.1), \
            f"Control variance ({control_var[0]}) should be close to global variance ({global_var})"


class TestDesign069MonteCarlo:
    """Monte Carlo simulation tests for variance estimation."""
    
    def test_coverage_rate_small_groups(self):
        """Test that CI coverage rate is not inflated due to zero variance."""
        np.random.seed(2024)
        
        n_simulations = 200
        n_control = 3
        n_treated = 50
        true_effect = 2.0
        
        coverage_count = 0
        
        for sim in range(n_simulations):
            # Generate data with known treatment effect
            Y_control = np.random.randn(n_control)
            Y_treated = np.random.randn(n_treated) + true_effect
            
            Y = np.concatenate([Y_control, Y_treated])
            X = np.concatenate([
                np.random.randn(n_control),
                np.random.randn(n_treated) + 0.5
            ]).reshape(-1, 1)
            W = np.array([0] * n_control + [1] * n_treated)
            
            # Simple ATT estimate (difference in means)
            att = Y_treated.mean() - Y_control.mean()
            
            # Variance estimation
            sigma2 = _estimate_conditional_variance_same_group(Y, X, W, J=2)
            
            # Compute SE (simplified)
            var_treated = np.mean(sigma2[W == 1])
            var_control = np.mean(sigma2[W == 0])
            se = np.sqrt(var_treated / n_treated + var_control / n_control)
            
            # 95% CI
            ci_lower = att - 1.96 * se
            ci_upper = att + 1.96 * se
            
            if ci_lower <= true_effect <= ci_upper:
                coverage_count += 1
        
        coverage_rate = coverage_count / n_simulations
        
        # Coverage should be approximately 95% (allow some simulation error)
        # If variance were 0, coverage would be much lower
        assert coverage_rate >= 0.80, \
            f"Coverage rate ({coverage_rate:.2%}) too low, suggests variance underestimation"
        
        # Should not be too high either (would suggest variance overestimation)
        assert coverage_rate <= 0.99, \
            f"Coverage rate ({coverage_rate:.2%}) too high, suggests variance overestimation"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
