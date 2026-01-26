"""
DESIGN-053: Detrend Time Variable Centering Tests

This module tests the numerical stability improvements made to detrend_unit()
and detrendq_unit() functions in transformations.py.

The fix centers the time variable at its pre-treatment mean before OLS
estimation, which:
1. Reduces the condition number of X'X from O(t_max^2) to O(1)
2. Prevents precision loss for large time indices (e.g., years 2000+)
3. Does not change the final residuals (mathematical invariance)

Test Categories:
- Numerical stability verification
- Mathematical invariance tests
- Extreme value stress tests
- Stata E2E validation
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from lwdid.transformations import detrend_unit, detrendq_unit, apply_rolling_transform


class TestDetrendNumericalStability:
    """Tests for numerical stability improvements in detrend functions."""

    def test_condition_number_improvement_detrend(self):
        """Verify centering significantly reduces condition number of X'X.
        
        For year-based time indices (e.g., 2000-2010), centering should
        reduce condition number by several orders of magnitude.
        """
        # Create data with large time indices
        np.random.seed(42)
        years = np.arange(2000, 2011)
        n = len(years)
        
        # Build design matrices
        # Without centering
        X_no_center = sm.add_constant(years.astype(float))
        cond_no_center = np.linalg.cond(X_no_center.T @ X_no_center)
        
        # With centering
        t_mean = years.mean()
        t_centered = years - t_mean
        X_centered = sm.add_constant(t_centered.astype(float))
        cond_centered = np.linalg.cond(X_centered.T @ X_centered)
        
        # Centering should improve condition number by at least 10^9
        assert cond_no_center / cond_centered > 1e9, \
            f"Condition number improvement insufficient: {cond_no_center / cond_centered:.2e}"
        
        # Centered condition number should be small (close to identity)
        assert cond_centered < 100, \
            f"Centered condition number too high: {cond_centered:.2e}"

    def test_condition_number_improvement_detrendq(self):
        """Verify centering reduces condition number for quarterly model."""
        np.random.seed(123)
        
        # Large quarterly time indices
        tindex = np.arange(8081, 8093)  # Example: year*4 + quarter
        quarters = np.tile([1, 2, 3, 4], 3)
        
        # Build design matrices
        q_dummies = pd.get_dummies(
            pd.Categorical(quarters, categories=[1, 2, 3, 4]),
            drop_first=True, prefix='q', dtype=float
        )
        
        # Without centering
        X_no_center = np.column_stack([
            np.ones(len(tindex)),
            tindex.astype(float),
            q_dummies.values
        ])
        cond_no_center = np.linalg.cond(X_no_center.T @ X_no_center)
        
        # With centering
        t_mean = tindex.mean()
        t_centered = tindex - t_mean
        X_centered = np.column_stack([
            np.ones(len(tindex)),
            t_centered.astype(float),
            q_dummies.values
        ])
        cond_centered = np.linalg.cond(X_centered.T @ X_centered)
        
        assert cond_no_center / cond_centered > 1e6, \
            f"Condition number improvement insufficient for detrendq"


class TestDetrendMathematicalInvariance:
    """Tests verifying centering does not change residuals."""

    def test_detrend_residuals_invariant_to_centering(self):
        """Verify centering produces identical residuals to non-centering.
        
        Mathematical proof: Y = α' + β(t - t̄) = (α' - βt̄) + βt = α + βt
        Therefore predicted values and residuals are invariant.
        """
        np.random.seed(999)
        t = np.array([2000, 2001, 2002, 2003, 2004, 2005], dtype=float)
        y = 100 + 2 * t + np.random.normal(0, 0.1, 6)
        post = np.array([0, 0, 0, 0, 1, 1])
        
        data = pd.DataFrame({'tindex': t, 'y': y, 'post': post})
        
        # Method 1: Without centering (manual OLS)
        pre_data = data[data['post'] == 0]
        X_no_center = sm.add_constant(pre_data['tindex'].values)
        model = sm.OLS(pre_data['y'].values, X_no_center).fit()
        X_all = sm.add_constant(data['tindex'].values)
        yhat_no_center = model.predict(X_all)
        ydot_no_center = data['y'].values - yhat_no_center
        
        # Method 2: With centering (using detrend_unit)
        _, ydot_centered = detrend_unit(data, 'y', 'tindex', 'post')
        
        # Residuals should be identical within floating point precision.
        # Note: For large time values (e.g., 2000+), floating point arithmetic
        # can introduce differences up to ~1e-9 due to different computation paths.
        np.testing.assert_allclose(
            ydot_no_center, ydot_centered,
            rtol=1e-8, atol=1e-8,
            err_msg="Centering changed residuals unexpectedly"
        )

    def test_detrendq_residuals_invariant_to_centering(self):
        """Verify centering produces identical residuals for quarterly model."""
        np.random.seed(456)
        
        # Create quarterly data
        tindex = np.arange(2020, 2032, dtype=float)
        quarters = np.tile([1, 2, 3, 4], 3)
        seasonal = np.array([0, 2, 1, 3])
        y = 50 + 0.5 * tindex + np.tile(seasonal, 3) + np.random.normal(0, 0.5, 12)
        post = np.array([0] * 8 + [1] * 4)
        
        data = pd.DataFrame({
            'tindex': tindex,
            'quarter': quarters,
            'y': y,
            'post': post
        })
        
        # Manual non-centered computation
        pre_data = data[data['post'] == 0].copy()
        q_cat = pd.Categorical(pre_data['quarter'], categories=[1, 2, 3, 4])
        q_dummies = pd.get_dummies(q_cat, drop_first=True, prefix='q', dtype=float)
        
        X_pre = np.column_stack([
            np.ones(len(pre_data)),
            pre_data['tindex'].values,
            q_dummies.values
        ])
        model = sm.OLS(pre_data['y'].values, X_pre).fit()
        
        q_cat_all = pd.Categorical(data['quarter'], categories=[1, 2, 3, 4])
        q_dummies_all = pd.get_dummies(q_cat_all, drop_first=True, prefix='q', dtype=float)
        
        X_all = np.column_stack([
            np.ones(len(data)),
            data['tindex'].values,
            q_dummies_all.values
        ])
        yhat = model.predict(X_all)
        ydot_no_center = data['y'].values - yhat
        
        # Using detrendq_unit with centering
        _, ydot_centered = detrendq_unit(data, 'y', 'tindex', 'quarter', 'post')
        
        np.testing.assert_allclose(
            ydot_no_center, ydot_centered,
            rtol=1e-9, atol=1e-9,
            err_msg="Centering changed detrendq residuals unexpectedly"
        )


class TestDetrendExtremeValues:
    """Stress tests with extreme time values."""

    def test_detrend_extreme_time_values(self):
        """Verify detrend handles extreme time indices (e.g., 1,000,000)."""
        np.random.seed(42)
        
        # Extreme time values
        t = np.array([1000000, 1000001, 1000002, 1000003, 1000004, 1000005], dtype=float)
        y = 50 + 0.001 * t + np.random.normal(0, 1, 6)
        post = np.array([0, 0, 0, 0, 1, 1])
        
        data = pd.DataFrame({'tindex': t, 'y': y, 'post': post})
        
        # Should complete without numerical issues
        yhat, ydot = detrend_unit(data, 'y', 'tindex', 'post')
        
        # Pre-period residuals should have near-zero mean
        pre_residuals = ydot[post == 0]
        assert np.abs(pre_residuals.mean()) < 1e-10, \
            f"Pre-period residual mean too large: {pre_residuals.mean()}"
        
        # All values should be finite
        assert np.all(np.isfinite(ydot)), "Non-finite residuals detected"

    def test_detrend_small_time_range(self):
        """Verify detrend works correctly with small time range (t=1,2,3,...)."""
        np.random.seed(789)
        
        t = np.array([1, 2, 3, 4, 5, 6], dtype=float)
        y = 10 + 2 * t + np.random.normal(0, 0.1, 6)
        post = np.array([0, 0, 0, 0, 1, 1])
        
        data = pd.DataFrame({'tindex': t, 'y': y, 'post': post})
        
        yhat, ydot = detrend_unit(data, 'y', 'tindex', 'post')
        
        # Pre-period residuals should be near zero
        pre_residuals = ydot[post == 0]
        assert np.allclose(pre_residuals.mean(), 0, atol=1e-10)


class TestApplyRollingTransformIntegration:
    """Integration tests for apply_rolling_transform with detrend."""

    def test_apply_rolling_transform_detrend_large_years(self):
        """Test apply_rolling_transform with year-based time indices."""
        np.random.seed(42)
        
        # Create panel data with large year values
        data_rows = []
        for unit in [1, 2, 3]:
            alpha = 100 + unit * 10
            beta = 2 + unit * 0.1
            
            for year in [2000, 2001, 2002, 2003, 2004, 2005]:
                post = 1 if year >= 2004 else 0
                treatment = 5 if (unit == 1 and post) else 0
                y = alpha + beta * year + treatment + np.random.normal(0, 0.1)
                
                data_rows.append({
                    'id': unit,
                    'year': year,
                    'd': 1 if unit == 1 else 0,
                    'post_': post,
                    'tindex': year,
                    'y': y
                })
        
        data = pd.DataFrame(data_rows)
        
        # Apply transform
        result = apply_rolling_transform(
            data=data,
            y='y',
            ivar='id',
            tindex='tindex',
            post='post_',
            rolling='detrend',
            tpost1=2004
        )
        
        # Check pre-period residuals are near zero per unit
        for unit_id in [1, 2, 3]:
            unit_pre = result[(result['id'] == unit_id) & (result['post_'] == 0)]
            assert np.allclose(unit_pre['ydot'].mean(), 0, atol=1e-10), \
                f"Unit {unit_id} pre-period mean not zero"
        
        # Treatment effect should be captured
        treated_postavg = result[result['id'] == 1]['ydot_postavg'].iloc[0]
        control_postavg = result[result['id'] != 1].groupby('id')['ydot_postavg'].first().mean()
        estimated_att = treated_postavg - control_postavg
        
        # Should be close to true treatment effect of 5
        assert abs(estimated_att - 5.0) < 1.0, \
            f"ATT estimate {estimated_att} too far from true effect 5.0"


class TestStataE2EValidation:
    """End-to-end validation tests comparing Python and Stata results.
    
    These tests use pre-computed Stata reference values to verify
    numerical equivalence between implementations.
    """

    # Stata reference values (computed independently)
    STATA_REFERENCE = {
        'unit_1': {
            'ydot': [0.04438477, -0.05771484, -0.01772461, 0.03105469, 4.8166504, 4.7780273],
            'ydot_postavg': 4.7973389
        },
        'unit_2': {
            'ydot': [0.03225098, -0.00556641, -0.08562012, 0.05893555, 0.00183105, 0.04506836],
            'ydot_postavg': 0.02344971
        },
        'unit_3': {
            'ydot': [0.08989258, -0.10366211, -0.06235352, 0.07612305, 0.0534668, 0.20805664],
            'ydot_postavg': 0.13076172
        },
        'unit_4': {
            'ydot': [0.01010742, -0.08964844, 0.14897461, -0.06943359, -0.08935547, -0.28798828],
            'ydot_postavg': -0.18867187
        }
    }

    @pytest.fixture
    def e2e_test_data(self):
        """Load E2E test data."""
        import os
        csv_path = os.path.join(
            os.path.dirname(__file__),
            'data', 'detrend_e2e_test.csv'
        )
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        
        # Generate data if not exists
        np.random.seed(42)
        data_rows = []
        for unit in [1, 2, 3, 4]:
            is_treated = 1 if unit == 1 else 0
            if unit == 1:
                alpha, beta = 100, 2.0
            elif unit == 2:
                alpha, beta = 80, 1.5
            elif unit == 3:
                alpha, beta = 120, 2.5
            else:
                alpha, beta = 90, 1.8
            
            for year in [2000, 2001, 2002, 2003, 2004, 2005]:
                post = 1 if year >= 2004 else 0
                treatment = 5.0 if (is_treated and post) else 0.0
                noise = np.random.normal(0, 0.1)
                y = alpha + beta * year + treatment + noise
                
                data_rows.append({
                    'id': unit,
                    'year': year,
                    'd': is_treated,
                    'post': post,
                    'y': round(y, 4)
                })
        
        return pd.DataFrame(data_rows)

    def test_python_stata_ydot_equivalence(self, e2e_test_data):
        """Verify Python ydot values match Stata reference within tolerance."""
        data = e2e_test_data.copy()
        data['d_'] = data['d'].astype(int)
        data['post_'] = data['post'].astype(int)
        data['tindex'] = data['year']
        
        result = apply_rolling_transform(
            data=data,
            y='y',
            ivar='id',
            tindex='tindex',
            post='post_',
            rolling='detrend',
            tpost1=2004
        )
        
        # Compare each unit
        for unit_id in [1, 2, 3, 4]:
            unit_result = result[result['id'] == unit_id]['ydot'].values
            stata_ref = self.STATA_REFERENCE[f'unit_{unit_id}']['ydot']
            
            # Allow 1e-3 tolerance (accounts for CSV precision loss)
            np.testing.assert_allclose(
                unit_result, stata_ref,
                rtol=1e-3, atol=1e-3,
                err_msg=f"Unit {unit_id} ydot mismatch vs Stata"
            )

    def test_python_stata_att_equivalence(self, e2e_test_data):
        """Verify ATT estimate matches Stata reference."""
        data = e2e_test_data.copy()
        data['d_'] = data['d'].astype(int)
        data['post_'] = data['post'].astype(int)
        data['tindex'] = data['year']
        
        result = apply_rolling_transform(
            data=data,
            y='y',
            ivar='id',
            tindex='tindex',
            post='post_',
            rolling='detrend',
            tpost1=2004
        )
        
        # Python ATT
        treated_postavg = result[result['d'] == 1]['ydot_postavg'].iloc[0]
        control_postavg = result[result['d'] == 0].groupby('id')['ydot_postavg'].first().mean()
        python_att = treated_postavg - control_postavg
        
        # Stata ATT (from reference values)
        stata_treated = self.STATA_REFERENCE['unit_1']['ydot_postavg']
        stata_control = np.mean([
            self.STATA_REFERENCE['unit_2']['ydot_postavg'],
            self.STATA_REFERENCE['unit_3']['ydot_postavg'],
            self.STATA_REFERENCE['unit_4']['ydot_postavg']
        ])
        stata_att = stata_treated - stata_control
        
        # ATT should match within 1e-3
        assert abs(python_att - stata_att) < 1e-3, \
            f"ATT mismatch: Python={python_att:.6f}, Stata={stata_att:.6f}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
