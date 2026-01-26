"""
Unit tests for BUG-295, BUG-296, BUG-297 fixes.

BUG-295: core.py trim_threshold boundary condition inconsistency
    - Changed from (0, 0.5] to (0, 0.5) for consistency with staggered/estimators.py
    - At threshold=0.5, propensity scores clip to [0.5, 0.5], excluding all observations

BUG-296: staggered/estimators.py paper-style conditional variance boundary
    - Changed from sigma2=0.0 (anti-conservative) to raising ValueError
    - J_actual <= 0 should never occur given J >= 1 validation

BUG-297: staggered/randomization.py duplicate NaN validation (dead code)
    - Already fixed: duplicate NaN check removed, only entry-point validation remains
"""
import pytest
import numpy as np
import pandas as pd
import warnings

from lwdid.core import lwdid
from lwdid.staggered.estimators import estimate_ipwra, estimate_ipw
from lwdid.staggered.randomization import randomization_inference_staggered, RandomizationError


class TestBug295TrimThresholdBoundary:
    """Test trim_threshold parameter validation boundary condition."""

    @pytest.fixture
    def common_timing_data(self):
        """Create minimal common timing panel data for testing."""
        np.random.seed(42)
        n_units = 50
        n_periods = 6
        treatment_period = 4
        
        data = []
        for i in range(n_units):
            treated = i < n_units // 2
            x1 = np.random.randn()
            x2 = np.random.randn()
            
            for t in range(1, n_periods + 1):
                post = 1 if t >= treatment_period else 0
                y = 1.0 + 0.5 * x1 + 0.3 * x2 + np.random.randn() * 0.5
                if treated and post:
                    y += 1.0  # Treatment effect
                
                data.append({
                    'id': i,
                    'year': t,
                    'y': y,
                    'treated': int(treated),
                    'post': post,
                    'x1': x1,
                    'x2': x2,
                })
        
        return pd.DataFrame(data)

    @pytest.fixture
    def staggered_data(self):
        """Create minimal staggered panel data for testing."""
        np.random.seed(42)
        n_units = 40
        n_periods = 8
        
        data = []
        for i in range(n_units):
            # Cohort assignment
            if i < 10:
                gvar = np.inf  # Never treated
            elif i < 20:
                gvar = 4
            elif i < 30:
                gvar = 5
            else:
                gvar = 6
            
            x1 = np.random.randn()
            x2 = np.random.randn()
            
            for t in range(1, n_periods + 1):
                treated = gvar != np.inf and t >= gvar
                y = 1.0 + 0.5 * x1 + 0.3 * x2 + np.random.randn() * 0.5
                if treated:
                    y += 0.8  # Treatment effect
                
                data.append({
                    'id': i,
                    'year': t,
                    'gvar': gvar,
                    'y': y,
                    'x1': x1,
                    'x2': x2,
                })
        
        return pd.DataFrame(data)

    def test_trim_threshold_0_5_raises_error_common_timing(self, common_timing_data):
        """Test that trim_threshold=0.5 raises ValueError in common timing mode."""
        with pytest.raises(ValueError) as exc_info:
            lwdid(
                data=common_timing_data,
                y='y',
                ivar='id',
                tvar='year',
                d='treated',
                post='post',
                estimator='ipw',
                controls=['x1', 'x2'],
                trim_threshold=0.5,
            )
        
        assert "trim_threshold" in str(exc_info.value).lower()
        assert "0.5" in str(exc_info.value)

    def test_trim_threshold_0_5_raises_error_staggered(self, staggered_data):
        """Test that trim_threshold=0.5 raises ValueError in staggered mode."""
        with pytest.raises(ValueError) as exc_info:
            lwdid(
                data=staggered_data,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                estimator='ipw',
                controls=['x1', 'x2'],
                trim_threshold=0.5,
            )
        
        assert "trim_threshold" in str(exc_info.value).lower()
        assert "0.5" in str(exc_info.value)

    def test_trim_threshold_0_49_is_valid(self, common_timing_data):
        """Test that trim_threshold=0.49 passes validation."""
        # Should not raise - 0.49 is within valid range (0, 0.5)
        try:
            result = lwdid(
                data=common_timing_data,
                y='y',
                ivar='id',
                tvar='year',
                d='treated',
                post='post',
                estimator='ipw',
                controls=['x1', 'x2'],
                trim_threshold=0.49,
            )
            # Result should be returned without error
            assert result is not None
        except ValueError as e:
            # Only fail if error is about trim_threshold
            if "trim_threshold" in str(e).lower():
                pytest.fail(f"trim_threshold=0.49 should be valid, got: {e}")
            # Other ValueErrors (e.g., data issues) are acceptable
            pass

    def test_trim_threshold_0_01_is_valid(self, common_timing_data):
        """Test that trim_threshold=0.01 (default) passes validation."""
        result = lwdid(
            data=common_timing_data,
            y='y',
            ivar='id',
            tvar='year',
            d='treated',
            post='post',
            estimator='ipw',
            controls=['x1', 'x2'],
            trim_threshold=0.01,
        )
        assert result is not None

    def test_trim_threshold_zero_raises_error(self, common_timing_data):
        """Test that trim_threshold=0 raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            lwdid(
                data=common_timing_data,
                y='y',
                ivar='id',
                tvar='year',
                d='treated',
                post='post',
                estimator='ipw',
                controls=['x1', 'x2'],
                trim_threshold=0.0,
            )
        
        assert "trim_threshold" in str(exc_info.value).lower()

    def test_trim_threshold_negative_raises_error(self, common_timing_data):
        """Test that negative trim_threshold raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            lwdid(
                data=common_timing_data,
                y='y',
                ivar='id',
                tvar='year',
                d='treated',
                post='post',
                estimator='ipw',
                controls=['x1', 'x2'],
                trim_threshold=-0.1,
            )
        
        assert "trim_threshold" in str(exc_info.value).lower()


class TestBug296ConditionalVarianceBoundary:
    """Test conditional variance estimation boundary condition handling."""

    @pytest.fixture
    def staggered_ipwra_data(self):
        """Create staggered panel data suitable for IPWRA estimation."""
        np.random.seed(123)
        n_units = 60
        n_periods = 8
        
        data = []
        for i in range(n_units):
            # Cohort assignment with good overlap
            if i < 20:
                gvar = np.inf  # Never treated
            elif i < 40:
                gvar = 4
            else:
                gvar = 5
            
            # Covariates
            x1 = np.random.randn()
            x2 = np.random.randn()
            
            for t in range(1, n_periods + 1):
                treated = gvar != np.inf and t >= gvar
                # Outcome with treatment effect
                y = 2.0 + 0.5 * x1 + 0.3 * x2 + np.random.randn() * 0.3
                if treated:
                    y += 1.0
                
                data.append({
                    'id': i,
                    'year': t,
                    'gvar': gvar,
                    'y': y,
                    'x1': x1,
                    'x2': x2,
                })
        
        return pd.DataFrame(data)

    def test_conditional_variance_normal_case_no_error(self, staggered_ipwra_data):
        """Test that normal IPWRA estimation does not trigger J_actual <= 0 error."""
        # Normal case should complete without ValueError about J_actual
        result = lwdid(
            data=staggered_ipwra_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            estimator='ipwra',
            controls=['x1', 'x2'],
            rolling='demean',
        )
        
        # Should return valid results
        assert result is not None
        assert hasattr(result, 'att')
        assert np.isfinite(result.att)

    def test_psm_estimator_normal_case(self, staggered_ipwra_data):
        """Test that PSM estimation works normally (uses conditional variance)."""
        result = lwdid(
            data=staggered_ipwra_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            estimator='psm',
            controls=['x1', 'x2'],
            rolling='demean',
            n_neighbors=3,
        )
        
        assert result is not None
        assert np.isfinite(result.att)


class TestBug297NaNValidation:
    """Test observed_att NaN validation in randomization inference.
    
    Note: Use target='cohort_time' to test NaN/Inf validation independently
    of the never-treated unit requirement for overall effects.
    """

    @pytest.fixture
    def ri_staggered_data(self):
        """Create staggered data suitable for randomization inference."""
        np.random.seed(456)
        n_units = 40
        n_periods = 8
        
        data = []
        for i in range(n_units):
            # Cohort assignment (no never-treated needed for cohort_time target)
            if i < 20:
                gvar = 4  # Treated cohort 1 - 20 units
            else:
                gvar = 5  # Treated cohort 2 - 20 units
            
            for t in range(1, n_periods + 1):
                y = np.random.randn() + (0.5 if t >= gvar else 0)
                data.append({
                    'id': i,
                    'year': t,
                    'gvar': gvar,
                    'y': y,
                })
        
        return pd.DataFrame(data)

    def test_nan_observed_att_raises_error_at_entry(self, ri_staggered_data):
        """Test that NaN observed_att raises error at function entry."""
        with pytest.raises(RandomizationError) as exc_info:
            randomization_inference_staggered(
                data=ri_staggered_data,
                gvar='gvar',
                ivar='id',
                tvar='year',
                y='y',
                observed_att=np.nan,
                rireps=50,
                rolling='demean',
                target='cohort_time',
                target_cohort=4,
                target_period=5,
            )
        
        assert "NaN" in str(exc_info.value)
        assert "observed ATT" in str(exc_info.value).lower() or "att" in str(exc_info.value).lower()

    def test_inf_observed_att_raises_error_at_entry(self, ri_staggered_data):
        """Test that Inf observed_att raises error at function entry."""
        with pytest.raises(RandomizationError) as exc_info:
            randomization_inference_staggered(
                data=ri_staggered_data,
                gvar='gvar',
                ivar='id',
                tvar='year',
                y='y',
                observed_att=np.inf,
                rireps=50,
                rolling='demean',
                target='cohort_time',
                target_cohort=4,
                target_period=5,
            )
        
        assert "infinite" in str(exc_info.value).lower() or "inf" in str(exc_info.value).lower()

    def test_valid_observed_att_completes(self):
        """Test that valid observed_att allows RI to complete."""
        # Create data with never-treated units for proper RI
        np.random.seed(789)
        n_units = 40
        n_periods = 8
        
        data = []
        for i in range(n_units):
            # Include never-treated units for proper control
            if i < 16:
                gvar = np.inf  # Never treated - 16 units
            elif i < 28:
                gvar = 4  # Treated cohort 1 - 12 units
            else:
                gvar = 5  # Treated cohort 2 - 12 units
            
            for t in range(1, n_periods + 1):
                y = np.random.randn() + (0.5 if gvar != np.inf and t >= gvar else 0)
                data.append({
                    'id': i,
                    'year': t,
                    'gvar': gvar,
                    'y': y,
                })
        
        ri_data = pd.DataFrame(data)
        
        # Count never-treated units
        n_never_treated = len(ri_data[ri_data['gvar'] == np.inf]['id'].unique())
        
        # Should complete without error
        result = randomization_inference_staggered(
            data=ri_data,
            gvar='gvar',
            ivar='id',
            tvar='year',
            y='y',
            observed_att=0.5,
            rireps=50,
            rolling='demean',
            seed=42,
            n_never_treated=n_never_treated,
        )
        
        assert result is not None
        assert hasattr(result, 'p_value')
        assert 0 <= result.p_value <= 1


class TestBug295EstimatorsConsistency:
    """Test trim_threshold validation consistency across estimators.py."""

    @pytest.fixture
    def estimation_data(self):
        """Create data for direct estimator function testing."""
        np.random.seed(789)
        n = 200
        
        # Generate data with good overlap
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        
        # Propensity score model
        logit = 0.5 * x1 + 0.3 * x2
        prob = 1 / (1 + np.exp(-logit))
        d = (np.random.rand(n) < prob).astype(int)
        
        # Outcome model
        y = 2.0 + 0.5 * x1 + 0.3 * x2 + 1.0 * d + np.random.randn(n) * 0.5
        
        return pd.DataFrame({
            'y': y,
            'd': d,
            'x1': x1,
            'x2': x2,
        })

    def test_estimate_ipw_trim_0_5_raises_error(self, estimation_data):
        """Test estimate_ipw rejects trim_threshold=0.5."""
        with pytest.raises(ValueError) as exc_info:
            estimate_ipw(
                data=estimation_data,
                y='y',
                d='d',
                propensity_controls=['x1', 'x2'],
                trim_threshold=0.5,
            )
        
        assert "trim_threshold" in str(exc_info.value).lower()

    def test_estimate_ipwra_trim_0_5_raises_error(self, estimation_data):
        """Test estimate_ipwra rejects trim_threshold=0.5."""
        with pytest.raises(ValueError) as exc_info:
            estimate_ipwra(
                data=estimation_data,
                y='y',
                d='d',
                controls=['x1', 'x2'],
                trim_threshold=0.5,
            )
        
        assert "trim_threshold" in str(exc_info.value).lower()

    def test_estimate_ipw_trim_0_49_valid(self, estimation_data):
        """Test estimate_ipw accepts trim_threshold=0.49."""
        result = estimate_ipw(
            data=estimation_data,
            y='y',
            d='d',
            propensity_controls=['x1', 'x2'],
            trim_threshold=0.49,
        )
        
        assert result is not None
        assert hasattr(result, 'att')

    def test_estimate_ipwra_trim_0_49_valid(self, estimation_data):
        """Test estimate_ipwra accepts trim_threshold=0.49."""
        result = estimate_ipwra(
            data=estimation_data,
            y='y',
            d='d',
            controls=['x1', 'x2'],
            trim_threshold=0.49,
        )
        
        assert result is not None
        assert hasattr(result, 'att')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
