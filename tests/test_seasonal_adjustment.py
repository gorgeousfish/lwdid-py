"""
Unit tests for seasonal adjustment enhancement.

Tests cover:
- demeanq_unit() and detrendq_unit() with Q=4, 12, 52
- transform_staggered_demeanq() and transform_staggered_detrendq()
- Validation functions for seasonal data
- Edge cases and boundary conditions
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from lwdid.transformations import (
    demeanq_unit,
    detrendq_unit,
    apply_rolling_transform,
    _validate_seasonal_transform_requirements,
)
from lwdid.staggered.transformations import (
    transform_staggered_demeanq,
    transform_staggered_detrendq,
    _compute_pre_treatment_seasonal_mean,
    _compute_pre_treatment_seasonal_trend,
)
from lwdid.validation import (
    validate_season_coverage,
    validate_season_diversity,
)
from lwdid.exceptions import (
    InsufficientPrePeriodsError,
    InsufficientQuarterDiversityError,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def quarterly_unit_data():
    """Create quarterly panel data for a single unit."""
    np.random.seed(42)
    n_periods = 12  # 3 years of quarterly data
    
    # Generate seasonal pattern: base + trend + seasonal + noise
    base = 100
    trend = 0.5
    seasonal_effects = {1: 0, 2: 5, 3: 10, 4: 3}  # Q1 is reference
    
    data = []
    for t in range(1, n_periods + 1):
        quarter = ((t - 1) % 4) + 1
        y = base + trend * t + seasonal_effects[quarter] + np.random.normal(0, 1)
        post = 1 if t > 8 else 0  # Treatment starts at t=9
        data.append({'tindex': t, 'quarter': quarter, 'y': y, 'post': post})
    
    return pd.DataFrame(data)


@pytest.fixture
def monthly_unit_data():
    """Create monthly panel data for a single unit."""
    np.random.seed(42)
    n_periods = 36  # 3 years of monthly data
    
    # Generate seasonal pattern
    base = 100
    trend = 0.2
    # Seasonal effects: higher in summer months
    seasonal_effects = {
        1: 0, 2: 1, 3: 3, 4: 5, 5: 8, 6: 12,
        7: 15, 8: 14, 9: 10, 10: 6, 11: 3, 12: 1
    }
    
    data = []
    for t in range(1, n_periods + 1):
        month = ((t - 1) % 12) + 1
        y = base + trend * t + seasonal_effects[month] + np.random.normal(0, 1)
        post = 1 if t > 24 else 0  # Treatment starts at t=25
        data.append({'tindex': t, 'month': month, 'y': y, 'post': post})
    
    return pd.DataFrame(data)


@pytest.fixture
def staggered_quarterly_data():
    """Create staggered adoption panel data with quarterly seasonality."""
    np.random.seed(42)
    n_units = 10
    n_periods = 20
    
    # Cohort assignments: some treated at t=8, some at t=12, some never
    cohorts = {
        1: 8, 2: 8, 3: 8,      # Early adopters
        4: 12, 5: 12, 6: 12,   # Late adopters
        7: np.inf, 8: np.inf, 9: np.inf, 10: np.inf  # Never treated
    }
    
    seasonal_effects = {1: 0, 2: 5, 3: 10, 4: 3}
    
    data = []
    for unit_id in range(1, n_units + 1):
        base = 100 + unit_id * 5  # Unit-specific intercept
        trend = 0.3 + unit_id * 0.05  # Unit-specific trend
        
        for t in range(1, n_periods + 1):
            quarter = ((t - 1) % 4) + 1
            gvar = cohorts[unit_id]
            
            # Treatment effect
            tau = 0
            if not np.isinf(gvar) and t >= gvar:
                tau = 10  # Constant treatment effect
            
            y = base + trend * t + seasonal_effects[quarter] + tau + np.random.normal(0, 1)
            
            data.append({
                'id': unit_id,
                'tindex': t,
                'quarter': quarter,
                'y': y,
                'gvar': gvar
            })
    
    return pd.DataFrame(data)


# =============================================================================
# Tests for demeanq_unit with different Q values
# =============================================================================

class TestDemeanqUnit:
    """Tests for demeanq_unit function."""
    
    def test_quarterly_basic(self, quarterly_unit_data):
        """Test basic quarterly demeaning (Q=4)."""
        yhat, ydot = demeanq_unit(
            quarterly_unit_data, 'y', 'quarter', 'post', Q=4
        )
        
        # Check output shapes
        assert len(yhat) == len(quarterly_unit_data)
        assert len(ydot) == len(quarterly_unit_data)
        
        # Pre-treatment residuals should have mean close to zero
        pre_mask = quarterly_unit_data['post'] == 0
        pre_residuals = ydot[pre_mask]
        assert abs(np.nanmean(pre_residuals)) < 0.1
    
    def test_monthly_basic(self, monthly_unit_data):
        """Test monthly demeaning (Q=12)."""
        yhat, ydot = demeanq_unit(
            monthly_unit_data, 'y', 'month', 'post', Q=12
        )
        
        # Check output shapes
        assert len(yhat) == len(monthly_unit_data)
        assert len(ydot) == len(monthly_unit_data)
        
        # Pre-treatment residuals should have mean close to zero
        pre_mask = monthly_unit_data['post'] == 0
        pre_residuals = ydot[pre_mask]
        assert abs(np.nanmean(pre_residuals)) < 0.1
    
    def test_insufficient_observations(self):
        """Test that insufficient observations returns NaN."""
        # Only 3 observations for Q=4 (need at least 5)
        data = pd.DataFrame({
            'tindex': [1, 2, 3, 4, 5],
            'quarter': [1, 2, 3, 1, 2],
            'y': [10, 12, 15, 11, 13],
            'post': [0, 0, 0, 1, 1]
        })
        
        with pytest.warns(UserWarning, match="Insufficient"):
            yhat, ydot = demeanq_unit(data, 'y', 'quarter', 'post', Q=4)
        
        # Should return NaN arrays
        assert np.all(np.isnan(yhat))
        assert np.all(np.isnan(ydot))
    
    def test_seasonal_coefficients_correctness(self):
        """Test that seasonal coefficients are correctly estimated."""
        # Create data with known seasonal pattern
        np.random.seed(123)
        n_pre = 16  # 4 complete years
        
        # Known seasonal effects: Q1=0 (ref), Q2=5, Q3=10, Q4=3
        true_mu = 100
        true_gamma = {1: 0, 2: 5, 3: 10, 4: 3}
        
        data = []
        for t in range(1, n_pre + 5):  # 4 post periods
            quarter = ((t - 1) % 4) + 1
            y = true_mu + true_gamma[quarter]
            post = 1 if t > n_pre else 0
            data.append({'tindex': t, 'quarter': quarter, 'y': y, 'post': post})
        
        df = pd.DataFrame(data)
        yhat, ydot = demeanq_unit(df, 'y', 'quarter', 'post', Q=4)
        
        # Residuals should be very close to zero (no noise in data)
        assert_allclose(ydot, 0, atol=1e-10)


class TestDetrendqUnit:
    """Tests for detrendq_unit function."""
    
    def test_quarterly_basic(self, quarterly_unit_data):
        """Test basic quarterly detrending (Q=4)."""
        yhat, ydot = detrendq_unit(
            quarterly_unit_data, 'y', 'tindex', 'quarter', 'post', Q=4
        )
        
        # Check output shapes
        assert len(yhat) == len(quarterly_unit_data)
        assert len(ydot) == len(quarterly_unit_data)
        
        # Pre-treatment residuals should have mean close to zero
        pre_mask = quarterly_unit_data['post'] == 0
        pre_residuals = ydot[pre_mask]
        assert abs(np.nanmean(pre_residuals)) < 0.5
    
    def test_monthly_basic(self, monthly_unit_data):
        """Test monthly detrending (Q=12)."""
        yhat, ydot = detrendq_unit(
            monthly_unit_data, 'y', 'tindex', 'month', 'post', Q=12
        )
        
        # Check output shapes
        assert len(yhat) == len(monthly_unit_data)
        assert len(ydot) == len(monthly_unit_data)
    
    def test_trend_removal(self):
        """Test that linear trend is correctly removed."""
        # Create data with known trend and seasonal pattern
        n_pre = 20
        true_alpha = 100
        true_beta = 2.0
        true_gamma = {1: 0, 2: 5, 3: 10, 4: 3}
        
        data = []
        for t in range(1, n_pre + 5):
            quarter = ((t - 1) % 4) + 1
            y = true_alpha + true_beta * t + true_gamma[quarter]
            post = 1 if t > n_pre else 0
            data.append({'tindex': t, 'quarter': quarter, 'y': y, 'post': post})
        
        df = pd.DataFrame(data)
        yhat, ydot = detrendq_unit(df, 'y', 'tindex', 'quarter', 'post', Q=4)
        
        # Residuals should be very close to zero
        assert_allclose(ydot, 0, atol=1e-8)


# =============================================================================
# Tests for Staggered Seasonal Transformations
# =============================================================================

class TestStaggeredDemeanq:
    """Tests for transform_staggered_demeanq function."""
    
    def test_basic_functionality(self, staggered_quarterly_data):
        """Test basic staggered seasonal demeaning."""
        result = transform_staggered_demeanq(
            staggered_quarterly_data,
            y='y',
            ivar='id',
            tvar='tindex',
            gvar='gvar',
            season_var='quarter',
            Q=4
        )
        
        # Check that transformation columns were created
        # Cohort 8 should have columns for r=8,9,...,20
        assert 'ydot_g8_r8' in result.columns
        assert 'ydot_g8_r20' in result.columns
        
        # Cohort 12 should have columns for r=12,13,...,20
        assert 'ydot_g12_r12' in result.columns
        assert 'ydot_g12_r20' in result.columns
    
    def test_output_not_all_nan(self, staggered_quarterly_data):
        """Test that output contains valid (non-NaN) values."""
        result = transform_staggered_demeanq(
            staggered_quarterly_data,
            y='y',
            ivar='id',
            tvar='tindex',
            gvar='gvar',
            season_var='quarter',
            Q=4
        )
        
        # At least some values should be non-NaN
        ydot_cols = [c for c in result.columns if c.startswith('ydot_g')]
        for col in ydot_cols:
            assert not result[col].isna().all(), f"Column {col} is all NaN"
    
    def test_missing_columns_error(self):
        """Test that missing columns raise ValueError."""
        data = pd.DataFrame({'x': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Missing required columns"):
            transform_staggered_demeanq(
                data, y='y', ivar='id', tvar='t', gvar='g', season_var='q'
            )
    
    def test_invalid_season_values(self, staggered_quarterly_data):
        """Test that invalid season values raise ValueError."""
        data = staggered_quarterly_data.copy()
        data['quarter'] = data['quarter'] + 10  # Invalid values 11-14
        
        with pytest.raises(ValueError, match="outside valid range"):
            transform_staggered_demeanq(
                data, y='y', ivar='id', tvar='tindex', gvar='gvar',
                season_var='quarter', Q=4
            )


class TestStaggeredDetrendq:
    """Tests for transform_staggered_detrendq function."""
    
    def test_basic_functionality(self, staggered_quarterly_data):
        """Test basic staggered seasonal detrending."""
        result = transform_staggered_detrendq(
            staggered_quarterly_data,
            y='y',
            ivar='id',
            tvar='tindex',
            gvar='gvar',
            season_var='quarter',
            Q=4
        )
        
        # Check that transformation columns were created
        assert 'ycheck_g8_r8' in result.columns
        assert 'ycheck_g12_r12' in result.columns
    
    def test_output_not_all_nan(self, staggered_quarterly_data):
        """Test that output contains valid values."""
        result = transform_staggered_detrendq(
            staggered_quarterly_data,
            y='y',
            ivar='id',
            tvar='tindex',
            gvar='gvar',
            season_var='quarter',
            Q=4
        )
        
        ycheck_cols = [c for c in result.columns if c.startswith('ycheck_g')]
        for col in ycheck_cols:
            assert not result[col].isna().all(), f"Column {col} is all NaN"


# =============================================================================
# Tests for Validation Functions
# =============================================================================

class TestSeasonValidation:
    """Tests for seasonal validation functions."""
    
    def test_season_coverage_valid(self):
        """Test that valid season coverage passes."""
        data = pd.DataFrame({
            'id': [1]*8 + [2]*8,
            'quarter': [1,2,3,4,1,2,3,4] * 2,
            'post': [0,0,0,0,1,1,1,1] * 2
        })
        
        # Should not raise
        validate_season_coverage(data, 'id', 'quarter', 'post', Q=4)
    
    def test_season_coverage_missing(self):
        """Test that missing season coverage raises error."""
        data = pd.DataFrame({
            'id': [1]*6,
            'quarter': [1,2,3,1,2,4],  # Q4 not in pre, but in post
            'post': [0,0,0,1,1,1]
        })
        
        with pytest.raises(InsufficientQuarterDiversityError):
            validate_season_coverage(data, 'id', 'quarter', 'post', Q=4)
    
    def test_season_diversity_valid(self):
        """Test that valid season diversity passes."""
        data = pd.DataFrame({
            'id': [1]*8,
            'quarter': [1,2,3,4,1,2,3,4],
            'post': [0,0,0,0,1,1,1,1]
        })
        
        # Should not raise
        validate_season_diversity(data, 'id', 'quarter', 'post', Q=4)
    
    def test_season_diversity_insufficient(self):
        """Test that insufficient diversity raises error."""
        data = pd.DataFrame({
            'id': [1]*4,
            'quarter': [1,1,1,2],  # Only 1 quarter in pre-period
            'post': [0,0,0,1]
        })
        
        with pytest.raises(InsufficientQuarterDiversityError):
            validate_season_diversity(data, 'id', 'quarter', 'post', Q=4)


# =============================================================================
# Tests for apply_rolling_transform with Q parameter
# =============================================================================

class TestApplyRollingTransformSeasonal:
    """Tests for apply_rolling_transform with seasonal methods."""
    
    def test_demeanq_with_Q4(self):
        """Test demeanq with Q=4 (quarterly)."""
        np.random.seed(42)
        data = pd.DataFrame({
            'id': [1]*12 + [2]*12,
            'tindex': list(range(1, 13)) * 2,
            'quarter': [1,2,3,4]*3 * 2,
            'y': np.random.randn(24) + 100,
            'post': [0]*8 + [1]*4 + [0]*8 + [1]*4
        })
        
        result = apply_rolling_transform(
            data, y='y', ivar='id', tindex='tindex', post='post',
            rolling='demeanq', tpost1=9, season_var='quarter', Q=4
        )
        
        assert 'ydot' in result.columns
        assert 'ydot_postavg' in result.columns
        assert 'firstpost' in result.columns
    
    def test_detrendq_with_Q4(self):
        """Test detrendq with Q=4 (quarterly)."""
        np.random.seed(42)
        data = pd.DataFrame({
            'id': [1]*12 + [2]*12,
            'tindex': list(range(1, 13)) * 2,
            'quarter': [1,2,3,4]*3 * 2,
            'y': np.random.randn(24) + 100,
            'post': [0]*8 + [1]*4 + [0]*8 + [1]*4
        })
        
        result = apply_rolling_transform(
            data, y='y', ivar='id', tindex='tindex', post='post',
            rolling='detrendq', tpost1=9, season_var='quarter', Q=4
        )
        
        assert 'ydot' in result.columns
    
    def test_backward_compatibility_quarter_param(self):
        """Test backward compatibility with 'quarter' parameter."""
        np.random.seed(42)
        data = pd.DataFrame({
            'id': [1]*12,
            'tindex': list(range(1, 13)),
            'qtr': [1,2,3,4]*3,
            'y': np.random.randn(12) + 100,
            'post': [0]*8 + [1]*4
        })
        
        # Using 'quarter' parameter (old API)
        result = apply_rolling_transform(
            data, y='y', ivar='id', tindex='tindex', post='post',
            rolling='demeanq', tpost1=9, quarter='qtr'
        )
        
        assert 'ydot' in result.columns


# =============================================================================
# Numerical Verification Tests
# =============================================================================

class TestNumericalCorrectness:
    """Numerical verification tests using known values."""
    
    def test_demeanq_manual_calculation(self):
        """Verify demeanq against manual calculation."""
        # Simple case: 8 pre-treatment periods, 4 post
        # Known seasonal effects
        data = pd.DataFrame({
            'tindex': list(range(1, 13)),
            'quarter': [1,2,3,4,1,2,3,4,1,2,3,4],
            'y': [100, 105, 110, 103,  # Year 1
                  100, 105, 110, 103,  # Year 2 (same pattern)
                  100, 105, 110, 103], # Year 3 (post)
            'post': [0]*8 + [1]*4
        })
        
        yhat, ydot = demeanq_unit(data, 'y', 'quarter', 'post', Q=4)
        
        # With perfect seasonal pattern, residuals should be zero
        assert_allclose(ydot, 0, atol=1e-10)
    
    def test_detrendq_manual_calculation(self):
        """Verify detrendq against manual calculation."""
        # Linear trend + seasonal pattern
        data = pd.DataFrame({
            'tindex': list(range(1, 13)),
            'quarter': [1,2,3,4,1,2,3,4,1,2,3,4],
            'y': [100 + t + {1:0, 2:5, 3:10, 4:3}[((t-1)%4)+1] 
                  for t in range(1, 13)],
            'post': [0]*8 + [1]*4
        })
        
        yhat, ydot = detrendq_unit(data, 'y', 'tindex', 'quarter', 'post', Q=4)
        
        # With perfect trend + seasonal pattern, residuals should be zero
        assert_allclose(ydot, 0, atol=1e-8)


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_minimum_observations_demeanq(self):
        """Test with exactly minimum required observations for demeanq."""
        # Q=4 requires at least 5 pre-treatment observations
        data = pd.DataFrame({
            'tindex': [1, 2, 3, 4, 5, 6, 7],
            'quarter': [1, 2, 3, 4, 1, 2, 3],
            'y': [100, 105, 110, 103, 101, 106, 111],
            'post': [0, 0, 0, 0, 0, 1, 1]
        })
        
        yhat, ydot = demeanq_unit(data, 'y', 'quarter', 'post', Q=4)
        
        # Should produce valid output (not all NaN)
        assert not np.all(np.isnan(ydot))
    
    def test_missing_season_in_post(self):
        """Test when post-treatment has season not in pre-treatment."""
        data = pd.DataFrame({
            'id': [1]*6,
            'tindex': [1, 2, 3, 4, 5, 6],
            'quarter': [1, 2, 3, 1, 2, 4],  # Q4 only in post
            'y': [100, 105, 110, 101, 106, 103],
            'post': [0, 0, 0, 1, 1, 1]
        })
        
        # Should raise validation error
        with pytest.raises(InsufficientQuarterDiversityError):
            validate_season_coverage(data, 'id', 'quarter', 'post', Q=4)
    
    def test_single_unit_staggered(self):
        """Test staggered transformation with single treated unit."""
        data = pd.DataFrame({
            'id': [1]*12 + [2]*12,
            'tindex': list(range(1, 13)) * 2,
            'quarter': [1,2,3,4,1,2,3,4,1,2,3,4] * 2,
            'y': np.random.randn(24) + 100,
            'gvar': [7]*12 + [np.inf]*12  # Unit 1 treated at t=7, Unit 2 never
        })
        
        result = transform_staggered_demeanq(
            data, y='y', ivar='id', tvar='tindex', gvar='gvar',
            season_var='quarter', Q=4
        )
        
        # Should have columns for cohort 7
        assert 'ydot_g7_r7' in result.columns


# =============================================================================
# Test detect_frequency
# =============================================================================

class TestDetectFrequency:
    """Tests for automatic frequency detection."""
    
    def test_quarterly_detection_by_obs_per_year(self):
        """Test detection of quarterly data by observations per year."""
        from lwdid.validation import detect_frequency
        
        # Create quarterly data with year-like time variable
        np.random.seed(42)
        data = pd.DataFrame({
            'id': [1]*12 + [2]*12,
            'year': [2020]*4 + [2021]*4 + [2022]*4 + [2020]*4 + [2021]*4 + [2022]*4,
            'y': np.random.randn(24)
        })
        
        result = detect_frequency(data, tvar='year', ivar='id')
        
        assert result['frequency'] == 'quarterly'
        assert result['Q'] == 4
        assert result['confidence'] > 0.5
    
    def test_monthly_detection_by_obs_per_year(self):
        """Test detection of monthly data by observations per year."""
        from lwdid.validation import detect_frequency
        
        # Create monthly data
        np.random.seed(42)
        data = pd.DataFrame({
            'id': [1]*24 + [2]*24,
            'year': [2020]*12 + [2021]*12 + [2020]*12 + [2021]*12,
            'y': np.random.randn(48)
        })
        
        result = detect_frequency(data, tvar='year', ivar='id')
        
        assert result['frequency'] == 'monthly'
        assert result['Q'] == 12
        assert result['confidence'] > 0.5
    
    def test_annual_detection(self):
        """Test detection of annual data."""
        from lwdid.validation import detect_frequency
        
        # Create annual data
        np.random.seed(42)
        data = pd.DataFrame({
            'id': [1]*5 + [2]*5,
            'year': [2018, 2019, 2020, 2021, 2022] * 2,
            'y': np.random.randn(10)
        })
        
        result = detect_frequency(data, tvar='year', ivar='id')
        
        assert result['frequency'] == 'annual'
        assert result['Q'] == 1
        assert result['confidence'] > 0.5
    
    def test_datetime_quarterly_detection(self):
        """Test detection of quarterly data from datetime variable."""
        from lwdid.validation import detect_frequency
        
        # Create quarterly datetime data
        dates = pd.date_range('2020-01-01', periods=12, freq='QE')
        data = pd.DataFrame({
            'id': [1]*12,
            'date': dates,
            'y': np.random.randn(12)
        })
        
        result = detect_frequency(data, tvar='date', ivar='id')
        
        assert result['frequency'] == 'quarterly'
        assert result['Q'] == 4
    
    def test_datetime_monthly_detection(self):
        """Test detection of monthly data from datetime variable."""
        from lwdid.validation import detect_frequency
        
        # Create monthly datetime data
        dates = pd.date_range('2020-01-01', periods=24, freq='ME')
        data = pd.DataFrame({
            'id': [1]*24,
            'date': dates,
            'y': np.random.randn(24)
        })
        
        result = detect_frequency(data, tvar='date', ivar='id')
        
        assert result['frequency'] == 'monthly'
        assert result['Q'] == 12
    
    def test_insufficient_data(self):
        """Test handling of insufficient data for detection."""
        from lwdid.validation import detect_frequency
        
        # Single observation
        data = pd.DataFrame({
            'id': [1],
            'year': [2020],
            'y': [1.0]
        })
        
        with pytest.warns(UserWarning, match="Insufficient time values"):
            result = detect_frequency(data, tvar='year', ivar='id')
        
        assert result['frequency'] is None
        assert result['Q'] is None
    
    def test_missing_column(self):
        """Test handling of missing time column."""
        from lwdid.validation import detect_frequency
        
        data = pd.DataFrame({
            'id': [1, 2],
            'y': [1.0, 2.0]
        })
        
        with pytest.warns(UserWarning, match="not found in data"):
            result = detect_frequency(data, tvar='nonexistent', ivar='id')
        
        assert result['frequency'] is None
    
    def test_integer_index_ambiguous(self):
        """Test that consecutive integer index produces low confidence."""
        from lwdid.validation import detect_frequency
        
        # Consecutive integer time index (ambiguous)
        data = pd.DataFrame({
            'id': [1]*10,
            'tindex': list(range(1, 11)),
            'y': np.random.randn(10)
        })
        
        result = detect_frequency(data, tvar='tindex', ivar='id')
        
        # Should have low confidence for ambiguous case
        assert result['confidence'] <= 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


# =============================================================================
# Test auto_detect_frequency parameter in lwdid()
# =============================================================================

class TestAutoDetectFrequency:
    """Tests for auto_detect_frequency parameter in lwdid()."""
    
    def test_auto_detect_quarterly(self):
        """Test auto-detection of quarterly data."""
        from lwdid import lwdid
        
        np.random.seed(42)
        # Create quarterly panel data
        n_units = 10
        n_years = 5
        
        data = []
        for unit in range(1, n_units + 1):
            for year in range(2018, 2018 + n_years):
                for quarter in range(1, 5):
                    t = (year - 2018) * 4 + quarter
                    treated = 1 if unit <= 5 else 0
                    post = 1 if year >= 2021 else 0
                    y = 100 + unit * 5 + t * 0.5 + np.random.normal(0, 1)
                    if treated and post:
                        y += 10  # Treatment effect
                    data.append({
                        'id': unit,
                        'year': year,
                        'quarter': quarter,
                        'y': y,
                        'd': treated,
                        'post': post
                    })
        
        df = pd.DataFrame(data)
        
        # Test with auto_detect_frequency=True
        result = lwdid(
            df, y='y', d='d', ivar='id',
            tvar=['year', 'quarter'],
            post='post',
            rolling='demeanq',
            season_var='quarter',
            auto_detect_frequency=True
        )
        
        # Should complete without error
        assert result is not None
        assert hasattr(result, 'att')
    
    def test_auto_detect_with_explicit_Q_override(self):
        """Test that explicit Q overrides auto-detection."""
        from lwdid import lwdid
        
        np.random.seed(42)
        # Create quarterly data with explicit Q=4
        n_units = 10
        n_years = 5
        
        data = []
        for unit in range(1, n_units + 1):
            for year in range(2018, 2018 + n_years):
                for quarter in range(1, 5):
                    t = (year - 2018) * 4 + quarter
                    treated = 1 if unit <= 5 else 0
                    post = 1 if year >= 2021 else 0
                    y = 100 + unit * 5 + t * 0.2 + np.random.normal(0, 1)
                    if treated and post:
                        y += 10
                    data.append({
                        'id': unit,
                        'year': year,
                        'quarter': quarter,
                        'y': y,
                        'd': treated,
                        'post': post
                    })
        
        df = pd.DataFrame(data)
        
        # Explicit Q=4 should be used even with auto_detect_frequency=True
        result = lwdid(
            df, y='y', d='d', ivar='id',
            tvar=['year', 'quarter'],
            post='post',
            rolling='demeanq',
            season_var='quarter',
            Q=4,  # Explicit Q
            auto_detect_frequency=True
        )
        
        assert result is not None
    
    def test_auto_detect_disabled_by_default(self):
        """Test that auto_detect_frequency is disabled by default."""
        from lwdid import lwdid
        
        np.random.seed(42)
        n_units = 10
        n_periods = 20
        
        data = []
        for unit in range(1, n_units + 1):
            for t in range(1, n_periods + 1):
                year = 2018 + (t - 1) // 4
                quarter = ((t - 1) % 4) + 1
                treated = 1 if unit <= 5 else 0
                post = 1 if t > 12 else 0
                y = 100 + unit * 5 + t * 0.5 + np.random.normal(0, 1)
                if treated and post:
                    y += 10
                data.append({
                    'id': unit,
                    'year': year,
                    'quarter': quarter,
                    'y': y,
                    'd': treated,
                    'post': post
                })
        
        df = pd.DataFrame(data)
        
        # Default behavior (auto_detect_frequency=False)
        result = lwdid(
            df, y='y', d='d', ivar='id',
            tvar=['year', 'quarter'],
            post='post',
            rolling='demeanq',
            season_var='quarter'
        )
        
        assert result is not None
    
    def test_auto_detect_invalid_type(self):
        """Test that invalid auto_detect_frequency type raises error."""
        from lwdid import lwdid
        
        np.random.seed(42)
        data = pd.DataFrame({
            'id': [1]*8,
            'year': [2020]*4 + [2021]*4,
            'quarter': [1, 2, 3, 4, 1, 2, 3, 4],
            'y': np.random.randn(8) + 100,
            'd': [1]*8,
            'post': [0]*4 + [1]*4
        })
        
        with pytest.raises(TypeError, match="auto_detect_frequency.*boolean"):
            lwdid(
                data, y='y', d='d', ivar='id',
                tvar=['year', 'quarter'],
                post='post',
                rolling='demeanq',
                season_var='quarter',
                auto_detect_frequency='yes'  # Invalid type
            )
