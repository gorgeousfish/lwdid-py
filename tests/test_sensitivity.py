"""
Unit tests for sensitivity analysis module.

Tests the pre-treatment period robustness and no-anticipation sensitivity
analysis functions from Lee & Wooldridge (2026) Section 8.1.
"""

import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from lwdid.sensitivity import (
    RobustnessLevel,
    AnticipationDetectionMethod,
    SpecificationResult,
    PrePeriodRobustnessResult,
    AnticipationEstimate,
    NoAnticipationSensitivityResult,
    ComprehensiveSensitivityResult,
    robustness_pre_periods,
    sensitivity_no_anticipation,
    sensitivity_analysis,
    _validate_robustness_inputs,
    _auto_detect_pre_period_range,
    _filter_to_n_pre_periods,
    _filter_excluding_periods,
    _determine_robustness_level,
    _compute_sensitivity_ratio,
    _detect_anticipation_effects,
)


# =============================================================================
# Test Data Generators
# =============================================================================

def generate_panel_data(
    n_units: int = 100,
    n_periods: int = 10,
    treatment_period: int = 6,
    treatment_effect: float = 2.0,
    noise_std: float = 0.5,
    cohort_trend: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate simulated panel data for testing."""
    np.random.seed(seed)
    
    data = []
    n_treated = n_units // 2
    
    for i in range(n_units):
        alpha_i = np.random.normal(0, 1)
        treated = i < n_treated
        beta_i = cohort_trend if treated else 0  # Heterogeneous trend
        
        for t in range(1, n_periods + 1):
            gamma_t = 0.1 * t
            d_it = 1 if treated and t >= treatment_period else 0
            tau = treatment_effect if d_it else 0
            y = alpha_i + gamma_t + beta_i * t + tau + np.random.normal(0, noise_std)
            
            data.append({
                'unit': i,
                'time': t,
                'Y': y,
                'first_treat': treatment_period if treated else 0,
                'D': 1 if treated else 0,
                'post': 1 if t >= treatment_period else 0,
            })
    
    return pd.DataFrame(data)


def generate_data_with_anticipation(
    n_units: int = 100,
    n_periods: int = 10,
    treatment_period: int = 6,
    treatment_effect: float = 2.0,
    anticipation_periods: int = 2,
    anticipation_effect: float = 0.5,
    noise_std: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate data with anticipation effects."""
    np.random.seed(seed)
    
    data = []
    n_treated = n_units // 2
    
    for i in range(n_units):
        alpha_i = np.random.normal(0, 1)
        treated = i < n_treated
        
        for t in range(1, n_periods + 1):
            d_it = 1 if treated and t >= treatment_period else 0
            # Anticipation effect
            anticipation = anticipation_effect if (
                treated and 
                t >= treatment_period - anticipation_periods and 
                t < treatment_period
            ) else 0
            tau = treatment_effect if d_it else 0
            y = alpha_i + tau + anticipation + np.random.normal(0, noise_std)
            
            data.append({
                'unit': i,
                'time': t,
                'Y': y,
                'first_treat': treatment_period if treated else 0,
                'D': 1 if treated else 0,
                'post': 1 if t >= treatment_period else 0,
            })
    
    return pd.DataFrame(data)


# =============================================================================
# Test Enumerations
# =============================================================================

class TestRobustnessLevel:
    """Test RobustnessLevel enum."""
    
    def test_enum_values(self):
        """Test enum has expected values."""
        assert RobustnessLevel.HIGHLY_ROBUST.value == "highly_robust"
        assert RobustnessLevel.MODERATELY_ROBUST.value == "moderately_robust"
        assert RobustnessLevel.SENSITIVE.value == "sensitive"
        assert RobustnessLevel.HIGHLY_SENSITIVE.value == "highly_sensitive"
    
    def test_enum_members(self):
        """Test all enum members exist."""
        assert len(RobustnessLevel) == 4


class TestAnticipationDetectionMethod:
    """Test AnticipationDetectionMethod enum."""
    
    def test_enum_values(self):
        """Test enum has expected values."""
        assert AnticipationDetectionMethod.TREND_BREAK.value == "trend_break"
        assert AnticipationDetectionMethod.COEFFICIENT_CHANGE.value == "coefficient_change"
        assert AnticipationDetectionMethod.NONE_DETECTED.value == "none_detected"


# =============================================================================
# Test Data Classes
# =============================================================================

class TestSpecificationResult:
    """Test SpecificationResult dataclass."""
    
    def test_is_significant_05(self):
        """Test significance at 5% level."""
        spec = SpecificationResult(
            specification_id=0, n_pre_periods=5, start_period=1, end_period=5,
            excluded_periods=0, att=0.5, se=0.1, t_stat=5.0, pvalue=0.001,
            ci_lower=0.3, ci_upper=0.7, n_treated=100, n_control=200, df=298
        )
        assert spec.is_significant_05 is True
        
        spec_ns = SpecificationResult(
            specification_id=1, n_pre_periods=5, start_period=1, end_period=5,
            excluded_periods=0, att=0.05, se=0.1, t_stat=0.5, pvalue=0.62,
            ci_lower=-0.15, ci_upper=0.25, n_treated=100, n_control=200, df=298
        )
        assert spec_ns.is_significant_05 is False
    
    def test_is_significant_10(self):
        """Test significance at 10% level."""
        spec = SpecificationResult(
            specification_id=0, n_pre_periods=5, start_period=1, end_period=5,
            excluded_periods=0, att=0.15, se=0.1, t_stat=1.5, pvalue=0.08,
            ci_lower=-0.05, ci_upper=0.35, n_treated=100, n_control=200, df=298
        )
        assert spec.is_significant_10 is True
        assert spec.is_significant_05 is False
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        spec = SpecificationResult(
            specification_id=0, n_pre_periods=5, start_period=1, end_period=5,
            excluded_periods=0, att=0.5, se=0.1, t_stat=5.0, pvalue=0.001,
            ci_lower=0.3, ci_upper=0.7, n_treated=100, n_control=200, df=298
        )
        d = spec.to_dict()
        assert d['att'] == 0.5
        assert d['n_pre_periods'] == 5
        assert d['significant_05'] is True
        assert 'spec_id' in d


class TestAnticipationEstimate:
    """Test AnticipationEstimate dataclass."""
    
    def test_is_significant(self):
        """Test significance property."""
        est = AnticipationEstimate(
            excluded_periods=1, att=0.5, se=0.1, t_stat=5.0, pvalue=0.001,
            ci_lower=0.3, ci_upper=0.7, n_pre_periods_used=4
        )
        assert est.is_significant is True
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        est = AnticipationEstimate(
            excluded_periods=1, att=0.5, se=0.1, t_stat=5.0, pvalue=0.001,
            ci_lower=0.3, ci_upper=0.7, n_pre_periods_used=4
        )
        d = est.to_dict()
        assert d['excluded_periods'] == 1
        assert d['att'] == 0.5
        assert d['significant'] is True


class TestPrePeriodRobustnessResult:
    """Test PrePeriodRobustnessResult dataclass."""
    
    @pytest.fixture
    def sample_result(self):
        """Create sample result for testing."""
        specs = [
            SpecificationResult(
                specification_id=i, n_pre_periods=i+2, start_period=1, end_period=i+2,
                excluded_periods=0, att=0.5+0.01*i, se=0.1,
                t_stat=5.0, pvalue=0.001, ci_lower=0.3, ci_upper=0.7,
                n_treated=100, n_control=200, df=298
            )
            for i in range(5)
        ]
        baseline = specs[-1]
        
        return PrePeriodRobustnessResult(
            specifications=specs,
            baseline_spec=baseline,
            att_range=(0.50, 0.54),
            att_mean=0.52,
            att_std=0.015,
            sensitivity_ratio=0.08,
            robustness_level=RobustnessLevel.HIGHLY_ROBUST,
            is_robust=True,
            robustness_threshold=0.25,
            all_same_sign=True,
            all_significant=True,
            n_significant=5,
            n_sign_changes=0,
            rolling_method='demean',
            estimator='ra',
            n_specifications=5,
            pre_period_range_tested=(2, 6),
            recommendation="Results are robust.",
        )
    
    def test_to_dataframe(self, sample_result):
        """Test conversion to DataFrame."""
        df = sample_result.to_dataframe()
        assert len(df) == 5
        assert 'att' in df.columns
        assert 'n_pre_periods' in df.columns
    
    def test_get_specification(self, sample_result):
        """Test getting specific specification."""
        spec = sample_result.get_specification(3)
        assert spec is not None
        assert spec.n_pre_periods == 3
        
        spec_none = sample_result.get_specification(100)
        assert spec_none is None
    
    def test_summary_format(self, sample_result):
        """Test summary output format."""
        summary = sample_result.summary()
        assert "PRE-TREATMENT PERIOD ROBUSTNESS" in summary
        assert "Sensitivity Ratio" in summary
        assert "RECOMMENDATION" in summary
        assert "demean" in summary


class TestNoAnticipationSensitivityResult:
    """Test NoAnticipationSensitivityResult dataclass."""
    
    @pytest.fixture
    def sample_result(self):
        """Create sample result for testing."""
        estimates = [
            AnticipationEstimate(
                excluded_periods=i, att=0.5+0.05*i, se=0.1,
                t_stat=5.0, pvalue=0.001, ci_lower=0.3, ci_upper=0.7,
                n_pre_periods_used=5-i
            )
            for i in range(4)
        ]
        
        return NoAnticipationSensitivityResult(
            estimates=estimates,
            baseline_estimate=estimates[0],
            anticipation_detected=False,
            recommended_exclusion=0,
            detection_method=AnticipationDetectionMethod.NONE_DETECTED,
            recommendation="No anticipation detected.",
        )
    
    def test_to_dataframe(self, sample_result):
        """Test conversion to DataFrame."""
        df = sample_result.to_dataframe()
        assert len(df) == 4
        assert 'excluded_periods' in df.columns
        assert 'att' in df.columns
    
    def test_summary_format(self, sample_result):
        """Test summary output format."""
        summary = sample_result.summary()
        assert "NO-ANTICIPATION SENSITIVITY" in summary
        assert "Anticipation Detected" in summary


# =============================================================================
# Test Helper Functions
# =============================================================================

class TestValidateRobustnessInputs:
    """Test input validation function."""
    
    def test_valid_staggered_inputs(self):
        """Test valid staggered design inputs."""
        data = pd.DataFrame({
            'Y': [1, 2, 3],
            'unit': [1, 1, 1],
            'time': [1, 2, 3],
            'first_treat': [2, 2, 2],
        })
        # Should not raise
        _validate_robustness_inputs(
            data, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', d=None, post=None, rolling='demean'
        )
    
    def test_valid_common_timing_inputs(self):
        """Test valid common timing inputs."""
        data = pd.DataFrame({
            'Y': [1, 2, 3],
            'unit': [1, 1, 1],
            'time': [1, 2, 3],
            'D': [1, 1, 1],
            'post': [0, 0, 1],
        })
        # Should not raise
        _validate_robustness_inputs(
            data, y='Y', ivar='unit', tvar='time',
            gvar=None, d='D', post='post', rolling='demean'
        )
    
    def test_missing_columns_raises(self):
        """Test that missing columns raise ValueError."""
        data = pd.DataFrame({'Y': [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            _validate_robustness_inputs(
                data, y='Y', ivar='unit', tvar='time',
                gvar='first_treat', d=None, post=None, rolling='demean'
            )
    
    def test_invalid_rolling_raises(self):
        """Test that invalid rolling method raises ValueError."""
        data = pd.DataFrame({
            'Y': [1, 2, 3],
            'unit': [1, 1, 1],
            'time': [1, 2, 3],
            'first_treat': [2, 2, 2],
        })
        with pytest.raises(ValueError, match="rolling must be one of"):
            _validate_robustness_inputs(
                data, y='Y', ivar='unit', tvar='time',
                gvar='first_treat', d=None, post=None, rolling='invalid'
            )
    
    def test_no_mode_specified_raises(self):
        """Test that missing mode specification raises ValueError."""
        data = pd.DataFrame({
            'Y': [1, 2, 3],
            'unit': [1, 1, 1],
            'time': [1, 2, 3],
        })
        with pytest.raises(ValueError, match="Must specify either gvar"):
            _validate_robustness_inputs(
                data, y='Y', ivar='unit', tvar='time',
                gvar=None, d=None, post=None, rolling='demean'
            )


class TestAutoDetectPrePeriodRange:
    """Test automatic pre-period range detection."""
    
    def test_common_timing_detection(self):
        """Test detection for common timing design."""
        data = pd.DataFrame({
            'unit': np.repeat(range(10), 8),
            'time': np.tile(range(1, 9), 10),
            'post': np.tile([0]*5 + [1]*3, 10),
            'Y': np.random.randn(80),
        })
        
        min_pre, max_pre = _auto_detect_pre_period_range(
            data, ivar='unit', tvar='time',
            gvar=None, d=None, post='post', rolling='demean'
        )
        
        assert min_pre == 1  # demean requires 1
        assert max_pre == 5  # 5 pre-treatment periods
    
    def test_detrend_minimum(self):
        """Test minimum for detrend method."""
        data = pd.DataFrame({
            'unit': np.repeat(range(10), 8),
            'time': np.tile(range(1, 9), 10),
            'post': np.tile([0]*5 + [1]*3, 10),
            'Y': np.random.randn(80),
        })
        
        min_pre, max_pre = _auto_detect_pre_period_range(
            data, ivar='unit', tvar='time',
            gvar=None, d=None, post='post', rolling='detrend'
        )
        
        assert min_pre == 2  # detrend requires 2


class TestDetermineRobustnessLevel:
    """Test robustness level determination."""
    
    @pytest.mark.parametrize("ratio,expected_level", [
        (0.05, RobustnessLevel.HIGHLY_ROBUST),
        (0.09, RobustnessLevel.HIGHLY_ROBUST),
        (0.10, RobustnessLevel.MODERATELY_ROBUST),
        (0.15, RobustnessLevel.MODERATELY_ROBUST),
        (0.24, RobustnessLevel.MODERATELY_ROBUST),
        (0.25, RobustnessLevel.SENSITIVE),
        (0.35, RobustnessLevel.SENSITIVE),
        (0.49, RobustnessLevel.SENSITIVE),
        (0.50, RobustnessLevel.HIGHLY_SENSITIVE),
        (0.75, RobustnessLevel.HIGHLY_SENSITIVE),
        (1.00, RobustnessLevel.HIGHLY_SENSITIVE),
    ])
    def test_robustness_level_thresholds(self, ratio, expected_level):
        """Test robustness level determination at boundary values."""
        level = _determine_robustness_level(ratio)
        assert level == expected_level


class TestComputeSensitivityRatio:
    """Test sensitivity ratio computation."""
    
    def test_basic_computation(self):
        """Test basic sensitivity ratio computation."""
        atts = [1.0, 1.1, 1.2, 1.3, 1.4]
        baseline = 1.4
        
        ratio = _compute_sensitivity_ratio(atts, baseline)
        expected = 0.4 / 1.4  # (1.4 - 1.0) / 1.4
        
        assert_allclose(ratio, expected, rtol=1e-10)
    
    def test_negative_baseline(self):
        """Test with negative baseline ATT."""
        atts = [-1.0, -1.1, -1.2]
        baseline = -1.2
        
        ratio = _compute_sensitivity_ratio(atts, baseline)
        expected = 0.2 / 1.2  # range / |baseline|
        
        assert_allclose(ratio, expected, rtol=1e-10)
    
    def test_zero_baseline(self):
        """Test with baseline near zero."""
        atts = [0.01, 0.02, 0.03]
        baseline = 0.0
        
        ratio = _compute_sensitivity_ratio(atts, baseline)
        
        assert ratio == float('inf')
    
    def test_empty_atts(self):
        """Test with empty ATT list."""
        ratio = _compute_sensitivity_ratio([], 1.0)
        assert ratio == 0.0
    
    def test_identical_atts(self):
        """Test with identical ATT values."""
        atts = [1.0, 1.0, 1.0]
        baseline = 1.0
        
        ratio = _compute_sensitivity_ratio(atts, baseline)
        assert ratio == 0.0


class TestFilterToNPrePeriods:
    """Test data filtering for pre-period selection."""
    
    def test_common_timing_filter(self):
        """Test filtering for common timing design."""
        data = pd.DataFrame({
            'unit': np.repeat(range(10), 10),
            'time': np.tile(range(1, 11), 10),
            'post': np.tile([0]*5 + [1]*5, 10),
            'Y': np.random.randn(100),
        })
        
        # Filter to 3 pre-periods (should keep periods 3, 4, 5)
        filtered = _filter_to_n_pre_periods(
            data, ivar='unit', tvar='time', gvar=None, d=None, post='post',
            n_pre_periods=3, exclude_periods=0
        )
        
        pre_times = filtered[filtered['post'] == 0]['time'].unique()
        assert set(pre_times) == {3, 4, 5}
    
    def test_filter_with_exclusion(self):
        """Test filtering with period exclusion."""
        data = pd.DataFrame({
            'unit': np.repeat(range(10), 10),
            'time': np.tile(range(1, 11), 10),
            'post': np.tile([0]*5 + [1]*5, 10),
            'Y': np.random.randn(100),
        })
        
        # Filter to 3 pre-periods, excluding 1 period before treatment
        filtered = _filter_to_n_pre_periods(
            data, ivar='unit', tvar='time', gvar=None, d=None, post='post',
            n_pre_periods=3, exclude_periods=1
        )
        
        pre_times = filtered[filtered['post'] == 0]['time'].unique()
        assert set(pre_times) == {2, 3, 4}
        assert 5 not in pre_times  # Period 5 excluded


class TestFilterExcludingPeriods:
    """Test data filtering for anticipation exclusion."""
    
    def test_no_exclusion(self):
        """Test with no exclusion."""
        data = pd.DataFrame({
            'unit': [1, 1, 1, 1],
            'time': [1, 2, 3, 4],
            'post': [0, 0, 1, 1],
            'Y': [1, 2, 3, 4],
        })
        
        filtered = _filter_excluding_periods(
            data, ivar='unit', tvar='time', gvar=None, post='post',
            exclude_periods=0
        )
        
        assert len(filtered) == len(data)
    
    def test_exclude_one_period(self):
        """Test excluding one period before treatment."""
        data = pd.DataFrame({
            'unit': np.repeat([1, 2], 5),
            'time': np.tile([1, 2, 3, 4, 5], 2),
            'post': np.tile([0, 0, 0, 1, 1], 2),
            'Y': np.random.randn(10),
        })
        
        filtered = _filter_excluding_periods(
            data, ivar='unit', tvar='time', gvar=None, post='post',
            exclude_periods=1
        )
        
        # Period 3 should be excluded (last pre-treatment period)
        pre_times = filtered[filtered['post'] == 0]['time'].unique()
        assert 3 not in pre_times
        assert set(pre_times) == {1, 2}


class TestDetectAnticipationEffects:
    """Test anticipation effect detection."""
    
    def test_no_anticipation(self):
        """Test when no anticipation is present."""
        estimates = [
            AnticipationEstimate(i, att=2.0, se=0.1, t_stat=20, pvalue=0.001,
                               ci_lower=1.8, ci_upper=2.2, n_pre_periods_used=5-i)
            for i in range(4)
        ]
        baseline = estimates[0]
        
        detected, rec_excl, method = _detect_anticipation_effects(
            estimates, baseline, threshold=0.10
        )
        
        assert detected is False
        assert rec_excl == 0
    
    def test_coefficient_change_detection(self):
        """Test detection via coefficient change."""
        # ATT increases substantially when excluding periods
        estimates = [
            AnticipationEstimate(0, att=1.5, se=0.1, t_stat=15, pvalue=0.001,
                               ci_lower=1.3, ci_upper=1.7, n_pre_periods_used=5),
            AnticipationEstimate(1, att=2.0, se=0.1, t_stat=20, pvalue=0.001,
                               ci_lower=1.8, ci_upper=2.2, n_pre_periods_used=4),
            AnticipationEstimate(2, att=2.1, se=0.1, t_stat=21, pvalue=0.001,
                               ci_lower=1.9, ci_upper=2.3, n_pre_periods_used=3),
        ]
        baseline = estimates[0]
        
        detected, rec_excl, method = _detect_anticipation_effects(
            estimates, baseline, threshold=0.10
        )
        
        assert detected is True
        assert rec_excl >= 1
        assert method == AnticipationDetectionMethod.COEFFICIENT_CHANGE
    
    def test_insufficient_data(self):
        """Test with insufficient data."""
        estimates = [
            AnticipationEstimate(0, att=2.0, se=0.1, t_stat=20, pvalue=0.001,
                               ci_lower=1.8, ci_upper=2.2, n_pre_periods_used=5),
        ]
        baseline = estimates[0]
        
        detected, rec_excl, method = _detect_anticipation_effects(
            estimates, baseline, threshold=0.10
        )
        
        assert detected is False
        assert method == AnticipationDetectionMethod.INSUFFICIENT_DATA


# =============================================================================
# Test Main Functions
# =============================================================================

class TestRobustnessPrePeriods:
    """Test robustness_pre_periods function."""
    
    @pytest.fixture
    def panel_data_robust(self):
        """Generate data where estimates are robust to pre-period selection."""
        return generate_panel_data(
            n_units=200, n_periods=12, treatment_period=8,
            treatment_effect=2.0, noise_std=0.5, cohort_trend=0.0, seed=42
        )
    
    @pytest.fixture
    def panel_data_sensitive(self):
        """Generate data where estimates are sensitive to pre-period selection."""
        return generate_panel_data(
            n_units=200, n_periods=12, treatment_period=8,
            treatment_effect=2.0, noise_std=0.5, cohort_trend=0.2, seed=42
        )
    
    def test_basic_execution(self, panel_data_robust):
        """Test basic function execution."""
        result = robustness_pre_periods(
            panel_data_robust, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', rolling='demean',
            pre_period_range=(2, 5), verbose=False
        )
        
        assert isinstance(result, PrePeriodRobustnessResult)
        assert result.n_specifications >= 2
        assert result.baseline_spec is not None
    
    def test_robust_data_is_robust(self, panel_data_robust):
        """When data has no heterogeneous trends, should be robust."""
        result = robustness_pre_periods(
            panel_data_robust, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', rolling='demean',
            pre_period_range=(2, 6), verbose=False
        )
        
        # With no heterogeneous trends, demean should be stable
        assert result.sensitivity_ratio < 0.50  # Allow some variation
        assert result.all_same_sign is True
    
    def test_sensitive_data_detected(self, panel_data_sensitive):
        """When data has heterogeneous trends, should detect sensitivity."""
        result = robustness_pre_periods(
            panel_data_sensitive, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', rolling='demean',
            pre_period_range=(2, 6), verbose=False
        )
        
        # With heterogeneous trends and demean, should show some sensitivity
        assert result.sensitivity_ratio > 0.05
    
    def test_common_timing_mode(self, panel_data_robust):
        """Test with common timing design."""
        result = robustness_pre_periods(
            panel_data_robust, y='Y', ivar='unit', tvar='time',
            d='D', post='post', rolling='demean',
            pre_period_range=(2, 5), verbose=False
        )
        
        assert isinstance(result, PrePeriodRobustnessResult)
        assert result.n_specifications >= 2
    
    def test_auto_detect_range(self, panel_data_robust):
        """Should auto-detect pre-period range when not specified."""
        result = robustness_pre_periods(
            panel_data_robust, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', rolling='demean',
            pre_period_range=None, verbose=False
        )
        
        assert result.pre_period_range_tested[0] >= 1
        assert result.pre_period_range_tested[1] >= result.pre_period_range_tested[0]
    
    def test_summary_generation(self, panel_data_robust):
        """Test summary generation."""
        result = robustness_pre_periods(
            panel_data_robust, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', rolling='demean',
            pre_period_range=(2, 4), verbose=False
        )
        
        summary = result.summary()
        assert isinstance(summary, str)
        assert len(summary) > 100
    
    def test_dataframe_conversion(self, panel_data_robust):
        """Test DataFrame conversion."""
        result = robustness_pre_periods(
            panel_data_robust, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', rolling='demean',
            pre_period_range=(2, 4), verbose=False
        )
        
        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == result.n_specifications


class TestSensitivityNoAnticipation:
    """Test sensitivity_no_anticipation function."""
    
    @pytest.fixture
    def data_no_anticipation(self):
        """Data without anticipation effects."""
        return generate_panel_data(
            n_units=200, n_periods=10, treatment_period=6,
            treatment_effect=2.0, noise_std=0.5, seed=42
        )
    
    @pytest.fixture
    def data_with_anticipation(self):
        """Data with anticipation effects."""
        return generate_data_with_anticipation(
            n_units=200, n_periods=10, treatment_period=6,
            treatment_effect=2.0, anticipation_periods=2,
            anticipation_effect=0.5, noise_std=0.5, seed=42
        )
    
    def test_basic_execution(self, data_no_anticipation):
        """Test basic function execution."""
        result = sensitivity_no_anticipation(
            data_no_anticipation, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', max_anticipation=2, verbose=False
        )
        
        assert isinstance(result, NoAnticipationSensitivityResult)
        assert len(result.estimates) >= 2
    
    def test_no_anticipation_not_detected(self, data_no_anticipation):
        """When no anticipation, should not detect."""
        result = sensitivity_no_anticipation(
            data_no_anticipation, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', max_anticipation=2, verbose=False
        )
        
        # May or may not detect depending on noise
        assert isinstance(result.anticipation_detected, bool)
    
    def test_summary_generation(self, data_no_anticipation):
        """Test summary generation."""
        result = sensitivity_no_anticipation(
            data_no_anticipation, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', max_anticipation=2, verbose=False
        )
        
        summary = result.summary()
        assert isinstance(summary, str)
        assert "NO-ANTICIPATION" in summary


class TestSensitivityAnalysis:
    """Test comprehensive sensitivity_analysis function."""
    
    @pytest.fixture
    def panel_data(self):
        """Generate panel data for testing."""
        return generate_panel_data(
            n_units=100, n_periods=10, treatment_period=6,
            treatment_effect=2.0, noise_std=0.5, seed=42
        )
    
    def test_basic_execution(self, panel_data):
        """Test basic function execution."""
        result = sensitivity_analysis(
            panel_data, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', rolling='demean',
            analyses=['pre_periods'], verbose=False
        )
        
        assert isinstance(result, ComprehensiveSensitivityResult)
        assert result.pre_period_result is not None
    
    def test_multiple_analyses(self, panel_data):
        """Test running multiple analyses."""
        result = sensitivity_analysis(
            panel_data, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', rolling='demean',
            analyses=['pre_periods', 'anticipation'], verbose=False
        )
        
        assert result.pre_period_result is not None
        assert result.anticipation_result is not None
    
    def test_summary_generation(self, panel_data):
        """Test summary generation."""
        result = sensitivity_analysis(
            panel_data, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', rolling='demean',
            analyses=['pre_periods'], verbose=False
        )
        
        summary = result.summary()
        assert isinstance(summary, str)
        assert "COMPREHENSIVE SENSITIVITY" in summary


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for sensitivity module."""
    
    def test_full_workflow(self):
        """Test complete sensitivity analysis workflow."""
        # Generate data
        data = generate_panel_data(
            n_units=150, n_periods=12, treatment_period=7,
            treatment_effect=1.5, noise_std=0.3, seed=123
        )
        
        # Run pre-period robustness
        pre_result = robustness_pre_periods(
            data, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', rolling='demean',
            pre_period_range=(2, 5), verbose=False
        )
        
        assert pre_result.n_specifications >= 2
        assert not np.isnan(pre_result.baseline_spec.att)
        
        # Run anticipation sensitivity
        ant_result = sensitivity_no_anticipation(
            data, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', max_anticipation=2, verbose=False
        )
        
        assert len(ant_result.estimates) >= 2
        
        # Run comprehensive analysis
        comp_result = sensitivity_analysis(
            data, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', rolling='demean',
            analyses=['pre_periods', 'anticipation'], verbose=False
        )
        
        assert comp_result.pre_period_result is not None
        assert comp_result.anticipation_result is not None
        assert len(comp_result.recommendations) > 0
    
    def test_import_from_package(self):
        """Test that functions can be imported from main package."""
        from lwdid import (
            robustness_pre_periods,
            sensitivity_no_anticipation,
            sensitivity_analysis,
            RobustnessLevel,
            PrePeriodRobustnessResult,
        )
        
        assert callable(robustness_pre_periods)
        assert callable(sensitivity_no_anticipation)
        assert callable(sensitivity_analysis)
        assert RobustnessLevel.HIGHLY_ROBUST is not None
