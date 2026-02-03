"""
Unit tests for selection mechanism diagnostics.

Tests the selection_diagnostics module which implements diagnostic tools
for assessing potential selection bias in unbalanced panel data, following
Lee & Wooldridge (2025) Section 4.4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import warnings

from lwdid.selection_diagnostics import (
    diagnose_selection_mechanism,
    get_unit_missing_stats,
    plot_missing_pattern,
    MissingPattern,
    SelectionRisk,
    BalanceStatistics,
    AttritionAnalysis,
    SelectionDiagnostics,
    SelectionTestResult,
    UnitMissingStats,
    _validate_diagnostic_inputs,
    _is_never_treated,
    _compute_balance_statistics,
    _compute_attrition_analysis,
    _classify_missing_pattern,
    _assess_selection_risk,
    _compute_missing_rates,
    _compute_unit_stats,
)
from lwdid.exceptions import UnbalancedPanelError


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def balanced_panel_data():
    """Create balanced panel data with no missing observations."""
    np.random.seed(42)
    n_units = 50
    n_periods = 10
    
    data = []
    for i in range(n_units):
        gvar = 6 if i < 25 else 0  # Half treated at t=6
        for t in range(1, n_periods + 1):
            data.append({
                'unit_id': i,
                'year': t,
                'y': np.random.normal(10 + i * 0.1, 1),
                'gvar': gvar,
                'x1': np.random.normal(0, 1),
            })
    return pd.DataFrame(data)


@pytest.fixture
def unbalanced_panel_data():
    """Create unbalanced panel data with missing observations."""
    np.random.seed(42)
    n_units = 50
    n_periods = 10
    
    data = []
    for i in range(n_units):
        gvar = 6 if i < 25 else 0
        # Some units have missing periods
        missing_periods = set()
        if i % 5 == 0:  # 20% of units have missing data
            missing_periods = {np.random.randint(1, n_periods + 1) for _ in range(2)}
        
        for t in range(1, n_periods + 1):
            if t not in missing_periods:
                data.append({
                    'unit_id': i,
                    'year': t,
                    'y': np.random.normal(10 + i * 0.1, 1),
                    'gvar': gvar,
                    'x1': np.random.normal(0, 1),
                })
    return pd.DataFrame(data)


@pytest.fixture
def high_attrition_data():
    """Create data with high attrition rate (>50%)."""
    np.random.seed(42)
    
    data = []
    for i in range(100):
        gvar = 6 if i < 50 else 0
        # High attrition: only 30% of units have all periods
        n_periods = np.random.choice([3, 5, 10], p=[0.5, 0.2, 0.3])
        for t in range(1, n_periods + 1):
            data.append({
                'unit_id': i,
                'year': t,
                'y': np.random.randn(),
                'gvar': gvar,
            })
    
    return pd.DataFrame(data)


# =============================================================================
# Test MissingPattern Enum
# =============================================================================

class TestMissingPattern:
    """Tests for MissingPattern enum."""
    
    def test_enum_values(self):
        """Verify enum values match expected strings."""
        assert MissingPattern.MCAR.value == "missing_completely_at_random"
        assert MissingPattern.MAR.value == "missing_at_random"
        assert MissingPattern.MNAR.value == "missing_not_at_random"
        assert MissingPattern.UNKNOWN.value == "unknown"
    
    def test_enum_members(self):
        """Verify all expected enum members exist."""
        members = list(MissingPattern)
        assert len(members) == 4
        assert MissingPattern.MCAR in members
        assert MissingPattern.MAR in members
        assert MissingPattern.MNAR in members
        assert MissingPattern.UNKNOWN in members


# =============================================================================
# Test SelectionRisk Enum
# =============================================================================

class TestSelectionRisk:
    """Tests for SelectionRisk enum."""
    
    def test_enum_values(self):
        """Verify enum values match expected strings."""
        assert SelectionRisk.LOW.value == "low"
        assert SelectionRisk.MEDIUM.value == "medium"
        assert SelectionRisk.HIGH.value == "high"
        assert SelectionRisk.UNKNOWN.value == "unknown"
    
    def test_enum_members(self):
        """Verify all expected enum members exist."""
        members = list(SelectionRisk)
        assert len(members) == 4


# =============================================================================
# Test BalanceStatistics Dataclass
# =============================================================================

class TestBalanceStatistics:
    """Tests for BalanceStatistics dataclass."""
    
    def test_balanced_panel(self):
        """Balanced panel should have balance_ratio = 1.0."""
        stats = BalanceStatistics(
            is_balanced=True,
            n_units=100,
            n_periods=10,
            min_obs_per_unit=10,
            max_obs_per_unit=10,
            mean_obs_per_unit=10.0,
            std_obs_per_unit=0.0,
            balance_ratio=1.0,
        )
        assert stats.is_balanced
        assert stats.balance_ratio == 1.0
        assert stats.min_obs_per_unit == stats.max_obs_per_unit
    
    def test_unbalanced_panel(self):
        """Unbalanced panel should have balance_ratio < 1.0."""
        stats = BalanceStatistics(
            is_balanced=False,
            n_units=100,
            n_periods=10,
            min_obs_per_unit=5,
            max_obs_per_unit=10,
            mean_obs_per_unit=8.0,
            std_obs_per_unit=2.0,
            balance_ratio=0.5,
        )
        assert not stats.is_balanced
        assert stats.balance_ratio == 0.5
        assert stats.min_obs_per_unit < stats.max_obs_per_unit
    
    def test_method_usability_defaults(self):
        """Default method usability should be 100%."""
        stats = BalanceStatistics(
            is_balanced=True,
            n_units=100,
            n_periods=10,
            min_obs_per_unit=10,
            max_obs_per_unit=10,
            mean_obs_per_unit=10.0,
            std_obs_per_unit=0.0,
            balance_ratio=1.0,
        )
        assert stats.pct_usable_demean == 100.0
        assert stats.pct_usable_detrend == 100.0


# =============================================================================
# Test AttritionAnalysis Dataclass
# =============================================================================

class TestAttritionAnalysis:
    """Tests for AttritionAnalysis dataclass."""
    
    def test_no_attrition(self):
        """No attrition should have attrition_rate = 0."""
        analysis = AttritionAnalysis(
            n_units_complete=100,
            n_units_partial=0,
            attrition_rate=0.0,
        )
        assert analysis.attrition_rate == 0.0
        assert analysis.n_units_partial == 0
    
    def test_high_attrition(self):
        """High attrition should have attrition_rate > 0.5."""
        analysis = AttritionAnalysis(
            n_units_complete=30,
            n_units_partial=70,
            attrition_rate=0.7,
        )
        assert analysis.attrition_rate == 0.7
        assert analysis.n_units_partial > analysis.n_units_complete


# =============================================================================
# Test SelectionDiagnostics Dataclass
# =============================================================================

class TestSelectionDiagnostics:
    """Tests for SelectionDiagnostics dataclass."""
    
    def test_summary_generation(self, balanced_panel_data):
        """Summary should be a non-empty string with expected sections."""
        diag = diagnose_selection_mechanism(
            balanced_panel_data,
            y='y', ivar='unit_id', tvar='year', gvar='gvar',
            verbose=False
        )
        summary = diag.summary()
        
        assert isinstance(summary, str)
        assert len(summary) > 100
        assert "SELECTION MECHANISM DIAGNOSTICS" in summary
        assert "PANEL BALANCE" in summary
        assert "MISSING DATA" in summary
        assert "SELECTION RISK" in summary
    
    def test_to_dict(self, balanced_panel_data):
        """to_dict should return a dictionary with expected keys."""
        diag = diagnose_selection_mechanism(
            balanced_panel_data,
            y='y', ivar='unit_id', tvar='year', gvar='gvar',
            verbose=False
        )
        result = diag.to_dict()
        
        assert isinstance(result, dict)
        assert 'missing_pattern' in result
        assert 'selection_risk' in result
        assert 'balance_statistics' in result
        assert 'recommendations' in result


# =============================================================================
# Test Helper Functions
# =============================================================================

class TestValidateDiagnosticInputs:
    """Tests for _validate_diagnostic_inputs function."""
    
    def test_valid_inputs(self, balanced_panel_data):
        """Valid inputs should not raise."""
        _validate_diagnostic_inputs(
            balanced_panel_data, 'y', 'unit_id', 'year', 'gvar'
        )
    
    def test_missing_column(self, balanced_panel_data):
        """Missing column should raise ValueError."""
        with pytest.raises(ValueError, match="Missing required columns"):
            _validate_diagnostic_inputs(
                balanced_panel_data, 'nonexistent', 'unit_id', 'year', 'gvar'
            )
    
    def test_non_numeric_time(self, balanced_panel_data):
        """Non-numeric time variable should raise TypeError."""
        data = balanced_panel_data.copy()
        data['year'] = data['year'].astype(str)
        
        with pytest.raises(TypeError, match="must be numeric"):
            _validate_diagnostic_inputs(data, 'y', 'unit_id', 'year', 'gvar')
    
    def test_insufficient_data(self):
        """Insufficient data should raise ValueError."""
        data = pd.DataFrame({
            'unit_id': [1, 1],
            'year': [1, 2],
            'y': [1.0, 2.0],
        })
        
        with pytest.raises(ValueError, match="at least 2 unique units"):
            _validate_diagnostic_inputs(data, 'y', 'unit_id', 'year', None)


class TestIsNeverTreated:
    """Tests for _is_never_treated function."""
    
    def test_zero_is_never_treated(self):
        """Zero should be never-treated."""
        assert _is_never_treated(0, [0, np.inf])
    
    def test_inf_is_never_treated(self):
        """Infinity should be never-treated."""
        assert _is_never_treated(np.inf, [0, np.inf])
    
    def test_nan_is_never_treated(self):
        """NaN should be never-treated."""
        assert _is_never_treated(np.nan, [0, np.inf])
    
    def test_positive_int_is_treated(self):
        """Positive integer should be treated."""
        assert not _is_never_treated(6, [0, np.inf])


# =============================================================================
# Test Main Diagnostic Function
# =============================================================================

class TestDiagnoseSelectionMechanism:
    """Tests for diagnose_selection_mechanism function."""
    
    def test_balanced_panel_detection(self, balanced_panel_data):
        """Balanced panel should be correctly identified."""
        diag = diagnose_selection_mechanism(
            balanced_panel_data,
            y='y', ivar='unit_id', tvar='year', gvar='gvar',
            verbose=False
        )
        assert diag.balance_statistics.is_balanced
        assert diag.balance_statistics.balance_ratio == 1.0
    
    def test_unbalanced_panel_detection(self, unbalanced_panel_data):
        """Unbalanced panel should be correctly identified."""
        diag = diagnose_selection_mechanism(
            unbalanced_panel_data,
            y='y', ivar='unit_id', tvar='year', gvar='gvar',
            verbose=False
        )
        assert not diag.balance_statistics.is_balanced
        assert diag.balance_statistics.balance_ratio < 1.0
    
    def test_missing_rate_calculation(self, unbalanced_panel_data):
        """Missing rate should be correctly calculated."""
        diag = diagnose_selection_mechanism(
            unbalanced_panel_data,
            y='y', ivar='unit_id', tvar='year', gvar='gvar',
            verbose=False
        )
        assert 0 < diag.missing_rate_overall < 1
        assert len(diag.missing_rate_by_period) > 0
    
    def test_selection_risk_low_for_balanced(self, balanced_panel_data):
        """Balanced panel should have low selection risk."""
        diag = diagnose_selection_mechanism(
            balanced_panel_data,
            y='y', ivar='unit_id', tvar='year', gvar='gvar',
            verbose=False
        )
        assert diag.selection_risk == SelectionRisk.LOW
    
    def test_recommendations_provided(self, unbalanced_panel_data):
        """Unbalanced panel should generate recommendations."""
        diag = diagnose_selection_mechanism(
            unbalanced_panel_data,
            y='y', ivar='unit_id', tvar='year', gvar='gvar',
            verbose=False
        )
        assert len(diag.recommendations) > 0
    
    def test_high_attrition_detection(self, high_attrition_data):
        """High attrition should be detected."""
        diag = diagnose_selection_mechanism(
            high_attrition_data,
            y='y', ivar='unit_id', tvar='year', gvar='gvar',
            verbose=False
        )
        assert diag.attrition_analysis.attrition_rate > 0.5
    
    def test_verbose_output(self, balanced_panel_data, capsys):
        """verbose=True should print summary."""
        diagnose_selection_mechanism(
            balanced_panel_data,
            y='y', ivar='unit_id', tvar='year', gvar='gvar',
            verbose=True
        )
        captured = capsys.readouterr()
        assert "SELECTION MECHANISM DIAGNOSTICS" in captured.out
    
    def test_without_gvar(self, balanced_panel_data):
        """Should work without gvar (common timing)."""
        data = balanced_panel_data.drop(columns=['gvar'])
        diag = diagnose_selection_mechanism(
            data,
            y='y', ivar='unit_id', tvar='year', gvar=None,
            verbose=False
        )
        assert diag is not None
        assert diag.balance_statistics.is_balanced


# =============================================================================
# Test get_unit_missing_stats Function
# =============================================================================

class TestGetUnitMissingStats:
    """Tests for get_unit_missing_stats function."""
    
    def test_returns_dataframe(self, balanced_panel_data):
        """Should return a DataFrame."""
        stats_df = get_unit_missing_stats(
            balanced_panel_data,
            y='y', ivar='unit_id', tvar='year', gvar='gvar'
        )
        assert isinstance(stats_df, pd.DataFrame)
    
    def test_one_row_per_unit(self, balanced_panel_data):
        """Should have one row per unit."""
        stats_df = get_unit_missing_stats(
            balanced_panel_data,
            y='y', ivar='unit_id', tvar='year', gvar='gvar'
        )
        n_units = balanced_panel_data['unit_id'].nunique()
        assert len(stats_df) == n_units
    
    def test_expected_columns(self, balanced_panel_data):
        """Should have expected columns."""
        stats_df = get_unit_missing_stats(
            balanced_panel_data,
            y='y', ivar='unit_id', tvar='year', gvar='gvar'
        )
        expected_cols = [
            'unit_id', 'cohort', 'is_treated', 'n_observed', 'n_missing',
            'missing_rate', 'can_use_demean', 'can_use_detrend'
        ]
        for col in expected_cols:
            assert col in stats_df.columns
    
    def test_balanced_panel_no_missing(self, balanced_panel_data):
        """Balanced panel should have no missing observations."""
        stats_df = get_unit_missing_stats(
            balanced_panel_data,
            y='y', ivar='unit_id', tvar='year', gvar='gvar'
        )
        assert (stats_df['missing_rate'] == 0).all()
        assert (stats_df['can_use_demean']).all()
        assert (stats_df['can_use_detrend']).all()


# =============================================================================
# Test plot_missing_pattern Function
# =============================================================================

class TestPlotMissingPattern:
    """Tests for plot_missing_pattern function."""
    
    def test_returns_figure(self, balanced_panel_data):
        """Should return a matplotlib Figure."""
        import matplotlib.pyplot as plt
        
        fig = plot_missing_pattern(
            balanced_panel_data,
            ivar='unit_id', tvar='year', y='y', gvar='gvar'
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_sort_by_cohort(self, balanced_panel_data):
        """Should work with sort_by='cohort'."""
        import matplotlib.pyplot as plt
        
        fig = plot_missing_pattern(
            balanced_panel_data,
            ivar='unit_id', tvar='year', y='y', gvar='gvar',
            sort_by='cohort'
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_sort_by_missing_rate(self, unbalanced_panel_data):
        """Should work with sort_by='missing_rate'."""
        import matplotlib.pyplot as plt
        
        fig = plot_missing_pattern(
            unbalanced_panel_data,
            ivar='unit_id', tvar='year', y='y', gvar='gvar',
            sort_by='missing_rate'
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_max_units_sampling(self, balanced_panel_data):
        """Should sample when max_units is exceeded."""
        import matplotlib.pyplot as plt
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig = plot_missing_pattern(
                balanced_panel_data,
                ivar='unit_id', tvar='year', y='y',
                max_units=10
            )
            
            # Should warn about sampling
            assert any("sample" in str(warning.message).lower() for warning in w)
        
        plt.close(fig)


# =============================================================================
# Test balanced_panel Parameter
# =============================================================================

class TestBalancedPanelParameter:
    """Tests for balanced_panel parameter in lwdid()."""
    
    @pytest.fixture
    def unbalanced_data(self):
        """Create unbalanced panel data."""
        return pd.DataFrame({
            'unit_id': [1, 1, 1, 2, 2],  # Unit 2 has only 2 periods
            'year': [2000, 2001, 2002, 2000, 2001],
            'y': [1.0, 2.0, 3.0, 1.5, 2.5],
            'gvar': [2001, 2001, 2001, 0, 0],
        })
    
    def test_balanced_panel_error_raises(self, unbalanced_data):
        """balanced_panel='error' should raise for unbalanced data."""
        from lwdid import lwdid
        
        with pytest.raises(UnbalancedPanelError) as exc_info:
            lwdid(
                unbalanced_data,
                y='y', gvar='gvar', ivar='unit_id', tvar='year',
                balanced_panel='error'
            )
        
        assert exc_info.value.min_obs == 2
        assert exc_info.value.max_obs == 3
        assert exc_info.value.n_incomplete_units == 1
    
    def test_balanced_panel_warn_no_error(self, unbalanced_data):
        """balanced_panel='warn' should not raise UnbalancedPanelError."""
        from lwdid import lwdid
        
        # This may raise other errors, but not UnbalancedPanelError
        try:
            lwdid(
                unbalanced_data,
                y='y', gvar='gvar', ivar='unit_id', tvar='year',
                balanced_panel='warn'
            )
        except UnbalancedPanelError:
            pytest.fail("Should not raise UnbalancedPanelError with balanced_panel='warn'")
        except Exception:
            pass  # Other errors are OK for this test
    
    def test_invalid_balanced_panel_value(self, unbalanced_data):
        """Invalid balanced_panel value should raise ValueError."""
        from lwdid import lwdid
        
        with pytest.raises(ValueError, match="balanced_panel must be"):
            lwdid(
                unbalanced_data,
                y='y', gvar='gvar', ivar='unit_id', tvar='year',
                balanced_panel='invalid'
            )


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_missing_observation(self):
        """Test with exactly one missing observation."""
        data = pd.DataFrame({
            'unit_id': [1, 1, 1, 2, 2],  # Unit 2 missing one period
            'year': [1, 2, 3, 1, 2],
            'y': [1.0, 2.0, 3.0, 1.5, 2.5],
        })
        
        diag = diagnose_selection_mechanism(
            data, y='y', ivar='unit_id', tvar='year', verbose=False
        )
        
        assert not diag.balance_statistics.is_balanced
        assert diag.missing_rate_overall > 0
    
    def test_all_units_complete(self):
        """Test when all units have complete observations."""
        data = pd.DataFrame({
            'unit_id': [1, 1, 2, 2],
            'year': [1, 2, 1, 2],
            'y': [1.0, 2.0, 1.5, 2.5],
        })
        
        diag = diagnose_selection_mechanism(
            data, y='y', ivar='unit_id', tvar='year', verbose=False
        )
        
        assert diag.balance_statistics.is_balanced
        assert diag.missing_rate_overall == 0
    
    def test_late_entry_pattern(self):
        """Test with late entry pattern (units join after period 1)."""
        data = []
        for i in range(50):
            # Half of units enter late
            start_period = 1 if i < 25 else 5
            for t in range(start_period, 11):
                data.append({
                    'unit_id': i,
                    'year': t,
                    'y': np.random.randn(),
                })
        
        df = pd.DataFrame(data)
        
        diag = diagnose_selection_mechanism(
            df, y='y', ivar='unit_id', tvar='year', verbose=False
        )
        
        # Should detect late entry
        assert diag.attrition_analysis.late_entry_rate > 0.4
    
    def test_early_dropout_pattern(self):
        """Test with early dropout pattern (units leave before final period)."""
        data = []
        for i in range(50):
            # Half of units drop out early
            end_period = 10 if i < 25 else 5
            for t in range(1, end_period + 1):
                data.append({
                    'unit_id': i,
                    'year': t,
                    'y': np.random.randn(),
                })
        
        df = pd.DataFrame(data)
        
        diag = diagnose_selection_mechanism(
            df, y='y', ivar='unit_id', tvar='year', verbose=False
        )
        
        # Should detect early dropout
        assert diag.attrition_analysis.early_dropout_rate > 0.4
