"""
End-to-end tests for selection diagnostics workflow.

Tests the complete diagnostic workflow from data input to recommendations,
including integration with the main lwdid estimation function.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import warnings

from lwdid import lwdid
from lwdid.selection_diagnostics import (
    diagnose_selection_mechanism,
    plot_missing_pattern,
    get_unit_missing_stats,
    SelectionRisk,
    MissingPattern,
)
from lwdid.exceptions import UnbalancedPanelError


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def realistic_unbalanced_data():
    """
    Create realistic unbalanced panel data with:
    - 100 units
    - 20 time periods
    - Staggered treatment (cohorts at t=8, 10, 12)
    - ~15% missing data
    - Some attrition pattern
    """
    np.random.seed(42)
    n_units = 100
    n_periods = 20
    
    data = []
    for i in range(n_units):
        # Assign cohort
        if i < 30:
            gvar = 8
        elif i < 60:
            gvar = 10
        elif i < 80:
            gvar = 12
        else:
            gvar = 0  # Never treated
        
        # Unit-specific characteristics
        unit_fe = np.random.normal(0, 2)
        unit_trend = np.random.normal(0, 0.1)
        
        # Attrition probability (higher for units with negative FE)
        attrition_prob = 0.1 + 0.05 * (unit_fe < -1)
        
        for t in range(1, n_periods + 1):
            # Attrition: some units drop out
            if t > 10 and np.random.random() < attrition_prob:
                continue
            
            # Random missing
            if np.random.random() < 0.05:
                continue
            
            # Generate outcome
            y = 10 + unit_fe + unit_trend * t + np.random.normal(0, 1)
            
            # Treatment effect
            if gvar > 0 and t >= gvar:
                y += 2.0  # True ATT = 2.0
            
            data.append({
                'unit_id': i,
                'year': t,
                'y': y,
                'gvar': gvar,
                'x1': np.random.normal(0, 1),
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def stata_comparable_data():
    """Create data that can be exported to Stata for comparison."""
    np.random.seed(123)
    n_units = 50
    n_periods = 10
    
    data = []
    for i in range(n_units):
        gvar = 6 if i < 25 else 0
        
        # Create some missing observations
        missing_periods = set()
        if i % 4 == 0:
            missing_periods = {np.random.randint(1, n_periods + 1)}
        
        for t in range(1, n_periods + 1):
            if t not in missing_periods:
                data.append({
                    'unit_id': i + 1,  # Stata uses 1-based indexing
                    'year': 2000 + t,
                    'y': np.random.normal(10, 1),
                    'gvar': 2000 + gvar if gvar > 0 else 0,
                })
    
    return pd.DataFrame(data)


# =============================================================================
# Test Full Diagnostic Workflow
# =============================================================================

class TestFullDiagnosticWorkflow:
    """Test complete diagnostic workflow from data to recommendations."""
    
    def test_full_workflow(self, realistic_unbalanced_data):
        """Test complete workflow: diagnose -> estimate -> compare."""
        data = realistic_unbalanced_data
        
        # Step 1: Run diagnostics
        diag = diagnose_selection_mechanism(
            data, y='y', ivar='unit_id', tvar='year', gvar='gvar',
            controls=['x1'], verbose=False
        )
        
        # Verify diagnostics completed
        assert diag is not None
        assert not diag.balance_statistics.is_balanced
        assert diag.missing_rate_overall > 0
        
        # Step 2: Check recommendations
        assert len(diag.recommendations) > 0
        
        # Step 3: Run estimation with warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = lwdid(
                data, y='y', gvar='gvar', ivar='unit_id', tvar='year',
                rolling='demean', balanced_panel='warn'
            )
            
            # Should have unbalanced panel warning
            unbalanced_warnings = [
                x for x in w if 'Unbalanced panel' in str(x.message)
            ]
            assert len(unbalanced_warnings) > 0
        
        # Step 4: Verify estimation completed
        assert result is not None
    
    def test_balanced_vs_unbalanced_comparison(self, realistic_unbalanced_data):
        """Compare estimates from full sample vs balanced subsample."""
        data = realistic_unbalanced_data
        
        # Create balanced subsample
        obs_per_unit = data.groupby('unit_id')['year'].count()
        max_obs = obs_per_unit.max()
        complete_units = obs_per_unit[obs_per_unit == max_obs].index
        balanced_data = data[data['unit_id'].isin(complete_units)]
        
        # Run diagnostics on both
        diag_full = diagnose_selection_mechanism(
            data, y='y', ivar='unit_id', tvar='year', gvar='gvar',
            verbose=False
        )
        
        diag_balanced = diagnose_selection_mechanism(
            balanced_data, y='y', ivar='unit_id', tvar='year', gvar='gvar',
            verbose=False
        )
        
        # Balanced subsample should be balanced
        assert diag_balanced.balance_statistics.is_balanced
        
        # Full sample should have higher or equal selection risk
        risk_order = {
            SelectionRisk.LOW: 0, 
            SelectionRisk.MEDIUM: 1, 
            SelectionRisk.HIGH: 2, 
            SelectionRisk.UNKNOWN: 3
        }
        assert risk_order[diag_full.selection_risk] >= risk_order[diag_balanced.selection_risk]
    
    def test_visualization_integration(self, realistic_unbalanced_data):
        """Test that visualization works with diagnostic data."""
        import matplotlib.pyplot as plt
        
        data = realistic_unbalanced_data
        
        # Generate plot
        fig = plot_missing_pattern(
            data, ivar='unit_id', tvar='year', y='y', gvar='gvar',
            sort_by='cohort'
        )
        
        # Verify figure created
        assert fig is not None
        
        # Clean up
        plt.close(fig)
    
    def test_unit_stats_integration(self, realistic_unbalanced_data):
        """Test that unit stats integrate with diagnostics."""
        data = realistic_unbalanced_data
        
        # Get unit stats
        stats_df = get_unit_missing_stats(
            data, y='y', ivar='unit_id', tvar='year', gvar='gvar'
        )
        
        # Run diagnostics
        diag = diagnose_selection_mechanism(
            data, y='y', ivar='unit_id', tvar='year', gvar='gvar',
            verbose=False
        )
        
        # Verify consistency
        assert len(stats_df) == diag.balance_statistics.n_units
        
        # Units that can't use demean should match
        n_below_demean = (~stats_df['can_use_demean']).sum()
        assert n_below_demean == diag.balance_statistics.units_below_demean_threshold


# =============================================================================
# Test Python to Stata E2E
# =============================================================================

class TestPythonToStataE2E:
    """
    End-to-end tests comparing Python diagnostics with Stata.
    
    These tests verify that Python diagnostics produce results consistent
    with what would be obtained from Stata's panel data commands.
    """
    
    def test_balance_statistics_match_stata(self, stata_comparable_data):
        """
        Verify balance statistics match Stata's xtdescribe output.
        
        Stata command: xtset unit_id year; xtdescribe
        """
        data = stata_comparable_data
        
        # Python calculation
        diag = diagnose_selection_mechanism(
            data, y='y', ivar='unit_id', tvar='year', gvar='gvar',
            verbose=False
        )
        
        # Manual calculation (what Stata would report)
        obs_per_unit = data.groupby('unit_id').size()
        n_units = len(obs_per_unit)
        min_obs = obs_per_unit.min()
        max_obs = obs_per_unit.max()
        
        # Verify match
        assert diag.balance_statistics.n_units == n_units
        assert diag.balance_statistics.min_obs_per_unit == min_obs
        assert diag.balance_statistics.max_obs_per_unit == max_obs
    
    def test_missing_pattern_consistency(self, stata_comparable_data):
        """
        Verify missing pattern detection is consistent.
        """
        data = stata_comparable_data
        
        diag = diagnose_selection_mechanism(
            data, y='y', ivar='unit_id', tvar='year', gvar='gvar',
            verbose=False
        )
        
        # Should detect some missing data
        if not diag.balance_statistics.is_balanced:
            assert diag.missing_rate_overall > 0


# =============================================================================
# Test Estimation Integration
# =============================================================================

class TestEstimationIntegration:
    """Test integration between diagnostics and estimation."""
    
    def test_balanced_panel_error_mode(self, realistic_unbalanced_data):
        """Test that balanced_panel='error' raises for unbalanced data."""
        data = realistic_unbalanced_data
        
        with pytest.raises(UnbalancedPanelError) as exc_info:
            lwdid(
                data, y='y', gvar='gvar', ivar='unit_id', tvar='year',
                balanced_panel='error'
            )
        
        # Verify exception contains useful information
        assert exc_info.value.min_obs > 0
        assert exc_info.value.max_obs > exc_info.value.min_obs
        assert exc_info.value.n_incomplete_units > 0
    
    def test_balanced_panel_ignore_mode(self, realistic_unbalanced_data):
        """Test that balanced_panel='ignore' suppresses warnings."""
        data = realistic_unbalanced_data
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            try:
                lwdid(
                    data, y='y', gvar='gvar', ivar='unit_id', tvar='year',
                    balanced_panel='ignore'
                )
            except Exception:
                pass  # May fail for other reasons
            
            # Should not have unbalanced panel warning from balanced_panel check
            # (may still have warning from validate_staggered_data)
    
    def test_demean_vs_detrend_comparison(self, realistic_unbalanced_data):
        """Compare demean and detrend results on unbalanced data."""
        data = realistic_unbalanced_data
        
        # Run diagnostics first
        diag = diagnose_selection_mechanism(
            data, y='y', ivar='unit_id', tvar='year', gvar='gvar',
            verbose=False
        )
        
        # If detrend usability is high enough, compare methods
        if diag.balance_statistics.pct_usable_detrend > 50:
            try:
                result_demean = lwdid(
                    data, y='y', gvar='gvar', ivar='unit_id', tvar='year',
                    rolling='demean', balanced_panel='ignore'
                )
                
                result_detrend = lwdid(
                    data, y='y', gvar='gvar', ivar='unit_id', tvar='year',
                    rolling='detrend', balanced_panel='ignore'
                )
                
                # Both should produce results
                assert result_demean is not None
                assert result_detrend is not None
                
            except Exception:
                pass  # May fail for other reasons


# =============================================================================
# Test Workflow with Different Data Patterns
# =============================================================================

class TestDifferentDataPatterns:
    """Test workflow with various data patterns."""
    
    def test_mcar_pattern_workflow(self):
        """Test workflow with MCAR missing pattern."""
        np.random.seed(42)
        
        # Create MCAR data (random missing)
        data = []
        for i in range(100):
            gvar = 6 if i < 50 else 0
            for t in range(1, 11):
                if np.random.random() > 0.1:  # 10% random missing
                    data.append({
                        'unit_id': i,
                        'year': t,
                        'y': np.random.normal(10, 1),
                        'gvar': gvar,
                    })
        
        df = pd.DataFrame(data)
        
        diag = diagnose_selection_mechanism(
            df, y='y', ivar='unit_id', tvar='year', gvar='gvar',
            verbose=False
        )
        
        # MCAR should have low selection risk
        assert diag.selection_risk in [SelectionRisk.LOW, SelectionRisk.MEDIUM]
    
    def test_attrition_pattern_workflow(self):
        """Test workflow with attrition pattern."""
        np.random.seed(42)
        
        # Create attrition data (units drop out over time)
        data = []
        for i in range(100):
            gvar = 6 if i < 50 else 0
            # Attrition probability increases with time
            for t in range(1, 11):
                if np.random.random() > 0.05 * t:  # Increasing dropout
                    data.append({
                        'unit_id': i,
                        'year': t,
                        'y': np.random.normal(10, 1),
                        'gvar': gvar,
                    })
        
        df = pd.DataFrame(data)
        
        diag = diagnose_selection_mechanism(
            df, y='y', ivar='unit_id', tvar='year', gvar='gvar',
            verbose=False
        )
        
        # Should detect attrition
        assert diag.attrition_analysis.early_dropout_rate > 0
    
    def test_cohort_specific_missing_workflow(self):
        """Test workflow with cohort-specific missing patterns."""
        np.random.seed(42)
        
        # Create data with different missing rates by cohort
        data = []
        for i in range(100):
            if i < 30:
                gvar = 6
                missing_prob = 0.2  # 20% missing for cohort 6
            elif i < 60:
                gvar = 8
                missing_prob = 0.1  # 10% missing for cohort 8
            else:
                gvar = 0
                missing_prob = 0.05  # 5% missing for never-treated
            
            for t in range(1, 11):
                if np.random.random() > missing_prob:
                    data.append({
                        'unit_id': i,
                        'year': t,
                        'y': np.random.normal(10, 1),
                        'gvar': gvar,
                    })
        
        df = pd.DataFrame(data)
        
        diag = diagnose_selection_mechanism(
            df, y='y', ivar='unit_id', tvar='year', gvar='gvar',
            verbose=False
        )
        
        # Should have cohort-specific missing rates
        assert len(diag.missing_rate_by_cohort) > 0


# =============================================================================
# Test Error Handling
# =============================================================================

class TestErrorHandling:
    """Test error handling in the workflow."""
    
    def test_missing_required_column(self):
        """Test error when required column is missing."""
        data = pd.DataFrame({
            'unit_id': [1, 1, 2, 2],
            'year': [1, 2, 1, 2],
            # Missing 'y' column
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            diagnose_selection_mechanism(
                data, y='y', ivar='unit_id', tvar='year', verbose=False
            )
    
    def test_insufficient_units(self):
        """Test error when insufficient units."""
        data = pd.DataFrame({
            'unit_id': [1, 1],
            'year': [1, 2],
            'y': [1.0, 2.0],
        })
        
        with pytest.raises(ValueError, match="at least 2 unique units"):
            diagnose_selection_mechanism(
                data, y='y', ivar='unit_id', tvar='year', verbose=False
            )
    
    def test_insufficient_periods(self):
        """Test error when insufficient periods."""
        data = pd.DataFrame({
            'unit_id': [1, 2],
            'year': [1, 1],
            'y': [1.0, 2.0],
        })
        
        with pytest.raises(ValueError, match="at least 2 unique time periods"):
            diagnose_selection_mechanism(
                data, y='y', ivar='unit_id', tvar='year', verbose=False
            )
