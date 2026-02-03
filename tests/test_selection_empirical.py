"""
Empirical data tests for selection diagnostics.

Uses real datasets to verify diagnostic functionality in realistic settings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import os

from lwdid.selection_diagnostics import (
    diagnose_selection_mechanism,
    plot_missing_pattern,
    get_unit_missing_stats,
    SelectionRisk,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def cattaneo_data():
    """Load Cattaneo data if available."""
    # Try multiple possible paths
    possible_paths = [
        'cattaneo2.dta',
        'data/cattaneo2.dta',
        '../cattaneo2.dta',
        os.path.join(os.path.dirname(__file__), '..', 'data', 'cattaneo2.dta'),
        os.path.join(os.path.dirname(__file__), '..', '..', 'cattaneo2.dta'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return pd.read_stata(path)
    
    pytest.skip("Cattaneo data not available")


@pytest.fixture
def nlswork_data():
    """Load NLS Work data if available."""
    possible_paths = [
        'nlswork_did.csv',
        'data/nlswork_did.csv',
        '../nlswork_did.csv',
        os.path.join(os.path.dirname(__file__), '..', 'data', 'nlswork_did.csv'),
        os.path.join(os.path.dirname(__file__), '..', '..', 'nlswork_did.csv'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return pd.read_csv(path)
    
    pytest.skip("NLS Work data not available")


@pytest.fixture
def project_test_data():
    """Load project test data."""
    possible_paths = [
        'auto_test.csv',
        'auto_stata_test.csv',
        'test_ipw_bug009.csv',
        '../auto_test.csv',
        '../auto_stata_test.csv',
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return pd.read_csv(path)
    
    pytest.skip("No project test data available")


# =============================================================================
# Test with Cattaneo Data
# =============================================================================

class TestCattaneoData:
    """Tests using Cattaneo (2010) maternal smoking data."""
    
    def test_diagnostics_run_successfully(self, cattaneo_data):
        """Diagnostics should complete without error on real data."""
        # This is cross-sectional data, so we need to create a pseudo-panel
        # For testing purposes, we'll just verify the function handles it
        
        # Create a simple panel structure for testing
        data = cattaneo_data.copy()
        data['unit_id'] = range(len(data))
        data['year'] = 1  # Single period
        
        # This should handle gracefully (single period)
        try:
            diag = diagnose_selection_mechanism(
                data,
                y='bweight',
                ivar='unit_id',
                tvar='year',
                verbose=False
            )
            assert diag is not None
        except ValueError as e:
            # Expected: may fail due to single period
            assert "period" in str(e).lower() or "time" in str(e).lower()


# =============================================================================
# Test with NLS Work Data
# =============================================================================

class TestNLSWorkData:
    """Tests using NLS Work panel data."""
    
    def test_diagnostics_on_real_panel(self, nlswork_data):
        """Test diagnostics on real panel data."""
        data = nlswork_data
        
        # Identify required columns (adjust based on actual data structure)
        id_cols = [c for c in data.columns if 'id' in c.lower() or 'unit' in c.lower()]
        time_cols = [c for c in data.columns if 'year' in c.lower() or 'time' in c.lower() or c.lower() == 't']
        y_cols = [c for c in data.columns if c.lower() in ['y', 'outcome', 'y_obs', 'ln_wage', 'wage']]
        
        if not id_cols or not time_cols:
            pytest.skip("Required columns not found in NLS Work data")
        
        # Find an outcome variable
        if not y_cols:
            # Try to find any numeric column
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            y_cols = [c for c in numeric_cols if c not in id_cols + time_cols]
        
        if not y_cols:
            pytest.skip("No outcome variable found")
        
        y_col = y_cols[0]
        ivar = id_cols[0]
        tvar = time_cols[0]
        
        diag = diagnose_selection_mechanism(
            data,
            y=y_col,
            ivar=ivar,
            tvar=tvar,
            verbose=False
        )
        
        # Verify diagnostics completed
        assert diag is not None
        assert diag.balance_statistics.n_units > 0
        assert diag.balance_statistics.n_periods > 0
        
        # NLS Work is typically unbalanced
        # (workers enter/exit the labor force)
        print(f"NLS Work diagnostics:")
        print(f"  - Balanced: {diag.balance_statistics.is_balanced}")
        print(f"  - Missing rate: {diag.missing_rate_overall:.1%}")
        print(f"  - Selection risk: {diag.selection_risk.value}")
    
    def test_visualization_on_real_data(self, nlswork_data):
        """Test visualization on real panel data."""
        import matplotlib.pyplot as plt
        
        data = nlswork_data
        
        id_cols = [c for c in data.columns if 'id' in c.lower() or 'unit' in c.lower()]
        time_cols = [c for c in data.columns if 'year' in c.lower() or 'time' in c.lower() or c.lower() == 't']
        
        if not id_cols or not time_cols:
            pytest.skip("Required columns not found")
        
        ivar = id_cols[0]
        tvar = time_cols[0]
        
        # Subsample for visualization (full data may be too large)
        sample_units = data[ivar].unique()[:100]
        sample_data = data[data[ivar].isin(sample_units)]
        
        fig = plot_missing_pattern(
            sample_data,
            ivar=ivar,
            tvar=tvar,
            sort_by='missing_rate'
        )
        
        assert fig is not None
        
        # Clean up
        plt.close(fig)


# =============================================================================
# Test with Project Data
# =============================================================================

class TestProjectData:
    """Tests using project's own test data."""
    
    def test_diagnostics_on_project_data(self, project_test_data):
        """Test diagnostics on project's test data."""
        data = project_test_data
        
        # Identify columns
        id_cols = [c for c in data.columns if 'id' in c.lower() or 'unit' in c.lower() or c.lower() == 'i']
        time_cols = [c for c in data.columns if 'year' in c.lower() or 'time' in c.lower() or c.lower() == 't']
        y_cols = [c for c in data.columns if c.lower() in ['y', 'outcome', 'y_obs']]
        
        if not id_cols or not time_cols or not y_cols:
            # Try alternative column names
            if 'i' in data.columns:
                id_cols = ['i']
            if 't' in data.columns:
                time_cols = ['t']
            if 'y' in data.columns:
                y_cols = ['y']
        
        if not id_cols or not time_cols or not y_cols:
            pytest.skip("Required columns not identified")
        
        diag = diagnose_selection_mechanism(
            data,
            y=y_cols[0],
            ivar=id_cols[0],
            tvar=time_cols[0],
            verbose=False
        )
        
        assert diag is not None
        print(f"Project data diagnostics: {diag.selection_risk.value}")


# =============================================================================
# Test Edge Cases with Real-World-Like Data
# =============================================================================

class TestEdgeCases:
    """Test edge cases with real-world-like data."""
    
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
    
    def test_high_attrition_data(self):
        """Test with high attrition rate (>50%)."""
        np.random.seed(42)
        
        data = []
        for i in range(100):
            # High attrition: only 30% of units have all periods
            n_periods = np.random.choice([3, 5, 10], p=[0.5, 0.2, 0.3])
            for t in range(1, n_periods + 1):
                data.append({
                    'unit_id': i,
                    'year': t,
                    'y': np.random.randn(),
                })
        
        df = pd.DataFrame(data)
        
        diag = diagnose_selection_mechanism(
            df, y='y', ivar='unit_id', tvar='year', verbose=False
        )
        
        # Should detect high attrition
        assert diag.attrition_analysis.attrition_rate > 0.5
        
        # Should have elevated selection risk
        assert diag.selection_risk in [SelectionRisk.MEDIUM, SelectionRisk.HIGH]
    
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
    
    def test_cohort_specific_attrition(self):
        """Test with cohort-specific attrition patterns."""
        np.random.seed(42)
        
        data = []
        for i in range(100):
            # Assign cohort
            if i < 40:
                gvar = 6
                attrition_prob = 0.3  # High attrition for cohort 6
            elif i < 70:
                gvar = 8
                attrition_prob = 0.1  # Low attrition for cohort 8
            else:
                gvar = 0
                attrition_prob = 0.05  # Very low for never-treated
            
            for t in range(1, 11):
                if t > 5 and np.random.random() < attrition_prob:
                    continue
                data.append({
                    'unit_id': i,
                    'year': t,
                    'y': np.random.randn(),
                    'gvar': gvar,
                })
        
        df = pd.DataFrame(data)
        
        diag = diagnose_selection_mechanism(
            df, y='y', ivar='unit_id', tvar='year', gvar='gvar', verbose=False
        )
        
        # Should have cohort-specific attrition rates
        assert len(diag.attrition_analysis.attrition_by_cohort) > 0
        
        # Cohort 6 should have higher attrition than cohort 8
        if 6 in diag.attrition_analysis.attrition_by_cohort and \
           8 in diag.attrition_analysis.attrition_by_cohort:
            assert diag.attrition_analysis.attrition_by_cohort[6] > \
                   diag.attrition_analysis.attrition_by_cohort[8]
    
    def test_treatment_related_dropout(self):
        """Test with dropout related to treatment timing."""
        np.random.seed(42)
        
        data = []
        for i in range(100):
            gvar = 6 if i < 50 else 0
            
            for t in range(1, 11):
                # Treated units more likely to drop out after treatment
                if gvar > 0 and t > gvar and np.random.random() < 0.2:
                    continue
                
                data.append({
                    'unit_id': i,
                    'year': t,
                    'y': np.random.randn(),
                    'gvar': gvar,
                })
        
        df = pd.DataFrame(data)
        
        diag = diagnose_selection_mechanism(
            df, y='y', ivar='unit_id', tvar='year', gvar='gvar', verbose=False
        )
        
        # Should detect dropout after treatment
        assert diag.attrition_analysis.dropout_after_treatment > 0
    
    def test_sparse_panel(self):
        """Test with very sparse panel (many missing observations)."""
        np.random.seed(42)
        
        data = []
        for i in range(50):
            # Each unit observed in only 3 random periods
            observed_periods = np.random.choice(range(1, 11), size=3, replace=False)
            for t in observed_periods:
                data.append({
                    'unit_id': i,
                    'year': t,
                    'y': np.random.randn(),
                })
        
        df = pd.DataFrame(data)
        
        diag = diagnose_selection_mechanism(
            df, y='y', ivar='unit_id', tvar='year', verbose=False
        )
        
        # Should detect high missing rate
        assert diag.missing_rate_overall > 0.5
        
        # Should have high selection risk
        assert diag.selection_risk in [SelectionRisk.MEDIUM, SelectionRisk.HIGH]
    
    def test_nearly_balanced_panel(self):
        """Test with nearly balanced panel (only 1-2 units incomplete)."""
        data = []
        for i in range(50):
            # Only 2 units have missing data
            if i < 2:
                periods = range(1, 9)  # Missing last 2 periods
            else:
                periods = range(1, 11)  # Complete
            
            for t in periods:
                data.append({
                    'unit_id': i,
                    'year': t,
                    'y': np.random.randn(),
                })
        
        df = pd.DataFrame(data)
        
        diag = diagnose_selection_mechanism(
            df, y='y', ivar='unit_id', tvar='year', verbose=False
        )
        
        # Should detect as unbalanced but low risk
        assert not diag.balance_statistics.is_balanced
        assert diag.selection_risk == SelectionRisk.LOW
    
    def test_unit_stats_consistency(self):
        """Test that unit stats are consistent with diagnostics."""
        np.random.seed(42)
        
        data = []
        for i in range(30):
            gvar = 6 if i < 15 else 0
            n_periods = 10 if i % 3 != 0 else 7  # Some units incomplete
            
            for t in range(1, n_periods + 1):
                data.append({
                    'unit_id': i,
                    'year': t,
                    'y': np.random.randn(),
                    'gvar': gvar,
                })
        
        df = pd.DataFrame(data)
        
        # Get diagnostics
        diag = diagnose_selection_mechanism(
            df, y='y', ivar='unit_id', tvar='year', gvar='gvar', verbose=False
        )
        
        # Get unit stats
        stats_df = get_unit_missing_stats(
            df, y='y', ivar='unit_id', tvar='year', gvar='gvar'
        )
        
        # Verify consistency
        assert len(stats_df) == diag.balance_statistics.n_units
        
        # Total missing should match
        total_missing_from_stats = stats_df['n_missing'].sum()
        total_obs_from_stats = stats_df['n_observed'].sum()
        expected_total = diag.balance_statistics.n_units * diag.balance_statistics.n_periods
        
        assert total_missing_from_stats + total_obs_from_stats == expected_total
