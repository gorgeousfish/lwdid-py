"""
BUG-022 and BUG-023 Regression Tests

Tests for staggered results export functionality:
- BUG-022: to_csv() for staggered results
- BUG-023: to_latex() for staggered results

These tests ensure that staggered DiD results can be correctly exported
to CSV and LaTeX formats using the generic to_csv() and to_latex() methods.

File: tests/staggered/test_bug022_023_staggered_export.py
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest


def create_mock_staggered_results(aggregate: str = 'overall'):
    """
    Create mock staggered results for testing export functionality.
    
    Parameters
    ----------
    aggregate : str
        Aggregation method: 'none', 'cohort', 'overall'
        
    Returns
    -------
    results_dict : dict
        Dictionary of result fields
    metadata : dict
        Dictionary of metadata fields
    """
    cohorts = [2005, 2006, 2007, 2008, 2009]
    
    # Create mock att_by_cohort_time DataFrame
    att_by_cohort_time_rows = []
    for g in cohorts:
        for r in range(g, 2011):
            att_by_cohort_time_rows.append({
                'cohort': g,
                'period': r,
                'att': np.random.uniform(0.05, 0.20),
                'se': np.random.uniform(0.02, 0.05),
                'ci_lower': np.random.uniform(0.01, 0.05),
                'ci_upper': np.random.uniform(0.15, 0.25),
                't_stat': np.random.uniform(2.0, 5.0),
                'pvalue': np.random.uniform(0.01, 0.05),
            })
    att_by_cohort_time = pd.DataFrame(att_by_cohort_time_rows)
    
    # Create mock att_by_cohort DataFrame
    att_by_cohort = pd.DataFrame({
        'cohort': cohorts,
        'att': [0.11, 0.14, 0.11, 0.12, 0.10],
        'se': [0.025, 0.03, 0.028, 0.027, 0.026],
        'ci_lower': [0.06, 0.08, 0.055, 0.067, 0.05],
        'ci_upper': [0.16, 0.20, 0.165, 0.173, 0.15],
    })
    
    results_dict = {
        'is_staggered': True,
        'att': 0.0,  # Not used in staggered mode
        'se_att': 0.0,
        't_stat': 0.0,
        'pvalue': 1.0,
        'ci_lower': 0.0,
        'ci_upper': 0.0,
        'nobs': 550,
        'n_treated': 21,
        'n_control': 29,
        'df_resid': 48,
        'params': np.array([0.0]),
        'bse': np.array([0.0]),
        'vcov': np.array([[0.0]]),
        'resid': np.zeros(50),
        'vce_type': 'hc3',
        
        # Staggered-specific fields
        'cohorts': cohorts,
        'cohort_sizes': {g: 3 + (i % 3) for i, g in enumerate(cohorts)},
        'att_by_cohort_time': att_by_cohort_time,
        'att_by_cohort': att_by_cohort if aggregate != 'none' else pd.DataFrame(),
        'cohort_weights': {g: 0.2 for g in cohorts} if aggregate != 'none' else {},
        'n_never_treated': 29,
        'control_group': 'never_treated',
        'control_group_used': 'never_treated',
        'aggregate': aggregate,
    }
    
    # Add overall effect fields if aggregate='overall'
    if aggregate == 'overall':
        results_dict.update({
            'att_overall': 0.115,
            'se_overall': 0.02,
            't_stat_overall': 5.75,
            'pvalue_overall': 0.0001,
            'ci_overall_lower': 0.075,
            'ci_overall_upper': 0.155,
        })
    else:
        results_dict.update({
            'att_overall': None,
            'se_overall': None,
            't_stat_overall': None,
            'pvalue_overall': None,
            'ci_overall_lower': None,
            'ci_overall_upper': None,
        })
    
    metadata = {
        'K': 5,
        'tpost1': 2005,
        'N_treated': 21,
        'N_control': 29,
        'depvar': 'lhomicide',
        'ivar': 'sid',
        'tvar': 'year',
        'rolling': 'demean',
        'is_staggered': True,
        'estimator': 'ra',
    }
    
    return results_dict, metadata


# ============================================================
# Section 1: BUG-022 Tests - to_csv() for staggered results
# ============================================================

class TestBug022ToCsvStaggered:
    """
    BUG-022 Regression Tests: to_csv() for staggered results
    
    Validates that:
    1. to_csv() correctly dispatches for staggered results
    2. CSV output contains att_by_cohort_time data
    3. CSV has expected columns
    4. Error is raised when att_by_cohort_time is missing
    """
    
    def test_to_csv_staggered_basic(self):
        """to_csv() should export att_by_cohort_time for staggered results"""
        from lwdid.results import LWDIDResults
        
        results_dict, metadata = create_mock_staggered_results(aggregate='overall')
        results = LWDIDResults(results_dict, metadata, None)
        
        assert results.is_staggered, "Should be staggered result"
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = f.name
        
        try:
            results.to_csv(path)
            assert os.path.exists(path), "CSV file should be created"
            
            # Verify CSV content
            df = pd.read_csv(path)
            assert 'cohort' in df.columns, "Should have cohort column"
            assert 'period' in df.columns, "Should have period column"
            assert 'att' in df.columns, "Should have att column"
            assert 'se' in df.columns, "Should have se column"
            assert len(df) > 0, "CSV should have data rows"
        finally:
            if os.path.exists(path):
                os.remove(path)
    
    def test_to_csv_staggered_aggregate_none(self):
        """to_csv() should work for aggregate='none'"""
        from lwdid.results import LWDIDResults
        
        results_dict, metadata = create_mock_staggered_results(aggregate='none')
        results = LWDIDResults(results_dict, metadata, None)
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = f.name
        
        try:
            results.to_csv(path)
            assert os.path.exists(path)
            
            df = pd.read_csv(path)
            assert len(df) > 0
        finally:
            if os.path.exists(path):
                os.remove(path)
    
    def test_to_csv_staggered_aggregate_cohort(self):
        """to_csv() should work for aggregate='cohort'"""
        from lwdid.results import LWDIDResults
        
        results_dict, metadata = create_mock_staggered_results(aggregate='cohort')
        results = LWDIDResults(results_dict, metadata, None)
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = f.name
        
        try:
            results.to_csv(path)
            assert os.path.exists(path)
            
            df = pd.read_csv(path)
            assert 'cohort' in df.columns
            assert len(df) > 0
        finally:
            if os.path.exists(path):
                os.remove(path)
    
    def test_to_csv_staggered_missing_att_by_cohort_time_raises(self):
        """to_csv() should raise ValueError when att_by_cohort_time is missing"""
        from lwdid.results import LWDIDResults
        
        results_dict, metadata = create_mock_staggered_results(aggregate='overall')
        results_dict['att_by_cohort_time'] = None  # Remove the data
        results = LWDIDResults(results_dict, metadata, None)
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = f.name
        
        try:
            with pytest.raises(ValueError, match="att_by_cohort_time"):
                results.to_csv(path)
        finally:
            if os.path.exists(path):
                os.remove(path)
    
    def test_to_csv_staggered_empty_att_by_cohort_time_raises(self):
        """to_csv() should raise ValueError when att_by_cohort_time is empty"""
        from lwdid.results import LWDIDResults
        
        results_dict, metadata = create_mock_staggered_results(aggregate='overall')
        results_dict['att_by_cohort_time'] = pd.DataFrame()  # Empty DataFrame
        results = LWDIDResults(results_dict, metadata, None)
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = f.name
        
        try:
            with pytest.raises(ValueError, match="att_by_cohort_time"):
                results.to_csv(path)
        finally:
            if os.path.exists(path):
                os.remove(path)


# ============================================================
# Section 2: BUG-023 Tests - to_latex() for staggered results
# ============================================================

class TestBug023ToLatexStaggered:
    """
    BUG-023 Regression Tests: to_latex() for staggered results
    
    Validates that:
    1. to_latex() correctly dispatches to to_latex_staggered()
    2. LaTeX output contains proper tables
    3. LaTeX includes summary, cohort-time effects, cohort effects
    4. Error is raised for non-staggered results calling to_latex_staggered()
    """
    
    def test_to_latex_staggered_basic(self):
        """to_latex() should dispatch to to_latex_staggered() for staggered results"""
        from lwdid.results import LWDIDResults
        
        results_dict, metadata = create_mock_staggered_results(aggregate='overall')
        results = LWDIDResults(results_dict, metadata, None)
        
        assert results.is_staggered, "Should be staggered result"
        
        with tempfile.NamedTemporaryFile(suffix='.tex', delete=False) as f:
            path = f.name
        
        try:
            results.to_latex(path)
            assert os.path.exists(path), "LaTeX file should be created"
            
            # Verify LaTeX content
            with open(path, 'r') as f:
                content = f.read()
            
            assert 'tabular' in content, "Should contain tabular environment"
            assert len(content) > 100, "LaTeX should have substantial content"
        finally:
            if os.path.exists(path):
                os.remove(path)
    
    def test_to_latex_staggered_contains_sections(self):
        """to_latex_staggered() should include all relevant sections"""
        from lwdid.results import LWDIDResults
        
        results_dict, metadata = create_mock_staggered_results(aggregate='overall')
        results = LWDIDResults(results_dict, metadata, None)
        
        with tempfile.NamedTemporaryFile(suffix='.tex', delete=False) as f:
            path = f.name
        
        try:
            results.to_latex_staggered(path)
            
            with open(path, 'r') as f:
                content = f.read()
            
            # Should have summary section
            assert 'Summary Statistics' in content or 'Estimator' in content
            
            # Should have cohort-time effects
            assert 'Cohort-Time Effects' in content or 'cohort' in content.lower()
        finally:
            if os.path.exists(path):
                os.remove(path)
    
    def test_to_latex_staggered_aggregate_none(self):
        """to_latex() should work for aggregate='none'"""
        from lwdid.results import LWDIDResults
        
        results_dict, metadata = create_mock_staggered_results(aggregate='none')
        results = LWDIDResults(results_dict, metadata, None)
        
        with tempfile.NamedTemporaryFile(suffix='.tex', delete=False) as f:
            path = f.name
        
        try:
            results.to_latex(path)
            assert os.path.exists(path)
            
            with open(path, 'r') as f:
                content = f.read()
            assert len(content) > 0
        finally:
            if os.path.exists(path):
                os.remove(path)
    
    def test_to_latex_staggered_aggregate_cohort(self):
        """to_latex() should work for aggregate='cohort'"""
        from lwdid.results import LWDIDResults
        
        results_dict, metadata = create_mock_staggered_results(aggregate='cohort')
        results = LWDIDResults(results_dict, metadata, None)
        
        with tempfile.NamedTemporaryFile(suffix='.tex', delete=False) as f:
            path = f.name
        
        try:
            results.to_latex(path)
            assert os.path.exists(path)
            
            with open(path, 'r') as f:
                content = f.read()
            
            # Should include cohort effects section
            assert 'Cohort Effects' in content or 'cohort' in content.lower()
        finally:
            if os.path.exists(path):
                os.remove(path)
    
    def test_to_latex_staggered_not_staggered_raises(self):
        """to_latex_staggered() should raise ValueError for non-staggered results"""
        from lwdid.results import LWDIDResults
        
        # Create common timing results
        results_dict = {
            'is_staggered': False,
            'att': 0.1,
            'se_att': 0.05,
            't_stat': 2.0,
            'pvalue': 0.05,
            'ci_lower': 0.0,
            'ci_upper': 0.2,
            'nobs': 100,
            'n_treated': 10,
            'n_control': 90,
            'df_resid': 98,
            'params': np.array([0.1]),
            'bse': np.array([0.05]),
            'vcov': np.array([[0.0025]]),
            'resid': np.zeros(100),
            'vce_type': 'hc3',
        }
        
        metadata = {
            'K': 5,
            'tpost1': 6,
            'N_treated': 10,
            'N_control': 90,
            'depvar': 'y',
            'ivar': 'id',
            'tvar': 'year',
            'rolling': 'demean',
            'is_staggered': False,
        }
        
        results = LWDIDResults(results_dict, metadata, None)
        
        with pytest.raises(ValueError, match="staggered"):
            results.to_latex_staggered('/tmp/test.tex')


# ============================================================
# Section 3: Integration Tests - Real Data
# ============================================================

class TestStaggeredExportIntegration:
    """
    Integration tests using real staggered estimation.
    
    These tests ensure end-to-end functionality with actual lwdid estimation.
    """
    
    @pytest.fixture
    def castle_data(self):
        """Load castle.csv if available"""
        candidates = [
            os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'castle.csv'),
            os.path.join(os.path.dirname(__file__), '..', 'data', 'castle.csv'),
            'data/castle.csv',
        ]
        for path in candidates:
            if os.path.exists(path):
                return pd.read_csv(path)
        pytest.skip("Castle data file not found")
    
    @pytest.mark.slow
    def test_real_staggered_to_csv(self, castle_data):
        """Test to_csv() with real staggered estimation"""
        from lwdid import lwdid
        
        # Prepare data
        data = castle_data.copy()
        
        # Create gvar column from effyear
        data['gvar'] = data['effyear'].replace(np.inf, 0)
        
        # Run staggered estimation
        results = lwdid(
            data,
            y='lhomicide',
            ivar='sid',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            vce='hc3',
            aggregate='overall'
        )
        
        assert results.is_staggered, "Should be staggered result"
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = f.name
        
        try:
            results.to_csv(path)
            assert os.path.exists(path)
            
            df = pd.read_csv(path)
            assert 'cohort' in df.columns
            assert 'period' in df.columns
            assert len(df) > 0
        finally:
            if os.path.exists(path):
                os.remove(path)
    
    @pytest.mark.slow
    def test_real_staggered_to_latex(self, castle_data):
        """Test to_latex() with real staggered estimation"""
        from lwdid import lwdid
        
        # Prepare data
        data = castle_data.copy()
        
        # Create gvar column from effyear
        data['gvar'] = data['effyear'].replace(np.inf, 0)
        
        # Run staggered estimation
        results = lwdid(
            data,
            y='lhomicide',
            ivar='sid',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            vce='hc3',
            aggregate='overall'
        )
        
        with tempfile.NamedTemporaryFile(suffix='.tex', delete=False) as f:
            path = f.name
        
        try:
            results.to_latex(path)
            assert os.path.exists(path)
            
            with open(path, 'r') as f:
                content = f.read()
            
            assert 'tabular' in content
            assert len(content) > 100
        finally:
            if os.path.exists(path):
                os.remove(path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
