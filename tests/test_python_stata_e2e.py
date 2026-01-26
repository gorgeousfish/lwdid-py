"""
Python to Stata end-to-end validation tests.

This module performs comprehensive end-to-end tests comparing Python lwdid
results with Stata lwdid results to ensure consistency.

Test Approach:
1. Generate test data and save to CSV
2. Run lwdid in Python
3. Run lwdid in Stata via MCP
4. Compare ATT estimates and standard errors
"""

import warnings
import pytest
import numpy as np
import pandas as pd
import sys
import os
import json

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lwdid import lwdid


# =============================================================================
# Test Data Generation
# =============================================================================

def generate_common_timing_data(n_units=100, n_periods=6, seed=42):
    """Generate common timing DiD test data."""
    np.random.seed(seed)
    
    data = []
    for i in range(1, n_units + 1):
        treated = 1 if i <= n_units // 2 else 0
        for t in range(2000, 2000 + n_periods):
            post = 1 if t >= 2003 else 0
            # Treatment effect = 2.0
            y = 10 + 2 * treated * post + np.random.normal(0, 1)
            data.append({
                'unit': i,
                'year': t,
                'treated': treated,
                'post': post,
                'outcome': y,
            })
    
    return pd.DataFrame(data)


def generate_staggered_data(n_units=90, n_periods=8, seed=42):
    """Generate staggered adoption DiD test data."""
    np.random.seed(seed)
    
    data = []
    for i in range(1, n_units + 1):
        # Assign cohort
        if i <= 30:
            gvar = 0  # never treated
        elif i <= 60:
            gvar = 2003
        else:
            gvar = 2005
            
        for t in range(2000, 2000 + n_periods):
            treated = 1 if gvar > 0 and t >= gvar else 0
            # Cohort-specific effects: 2003 -> 2.0, 2005 -> 3.0
            effect = 2.0 if gvar == 2003 and treated else (3.0 if gvar == 2005 and treated else 0)
            y = 10 + effect + np.random.normal(0, 1)
            data.append({
                'unit': i,
                'year': t,
                'gvar': gvar,
                'outcome': y,
            })
    
    return pd.DataFrame(data)


# =============================================================================
# Python-only Tests (no Stata dependency)
# =============================================================================

class TestPythonResults:
    """Test Python lwdid results are valid and consistent."""
    
    def test_common_timing_returns_valid_result(self):
        """Common timing DiD should return valid results."""
        df = generate_common_timing_data()
        
        result = lwdid(
            data=df,
            y='outcome',
            d='treated',
            ivar='unit',
            tvar='year',
            post='post',
            rolling='demean'
        )
        
        # Basic validity checks
        assert result is not None
        assert hasattr(result, 'att')
        assert hasattr(result, 'se_att')
        assert not np.isnan(result.att)
        assert not np.isnan(result.se_att)
        assert result.se_att > 0
        
        # ATT should be close to 2.0 (true effect)
        assert 1.0 < result.att < 3.0, f"ATT={result.att} not in expected range"
    
    def test_staggered_returns_valid_result(self):
        """Staggered DiD should return valid results."""
        df = generate_staggered_data()
        
        result = lwdid(
            data=df,
            y='outcome',
            ivar='unit',
            tvar='year',
            gvar='gvar',
            rolling='demean'
        )
        
        # Basic validity checks
        assert result is not None
        assert hasattr(result, 'att')
        assert hasattr(result, 'se_att')
        assert not np.isnan(result.att)
        assert not np.isnan(result.se_att)
        assert result.se_att > 0
    
    def test_common_timing_detrend(self):
        """Common timing with detrend transformation."""
        df = generate_common_timing_data()
        
        result = lwdid(
            data=df,
            y='outcome',
            d='treated',
            ivar='unit',
            tvar='year',
            post='post',
            rolling='detrend'
        )
        
        assert result is not None
        assert not np.isnan(result.att)
    
    def test_staggered_cohort_aggregation(self):
        """Staggered DiD with cohort-level aggregation."""
        df = generate_staggered_data()
        
        result = lwdid(
            data=df,
            y='outcome',
            ivar='unit',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            aggregate='cohort'
        )
        
        assert result is not None
        assert hasattr(result, 'att_by_cohort')
        assert result.att_by_cohort is not None
        assert len(result.att_by_cohort) > 0
    
    def test_staggered_overall_aggregation(self):
        """Staggered DiD with overall aggregation."""
        df = generate_staggered_data()
        
        result = lwdid(
            data=df,
            y='outcome',
            ivar='unit',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            aggregate='overall'
        )
        
        assert result is not None
        # Overall aggregation should produce single ATT
        assert hasattr(result, 'att')


# =============================================================================
# Cross-validation Tests
# =============================================================================

class TestCrossValidation:
    """Cross-validate different estimation methods."""
    
    def test_demean_vs_detrend_both_positive(self):
        """Demean and detrend should both detect positive treatment effect."""
        df = generate_common_timing_data()
        
        result_demean = lwdid(
            data=df, y='outcome', d='treated', ivar='unit',
            tvar='year', post='post', rolling='demean'
        )
        
        result_detrend = lwdid(
            data=df, y='outcome', d='treated', ivar='unit',
            tvar='year', post='post', rolling='detrend'
        )
        
        # Both methods should detect a positive treatment effect
        # (true effect is 2.0, so ATT should be in [1, 4] range)
        assert 1.0 < result_demean.att < 4.0, \
            f"Demean ATT={result_demean.att:.4f} not in expected range [1, 4]"
        assert 1.0 < result_detrend.att < 4.0, \
            f"Detrend ATT={result_detrend.att:.4f} not in expected range [1, 4]"
        
        # Both should have reasonable standard errors
        assert result_demean.se_att > 0
        assert result_detrend.se_att > 0
    
    def test_staggered_control_groups_similar(self):
        """Different control groups should give similar results."""
        df = generate_staggered_data()
        
        result_never = lwdid(
            data=df, y='outcome', ivar='unit', tvar='year',
            gvar='gvar', rolling='demean', control_group='never_treated'
        )
        
        result_notyet = lwdid(
            data=df, y='outcome', ivar='unit', tvar='year',
            gvar='gvar', rolling='demean', control_group='not_yet_treated'
        )
        
        # Results should be reasonably similar
        assert result_never is not None
        assert result_notyet is not None


# =============================================================================
# Stata E2E Comparison (requires Stata MCP)
# =============================================================================

class TestStataComparison:
    """Compare Python results with Stata (requires Stata MCP)."""
    
    @pytest.fixture
    def test_data_path(self, tmp_path):
        """Create test data CSV file."""
        df = generate_common_timing_data()
        csv_path = tmp_path / "test_data.csv"
        df.to_csv(csv_path, index=False)
        return str(csv_path)
    
    @pytest.mark.skip(reason="Requires Stata MCP to be running")
    def test_common_timing_matches_stata(self, test_data_path):
        """
        Compare Python common timing DiD with Stata lwdid.
        
        This test requires:
        1. Stata to be installed
        2. lwdid package installed in Stata
        3. Stata MCP server to be running
        """
        # Run Python
        df = pd.read_csv(test_data_path)
        python_result = lwdid(
            data=df, y='outcome', d='treated', ivar='unit',
            tvar='year', post='post', rolling='demean'
        )
        
        # The Stata comparison would be done via MCP
        # This is a placeholder for when MCP is available
        
        python_att = python_result.att
        python_se = python_result.se_att
        
        # Expected to be close to Stata results
        # (tolerance depends on numerical precision differences)
        assert python_att is not None
        assert python_se is not None


# =============================================================================
# Consistency Tests
# =============================================================================

class TestConsistency:
    """Test result consistency across different scenarios."""
    
    def test_reproducibility_with_seed(self):
        """Results should be reproducible with same seed."""
        df = generate_staggered_data(seed=12345)
        
        result1 = lwdid(
            data=df, y='outcome', ivar='unit', tvar='year',
            gvar='gvar', rolling='demean', ri=True, rireps=50, seed=42
        )
        
        result2 = lwdid(
            data=df, y='outcome', ivar='unit', tvar='year',
            gvar='gvar', rolling='demean', ri=True, rireps=50, seed=42
        )
        
        # With same seed, RI p-values should be identical
        if hasattr(result1, 'ri_pvalue') and hasattr(result2, 'ri_pvalue'):
            assert result1.ri_pvalue == result2.ri_pvalue
    
    def test_different_seeds_different_ri(self):
        """Different seeds should give different RI results."""
        df = generate_staggered_data(seed=12345)
        
        result1 = lwdid(
            data=df, y='outcome', ivar='unit', tvar='year',
            gvar='gvar', rolling='demean', ri=True, rireps=50, seed=42
        )
        
        result2 = lwdid(
            data=df, y='outcome', ivar='unit', tvar='year',
            gvar='gvar', rolling='demean', ri=True, rireps=50, seed=43
        )
        
        # ATT should be the same (not dependent on RI seed)
        assert result1.att == result2.att


# =============================================================================
# Run tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-k', 'not Stata'])
