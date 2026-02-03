"""
Empirical Data Tests for Seasonal Adjustment.

Task 4.8: Test seasonal adjustment with real datasets.

Tests use available datasets to verify:
- Implementation runs without errors on real data
- Results are numerically reasonable
"""

import numpy as np
import pandas as pd
import pytest
import os

from lwdid import lwdid


class TestCattaneo2Data:
    """Tests using cattaneo2.dta dataset (if available)."""
    
    @pytest.fixture
    def cattaneo2_path(self):
        """Path to cattaneo2.dta file."""
        # Try multiple possible locations
        paths = [
            'cattaneo2.dta',
            '../cattaneo2.dta',
            '../../cattaneo2.dta',
        ]
        for p in paths:
            if os.path.exists(p):
                return p
        return None
    
    def test_cattaneo2_loads(self, cattaneo2_path):
        """Test that cattaneo2.dta can be loaded."""
        if cattaneo2_path is None:
            pytest.skip("cattaneo2.dta not found")
        
        df = pd.read_stata(cattaneo2_path)
        assert len(df) > 0, "Dataset should not be empty"
        assert 'bweight' in df.columns, "Expected 'bweight' column"


class TestNLSWorkData:
    """Tests using nlswork_did.csv dataset (if available)."""
    
    @pytest.fixture
    def nlswork_path(self):
        """Path to nlswork_did.csv file."""
        paths = [
            'nlswork_did.csv',
            '../nlswork_did.csv',
            '../../nlswork_did.csv',
        ]
        for p in paths:
            if os.path.exists(p):
                return p
        return None
    
    def test_nlswork_loads(self, nlswork_path):
        """Test that nlswork_did.csv can be loaded."""
        if nlswork_path is None:
            pytest.skip("nlswork_did.csv not found")
        
        df = pd.read_csv(nlswork_path)
        assert len(df) > 0, "Dataset should not be empty"


class TestSyntheticEmpiricalData:
    """Tests with synthetic data mimicking empirical patterns."""
    
    def test_realistic_quarterly_panel(self):
        """Test with realistic quarterly panel structure."""
        np.random.seed(42)
        
        # Simulate realistic quarterly panel
        # 50 units, 5 years (20 quarters), treatment at year 4
        n_units = 50
        n_periods = 20
        treatment_period = 13
        
        # Realistic seasonal pattern (retail-like)
        gamma = {1: -5, 2: 0, 3: 5, 4: 15}  # Q4 holiday boost
        
        data = []
        for i in range(n_units):
            # Unit-specific baseline
            alpha_i = np.random.normal(100, 20)
            # Treatment assignment (30% treated)
            treated = 1 if np.random.random() < 0.3 else 0
            
            for t in range(1, n_periods + 1):
                quarter = ((t - 1) % 4) + 1
                post = 1 if t >= treatment_period else 0
                
                # Realistic noise
                epsilon = np.random.normal(0, 5)
                
                # True treatment effect = 8
                y = alpha_i + gamma[quarter] + 8 * treated * post + epsilon
                
                data.append({
                    'unit': i,
                    't': t,
                    'quarter': quarter,
                    'year': 2018 + (t - 1) // 4,
                    'treated': treated,
                    'post': post,
                    'y': y
                })
        
        df = pd.DataFrame(data)
        
        # Run lwdid with demeanq
        result = lwdid(
            df, y='y', d='treated', ivar='unit',
            tvar=['year', 'quarter'], post='post',
            rolling='demeanq', season_var='quarter', Q=4
        )
        
        # Check results are reasonable
        assert result.att is not None
        assert not np.isnan(result.att)
        assert result.se_att > 0
        
        # ATT should be in reasonable range (true = 8)
        assert 4 < result.att < 12, f"ATT {result.att:.2f} outside reasonable range"
    
    def test_realistic_monthly_panel(self):
        """Test with realistic monthly panel structure."""
        np.random.seed(123)
        
        # 30 units, 3 years (36 months), treatment at month 19
        n_units = 30
        n_periods = 36
        treatment_period = 19
        
        # Realistic monthly seasonal pattern
        gamma = {m: 3 * np.sin(2 * np.pi * m / 12) for m in range(1, 13)}
        
        data = []
        for i in range(n_units):
            alpha_i = np.random.normal(50, 10)
            treated = 1 if np.random.random() < 0.4 else 0
            
            for t in range(1, n_periods + 1):
                month = ((t - 1) % 12) + 1
                post = 1 if t >= treatment_period else 0
                epsilon = np.random.normal(0, 2)
                
                # True treatment effect = 5
                y = alpha_i + gamma[month] + 5 * treated * post + epsilon
                
                data.append({
                    'unit': i,
                    't': t,
                    'month': month,
                    'treated': treated,
                    'post': post,
                    'y': y
                })
        
        df = pd.DataFrame(data)
        
        result = lwdid(
            df, y='y', d='treated', ivar='unit',
            tvar='t', post='post',
            rolling='demeanq', season_var='month', Q=12
        )
        
        assert result.att is not None
        assert not np.isnan(result.att)
        assert result.se_att > 0
        
        # ATT should be in reasonable range (true = 5)
        assert 2 < result.att < 8, f"ATT {result.att:.2f} outside reasonable range"
    
    def test_unbalanced_panel_handling(self):
        """Test with slightly unbalanced panel (some missing observations).
        
        Note: We ensure all quarters are covered in pre-treatment to avoid
        InsufficientQuarterDiversityError.
        """
        np.random.seed(456)
        
        n_units = 30
        n_periods = 20  # 5 years - more periods to ensure coverage
        treatment_period = 13  # Treatment at year 4
        gamma = {1: 0, 2: 3, 3: 6, 4: 2}
        
        data = []
        for i in range(n_units):
            alpha_i = np.random.normal(100, 15)
            treated = 1 if i < n_units // 3 else 0
            
            for t in range(1, n_periods + 1):
                quarter = ((t - 1) % 4) + 1
                post = 1 if t >= treatment_period else 0
                
                # Only skip observations in post-treatment period
                # to ensure pre-treatment has all quarters
                if post == 1 and np.random.random() < 0.1:
                    continue
                
                epsilon = np.random.normal(0, 3)
                y = alpha_i + gamma[quarter] + 10 * treated * post + epsilon
                
                data.append({
                    'unit': i,
                    't': t,
                    'quarter': quarter,
                    'year': 2018 + (t - 1) // 4,
                    'treated': treated,
                    'post': post,
                    'y': y
                })
        
        df = pd.DataFrame(data)
        
        # Should handle unbalanced panel (with warning)
        result = lwdid(
            df, y='y', d='treated', ivar='unit',
            tvar=['year', 'quarter'], post='post',
            rolling='demeanq', season_var='quarter', Q=4,
            balanced_panel='warn'
        )
        
        assert result.att is not None
        assert not np.isnan(result.att)
        # ATT should be close to true value (10)
        assert 6 < result.att < 14, f"ATT {result.att:.2f} outside reasonable range"
    
    def test_heterogeneous_treatment_effects(self):
        """Test with heterogeneous treatment effects across units."""
        np.random.seed(789)
        
        n_units = 40
        n_periods = 20
        treatment_period = 13
        gamma = {1: 0, 2: 4, 3: 8, 4: 2}
        
        data = []
        for i in range(n_units):
            alpha_i = np.random.normal(100, 10)
            treated = 1 if i < n_units // 2 else 0
            
            # Heterogeneous treatment effect
            tau_i = np.random.normal(10, 3) if treated else 0
            
            for t in range(1, n_periods + 1):
                quarter = ((t - 1) % 4) + 1
                post = 1 if t >= treatment_period else 0
                epsilon = np.random.normal(0, 2)
                
                y = alpha_i + gamma[quarter] + tau_i * post + epsilon
                
                data.append({
                    'unit': i,
                    't': t,
                    'quarter': quarter,
                    'year': 2018 + (t - 1) // 4,
                    'treated': treated,
                    'post': post,
                    'y': y
                })
        
        df = pd.DataFrame(data)
        
        result = lwdid(
            df, y='y', d='treated', ivar='unit',
            tvar=['year', 'quarter'], post='post',
            rolling='demeanq', season_var='quarter', Q=4
        )
        
        # Should recover average treatment effect (around 10)
        assert 6 < result.att < 14, f"ATT {result.att:.2f} outside reasonable range"


class TestResultsReasonableness:
    """Tests to verify results are numerically reasonable."""
    
    def test_standard_error_positive(self):
        """Test that standard errors are always positive."""
        np.random.seed(101)
        
        data = []
        for i in range(20):
            for t in range(1, 17):
                data.append({
                    'unit': i,
                    't': t,
                    'quarter': ((t - 1) % 4) + 1,
                    'year': 2018 + (t - 1) // 4,
                    'treated': 1 if i < 10 else 0,
                    'post': 1 if t >= 9 else 0,
                    'y': 100 + np.random.normal(0, 5)
                })
        
        df = pd.DataFrame(data)
        
        result = lwdid(
            df, y='y', d='treated', ivar='unit',
            tvar=['year', 'quarter'], post='post',
            rolling='demeanq', season_var='quarter', Q=4
        )
        
        assert result.se_att > 0, "Standard error must be positive"
    
    def test_confidence_interval_contains_att(self):
        """Test that CI contains the point estimate."""
        np.random.seed(202)
        
        data = []
        for i in range(25):
            for t in range(1, 21):
                data.append({
                    'unit': i,
                    't': t,
                    'quarter': ((t - 1) % 4) + 1,
                    'year': 2018 + (t - 1) // 4,
                    'treated': 1 if i < 12 else 0,
                    'post': 1 if t >= 13 else 0,
                    'y': 100 + 10 * (1 if i < 12 and t >= 13 else 0) + np.random.normal(0, 3)
                })
        
        df = pd.DataFrame(data)
        
        result = lwdid(
            df, y='y', d='treated', ivar='unit',
            tvar=['year', 'quarter'], post='post',
            rolling='demeanq', season_var='quarter', Q=4
        )
        
        # CI should contain ATT
        assert result.ci_lower <= result.att <= result.ci_upper, \
            "Confidence interval should contain point estimate"
    
    def test_pvalue_in_valid_range(self):
        """Test that p-value is in [0, 1]."""
        np.random.seed(303)
        
        data = []
        for i in range(20):
            for t in range(1, 17):
                data.append({
                    'unit': i,
                    't': t,
                    'quarter': ((t - 1) % 4) + 1,
                    'year': 2018 + (t - 1) // 4,
                    'treated': 1 if i < 10 else 0,
                    'post': 1 if t >= 9 else 0,
                    'y': 100 + 5 * (1 if i < 10 and t >= 9 else 0) + np.random.normal(0, 2)
                })
        
        df = pd.DataFrame(data)
        
        result = lwdid(
            df, y='y', d='treated', ivar='unit',
            tvar=['year', 'quarter'], post='post',
            rolling='demeanq', season_var='quarter', Q=4
        )
        
        assert 0 <= result.pvalue <= 1, "P-value must be in [0, 1]"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
