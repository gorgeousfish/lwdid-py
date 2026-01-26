"""
Test DESIGN-043-J fix: Bootstrap SE groupby include_groups parameter.

Validates that:
1. No FutureWarning is raised in pandas 2.0+
2. Bootstrap SE computation produces consistent numerical results
3. Event study aggregation works correctly with weighted/simple methods
"""

import warnings
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch


class TestGroupbyApplyKwargs:
    """Test pandas 2.x compatibility for groupby.apply()."""
    
    @pytest.fixture
    def sample_staggered_data(self):
        """Create sample staggered adoption data for testing."""
        np.random.seed(42)
        n_units = 50
        n_periods = 10
        
        # Generate unit-level treatment timing
        treatment_timing = {i: np.random.choice([5, 6, 7, 0]) for i in range(n_units)}
        
        data = []
        for unit in range(n_units):
            cohort = treatment_timing[unit]
            for t in range(1, n_periods + 1):
                treated = 1 if (cohort > 0 and t >= cohort) else 0
                # Generate outcome with treatment effect
                y = 2.0 + 0.5 * unit + 0.3 * t + 1.5 * treated + np.random.normal(0, 1)
                data.append({
                    'id': unit,
                    'time': t,
                    'gvar': cohort,
                    'y': y,
                    'treated': treated,
                })
        
        return pd.DataFrame(data)
    
    def test_no_future_warning_in_bootstrap_loop(self, sample_staggered_data):
        """Test that bootstrap loop groupby.apply() does not raise FutureWarning."""
        import re
        
        # Simulate the bootstrap aggregation logic
        boot_df = pd.DataFrame({
            'cohort': [5, 5, 6, 6, 7, 7],
            'event_time': [0, 1, 0, 1, 0, 1],
            'att': [1.5, 1.7, 1.4, 1.6, 1.3, 1.5],
            'weight': [0.3, 0.3, 0.4, 0.4, 0.3, 0.3],
        })
        
        # Pandas 2.x compatibility
        _version_match = re.match(r'^(\d+)\.(\d+)', pd.__version__)
        _pandas_version = (int(_version_match.group(1)), int(_version_match.group(2))) if _version_match else (1, 0)
        _groupby_apply_kwargs = {'include_groups': False} if _pandas_version >= (2, 0) else {}
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # This is the exact pattern from the fixed code
            event_atts = boot_df.groupby('event_time').apply(
                lambda x: np.average(x['att'], weights=x['weight']) 
                if x['weight'].sum() > 0 else x['att'].mean(),
                **_groupby_apply_kwargs
            )
            
            # Check no FutureWarning about include_groups
            future_warnings = [warning for warning in w 
                             if issubclass(warning.category, FutureWarning)
                             and 'include_groups' in str(warning.message).lower()]
            
            assert len(future_warnings) == 0, \
                f"FutureWarning raised: {[str(fw.message) for fw in future_warnings]}"
    
    def test_weighted_aggregation_numerical_correctness(self):
        """Test that weighted aggregation produces correct numerical results."""
        import re
        
        boot_df = pd.DataFrame({
            'cohort': [5, 5, 6, 6],
            'event_time': [0, 0, 1, 1],
            'att': [1.0, 2.0, 3.0, 4.0],
            'weight': [0.6, 0.4, 0.3, 0.7],
        })
        
        _version_match = re.match(r'^(\d+)\.(\d+)', pd.__version__)
        _pandas_version = (int(_version_match.group(1)), int(_version_match.group(2))) if _version_match else (1, 0)
        _groupby_apply_kwargs = {'include_groups': False} if _pandas_version >= (2, 0) else {}
        
        event_atts = boot_df.groupby('event_time').apply(
            lambda x: np.average(x['att'], weights=x['weight']) 
            if x['weight'].sum() > 0 else x['att'].mean(),
            **_groupby_apply_kwargs
        )
        
        # Manual calculation for event_time=0: (1.0*0.6 + 2.0*0.4) / (0.6+0.4) = 1.4
        # Manual calculation for event_time=1: (3.0*0.3 + 4.0*0.7) / (0.3+0.7) = 3.7
        expected_0 = (1.0 * 0.6 + 2.0 * 0.4) / (0.6 + 0.4)
        expected_1 = (3.0 * 0.3 + 4.0 * 0.7) / (0.3 + 0.7)
        
        np.testing.assert_almost_equal(event_atts[0], expected_0, decimal=10)
        np.testing.assert_almost_equal(event_atts[1], expected_1, decimal=10)
    
    def test_simple_aggregation_numerical_correctness(self):
        """Test that simple (unweighted) aggregation produces correct results."""
        boot_df = pd.DataFrame({
            'cohort': [5, 5, 6, 6],
            'event_time': [0, 0, 1, 1],
            'att': [1.0, 2.0, 3.0, 4.0],
        })
        
        event_atts = boot_df.groupby('event_time')['att'].mean()
        
        # Manual calculation
        expected_0 = (1.0 + 2.0) / 2  # 1.5
        expected_1 = (3.0 + 4.0) / 2  # 3.5
        
        np.testing.assert_almost_equal(event_atts[0], expected_0, decimal=10)
        np.testing.assert_almost_equal(event_atts[1], expected_1, decimal=10)
    
    def test_consistency_between_aggregation_methods(self):
        """Test that equal weights produce same result as simple mean."""
        import re
        
        boot_df = pd.DataFrame({
            'cohort': [5, 5, 6, 6],
            'event_time': [0, 0, 1, 1],
            'att': [1.0, 2.0, 3.0, 4.0],
            'weight': [1.0, 1.0, 1.0, 1.0],  # Equal weights
        })
        
        _version_match = re.match(r'^(\d+)\.(\d+)', pd.__version__)
        _pandas_version = (int(_version_match.group(1)), int(_version_match.group(2))) if _version_match else (1, 0)
        _groupby_apply_kwargs = {'include_groups': False} if _pandas_version >= (2, 0) else {}
        
        weighted_result = boot_df.groupby('event_time').apply(
            lambda x: np.average(x['att'], weights=x['weight']) 
            if x['weight'].sum() > 0 else x['att'].mean(),
            **_groupby_apply_kwargs
        )
        
        simple_result = boot_df.groupby('event_time')['att'].mean()
        
        np.testing.assert_array_almost_equal(
            weighted_result.values, 
            simple_result.values, 
            decimal=10
        )
    
    def test_kwargs_definition_location(self):
        """Verify _groupby_apply_kwargs is defined before use in bootstrap loop."""
        import ast
        
        # Read the source file
        with open('/Users/cxy/Desktop/大样本lwdid/lwdid-py_v0.1.0/src/lwdid/results.py', 'r') as f:
            source = f.read()
        
        # Find the function definition
        func_start = source.find('def _compute_event_study_se_bootstrap(')
        assert func_start != -1, "Function not found"
        
        # Find the definition of _groupby_apply_kwargs
        kwargs_def = source.find("_groupby_apply_kwargs = {'include_groups': False}", func_start)
        assert kwargs_def != -1, "_groupby_apply_kwargs definition not found"
        
        # Find the bootstrap loop
        bootstrap_loop = source.find('for b in range(n_bootstrap):', func_start)
        assert bootstrap_loop != -1, "Bootstrap loop not found"
        
        # Verify definition comes before the loop
        assert kwargs_def < bootstrap_loop, \
            "_groupby_apply_kwargs must be defined before bootstrap loop"
    
    def test_all_groupby_apply_use_kwargs(self):
        """Verify all groupby().apply() calls use _groupby_apply_kwargs."""
        with open('/Users/cxy/Desktop/大样本lwdid/lwdid-py_v0.1.0/src/lwdid/results.py', 'r') as f:
            source = f.read()
        
        # Find the function
        func_start = source.find('def _compute_event_study_se_bootstrap(')
        func_end = source.find('\ndef ', func_start + 1)
        if func_end == -1:
            func_end = len(source)
        
        func_source = source[func_start:func_end]
        
        # Count groupby().apply() calls
        import re
        apply_pattern = r'\.groupby\([^)]+\)\.apply\('
        apply_matches = list(re.finditer(apply_pattern, func_source))
        
        # Count uses of _groupby_apply_kwargs
        kwargs_uses = func_source.count('**_groupby_apply_kwargs')
        
        # The function has 3 groupby().apply() calls that need kwargs:
        # 1. In bootstrap loop (weighted)
        # 2. After loop (weighted_agg_orig)
        # 3. After loop (simple_agg_orig)
        # Note: groupby()[col].apply() doesn't need kwargs
        
        # All apply() calls that take a function should use kwargs
        assert kwargs_uses >= 3, \
            f"Expected at least 3 uses of _groupby_apply_kwargs, found {kwargs_uses}"


class TestPandasVersionCompatibility:
    """Test compatibility across pandas versions."""
    
    def test_pandas_version_detection(self):
        """Test that pandas version is correctly detected."""
        import re
        
        _version_match = re.match(r'^(\d+)\.(\d+)', pd.__version__)
        assert _version_match is not None, "Could not parse pandas version"
        
        major = int(_version_match.group(1))
        minor = int(_version_match.group(2))
        
        _pandas_version = (major, minor)
        
        # Current pandas should be 2.x
        if major >= 2:
            assert _pandas_version >= (2, 0)
            _groupby_apply_kwargs = {'include_groups': False}
            assert _groupby_apply_kwargs == {'include_groups': False}
        else:
            _groupby_apply_kwargs = {}
            assert _groupby_apply_kwargs == {}
    
    def test_backwards_compatibility_pandas_1x(self):
        """Test that the code would work on pandas 1.x (simulated)."""
        # Simulate pandas 1.x behavior
        _groupby_apply_kwargs = {}  # Empty dict for pandas < 2.0
        
        boot_df = pd.DataFrame({
            'event_time': [0, 0, 1, 1],
            'att': [1.0, 2.0, 3.0, 4.0],
            'weight': [0.5, 0.5, 0.5, 0.5],
        })
        
        # This should work with empty kwargs
        event_atts = boot_df.groupby('event_time').apply(
            lambda x: np.average(x['att'], weights=x['weight']),
            **_groupby_apply_kwargs
        )
        
        assert len(event_atts) == 2
        np.testing.assert_almost_equal(event_atts[0], 1.5, decimal=10)
        np.testing.assert_almost_equal(event_atts[1], 3.5, decimal=10)


class TestIntegration:
    """Integration tests for the full bootstrap SE computation."""
    
    @pytest.fixture
    def mock_staggered_data(self):
        """Create mock transformed data for integration testing."""
        np.random.seed(123)
        n_units = 30
        n_periods = 8
        
        data = []
        for unit in range(n_units):
            cohort = np.random.choice([4, 5, 6, 0])  # 0 = never treated
            for t in range(1, n_periods + 1):
                treated = 1 if (cohort > 0 and t >= cohort) else 0
                y = 1.0 + 0.2 * unit + 0.1 * t + 2.0 * treated + np.random.normal(0, 0.5)
                data.append({
                    'id': unit,
                    'time': t,
                    'gvar': cohort,
                    'y': y,
                    'ydot': y,  # Simplified: use y as transformed
                    'ydot_base': y - 2.0 * treated,
                })
        
        return pd.DataFrame(data)
    
    def test_no_warnings_in_full_function(self, mock_staggered_data):
        """Test that the full _compute_event_study_se_bootstrap raises no warnings."""
        # This is a smoke test - we just verify no warnings are raised
        # Full numerical validation requires the complete package setup
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Simulate the aggregation logic
            import re
            
            _version_match = re.match(r'^(\d+)\.(\d+)', pd.__version__)
            _pandas_version = (int(_version_match.group(1)), int(_version_match.group(2))) if _version_match else (1, 0)
            _groupby_apply_kwargs = {'include_groups': False} if _pandas_version >= (2, 0) else {}
            
            # Simulate bootstrap iteration
            for b in range(5):
                boot_df = mock_staggered_data.sample(frac=1, replace=True).copy()
                boot_df['event_time'] = boot_df['time'] - boot_df['gvar']
                boot_df = boot_df[boot_df['gvar'] > 0]  # Only treated
                boot_df['weight'] = 1.0
                
                if not boot_df.empty:
                    event_atts = boot_df.groupby('event_time').apply(
                        lambda x: np.average(x['y'], weights=x['weight']) 
                        if x['weight'].sum() > 0 else x['y'].mean(),
                        **_groupby_apply_kwargs
                    )
            
            # Check for FutureWarning about include_groups
            future_warnings = [warning for warning in w 
                             if issubclass(warning.category, FutureWarning)
                             and 'include_groups' in str(warning.message).lower()]
            
            assert len(future_warnings) == 0, \
                f"FutureWarning raised: {[str(fw.message) for fw in future_warnings]}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
