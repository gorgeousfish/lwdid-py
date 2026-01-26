"""
Unit Tests for BUG-021: Column Name Conflict Detection in Transformations

Tests that transform_staggered_demean() and transform_staggered_detrend()
properly detect and raise errors when output column names conflict with
existing columns in the input data.
"""

import warnings
import pytest
import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from lwdid.staggered.transformations import (
    transform_staggered_demean,
    transform_staggered_detrend,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def basic_staggered_data():
    """Basic staggered data for testing."""
    return pd.DataFrame({
        'id': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
        'year': [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
        'y': [10, 12, 14, 20, 15, 16, 17, 18, 5, 6, 7, 8],
        'gvar': [3, 3, 3, 3, 4, 4, 4, 4, 0, 0, 0, 0]
    })


@pytest.fixture
def detrend_data():
    """Data with at least 2 pre-treatment periods for detrending."""
    return pd.DataFrame({
        'id': [1] * 6 + [2] * 6,
        'year': [1, 2, 3, 4, 5, 6] * 2,
        'y': [12, 14, 16, 28, 32, 36,  # unit 1: trend=2, effect=10 at t>=4
              5, 7, 9, 11, 13, 15],    # unit 2 (control): trend=2
        'gvar': [4] * 6 + [0] * 6
    })


# =============================================================================
# Test 1: Demean Column Conflict Detection
# =============================================================================

class TestDemeanColumnConflict:
    """Tests for column conflict detection in transform_staggered_demean()."""
    
    def test_demean_column_conflict_raises_error(self, basic_staggered_data):
        """Test that existing ydot column raises ValueError."""
        # Add a conflicting column
        data = basic_staggered_data.copy()
        data['ydot_g3_r3'] = 999  # This will conflict
        
        with pytest.raises(ValueError, match="Column name conflict detected"):
            transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
    
    def test_demean_multiple_conflicts(self, basic_staggered_data):
        """Test that multiple conflicting columns are detected."""
        data = basic_staggered_data.copy()
        data['ydot_g3_r3'] = 999
        data['ydot_g3_r4'] = 888
        data['ydot_g4_r4'] = 777
        
        with pytest.raises(ValueError, match="Column name conflict detected"):
            transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
    
    def test_demean_error_message_contains_column_names(self, basic_staggered_data):
        """Test that error message lists the conflicting column names."""
        data = basic_staggered_data.copy()
        data['ydot_g3_r4'] = 999
        
        with pytest.raises(ValueError) as exc_info:
            transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        error_msg = str(exc_info.value)
        assert 'ydot_g3_r4' in error_msg
    
    def test_demean_no_conflict_with_different_prefix(self, basic_staggered_data):
        """Test that columns with different prefixes don't cause conflict."""
        data = basic_staggered_data.copy()
        data['ycheck_g3_r3'] = 999  # Different prefix, should NOT conflict
        data['xdot_g3_r3'] = 888    # Different prefix
        data['ydot_other'] = 777    # Different format
        
        # Should work without error
        result = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        assert 'ydot_g3_r3' in result.columns
        assert 'ycheck_g3_r3' in result.columns  # Original preserved
    
    def test_demean_basic_no_conflict(self, basic_staggered_data):
        """Test that normal data without conflicts works fine."""
        result = transform_staggered_demean(
            basic_staggered_data, 'y', 'id', 'year', 'gvar'
        )
        
        # Should have created expected columns
        assert 'ydot_g3_r3' in result.columns
        assert 'ydot_g3_r4' in result.columns
        assert 'ydot_g4_r4' in result.columns
        
        # Verify values
        unit1_r3 = result.loc[(result['id'] == 1) & (result['year'] == 3), 'ydot_g3_r3'].iloc[0]
        assert np.isclose(unit1_r3, 3)  # 14 - 11 = 3


# =============================================================================
# Test 2: Detrend Column Conflict Detection
# =============================================================================

class TestDetrendColumnConflict:
    """Tests for column conflict detection in transform_staggered_detrend()."""
    
    def test_detrend_column_conflict_raises_error(self, detrend_data):
        """Test that existing ycheck column raises ValueError."""
        data = detrend_data.copy()
        data['ycheck_g4_r4'] = 999  # This will conflict
        
        with pytest.raises(ValueError, match="Column name conflict detected"):
            transform_staggered_detrend(data, 'y', 'id', 'year', 'gvar')
    
    def test_detrend_multiple_conflicts(self, detrend_data):
        """Test that multiple conflicting columns are detected."""
        data = detrend_data.copy()
        data['ycheck_g4_r4'] = 999
        data['ycheck_g4_r5'] = 888
        data['ycheck_g4_r6'] = 777
        
        with pytest.raises(ValueError, match="Column name conflict detected"):
            transform_staggered_detrend(data, 'y', 'id', 'year', 'gvar')
    
    def test_detrend_error_message_contains_column_names(self, detrend_data):
        """Test that error message lists the conflicting column names."""
        data = detrend_data.copy()
        data['ycheck_g4_r5'] = 999
        
        with pytest.raises(ValueError) as exc_info:
            transform_staggered_detrend(data, 'y', 'id', 'year', 'gvar')
        
        error_msg = str(exc_info.value)
        assert 'ycheck_g4_r5' in error_msg
    
    def test_detrend_no_conflict_with_different_prefix(self, detrend_data):
        """Test that columns with different prefixes don't cause conflict."""
        data = detrend_data.copy()
        data['ydot_g4_r4'] = 999     # Different prefix (demean format)
        data['xcheck_g4_r4'] = 888   # Different prefix
        
        # Should work without error
        result = transform_staggered_detrend(data, 'y', 'id', 'year', 'gvar')
        
        assert 'ycheck_g4_r4' in result.columns
        assert 'ydot_g4_r4' in result.columns  # Original preserved
    
    def test_detrend_basic_no_conflict(self, detrend_data):
        """Test that normal data without conflicts works fine."""
        result = transform_staggered_detrend(
            detrend_data, 'y', 'id', 'year', 'gvar'
        )
        
        # Should have created expected columns
        assert 'ycheck_g4_r4' in result.columns
        assert 'ycheck_g4_r5' in result.columns
        assert 'ycheck_g4_r6' in result.columns
        
        # Verify detrending (unit 1: Y = 10 + 2*t, effect = 10)
        ycheck_u1_r4 = result.loc[(result['id'] == 1) & (result['year'] == 4), 'ycheck_g4_r4'].iloc[0]
        assert np.isclose(ycheck_u1_r4, 10, atol=0.1)


# =============================================================================
# Test 3: Boundary Cases
# =============================================================================

class TestBoundaryConflictCases:
    """Tests for boundary conditions in column conflict detection."""
    
    def test_single_cohort_single_period_conflict(self):
        """Test conflict detection with single cohort and single post period."""
        data = pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2],
            'year': [1, 2, 3, 1, 2, 3],  # T_max = 3
            'y': [10, 12, 20, 5, 6, 8],
            'gvar': [3, 3, 3, 0, 0, 0],  # cohort 3, only r=3 post period
            'ydot_g3_r3': [0, 0, 0, 0, 0, 0]  # Conflict!
        })
        
        with pytest.raises(ValueError, match="ydot_g3_r3"):
            transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
    
    def test_all_columns_conflict(self, basic_staggered_data):
        """Test when all generated columns would conflict."""
        data = basic_staggered_data.copy()
        # Add all expected columns
        data['ydot_g3_r3'] = 0
        data['ydot_g3_r4'] = 0
        data['ydot_g4_r4'] = 0
        
        with pytest.raises(ValueError) as exc_info:
            transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        error_msg = str(exc_info.value)
        assert "Total conflicts: 3" in error_msg
    
    def test_case_sensitivity(self, basic_staggered_data):
        """Test that column name matching is case-sensitive."""
        data = basic_staggered_data.copy()
        # Different case should NOT conflict
        data['YDOT_G3_R3'] = 999
        data['Ydot_g3_r3'] = 888
        
        # Should work without error (case-sensitive comparison)
        result = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        assert 'ydot_g3_r3' in result.columns
        assert 'YDOT_G3_R3' in result.columns  # Original preserved
    
    def test_conflict_count_more_than_five(self):
        """Test that error message truncates when more than 5 conflicts."""
        data = pd.DataFrame({
            'id': [1] * 10 + [2] * 10,
            'year': list(range(1, 11)) * 2,
            'y': [10 + i for i in range(10)] * 2,
            'gvar': [3] * 10 + [0] * 10  # cohort 3, post periods 3-10 (8 periods)
        })
        
        # Add 6 conflicting columns
        for r in range(3, 9):
            data[f'ydot_g3_r{r}'] = 0
        
        with pytest.raises(ValueError) as exc_info:
            transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        error_msg = str(exc_info.value)
        assert "..." in error_msg  # Truncation indicator
        assert "Total conflicts: 6" in error_msg
    
    def test_partial_column_conflict_demean(self, basic_staggered_data):
        """Test partial conflict (only some columns conflict)."""
        data = basic_staggered_data.copy()
        data['ydot_g3_r3'] = 999  # Only 1 of 3 conflicts
        
        with pytest.raises(ValueError) as exc_info:
            transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        error_msg = str(exc_info.value)
        assert "ydot_g3_r3" in error_msg
        assert "Total conflicts: 1" in error_msg
    
    def test_partial_column_conflict_detrend(self, detrend_data):
        """Test partial conflict for detrend function."""
        data = detrend_data.copy()
        data['ycheck_g4_r5'] = 999  # Only 1 of 3 conflicts
        
        with pytest.raises(ValueError) as exc_info:
            transform_staggered_detrend(data, 'y', 'id', 'year', 'gvar')
        
        error_msg = str(exc_info.value)
        assert "ycheck_g4_r5" in error_msg
        assert "Total conflicts: 1" in error_msg


# =============================================================================
# Test 4: Integration with Real Data Patterns
# =============================================================================

class TestRealDataPatterns:
    """Tests mimicking real-world data patterns."""
    
    def test_castle_law_style_data_no_conflict(self):
        """Test with Castle Law style data (year cohorts like 2005, 2006)."""
        data = pd.DataFrame({
            'id': [1] * 5 + [2] * 5 + [3] * 5 + [4] * 5,
            'year': [2003, 2004, 2005, 2006, 2007] * 4,
            'y': [10, 11, 15, 16, 17,   # cohort 2005
                  20, 21, 22, 28, 29,   # cohort 2006
                  30, 31, 32, 33, 40,   # cohort 2007
                  40, 41, 42, 43, 44],  # never treated
            'gvar': [2005] * 5 + [2006] * 5 + [2007] * 5 + [0] * 5
        })
        
        # Should work without error
        result = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        # Verify expected columns
        assert 'ydot_g2005_r2005' in result.columns
        assert 'ydot_g2006_r2006' in result.columns
        assert 'ydot_g2007_r2007' in result.columns
    
    def test_castle_law_style_with_conflict(self):
        """Test that conflict is detected in Castle Law style data."""
        data = pd.DataFrame({
            'id': [1] * 5 + [2] * 5,
            'year': [2003, 2004, 2005, 2006, 2007] * 2,
            'y': [10, 11, 15, 16, 17, 40, 41, 42, 43, 44],
            'gvar': [2005] * 5 + [0] * 5,
            'ydot_g2005_r2006': [0] * 10  # Conflict!
        })
        
        with pytest.raises(ValueError, match="ydot_g2005_r2006"):
            transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
    
    def test_user_data_with_similar_but_different_columns(self):
        """Test that similar but different column names don't conflict."""
        data = pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2],
            'year': [1, 2, 3, 1, 2, 3],
            'y': [10, 12, 20, 5, 6, 8],
            'gvar': [3, 3, 3, 0, 0, 0],
            # Similar but different patterns
            'ydot_g3': 0,          # Missing _r part
            'ydot_r3': 0,          # Missing _g part
            'ydot_g33_r3': 0,      # Different cohort number
            'ydot_g3_r33': 0,      # Different period number
            'my_ydot_g3_r3': 0,    # Has prefix
            'ydot_g3_r3_extra': 0, # Has suffix
        })
        
        # Should work without error
        result = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        assert 'ydot_g3_r3' in result.columns
        # All original columns preserved
        assert 'ydot_g3' in result.columns
        assert 'ydot_r3' in result.columns


# =============================================================================
# Test 5: Error Message Quality
# =============================================================================

class TestErrorMessageQuality:
    """Tests for the quality and clarity of error messages."""
    
    def test_error_message_mentions_function_name_demean(self, basic_staggered_data):
        """Test that demean error mentions the correct function."""
        data = basic_staggered_data.copy()
        data['ydot_g3_r3'] = 0
        
        with pytest.raises(ValueError) as exc_info:
            transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        assert "transform_staggered_demean" in str(exc_info.value)
    
    def test_error_message_mentions_function_name_detrend(self, detrend_data):
        """Test that detrend error mentions the correct function."""
        data = detrend_data.copy()
        data['ycheck_g4_r4'] = 0
        
        with pytest.raises(ValueError) as exc_info:
            transform_staggered_detrend(data, 'y', 'id', 'year', 'gvar')
        
        assert "transform_staggered_detrend" in str(exc_info.value)
    
    def test_error_message_provides_guidance(self, basic_staggered_data):
        """Test that error message tells user how to fix the issue."""
        data = basic_staggered_data.copy()
        data['ydot_g3_r3'] = 0
        
        with pytest.raises(ValueError) as exc_info:
            transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        error_msg = str(exc_info.value)
        assert "rename" in error_msg.lower() or "Please" in error_msg


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
