"""
Unit tests for BUG-203, BUG-204, and BUG-205 fixes.

BUG-203: aggregation.py controls missing value handling
BUG-204: control_groups.py cohort rounding consistency
BUG-205: results.py LaTeX escape function double-escaping
"""

import warnings
import numpy as np
import pandas as pd
import pytest


class TestBug203ControlsMissingValues:
    """Test BUG-203: Controls missing value handling in aggregation functions."""

    @pytest.fixture
    def staggered_data_with_missing_controls(self):
        """Create staggered panel data with missing values in control variable."""
        np.random.seed(42)
        n_units = 20
        n_periods = 8
        
        data = []
        for i in range(n_units):
            if i < 5:
                gvar = np.inf  # Never-treated
            elif i < 10:
                gvar = 2004  # Cohort 2004
            elif i < 15:
                gvar = 2005  # Cohort 2005
            else:
                gvar = 2006  # Cohort 2006
            
            for t in range(2000, 2000 + n_periods):
                y = np.random.normal(10 + i * 0.5, 1)
                if gvar != np.inf and t >= gvar:
                    y += 2  # Treatment effect
                
                # Control variable with some missing values
                ctrl = np.random.normal(5, 1) if np.random.random() > 0.1 else np.nan
                
                data.append({
                    'id': i,
                    'year': t,
                    'y': y,
                    'gvar': gvar,
                    'ctrl': ctrl,
                })
        
        return pd.DataFrame(data)

    def test_aggregate_to_cohort_handles_missing_controls(self, staggered_data_with_missing_controls):
        """Test that aggregate_to_cohort properly drops rows with missing controls."""
        from lwdid.staggered.aggregation import aggregate_to_cohort
        from lwdid.staggered.transformations import transform_staggered_demean
        
        data = staggered_data_with_missing_controls
        
        # Apply transformation
        data_transformed = transform_staggered_demean(
            data=data,
            gvar='gvar',
            ivar='id',
            tvar='year',
            y='y',
        )
        
        # This should not raise an error when controls have NaN
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = aggregate_to_cohort(
                data_transformed=data_transformed,
                gvar='gvar',
                ivar='id',
                tvar='year',
                cohorts=[2004, 2005, 2006],
                T_max=2007,
                transform_type='demean',
                controls=['ctrl'],
            )
            
            # Should get valid results
            assert len(results) > 0
            for effect in results:
                assert np.isfinite(effect.att)

    def test_aggregate_to_overall_handles_missing_controls(self, staggered_data_with_missing_controls):
        """Test that aggregate_to_overall properly drops rows with missing controls."""
        from lwdid.staggered.aggregation import aggregate_to_overall
        from lwdid.staggered.transformations import transform_staggered_demean
        
        data = staggered_data_with_missing_controls
        
        # Apply transformation
        data_transformed = transform_staggered_demean(
            data=data,
            gvar='gvar',
            ivar='id',
            tvar='year',
            y='y',
        )
        
        # This should not raise an error when controls have NaN
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = aggregate_to_overall(
                data_transformed=data_transformed,
                gvar='gvar',
                ivar='id',
                tvar='year',
                transform_type='demean',
                controls=['ctrl'],
            )
            
            # Should get valid result
            assert np.isfinite(result.att)
            assert np.isfinite(result.se)


class TestBug204CohortRoundingConsistency:
    """Test BUG-204: Cohort rounding consistency in control_groups.py."""

    def test_get_all_control_masks_integer_cohorts(self):
        """Test that integer cohorts work correctly."""
        from lwdid.staggered.control_groups import get_all_control_masks, ControlGroupStrategy
        
        # Create simple panel data
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3, 4, 4],
            'year': [2000, 2001] * 4,
            'gvar': [2001, 2001, 2002, 2002, np.inf, np.inf, np.inf, np.inf],
        })
        
        masks = get_all_control_masks(
            data=data,
            gvar='gvar',
            ivar='id',
            cohorts=[2001],
            T_max=2002,
            strategy=ControlGroupStrategy.NOT_YET_TREATED,
        )
        
        # Should have masks for cohort 2001, periods 2001 and 2002
        assert (2001, 2001.0) in masks
        assert (2001, 2002.0) in masks

    def test_get_all_control_masks_warns_on_non_integer_cohort(self):
        """Test that non-integer cohorts trigger a warning."""
        from lwdid.staggered.control_groups import get_all_control_masks, ControlGroupStrategy
        
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3],
            'year': [2000, 2001] * 3,
            'gvar': [2001.6, 2001.6, np.inf, np.inf, np.inf, np.inf],
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            masks = get_all_control_masks(
                data=data,
                gvar='gvar',
                ivar='id',
                cohorts=[2001.6],
                T_max=2003,
                strategy=ControlGroupStrategy.NOT_YET_TREATED,
            )
            
            # Should issue a warning about non-integer cohort
            warning_messages = [str(warning.message) for warning in w]
            assert any('not an integer' in msg or 'Rounding' in msg for msg in warning_messages), \
                f"Expected warning about non-integer cohort, got: {warning_messages}"

    def test_cohort_rounding_consistency_with_safe_int_cohort(self):
        """Test that cohort handling is consistent with safe_int_cohort behavior."""
        from lwdid.validation import safe_int_cohort, COHORT_FLOAT_TOLERANCE
        
        # Integer values should pass through unchanged
        assert safe_int_cohort(2005) == 2005
        assert safe_int_cohort(2005.0) == 2005
        
        # Values very close to integer should be accepted
        assert safe_int_cohort(2005.0 + COHORT_FLOAT_TOLERANCE / 2) == 2005
        
        # Non-integer values should raise ValueError
        with pytest.raises(ValueError):
            safe_int_cohort(2005.5)
        
        with pytest.raises(ValueError):
            safe_int_cohort(2005.1)


class TestBug205LatexEscapeDoubleEscaping:
    """Test BUG-205: LaTeX escape function double-escaping prevention."""

    def test_latex_escape_basic_characters(self):
        """Test basic special character escaping."""
        from lwdid.results import _latex_escape_string
        
        # Underscore
        assert _latex_escape_string('price_index') == r'price\_index'
        
        # Percent
        assert _latex_escape_string('rate%') == r'rate\%'
        
        # Ampersand
        assert _latex_escape_string('A&B') == r'A\&B'
        
        # Hash
        assert _latex_escape_string('item#1') == r'item\#1'
        
        # Dollar
        assert _latex_escape_string('$value') == r'\$value'

    def test_latex_escape_no_double_escaping(self):
        """Test that already-escaped characters are not double-escaped."""
        from lwdid.results import _latex_escape_string
        
        # Already escaped underscore should not be double-escaped
        assert _latex_escape_string(r'price\_index') == r'price\_index'
        
        # Already escaped percent should not be double-escaped
        assert _latex_escape_string(r'rate\%') == r'rate\%'
        
        # Already escaped ampersand should not be double-escaped
        assert _latex_escape_string(r'A\&B') == r'A\&B'
        
        # Already escaped hash should not be double-escaped
        assert _latex_escape_string(r'item\#1') == r'item\#1'
        
        # Already escaped dollar should not be double-escaped
        assert _latex_escape_string(r'\$value') == r'\$value'

    def test_latex_escape_mixed_content(self):
        """Test strings with mix of escaped and unescaped characters."""
        from lwdid.results import _latex_escape_string
        
        # Mix of escaped underscore and unescaped ampersand
        result = _latex_escape_string(r'price\_index & rate')
        assert result == r'price\_index \& rate'
        
        # Multiple unescaped characters
        result = _latex_escape_string('col_1 & col_2 % notes')
        assert result == r'col\_1 \& col\_2 \% notes'

    def test_latex_escape_empty_string(self):
        """Test empty string handling."""
        from lwdid.results import _latex_escape_string
        
        assert _latex_escape_string('') == ''

    def test_latex_escape_no_special_chars(self):
        """Test string with no special characters."""
        from lwdid.results import _latex_escape_string
        
        assert _latex_escape_string('simple_text') == r'simple\_text'
        assert _latex_escape_string('nospecialchars') == 'nospecialchars'


class TestIntegration:
    """Integration tests for all three bug fixes."""

    def test_full_staggered_workflow_with_controls(self):
        """Test complete staggered DiD workflow with controls containing NaN."""
        from lwdid import lwdid
        
        np.random.seed(123)
        n_units = 30
        n_periods = 10
        
        data = []
        for i in range(n_units):
            if i < 8:
                gvar = np.inf
            elif i < 16:
                gvar = 2005
            elif i < 24:
                gvar = 2006
            else:
                gvar = 2007
            
            for t in range(2000, 2000 + n_periods):
                y = 10 + i * 0.3 + t * 0.1 + np.random.normal(0, 0.5)
                if gvar != np.inf and t >= gvar:
                    y += 2
                
                # Control with occasional missing values
                ctrl = np.random.normal(5, 1) if np.random.random() > 0.05 else np.nan
                
                data.append({
                    'id': i,
                    'year': t,
                    'y': y,
                    'gvar': gvar,
                    'ctrl': ctrl,
                })
        
        df = pd.DataFrame(data)
        
        # Should complete without error
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            results = lwdid(
                data=df,
                y='y',
                gvar='gvar',
                ivar='id',
                tvar='year',
                rolling='demean',
                aggregate='cohort',
                controls=['ctrl'],
            )
            
            # Verify results are valid
            assert results is not None
            assert results.att_by_cohort is not None
            assert len(results.att_by_cohort) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
