"""
Tests for staggered DiD data validation functions.

Validates ``is_never_treated()`` and ``validate_staggered_data()`` for
correct enforcement of panel structure requirements in staggered adoption
designs.

Validates the input validation layer (Section 4, data requirements) of the
Lee-Wooldridge Difference-in-Differences framework.

References
----------
Lee, S. & Wooldridge, J. M. (2025). A Simple Transformation Approach to
    Difference-in-Differences Estimation for Panel Data. SSRN 4516518.
"""

import numpy as np
import pandas as pd
import pytest

from lwdid.validation import is_never_treated, validate_staggered_data
from lwdid.exceptions import InvalidStaggeredDataError, MissingRequiredColumnError


# =============================================================================
# TestIsNeverTreated - Tests for is_never_treated() helper function
# =============================================================================

class TestIsNeverTreated:
    """Test is_never_treated() helper function"""
    
    def test_zero_is_never_treated(self):
        """0 should be identified as never treated"""
        assert is_never_treated(0) is True
        assert is_never_treated(0.0) is True
    
    def test_inf_is_never_treated(self):
        """np.inf should be identified as never treated"""
        assert is_never_treated(np.inf) is True
        assert is_never_treated(float('inf')) is True
    
    def test_nan_is_never_treated(self):
        """NaN should be identified as never treated"""
        assert is_never_treated(np.nan) is True
        assert is_never_treated(float('nan')) is True
    
    def test_none_is_never_treated(self):
        """None should be identified as never treated"""
        assert is_never_treated(None) is True
    
    def test_positive_int_is_treated(self):
        """Positive integers should be identified as treated"""
        assert is_never_treated(2005) is False
        assert is_never_treated(2006) is False
        assert is_never_treated(1) is False
    
    def test_positive_float_is_treated(self):
        """Positive floats should be identified as treated"""
        assert is_never_treated(2005.0) is False
        assert is_never_treated(0.1) is False
    
    def test_negative_is_treated(self):
        """Negative values should be identified as treated (invalid, but not NT)
        
        Note: Negative values are rejected by validate_staggered_data(),
        but is_never_treated() itself doesn't handle this check.
        """
        assert is_never_treated(-1) is False
    
    def test_negative_inf_raises_error(self):
        """-np.inf should raise InvalidStaggeredDataError.
        
        Note: Negative infinity is not a valid gvar value. The function
        explicitly rejects it to prevent ambiguity. Use 0, np.inf, or NaN
        to indicate never-treated units.
        """
        with pytest.raises(InvalidStaggeredDataError, match="Negative infinity"):
            is_never_treated(-np.inf)


# =============================================================================
# TestValidateStaggeredData - Core validation function tests
# =============================================================================

class TestValidateStaggeredData:
    """Test validate_staggered_data() main function"""
    
    @pytest.fixture
    def basic_staggered_data(self):
        """Basic staggered data fixture"""
        return pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'year': [2000, 2001, 2002, 2000, 2001, 2002, 2000, 2001, 2002],
            'y': [1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 1.2, 2.2, 3.2],
            'gvar': [2001, 2001, 2001, 2002, 2002, 2002, 0, 0, 0]
        })
    
    def test_basic_validation_success(self, basic_staggered_data):
        """Basic validation should pass"""
        result = validate_staggered_data(
            data=basic_staggered_data,
            gvar='gvar',
            ivar='id',
            tvar='year',
            y='y'
        )
        
        assert result['cohorts'] == [2001, 2002]
        assert result['n_cohorts'] == 2
        assert result['n_treated'] == 2
        assert result['n_never_treated'] == 1
        assert result['has_never_treated'] is True
        assert result['T_min'] == 2000
        assert result['T_max'] == 2002
        assert result['N_total'] == 3
        assert result['N_obs'] == 9
    
    def test_missing_gvar_column_raises(self, basic_staggered_data):
        """Missing gvar column should raise MissingRequiredColumnError"""
        with pytest.raises(MissingRequiredColumnError, match="not found in data"):
            validate_staggered_data(
                data=basic_staggered_data,
                gvar='nonexistent',
                ivar='id',
                tvar='year',
                y='y'
            )
    
    def test_string_gvar_raises(self, basic_staggered_data):
        """String type gvar should raise InvalidStaggeredDataError"""
        data = basic_staggered_data.copy()
        data['gvar'] = data['gvar'].astype(str)
        
        with pytest.raises(InvalidStaggeredDataError, match="must be numeric"):
            validate_staggered_data(data, 'gvar', 'id', 'year', 'y')
    
    def test_negative_gvar_raises(self, basic_staggered_data):
        """Negative gvar values should raise InvalidStaggeredDataError"""
        data = basic_staggered_data.copy()
        data.loc[data['id'] == 1, 'gvar'] = -2001
        
        with pytest.raises(InvalidStaggeredDataError, match="negative values"):
            validate_staggered_data(data, 'gvar', 'id', 'year', 'y')
    
    def test_no_cohorts_raises(self):
        """All units being never treated should raise InvalidStaggeredDataError"""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'year': [2000, 2001, 2000, 2001],
            'y': [1.0, 2.0, 1.5, 2.5],
            'gvar': [0, 0, 0, 0]  # All NT
        })
        
        with pytest.raises(InvalidStaggeredDataError, match="No treatment cohorts"):
            validate_staggered_data(data, 'gvar', 'id', 'year', 'y')
    
    def test_no_never_treated_warning(self):
        """No NT units should emit warning"""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'year': [2000, 2001, 2000, 2001],
            'y': [1.0, 2.0, 1.5, 2.5],
            'gvar': [2001, 2001, 2001, 2001]  # All treated
        })
        
        result = validate_staggered_data(data, 'gvar', 'id', 'year', 'y')
        
        assert result['n_never_treated'] == 0
        assert result['has_never_treated'] is False
        assert len(result['warnings']) > 0
        assert any('No never-treated' in w for w in result['warnings'])
    
    def test_unbalanced_panel_warning(self):
        """Unbalanced panel should emit warning"""
        data = pd.DataFrame({
            'id': [1, 1, 1, 2, 2],  # id=2 has only 2 periods
            'year': [2000, 2001, 2002, 2000, 2001],
            'y': [1.0, 2.0, 3.0, 1.5, 2.5],
            'gvar': [2001, 2001, 2001, 0, 0]
        })
        
        result = validate_staggered_data(data, 'gvar', 'id', 'year', 'y')
        
        assert len(result['warnings']) > 0
        assert any('Unbalanced panel' in w for w in result['warnings'])
    
    def test_inf_recognized_as_nt(self):
        """np.inf should be recognized as never treated"""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'year': [2000, 2001, 2000, 2001],
            'y': [1.0, 2.0, 1.5, 2.5],
            'gvar': [2001, 2001, np.inf, np.inf]
        })
        
        result = validate_staggered_data(data, 'gvar', 'id', 'year', 'y')
        
        assert result['n_never_treated'] == 1
        assert result['n_treated'] == 1
    
    def test_nan_recognized_as_nt(self):
        """NaN should be recognized as never treated"""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'year': [2000, 2001, 2000, 2001],
            'y': [1.0, 2.0, 1.5, 2.5],
            'gvar': [2001, 2001, np.nan, np.nan]
        })
        
        result = validate_staggered_data(data, 'gvar', 'id', 'year', 'y')
        
        assert result['n_never_treated'] == 1
        assert result['n_treated'] == 1
    
    def test_cohorts_sorted_ascending(self, basic_staggered_data):
        """Cohorts should be sorted in ascending order"""
        result = validate_staggered_data(
            data=basic_staggered_data,
            gvar='gvar',
            ivar='id',
            tvar='year',
            y='y'
        )
        
        assert result['cohorts'] == sorted(result['cohorts'])
    
    def test_cohort_sizes_correct(self, basic_staggered_data):
        """Cohort sizes should be correct"""
        result = validate_staggered_data(
            data=basic_staggered_data,
            gvar='gvar',
            ivar='id',
            tvar='year',
            y='y'
        )
        
        assert result['cohort_sizes'] == {2001: 1, 2002: 1}
    
    def test_y_missing_warning(self, basic_staggered_data):
        """Missing values in y should emit warning"""
        data = basic_staggered_data.copy()
        data.loc[0, 'y'] = np.nan
        
        result = validate_staggered_data(
            data=data,
            gvar='gvar',
            ivar='id',
            tvar='year',
            y='y'
        )
        
        assert any('missing values' in w for w in result['warnings'])
    
    def test_earliest_cohort_no_pretreatment_warning(self):
        """Earliest cohort with no pre-treatment period should emit warning"""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'year': [2000, 2001, 2000, 2001],
            'y': [1.0, 2.0, 1.5, 2.5],
            'gvar': [2000, 2000, 0, 0]  # Cohort starts at T_min
        })
        
        result = validate_staggered_data(data, 'gvar', 'id', 'year', 'y')
        
        assert any('no pre-treatment' in w for w in result['warnings'])


# =============================================================================
# TestCastleLawValidation - Castle Law data end-to-end tests
# =============================================================================

class TestCastleLawValidation:
    """Castle Law data validation tests"""
    
    def test_castle_validation_passes(self, castle_data):
        """Castle Law data should pass validation"""
        result = validate_staggered_data(
            data=castle_data,
            gvar='gvar',
            ivar='sid',
            tvar='year',
            y='lhomicide'
        )
        
        # Should not raise exception
        assert result is not None
    
    def test_castle_cohorts_correct(self, castle_data):
        """Castle Law should have 5 correct cohorts"""
        result = validate_staggered_data(
            data=castle_data,
            gvar='gvar',
            ivar='sid',
            tvar='year',
            y='lhomicide'
        )
        
        expected_cohorts = [2005, 2006, 2007, 2008, 2009]
        assert result['cohorts'] == expected_cohorts
        assert result['n_cohorts'] == 5
    
    def test_castle_cohort_sizes_correct(self, castle_data):
        """Castle Law cohort sizes should be correct"""
        result = validate_staggered_data(
            data=castle_data,
            gvar='gvar',
            ivar='sid',
            tvar='year',
            y='lhomicide'
        )
        
        expected_sizes = {2005: 1, 2006: 13, 2007: 4, 2008: 2, 2009: 1}
        assert result['cohort_sizes'] == expected_sizes
    
    def test_castle_treated_count_correct(self, castle_data):
        """Castle Law treated/NT counts should be correct"""
        result = validate_staggered_data(
            data=castle_data,
            gvar='gvar',
            ivar='sid',
            tvar='year',
            y='lhomicide'
        )
        
        assert result['n_treated'] == 21
        assert result['n_never_treated'] == 29
        assert result['has_never_treated'] is True
    
    def test_castle_time_range_correct(self, castle_data):
        """Castle Law time range should be correct"""
        result = validate_staggered_data(
            data=castle_data,
            gvar='gvar',
            ivar='sid',
            tvar='year',
            y='lhomicide'
        )
        
        assert result['T_min'] == 2000
        assert result['T_max'] == 2010
    
    def test_castle_total_units_correct(self, castle_data):
        """Castle Law total units should be correct"""
        result = validate_staggered_data(
            data=castle_data,
            gvar='gvar',
            ivar='sid',
            tvar='year',
            y='lhomicide'
        )
        
        # N_total = 50 states
        assert result['N_total'] == 50
        # N_obs = 50 states × 11 years = 550
        assert result['N_obs'] == 550
        # Cross-check: treated + NT = total
        assert result['n_treated'] + result['n_never_treated'] == 50
    
    def test_castle_dinf_consistency(self, castle_data):
        """Verify gvar and dinf column consistency"""
        # dinf=1 indicates never treated
        n_nt_from_dinf = castle_data.groupby('sid')['dinf'].first().sum()
        
        result = validate_staggered_data(
            data=castle_data,
            gvar='gvar',
            ivar='sid',
            tvar='year',
            y='lhomicide'
        )
        
        assert result['n_never_treated'] == n_nt_from_dinf


# =============================================================================
# TestValidationEdgeCases - Edge case tests
# =============================================================================

class TestValidationEdgeCases:
    """Edge case tests"""
    
    def test_single_cohort(self):
        """Single cohort should work correctly"""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'year': [2000, 2001, 2000, 2001],
            'y': [1.0, 2.0, 1.5, 2.5],
            'gvar': [2001, 2001, 0, 0]
        })
        
        result = validate_staggered_data(data, 'gvar', 'id', 'year', 'y')
        
        assert result['cohorts'] == [2001]
        assert result['n_cohorts'] == 1
    
    def test_large_cohort_numbers(self):
        """Large cohort year values should work correctly"""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'year': [2020, 2025, 2020, 2025],
            'y': [1.0, 2.0, 1.5, 2.5],
            'gvar': [2025, 2025, 0, 0]
        })
        
        result = validate_staggered_data(data, 'gvar', 'id', 'year', 'y')
        
        assert result['cohorts'] == [2025]
    
    def test_float_cohort_values(self):
        """Float cohort values should be converted to integers"""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'year': [2000, 2001, 2000, 2001],
            'y': [1.0, 2.0, 1.5, 2.5],
            'gvar': [2001.0, 2001.0, 0.0, 0.0]
        })
        
        result = validate_staggered_data(data, 'gvar', 'id', 'year', 'y')
        
        # Cohorts should be integer list
        assert result['cohorts'] == [2001]
        assert all(isinstance(g, int) for g in result['cohorts'])
    
    def test_mixed_nt_representations(self):
        """Mixed NT representations should all be identified"""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3, 4, 4],
            'year': [2000, 2001] * 4,
            'y': [1.0, 2.0] * 4,
            'gvar': [2001, 2001, 0, 0, np.inf, np.inf, np.nan, np.nan]
        })
        
        result = validate_staggered_data(data, 'gvar', 'id', 'year', 'y')
        
        assert result['n_treated'] == 1
        assert result['n_never_treated'] == 3  # 0, inf, nan each have 1 unit
    
    def test_empty_warnings_when_valid(self):
        """Fully valid data should have no warnings"""
        data = pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'year': [2000, 2001, 2002, 2000, 2001, 2002, 2000, 2001, 2002],
            'y': [1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 1.2, 2.2, 3.2],
            'gvar': [2002, 2002, 2002, 0, 0, 0, 0, 0, 0]  # Has NT, has pre-period
        })
        
        result = validate_staggered_data(data, 'gvar', 'id', 'year', 'y')
        
        # Should have no warnings
        assert len(result['warnings']) == 0
    
    def test_gvar_time_varying_raises(self):
        """gvar varying within unit should raise error"""
        data = pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2],
            'year': [2000, 2001, 2002, 2000, 2001, 2002],
            'y': [1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
            'gvar': [2001, 2002, 2002, 0, 0, 0]  # id=1's gvar is inconsistent!
        })
        
        with pytest.raises(InvalidStaggeredDataError, match="time-invariant"):
            validate_staggered_data(data, 'gvar', 'id', 'year', 'y')
    
    def test_negative_inf_raises_error(self):
        """-np.inf should trigger negative value check error
        
        Key behavior:
        - The negative value check (line ~1190) executes before is_never_treated()
        - `-np.inf < 0` is True, so it's identified as negative
        - Therefore -np.inf raises InvalidStaggeredDataError, not treated as NT
        """
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'year': [2000, 2001, 2000, 2001],
            'y': [1.0, 2.0, 1.5, 2.5],
            'gvar': [2001, 2001, -np.inf, -np.inf]
        })
        
        # Current implementation: -np.inf triggers negative value check error
        with pytest.raises(InvalidStaggeredDataError, match="negative"):
            validate_staggered_data(data, 'gvar', 'id', 'year', 'y')
    
    def test_empty_data_raises(self):
        """Empty DataFrame should raise error"""
        data = pd.DataFrame({'id': [], 'year': [], 'y': [], 'gvar': []})
        
        with pytest.raises(InvalidStaggeredDataError, match="empty"):
            validate_staggered_data(data, 'gvar', 'id', 'year', 'y')
    
    def test_controls_missing_raises(self):
        """Missing control columns should raise error"""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'year': [2000, 2001, 2000, 2001],
            'y': [1.0, 2.0, 1.5, 2.5],
            'gvar': [2001, 2001, 0, 0]
        })
        
        with pytest.raises(MissingRequiredColumnError, match="Control variable"):
            validate_staggered_data(data, 'gvar', 'id', 'year', 'y', 
                                   controls=['nonexistent_control'])
    
    def test_cohort_outside_range_warning(self):
        """Cohort outside time range should emit warning"""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'year': [2000, 2001, 2000, 2001],
            'y': [1.0, 2.0, 1.5, 2.5],
            'gvar': [2005, 2005, 0, 0]  # 2005 > T_max=2001
        })
        
        result = validate_staggered_data(data, 'gvar', 'id', 'year', 'y')
        
        assert any('outside' in w for w in result['warnings'])
    
    def test_only_one_nt_warning(self):
        """Only 1 NT unit should emit warning"""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3],
            'year': [2000, 2001, 2000, 2001, 2000, 2001],
            'y': [1.0, 2.0, 1.5, 2.5, 1.2, 2.2],
            'gvar': [2001, 2001, 2001, 2001, 0, 0]  # Only 1 NT unit
        })
        
        result = validate_staggered_data(data, 'gvar', 'id', 'year', 'y')
        
        assert result['n_never_treated'] == 1
        assert any('Only 1 never-treated' in w for w in result['warnings'])


# =============================================================================
# TestModuleImports - Test that exports work correctly
# =============================================================================

class TestModuleImports:
    """Test module imports and exports"""
    
    def test_import_from_lwdid(self):
        """Should be able to import from top-level lwdid module"""
        from lwdid import is_never_treated, validate_staggered_data
        from lwdid import InvalidStaggeredDataError, NoNeverTreatedError
        
        # Verify functions work
        assert is_never_treated(0) is True
        assert callable(validate_staggered_data)
    
    def test_exception_inheritance(self):
        """Exception classes should have correct inheritance"""
        from lwdid import InvalidStaggeredDataError, NoNeverTreatedError, LWDIDError
        
        # InvalidStaggeredDataError should inherit from LWDIDError
        assert issubclass(InvalidStaggeredDataError, LWDIDError)
        
        # NoNeverTreatedError should inherit from InsufficientDataError
        from lwdid import InsufficientDataError
        assert issubclass(NoNeverTreatedError, InsufficientDataError)
    
    def test_exception_can_be_caught(self):
        """Exceptions should be catchable by parent class"""
        from lwdid import LWDIDError
        
        data = pd.DataFrame({'id': [], 'year': [], 'y': [], 'gvar': []})
        
        with pytest.raises(LWDIDError):
            validate_staggered_data(data, 'gvar', 'id', 'year', 'y')
