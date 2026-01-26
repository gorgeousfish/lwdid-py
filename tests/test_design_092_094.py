"""
Tests for DESIGN-092 and DESIGN-094 fixes.

DESIGN-092: get_all_control_masks() T_min parameter removal
    - T_min parameter has been removed per paper algorithm (control group
      selection does not depend on T_min, only on current period r and T_max)
    - Stata reference implementations do not use T_min for control groups
    - Function should work correctly without T_min parameter

DESIGN-094: detrendq_unit() degrees of freedom check consistency
    - Changed from n_valid < n_params to n_valid <= n_params
    - Ensures df >= 1 for reliable statistical inference
    - Consistent with demeanq_unit() and apply_rolling_transform()
"""

import warnings
import numpy as np
import pandas as pd
import pytest

from lwdid.staggered.control_groups import (
    get_all_control_masks,
    ControlGroupStrategy,
)
from lwdid.transformations import detrendq_unit, demeanq_unit


class TestDESIGN092_TMinParameterRemoval:
    """Tests for DESIGN-092: T_min parameter removal.
    
    The T_min parameter was removed from get_all_control_masks() because:
    1. Paper algorithm: Control group A_{r+1} = D_{r+1} + ... + D_T + D_inf
       depends only on period r and T_max, not T_min
    2. Stata implementations do not use T_min for control group selection
    3. T_min is only relevant in transformation phase (demean/detrend)
    """

    @pytest.fixture
    def basic_panel_data(self):
        """Create basic panel data for control group tests."""
        return pd.DataFrame({
            'id': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            'year': [2000, 2001, 2002, 2003] * 3,
            'y': [10, 12, 14, 16, 20, 22, 24, 26, 30, 32, 34, 36],
            'gvar': [2002, 2002, 2002, 2002,  # Unit 1: treated at 2002
                     2003, 2003, 2003, 2003,  # Unit 2: treated at 2003
                     0, 0, 0, 0]               # Unit 3: never treated
        })

    def test_function_works_without_t_min(self, basic_panel_data):
        """get_all_control_masks should work correctly without T_min parameter."""
        result = get_all_control_masks(
            data=basic_panel_data,
            gvar='gvar',
            ivar='id',
            cohorts=[2002, 2003],
            T_max=2003,
            strategy=ControlGroupStrategy.NOT_YET_TREATED
        )
        
        # Result should be valid
        assert len(result) > 0, "Function should return non-empty dict"
        
        # Verify expected keys for cohort 2002: periods 2002, 2003
        assert (2002, 2002.0) in result
        assert (2002, 2003.0) in result
        
        # Verify expected keys for cohort 2003: period 2003 only
        assert (2003, 2003.0) in result

    def test_control_masks_are_correct(self, basic_panel_data):
        """Verify control masks are computed correctly per paper algorithm.
        
        For period r, control group is: {i : gvar_i > r} U {i : i in NT}
        """
        result = get_all_control_masks(
            data=basic_panel_data,
            gvar='gvar',
            ivar='id',
            cohorts=[2002],
            T_max=2003,
            strategy=ControlGroupStrategy.NOT_YET_TREATED
        )
        
        # For cohort 2002, period 2002:
        # Control = {gvar > 2002} U {NT} = {unit 2 (gvar=2003)} U {unit 3 (NT)}
        mask_2002_2002 = result[(2002, 2002.0)]
        assert mask_2002_2002[2] == True, "Unit 2 (gvar=2003 > 2002) should be control"
        assert mask_2002_2002[3] == True, "Unit 3 (never-treated) should be control"
        assert mask_2002_2002[1] == False, "Unit 1 (gvar=2002) should not be control"
        
        # For cohort 2002, period 2003:
        # Control = {gvar > 2003} U {NT} = {} U {unit 3 (NT)} = {unit 3}
        mask_2002_2003 = result[(2002, 2003.0)]
        assert mask_2002_2003[3] == True, "Unit 3 (never-treated) should be control"
        assert mask_2002_2003[1] == False, "Unit 1 (gvar=2002) should not be control"
        assert mask_2002_2003[2] == False, "Unit 2 (gvar=2003) should not be control"

    def test_never_treated_strategy(self, basic_panel_data):
        """Verify NEVER_TREATED strategy only includes NT units."""
        result = get_all_control_masks(
            data=basic_panel_data,
            gvar='gvar',
            ivar='id',
            cohorts=[2002],
            T_max=2003,
            strategy=ControlGroupStrategy.NEVER_TREATED
        )
        
        # For any period with NEVER_TREATED strategy:
        # Control = {i : i in NT} only
        for key, mask in result.items():
            assert mask[3] == True, "Unit 3 (never-treated) should be control"
            assert mask[1] == False, "Unit 1 (treated) should not be control"
            assert mask[2] == False, "Unit 2 (treated) should not be control"


class TestDESIGN094_DetrendqDFCheck:
    """Tests for DESIGN-094: detrendq_unit degrees of freedom check."""

    @pytest.fixture
    def unit_data_exactly_k_obs(self):
        """
        Create unit data with exactly k = n_params observations.
        
        For detrendq: y ~ 1 + tindex + i.quarter
        With 2 quarters: k = 1 (intercept) + 1 (tindex) + 1 (Q2 dummy) = 3 params
        With 3 observations: n = k, df = 0 (should be rejected)
        """
        return pd.DataFrame({
            'unit': [1] * 6,
            'tindex': [1, 2, 3, 4, 5, 6],
            'y': [10.0, 12.0, 14.0, 20.0, 25.0, 30.0],
            'quarter': [1, 2, 1, 2, 1, 2],  # 2 unique quarters
            'post': [0, 0, 0, 1, 1, 1]      # 3 pre-treatment obs
        })

    @pytest.fixture
    def unit_data_k_plus_one_obs(self):
        """
        Create unit data with k + 1 observations (df = 1).
        
        With 2 quarters: k = 3 params, need n = 4 for df = 1
        """
        return pd.DataFrame({
            'unit': [1] * 7,
            'tindex': [1, 2, 3, 4, 5, 6, 7],
            'y': [10.0, 12.0, 14.0, 16.0, 25.0, 30.0, 35.0],
            'quarter': [1, 2, 1, 2, 1, 2, 1],  # 2 unique quarters
            'post': [0, 0, 0, 0, 1, 1, 1]       # 4 pre-treatment obs
        })

    def test_detrendq_rejects_exactly_k_observations(self, unit_data_exactly_k_obs):
        """
        detrendq_unit should return NaN when n_valid = n_params (df = 0).
        
        This ensures df >= 1 for reliable variance estimation.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            yhat, ydot = detrendq_unit(
                unit_data_exactly_k_obs,
                y='y',
                tindex='tindex',
                quarter='quarter',
                post='post'
            )
            
            # Should return NaN arrays
            assert np.all(np.isnan(yhat)), \
                "detrendq_unit should return NaN yhat when df = 0"
            assert np.all(np.isnan(ydot)), \
                "detrendq_unit should return NaN ydot when df = 0"
            
            # Should emit warning
            user_warnings = [warning for warning in w 
                            if issubclass(warning.category, UserWarning)]
            assert len(user_warnings) >= 1, \
                "detrendq_unit should warn when df = 0"
            
            # Check warning mentions insufficient observations
            warning_msgs = [str(warning.message) for warning in user_warnings]
            assert any("Insufficient" in msg for msg in warning_msgs), \
                "Warning should mention 'Insufficient'"

    def test_detrendq_accepts_k_plus_one_observations(self, unit_data_k_plus_one_obs):
        """
        detrendq_unit should succeed when n_valid = n_params + 1 (df = 1).
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            yhat, ydot = detrendq_unit(
                unit_data_k_plus_one_obs,
                y='y',
                tindex='tindex',
                quarter='quarter',
                post='post'
            )
            
            # Should return valid arrays (not all NaN)
            assert not np.all(np.isnan(yhat)), \
                "detrendq_unit should return valid yhat when df >= 1"
            assert not np.all(np.isnan(ydot)), \
                "detrendq_unit should return valid ydot when df >= 1"
            
            # Should not emit insufficient observations warning
            user_warnings = [warning for warning in w 
                            if issubclass(warning.category, UserWarning)]
            insufficient_warnings = [w for w in user_warnings 
                                    if "Insufficient" in str(w.message)]
            assert len(insufficient_warnings) == 0, \
                "detrendq_unit should not warn about insufficient obs when df >= 1"

    def test_demeanq_detrendq_consistency(self):
        """
        demeanq_unit and detrendq_unit should use consistent df requirements.
        
        Both should require df >= 1 (n > k).
        """
        # demeanq: y ~ 1 + i.quarter, with 2 quarters: k = 2
        # Need n >= 3 for df >= 1
        data_demeanq = pd.DataFrame({
            'unit': [1] * 5,
            'tindex': [1, 2, 3, 4, 5],
            'y': [10.0, 12.0, 20.0, 25.0, 30.0],
            'quarter': [1, 2, 1, 2, 1],  # 2 unique quarters, k = 2
            'post': [0, 0, 1, 1, 1]       # 2 pre-treatment obs, n = k, df = 0
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # demeanq should reject n = k = 2
            _, ydot_demeanq = demeanq_unit(
                data_demeanq, y='y', quarter='quarter', post='post'
            )
            
            demeanq_rejected = np.all(np.isnan(ydot_demeanq))
        
        # detrendq with same boundary condition
        data_detrendq = pd.DataFrame({
            'unit': [1] * 6,
            'tindex': [1, 2, 3, 4, 5, 6],
            'y': [10.0, 12.0, 14.0, 20.0, 25.0, 30.0],
            'quarter': [1, 2, 1, 2, 1, 2],  # 2 quarters, k = 3
            'post': [0, 0, 0, 1, 1, 1]       # 3 pre-treatment obs, n = k, df = 0
        })
        
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            
            # detrendq should also reject n = k = 3
            _, ydot_detrendq = detrendq_unit(
                data_detrendq, y='y', tindex='tindex', 
                quarter='quarter', post='post'
            )
            
            detrendq_rejected = np.all(np.isnan(ydot_detrendq))
        
        # Both should consistently reject df = 0
        assert demeanq_rejected == detrendq_rejected, \
            "demeanq_unit and detrendq_unit should be consistent in df requirements"


class TestDESIGN094_BoundaryConditions:
    """Additional boundary condition tests for DESIGN-094."""

    def test_detrendq_with_one_quarter_boundary(self):
        """
        Test detrendq with single quarter (minimum seasonal variation).
        
        With 1 quarter: k = 1 (intercept) + 1 (tindex) + 0 (no dummies) = 2
        Need n >= 3 for df >= 1
        """
        # n = 2 (exactly k), should be rejected
        data_k = pd.DataFrame({
            'unit': [1] * 4,
            'tindex': [1, 2, 3, 4],
            'y': [10.0, 12.0, 20.0, 25.0],
            'quarter': [1, 1, 1, 1],  # Single quarter
            'post': [0, 0, 1, 1]       # 2 pre-treatment obs, k = 2
        })
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, ydot = detrendq_unit(
                data_k, y='y', tindex='tindex', quarter='quarter', post='post'
            )
            assert np.all(np.isnan(ydot)), \
                "detrendq should reject n = k (single quarter case)"
        
        # n = 3 (k + 1), should succeed
        data_k_plus_1 = pd.DataFrame({
            'unit': [1] * 5,
            'tindex': [1, 2, 3, 4, 5],
            'y': [10.0, 12.0, 14.0, 25.0, 30.0],
            'quarter': [1, 1, 1, 1, 1],  # Single quarter
            'post': [0, 0, 0, 1, 1]       # 3 pre-treatment obs
        })
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, ydot = detrendq_unit(
                data_k_plus_1, y='y', tindex='tindex', quarter='quarter', post='post'
            )
            assert not np.all(np.isnan(ydot)), \
                "detrendq should accept n = k + 1 (single quarter case)"

    def test_detrendq_warning_message_content(self):
        """Verify the warning message contains correct information."""
        # Create data with n = k (should be rejected)
        data = pd.DataFrame({
            'unit': [1] * 6,
            'tindex': [1, 2, 3, 4, 5, 6],
            'y': [10.0, 12.0, 14.0, 20.0, 25.0, 30.0],
            'quarter': [1, 2, 1, 2, 1, 2],  # 2 quarters, k = 3
            'post': [0, 0, 0, 1, 1, 1]       # 3 pre-treatment obs
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            detrendq_unit(
                data, y='y', tindex='tindex', quarter='quarter', post='post'
            )
            
            # Find the relevant warning
            user_warnings = [warning for warning in w 
                            if issubclass(warning.category, UserWarning)
                            and "Insufficient" in str(warning.message)]
            
            assert len(user_warnings) == 1
            warning_msg = str(user_warnings[0].message)
            
            # Check message content
            assert "found 3 valid observations" in warning_msg, \
                "Warning should mention the actual number of observations"
            assert "require at least 4" in warning_msg, \
                "Warning should mention the required minimum (k + 1)"
            assert "df >= 1" in warning_msg, \
                "Warning should explain the df requirement"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
