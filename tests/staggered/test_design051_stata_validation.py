"""
Stata Numerical Validation Test for DESIGN-051 Fix.

This test validates that the Python implementation of staggered demeaning
transformation produces identical results to Stata after the DESIGN-051 fix
(converting time variable to integer to avoid floating-point comparison issues).

Test Data Structure:
- 4 units × 6 periods
- Unit 1: cohort g=4, effect=10 starting at t=4
- Unit 2: cohort g=5, effect=15 starting at t=5
- Unit 3: cohort g=6, effect=20 starting at t=6
- Unit 4: never treated (gvar=0)
"""

import numpy as np
import pandas as pd
import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from lwdid.staggered.transformations import transform_staggered_demean


class TestDesign051StataValidation:
    """Numerical validation tests comparing Python vs Stata results."""
    
    @pytest.fixture
    def stata_data(self):
        """Load Stata validation data or generate equivalent test data."""
        csv_path = os.path.join(os.path.dirname(__file__), 'stata_demean_validation.csv')
        
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        else:
            # Generate equivalent test data if CSV not available
            return self._generate_test_data()
    
    def _generate_test_data(self):
        """Generate test data matching Stata structure."""
        # Same structure as Stata: 4 units × 6 periods
        data = pd.DataFrame({
            'id': [1]*6 + [2]*6 + [3]*6 + [4]*6,
            'year': [1,2,3,4,5,6]*4,
            'gvar': [4]*6 + [5]*6 + [6]*6 + [0]*6,
            'y': [
                # Unit 1: y = 10 + 2*1 + 0.5*t = 12 + 0.5*t, effect=10 at t>=4
                12.5, 13, 13.5, 24, 24.5, 25,
                # Unit 2: y = 10 + 2*2 + 0.5*t = 14 + 0.5*t, effect=15 at t>=5
                14.5, 15, 15.5, 16, 31.5, 32,
                # Unit 3: y = 10 + 2*3 + 0.5*t = 16 + 0.5*t, effect=20 at t>=6
                16.5, 17, 17.5, 18, 18.5, 39,
                # Unit 4: y = 10 + 2*4 + 0.5*t = 18 + 0.5*t, never treated
                18.5, 19, 19.5, 20, 20.5, 21,
            ]
        })
        return data
    
    @pytest.fixture
    def stata_expected_values(self):
        """Expected values from Stata computation."""
        return {
            # ydot_g4_r4: y - pre_mean_g4 at year=4
            # pre_mean_g4 = mean(y for t<4) = mean(y for t=1,2,3)
            'ydot_g4_r4': {
                1: 11.0,    # 24 - 13 = 11
                2: 1.0,     # 16 - 15 = 1
                3: 1.0,     # 18 - 17 = 1
                4: 1.0,     # 20 - 19 = 1
            },
            'ydot_g4_r5': {
                1: 11.5,    # 24.5 - 13 = 11.5
                2: 16.5,    # 31.5 - 15 = 16.5
                3: 1.5,     # 18.5 - 17 = 1.5
                4: 1.5,     # 20.5 - 19 = 1.5
            },
            'ydot_g4_r6': {
                1: 12.0,    # 25 - 13 = 12
                2: 17.0,    # 32 - 15 = 17
                3: 22.0,    # 39 - 17 = 22
                4: 2.0,     # 21 - 19 = 2
            },
            'ydot_g5_r5': {
                # pre_mean_g5 = mean(y for t<5) = mean(y for t=1,2,3,4)
                1: 8.75,    # 24.5 - 15.75 = 8.75
                2: 16.25,   # 31.5 - 15.25 = 16.25
                3: 1.25,    # 18.5 - 17.25 = 1.25
                4: 1.25,    # 20.5 - 19.25 = 1.25
            },
            'ydot_g5_r6': {
                1: 9.25,    # 25 - 15.75 = 9.25
                2: 16.75,   # 32 - 15.25 = 16.75
                3: 21.75,   # 39 - 17.25 = 21.75
                4: 1.75,    # 21 - 19.25 = 1.75
            },
            'ydot_g6_r6': {
                # pre_mean_g6 = mean(y for t<6) = mean(y for t=1,2,3,4,5)
                1: 7.5,     # 25 - 17.5 = 7.5
                2: 13.5,    # 32 - 18.5 = 13.5
                3: 21.5,    # 39 - 17.5 = 21.5
                4: 1.5,     # 21 - 19.5 = 1.5
            },
        }
    
    def test_demean_matches_stata_cohort_g4(self, stata_expected_values):
        """Test Python demean transformation matches Stata for cohort g=4."""
        data = self._generate_test_data()
        result = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        # Verify ydot_g4_r4
        for unit_id, expected_val in stata_expected_values['ydot_g4_r4'].items():
            actual = result.loc[(result['id']==unit_id) & (result['year']==4), 'ydot_g4_r4'].iloc[0]
            assert np.isclose(actual, expected_val, atol=1e-10), \
                f"Unit {unit_id} ydot_g4_r4: expected {expected_val}, got {actual}"
        
        # Verify ydot_g4_r5
        for unit_id, expected_val in stata_expected_values['ydot_g4_r5'].items():
            actual = result.loc[(result['id']==unit_id) & (result['year']==5), 'ydot_g4_r5'].iloc[0]
            assert np.isclose(actual, expected_val, atol=1e-10), \
                f"Unit {unit_id} ydot_g4_r5: expected {expected_val}, got {actual}"
        
        # Verify ydot_g4_r6
        for unit_id, expected_val in stata_expected_values['ydot_g4_r6'].items():
            actual = result.loc[(result['id']==unit_id) & (result['year']==6), 'ydot_g4_r6'].iloc[0]
            assert np.isclose(actual, expected_val, atol=1e-10), \
                f"Unit {unit_id} ydot_g4_r6: expected {expected_val}, got {actual}"
    
    def test_demean_matches_stata_cohort_g5(self, stata_expected_values):
        """Test Python demean transformation matches Stata for cohort g=5."""
        data = self._generate_test_data()
        result = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        # Verify ydot_g5_r5
        for unit_id, expected_val in stata_expected_values['ydot_g5_r5'].items():
            actual = result.loc[(result['id']==unit_id) & (result['year']==5), 'ydot_g5_r5'].iloc[0]
            assert np.isclose(actual, expected_val, atol=1e-10), \
                f"Unit {unit_id} ydot_g5_r5: expected {expected_val}, got {actual}"
        
        # Verify ydot_g5_r6
        for unit_id, expected_val in stata_expected_values['ydot_g5_r6'].items():
            actual = result.loc[(result['id']==unit_id) & (result['year']==6), 'ydot_g5_r6'].iloc[0]
            assert np.isclose(actual, expected_val, atol=1e-10), \
                f"Unit {unit_id} ydot_g5_r6: expected {expected_val}, got {actual}"
    
    def test_demean_matches_stata_cohort_g6(self, stata_expected_values):
        """Test Python demean transformation matches Stata for cohort g=6."""
        data = self._generate_test_data()
        result = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        # Verify ydot_g6_r6
        for unit_id, expected_val in stata_expected_values['ydot_g6_r6'].items():
            actual = result.loc[(result['id']==unit_id) & (result['year']==6), 'ydot_g6_r6'].iloc[0]
            assert np.isclose(actual, expected_val, atol=1e-10), \
                f"Unit {unit_id} ydot_g6_r6: expected {expected_val}, got {actual}"
    
    def test_demean_with_float_time_matches_stata(self, stata_expected_values):
        """Test that float time values produce same results as Stata.
        
        This is the key test for DESIGN-051 fix.
        """
        data = self._generate_test_data()
        # Convert year to float (simulating CSV import)
        data['year'] = data['year'].astype(float)
        
        result = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        # Verify all cohort g=4 results match Stata
        for col, expected_dict in [
            ('ydot_g4_r4', stata_expected_values['ydot_g4_r4']),
            ('ydot_g4_r5', stata_expected_values['ydot_g4_r5']),
            ('ydot_g4_r6', stata_expected_values['ydot_g4_r6']),
        ]:
            year = int(col[-1])  # Extract period from column name
            for unit_id, expected_val in expected_dict.items():
                actual = result.loc[(result['id']==unit_id) & (result['year']==year), col].iloc[0]
                assert np.isclose(actual, expected_val, atol=1e-10), \
                    f"Float time: Unit {unit_id} {col}: expected {expected_val}, got {actual}"
    
    def test_demean_with_precision_issues_matches_stata(self, stata_expected_values):
        """Test that values with floating-point precision issues match Stata.
        
        Critical test for DESIGN-051: simulates the exact failure scenario.
        """
        data = self._generate_test_data()
        
        # Introduce floating-point precision issues
        data['year'] = data['year'].astype(float)
        # Add tiny epsilon to some values (simulating accumulated floating-point errors)
        precision_noise = 1e-14  # Near machine epsilon for float64
        data.loc[data['year'] == 4.0, 'year'] = 4.0 + precision_noise
        data.loc[data['year'] == 5.0, 'year'] = 5.0 - precision_noise
        
        result = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        # After integer conversion (rounding), these should match Stata exactly
        for unit_id, expected_val in stata_expected_values['ydot_g4_r4'].items():
            actual = result.loc[(result['id']==unit_id) & (result['year']==4), 'ydot_g4_r4'].iloc[0]
            assert np.isclose(actual, expected_val, atol=1e-10), \
                f"Precision issue: Unit {unit_id} ydot_g4_r4: expected {expected_val}, got {actual}"
    
    def test_full_stata_comparison_from_csv(self, stata_data):
        """Full comparison against Stata CSV output if available."""
        # Check if we have the Stata CSV
        csv_path = os.path.join(os.path.dirname(__file__), 'stata_demean_validation.csv')
        if not os.path.exists(csv_path):
            pytest.skip("Stata validation CSV not available")
        
        # Generate Python results
        data = stata_data[['id', 'year', 'gvar', 'y']].copy()
        result = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        # Compare each ydot column
        ydot_cols = ['ydot_g4_r4', 'ydot_g4_r5', 'ydot_g4_r6', 
                     'ydot_g5_r5', 'ydot_g5_r6', 'ydot_g6_r6']
        
        for col in ydot_cols:
            if col not in stata_data.columns:
                continue
                
            for idx in result.index:
                stata_val = stata_data.loc[idx, col] if col in stata_data.columns else np.nan
                python_val = result.loc[idx, col]
                
                # Both NaN is OK
                if pd.isna(stata_val) and pd.isna(python_val):
                    continue
                # One NaN, one not is a failure
                if pd.isna(stata_val) != pd.isna(python_val):
                    assert False, f"NaN mismatch at idx={idx}, col={col}: Stata={stata_val}, Python={python_val}"
                # Both have values - compare
                assert np.isclose(python_val, stata_val, atol=1e-10), \
                    f"Value mismatch at idx={idx}, col={col}: Stata={stata_val}, Python={python_val}"


class TestDesign051TreatmentEffectRecovery:
    """Test that true treatment effects are correctly recovered after transformation."""
    
    def test_treatment_effect_recovery_demean(self):
        """Verify that demeaning correctly reveals treatment effects.
        
        In DiD, the treatment effect is identified by:
        ATT = E[ydot | treated] - E[ydot | control] at post-treatment period
        """
        # Data with known treatment effects
        data = pd.DataFrame({
            'id': [1]*6 + [2]*6,
            'year': [1,2,3,4,5,6]*2,
            'gvar': [4]*6 + [0]*6,
            'y': [
                # Unit 1 (treated at t=4): baseline trend + effect=10 at t>=4
                10, 12, 14, 26, 28, 30,  # y = 8 + 2*t, effect = 10
                # Unit 2 (control): same baseline trend, no effect
                10, 12, 14, 16, 18, 20,  # y = 8 + 2*t
            ]
        })
        
        result = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        # At t=4: ydot = y - pre_mean where pre_mean = mean(y for t=1,2,3)
        # Unit 1: pre_mean = (10+12+14)/3 = 12, ydot = 26 - 12 = 14
        # Unit 2: pre_mean = (10+12+14)/3 = 12, ydot = 16 - 12 = 4
        # ATT = 14 - 4 = 10 ✓
        
        ydot_treated = result.loc[(result['id']==1) & (result['year']==4), 'ydot_g4_r4'].iloc[0]
        ydot_control = result.loc[(result['id']==2) & (result['year']==4), 'ydot_g4_r4'].iloc[0]
        
        implied_att = ydot_treated - ydot_control
        assert np.isclose(implied_att, 10, atol=1e-10), \
            f"Treatment effect recovery failed: expected 10, got {implied_att}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
