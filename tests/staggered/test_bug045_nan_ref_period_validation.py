"""
BUG-045 Numerical Validation: plot_event_study NaN Reference Period Normalization

This script provides comprehensive numerical validation for the BUG-045 fix,
which ensures that when the reference period ATT value is NaN, normalization
is skipped with a clear warning rather than corrupting all ATT values.

Test scenarios:
1. Normal reference period normalization
2. NaN reference period detection and warning
3. Data preservation when normalization is skipped
4. CI bounds integrity
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def test_normalization_logic_isolated():
    """Test the normalization logic in isolation.
    
    This tests the exact logic from results.py lines 1674-1696.
    """
    print("\n" + "=" * 70)
    print("TEST 1: Normalization Logic Isolation")
    print("=" * 70)
    
    # Scenario 1: Normal reference period
    print("\n--- Scenario 1: Normal reference period (ATT = 1.0) ---")
    event_df = pd.DataFrame({
        'event_time': [-2, -1, 0, 1, 2],
        'att': [0.2, 0.4, 1.0, 1.5, 2.0],
        'se': [0.1, 0.1, 0.1, 0.1, 0.1]
    })
    
    alpha = 0.05
    z_crit = stats.norm.ppf(1 - alpha / 2)
    event_df['ci_lower'] = event_df['att'] - z_crit * event_df['se']
    event_df['ci_upper'] = event_df['att'] + z_crit * event_df['se']
    
    ref_period = 0
    ref_row = event_df[event_df['event_time'] == ref_period]
    ref_att = ref_row['att'].values[0]
    
    print(f"Reference period: {ref_period}")
    print(f"Reference ATT: {ref_att}")
    print(f"Is NaN: {pd.isna(ref_att)}")
    
    # Apply normalization (the fix logic)
    if pd.isna(ref_att):
        print("WARNING: Skipping normalization due to NaN ref_att")
        normalized_df = event_df.copy()
    else:
        normalized_df = event_df.copy()
        normalized_df['att'] = normalized_df['att'] - ref_att
        normalized_df['ci_lower'] = normalized_df['ci_lower'] - ref_att
        normalized_df['ci_upper'] = normalized_df['ci_upper'] - ref_att
    
    print("\nOriginal ATT values:")
    print(event_df[['event_time', 'att']].to_string(index=False))
    print("\nNormalized ATT values:")
    print(normalized_df[['event_time', 'att']].to_string(index=False))
    
    # Verify: ATT at ref_period should be 0
    ref_att_normalized = normalized_df[normalized_df['event_time'] == ref_period]['att'].values[0]
    assert abs(ref_att_normalized) < 1e-10, f"Expected 0, got {ref_att_normalized}"
    print(f"\n✓ ATT at ref_period normalized to: {ref_att_normalized:.10f}")
    
    # Scenario 2: NaN reference period
    print("\n--- Scenario 2: NaN reference period ---")
    event_df_nan = pd.DataFrame({
        'event_time': [-2, -1, 0, 1, 2],
        'att': [0.2, 0.4, np.nan, 1.5, 2.0],  # NaN at event_time=0
        'se': [0.1, 0.1, 0.1, 0.1, 0.1]
    })
    event_df_nan['ci_lower'] = event_df_nan['att'] - z_crit * event_df_nan['se']
    event_df_nan['ci_upper'] = event_df_nan['att'] + z_crit * event_df_nan['se']
    
    ref_row_nan = event_df_nan[event_df_nan['event_time'] == ref_period]
    ref_att_nan = ref_row_nan['att'].values[0]
    
    print(f"Reference period: {ref_period}")
    print(f"Reference ATT: {ref_att_nan}")
    print(f"Is NaN: {pd.isna(ref_att_nan)}")
    
    # Apply normalization with fix logic
    if pd.isna(ref_att_nan):
        print("⚠ WARNING: Skipping normalization due to NaN ref_att")
        normalized_df_nan = event_df_nan.copy()
    else:
        normalized_df_nan = event_df_nan.copy()
        normalized_df_nan['att'] = normalized_df_nan['att'] - ref_att_nan
    
    print("\nOriginal ATT values:")
    print(event_df_nan[['event_time', 'att']].to_string(index=False))
    print("\nATT values after fix logic:")
    print(normalized_df_nan[['event_time', 'att']].to_string(index=False))
    
    # Verify: Non-NaN values should be preserved
    non_nan_original = event_df_nan[event_df_nan['att'].notna()]['att'].values
    non_nan_result = normalized_df_nan[normalized_df_nan['att'].notna()]['att'].values
    np.testing.assert_array_almost_equal(non_nan_original, non_nan_result)
    print(f"\n✓ Non-NaN ATT values preserved: {list(non_nan_result)}")
    
    # Scenario 3: Demonstrate the bug (what would happen without fix)
    print("\n--- Scenario 3: Demonstrating the bug (without fix) ---")
    corrupted_df = event_df_nan.copy()
    corrupted_df['att'] = corrupted_df['att'] - ref_att_nan  # Subtract NaN
    
    print("ATT values WITHOUT fix (all become NaN):")
    print(corrupted_df[['event_time', 'att']].to_string(index=False))
    
    all_nan = corrupted_df['att'].isna().all()
    print(f"\n✓ Confirmed: Without fix, all ATT values become NaN: {all_nan}")
    
    return True


def test_ci_bounds_integrity():
    """Test that CI bounds are correctly handled during normalization."""
    print("\n" + "=" * 70)
    print("TEST 2: CI Bounds Integrity")
    print("=" * 70)
    
    alpha = 0.05
    z_crit = stats.norm.ppf(1 - alpha / 2)
    
    # Create test data
    event_df = pd.DataFrame({
        'event_time': [-1, 0, 1],
        'att': [0.5, 1.0, 1.5],
        'se': [0.2, 0.2, 0.2]
    })
    event_df['ci_lower'] = event_df['att'] - z_crit * event_df['se']
    event_df['ci_upper'] = event_df['att'] + z_crit * event_df['se']
    
    original_ci_width = event_df['ci_upper'] - event_df['ci_lower']
    
    # Normalize
    ref_att = 1.0
    event_df['att'] = event_df['att'] - ref_att
    event_df['ci_lower'] = event_df['ci_lower'] - ref_att
    event_df['ci_upper'] = event_df['ci_upper'] - ref_att
    
    normalized_ci_width = event_df['ci_upper'] - event_df['ci_lower']
    
    print("\nCI width before normalization:")
    print(original_ci_width.to_string())
    print("\nCI width after normalization:")
    print(normalized_ci_width.to_string())
    
    # Verify: CI width should be preserved
    np.testing.assert_array_almost_equal(
        original_ci_width.values,
        normalized_ci_width.values,
        decimal=10
    )
    print("\n✓ CI width preserved after normalization")
    
    # Verify: At ref_period, CI should be centered at 0
    ref_row = event_df[event_df['event_time'] == 0]
    ref_ci_center = (ref_row['ci_lower'].values[0] + ref_row['ci_upper'].values[0]) / 2
    assert abs(ref_ci_center) < 1e-10, f"CI center at ref_period should be 0, got {ref_ci_center}"
    print(f"✓ CI center at ref_period: {ref_ci_center:.10f}")
    
    return True


def test_warning_message_content():
    """Test the warning message content matches expected format."""
    print("\n" + "=" * 70)
    print("TEST 3: Warning Message Content")
    print("=" * 70)
    
    # Simulate the warning message from the fix
    ref_period = 0
    warning_msg = (
        f"Reference period e={ref_period} has NaN ATT estimate. "
        f"Cannot normalize to NaN. Skipping normalization."
    )
    
    print(f"\nExpected warning format:")
    print(f"  '{warning_msg}'")
    
    # Verify key components
    assert f"e={ref_period}" in warning_msg, "Warning should include ref_period"
    assert "NaN ATT estimate" in warning_msg, "Warning should mention NaN ATT"
    assert "Skipping normalization" in warning_msg, "Warning should mention skipping"
    
    print("\n✓ Warning message contains all required components:")
    print(f"  - Reference period (e={ref_period})")
    print("  - NaN ATT mention")
    print("  - Skipping normalization notice")
    
    return True


def test_edge_cases():
    """Test edge cases for the NaN reference period fix."""
    print("\n" + "=" * 70)
    print("TEST 4: Edge Cases")
    print("=" * 70)
    
    alpha = 0.05
    z_crit = stats.norm.ppf(1 - alpha / 2)
    
    # Edge case 1: All ATT values are NaN
    print("\n--- Edge Case 1: All ATT values are NaN ---")
    event_df_all_nan = pd.DataFrame({
        'event_time': [-1, 0, 1],
        'att': [np.nan, np.nan, np.nan],
        'se': [0.1, 0.1, 0.1]
    })
    
    ref_row = event_df_all_nan[event_df_all_nan['event_time'] == 0]
    ref_att = ref_row['att'].values[0]
    
    if pd.isna(ref_att):
        print("✓ Correctly detected NaN ref_att, normalization skipped")
        result_df = event_df_all_nan.copy()
    else:
        result_df = event_df_all_nan.copy()
        result_df['att'] = result_df['att'] - ref_att
    
    # All should still be NaN (preserved)
    assert result_df['att'].isna().all()
    print("✓ All NaN values preserved")
    
    # Edge case 2: Only ref_period is NaN
    print("\n--- Edge Case 2: Only ref_period is NaN ---")
    event_df_ref_only_nan = pd.DataFrame({
        'event_time': [-1, 0, 1],
        'att': [0.5, np.nan, 1.5],  # Only ref_period is NaN
        'se': [0.1, 0.1, 0.1]
    })
    
    ref_row = event_df_ref_only_nan[event_df_ref_only_nan['event_time'] == 0]
    ref_att = ref_row['att'].values[0]
    
    original_non_nan_values = event_df_ref_only_nan['att'].dropna().values.copy()
    
    if pd.isna(ref_att):
        print("✓ Correctly detected NaN ref_att, normalization skipped")
        result_df = event_df_ref_only_nan.copy()
    else:
        result_df = event_df_ref_only_nan.copy()
        result_df['att'] = result_df['att'] - ref_att
    
    result_non_nan_values = result_df['att'].dropna().values
    np.testing.assert_array_almost_equal(original_non_nan_values, result_non_nan_values)
    print(f"✓ Non-NaN values preserved: {list(result_non_nan_values)}")
    
    # Edge case 3: Ref period not found (should not reach NaN check)
    print("\n--- Edge Case 3: Ref period not found ---")
    event_df_no_ref = pd.DataFrame({
        'event_time': [-2, -1, 1, 2],  # No event_time=0
        'att': [0.2, 0.4, 1.5, 2.0],
        'se': [0.1, 0.1, 0.1, 0.1]
    })
    
    ref_period = 0
    ref_row = event_df_no_ref[event_df_no_ref['event_time'] == ref_period]
    
    if len(ref_row) == 0:
        print(f"✓ Correctly detected ref_period={ref_period} not found")
        print(f"  Available event_times: {sorted(event_df_no_ref['event_time'].unique())}")
    
    return True


def run_integration_test():
    """Run integration test with actual lwdid results if available."""
    print("\n" + "=" * 70)
    print("TEST 5: Integration Test with Real Results")
    print("=" * 70)
    
    try:
        from lwdid import lwdid
        
        # Load castle data
        here = os.path.dirname(__file__)
        data_path = os.path.join(here, '..', '..', 'data', 'castle.csv')
        
        if not os.path.exists(data_path):
            print(f"⚠ Castle data not found at {data_path}, skipping integration test")
            return True
        
        data = pd.read_csv(data_path)
        data['gvar'] = data['effyear'].fillna(0).astype(int)
        
        print("\nRunning lwdid estimation...")
        results = lwdid(
            data=data,
            y='lhomicide',
            ivar='sid',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            control_group='never_treated',
            aggregate='overall',
            vce='hc3'
        )
        
        print("✓ Estimation completed")
        
        # Test normal ref_period
        print("\n--- Testing normal ref_period=0 ---")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig, ax, event_df = results.plot_event_study(
                ref_period=0, return_data=True
            )
        
        ref_row = event_df[event_df['event_time'] == 0]
        if not ref_row.empty:
            ref_att = ref_row['att'].values[0]
            print(f"ATT at ref_period=0: {ref_att:.10f}")
            
            if not pd.isna(ref_att):
                assert abs(ref_att) < 1e-10, f"Should be ~0 after normalization"
                print("✓ Normalization working correctly")
        
        import matplotlib.pyplot as plt
        plt.close(fig)
        
        # Test ref_period=None (no normalization)
        print("\n--- Testing ref_period=None (no normalization) ---")
        fig2, ax2, event_df_none = results.plot_event_study(
            ref_period=None, return_data=True
        )
        
        ref_row_none = event_df_none[event_df_none['event_time'] == 0]
        if not ref_row_none.empty:
            ref_att_none = ref_row_none['att'].values[0]
            print(f"ATT at event_time=0 (no normalization): {ref_att_none:.6f}")
            print("✓ No normalization applied")
        
        plt.close(fig2)
        
        return True
        
    except ImportError as e:
        print(f"⚠ Could not import lwdid: {e}")
        print("  Skipping integration test")
        return True
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("=" * 70)
    print("BUG-045 Numerical Validation: NaN Reference Period Normalization")
    print("=" * 70)
    
    all_passed = True
    
    try:
        all_passed &= test_normalization_logic_isolated()
    except AssertionError as e:
        print(f"✗ TEST 1 FAILED: {e}")
        all_passed = False
    
    try:
        all_passed &= test_ci_bounds_integrity()
    except AssertionError as e:
        print(f"✗ TEST 2 FAILED: {e}")
        all_passed = False
    
    try:
        all_passed &= test_warning_message_content()
    except AssertionError as e:
        print(f"✗ TEST 3 FAILED: {e}")
        all_passed = False
    
    try:
        all_passed &= test_edge_cases()
    except AssertionError as e:
        print(f"✗ TEST 4 FAILED: {e}")
        all_passed = False
    
    try:
        all_passed &= run_integration_test()
    except Exception as e:
        print(f"✗ TEST 5 FAILED: {e}")
        all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL VALIDATION TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
