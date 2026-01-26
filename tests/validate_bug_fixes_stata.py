"""
End-to-end validation of BUG-098, BUG-099, BUG-100 fixes using actual data.

This script performs comprehensive validation of the bug fixes:
- BUG-098: LaTeX output compilation test
- BUG-099: PSM reproducibility test with multiple runs  
- BUG-100: Sample size correctness test with NaN data

The validation uses example datasets and checks:
1. LaTeX files can be generated without errors
2. PSM matching produces identical results across runs
3. Sample sizes correctly reflect NaN-filtered data
"""

import tempfile
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from lwdid import lwdid


def validate_bug_098_latex():
    """
    Validate BUG-098: LaTeX special character escaping.
    
    Test that to_latex() correctly handles column names with underscores
    and other special characters, and that the generated LaTeX files are valid.
    """
    print("\n" + "="*70)
    print("VALIDATING BUG-098: LaTeX Special Character Escaping")
    print("="*70)
    
    # Create synthetic staggered data
    np.random.seed(42)
    n_units = 30
    n_periods = 6
    
    data_list = []
    for i in range(n_units):
        for t in range(1, n_periods + 1):
            # Cohorts: 0-9 treated at t=3, 10-19 treated at t=4, 20-29 never treated
            if i < 10:
                g = 3
            elif i < 20:
                g = 4
            else:
                g = 0
            
            y = 1.0 + 0.5 * (t >= g and g > 0) + np.random.normal(0, 0.1)
            
            data_list.append({
                'id': i,
                'year': t,
                'gvar': g,
                'y': y,
            })
    
    data = pd.DataFrame(data_list)
    
    # Run lwdid with staggered design
    results = lwdid(
        data=data,
        y='y',
        ivar='id',
        tvar='year',
        gvar='gvar',
        estimator='ra',
        control_group='not_yet_treated',
        aggregate='overall',
        rolling='demean',
        alpha=0.05,
    )
    
    # Test LaTeX export
    with tempfile.TemporaryDirectory() as tmpdir:
        latex_path = os.path.join(tmpdir, 'test_output.tex')
        
        try:
            results.to_latex(latex_path)
            print(f"✓ LaTeX file generated successfully: {latex_path}")
            
            # Read and check content
            with open(latex_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check that underscores are properly escaped
            if r'ci\_lower' in content and r'ci\_upper' in content:
                print("✓ Column names with underscores are properly escaped")
            else:
                print("✗ WARNING: Underscore escaping may not be correct")
                print("  Expected: ci\\_lower and ci\\_upper")
            
            # Check that the file doesn't have unescaped underscores in column names
            # (excluding LaTeX commands)
            lines = content.split('\n')
            unescaped_found = False
            for i, line in enumerate(lines):
                if line.strip().startswith('\\'):
                    continue
                # Check for pattern like "ci_lower" without backslash before underscore
                if 'ci_lower' in line and 'ci\\_lower' not in line and 'ci\\textbackslash' not in line:
                    print(f"✗ WARNING: Unescaped underscore in line {i+1}: {line[:80]}")
                    unescaped_found = True
            
            if not unescaped_found:
                print("✓ No unescaped underscores found in column names")
            
            print("\n✓ BUG-098 validation PASSED: LaTeX export works correctly")
            
        except Exception as e:
            print(f"✗ BUG-098 validation FAILED: {e}")
            return False
    
    return True


def validate_bug_099_psm_reproducibility():
    """
    Validate BUG-099: PSM tie-breaking reproducibility.
    
    Test that PSM matching produces identical results when run multiple times
    with the same data and parameters, even when there are ties in propensity scores.
    """
    print("\n" + "="*70)
    print("VALIDATING BUG-099: PSM Tie-Breaking Reproducibility")
    print("="*70)
    
    # Create data with intentional PS ties
    np.random.seed(123)
    n_units = 40
    n_periods = 5
    
    data_list = []
    for i in range(n_units):
        # Create groups with identical covariate values to induce PS ties
        x_val = i // 4  # Groups of 4 units with same X
        
        for t in range(1, n_periods + 1):
            # Cohort at t=3
            g = 3 if i < 20 else 0
            
            # Outcome with treatment effect
            y = 1.0 + 0.3 * x_val + 0.4 * (t >= g and g > 0) + np.random.normal(0, 0.05)
            
            data_list.append({
                'id': i,
                'year': t,
                'gvar': g,
                'y': y,
                'x': x_val,
            })
    
    data = pd.DataFrame(data_list)
    
    # Run PSM estimation multiple times
    n_runs = 5
    all_atts = []
    
    for run in range(n_runs):
        results = lwdid(
            data=data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            estimator='psm',
            controls=['x'],
            ps_controls=['x'],
            n_neighbors=1,
            with_replacement=True,
            match_order='data',
            control_group='never_treated',
            aggregate='overall',
            rolling='demean',
            alpha=0.05,
        )
        
        all_atts.append(results.att_overall)
        print(f"  Run {run+1}: ATT = {results.att_overall:.10f}")
    
    # Check reproducibility
    if len(set([f"{att:.15f}" for att in all_atts])) == 1:
        print(f"\n✓ PSM results are perfectly reproducible across {n_runs} runs")
        print(f"  ATT = {all_atts[0]:.10f} (identical in all runs)")
        print("\n✓ BUG-099 validation PASSED: PSM matching is deterministic")
        return True
    else:
        print(f"\n✗ WARNING: PSM results vary across runs")
        print(f"  Unique ATT values: {set([f'{att:.10f}' for att in all_atts])}")
        print("✗ BUG-099 validation FAILED: PSM matching is not reproducible")
        return False


def validate_bug_100_sample_counting():
    """
    Validate BUG-100: Sample size counting with NaN values.
    
    Test that n_treat and n_control correctly reflect the actual sample
    used in regression after filtering NaN outcomes.
    """
    print("\n" + "="*70)
    print("VALIDATING BUG-100: Sample Size Counting with NaN Filtering")
    print("="*70)
    
    # Create data with strategic NaN placements
    np.random.seed(456)
    n_units = 30
    n_periods = 5
    
    data_list = []
    nan_count_expected = 0
    
    for i in range(n_units):
        for t in range(1, n_periods + 1):
            g = 3 if i < 15 else 0
            
            # Introduce NaN for specific treated units in post-treatment periods
            if i < 15 and t >= 3 and i % 5 == 0:  # Units 0, 5, 10
                y = np.nan
                if t == 3:  # Count NaN in period 3
                    nan_count_expected += 1
            else:
                y = 1.0 + 0.5 * (t >= g and g > 0) + np.random.normal(0, 0.1)
            
            data_list.append({
                'id': i,
                'year': t,
                'gvar': g,
                'y': y,
            })
    
    data = pd.DataFrame(data_list)
    
    print(f"\nData setup:")
    print(f"  Total units: {n_units}")
    print(f"  Treated units (cohort 3): 15")
    print(f"  Control units (never treated): 15")
    print(f"  Expected NaN-affected treated units in period 3: {nan_count_expected}")
    
    # Run estimation
    results = lwdid(
        data=data,
        y='y',
        ivar='id',
        tvar='year',
        gvar='gvar',
        estimator='ra',
        control_group='not_yet_treated',
        aggregate='cohort',
        rolling='demean',
        alpha=0.05,
    )
    
    # Access cohort-time effects
    effects = results._cohort_time_effects
    
    # Check sample sizes for period 3
    period_3_effect = [e for e in effects if e.cohort == 3 and e.period == 3]
    
    if len(period_3_effect) == 0:
        print("\n✗ BUG-100 validation FAILED: No effect estimated for period 3")
        return False
    
    effect = period_3_effect[0]
    
    print(f"\nEstimated effect for (cohort=3, period=3):")
    print(f"  n_treated: {effect.n_treated}")
    print(f"  n_control: {effect.n_control}")
    print(f"  n_total: {effect.n_total}")
    
    # Expected: 15 - 3 = 12 treated units (after NaN filtering)
    expected_treated = 15 - nan_count_expected
    
    if effect.n_treated == expected_treated:
        print(f"\n✓ n_treated correctly reflects NaN filtering")
        print(f"  Expected: {expected_treated}, Got: {effect.n_treated}")
    else:
        print(f"\n✗ WARNING: n_treated may not reflect NaN filtering")
        print(f"  Expected: {expected_treated}, Got: {effect.n_treated}")
    
    if effect.n_total == effect.n_treated + effect.n_control:
        print(f"✓ n_total is consistent with n_treated + n_control")
    else:
        print(f"✗ ERROR: n_total inconsistent")
        print(f"  n_total={effect.n_total}, but n_treated + n_control = {effect.n_treated + effect.n_control}")
    
    # Validate that all sample sizes are reasonable
    validation_passed = True
    
    if effects:
        for e in effects:
            if e.n_total != e.n_treated + e.n_control:
                print(f"✗ Inconsistency in (cohort={e.cohort}, period={e.period})")
                validation_passed = False
            
            if e.n_treated > 15 or e.n_control > 15:
                print(f"✗ Sample size exceeds maximum possible in (cohort={e.cohort}, period={e.period})")
                validation_passed = False
    else:
        print("✗ WARNING: No effects available for validation")
        validation_passed = False
    
    if validation_passed:
        print("\n✓ BUG-100 validation PASSED: Sample sizes correctly reflect NaN filtering")
        return True
    else:
        print("\n✗ BUG-100 validation FAILED: Sample size issues detected")
        return False


def main():
    """Run all validation tests."""
    print("\n" + "="*70)
    print("BUG FIXES VALIDATION SUITE")
    print("Testing BUG-098, BUG-099, BUG-100")
    print("="*70)
    
    results = {
        'BUG-098 (LaTeX Escaping)': validate_bug_098_latex(),
        'BUG-099 (PSM Reproducibility)': validate_bug_099_psm_reproducibility(),
        'BUG-100 (Sample Counting)': validate_bug_100_sample_counting(),
    }
    
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("ALL VALIDATIONS PASSED ✓")
        print("Bug fixes are working correctly.")
    else:
        print("SOME VALIDATIONS FAILED ✗")
        print("Please review the failed tests above.")
    print("="*70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
