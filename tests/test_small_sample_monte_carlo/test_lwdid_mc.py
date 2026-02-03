# -*- coding: utf-8 -*-
"""
Test script to run Monte Carlo with actual lwdid package.
"""

import sys
from pathlib import Path

# Add fixtures paths
fixtures_path = Path(__file__).parent / 'fixtures'
parent_fixtures = Path(__file__).parent.parent / 'test_common_timing' / 'fixtures'
sys.path.insert(0, str(fixtures_path))
sys.path.insert(0, str(parent_fixtures))

from monte_carlo_runner import run_small_sample_monte_carlo, run_all_scenarios_monte_carlo, generate_comparison_table

def main():
    print("Running Monte Carlo with actual lwdid package...")
    print("=" * 60)
    
    # Run all scenarios with actual lwdid package
    all_results = run_all_scenarios_monte_carlo(
        n_reps=100,
        estimators=['demeaning', 'detrending'],
        seed=42,
        verbose=True,
        use_lwdid=True,
    )
    
    # Generate comparison table
    print(f'\n{"=" * 60}')
    print(f'Comparison Table (lwdid package):')
    print(f'{"=" * 60}')
    df = generate_comparison_table(all_results)
    print(df.to_string(index=False))
    
    # Paper Table 2 reference values
    print(f'\n{"=" * 60}')
    print(f'Paper Table 2 Reference Values:')
    print(f'{"=" * 60}')
    print(f'Scenario 1 (p=0.32):')
    print(f'  Detrending: Bias=0.009, SD=1.73, RMSE=1.734, Coverage=96%')
    print(f'Scenario 2 (p=0.24):')
    print(f'  Detrending: Bias=-0.042, SD=1.89, RMSE=1.892, Coverage=95%')
    print(f'Scenario 3 (p=0.17):')
    print(f'  Detrending: Bias=0.165, SD=2.37, RMSE=2.380, Coverage=95%')

if __name__ == '__main__':
    main()
