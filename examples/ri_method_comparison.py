"""
Demonstration: Bootstrap vs Permutation RI Methods

This script demonstrates the difference between the two randomization
inference methods available in lwdid:

1. Bootstrap (with-replacement): Stata-compatible, default
2. Permutation (without-replacement): Classical Fisher RI

Author: lwdid development team
Date: 2025-11-12
"""

import numpy as np
import pandas as pd
from lwdid import lwdid


def create_example_data(N=30, N1=12, T0=4, T1=4, seed=42):
    """
    Create example panel data for demonstration
    
    Parameters
    ----------
    N : int
        Total number of units
    N1 : int
        Number of treated units
    T0 : int
        Number of pre-treatment periods
    T1 : int
        Number of post-treatment periods
    seed : int
        Random seed
    
    Returns
    -------
    pd.DataFrame
        Panel data with columns: unit, time, d, post, y
    """
    np.random.seed(seed)
    
    data = []
    for i in range(N):
        # Treatment assignment
        d_i = 1 if i < N1 else 0
        
        # Unit-specific effect
        alpha_i = np.random.normal(0, 2)
        
        for t in range(T0 + T1):
            # Post-treatment indicator
            post_t = 1 if t >= T0 else 0
            
            # Outcome: y = alpha_i + beta*t + tau*d*post + epsilon
            y_it = (
                alpha_i +                      # Unit fixed effect
                0.5 * t +                      # Time trend
                3.0 * d_i * post_t +          # Treatment effect (ATT=3)
                np.random.normal(0, 1)        # Idiosyncratic error
            )
            
            data.append({
                'unit': i,
                'time': t,
                'd': d_i,
                'post': post_t,
                'y': y_it
            })
    
    return pd.DataFrame(data)


def main():
    """Main demonstration"""
    
    print("=" * 80)
    print("RI Method Comparison: Bootstrap vs Permutation")
    print("=" * 80)
    print()
    
    # Create example data
    print("Creating example data...")
    print("  N = 30 units (12 treated, 18 control)")
    print("  T = 8 periods (4 pre, 4 post)")
    print("  True ATT = 3.0")
    print()
    
    data = create_example_data(N=30, N1=12, T0=4, T1=4, seed=123)
    
    # Method 1: Bootstrap (default, Stata-compatible)
    print("-" * 80)
    print("Method 1: Bootstrap (with-replacement)")
    print("-" * 80)
    print("This is the default method, compatible with Stata lwdid.ado")
    print("Treatment group size N1 may vary across replications")
    print()
    
    result_bootstrap = lwdid(
        data, 'y', 'd', 'unit', 'time', 'post', 'demean',
        ri=True, rireps=1000, seed=456, ri_method='bootstrap'
    )
    
    print(f"ATT estimate:     {result_bootstrap.att:>8.4f}")
    print(f"Standard error:   {result_bootstrap.se_att:>8.4f}")
    print(f"t-statistic:      {result_bootstrap.t_stat:>8.4f}")
    print(f"OLS p-value:      {result_bootstrap.pvalue:>8.4f}")
    print()
    print(f"RI p-value:       {result_bootstrap.ri_pvalue:>8.4f}")
    print(f"RI method:        {result_bootstrap.ri_method}")
    print(f"RI replications:  {result_bootstrap.ri_valid}/{result_bootstrap.rireps}")
    print(f"RI failures:      {result_bootstrap.ri_failed}")
    print()
    
    # Method 2: Permutation (Fisher RI)
    print("-" * 80)
    print("Method 2: Permutation (without-replacement)")
    print("-" * 80)
    print("This is the classical Fisher randomization inference method")
    print("Treatment group size N1 is fixed across all permutations")
    print("Matches the ritest command behavior mentioned in the paper")
    print()
    
    result_permutation = lwdid(
        data, 'y', 'd', 'unit', 'time', 'post', 'demean',
        ri=True, rireps=1000, seed=456, ri_method='permutation'
    )
    
    print(f"ATT estimate:     {result_permutation.att:>8.4f}")
    print(f"Standard error:   {result_permutation.se_att:>8.4f}")
    print(f"t-statistic:      {result_permutation.t_stat:>8.4f}")
    print(f"OLS p-value:      {result_permutation.pvalue:>8.4f}")
    print()
    print(f"RI p-value:       {result_permutation.ri_pvalue:>8.4f}")
    print(f"RI method:        {result_permutation.ri_method}")
    print(f"RI replications:  {result_permutation.ri_valid}/{result_permutation.rireps}")
    print(f"RI failures:      {result_permutation.ri_failed}")
    print()
    
    # Comparison
    print("=" * 80)
    print("Comparison")
    print("=" * 80)
    print()
    print(f"ATT estimates are identical:  {result_bootstrap.att:.4f} vs {result_permutation.att:.4f}")
    print(f"OLS p-values are identical:   {result_bootstrap.pvalue:.4f} vs {result_permutation.pvalue:.4f}")
    print()
    print(f"RI p-values differ:")
    print(f"  Bootstrap:    {result_bootstrap.ri_pvalue:.4f}")
    print(f"  Permutation:  {result_permutation.ri_pvalue:.4f}")
    print(f"  Difference:   {abs(result_bootstrap.ri_pvalue - result_permutation.ri_pvalue):.4f}")
    print()
    print(f"Failure rates:")
    print(f"  Bootstrap:    {result_bootstrap.ri_failed}/{result_bootstrap.rireps} "
          f"({100*result_bootstrap.ri_failed/result_bootstrap.rireps:.1f}%)")
    print(f"  Permutation:  {result_permutation.ri_failed}/{result_permutation.rireps} "
          f"({100*result_permutation.ri_failed/result_permutation.rireps:.1f}%)")
    print()
    
    # Recommendations
    print("=" * 80)
    print("Recommendations")
    print("=" * 80)
    print()
    print("Use BOOTSTRAP when:")
    print("  - You need exact compatibility with Stata lwdid.ado results")
    print("  - You are replicating existing analyses based on bootstrap")
    print("  - Sample size is moderate and treatment group is balanced")
    print()
    print("Use PERMUTATION when:")
    print("  - You need classical Fisher RI theoretical guarantees")
    print("  - You want to match ritest command results")
    print("  - Sample size is small (N < 30)")
    print("  - Treatment group is very small (N1 < 5) or very large (N1 > N-5)")
    print("  - You want to avoid potential failures due to extreme N1 values")
    print()
    print("For new analyses, PERMUTATION is generally recommended.")
    print()
    
    # Show full summary for permutation
    print("=" * 80)
    print("Full Summary (Permutation Method)")
    print("=" * 80)
    print()
    print(result_permutation.summary())


if __name__ == '__main__':
    main()

