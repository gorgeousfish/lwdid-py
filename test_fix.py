"""Quick test to verify the fix works."""
from lwdid.trend_diagnostics import test_parallel_trends, _STAGGERED_AVAILABLE
print(f'Staggered module available: {_STAGGERED_AVAILABLE}')

# Quick test with small data
import numpy as np
import pandas as pd

np.random.seed(42)
n_units, n_periods = 50, 8
data = []
for i in range(n_units):
    is_treated = i < n_units // 2
    first_treat = 6 if is_treated else np.inf
    unit_fe = np.random.normal(0, 1)
    for t in range(1, n_periods + 1):
        y = 10 + unit_fe + 0.2 * t  # Same trend for all
        if is_treated and t >= 6:
            y += 2.0
        y += np.random.normal(0, 0.5)
        data.append({'unit': i, 'time': t, 'Y': y, 'first_treat': first_treat})

df = pd.DataFrame(data)
print(f'Data shape: {df.shape}')

# Run test
result = test_parallel_trends(df, y='Y', ivar='unit', tvar='time', gvar='first_treat', verbose=False)
print(f'Reject null: {result.reject_null}')
print(f'P-value: {result.pvalue:.4f}')
print(f'N pre-trend estimates: {len(result.pre_trend_estimates)}')
print(f'Method: {result.method}')

# Print pre-trend estimates
if result.pre_trend_estimates:
    print('\nPre-trend estimates:')
    for est in result.pre_trend_estimates:
        print(f'  e={est.event_time}: ATT={est.att:.4f}, SE={est.se:.4f}, p={est.pvalue:.4f}')
