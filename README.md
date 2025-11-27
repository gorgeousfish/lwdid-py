# lwdid: Difference-in-Differences Estimator for Small Cross-Sectional Samples

![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-AGPL--3.0-green.svg)

Python implementation of the Lee and Wooldridge (2025) difference-in-differences estimator for panel data with small cross-sectional sample sizes.

## Overview

This package implements the methodology described in Lee and Wooldridge (2025), providing valid inference for difference-in-differences estimation when the number of treated or control units is small.

**Reference**: Lee, S. J., and Wooldridge, J. M. (2025). Simple Approaches to Inference with Difference-in-Differences Estimators with Small Cross-Sectional Sample Sizes. *Available at [SSRN 5325686](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5325686)*.

**Authors**: Xuanyu Cai, Wenli Xu

### Key Features

The package provides inference for small cross-sectional samples by transforming panel data into cross-sectional regressions:

- Designed for settings with small numbers of treated or control units
- Exact t-based inference available under classical linear model assumptions (normality and homoskedasticity)
- Works best with large time dimensions, where the central limit theorem across time supports normality
- Serial correlation handled through unit-specific transformations
- Unit-specific linear trends and seasonal patterns
- Heteroskedasticity-robust inference (HC1/HC3) for moderate sample sizes
- Randomization inference for finite-sample validity without distributional assumptions

### Transformation Methods

Four transformation methods are available:

- **demean**: Unit-specific demeaning (Procedure 2.1)
- **detrend**: Unit-specific detrending (Procedure 3.1)
- **demeanq**: Quarterly demeaning with seasonal effects
- **detrendq**: Quarterly detrending with linear trends and seasonal effects

## Installation

```bash
pip install lwdid
```

Or install from source:

```bash
git clone https://github.com/gorgeousfish/lwdid-py.git
cd lwdid-py
pip install .
```

## Quick Start

### Basic Example

```python
import pandas as pd
from lwdid import lwdid

# Load panel data
data = pd.read_csv('smoking.csv')
# Note: 'd' is the column name for treatment indicator in this dataset

# Estimate ATT with exact inference
results = lwdid(
    data,
    y='lcigsale',      # outcome variable
    d='d',             # treatment indicator (0/1)
    ivar='state',      # unit identifier
    tvar='year',       # time variable
    post='post',       # post-treatment indicator
    rolling='detrend', # transformation: demean, detrend, demeanq, detrendq
    vce=None           # None: exact inference; 'hc3': heteroskedasticity-robust
)

# View results
print(results.summary())
print(f"ATT: {results.att:.4f} (SE: {results.se_att:.4f})")
print(f"95% CI: [{results.ci_lower:.4f}, {results.ci_upper:.4f}]")

# Export results
results.to_excel('results.xlsx')
results.to_latex('results.tex')
```

### Advanced Usage

**Randomization Inference**

```python
# Randomization inference for finite-sample validity without distributional assumptions
# Default: bootstrap resampling
# Alternative: permutation-based (Fisher randomization inference)
results = lwdid(
    data, 'lcigsale', 'd', 'state', 'year', 'post', 'detrend',
    ri=True,               # enable randomization inference
    ri_method='bootstrap', # 'bootstrap' (default) or 'permutation'
    rireps=1000,           # number of replications
    seed=42
)
print(f"RI p-value: {results.ri_pvalue:.4f}")
```

**Control Variables**

```python
# Include time-invariant control variables
# Note: Controls must be constant within each unit across all periods
# For time-varying variables, use pre-treatment mean or first value

# Create time-invariant controls from time-varying variables
data_with_controls = data.copy()
for var in ['retprice', 'beer']:
    # Use pre-treatment period mean
    pre_mean = data[data['post']==0].groupby('state')[var].mean()
    data_with_controls[f'{var}_pre'] = data_with_controls['state'].map(pre_mean)

results = lwdid(
    data_with_controls, 'lcigsale', 'd', 'state', 'year', 'post', 'detrend',
    controls=['retprice_pre', 'beer_pre'],  # time-invariant covariates
    vce='hc3'
)
```

**Quarterly Data**

```python
# Quarterly panel with seasonal effects
# Example: data with columns [unit, year, quarter, outcome, d, post]
results = lwdid(
    data, 'outcome', 'd', 'unit',
    tvar=['year', 'quarter'],  # composite time variable
    post='post',
    rolling='detrendq'         # quarterly detrending
)
```

## Capabilities

### Core Features

- **Transformation methods**: demean, detrend, demeanq, detrendq
- **Inference options**: Exact (under normality), HC1 robust, HC3 robust, cluster-robust
- **Control variables**: Time-invariant covariates with automatic centering
- **Period-specific effects**: Estimate ATT for each post-treatment period
- **Randomization inference**: Bootstrap (default) or permutation-based p-values for finite-sample validity
- **Visualization**: Time series plots comparing treated and control units
- **Export formats**: Excel (multi-sheet), CSV, LaTeX tables

### Validation

The implementation has been validated for numerical accuracy and consistency with the methodology described in Lee and Wooldridge (2025).

## Requirements

- Python ≥ 3.8, <3.13
- numpy ≥ 1.20, <3.0
- pandas ≥ 1.3, <3.0
- scipy ≥ 1.7, <2.0
- statsmodels ≥ 0.13, <1.0
- matplotlib ≥ 3.3 (visualization)
- openpyxl ≥ 3.1 (Excel export)

## API Reference

### Main Function

```python
lwdid(data, y, d, ivar, tvar, post, rolling, **options)
```

**Required Parameters**:
- `data` (DataFrame): Panel data in long format
- `y` (str): Outcome variable
- `d` (str): Unit-level treatment indicator Dᵢ (0/1)
  - **Important**: Must be time-invariant (constant within each unit across all periods)
  - Do **not** pass time-varying treatment indicator Wᵢₜ = Dᵢ × postₜ
  - If you have Wᵢₜ, construct Dᵢ first: `data['D_i'] = data.groupby('unit')['W_it'].transform('max')`
- `ivar` (str): Unit identifier
- `tvar` (str or list): Time variable (must be numeric)
  - Annual data: Single column name (str), e.g., `tvar='year'`
  - Quarterly data: List of two column names [year, quarter], e.g., `tvar=['year', 'quarter']`
  - **Important**: All time variables must contain numeric values (int or float)
- `post` (str): Post-treatment indicator (0/1)
- `rolling` (str): Transformation method
  - `'demean'`: Standard DiD with unit fixed effects
  - `'detrend'`: DiD with unit-specific linear trends
  - `'demeanq'`: Quarterly data with seasonal effects
  - `'detrendq'`: Quarterly data with trends and seasonal effects

**Optional Parameters**:
- `vce` (str or None): Variance estimator (default: `None`, case-insensitive)
  - `None`: Homoskedastic standard errors (exact inference under normality)
  - `'robust'` or `'hc1'`: HC1 heteroskedasticity-robust standard errors
  - `'hc3'`: HC3 small-sample adjusted heteroskedasticity-robust standard errors
  - `'cluster'`: Cluster-robust standard errors (requires `cluster_var`)
- `cluster_var` (str): Cluster variable for cluster-robust standard errors (required when `vce='cluster'`)
  - Must be a column name in the data
  - Clusters are typically defined at a higher aggregation level than units
  - Inference uses G-1 degrees of freedom, where G is the number of clusters
- `controls` (list of str): Time-invariant control variables
  - Controls are included only if both N₁ > K+1 and N₀ > K+1 (where K is the number of controls)
  - If conditions are not met, controls are excluded and a warning is issued
- `ri` (bool): Enable randomization inference (default: `False`)
- `ri_method` (str): Resampling method for randomization inference (default: `'bootstrap'`)
  - `'bootstrap'`: With-replacement resampling
  - `'permutation'`: Without-replacement permutation (Fisher randomization inference)
- `rireps` (int): Number of replications for randomization inference (default: 1000)
- `seed` (int): Random seed for reproducibility
- `graph` (bool): Generate visualization (default: `False`)
  - If plotting fails, a warning is issued and estimation continues unaffected
- `gid` (str/int): Unit identifier for plotting (default: `None` for treated group mean)
- `graph_options` (dict): Matplotlib plotting options (default: `None`)
  - Supported keys: `figsize`, `title`, `xlabel`, `ylabel`, `legend_loc`, `dpi`, `savefig`

**Returns**: `LWDIDResults` object with the following attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `att` | float | Average treatment effect on treated |
| `se_att` | float | Standard error |
| `t_stat` | float | t-statistic |
| `pvalue` | float | Two-sided p-value |
| `ci_lower`, `ci_upper` | float | 95% confidence interval |
| `att_by_period` | DataFrame | Period-specific treatment effects |
| `ri_pvalue` | float | Randomization inference p-value (if `ri=True`) |
| `rireps` | int | Number of RI replications (if `ri=True`) |
| `ri_method` | str | RI method used: 'bootstrap' or 'permutation' (if `ri=True`) |
| `ri_valid` | int | Number of successful RI replications (if `ri=True`) |
| `ri_failed` | int | Number of failed RI replications (if `ri=True`) |
| `nobs` | int | Number of observations in the cross-sectional regression (equals number of units) |
| `n_treated` | int | Number of treated units |
| `n_control` | int | Number of control units |
| `df_resid` | int | Residual degrees of freedom (N - K - 1) |
| `df_inference` | int | Degrees of freedom used for inference (G - 1 for cluster-robust SE, df_resid otherwise) |

**Methods**:
- `summary()`: Print formatted results table
- `plot(gid=None, graph_options=None)`: Visualize transformed outcomes over time
  - Plots residualized outcomes after removing unit-specific patterns
  - Useful for assessing parallel trends assumption
  - `gid`: Unit identifier to plot (default: treated group mean)
  - `graph_options`: Dictionary of matplotlib options
- `to_excel(path)`: Export to Excel workbook
- `to_csv(path)`: Export period-specific effects to CSV
- `to_latex(path)`: Export to LaTeX table

### Usage Guidelines

**Inference Choice**:
- Use `vce=None` for exact inference when N is small and normality is plausible
- Use `vce='hc3'` for moderate samples (N ≥ 10) or when heteroskedasticity is suspected
- Use `vce='cluster'` for cluster-robust inference (requires `cluster_var`)
  - Inference uses df = G - 1 (number of clusters minus 1)
  - Actual degrees of freedom stored in `results.df_inference`
- Use randomization inference (`ri=True`) for finite-sample validity without distributional assumptions
  - Randomization inference uses homoskedastic standard errors to construct the null distribution
  - The `vce` option affects only classical t-based inference, not the randomization inference p-value

**Data Format**:
- **Panel structure**:
  - Data must be in long format (one row per unit-time observation)
  - Each (unit, time) combination must be unique
  - Time index must form a continuous sequence
  - Panels may be balanced or unbalanced across units
- **Treatment timing** (common timing assumption):
  - All treated units must begin treatment in the same period
  - The `post` indicator must be a function of time only
  - Treatment must be persistent (no reversals)
  - Staggered adoption is not supported (see Lee and Wooldridge 2025, Section 7)
- **Time variable format**:
  - Annual data: Single numeric column (e.g., `tvar='year'`)
  - Quarterly data: Two numeric columns (e.g., `tvar=['year', 'quarter']`)
- **Reserved column names**: Avoid `d_`, `post_`, `tindex`, `tq`, `ydot`, `ydot_postavg`, `firstpost`

## Examples

### California Smoking Restrictions

```python
# Analysis with single treated unit (N_treated = 1, N_control = 38)
data = pd.read_csv('smoking.csv')
results = lwdid(
    data,
    y='lcigsale',
    d='d',
    ivar='state',
    tvar='year',
    post='post',
    rolling='detrend',
    vce=None
)
print(results.summary())
```

See `examples/smoking.ipynb` for complete example.

## Testing

The package includes comprehensive tests:

```bash
pytest tests/
```

## Authors

Xuanyu Cai, Wenli Xu

## Contributing

Contributions are welcome. Please submit bug reports or feature requests via the issue tracker.
