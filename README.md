# lwdid

---

## Lee–Wooldridge Difference-in-Differences for Panel Data

![Version](https://img.shields.io/badge/version-0.2.1-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/license-AGPL--3.0-green.svg)
[![Documentation](https://readthedocs.org/projects/lwdid/badge/?version=latest)](https://lwdid.readthedocs.io)

![From Panel to Cross-Section](image/image.png)

`lwdid` implements the rolling-transformation DiD workflow described in
Lee & Wooldridge (2025, 2026) with source-level support for:

- common timing designs (`d` + `post`),
- staggered adoption designs (`gvar`),
- seasonal adjustments (`Q`, `season_var`) for quarterly, monthly, or weekly data,
- robust/cluster inference with HC0–HC4 standard errors for OLS-based paths,
- IPW / IPWRA / PSM estimators with propensity-score workflows,
- pre-treatment dynamics and parallel trends testing,
- randomization inference (`bootstrap` / `permutation`),
- diagnostic toolkits for trend assessment, sensitivity analysis, and clustering,
- event-study plotting and export utilities.

## Statement of Need

Difference-in-differences (DiD) is one of the most widely used methods for
causal inference in the social sciences. However, existing implementations
often assume large cross-sectional sample sizes and may perform poorly when
the number of treated or control units is small. Lee & Wooldridge (2026)
propose a rolling-transformation approach that converts panel DiD into a
cross-sectional treatment effects problem, enabling exact t-based inference
under classical linear model assumptions — even with as few as N₀ ≥ 1
control and N₁ ≥ 1 treated units (N ≥ 3 total). Lee & Wooldridge (2025)
extend this framework to staggered adoption designs with cohort-time
specific effects, multiple estimators (RA, IPW, IPWRA, PSM), and flexible
aggregation strategies.

`lwdid` provides a unified Python implementation of both papers, filling a
gap in the existing ecosystem where no Python package offers:

- exact small-sample inference for DiD,
- unit-specific rolling transformations (demeaning and detrending),
- integrated support for both common timing and staggered adoption,
- seasonal adjustment for high-frequency panel data,
- pre-treatment dynamics for parallel trends assessment,
- built-in diagnostic toolkits for trend heterogeneity, sensitivity, and clustering.

## Overview

`lwdid` follows the Lee–Wooldridge rolling-transformation strategy: transform
panel outcomes at the unit level using pre-treatment information, then estimate
treatment effects in a cross-sectional framework.

The package supports three practical DiD workflows:

- **Common timing** with exact/robust OLS inference (`d`, `post`).
- **Staggered adoption** with cohort-time effects and aggregation (`gvar`).
- **Repeated cross-section aggregation** to panel format before DiD estimation.

Compared with a single fixed estimator pipeline, `lwdid` exposes multiple
transformation and estimator combinations (`ra`, `ipw`, `ipwra`, `psm`) while
keeping inference and output APIs unified via `LWDIDResults`.

## Requirements

### Software

- Python `>=3.10,<3.13`
- `numpy>=1.20,<3.0`
- `pandas>=1.3,<3.0`
- `scipy>=1.7,<2.0`
- `statsmodels>=0.13,<1.0`
- `scikit-learn>=1.0`
- `matplotlib>=3.3`
- `openpyxl>=3.1`

### Data

Before estimation, ensure:

- Long-format panel structure (one row per unit-time observation),
- Unique `(ivar, time)` keys,
- Valid treatment coding for the selected mode:
  - common timing: `d` + `post`,
  - staggered: `gvar` with `0/NaN/inf` as never treated,
- Sufficient pre-treatment observations for the selected rolling method.

Detailed validation rules are documented in **Data Requirements and Validation Rules** below.

## Installation

Install from PyPI:

```bash
pip install lwdid
```

## Quick Start

### 1) Common Timing (small-N or large-N)

```python
import pandas as pd
from lwdid import lwdid

df = pd.read_csv("data/smoking.csv")

res = lwdid(
    data=df,
    y="lcigsale",
    d="d",
    ivar="state",
    tvar="year",
    post="post",
    rolling="detrend",
    estimator="ra",
    vce="hc3",          # None / hc0 / hc1(robust) / hc2 / hc3 / hc4 / cluster
    alpha=0.05,
)

print(res.summary())
print(res.att, res.se_att, res.pvalue)
```

### 2) Staggered Adoption

```python
import pandas as pd
from lwdid import lwdid

df = pd.read_csv("data/castle.csv")
df["gvar"] = df["effyear"].fillna(0)  # 0 / NaN / inf => never treated

res = lwdid(
    data=df,
    y="lhomicide",
    ivar="sid",
    tvar="year",
    gvar="gvar",
    rolling="demean",
    estimator="ra",
    control_group="not_yet_treated",
    aggregate="none",
    vce="hc3",
    include_pretreatment=True,
)

res.plot_event_study(aggregation="weighted", title="Castle Law Event Study")
```

### 3) Repeated Cross-Section → Panel Aggregation

```python
from lwdid import aggregate_to_panel

agg = aggregate_to_panel(
    data=raw_df,
    unit_var="state",
    time_var="year",
    outcome_var="outcome",
    weight_var="survey_weight",   # optional
    treatment_var="treated",      # optional consistency check
)

panel_df = agg.panel_data
```

## Core API

```python
lwdid(
    data, y, d=None, ivar=None, tvar=None, post=None, rolling="demean", *,
    gvar=None, control_group="not_yet_treated", estimator="ra", aggregate="cohort",
    balanced_panel="warn", ps_controls=None, trim_threshold=0.01,
    return_diagnostics=False, n_neighbors=1, caliper=None, with_replacement=True,
    match_order="data", vce=None, controls=None, cluster_var=None, alpha=0.05,
    ri=False, rireps=1000, seed=None, ri_method="bootstrap",
    graph=False, gid=None, graph_options=None,
    season_var=None, Q=4, auto_detect_frequency=False,
    include_pretreatment=False, pretreatment_test=True, pretreatment_alpha=0.05,
    exclude_pre_periods=0
)
```

## Method Selection Guide

### Transformation Methods

| Method | Data Type | Problem It Addresses | Practical Advantage |
|---|---|---|---|
| `demean` | Annual/ordered panel | Level differences across units | Most direct implementation of rolling DiD baseline |
| `detrend` | Annual/ordered panel | Unit-specific linear trends (CHT-style concern) | More robust when pre-trends differ by cohort; recommended for small samples (Lee & Wooldridge, 2026) |
| `demeanq` | Seasonal panel (quarterly/monthly/weekly via `Q`) | Seasonal level effects + unit heterogeneity | Handles periodic structure without explicit FE model rewriting |
| `detrendq` | Seasonal panel | Unit trends + seasonal structure | Most robust seasonal option when both trends and seasonality matter |

All four transformation methods are supported for both common timing and staggered adoption designs.

### When to Use `detrend` vs `demean`

Under the Conditional Heterogeneous Trends (CHT) assumption (Lee & Wooldridge, 2025), each treatment cohort may have its own linear trend. The `detrend` method removes unit-specific linear trends, relaxing the standard parallel trends assumption. Use `detrend` when:

- Pre-treatment outcome trends differ visibly across cohorts,
- You suspect unit-specific linear trends (e.g., differential growth rates),
- Small-sample settings where HC3 + detrending provides the best coverage (Lee & Wooldridge, 2026).

### Estimators

| Estimator | Best-Suited Setting | Main Use Case | Advantage | Inference Distribution |
|---|---|---|---|---|
| `ra` | Small or large N, strong linear adjustment | Baseline ATT estimation | Transparent OLS path + full `vce` support | **t-distribution** |
| `ipw` | Covariate imbalance with PS model | Reweight controls to treated support | Simpler weighting estimator | **normal (z)** |
| `ipwra` | Potential model misspecification risk | Doubly robust ATT | Consistent if PS or outcome model is correct | **normal (z)** |
| `psm` | Matchable treated/control support | Nearest-neighbor ATT | Intuitive matched-sample interpretation | **normal (z)** |

Note: IPW, IPWRA, and PSM currently use normal-based inference. The `ra` estimator uses t-distribution inference with degrees of freedom df = N₁ + N₀ − 2, which is recommended for small samples per Lee & Wooldridge (2026). A future release will migrate IPW, IPWRA, and PSM to t-distribution inference as well.

## Inference Behavior

- `vce` options for OLS-based paths (`ra`): `None`, `hc0`, `hc1`/`robust`, `hc2`, `hc3`, `hc4`, `cluster`.
- HC3 is recommended for small samples (Lee & Wooldridge, 2026).
- Cluster OLS uses `df = G - 1` (`G` = number of clusters). When the unit of analysis is finer than the policy level (e.g., county data with state-level policy), cluster at the policy level (Lee & Wooldridge, 2026).
- `ra` computes p-values and CIs from t-distribution critical values with df = N₁ + N₀ − 2.
- `ipw`, `ipwra`, and `psm` compute p-values and CIs from normal critical values.
- `ri=True` enables randomization inference for exact p-values:
  - `ri_method="bootstrap"` (default),
  - `ri_method="permutation"`.
- Event-study weighted aggregation (`plot_event_study(aggregation="weighted")`) uses t-based CI construction with a conservative df strategy (minimum df across contributing cohorts).

### Minimum Sample Requirements

Following Lee & Wooldridge (2026), exact inference requires:

- N₀ ≥ 1 (at least 1 control unit),
- N₁ ≥ 1 (at least 1 treated unit),
- N = N₀ + N₁ ≥ 3 (at least 3 total units).

## Mode-Specific Parameters

### Common Timing Mode (`gvar=None`)

Required: `d`, `post`, `ivar`, `tvar`.

- `control_group` and `aggregate` are ignored in this mode.
- `d` must be time-invariant within unit.
- `post` must be monotone (no treatment reversal).

### Staggered Mode (`gvar` provided)

Required: `gvar`, `ivar`, `tvar`.

- `d` and `post` are ignored if provided together with `gvar`.
- `gvar` coding: positive values = first treatment period; `0` / `NaN` / `np.inf` = never treated.
- `control_group`: `not_yet_treated`, `never_treated`, `all_others`.
- `aggregate`: `none`, `cohort`, `overall`.
- `aggregate in {"cohort","overall"}` requires never-treated units; control strategy auto-switches to `never_treated`.

### Pre-treatment Dynamics and Parallel Trends

Set `include_pretreatment=True` (staggered mode) to compute pre-treatment transformed outcomes for parallel trends assessment. The transformation uses future pre-treatment periods {t+1, ..., g−1} as reference (Lee & Wooldridge, 2025):

- **Demeaning**: ẏ_{itg} = Y_{it} − mean(Y_{i,t+1}, ..., Y_{i,g−1})
- **Detrending**: Ÿ_{itg} = Y_{it} − fitted value from regressing future pre-treatment outcomes on time

The period t = g−1 serves as the anchor point (reference baseline) for event study visualization. Set `pretreatment_test=True` to run individual t-tests and a joint F-test for H₀: all pre-treatment ATT = 0.

### No-Anticipation Robustness

Set `exclude_pre_periods=k` to exclude the k periods immediately before treatment from the pre-treatment sample used for transformation. This addresses potential violations of the no-anticipation assumption when units may adjust behavior before formal treatment.

## Data Requirements and Validation Rules

- Long format panel (`one row = one unit-time observation`).
- Unique `(ivar, time)` combinations required.
- Minimum sample size check enforced (`N >= 3` units).
- `controls` are treated as time-invariant regressors.
- Seasonal modes require `season_var` (or legacy `tvar=[year, quarter]`) and valid values in `1..Q`.
- Reserved internal column names should not appear in raw input:
  - `d_`, `post_`, `tindex`, `tq`, `ydot`, `ydot_postavg`, `firstpost`.

### Unbalanced Panels

The `balanced_panel` parameter controls handling of unbalanced panels:

- `"warn"` (default): Issue a warning with selection mechanism diagnostics.
- `"error"`: Raise an error if the panel is unbalanced.
- `"ignore"`: Proceed silently.

Selection may depend on unobserved time-invariant heterogeneity, but cannot systematically depend on Y_{it}(∞) shocks (Lee & Wooldridge, 2025). Minimum pre-treatment observation requirements:

- `demean`: at least 1 pre-treatment period per unit.
- `detrend`: at least 2 pre-treatment periods per unit.
- `demeanq`: at least Q + 1 pre-treatment periods per unit.
- `detrendq`: at least Q + 2 pre-treatment periods per unit.

## Returned Object (`LWDIDResults`)

Core fields:

- `att`, `se_att`, `t_stat`, `pvalue`, `ci_lower`, `ci_upper`
- `nobs`, `n_treated`, `n_control`, `df_resid`, `df_inference`
- `att_by_period` (common timing)
- `att_by_cohort_time`, `att_by_cohort`, `att_overall` (staggered, depending on `aggregate`)
- `att_pre_treatment`, `parallel_trends_test` (if `include_pretreatment=True`)

Key methods:

- `summary()`
- `plot(...)` (common-timing style transformed-outcome plot)
- `plot_event_study(...)` (staggered event-study visualization)
- `to_excel(...)`, `to_csv(...)`, `to_latex(...)`

## Diagnostic Toolkits

`lwdid` includes built-in diagnostic modules for common methodological concerns:

### Trend Diagnostics

```python
from lwdid import test_parallel_trends, diagnose_heterogeneous_trends, recommend_transformation

# Test parallel trends assumption
pt_result = test_parallel_trends(data, y="outcome", ivar="unit", tvar="year", gvar="gvar")

# Diagnose heterogeneous trends across cohorts
ht_diag = diagnose_heterogeneous_trends(data, y="outcome", ivar="unit", tvar="year", gvar="gvar")

# Get transformation recommendation (demean vs detrend)
rec = recommend_transformation(data, y="outcome", ivar="unit", tvar="year", gvar="gvar")
```

### Sensitivity Analysis

```python
from lwdid import robustness_pre_periods, sensitivity_no_anticipation, sensitivity_analysis

# Robustness to number of pre-treatment periods
rob = robustness_pre_periods(data, y="outcome", ivar="unit", tvar="year", d="d", post="post")

# Sensitivity to no-anticipation assumption violation
na_sens = sensitivity_no_anticipation(data, y="outcome", ivar="unit", tvar="year", d="d", post="post")

# Comprehensive sensitivity analysis
comp = sensitivity_analysis(data, y="outcome", ivar="unit", tvar="year", d="d", post="post")
```

### Clustering Diagnostics

```python
from lwdid import diagnose_clustering, recommend_clustering_level

# Diagnose clustering structure
clust_diag = diagnose_clustering(data, ivar="county", potential_cluster_vars=["state", "region"], gvar="gvar")

# Get clustering level recommendation
rec = recommend_clustering_level(data, ivar="county", tvar="year", potential_cluster_vars=["state", "region"], gvar="gvar")
```

### Selection Diagnostics

```python
from lwdid import diagnose_selection_mechanism

# Diagnose selection mechanism for unbalanced panels
sel_diag = diagnose_selection_mechanism(data, ivar="unit", tvar="year", y="outcome")
```

## Current Boundaries and Caveats

- `graph=True` in staggered mode is not executed directly; use `results.plot_event_study()` after estimation.
- `balanced_panel="error"` is the strict enforcement mode; `warn`/`ignore` do not trigger the same hard check.
- When overlap is weak, propensity-score estimators may trim many observations (`trim_threshold`), potentially reducing effective sample size.
- IPWRA and PSM currently use normal-distribution-based inference; a future release will migrate these to t-distribution inference consistent with the `ra` path.

## Documentation

Full API documentation is available at [lwdid.readthedocs.io](https://lwdid.readthedocs.io).

## Version Notes

- Current package version: `0.2.1` (`pyproject.toml`).
- Includes generalized seasonal support (`Q`, `season_var`) and staggered seasonal transformations (`demeanq`, `detrendq` in staggered mode).
- Pre-treatment dynamics and parallel trends testing (`include_pretreatment`, `pretreatment_test`).
- No-anticipation robustness check (`exclude_pre_periods`).
- Diagnostic toolkits: trend diagnostics, sensitivity analysis, clustering diagnostics, selection diagnostics.
- Wild cluster bootstrap inference.

## Getting Help

- Report bugs or request features via [GitHub Issues](https://github.com/gorgeousfish/lwdid-py/issues).
- For questions about methodology, consult the referenced papers or open a discussion on the repository.

## Contributing

Contributions are welcome. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to report bugs, suggest features, and submit pull requests.

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).

## Citation

If you use `lwdid` in your research, please cite both the software and the underlying papers. A machine-readable citation file is provided in [CITATION.cff](CITATION.cff).

```bibtex
@software{cai_lwdid_2025,
  author  = {Cai, Xuanyu and Xu, Wenli},
  title   = {lwdid: Lee--Wooldridge Difference-in-Differences for Panel Data},
  version = {0.2.1},
  year    = {2025},
  url     = {https://github.com/gorgeousfish/lwdid-py}
}
```

## References

- Lee, Soo Jeong and Wooldridge, Jeffrey M., *Simple Approaches to Inference with Difference-in-Differences Estimators with Small Cross-Sectional Sample Sizes* (January 03, 2026). Available at SSRN: [https://ssrn.com/abstract=5325686](https://ssrn.com/abstract=5325686) or [http://dx.doi.org/10.2139/ssrn.5325686](https://dx.doi.org/10.2139/ssrn.5325686)
- Lee, Soo Jeong and Wooldridge, Jeffrey M., *A Simple Transformation Approach to Difference-in-Differences Estimation for Panel Data* (December 25, 2025). Available at SSRN: [https://ssrn.com/abstract=4516518](https://ssrn.com/abstract=4516518) or [http://dx.doi.org/10.2139/ssrn.4516518](https://dx.doi.org/10.2139/ssrn.4516518)

## License

This project is licensed under the **GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)**.
See `LICENSE` for the full license text.

## Authors

Xuanyu Cai, Wenli Xu
