# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.2.1 - 2026-02-14

### Performance

- **Common timing RI** (`randomization.py`): No-controls path uses direct ATT computation with batch vectorization; with-controls path uses pre-allocated design matrix + `np.linalg.lstsq`
- **Staggered RI** (`staggered/randomization.py`): `n_jobs` parameter for joblib parallel resampling; `SeedSequence`-derived child seeds for deterministic parallelism; `legacy_seed` for backward compatibility
- **Wild cluster bootstrap** (`inference/wild_bootstrap.py`): Vectorized OLS and batch matrix operations; shared pre-computation across grid points

---

## 0.2.0 - 2026-01-31

### Added

#### Staggered Adoption DiD (Lee & Wooldridge, 2025; SSRN 4516518)

- **`staggered/` subpackage**: Complete implementation of staggered adoption difference-in-differences with cohort-time specific effects, comprising 10 new modules:

  - `estimation.py`: Cohort-time specific ATT estimation with OLS-based regression adjustment
  - `estimators.py`: IPW, IPWRA, and PSM estimators for staggered designs
  - `aggregation.py`: Cohort-level and overall effect aggregation (§7 of paper)
  - `control_groups.py`: Control group selection strategies (never-treated / not-yet-treated) with `ControlGroupStrategy` enum
  - `transformations.py`: Cohort-specific rolling transformations (demean, detrend)
  - `transformations_pre.py`: Pre-treatment period rolling transformations (Appendix D)
  - `estimation_pre.py`: Pre-treatment effect estimation for parallel trends assessment
  - `parallel_trends.py`: Formal parallel trends testing with joint F-test
  - `randomization.py`: Randomization inference for staggered designs
- **3 new estimators** for staggered designs:

  - `ipw`: Inverse probability weighting with propensity score trimming
  - `ipwra`: Doubly robust estimation combining IPW with regression adjustment
  - `psm`: Propensity score matching with nearest-neighbor matching, caliper constraints, and replacement options
- **New `lwdid()` parameters for staggered designs**:

  - `gvar`: First treatment period column (activates staggered mode)
  - `control_group`: Control group strategy (`'not_yet_treated'` or `'never_treated'`)
  - `estimator`: Estimation method (`'ra'`, `'ipw'`, `'ipwra'`, `'psm'`)
  - `aggregate`: Aggregation level (`'none'`, `'cohort'`, `'overall'`)
  - `ps_controls`: Propensity score model covariates
  - `trim_threshold`: Propensity score trimming threshold (default 0.01)
  - `n_neighbors`: Number of nearest neighbors for PSM (default 1)
  - `caliper`: Maximum distance for PSM matching
  - `with_replacement`: Whether PSM uses replacement (default True)
  - `match_order`: PSM matching order (`'data'`)
  - `return_diagnostics`: Return diagnostic information (default False)
- **Pre-treatment dynamics** (Appendix D of SSRN 4516518):

  - `include_pretreatment`: Compute pre-treatment transformed outcomes for event study plots
  - `pretreatment_test`: Run parallel trends test on pre-treatment effects (default True)
  - `pretreatment_alpha`: Significance level for parallel trends test (default 0.05)
- **No-anticipation robustness**:

  - `exclude_pre_periods`: Number of pre-treatment periods to exclude from transformation window for robustness to anticipation effects
- **Event study visualization**: `plot_event_study()` method in `LWDIDResults` for staggered designs with weighted event-time aggregation (WATT)

#### Variance Estimation Enhancements

- **3 new HC variance estimators**: HC0 (White, 1980), HC2 (leverage-adjusted), HC4 (adaptive) added to `vce` parameter options alongside existing HC1 and HC3
- **`alpha` parameter**: Configurable significance level for confidence intervals (default 0.05, previously hardcoded)

#### Diagnostic Modules (4 new modules)

- **`clustering_diagnostics.py`**: Clustering level diagnostics and recommendations

  - `diagnose_clustering()`: Analyze cluster structure and effective sample sizes
  - `recommend_clustering_level()`: Suggest optimal clustering level
  - `check_clustering_consistency()`: Verify clustering consistency across specifications
- **`selection_diagnostics.py`**: Selection mechanism diagnostics for unbalanced panels

  - `diagnose_selection_mechanism()`: Comprehensive selection bias assessment
  - `get_unit_missing_stats()`: Unit-level missing data statistics
  - `plot_missing_pattern()`: Visualize missing data patterns
- **`sensitivity.py`**: Sensitivity analysis tools

  - `robustness_pre_periods()`: Pre-treatment period robustness checks
  - `sensitivity_no_anticipation()`: No-anticipation assumption sensitivity
  - `sensitivity_analysis()`: Comprehensive sensitivity analysis
  - `plot_sensitivity()`: Sensitivity analysis visualization
- **`trend_diagnostics.py`**: Parallel trends diagnostics

  - `test_parallel_trends()`: Formal parallel trends testing
  - `diagnose_heterogeneous_trends()`: Heterogeneous trend detection
  - `recommend_transformation()`: Data-driven transformation method recommendation
  - `plot_cohort_trends()`: Cohort-specific trend visualization

#### Inference

- **`inference/wild_bootstrap.py`**: Wild cluster bootstrap for cluster-robust inference
  - `wild_cluster_bootstrap()`: Rademacher and Webb weight distributions

#### Data Preprocessing

- **`preprocessing/aggregation.py`**: Repeated cross-section data aggregation
  - `aggregate_to_panel()`: Aggregate individual-level data to unit-period panel format
  - `AggregationResult` and `CellStatistics` data classes

#### Unbalanced Panel Support

- **`balanced_panel` parameter**: Control unbalanced panel handling (`'warn'`, `'error'`, `'ignore'`)
- **`UnbalancedPanelError`**: New exception with diagnostic attributes (`min_obs`, `max_obs`, `n_incomplete_units`)

#### Generalized Seasonal Adjustment (Q Parameter)

- **`Q` parameter**: Extended `demeanq` and `detrendq` transformations to support arbitrary seasonal periods:

  - `Q=4` (default): Quarterly data with 4 seasons per year
  - `Q=12`: Monthly data with 12 seasons per year
  - `Q=52`: Weekly data with 52 seasons per year
- **`season_var` parameter**: Season indicator column for monthly/weekly data. Acts as an alias for `quarter` parameter with extended range support.
- **`detect_frequency()` function**: Automatic data frequency detection based on time intervals and annual observation counts.
- **`auto_detect_frequency` parameter**: Optional automatic frequency detection in `lwdid()`.
- **Staggered seasonal support**:

  - `transform_staggered_demeanq()`: Cohort-specific seasonal demeaning
  - `transform_staggered_detrendq()`: Cohort-specific seasonal detrending

#### Exception Hierarchy Expansion

- **6 new exception classes**:

  - `UnbalancedPanelError`: Unbalanced panel detection with diagnostic attributes
  - `InvalidStaggeredDataError`: Staggered data validation failures
  - `NoNeverTreatedError`: Missing never-treated units when required for aggregation
  - `AggregationError`: Base class for aggregation errors
  - `InvalidAggregationError`: Invalid aggregation constraints (treatment varies within cell, etc.)
  - `InsufficientCellSizeError`: All cells below minimum size threshold
- **Enhanced `InsufficientPrePeriodsError`**: Added `cohort`, `available`, `required`, `excluded` attributes for staggered designs with `exclude_pre_periods`

### Changed

#### API Changes

- **`d` parameter**: Changed from required (`str`) to optional (`str | None = None`). Not needed when using `gvar` for staggered designs.
- **`ivar` parameter**: Changed from required (`str`) to optional (`str | None = None`).
- **`tvar` parameter**: Changed from required to optional (`str | list[str] | None = None`).
- **`post` parameter**: Changed from required (`str`) to optional (`str | None = None`). Not needed when using `gvar` for staggered designs.
- **`rolling` parameter**: Changed from required (`str`) to optional with default (`str = 'demean'`).
- **Type annotations**: Modernized to Python 3.10+ union syntax (`str | None` instead of `Optional[str]`).
- **Keyword-only arguments**: New staggered parameters are keyword-only (after `*` separator).

#### Seasonal Transformations

- **`demeanq_unit()`**: Now accepts `Q` parameter for generalized seasonal periods (default Q=4 for backward compatibility).
- **`detrendq_unit()`**: Now accepts `Q` parameter for generalized seasonal periods (default Q=4 for backward compatibility).
- **`validate_season_coverage()`**: Renamed from `validate_quarter_coverage()` with `Q` parameter support. Original function name retained as alias.
- **`validate_season_diversity()`**: Renamed from `validate_quarter_diversity()` with `Q` parameter support. Original function name retained as alias.
- **Minimum pre-treatment requirements**: Updated to depend on Q:
  - `demeanq`: n_pre ≥ Q + 1 per unit
  - `detrendq`: n_pre ≥ Q + 2 per unit

#### Numerical Stability

- Time centering in `detrend` transformation for improved numerical conditioning
- Variance threshold checks with NaN warnings for degenerate cases

#### Project Metadata

- **Description**: Changed from "Lee & Wooldridge Difference-in-Differences estimator for small cross-sectional sample sizes" to "Lee & Wooldridge Difference-in-Differences with rolling transformations for panel data"
- **New dependency**: `scikit-learn>=1.0` (required for PSM estimator)

### Fixed

- Division by zero in `randomization.py` when `failure_rate=1.0`

### Exports

- **70+ exported symbols** (up from 13 in v0.1.0), including all new diagnostic classes, enums, and utility functions

### Backward Compatibility

- All v0.1.0 common timing functionality remains fully compatible
- The `quarter` parameter remains supported as an alias for `season_var`
- Original validation function names (`validate_quarter_coverage`, `validate_quarter_diversity`) remain available as aliases
- Existing code using `demeanq`/`detrendq` with quarterly data continues to work without modification

---

## 0.1.0 - 2026-01-15

### Added

- Initial release of lwdid package
- **Core transformations**: `demean`, `detrend`, `demeanq`, `detrendq` (unit-specific rolling transformations)
- **Common timing DiD estimation**: Exact t-based inference under classical linear model (CLM) assumptions for small cross-sectional samples
- **Estimation method**: Regression adjustment (RA) only
- **Variance estimators**: Homoskedastic (None), heteroskedasticity-robust (HC1/`'robust'`, HC3), cluster-robust
- **Randomization inference**: Bootstrap and permutation methods for finite-sample p-values
- **Period-specific effects**: Separate treatment effect estimates for each post-treatment period
- **Control variables**: Time-invariant covariates support
- **Visualization**: Residualized outcome plots
- **Result export**: Excel, CSV, and LaTeX output formats
- **Exception hierarchy**: 12 typed exceptions inheriting from `LWDIDError`
- **9 source modules**: core, estimation, exceptions, randomization, results, transformations, validation, visualization, `__init__`
