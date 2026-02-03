# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-01-31

### Added

#### Generalized Seasonal Adjustment (Q Parameter)

- **`Q` parameter**: Extended `demeanq` and `detrendq` transformations to support arbitrary seasonal periods:
  - `Q=4` (default): Quarterly data with 4 seasons per year
  - `Q=12`: Monthly data with 12 seasons per year
  - `Q=52`: Weekly data with 52 seasons per year

- **`season_var` parameter**: New parameter to specify the season indicator column for monthly/weekly data. Acts as an alias for `quarter` parameter with extended range support.

- **`detect_frequency()` function**: Automatic data frequency detection based on time intervals and annual observation counts.

- **`auto_detect_frequency` parameter**: Optional automatic frequency detection in `lwdid()` function.

#### Staggered Design Seasonal Support

- **`transform_staggered_demeanq()`**: Cohort-specific seasonal demeaning for staggered adoption designs.

- **`transform_staggered_detrendq()`**: Cohort-specific seasonal detrending for staggered adoption designs.

### Changed

- **`demeanq_unit()`**: Now accepts `Q` parameter for generalized seasonal periods (default Q=4 for backward compatibility).

- **`detrendq_unit()`**: Now accepts `Q` parameter for generalized seasonal periods (default Q=4 for backward compatibility).

- **`validate_season_coverage()`**: Renamed from `validate_quarter_coverage()` with `Q` parameter support. Original function name retained as alias for backward compatibility.

- **`validate_season_diversity()`**: Renamed from `validate_quarter_diversity()` with `Q` parameter support. Original function name retained as alias for backward compatibility.

- **Minimum pre-treatment requirements**: Updated to depend on Q:
  - `demeanq`: n_pre ≥ Q + 1 per unit
  - `detrendq`: n_pre ≥ Q + 2 per unit

### Documentation

- Updated methodological notes with generalized seasonal transformation formulas.
- Added monthly data (Q=12) and weekly data (Q=52) usage examples.
- Updated user guide with Q parameter documentation for all seasonal transformations.

### Backward Compatibility

- All changes are backward compatible. Existing code using `demeanq`/`detrendq` with quarterly data will continue to work without modification.
- The `quarter` parameter remains supported as an alias for `season_var`.
- Original validation function names (`validate_quarter_coverage`, `validate_quarter_diversity`) remain available as aliases.

## [0.1.0] - 2026-01-15

### Added

- Initial release of lwdid package.
- Core transformations: `demean`, `detrend`, `demeanq`, `detrendq`.
- Common timing DiD estimation with exact t-based inference.
- Staggered adoption DiD with cohort-time specific effects.
- Multiple variance estimators: OLS, HC0-HC4, cluster-robust.
- Randomization inference (bootstrap and permutation methods).
- Multiple estimators: RA, IPW, IPWRA, PSM.
- Comprehensive result export: Excel, CSV, LaTeX.
- Event study visualization for staggered designs.
