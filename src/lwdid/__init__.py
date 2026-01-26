"""
Difference-in-differences estimation with unit-specific transformations.

This package implements difference-in-differences (DiD) methods using
unit-specific transformations that convert panel data into cross-sectional
form. The approach supports both common timing and staggered adoption designs,
with inference methods appropriate for small, moderate, and large samples.

The core methodology removes pre-treatment unit-specific patterns (means or
linear trends) before applying standard treatment effect estimators. This
transformation-based approach enables exact t-based inference for small
samples, heteroskedasticity-robust inference for moderate samples, and
propensity score-based methods for large samples.

Transformation Methods
----------------------
Four unit-specific transformation methods remove pre-treatment patterns:

- ``demean`` : Unit-specific demeaning (subtracts pre-treatment mean)
- ``detrend`` : Unit-specific linear detrending (removes linear trend)
- ``demeanq`` : Quarterly demeaning with seasonal fixed effects
- ``detrendq`` : Quarterly detrending with seasonal effects and trends

Staggered adoption designs support only ``demean`` and ``detrend``.

Estimation Methods
------------------
Multiple estimators accommodate different sample sizes and assumptions:

- ``ra`` : Regression adjustment (default). Enables exact inference under
  classical linear model assumptions for small samples.
- ``ipw`` : Inverse probability weighting. Reweights control observations
  using propensity scores for large samples.
- ``ipwra`` : Doubly robust estimation combining propensity score weighting
  with outcome regression. Consistent if either model is correctly specified.
- ``psm`` : Propensity score matching with nearest-neighbor matching,
  caliper constraints, and replacement options.

Variance Estimation
-------------------
Flexible standard error computation methods:

- Homoskedastic : Exact t-based inference under normality assumption
- Heteroskedasticity-robust : HC0 through HC4 estimators
- Cluster-robust : For within-group correlation structures

Design Support
--------------
- Common timing : All treated units begin treatment simultaneously
- Staggered adoption : Treatment timing varies across cohorts with flexible
  control group selection (never-treated or not-yet-treated)

Main Components
---------------
lwdid : callable
    Primary estimation function accepting panel data and returning treatment
    effect estimates with standard errors and inference statistics.

LWDIDResults : class
    Results container with ``summary()``, ``plot()``, and export methods
    for presenting and saving estimation outputs.

validate_staggered_data : callable
    Comprehensive data validation for staggered adoption designs.

is_never_treated : callable
    Utility for determining control group membership.

LWDIDError : exception
    Base exception class with specialized subclasses for data validation,
    parameter validation, and estimation failures.

Examples
--------
Common timing design:

>>> import pandas as pd
>>> from lwdid import lwdid
>>> # Assuming panel_data is a DataFrame with required columns
>>> results = lwdid(
...     data=panel_data,
...     y='outcome',
...     d='treated',
...     ivar='unit_id',
...     tvar='year',
...     post='post_treatment',
...     rolling='demean',
... )
>>> print(results.summary())  # doctest: +SKIP

Staggered adoption design:

>>> results = lwdid(
...     data=panel_data,
...     y='outcome',
...     ivar='unit_id',
...     tvar='year',
...     gvar='first_treat_year',
...     rolling='demean',
...     estimator='ra',
...     aggregate='overall',
... )
>>> print(results.summary())  # doctest: +SKIP

Notes
-----
The transformation-based approach removes pre-treatment unit-specific
heterogeneity before estimation. For common timing designs, this enables
exact inference even with small numbers of treated or control units (as
few as one treated unit with two controls). For staggered designs,
cohort-time specific effects are estimated and aggregated using
cohort-share or observation-share weighting schemes.

See Also
--------
lwdid : Main estimation function with full parameter documentation.
LWDIDResults : Results container with output methods.
"""

try:
    from importlib.metadata import version as _version
    __version__ = _version("lwdid")
except ImportError:
    # Python < 3.8 fallback
    try:
        import pkg_resources
        __version__ = pkg_resources.get_distribution("lwdid").version
    except Exception:
        __version__ = "0.1.0"  # Fallback if package not installed

from .core import lwdid
from .results import LWDIDResults
from .staggered.control_groups import ControlGroupStrategy
from .exceptions import (
    InsufficientDataError,
    InsufficientPrePeriodsError,
    InsufficientQuarterDiversityError,
    InvalidParameterError,
    InvalidRollingMethodError,
    InvalidStaggeredDataError,
    InvalidVCETypeError,
    LWDIDError,
    MissingRequiredColumnError,
    NoControlUnitsError,
    NoNeverTreatedError,
    NoTreatedUnitsError,
    RandomizationError,
    TimeDiscontinuityError,
    VisualizationError,
)
from .validation import is_never_treated, validate_staggered_data

__all__ = [
    # Package metadata
    '__version__',
    # Main API
    'lwdid',
    'LWDIDResults',
    # Staggered design utilities
    'ControlGroupStrategy',
    # Validation utilities
    'is_never_treated',
    'validate_staggered_data',
    # Exception hierarchy
    'LWDIDError',
    'InvalidParameterError',
    'InvalidRollingMethodError',
    'InvalidStaggeredDataError',
    'InvalidVCETypeError',
    'InsufficientDataError',
    'InsufficientPrePeriodsError',
    'InsufficientQuarterDiversityError',
    'NoTreatedUnitsError',
    'NoControlUnitsError',
    'NoNeverTreatedError',
    'TimeDiscontinuityError',
    'MissingRequiredColumnError',
    'RandomizationError',
    'VisualizationError',
]
