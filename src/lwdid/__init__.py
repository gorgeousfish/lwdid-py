"""
lwdid: Difference-in-Differences Estimator for Small Cross-Sectional Samples
=============================================================================

Python implementation of the Lee and Wooldridge (2025) difference-in-differences
estimator for panel data with small cross-sectional sample sizes.

This package implements the methodology described in Lee and Wooldridge (2025),
providing inference for difference-in-differences estimation when the number
of treated or control units is small.

Key Features
------------
- Small-sample inference: Designed for settings with small numbers of treated
  or control units. Exact t-based inference available under classical linear
  model assumptions (normality and homoskedasticity). Heteroskedasticity-robust
  standard errors (HC1/HC3) for moderate sample sizes
- Four transformation methods:

  * ``demean``: Unit-specific demeaning (Procedure 2.1)
  * ``detrend``: Unit-specific detrending (Procedure 3.1)
  * ``demeanq``: Quarterly demeaning with seasonal effects
  * ``detrendq``: Quarterly detrending with linear trends and seasonal effects

- Variance estimation: Homoskedastic (exact t-inference), HC1/HC3
  (heteroskedasticity-robust), and cluster-robust standard errors
- Randomization inference: Bootstrap or permutation-based p-values for
  finite-sample validity without distributional assumptions
- Period-specific effects: Separate treatment effect estimates for each
  post-treatment period
- Control variables: Time-invariant covariates
- Visualization: Residualized outcome plots
- Export: Excel, CSV, and LaTeX output formats

Main Components
---------------
lwdid : function
    Main estimation function. See ``help(lwdid)`` for detailed documentation.
LWDIDResults : class
    Results container with ``summary()``, ``plot()``, and export methods.
Exception hierarchy : module
    Typed exceptions inheriting from ``LWDIDError``.

Quick Start
-----------
>>> import pandas as pd
>>> from lwdid import lwdid
>>>
>>> # Load panel data
>>> data = pd.read_csv('smoking.csv')
>>>
>>> # Estimate ATT
>>> results = lwdid(
...     data=data,
...     y='lcigsale',
...     d='treated',
...     ivar='state',
...     tvar='year',
...     post='post',
...     rolling='demean'
... )
>>>
>>> # View results
>>> print(results.summary())
>>> results.plot()
>>>
>>> # Export results
>>> results.to_excel('results.xlsx')

Notes
-----
This implementation is designed for common-treatment-timing settings where all
treated units begin treatment in the same period. Staggered adoption is not
supported.

References
----------
Lee, S. J., and Wooldridge, J. M. (2025). Simple Approaches to Inference with
Difference-in-Differences Estimators with Small Cross-Sectional Sample Sizes.
Available at SSRN 5325686.
"""

# Export main function
from .core import lwdid

# Export results class
from .results import LWDIDResults

# Export exception classes
from .exceptions import (
    InsufficientDataError,
    InsufficientPrePeriodsError,
    InsufficientQuarterDiversityError,
    InvalidParameterError,
    InvalidRollingMethodError,
    InvalidVCETypeError,
    LWDIDError,
    MissingRequiredColumnError,
    NoControlUnitsError,
    NoTreatedUnitsError,
    RandomizationError,
    TimeDiscontinuityError,
    VisualizationError,
)

__all__ = [
    # Main function
    'lwdid',
    # Results class
    'LWDIDResults',
    # Exception classes
    'LWDIDError',
    'InvalidParameterError',
    'InvalidRollingMethodError',
    'InvalidVCETypeError',
    'InsufficientDataError',
    'InsufficientPrePeriodsError',
    'InsufficientQuarterDiversityError',
    'NoTreatedUnitsError',
    'NoControlUnitsError',
    'TimeDiscontinuityError',
    'MissingRequiredColumnError',
    'RandomizationError',
    'VisualizationError',
]

