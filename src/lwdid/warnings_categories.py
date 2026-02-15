"""
Warning category hierarchy for the lwdid package.

Provides structured warning categories for difference-in-differences
estimation, enabling selective filtering via Python's standard
``warnings.filterwarnings()`` mechanism. All warning classes inherit
from :class:`LWDIDWarning`, which itself inherits from :class:`UserWarning`,
preserving backward compatibility with existing filter rules.

The hierarchy covers small-sample conditions, overlap violations,
numerical instability, data quality issues, and convergence failures.

Examples
--------
Suppress only small-sample warnings while keeping others visible:

>>> import warnings
>>> from lwdid import SmallSampleWarning
>>> warnings.filterwarnings('ignore', category=SmallSampleWarning)

Suppress all lwdid warnings at once:

>>> warnings.filterwarnings('ignore', category=LWDIDWarning)
"""


class LWDIDWarning(UserWarning):
    """
    Base warning class for all lwdid package warnings.

    All custom warnings in the lwdid package inherit from this class,
    providing a common base for unified warning filtering. Because
    ``LWDIDWarning`` inherits from ``UserWarning``, existing calls to
    ``warnings.filterwarnings('ignore', category=UserWarning)`` will
    continue to suppress lwdid warnings.
    """
    pass


class SmallSampleWarning(LWDIDWarning):
    """
    Warning raised when sample size is too small for reliable inference.

    Triggered when the number of treated or control units in a
    (cohort, period) cell falls below the threshold required for
    stable estimation. Common in staggered adoption designs where
    late cohorts have few treated units.
    """
    pass


class OverlapWarning(LWDIDWarning):
    """
    Warning raised when the overlap (common support) assumption is suspect.

    Triggered when propensity scores are near 0 or 1, or when IPW
    weights exhibit extreme values, indicating poor overlap between
    treated and control covariate distributions.
    """
    pass


class NumericalWarning(LWDIDWarning):
    """
    Warning raised when numerical instability is detected.

    Triggered by near-singular matrices, extremely small standard
    errors, ill-conditioned design matrices, or other conditions
    that may compromise the precision of point estimates or
    variance calculations.
    """
    pass


class DataWarning(LWDIDWarning):
    """
    Warning raised for data quality issues.

    Triggered by missing values, implicit type coercions, duplicate
    observations, or other data anomalies that may affect estimation
    but do not prevent it from proceeding.
    """
    pass


class ConvergenceWarning(LWDIDWarning):
    """
    Warning raised when an iterative estimation procedure fails to converge.

    Triggered when logistic regression (propensity score estimation),
    matching algorithms, or other iterative methods do not reach the
    convergence criterion within the maximum number of iterations.
    """
    pass
