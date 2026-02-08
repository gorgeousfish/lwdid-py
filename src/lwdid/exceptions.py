"""
Exception hierarchy for the lwdid package.

Provides a structured exception hierarchy for difference-in-differences
estimation. All exceptions inherit from LWDIDError, enabling unified
error handling across the package.

The hierarchy covers parameter validation, data insufficiency, time series
requirements, randomization inference, visualization, and staggered designs.
"""


class LWDIDError(Exception):
    """
    Base exception class for all lwdid package errors.

    All custom exceptions in the lwdid package inherit from this class,
    providing a common base for unified error handling.
    """
    pass


class InvalidParameterError(LWDIDError):
    """
    Exception raised when input parameter validation fails.

    This is a general exception for invalid parameter values that do not
    fall into more specific categories. Common triggers include:

    - Invalid data types for control variables
    - Cluster variable specified without vce='cluster'
    - Other parameter constraint violations

    See Also
    --------
    InvalidRollingMethodError : For invalid rolling method values.
    InvalidVCETypeError : For invalid variance estimator types.
    """
    pass


class InvalidRollingMethodError(InvalidParameterError):
    """
    Exception raised when the rolling parameter has an invalid value.

    The rolling parameter must be one of: 'demean', 'detrend', 'demeanq', or
    'detrendq'. This exception is raised during input validation when an
    unsupported transformation method is specified.

    See Also
    --------
    InvalidParameterError : Parent class for parameter validation errors.
    """
    pass


class InvalidVCETypeError(InvalidParameterError):
    """
    Exception raised when the vce parameter has an invalid value.

    The vce (variance-covariance estimator) parameter must be one of: None,
    'robust', 'hc0', 'hc1', 'hc2', 'hc3', 'hc4', or 'cluster'. This
    exception is raised during estimation when an unsupported variance
    estimator type is specified.

    See Also
    --------
    InvalidParameterError : Parent class for parameter validation errors.
    """
    pass


class UnbalancedPanelError(LWDIDError):
    """
    Exception raised when balanced panel is required but data is unbalanced.
    
    This exception is raised when ``balanced_panel='error'`` is specified
    and the panel data contains units with different numbers of observations.
    
    Attributes
    ----------
    min_obs : int
        Minimum observations per unit in the panel.
    max_obs : int
        Maximum observations per unit in the panel.
    n_incomplete_units : int
        Number of units with fewer than max_obs observations.
    
    Notes
    -----
    Unbalanced panels arise when units have different numbers of observed
    time periods. Under standard selection assumptions, this is acceptable
    provided that missingness depends only on time-invariant unit heterogeneity
    and not on time-varying shocks. Users may want to enforce balanced panels
    for sensitivity analysis or when the selection mechanism is questionable.

    See Also
    --------
    LWDIDError : Base exception class.
    diagnose_selection_mechanism : Diagnostic tools for selection bias.
    """
    
    def __init__(
        self,
        message: str,
        min_obs: int,
        max_obs: int,
        n_incomplete_units: int,
    ):
        super().__init__(message)
        self.min_obs = min_obs
        self.max_obs = max_obs
        self.n_incomplete_units = n_incomplete_units


class InsufficientDataError(LWDIDError):
    """
    Exception raised when sample size is insufficient for estimation.

    This is a general exception for data insufficiency issues. More specific
    subclasses indicate the exact nature of the insufficiency (e.g., no treated
    units, no control units, insufficient pre-periods).

    See Also
    --------
    NoTreatedUnitsError : No units with treatment indicator d=1.
    NoControlUnitsError : No units with treatment indicator d=0.
    InsufficientPrePeriodsError : Insufficient pre-treatment periods.
    """
    pass


class NoTreatedUnitsError(InsufficientDataError):
    """
    Exception raised when there are no treated units in the data.

    Raised when all units have treatment indicator d=0 in the panel or
    regression sample. At least one treated unit (d=1) is required for
    difference-in-differences estimation.
    """
    pass


class NoControlUnitsError(InsufficientDataError):
    """
    Exception raised when there are no control units in the data.

    Raised when all units have treatment indicator d=1 in the panel or
    regression sample. At least one control unit (d=0) is required for
    difference-in-differences estimation.
    """
    pass


class InsufficientPrePeriodsError(InsufficientDataError):
    """
    Exception raised when pre-treatment periods are insufficient.

    Raised when the number of pre-treatment periods is too small for the chosen
    rolling transformation. Different methods have different minimum global and
    unit-level requirements:

    - demean: Each unit must have at least 1 pre-treatment observation.

    - detrend: (i) The panel must contain at least 2 pre-treatment periods in total
      (T0 >= 2); and (ii) Each unit must have at least 2 pre-treatment observations
      so that a unit-specific linear trend can be estimated.

    - demeanq: (i) The panel must contain at least 1 pre-treatment period in total
      (T0 >= 1); (ii) Each unit must have at least 1 pre-treatment observation; and
      (iii) For each unit, the number of pre-treatment observations must be at least
      the number of distinct pre-period quarters plus one, ``n_pre >= (#quarters_pre + 1)``,
      to ensure positive degrees of freedom when estimating quarterly fixed effects.

    - detrendq: (i) The panel must contain at least 2 pre-treatment periods in total
      (T0 >= 2); (ii) Each unit must have at least 2 pre-treatment observations; and
      (iii) For each unit, the number of pre-treatment observations must be at least
      1 plus the number of distinct pre-period quarters, ``n_pre >= (1 + #quarters_pre)``,
      to avoid rank deficiency when estimating a linear trend with quarterly effects.

    This exception is also raised in staggered adoption designs when
    ``exclude_pre_periods`` is specified and the remaining pre-treatment
    periods are insufficient for the chosen transformation method.

    Attributes
    ----------
    cohort : int or None
        The treatment cohort identifier that triggered the error.
        Only set in staggered adoption mode.
    available : int or None
        Number of pre-treatment periods remaining after exclusion.
        Only set when exclude_pre_periods is used.
    required : int or None
        Minimum number of pre-treatment periods required by the
        transformation method (1 for demean, 2 for detrend, etc.).
    excluded : int or None
        Number of periods excluded via exclude_pre_periods parameter.

    See Also
    --------
    lwdid.transformations.apply_rolling_transform : Applies rolling
        transformations and enforces pre-period requirements.

    Notes
    -----
    When the no-anticipation assumption may be violated, excluding periods
    immediately before treatment from the transformation window can provide
    robustness. For cohort g with ``exclude_pre_periods=k``, the pre-treatment
    window becomes {T_min, ..., g-1-k} instead of {T_min, ..., g-1}.
    """

    def __init__(
        self,
        message: str,
        cohort: int | None = None,
        available: int | None = None,
        required: int | None = None,
        excluded: int | None = None,
    ):
        super().__init__(message)
        self.cohort = cohort
        self.available = available
        self.required = required
        self.excluded = excluded


class InsufficientQuarterDiversityError(InsufficientDataError):
    """
    Exception raised when quarterly data requirements are not met.

    Raised for quarterly transformation methods (``demeanq``/``detrendq``) when
    quarter coverage is insufficient: post-treatment periods contain quarters
    that do not appear in the pre-treatment period for a given unit, preventing
    estimation of seasonal effects for those quarters.

    See Also
    --------
    lwdid.validation.validate_quarter_coverage : Quarter coverage validation.
    """
    pass


class TimeDiscontinuityError(LWDIDError):
    """
    Exception raised when time series is discontinuous or post variable is non-monotone.

    This exception is raised in two scenarios:

    1. **Time index discontinuity**:
       The time index has gaps, meaning there are missing periods in the
       sequence. A continuous time index is required for valid transformation
       and estimation.

    2. **Post variable non-monotonicity**:
       The post-treatment indicator is not monotone non-decreasing in time,
       suggesting that the policy was reversed or suspended. The method assumes
       absorbing treatment states.

    Both scenarios violate the assumptions required for valid difference-in-
    differences estimation and will cause estimation to fail or produce
    incorrect results.

    See Also
    --------
    LWDIDError : Base exception class.
    """
    pass


class MissingRequiredColumnError(LWDIDError):
    """
    Exception raised when input DataFrame is missing required columns.

    Required columns depend on the function call but typically include:

    - y: Outcome variable
    - d: Treatment indicator
    - ivar: Unit identifier
    - tvar: Time variable (single column or list of [year, quarter])
    - post: Post-treatment indicator
    - controls: Control variables (if specified)

    See Also
    --------
    LWDIDError : Base exception class.
    """
    pass


class RandomizationError(LWDIDError):
    """
    Exception raised when randomization inference (RI) fails.

    Common causes include invalid number of replications (rireps <= 0),
    missing required columns in input data, sample size too small for
    resampling (N < 3), invalid ri_method specification, or insufficient
    valid draws for reliable inference.

    See Also
    --------
    LWDIDError : Base exception class.
    """
    pass


class VisualizationError(LWDIDError):
    """
    Exception raised for visualization-related errors.

    Common causes include plot data missing required columns (such as
    transformed outcome, treatment indicator, or time index) or missing
    plotting backend (matplotlib not installed).

    See Also
    --------
    LWDIDError : Base exception class.
    """
    pass


# =============================================================================
# Staggered DiD Exceptions
# =============================================================================

class InvalidStaggeredDataError(LWDIDError):
    """
    Exception raised when staggered data validation fails.

    This exception is raised in the following scenarios:

    1. **Invalid gvar values**:
       - Negative values in gvar column
       - String values instead of numeric types
       - Values that cannot be interpreted as treatment cohort or never-treated

    2. **No valid cohorts**:
       - All units are never-treated (no treated cohorts to estimate effects for)
       - All gvar values are NaN/0/inf with no positive integers

    3. **Inconsistent gvar within unit**:
       - Same unit has different gvar values across time periods
       (gvar should be time-invariant within unit)

    Valid gvar values:

    - Positive integer: Treatment cohort (first treatment period)
    - 0: Never treated
    - np.inf: Never treated
    - NaN/None: Never treated

    See Also
    --------
    LWDIDError : Base exception class.
    NoNeverTreatedError : Raised when never-treated units are required but absent.
    """
    pass


class NoNeverTreatedError(InsufficientDataError):
    """
    Exception raised when never-treated units are required but absent.

    This exception is raised when:

    - aggregate='cohort' is specified but no never-treated units exist
    - aggregate='overall' is specified but no never-treated units exist

    Never-treated units are required as control group for cohort and overall
    effect aggregation because different cohorts use different pre-treatment
    periods for transformation, and only never-treated units can serve as a
    consistent reference across cohorts.

    For (g,r)-specific effects, not-yet-treated units can serve as controls,
    so this exception is not raised when aggregate='none'.

    See Also
    --------
    InsufficientDataError : Parent class for data insufficiency errors.
    InvalidStaggeredDataError : Raised for other staggered data validation failures.
    """
    pass


# =============================================================================
# Repeated Cross-Section Aggregation Exceptions
# =============================================================================

class AggregationError(LWDIDError):
    """
    Base exception class for aggregation-related errors.

    This is the parent class for all exceptions related to repeated
    cross-sectional data aggregation. Specific subclasses indicate
    the exact nature of the aggregation failure.

    See Also
    --------
    InvalidAggregationError : Raised when aggregation constraints are violated.
    InsufficientCellSizeError : Raised when all cells are below minimum size.
    """
    pass


class InvalidAggregationError(AggregationError):
    """
    Exception raised when aggregation constraints are violated.

    This exception is raised in the following scenarios:

    1. **Treatment varies within cell**:
       Treatment status is not constant within a (unit, period) cell,
       violating the assumption that treatment is assigned at the
       aggregation unit level.

    2. **gvar varies within unit**:
       Treatment timing (gvar) is not constant within a unit across
       all periods, violating the time-invariance assumption.

    3. **Duplicate column names**:
       Aggregation would result in duplicate column names in the output.

    See Also
    --------
    AggregationError : Parent class for aggregation errors.
    """
    pass


class InsufficientCellSizeError(AggregationError):
    """
    Exception raised when all cells are below minimum size threshold.

    This exception is raised when the min_cell_size parameter is specified
    and all (unit, period) cells have fewer observations than the threshold,
    resulting in an empty output panel.

    See Also
    --------
    AggregationError : Parent class for aggregation errors.
    """
    pass
