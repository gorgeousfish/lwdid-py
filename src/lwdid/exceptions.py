"""
Exception Classes Module

Defines exception hierarchy for the lwdid package.

Authors: Xuanyu Cai, Wenli Xu
"""


class LWDIDError(Exception):
    """
    Base exception class for all lwdid package errors.

    All custom exceptions in the lwdid package inherit from this class,
    allowing users to catch any lwdid-specific error with:

        try:
            results = lwdid(...)
        except LWDIDError as e:
            # Handle any lwdid error
            print(f"lwdid error: {e}")
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

    The rolling parameter must be one of: 'demean', 'detrend', 'demeanq', 'detrendq'.
    This exception is raised during input validation if an unsupported method is specified.

    Examples
    --------
    >>> lwdid(data, ..., rolling='invalid_method')  # doctest: +SKIP
    InvalidRollingMethodError: rolling must be one of ['demean', 'detrend', 'demeanq', 'detrendq']

    See Also
    --------
    lwdid.validation._validate_rolling_parameter : Function that performs this validation.
    """
    pass


class InvalidVCETypeError(InvalidParameterError):
    """
    Exception raised when the vce parameter has an invalid value.

    The vce parameter must be one of: None, 'robust', 'hc1', 'hc3', 'cluster'.
    This exception is raised during estimation if an unsupported variance
    estimator type is specified.

    Examples
    --------
    >>> lwdid(data, ..., vce='invalid_vce')  # doctest: +SKIP
    InvalidVCETypeError: vce must be one of [None, 'robust', 'hc1', 'hc3', 'cluster']

    See Also
    --------
    lwdid.estimation.estimate_att : Function that validates vce parameter.
    """
    pass


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

    This occurs when all units have treatment indicator d=0. At least one
    treated unit (d=1) is required for difference-in-differences estimation.

    Trigger condition: N_treated = 0 (no units with d=1 over the panel or in
    the main regression sample).
    """
    pass


class NoControlUnitsError(InsufficientDataError):
    """
    Exception raised when there are no control units in the data.

    This occurs when all units have treatment indicator d=1. At least one
    control unit (d=0) is required for difference-in-differences estimation.

    Trigger condition: N_control = 0 (no units with d=0 over the panel or in
    the main regression sample).
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

    See Also
    --------
    lwdid.transformations.apply_rolling_transform : Applies rolling
        transformations and enforces pre-period requirements.
    """
    pass


class InsufficientQuarterDiversityError(InsufficientDataError):
    """
    Exception raised when quarterly data requirements are not met.

    In the current ``lwdid()`` workflow this exception is raised for
    quarterly methods (``demeanq``/``detrendq``) when **quarter coverage is
    insufficient**: post-treatment periods contain quarter(s) that do not appear
    in the pre-treatment period for a given unit, so seasonal effects for those
    quarters cannot be estimated from pre-treatment data.

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
       The time index (tindex) has gaps, meaning there are missing periods in
       the sequence.

       Example: If the data contains years 2000, 2001, and 2003 but is missing
       2002, this creates a gap that violates the continuity assumption.

    2. **Post variable non-monotonicity**:
       The post-treatment indicator is not monotone non-decreasing in time,
       suggesting that the policy was reversed or suspended.

       Example: If post=0 in period 1, post=1 in period 2, and post=0 again in
       period 3, this indicates policy reversal which is not supported by the method.

    Both scenarios violate the assumptions of the Lee and Wooldridge (2025) method
    and will cause estimation to fail or produce incorrect results.

    See Also
    --------
    lwdid.validation._validate_time_continuity : Function that performs these checks.
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

    Examples
    --------
    >>> lwdid(data, y='outcome', d='treated', ...)  # doctest: +SKIP
    MissingRequiredColumnError: Required column 'outcome' not found in data

    See Also
    --------
    lwdid.validation._validate_required_columns : Function that performs this check.
    """
    pass


class RandomizationError(LWDIDError):
    """
    Exception raised when randomization inference (RI) fails.

    Trigger conditions include:

    - rireps <= 0 (invalid number of replications)
    - firstpost_df missing required columns (``ydot_postavg``, ``d_`` (treatment indicator))
    - Sample too small (N < 3, insufficient for resampling)
    - Invalid ri_method
    - Number of valid RI replications too small relative to rireps
      (insufficient valid draws for reliable inference)

    Examples
    --------
    >>> lwdid(data, ..., ri=True, rireps=0)  # doctest: +SKIP
    RandomizationError: rireps must be positive, got 0

    See Also
    --------
    lwdid.randomization.randomization_inference : Function that performs RI.
    """
    pass


class VisualizationError(LWDIDError):
    """
    Exception raised for visualization-related errors.

    Trigger conditions include:

    - Plot data missing required columns (for example, ``ydot``, ``d_`` (treatment indicator), ``tindex``)
    - Missing plotting backend (matplotlib not installed)

    Examples
    --------
    >>> results.plot()  # in an environment without matplotlib installed  # doctest: +SKIP
    VisualizationError: Install required dependencies: matplotlib>=3.3.

    See Also
    --------
    lwdid.visualization.plot_results : Function that generates plots.
    lwdid.results.LWDIDResults.plot : Method that calls plot_results.
    """
    pass
