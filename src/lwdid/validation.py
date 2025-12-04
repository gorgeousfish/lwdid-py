"""
Validation Module

Implements input validation and data preparation for the Lee and Wooldridge (2025)
difference-in-differences estimator.

"""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .exceptions import (
    InsufficientDataError,
    InsufficientQuarterDiversityError,
    InvalidParameterError,
    InvalidRollingMethodError,
    MissingRequiredColumnError,
    NoControlUnitsError,
    NoTreatedUnitsError,
    TimeDiscontinuityError,
)


def validate_and_prepare_data(
    data: pd.DataFrame,
    y: str,
    d: str,
    ivar: str,
    tvar: Union[str, List[str]],
    post: str,
    rolling: str,
    controls: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Validate input data and execute data preparation pipeline (Steps 0-6).

    This is the main entry point for all data validation and preparation in the
    lwdid package. It performs comprehensive checks and transformations to ensure
    data integrity before transformation and estimation.

    Pipeline Steps
    --------------
    1. **Input validation**:
       - DataFrame type check
       - Reserved column names check
       - Required columns existence check
       - Rolling parameter validation

    2. **Data type validation**:
       - Outcome variable numeric type check
       - Control variables numeric type check

    3. **Time-invariance validation**:
       - Treatment indicator time-invariance check
       - Control variables time-invariance check

    4. **Data preparation**:
       - String ID conversion to numeric codes
       - Time index creation (tindex)
       - Binary treatment/post indicator creation (d_, post_)
       - Missing value handling

    5. **Time structure validation**:
       - Time continuity validation
       - Post-treatment monotonicity check

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel data with one row per unit-time observation.
    y : str
        Outcome variable column name. Must be numeric.
    d : str
        Treatment indicator column name. Must be time-invariant (constant within unit).
        d=1 for treated units, d=0 for control units.
    ivar : str
        Unit identifier column name. Can be string or numeric.
    tvar : str or list of str
        Time variable column name(s):

        - str: Annual data (e.g., 'year')
        - [str, str]: Quarterly data (e.g., ['year', 'quarter'])

    post : str
        Post-treatment indicator column name. Must be monotone non-decreasing in time.
        post=0 for pre-treatment periods, post=1 for post-treatment periods.
    rolling : str
        Transformation method. Must be one of:

        - 'demean': Unit-specific demeaning
        - 'detrend': Unit-specific detrending
        - 'demeanq': Quarterly demeaning with seasonal effects
        - 'detrendq': Quarterly detrending with seasonal effects

    controls : list of str, optional
        Control variable column names. Must be numeric and time-invariant.
        Default: None (no controls).

    Returns
    -------
    data_clean : pd.DataFrame
        Cleaned and prepared data with the following modifications:

        - Original columns preserved
        - New columns added: tindex, d_, post_ (and tq for quarterly data)
        - String IDs converted to numeric codes (if applicable)
        - Missing values handled (rows with NaN in y, d, post, ivar, or time
          variables are dropped; missing values in control variables are
          handled later at the estimation stage)

    metadata : dict
        Metadata dictionary containing:

        - 'N': Total number of units
        - 'N_treated': Number of treated units (d_=1)
        - 'N_control': Number of control units (d_=0)
        - 'T': Total number of time periods
        - 'K': Number of pre-treatment periods
        - 'tpost1': First post-treatment period index
        - 'is_quarterly': Boolean indicating quarterly data
        - 'id_mapping': Dict mapping original string IDs to numeric codes (if applicable)

    Raises
    ------
    TypeError
        If data is not a pandas DataFrame.
    MissingRequiredColumnError
        If required columns are missing from data.
    InvalidParameterError
        If reserved column names exist in data.
    InvalidRollingMethodError
        If rolling parameter is not one of the four valid methods.
    InsufficientDataError
        If sample size is insufficient (no treated/control units).
    TimeDiscontinuityError
        If time series has gaps or post variable is non-monotone.
    InsufficientQuarterDiversityError
        If quarterly helper checks for demeanq/detrendq detect invalid quarter
        patterns (raised indirectly via quarterly validation utilities).

    Notes
    -----
    Reserved column names that cannot exist in input data:

    - 'd_': Binary treatment indicator (created internally)
    - 'post_': Binary post indicator (created internally)
    - 'tindex': Time index (created internally)
    - 'tq': Quarter index (created internally for quarterly data)
    - 'ydot': Residualized outcome (created in transformation)
    - 'ydot_postavg': Post-period average of ydot (created in transformation)
    - 'firstpost': Main regression sample indicator (created in transformation)

    Examples
    --------
    >>> data_clean, metadata = validate_and_prepare_data(
    ...     data=df,
    ...     y='outcome',
    ...     d='treated',
    ...     ivar='state',
    ...     tvar='year',
    ...     post='post',
    ...     rolling='demean'
    ... )
    >>> print(f"N={metadata['N']}, T={metadata['T']}, K={metadata['K']}")
    N=50, T=20, K=10

    See Also
    --------
    _validate_required_columns : Validates column existence.
    _validate_time_continuity : Validates time series continuity.
    validate_quarter_coverage : Validates quarter coverage for quarterly methods.
    """
    # DataFrame type check
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"Input data must be a pandas DataFrame. Got: {type(data).__name__}"
        )

    # Check for conflicts with internal column names
    _validate_no_reserved_columns(data)

    # Required columns validation
    _validate_required_columns(data, y, d, ivar, tvar, post, controls)

    # Outcome variable data type validation
    _validate_outcome_dtype(data, y)

    # Control variables data type validation
    _validate_controls_dtype(data, controls)

    # Treatment indicator time-invariance validation
    _validate_treatment_time_invariance(data, d, ivar)

    # Control variables time-invariance validation
    _validate_time_invariant_controls(data, ivar, controls)

    # Rolling parameter validation (case-insensitive)
    rolling = _validate_rolling_parameter(rolling, tvar)
    
    data_work, id_mapping = _convert_string_id(data, ivar)
    
    # Binarize treatment and post indicators
    d_numeric = pd.to_numeric(data_work[d], errors='coerce')
    post_numeric = pd.to_numeric(data_work[post], errors='coerce')
    data_work['d_'] = (d_numeric != 0).where(d_numeric.notna()).astype('Int64')
    data_work['post_'] = (post_numeric != 0).where(post_numeric.notna()).astype('Int64')
    
    # Drop observations with missing values in required variables
    required_vars = [y, ivar, 'post_', 'd_']

    if isinstance(tvar, str):
        required_vars.append(tvar)
    else:
        required_vars.extend(tvar)

    # Controls handled at estimation stage (after N₁, N₀ known)
    n_before = len(data_work)
    data_work = data_work.dropna(subset=required_vars, how='any').copy()
    n_after = len(data_work)

    if n_after < n_before:
        n_dropped = n_before - n_after
        var_display = []
        for var in required_vars:
            if var == 'd_':
                var_display.append(f"{d} (treatment indicator)")
            elif var == 'post_':
                var_display.append(f"{post} (post indicator)")
            else:
                var_display.append(var)
        vars_str = ', '.join(var_display)

        warnings.warn(
            f"Dropped {n_dropped} observations due to missing values in required variables: {vars_str}",
            UserWarning,
            stacklevel=3
        )
    
    data_work, is_quarterly = _create_time_index(data_work, tvar)

    if is_quarterly:
        year_var, quarter_var = tvar[0], tvar[1]
        dup_mask = data_work.duplicated([ivar, year_var, quarter_var], keep=False)
        if dup_mask.any():
            n_dup = dup_mask.sum()
            dup_examples = data_work[dup_mask][[ivar, year_var, quarter_var]].drop_duplicates().head(5)
            raise InvalidParameterError(
                f"Duplicate (ivar, year, quarter) observations found. "
                f"Each (unit, year, quarter) combination must be unique. "
                f"Found {n_dup} duplicate rows. "
                f"Examples of duplicated combinations:\n{dup_examples.to_string(index=False)}\n"
                f"Please check your data and remove duplicate observations."
                )
    else:
        dup_mask = data_work.duplicated([ivar, 'tindex'], keep=False)
        if dup_mask.any():
            n_dup = dup_mask.sum()
            if isinstance(tvar, str):
                dup_examples = data_work[dup_mask][[ivar, tvar, 'tindex']].drop_duplicates().head(5)
                raise InvalidParameterError(
                    f"Duplicate (ivar, tvar) observations found. "
                    f"Each (unit, time) combination must be unique. "
                    f"Found {n_dup} duplicate rows. "
                    f"Examples of duplicated combinations:\n{dup_examples.to_string(index=False)}\n"
                    f"Please check your data and remove duplicate observations."
                )

    K, tpost1, T = _validate_time_continuity(data_work)
    
    unit_treatment = data_work.groupby(ivar)['d_'].max()
    N = len(unit_treatment)
    N_treated = (unit_treatment == 1).sum()
    N_control = (unit_treatment == 0).sum()
    
    if N < 3:
        raise InsufficientDataError(
            f"Insufficient sample size: N={N} (need N >= 3)."
        )
    
    if N_treated < 1:
        raise NoTreatedUnitsError(
            f"No treated units found (d==1 in any period). Check treatment variable '{d}'."
        )
    
    if N_control < 1:
        raise NoControlUnitsError(
            f"No control units found (d==0). Check treatment variable '{d}'."
        )

    if N_treated == 1:
        warnings.warn(
            f"Only 1 treated unit found (N_treated=1). "
            f"Estimation with a single treated unit is technically feasible but highly unstable. "
            f"Results are extremely sensitive to this single unit and may not be reliable. "
            f"Consider: (1) checking treatment variable '{d}' for coding errors, "
            f"(2) verifying sample selection, (3) using alternative methods (e.g., synthetic control).",
            UserWarning,
            stacklevel=2
        )

    if N_control == 1:
        warnings.warn(
            f"Only 1 control unit found (N_control=1). "
            f"Estimation with a single control unit is technically feasible but highly unstable. "
            f"Results are extremely sensitive to this single unit and may not be reliable. "
            f"Consider: (1) checking treatment variable '{d}' for coding errors, "
            f"(2) verifying sample selection, (3) expanding the control group if possible.",
            UserWarning,
            stacklevel=2
        )

    metadata = {
        'N': N,
        'N_treated': N_treated,
        'N_control': N_control,
        'K': K,
        'tpost1': tpost1,
        'T': T,
        'id_mapping': id_mapping,
        'is_quarterly': is_quarterly,
        'depvar': y,
        'ivar': ivar,
        'tvar': tvar,
        'post': post,
        'd': d,
        'rolling': rolling,
    }
    
    return data_work, metadata


def _validate_no_reserved_columns(data: pd.DataFrame) -> None:
    """
    Check that input data does not contain reserved column names.

    lwdid uses several internal column names that should not exist in user data.
    If these columns exist, they will be silently overwritten, causing data loss
    and potentially incorrect results.

    Parameters
    ----------
    data : pd.DataFrame
        Input data to check for reserved column names.

    Raises
    ------
    InvalidParameterError
        If any reserved column names are found in the input data.

    Notes
    -----
    Reserved columns (created internally by lwdid):

    - ``d_`` (treatment indicator): Binary treatment indicator (0/1)
    - ``post_`` (post-period indicator): Binary post-period indicator (0/1)
    - ``tindex``: Time index (1, 2, 3, ...)
    - ``tq``: Quarter index (for quarterly data)
    - ``ydot``: Residualized outcome variable
    - ``ydot_postavg``: Post-period average of ydot
    - ``firstpost``: Main regression sample indicator

    Examples
    --------
    >>> data = pd.DataFrame({'y': [1, 2], 'd_': [0, 1]})  # doctest: +SKIP
    >>> _validate_no_reserved_columns(data)
    InvalidParameterError: Input data contains reserved column names: ['d_']

    See Also
    --------
    validate_and_prepare_data : Main validation function that calls this check.
    """
    RESERVED_COLUMNS = ['d_', 'post_', 'tindex', 'tq', 'ydot', 'ydot_postavg', 'firstpost']
    existing_reserved = [col for col in RESERVED_COLUMNS if col in data.columns]

    if existing_reserved:
        raise InvalidParameterError(
            f"Input data contains reserved column names: {existing_reserved}.\n"
            f"These columns are used internally by lwdid and will be overwritten.\n"
            f"Please rename these columns in your data before calling lwdid().\n"
            f"Reserved column names: {RESERVED_COLUMNS}"
        )


def _validate_required_columns(
    data: pd.DataFrame,
    y: str,
    d: str,
    ivar: str,
    tvar: Union[str, List[str]],
    post: str,
    controls: Optional[List[str]],
) -> None:
    """
    Validate existence of required columns
    """
    # Core variables
    required_cols = [y, d, ivar, post]

    # Time variables
    if isinstance(tvar, str):
        required_cols.append(tvar)
    else:
        required_cols.extend(tvar)

    # Control variables
    if controls:
        required_cols.extend(controls)

    # Check for missing columns
    missing_cols = [col for col in required_cols if col not in data.columns]

    if missing_cols:
        raise MissingRequiredColumnError(
            f"Required column(s) not found in data: {missing_cols}. "
            f"Available columns: {list(data.columns)}"
        )


def _validate_outcome_dtype(
    data: pd.DataFrame,
    y: str,
) -> None:
    """
    Validate outcome variable is numeric

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    y : str
        Outcome variable column name

    Raises
    ------
    InvalidParameterError
        If outcome variable is not numeric
    """
    dtype = data[y].dtype

    if not pd.api.types.is_numeric_dtype(dtype):
        raise InvalidParameterError(
            f"Outcome variable '{y}' must be numeric type. "
            f"Found dtype: '{dtype}'\n\n"
            f"Why this matters:\n"
            f"  - Outcome variable needs arithmetic operations (mean, subtraction)\n"
            f"  - Non-numeric types cannot be used in regression\n"
            f"  - String or categorical variables must be converted first\n\n"
            f"How to fix:\n"
            f"  1. If '{y}' should be numeric, convert it:\n"
            f"     data['{y}'] = pd.to_numeric(data['{y}'], errors='coerce')\n"
            f"  2. If '{y}' is categorical, you may need to:\n"
            f"     - Use a different outcome variable\n"
            f"     - Convert categories to numeric codes\n"
            f"  3. Check for data entry errors (e.g., text in numeric column)\n\n"
            f"Example of valid outcome: 10.5, 20, 30.7, ...\n"
            f"Example of invalid outcome: 'high', 'low', True, False"
        )

    if dtype == 'bool':
        warnings.warn(
            f"Outcome variable '{y}' has boolean type (True/False). "
            f"This will be treated as numeric (1/0). "
            f"If this is not your intent, please convert '{y}' to a proper numeric variable.",
            category=UserWarning,
            stacklevel=4
        )


def _validate_controls_dtype(
    data: pd.DataFrame,
    controls: Optional[List[str]],
) -> None:
    """
    Validate control variables are numeric

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    controls : Optional[List[str]]
        Control variable names

    Raises
    ------
    InvalidParameterError
        If any control is not numeric
    """
    if not controls:
        return

    non_numeric_controls = []
    for control in controls:
        dtype = data[control].dtype
        if not pd.api.types.is_numeric_dtype(dtype):
            non_numeric_controls.append((control, str(dtype)))

    if non_numeric_controls:
        error_details = "\n".join([
            f"  - '{col}' has dtype '{dtype}'"
            for col, dtype in non_numeric_controls
        ])
        raise InvalidParameterError(
            f"Control variables must be numeric type. Found non-numeric controls:\n"
            f"{error_details}\n\n"
            f"Suggestion: Convert categorical/string variables to numeric using:\n"
            f"  - One-hot encoding: pd.get_dummies(data, columns=[...])\n"
            f"  - Label encoding: data[col] = pd.factorize(data[col])[0]\n"
            f"  - Manual mapping: data[col] = data[col].map({{'A': 0, 'B': 1, ...}})"
        )


def _validate_treatment_time_invariance(
    data: pd.DataFrame,
    d: str,
    ivar: str,
) -> None:
    """
    Validate treatment indicator is time-invariant (D_i, not W_it)

    The method requires unit-level treatment status D_i (constant within unit),
    not time-varying W_it = D_i·post_t. See equations (2.1)-(2.2).

    Parameters
    ----------
    data : pd.DataFrame
        Panel data
    d : str
        Treatment indicator
    ivar : str
        Unit identifier

    Raises
    ------
    InvalidParameterError
        If treatment varies within any unit
    """
    within_unit_std = data.groupby(ivar)[d].std()
    max_std = within_unit_std.max()

    if pd.notna(max_std) and max_std > 1e-10:
        n_varying_units = (within_unit_std > 1e-10).sum()

        varying_units = within_unit_std[within_unit_std > 1e-10].head(3)
        example_details = "\n".join([
            f"    Unit {unit_id}: std = {std:.6f}"
            for unit_id, std in varying_units.items()
        ])

        raise InvalidParameterError(
            f"Treatment indicator '{d}' must be time-invariant (constant within each unit).\n"
            f"The paper (§2.1, eq. 2.1) requires unit-level D_i, not time-varying W_it.\n\n"
            f"Found {n_varying_units} units with time-varying treatment status:\n"
            f"{example_details}\n"
            f"{'  ...' if n_varying_units > 3 else ''}\n\n"
            f"Why this matters:\n"
            f"  - The paper defines D_i as unit-level treatment status (0 or 1, time-invariant)\n"
            f"  - W_it = D_i · post_t is the time-varying indicator (derived from D_i)\n"
            f"  - You should pass D_i as 'd', NOT W_it\n"
            f"  - Time-varying d violates the identification assumptions\n"
            f"  - Results cannot be interpreted under the paper's framework\n\n"
            f"How to fix:\n"
            f"  1. If you have D_i (unit-level): use it directly as 'd' parameter\n"
            f"  2. If you have W_it (time-varying): create D_i first:\n"
            f"     data['D_i'] = data.groupby('{ivar}')['{d}'].transform('max')\n"
            f"     Then use 'D_i' as the 'd' parameter\n"
            f"  3. Verify: Each unit should have the same d value in all periods\n\n"
            f"Example:\n"
            f"  Correct:   id=1 has d=1 in all periods (treated unit)\n"
            f"  Correct:   id=2 has d=0 in all periods (control unit)\n"
            f"  Incorrect: id=1 has d=0 in pre, d=1 in post (this is W_it, not D_i!)"
        )


def _validate_time_invariant_controls(
    data: pd.DataFrame,
    ivar: str,
    controls: Optional[List[str]],
) -> None:
    """
    Validate control variables are time-invariant (X_i, not X_it)

    Method requires time-constant controls X_i as in Section 2.2, equation (2.18).
    Centering uses X_i - X̄₁ with unit-level means.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data
    ivar : str
        Unit identifier
    controls : Optional[List[str]]
        Control variable names

    Raises
    ------
    InvalidParameterError
        If any control varies within units
    """
    if not controls:
        return

    time_varying_controls = []

    for control in controls:
        within_unit_std = data.groupby(ivar)[control].std()
        max_std = within_unit_std.max()

        if pd.notna(max_std) and max_std > 1e-10:
            n_varying_units = (within_unit_std > 1e-10).sum()
            time_varying_controls.append((control, max_std, n_varying_units))

    if time_varying_controls:
        error_details = "\n".join([
            f"  - '{col}': max within-unit std = {std:.6f}, "
            f"{n_units} units have time-varying values"
            for col, std, n_units in time_varying_controls
        ])

        raise InvalidParameterError(
            f"Control variables must be time-invariant (constant within each unit).\n"
            f"The paper (§2.2) requires 'time-constant controls X_i', not X_it.\n\n"
            f"Found time-varying controls:\n"
            f"{error_details}\n\n"
            f"Why this matters:\n"
            f"  - Time-varying controls violate the paper's theoretical assumptions\n"
            f"  - They can cause substantial estimation bias in ATT\n"
            f"  - The method uses unit-level X_i, not period-specific X_it\n\n"
            f"How to fix:\n"
            f"  1. Use first period value: data.groupby('{ivar}')[control].transform('first')\n"
            f"  2. Use unit mean: data.groupby('{ivar}')[control].transform('mean')\n"
            f"  3. Use pre-treatment value: data[data['post']==0].groupby('{ivar}')[control].first()\n"
            f"  4. Use domain-appropriate aggregation method\n\n"
            f"After aggregation, ensure each unit has a constant value across all periods."
        )


def _validate_rolling_parameter(rolling: str, tvar: Union[str, List[str]]) -> str:
    """
    Validate rolling parameter

    Parameters
    ----------
    rolling : str
        Transformation method (case-insensitive)
    tvar : Union[str, List[str]]
        Time variable

    Returns
    -------
    str
        Standardized rolling parameter (lowercase)
    """
    rolling_lower = rolling.lower()

    valid_methods = {'demean', 'detrend', 'demeanq', 'detrendq'}

    if rolling_lower not in valid_methods:
        raise InvalidRollingMethodError(
            f"rolling() must be one of: demean, detrend, demeanq, detrendq. "
            f"Got: '{rolling}'"
        )

    if rolling_lower in ['demeanq', 'detrendq']:
        if isinstance(tvar, str) or not hasattr(tvar, '__len__') or len(tvar) != 2:
            raise InvalidRollingMethodError(
                f"rolling('{rolling_lower}') requires tvar(year quarter). "
                f"Got: {tvar}"
            )

    return rolling_lower


def _convert_string_id(
    data: pd.DataFrame, ivar: str
) -> Tuple[pd.DataFrame, Optional[Dict]]:
    """
    Convert string unit IDs to numeric codes

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    ivar : str
        Unit identifier

    Returns
    -------
    data_copy : pd.DataFrame
        Copy with numeric IDs
    id_mapping : dict or None
        Bidirectional mapping if conversion occurred
    """
    data_work = data.copy()
    id_mapping = None

    if data_work[ivar].dtype == 'object' or pd.api.types.is_string_dtype(data_work[ivar]):
        codes, uniques = pd.factorize(data_work[ivar])
        codes_series = pd.Series(codes, index=data_work.index)
        codes_series = codes_series.replace(-1, pd.NA)
        data_work[ivar] = (codes_series + 1).astype('Int64')

        id_mapping = {
            'original_to_numeric': {orig: num + 1 for num, orig in enumerate(uniques)},
            'numeric_to_original': {num + 1: orig for num, orig in enumerate(uniques)},
        }

    return data_work, id_mapping


def _create_time_index(
    data: pd.DataFrame, tvar: Union[str, List[str]]
) -> Tuple[pd.DataFrame, bool]:
    """
    Create time index (tindex column)
    
    Returns
    -------
    data : pd.DataFrame
        Data with 'tindex' column added.
    is_quarterly : bool
        True if quarterly data, False if annual.
    """
    if isinstance(tvar, str):
        year_var = tvar
        year_numeric = pd.to_numeric(data[year_var], errors='coerce')

        if year_numeric.isna().any():
            invalid_mask = year_numeric.isna()
            invalid_values = data.loc[invalid_mask, year_var].unique()
            raise InvalidParameterError(
                f"Year variable '{year_var}' contains non-numeric values that cannot be converted to numbers.\n"
                f"Found invalid values: {list(invalid_values)}\n\n"
                f"Why this matters:\n"
                f"  - Time index calculation requires numeric year values\n"
                f"  - Non-numeric values (e.g., 'NA', 'missing', text) cannot be used in arithmetic operations\n"
                f"  - This would cause {invalid_mask.sum()} observations to be dropped silently\n\n"
                f"How to fix:\n"
                f"  1. Check your data for non-numeric year values\n"
                f"  2. Remove or fix rows with invalid year values\n"
                f"  3. Ensure all year values are numeric (e.g., 2000, 2001, ...)\n\n"
                f"Example of valid year values: 2000, 2001, 2002, ...\n"
                f"Example of invalid year values: 'NA', 'missing', '', None"
            )

        data['tindex'] = year_numeric - year_numeric.min() + 1
        is_quarterly = False
    else:
        year_var, quarter_var = tvar[0], tvar[1]

        year_numeric = pd.to_numeric(data[year_var], errors='coerce')
        quarter_numeric = pd.to_numeric(data[quarter_var], errors='coerce')

        if year_numeric.isna().any():
            invalid_mask = year_numeric.isna()
            invalid_values = data.loc[invalid_mask, year_var].unique()
            raise InvalidParameterError(
                f"Year variable '{year_var}' contains non-numeric values that cannot be converted to numbers.\n"
                f"Found invalid values: {list(invalid_values)}\n\n"
                f"Why this matters:\n"
                f"  - Quarterly time index calculation requires numeric year values\n"
                f"  - Non-numeric values (e.g., 'NA', 'missing', text) cannot be used in arithmetic operations\n"
                f"  - This would cause {invalid_mask.sum()} observations to be dropped silently\n\n"
                f"How to fix:\n"
                f"  1. Check your data for non-numeric year values\n"
                f"  2. Remove or fix rows with invalid year values\n"
                f"  3. Ensure all year values are numeric (e.g., 2000, 2001, ...)\n\n"
                f"Example of valid year values: 2000, 2001, 2002, ...\n"
                f"Example of invalid year values: 'NA', 'missing', '', None"
            )

        if quarter_numeric.isna().any():
            invalid_mask = quarter_numeric.isna()
            invalid_values = data.loc[invalid_mask, quarter_var].unique()
            raise InvalidParameterError(
                f"Quarter variable '{quarter_var}' contains non-numeric values that cannot be converted to numbers.\n"
                f"Found invalid values: {list(invalid_values)}\n\n"
                f"Why this matters:\n"
                f"  - Quarterly time index calculation requires numeric quarter values\n"
                f"  - Non-numeric values (e.g., 'NA', 'Q1', text) cannot be used in arithmetic operations\n"
                f"  - This would cause {invalid_mask.sum()} observations to be dropped silently\n\n"
                f"How to fix:\n"
                f"  1. Check your data for non-numeric quarter values\n"
                f"  2. Remove or fix rows with invalid quarter values\n"
                f"  3. Ensure all quarter values are numeric (1, 2, 3, or 4)\n\n"
                f"Example of valid quarter values: 1, 2, 3, 4\n"
                f"Example of invalid quarter values: 'Q1', 'NA', 'missing', '', None"
            )

        quarter_vals = quarter_numeric.unique()
        quarter_vals = quarter_vals[~pd.isna(quarter_vals)]
        if not all(q in [1, 2, 3, 4] for q in quarter_vals):
            raise InvalidParameterError(
                f"Quarter variable '{quarter_var}' contains invalid values. "
                f"Must be in {{1, 2, 3, 4}}. Found: {sorted(quarter_vals)}. "
                f"Please check your data and ensure quarter values are in the valid range."
            )

        data['tq'] = (year_numeric - 1960) * 4 + quarter_numeric
        data['tindex'] = data['tq'] - data['tq'].min() + 1
        is_quarterly = True
    
    return data, is_quarterly


def _validate_time_continuity(data: pd.DataFrame) -> Tuple[int, int, int]:
    """
    Validate time continuity and extract time structure

    Returns
    -------
    K : int
        Last pre-treatment period
    tpost1 : int
        First post-treatment period
    T : int
        Total number of periods

    Raises
    ------
    InsufficientDataError
        No pre or post observations
    TimeDiscontinuityError
        Time series has gaps
    InvalidParameterError
        Common timing violated
    """
    pre_period = data[data['post_'] == 0]
    post_period = data[data['post_'] == 1]

    if len(pre_period) == 0:
        raise InsufficientDataError(
            "No pre-treatment observations (post==0)."
        )

    if len(post_period) == 0:
        raise InsufficientDataError(
            "No post-treatment observations (post==1)."
        )

    K = int(data[data['post_'] == 0]['tindex'].max())
    tpost1 = int(data[data['post_'] == 1]['tindex'].min())
    T = int(data['tindex'].max())

    # Verify common timing: post must be constant across units at each time
    post_by_time = data.groupby('tindex')['post_'].nunique()
    if (post_by_time > 1).any():
        violating_times = post_by_time[post_by_time > 1].index.tolist()
        raise InvalidParameterError(
            f"Common timing assumption violated: 'post' must be a pure function of time. "
            f"Found time periods where 'post' varies across units: tindex={violating_times}. "
            f"Current implementation requires all units to switch from pre to post at the same time. "
            f"For staggered adoption scenarios, please refer to the extended methods in "
            f"Lee and Wooldridge (2025, Section 7)."
        )

    # Validate tindex continuity (gaps distort trend estimation)
    unique_tindex = sorted(data['tindex'].unique())
    expected_tindex = list(range(1, T + 1))

    if unique_tindex != expected_tindex:
        missing_periods = sorted(set(expected_tindex) - set(unique_tindex))
        raise TimeDiscontinuityError(
            f"Time series is discontinuous: tindex has gaps. "
            f"Expected continuous sequence: {expected_tindex}, "
            f"but found: {unique_tindex}. "
            f"Missing time periods: {missing_periods}. "
            f"\n\nThis is problematic because:"
            f"\n  - detrend/detrendq methods use tindex for linear regression, "
            f"and gaps distort trend estimation"
            f"\n  - K={K} and T={T} would represent 'max index' rather than 'period count'"
            f"\n\nPlease ensure your time variable (year or year+quarter) forms a "
            f"continuous sequence without gaps."
        )

    # Validate post monotonicity (persistent treatment, eq 2.2)
    post_by_time = data.groupby('tindex')['post_'].first().sort_index()

    if (post_by_time == 1).any():
        first_post_t = post_by_time[post_by_time == 1].index[0]

        periods_after_first_post = post_by_time[post_by_time.index > first_post_t]

        if (periods_after_first_post == 0).any():
            reversal_times = periods_after_first_post[periods_after_first_post == 0].index.tolist()
            raise TimeDiscontinuityError(
                f"Post variable is not monotone in time: policy appears to be reversed or suspended. "
                f"Found post=0 at time periods {reversal_times} after first post=1 at tindex={first_post_t}. "
                f"\n\nThis violates the paper's core assumption (§2.1, eq. 2.2): "
                f"post_t = 1 if t ∈ {{S, S+1,...,T}} and zero otherwise. "
                f"This definition implies 'once treated, always treated' (monotonicity). "
                f"\n\nPolicy reversals break the identification strategy because:"
                f"\n  - K (last pre-period) and tpost1 (first post-period) become contradictory"
                f"\n  - The pre/post dichotomy used in ydot/yddot transformations loses meaning"
                f"\n  - The parallel trends assumption cannot be tested or maintained"
                f"\n\nPlease ensure your 'post' variable is monotone non-decreasing in time. "
                f"If your intervention was truly reversed, this method is not applicable."
            )

    return K, tpost1, T


def validate_quarter_diversity(
    data: pd.DataFrame,
    ivar: str,
    quarter: str,
    post: str
) -> None:
    """
    Validate quarter diversity and coverage for seasonal effects identification

    Requires ≥2 distinct quarters per unit in pre-period and that all post-period
    quarters appear in pre-period.

    Parameters
    ----------
    data : pandas.DataFrame
        Panel data
    ivar : str
        Unit identifier
    quarter : str
        Quarter variable
    post : str
        Post-treatment indicator

    Raises
    ------
    InsufficientQuarterDiversityError
        Quarter diversity or coverage violated
    """
    for unit_id in data[ivar].unique():
        unit_pre_mask = (data[ivar] == unit_id) & (data[post] == 0)
        unit_post_mask = (data[ivar] == unit_id) & (data[post] == 1)
        unit_pre_data = data[unit_pre_mask]
        unit_post_data = data[unit_post_mask]

        unique_quarters = unit_pre_data[quarter].nunique()

        if unique_quarters < 2:
            found_quarters = sorted(unit_pre_data[quarter].unique())
            raise InsufficientQuarterDiversityError(
                f"Unit {unit_id} has only {unique_quarters} quarter(s) in pre-period. "
                f"demeanq/detrendq requires ≥2 different quarters per unit to identify seasonal effects. "
                f"Found quarters: {found_quarters}"
            )

        pre_quarters = set(unit_pre_data[quarter].unique())
        post_quarters = set(unit_post_data[quarter].unique())

        uncovered_quarters = post_quarters - pre_quarters

        if uncovered_quarters:
            raise InsufficientQuarterDiversityError(
                f"Unit {unit_id}: Post-treatment period contains quarter(s) {sorted(uncovered_quarters)} "
                f"that do not appear in the pre-treatment period. "
                f"demeanq/detrendq cannot estimate seasonal effects for quarters not observed in pre-period. "
                f"Pre-period quarters: {sorted(pre_quarters)}, Post-period quarters: {sorted(post_quarters)}. "
                f"Please ensure each unit's pre-treatment period covers all quarters that appear in post-treatment, "
                f"or use demean/detrend methods instead."
            )


def validate_quarter_coverage(
    data: pd.DataFrame,
    ivar: str,
    quarter: str,
    post: str
) -> None:
    """
    Validate post-period quarters appear in pre-period for each unit

    Quarterly methods assume seasonal effects are constant over time. Each
    post-period quarter must appear in pre-period to identify its coefficient.

    Parameters
    ----------
    data : pandas.DataFrame
        Panel data
    ivar : str
        Unit identifier
    quarter : str
        Quarter variable
    post : str
        Post-treatment indicator

    Raises
    ------
    InsufficientQuarterDiversityError
        Post-period quarter not in pre-period
    """
    for unit_id in data[ivar].unique():
        unit_pre_mask = (data[ivar] == unit_id) & (data[post] == 0)
        unit_post_mask = (data[ivar] == unit_id) & (data[post] == 1)
        unit_pre_data = data[unit_pre_mask]
        unit_post_data = data[unit_post_mask]

        pre_quarters = set(unit_pre_data[quarter].unique())
        post_quarters = set(unit_post_data[quarter].unique())

        uncovered_quarters = post_quarters - pre_quarters

        if uncovered_quarters:
            raise InsufficientQuarterDiversityError(
                f"Unit {unit_id}: Post-treatment period contains quarter(s) {sorted(uncovered_quarters)} "
                f"that do not appear in the pre-treatment period. "
                f"demeanq/detrendq cannot estimate seasonal effects for quarters not observed in pre-period. "
                f"Pre-period quarters: {sorted(pre_quarters)}, Post-period quarters: {sorted(post_quarters)}. "
                f"Please ensure each unit's pre-treatment period covers all quarters that appear in post-treatment, "
                f"or use demean/detrend methods instead."
            )


# =============================================================================
# Staggered DiD Validation Functions
# =============================================================================

def is_never_treated(gvar_value: Union[int, float]) -> bool:
    """
    Determine if a unit is never treated based on its gvar value.
    
    This is the single source of truth for never-treated status identification.
    All modules (validation, control_groups, aggregation) should use this function
    to ensure consistent NT determination.
    
    Parameters
    ----------
    gvar_value : int or float
        The gvar (first treatment period) value for a unit.
        
    Returns
    -------
    bool
        True if the unit is never treated, False otherwise.
        
    Notes
    -----
    A unit is considered never treated if its gvar value is:
    - NaN or None (missing value)
    - 0 (explicitly coded as never treated)
    - np.inf (infinity, explicitly coded as never treated)
    
    Positive integers indicate the first treatment period (cohort membership).
    Negative values are invalid and should be caught by validate_staggered_data().
    
    Examples
    --------
    >>> is_never_treated(0)
    True
    >>> is_never_treated(np.nan)
    True
    >>> is_never_treated(np.inf)
    True
    >>> is_never_treated(2005)
    False
    """
    if pd.isna(gvar_value):
        return True
    if gvar_value == 0:
        return True
    if np.isinf(gvar_value):
        return True
    return False


def validate_staggered_data(
    data: pd.DataFrame,
    gvar: str,
    ivar: str,
    tvar: Union[str, List[str]],
    y: str,
    controls: Optional[List[str]] = None,
) -> Dict:
    """
    Validate staggered DiD data and extract cohort structure.
    
    This function performs comprehensive validation for staggered settings,
    checking gvar column validity, cohort identification, and data integrity.
    
    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format.
    gvar : str
        Column name for first treatment period (cohort indicator).
        Valid values: positive integers (cohort), 0/inf/NaN (never treated).
    ivar : str
        Unit identifier column name.
    tvar : str or list of str
        Time variable column name(s).
    y : str
        Outcome variable column name.
    controls : list of str, optional
        Control variable column names.
        
    Returns
    -------
    dict
        Validation result dictionary containing:
        
        - 'cohorts': List[int], sorted list of treatment cohorts (excludes NT)
        - 'n_never_treated': int, number of never-treated units
        - 'n_treated': int, total number of treated units across all cohorts
        - 'cohort_sizes': Dict[int, int], {cohort: n_units} mapping
        - 'T_min': int, minimum time period in data
        - 'T_max': int, maximum time period in data
        - 'warnings': List[str], warning messages (e.g., gvar out of tvar range)
        
    Raises
    ------
    MissingRequiredColumnError
        If required columns (gvar, ivar, tvar, y) are missing from data.
    InvalidStaggeredDataError
        If gvar column contains invalid values (negative numbers, strings)
        or if there are no valid treatment cohorts.
        
    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'id': [1, 1, 2, 2, 3, 3],
    ...     'year': [2000, 2001, 2000, 2001, 2000, 2001],
    ...     'y': [1.0, 2.0, 1.5, 2.5, 1.2, 2.2],
    ...     'gvar': [2001, 2001, 0, 0, 0, 0]
    ... })
    >>> result = validate_staggered_data(data, 'gvar', 'id', 'year', 'y')
    >>> result['cohorts']
    [2001]
    >>> result['n_never_treated']
    2
    """
    from .exceptions import InvalidStaggeredDataError, MissingRequiredColumnError
    
    # Check DataFrame type
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be a pandas DataFrame, got {type(data).__name__}")
    
    if len(data) == 0:
        raise InvalidStaggeredDataError("Input data is empty")
    
    # Check required columns exist
    required_cols = {gvar: 'gvar', ivar: 'ivar', y: 'y'}
    
    # Handle tvar (can be string or list)
    if isinstance(tvar, str):
        required_cols[tvar] = 'tvar'
        tvar_cols = [tvar]
    else:
        for t in tvar:
            required_cols[t] = 'tvar'
        tvar_cols = list(tvar)
    
    missing_cols = [col for col in required_cols.keys() if col not in data.columns]
    if missing_cols:
        raise MissingRequiredColumnError(
            f"Required columns not found in data: {missing_cols}. "
            f"Available columns: {list(data.columns)}"
        )
    
    # Check controls if specified
    if controls:
        missing_controls = [c for c in controls if c not in data.columns]
        if missing_controls:
            raise MissingRequiredColumnError(
                f"Control variable columns not found: {missing_controls}"
            )
    
    # Validate gvar column data type
    if not pd.api.types.is_numeric_dtype(data[gvar]):
        raise InvalidStaggeredDataError(
            f"gvar column '{gvar}' must be numeric, got {data[gvar].dtype}. "
            f"Please convert string values to numeric (e.g., 'never' -> 0, '2005' -> 2005)."
        )
    
    # Get unit-level gvar (should be time-invariant)
    unit_gvar = data.groupby(ivar)[gvar].first()
    
    # Check gvar is time-invariant within each unit
    gvar_nunique = data.groupby(ivar)[gvar].nunique()
    inconsistent_units = gvar_nunique[gvar_nunique > 1].index.tolist()
    if inconsistent_units:
        raise InvalidStaggeredDataError(
            f"gvar must be time-invariant within each unit. "
            f"Units with varying gvar values: {inconsistent_units[:5]}"
            f"{'...' if len(inconsistent_units) > 5 else ''}"
        )
    
    # Check for negative values (invalid)
    non_na_gvar = unit_gvar.dropna()
    negative_gvar = non_na_gvar[non_na_gvar < 0]
    if len(negative_gvar) > 0:
        neg_values = sorted(negative_gvar.unique().tolist())[:5]
        raise InvalidStaggeredDataError(
            f"gvar column contains negative values, which are not valid: {neg_values}. "
            f"Valid values: positive integer (cohort), 0 (never treated), inf (never treated), NaN (never treated)."
        )
    
    # Identify never-treated units using vectorized operation
    never_treated_mask = unit_gvar.apply(is_never_treated)
    n_never_treated = int(never_treated_mask.sum())
    
    # Identify cohorts (positive gvar values, excluding NT values)
    treated_mask = ~never_treated_mask
    treated_gvar = unit_gvar[treated_mask]
    
    if len(treated_gvar) == 0:
        raise InvalidStaggeredDataError(
            "No treatment cohorts found in data. All units appear to be never-treated. "
            f"Found gvar values: {sorted(unit_gvar.dropna().unique().tolist()[:10])}"
        )
    
    cohorts = sorted([int(g) for g in treated_gvar.unique() if pd.notna(g) and g > 0 and not np.isinf(g)])
    
    if len(cohorts) == 0:
        raise InvalidStaggeredDataError(
            "No valid treatment cohorts found. Treatment cohorts must be positive integers. "
            f"Found gvar values: {sorted(unit_gvar.dropna().unique().tolist()[:10])}"
        )
    
    # Cohort sizes
    cohort_sizes = {}
    for g in cohorts:
        cohort_sizes[g] = int((unit_gvar == g).sum())
    
    n_treated = sum(cohort_sizes.values())
    
    # Get time range
    if len(tvar_cols) == 1:
        T_min = int(data[tvar_cols[0]].min())
        T_max = int(data[tvar_cols[0]].max())
    else:
        # For quarterly data, use year as primary
        T_min = int(data[tvar_cols[0]].min())
        T_max = int(data[tvar_cols[0]].max())
    
    # Generate warnings
    warning_list = []
    
    # Warn if no never-treated units (important for aggregate estimation)
    if n_never_treated == 0:
        warning_list.append(
            "No never-treated units found in data. "
            "Impact: Only (g,r)-specific effects can be estimated using not-yet-treated controls. "
            "Cohort effects (τ_g) and overall effects (τ_ω) cannot be estimated. "
            "Use aggregate='none' to estimate (g,r)-specific effects only."
        )
    
    # Warn if cohorts are outside tvar range
    cohorts_outside = [g for g in cohorts if g < T_min or g > T_max]
    if cohorts_outside:
        warning_list.append(
            f"Some cohorts are outside the observed time range [{T_min}, {T_max}]: {cohorts_outside}. "
            f"This may indicate data issues."
        )
    
    # Warn if earliest cohort has no pre-treatment period
    min_cohort = min(cohorts)
    if min_cohort <= T_min:
        warning_list.append(
            f"Earliest cohort ({min_cohort}) has no pre-treatment period. "
            f"Data starts at T_min={T_min}. Demeaning/detrending transformation may be unreliable for this cohort."
        )
    
    # Warn if very few never-treated units
    if n_never_treated == 1:
        warning_list.append(
            f"Only 1 never-treated unit found. "
            f"Inference for cohort and overall effects may be unreliable with very few NT units."
        )
    
    # Check for unbalanced panel
    panel_counts = data.groupby(ivar)[tvar_cols[0]].count()
    if panel_counts.nunique() > 1:
        min_obs = int(panel_counts.min())
        max_obs = int(panel_counts.max())
        warning_list.append(
            f"Unbalanced panel detected: observation counts range from {min_obs} to {max_obs}. "
            f"Missing periods will be handled automatically."
        )
    
    # Check for missing values in outcome variable
    n_missing_y = data[y].isna().sum()
    if n_missing_y > 0:
        pct_missing = n_missing_y / len(data) * 100
        warning_list.append(
            f"Outcome variable '{y}' has {n_missing_y} missing values ({pct_missing:.1f}%). "
            f"These observations will be excluded from estimation."
        )
    
    return {
        'cohorts': cohorts,
        'n_cohorts': len(cohorts),
        'n_never_treated': n_never_treated,
        'n_treated': n_treated,
        'has_never_treated': n_never_treated > 0,
        'cohort_sizes': cohort_sizes,
        'T_min': T_min,
        'T_max': T_max,
        'N_total': len(unit_gvar),
        'N_obs': len(data),
        'warnings': warning_list,
    }
