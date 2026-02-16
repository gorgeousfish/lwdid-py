"""
Shared fixtures and helpers for staggered DiD tests.

Provides:
- Stata reference data loading for cross-validation
- Subsample construction utilities matching Stata filtering logic
- Transformed outcome computation following Paper Procedure 4.1

References
----------
Lee, S. J. & Wooldridge, J. M. (2025). A Simple Transformation Approach to
    Difference-in-Differences Estimation for Panel Data. SSRN 4516518.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_STAGGERED_DIR = Path(__file__).resolve().parent
_TESTS_DIR = _STAGGERED_DIR.parent
_PACKAGE_ROOT = _TESTS_DIR.parent
_DATA_DIR = _PACKAGE_ROOT / "data"


def get_test_data_path(filename: str) -> str:
    """Return the absolute path to a file inside ``tests/staggered/reference_data/``."""
    return str(_STAGGERED_DIR / "reference_data" / filename)


# ---------------------------------------------------------------------------
# Stata reference values (hardcoded, verified against Stata teffects output)
# ---------------------------------------------------------------------------

STATA_IPWRA_RESULTS: Dict[Tuple[int, int], dict] = {
    # (cohort_g, period_r): {att, se, n_obs, n_treated, n_control}
    (4, 4): {
        "att": 4.3029238,
        "se": 0.42367713,
        "n_obs": 1000,
        "n_treated": 129,
        "n_control": 871,
        "condition": "f04",
    },
    (4, 5): {
        "att": 6.6112909,
        "se": 0.43215951,
        "n_obs": 891,
        "n_treated": 129,
        "n_control": 762,
        "condition": "f05 & ~g5",
    },
    (4, 6): {
        "att": 8.3343553,
        "se": 0.44138304,
        "n_obs": 781,
        "n_treated": 129,
        "n_control": 652,
        "condition": "f06 & (g5 + g6 != 1)",
    },
    (5, 5): {
        "att": 3.0283627,
        "se": 0.42077459,
        "n_obs": 871,
        "n_treated": 109,
        "n_control": 762,
        "condition": "f05 & ~g4",
    },
    (5, 6): {
        "att": 4.9326076,
        "se": 0.44026846,
        "n_obs": 761,
        "n_treated": 109,
        "n_control": 652,
        "condition": "f06 & (g4 + g6 != 1)",
    },
    (6, 6): {
        "att": 2.4200472,
        "se": 0.48314986,
        "n_obs": 762,
        "n_treated": 110,
        "n_control": 652,
        "condition": "f06 & (g4 + g5 != 1)",
    },
}

# Expected control-group cohorts per (g, r) pair (0 = never-treated)
EXPECTED_CONTROL_COHORTS: Dict[Tuple[int, int], set] = {
    (4, 4): {5, 6, 0},
    (4, 5): {6, 0},
    (4, 6): {0},        # only never-treated
    (5, 5): {6, 0},
    (5, 6): {0},
    (6, 6): {0},
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stata_ipwra_results() -> dict:
    """Return the Stata IPWRA reference results dictionary."""
    return STATA_IPWRA_RESULTS


@pytest.fixture(scope="module")
def expected_control_cohorts() -> dict:
    """Return the expected control-cohort sets for each (g, r) pair."""
    return EXPECTED_CONTROL_COHORTS


@pytest.fixture
def small_staggered_data() -> pd.DataFrame:
    """Generate a small staggered panel for fast unit tests.

    Layout
    ------
    - 100 units, 6 periods (t = 1..6)
    - Cohort allocation: ~12% g=4, ~11% g=5, ~11% g=6, ~66% never-treated
    - DGP: Y = 1 + 0.5*t + 0.3*x1 + 0.2*x2 + tau(g,r) + eps
    - Treatment effect: tau(g,r) = 1.5 + 0.5*(r-g) + 0.3*(g-4)
    """
    rng = np.random.default_rng(42)
    n_units, T = 100, 6

    cohort_probs = [0.66, 0.12, 0.11, 0.11]
    cohort_values = [0, 4, 5, 6]
    unit_cohorts = rng.choice(cohort_values, size=n_units, p=cohort_probs)

    x1 = rng.standard_normal(n_units)
    x2 = rng.standard_normal(n_units)

    records = []
    for i in range(n_units):
        g = unit_cohorts[i]
        for t in range(1, T + 1):
            y_base = 1 + 0.5 * t + 0.3 * x1[i] + 0.2 * x2[i] + rng.standard_normal()
            if g > 0 and t >= g:
                tau = 1.5 + 0.5 * (t - g) + 0.3 * (g - 4)
                y = y_base + tau
            else:
                y = y_base
            records.append({
                "id": i + 1,
                "year": t,
                "y": y,
                "gvar": g,
                "x1": x1[i],
                "x2": x2[i],
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Subsample construction (matches Stata filtering logic)
# ---------------------------------------------------------------------------

def build_subsample_for_gr(
    data: pd.DataFrame,
    cohort_g: int,
    period_r: int,
    gvar_col: str = "gvar",
    period_col: str = "year",
    id_col: str = "id",
    never_treated_values: Optional[List] = None,
) -> pd.DataFrame:
    """Build a (g, r)-specific subsample following Paper Procedure 4.1.

    The subsample satisfies :math:`D_{ig} + A_{r+1} = 1`, i.e. each unit is
    either in cohort *g* (treated) or not-yet-treated at period *r* (control).

    Parameters
    ----------
    data : pd.DataFrame
        Full panel dataset.
    cohort_g : int
        Target treatment cohort.
    period_r : int
        Target evaluation period.
    gvar_col, period_col, id_col : str
        Column names for cohort, period, and unit identifier.
    never_treated_values : list, optional
        Values of *gvar* that indicate never-treated status (default ``[0]``).

    Returns
    -------
    pd.DataFrame
        Subsample with an added ``d`` column (1 = treated, 0 = control).

    Raises
    ------
    ValueError
        If either the treated or control group is empty.

    Notes
    -----
    Control group uses **strict** inequality ``gvar > period_r``.
    """
    if never_treated_values is None:
        never_treated_values = [0]

    data_r = data[data[period_col] == period_r].copy()
    if len(data_r) == 0:
        raise ValueError(f"No observations found for period={period_r}")

    unit_gvar = data_r.set_index(id_col)[gvar_col]
    is_treated = unit_gvar == cohort_g
    is_control = unit_gvar.isin(never_treated_values) | (unit_gvar > period_r)

    valid_units = (is_treated | is_control)
    valid_ids = valid_units[valid_units].index
    subsample = data_r[data_r[id_col].isin(valid_ids)].copy()
    subsample["d"] = (subsample[gvar_col] == cohort_g).astype(int)

    n_treated = (subsample["d"] == 1).sum()
    n_control = (subsample["d"] == 0).sum()

    if n_treated == 0:
        raise ValueError(
            f"(g={cohort_g}, r={period_r}): Empty treated group."
        )
    if n_control == 0:
        raise ValueError(
            f"(g={cohort_g}, r={period_r}): Empty control group."
        )
    return subsample


def compute_transformed_outcome(
    data: pd.DataFrame,
    y_col: str,
    id_col: str,
    period_col: str,
    cohort_g: int,
    period_r: int,
) -> pd.Series:
    r"""Compute the demeaned outcome :math:`\hat{Y}_{irg}` for a (g, r) pair.

    .. math::

        \hat{Y}_{irg} = Y_{ir} - \frac{1}{g-1} \sum_{s=1}^{g-1} Y_{is}

    The pre-treatment mean is computed once per cohort *g* and held fixed
    across all post-treatment periods *r* (see Paper Section 4.12).

    Parameters
    ----------
    data : pd.DataFrame
        Full panel dataset.
    y_col, id_col, period_col : str
        Column names for outcome, unit, and period.
    cohort_g : int
        Treatment cohort (determines the pre-treatment window).
    period_r : int
        Evaluation period.

    Returns
    -------
    pd.Series
        Transformed outcome indexed by unit ID.
    """
    Y_r = data[data[period_col] == period_r].set_index(id_col)[y_col]

    pre_periods = list(range(1, cohort_g))
    if len(pre_periods) == 0:
        raise ValueError(f"cohort_g={cohort_g}: no pre-treatment periods")

    pre_mean = (
        data[data[period_col].isin(pre_periods)]
        .groupby(id_col)[y_col]
        .mean()
    )

    common_ids = Y_r.index.intersection(pre_mean.index)
    return Y_r.loc[common_ids] - pre_mean.loc[common_ids]
