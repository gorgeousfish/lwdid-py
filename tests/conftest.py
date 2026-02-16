"""
Root-level pytest configuration for the lwdid test suite.

Provides shared fixtures and helper utilities used across all test modules.
Fixtures include loaders for the empirical datasets used in the papers:

- Smoking data: California Proposition 99 (common timing case)
  Source: Lee & Wooldridge (2023), Table 3
- Castle Law data: Stand-Your-Ground laws (staggered adoption case)
  Source: Lee & Wooldridge (2025), Section 6 and Table A4

References
----------
Lee, S. J. & Wooldridge, J. M. (2023). A Simple Transformation Approach to
    Difference-in-Differences Estimation for Panel Data. SSRN 5325686.
Lee, S. J. & Wooldridge, J. M. (2025). A Simple Transformation Approach to
    Difference-in-Differences Estimation for Panel Data. SSRN 4516518.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_TESTS_DIR = Path(__file__).resolve().parent
_PACKAGE_ROOT = _TESTS_DIR.parent
_DATA_DIR = _PACKAGE_ROOT / "data"


def _find_data_file(filename: str) -> Path | None:
    """Locate a data file by searching standard locations.

    Parameters
    ----------
    filename : str
        Name of the data file (e.g. ``"smoking.csv"``).

    Returns
    -------
    Path or None
        Resolved path if found, otherwise ``None``.
    """
    candidates = [
        _DATA_DIR / filename,
        _TESTS_DIR / "data" / filename,
        Path("data") / filename,
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    return None


# ---------------------------------------------------------------------------
# Smoking data fixtures (common timing)
# ---------------------------------------------------------------------------

@pytest.fixture
def smoking_data_path() -> Path:
    """Return the resolved path to ``smoking.csv``.

    Skips the test if the file cannot be located.
    """
    path = _find_data_file("smoking.csv")
    if path is None:
        pytest.skip("smoking.csv not found in expected locations")
    return path


@pytest.fixture
def smoking_data(smoking_data_path) -> pd.DataFrame:
    """Load the California Proposition 99 smoking dataset.

    Returns a fresh copy on each invocation to prevent cross-test
    side effects.  The dataset contains 39 U.S. states observed from
    1970 to 2000.

    Columns of interest:
        - ``lcigsale``: log per-capita cigarette sales (outcome)
        - ``state``: state numeric identifier (unit ID)
        - ``year``: calendar year (time variable)
        - ``d``: treatment indicator (1 = California)
        - ``post``: post-treatment indicator (1 if year >= 1989)
    """
    return pd.read_csv(smoking_data_path).copy()


# ---------------------------------------------------------------------------
# Castle Law data fixtures (staggered adoption)
# ---------------------------------------------------------------------------

@pytest.fixture
def castle_data() -> pd.DataFrame:
    """Load and preprocess the Castle Doctrine / Stand-Your-Ground dataset.

    The dataset covers 50 U.S. states from 2000 to 2010, with staggered
    adoption of Castle Doctrine laws across five cohorts (2005-2009) and
    29 never-treated states.

    Pre-processing:
        - ``gvar`` is created from ``effyear`` with NaN mapped to 0
          (indicating never-treated status).

    Columns of interest:
        - ``lhomicide``: log homicide rate (outcome)
        - ``sid``: state numeric identifier (unit ID)
        - ``year``: calendar year (time variable)
        - ``effyear``: first treatment year (NaN = never treated)
        - ``gvar``: first treatment year (0 = never treated)

    Returns a fresh copy to prevent cross-test side effects.
    """
    data_path = _DATA_DIR / "castle.csv"
    if not data_path.exists():
        pytest.skip(f"Castle data not found: {data_path}")

    df = pd.read_csv(data_path)
    df["gvar"] = df["effyear"].fillna(0).astype(int)
    return df.copy()


# ---------------------------------------------------------------------------
# Synthetic data fixtures for unit tests
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_panel() -> pd.DataFrame:
    """Generate a minimal balanced panel for unit tests.

    Layout: 10 units x 6 periods, 3 treated (from period 4 onward).
    True ATT = 2.0 under a simple additive model.
    """
    rng = np.random.default_rng(42)
    records = []
    for unit in range(1, 11):
        treated = 1 if unit <= 3 else 0
        for t in range(1, 7):
            post = 1 if t >= 4 else 0
            y = 1.0 + 0.5 * unit + 0.3 * t + rng.normal(0, 0.1)
            if treated and post:
                y += 2.0  # true ATT
            records.append({
                "id": unit,
                "year": 2000 + t,
                "y": y,
                "d": treated,
                "post": post,
            })
    return pd.DataFrame(records)


@pytest.fixture
def staggered_panel() -> pd.DataFrame:
    """Generate a minimal staggered adoption panel for unit tests.

    Layout: 60 units x 10 periods, two treatment cohorts (g=6, g=8)
    plus 20 never-treated units.  True ATT = 1.5.
    """
    rng = np.random.default_rng(123)
    records = []
    for unit in range(1, 61):
        if unit <= 20:
            gvar = 6
        elif unit <= 40:
            gvar = 8
        else:
            gvar = 0  # never treated
        for t in range(1, 11):
            treated = 1 if gvar > 0 and t >= gvar else 0
            y = 2.0 + 0.2 * unit + 0.4 * t + rng.normal(0, 0.5)
            if treated:
                y += 1.5  # true ATT
            records.append({
                "id": unit,
                "year": 2000 + t,
                "y": y,
                "gvar": gvar,
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# pytest configuration
# ---------------------------------------------------------------------------

def pytest_configure(config):
    """Register custom markers for the lwdid test suite."""
    config.addinivalue_line("markers", "slow: marks tests that take > 10 seconds")
    config.addinivalue_line("markers", "integration: end-to-end integration tests")
    config.addinivalue_line(
        "markers", "stata_alignment: tests validated against Stata output"
    )
    config.addinivalue_line(
        "markers", "paper_validation: tests replicating published results"
    )
    config.addinivalue_line("markers", "monte_carlo: Monte Carlo simulation tests")
