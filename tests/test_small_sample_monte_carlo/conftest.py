# -*- coding: utf-8 -*-
"""
Pytest configuration and fixtures for small-sample Monte Carlo tests.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add fixtures path to sys.path
fixtures_path = Path(__file__).parent.parent / 'test_common_timing' / 'fixtures'
if str(fixtures_path) not in sys.path:
    sys.path.insert(0, str(fixtures_path))


@pytest.fixture
def small_sample_data():
    """Generate small-sample DGP data with default parameters."""
    from dgp_small_sample import generate_small_sample_dgp
    return generate_small_sample_dgp(seed=42)


@pytest.fixture
def small_sample_data_with_components():
    """Generate small-sample DGP data with components."""
    from dgp_small_sample import generate_small_sample_dgp
    return generate_small_sample_dgp(seed=42, return_components=True)


@pytest.fixture
def scenario_1_data():
    """Generate data for Scenario 1 (p=0.32)."""
    from dgp_small_sample import generate_small_sample_dgp_from_scenario
    return generate_small_sample_dgp_from_scenario('scenario_1', seed=42)


@pytest.fixture
def scenario_2_data():
    """Generate data for Scenario 2 (p=0.24)."""
    from dgp_small_sample import generate_small_sample_dgp_from_scenario
    return generate_small_sample_dgp_from_scenario('scenario_2', seed=42)


@pytest.fixture
def scenario_3_data():
    """Generate data for Scenario 3 (p=0.17)."""
    from dgp_small_sample import generate_small_sample_dgp_from_scenario
    return generate_small_sample_dgp_from_scenario('scenario_3', seed=42)


@pytest.fixture
def zero_noise_data():
    """Generate data with zero noise for exact formula verification."""
    from dgp_small_sample import generate_small_sample_dgp
    return generate_small_sample_dgp(
        sigma_eps=0.0,
        seed=42,
        return_components=True,
    )


# Test markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "stata: marks tests requiring Stata environment"
    )
    config.addinivalue_line(
        "markers", "monte_carlo: marks Monte Carlo simulation tests"
    )
    config.addinivalue_line(
        "markers", "numerical: marks numerical validation tests"
    )
