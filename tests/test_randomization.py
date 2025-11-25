"""Randomization Inference Tests

Tests for randomization inference (RI) functionality including:
- Reproducibility with fixed seeds
- P-value computation and validation
- Bootstrap resampling methods
"""

import os
import pandas as pd
import pytest

from lwdid import lwdid


@pytest.mark.slow
def test_ri_pvalue_reproducible_and_in_range():
    """Test randomization inference p-value reproducibility and range.

    Verifies that:
    1. RI p-values are reproducible with the same seed
    2. P-values fall within expected range for the smoking dataset
    3. Seed is correctly stored in results object

    Note: Range is relaxed (0.02-0.04) to account for randomization variability
    with bootstrap resampling when N_treated=1.
    """
    here = os.path.dirname(__file__)
    data = pd.read_csv(os.path.join(here, 'data', 'smoking.csv'))

    seed = 12345
    res1 = lwdid(
        data, y='lcigsale', d='d', ivar='state', tvar='year', post='post',
        rolling='detrend', vce='robust', ri=True, rireps=1000, seed=seed
    )

    # Verify seed is stored
    assert res1.ri_seed == seed

    # Verify p-value is in expected range
    assert 0.02 <= res1.ri_pvalue <= 0.04, \
        f"RI p-value {res1.ri_pvalue} outside expected range [0.02, 0.04]"

    # Verify reproducibility: same seed should give identical p-value
    res2 = lwdid(
        data, y='lcigsale', d='d', ivar='state', tvar='year', post='post',
        rolling='detrend', vce='robust', ri=True, rireps=1000, seed=seed
    )
    assert res2.ri_pvalue == res1.ri_pvalue, \
        "Same seed should produce identical RI p-values"


