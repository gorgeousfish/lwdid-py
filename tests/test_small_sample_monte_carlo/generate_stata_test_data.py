#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate test data for Stata comparison."""

import sys
from pathlib import Path

# Add fixtures paths
parent_fixtures = Path(__file__).parent.parent / 'test_common_timing' / 'fixtures'
sys.path.insert(0, str(parent_fixtures))

from dgp_small_sample import generate_small_sample_dgp

# Generate test data
data, params = generate_small_sample_dgp(seed=42)
print(f'Generated data: {len(data)} rows')
print(f'N_treated: {params["n_treated"]}, N_control: {params["n_control"]}')
print(f'True ATT: {params["tau"]}')

# Save to CSV for Stata
output_path = '/Users/cxy/Desktop/大样本lwdid/stata-mcp-folder/small_sample_test.csv'
data.to_csv(output_path, index=False)
print(f'Saved to {output_path}')
