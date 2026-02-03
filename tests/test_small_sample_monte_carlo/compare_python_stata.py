#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compare Python and Stata estimation results."""

import sys
from pathlib import Path

# Add fixtures paths
fixtures_path = Path(__file__).parent / 'fixtures'
parent_fixtures = Path(__file__).parent.parent / 'test_common_timing' / 'fixtures'
sys.path.insert(0, str(fixtures_path))
sys.path.insert(0, str(parent_fixtures))

from monte_carlo_runner import _estimate_manual_demean, _estimate_manual_detrend
from dgp_small_sample import generate_small_sample_dgp

# Generate same data as Stata test
data, params = generate_small_sample_dgp(seed=42)

print("=" * 60)
print("Python vs Stata Comparison (Unit-level Cross-sectional OLS)")
print("=" * 60)
print(f"N units: {params['n_units']}")
print(f"N treated: {params['n_treated']}")
print(f"N control: {params['n_control']}")
print(f"True ATT: {params['tau']}")
print()

# Python demeaning
demean_result = _estimate_manual_demean(data, treatment_start=11)
print("=== DEMEANING (Python) ===")
print(f"ATT: {demean_result['att']:.4f}")
print(f"SE (OLS): {demean_result['se_ols']:.4f}")
print(f"SE (HC3): {demean_result['se_hc3']:.4f}")
print(f"df: {demean_result['df']}")
print(f"N treated: {demean_result['n_treated']}, N control: {demean_result['n_control']}")
print()

# Python detrending
detrend_result = _estimate_manual_detrend(data, treatment_start=11)
print("=== DETRENDING (Python) ===")
print(f"ATT: {detrend_result['att']:.4f}")
print(f"SE (OLS): {detrend_result['se_ols']:.4f}")
print(f"SE (HC3): {detrend_result['se_hc3']:.4f}")
print(f"df: {detrend_result['df']}")
print(f"N treated: {detrend_result['n_treated']}, N control: {detrend_result['n_control']}")
print()

# Stata results (from log)
print("=== STATA RESULTS ===")
print("Demeaning ATT: 3.7353, SE (OLS): 4.4400")
print("Detrending ATT: -0.6981, SE (OLS): 1.6367, SE (HC3): 1.7151")
print()

print("=" * 60)
print("COMPARISON:")
print("=" * 60)
print(f"Demeaning ATT diff: {abs(demean_result['att'] - 3.7353):.6f}")
print(f"Demeaning SE diff: {abs(demean_result['se_ols'] - 4.4400):.6f}")
print(f"Detrending ATT diff: {abs(detrend_result['att'] - (-0.6981)):.6f}")
print(f"Detrending SE (OLS) diff: {abs(detrend_result['se_ols'] - 1.6367):.6f}")
print(f"Detrending SE (HC3) diff: {abs(detrend_result['se_hc3'] - 1.7151):.6f}")
print("=" * 60)
