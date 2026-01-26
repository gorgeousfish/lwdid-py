"""
Stata Consistency Verification Test

This test verifies that the Python lwdid estimators produce results
numerically consistent with Stata's teffects commands.

Stata Reference Results (seed=12345, N=200):
- IPW ATT: 1.3890146, SE: 0.15554898
- IPWRA ATT: 1.4185803, SE: 0.15233085
- PSM ATT: 1.3636744, SE: 0.19020659 (AI Robust SE)
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lwdid.staggered.estimators import estimate_ipw, estimate_ipwra, estimate_psm


# Stata reference values
STATA_IPW_ATT = 1.3890146
STATA_IPW_SE = 0.15554898
STATA_IPWRA_ATT = 1.4185803
STATA_IPWRA_SE = 0.15233085
STATA_PSM_ATT = 1.3636744
STATA_PSM_SE = 0.19020659


@pytest.fixture
def stata_data():
    """Load data exported from Stata."""
    data_path = os.path.join(os.path.dirname(__file__), 'stata_comparison_data.csv')
    if not os.path.exists(data_path):
        pytest.skip("Stata comparison data not found")
    return pd.read_csv(data_path)


class TestStataIPWConsistency:
    """Test IPW estimator consistency with Stata teffects ipw."""
    
    def test_ipw_att_consistency(self, stata_data):
        """Test IPW ATT estimate is close to Stata."""
        result = estimate_ipw(
            data=stata_data,
            y='y',
            d='d',
            propensity_controls=['x1', 'x2'],
            alpha=0.05,
        )
        
        rel_diff_att = abs(result.att - STATA_IPW_ATT) / abs(STATA_IPW_ATT)
        
        print(f"\nIPW Comparison:")
        print(f"  Python ATT: {result.att:.6f}")
        print(f"  Stata ATT:  {STATA_IPW_ATT:.6f}")
        print(f"  Relative diff: {rel_diff_att*100:.4f}%")
        print(f"  Python SE: {result.se:.6f}")
        print(f"  Stata SE:  {STATA_IPW_SE:.6f}")
        
        # Allow 5% relative difference for ATT
        assert rel_diff_att < 0.05, f"IPW ATT differs by {rel_diff_att*100:.2f}%"
    
    def test_ipw_se_consistency(self, stata_data):
        """Test IPW SE estimate is close to Stata."""
        result = estimate_ipw(
            data=stata_data,
            y='y',
            d='d',
            propensity_controls=['x1', 'x2'],
            alpha=0.05,
        )
        
        rel_diff_se = abs(result.se - STATA_IPW_SE) / abs(STATA_IPW_SE)
        
        # Allow 10% relative difference for SE (more variation expected)
        assert rel_diff_se < 0.10, f"IPW SE differs by {rel_diff_se*100:.2f}%"


class TestStataIPWRAConsistency:
    """Test IPWRA estimator consistency with Stata teffects ipwra."""
    
    def test_ipwra_att_consistency(self, stata_data):
        """Test IPWRA ATT estimate is close to Stata."""
        result = estimate_ipwra(
            data=stata_data,
            y='y',
            d='d',
            controls=['x1', 'x2'],
            propensity_controls=['x1', 'x2'],
            alpha=0.05,
        )
        
        rel_diff_att = abs(result.att - STATA_IPWRA_ATT) / abs(STATA_IPWRA_ATT)
        
        print(f"\nIPWRA Comparison:")
        print(f"  Python ATT: {result.att:.6f}")
        print(f"  Stata ATT:  {STATA_IPWRA_ATT:.6f}")
        print(f"  Relative diff: {rel_diff_att*100:.4f}%")
        print(f"  Python SE: {result.se:.6f}")
        print(f"  Stata SE:  {STATA_IPWRA_SE:.6f}")
        
        # Allow 5% relative difference for ATT
        assert rel_diff_att < 0.05, f"IPWRA ATT differs by {rel_diff_att*100:.2f}%"
    
    def test_ipwra_se_consistency(self, stata_data):
        """Test IPWRA SE estimate is close to Stata."""
        result = estimate_ipwra(
            data=stata_data,
            y='y',
            d='d',
            controls=['x1', 'x2'],
            propensity_controls=['x1', 'x2'],
            alpha=0.05,
        )
        
        rel_diff_se = abs(result.se - STATA_IPWRA_SE) / abs(STATA_IPWRA_SE)
        
        # Allow 10% relative difference for SE
        assert rel_diff_se < 0.10, f"IPWRA SE differs by {rel_diff_se*100:.2f}%"


class TestStataPSMConsistency:
    """Test PSM estimator consistency with Stata teffects psmatch."""
    
    def test_psm_att_consistency(self, stata_data):
        """Test PSM ATT estimate is close to Stata."""
        result = estimate_psm(
            data=stata_data,
            y='y',
            d='d',
            propensity_controls=['x1', 'x2'],
            n_neighbors=1,
            with_replacement=True,
            alpha=0.05,
        )
        
        rel_diff_att = abs(result.att - STATA_PSM_ATT) / abs(STATA_PSM_ATT)
        
        print(f"\nPSM Comparison:")
        print(f"  Python ATT: {result.att:.6f}")
        print(f"  Stata ATT:  {STATA_PSM_ATT:.6f}")
        print(f"  Relative diff: {rel_diff_att*100:.4f}%")
        print(f"  Python SE: {result.se:.6f}")
        print(f"  Stata SE:  {STATA_PSM_SE:.6f}")
        
        # Allow 5% relative difference for ATT
        assert rel_diff_att < 0.05, f"PSM ATT differs by {rel_diff_att*100:.2f}%"
    
    def test_psm_se_consistency(self, stata_data):
        """Test PSM SE estimate is close to Stata (AI Robust SE)."""
        result = estimate_psm(
            data=stata_data,
            y='y',
            d='d',
            propensity_controls=['x1', 'x2'],
            n_neighbors=1,
            with_replacement=True,
            se_method='abadie_imbens_full',
            alpha=0.05,
        )
        
        rel_diff_se = abs(result.se - STATA_PSM_SE) / abs(STATA_PSM_SE)
        
        # Allow 15% relative difference for SE (PSM SE has more variation)
        assert rel_diff_se < 0.15, f"PSM SE differs by {rel_diff_se*100:.2f}%"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
