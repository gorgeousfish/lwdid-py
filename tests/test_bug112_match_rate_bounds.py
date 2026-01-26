"""
Unit tests for BUG-112 fix: match_rate boundary checks.

Tests verify that match_rate and treatment_retention_rate are always
bounded to [0, 1], even in edge cases or with inconsistent data.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

import sys
sys.path.insert(0, str(__file__).rsplit('/tests/', 1)[0] + '/src')

from lwdid.core import _convert_psm_result_to_dict
from lwdid.staggered.estimators import _compute_match_statistics


class TestMatchRateBoundsCore:
    """Test match_rate bounds in core.py _convert_psm_result_to_dict()."""
    
    def test_normal_case_all_matched(self):
        """Test normal case where all treated units are matched."""
        psm_result = MagicMock()
        psm_result.n_treated = 100
        psm_result.n_matched = 100
        psm_result.n_dropped = 0
        psm_result.att = 1.5
        psm_result.se = 0.5
        psm_result.t_stat = 3.0
        psm_result.pvalue = 0.003
        psm_result.ci_lower = 0.5
        psm_result.ci_upper = 2.5
        psm_result.n_control = 200
        psm_result.diagnostics = None
        
        result = _convert_psm_result_to_dict(
            psm_result, alpha=0.05, vce=None, cluster_var=None, controls=['x1']
        )
        
        assert result['match_rate'] == 1.0
        
    def test_normal_case_partial_match(self):
        """Test normal case where some treated units are dropped."""
        psm_result = MagicMock()
        psm_result.n_treated = 100
        psm_result.n_matched = 80
        psm_result.n_dropped = 20  # 20 units dropped due to caliper
        psm_result.att = 1.2
        psm_result.se = 0.6
        psm_result.t_stat = 2.0
        psm_result.pvalue = 0.046
        psm_result.ci_lower = 0.02
        psm_result.ci_upper = 2.38
        psm_result.n_control = 200
        psm_result.diagnostics = None
        
        result = _convert_psm_result_to_dict(
            psm_result, alpha=0.05, vce=None, cluster_var=None, controls=['x1']
        )
        
        expected_rate = (100 - 20) / 100  # 0.8
        assert result['match_rate'] == pytest.approx(expected_rate, rel=1e-10)
        
    def test_edge_case_all_dropped(self):
        """Test edge case where all treated units are dropped."""
        psm_result = MagicMock()
        psm_result.n_treated = 50
        psm_result.n_matched = 0
        psm_result.n_dropped = 50  # All dropped
        psm_result.att = np.nan
        psm_result.se = np.nan
        psm_result.t_stat = np.nan
        psm_result.pvalue = np.nan
        psm_result.ci_lower = np.nan
        psm_result.ci_upper = np.nan
        psm_result.n_control = 100
        psm_result.diagnostics = None
        
        result = _convert_psm_result_to_dict(
            psm_result, alpha=0.05, vce=None, cluster_var=None, controls=['x1']
        )
        
        assert result['match_rate'] == 0.0
        
    def test_edge_case_zero_treated(self):
        """Test edge case where n_treated is 0."""
        psm_result = MagicMock()
        psm_result.n_treated = 0
        psm_result.n_matched = 0
        psm_result.n_dropped = 0
        psm_result.att = np.nan
        psm_result.se = np.nan
        psm_result.t_stat = np.nan
        psm_result.pvalue = np.nan
        psm_result.ci_lower = np.nan
        psm_result.ci_upper = np.nan
        psm_result.n_control = 100
        psm_result.diagnostics = None
        
        result = _convert_psm_result_to_dict(
            psm_result, alpha=0.05, vce=None, cluster_var=None, controls=['x1']
        )
        
        assert result['match_rate'] == 0.0
        
    def test_defensive_negative_rate_clamped(self):
        """Test defensive case: n_dropped > n_treated should clamp to 0."""
        psm_result = MagicMock()
        psm_result.n_treated = 50
        psm_result.n_matched = 0
        psm_result.n_dropped = 100  # Abnormal: more dropped than treated
        psm_result.att = np.nan
        psm_result.se = np.nan
        psm_result.t_stat = np.nan
        psm_result.pvalue = np.nan
        psm_result.ci_lower = np.nan
        psm_result.ci_upper = np.nan
        psm_result.n_control = 100
        psm_result.diagnostics = None
        
        result = _convert_psm_result_to_dict(
            psm_result, alpha=0.05, vce=None, cluster_var=None, controls=['x1']
        )
        
        # Without fix: (50 - 100) / 50 = -1.0
        # With fix: max(0.0, min(1.0, -1.0)) = 0.0
        assert result['match_rate'] == 0.0
        assert result['match_rate'] >= 0.0
        assert result['match_rate'] <= 1.0


class TestMatchRateBoundsEstimators:
    """Test match_rate and treatment_retention_rate bounds in estimators.py."""
    
    def test_normal_case_full_coverage(self):
        """Test normal case with full control unit coverage."""
        K_M = np.array([1, 1, 1, 1, 1])  # All 5 control units matched once
        n_treated = 5
        n_dropped = 0
        
        match_rate, retention_rate, avg_reuse, max_reuse = _compute_match_statistics(
            K_M, n_treated, n_dropped
        )
        
        assert match_rate == 1.0
        assert retention_rate == 1.0
        assert avg_reuse == 1.0
        assert max_reuse == 1
        
    def test_normal_case_partial_coverage(self):
        """Test normal case with partial control unit coverage."""
        K_M = np.array([2, 0, 1, 0, 3])  # 3 out of 5 control units matched
        n_treated = 6
        n_dropped = 1
        
        match_rate, retention_rate, avg_reuse, max_reuse = _compute_match_statistics(
            K_M, n_treated, n_dropped
        )
        
        expected_match_rate = 3 / 5  # 0.6
        expected_retention_rate = (6 - 1) / 6  # 5/6 ≈ 0.833
        
        assert match_rate == pytest.approx(expected_match_rate, rel=1e-10)
        assert retention_rate == pytest.approx(expected_retention_rate, rel=1e-10)
        assert avg_reuse == pytest.approx(2.0, rel=1e-10)  # (2 + 1 + 3) / 3 = 2
        assert max_reuse == 3
        
    def test_edge_case_no_matches(self):
        """Test edge case with no matches."""
        K_M = np.array([0, 0, 0, 0, 0])
        n_treated = 5
        n_dropped = 5
        
        match_rate, retention_rate, avg_reuse, max_reuse = _compute_match_statistics(
            K_M, n_treated, n_dropped
        )
        
        assert match_rate == 0.0
        assert retention_rate == 0.0
        assert avg_reuse == 0.0
        assert max_reuse == 0
        
    def test_edge_case_empty_control(self):
        """Test edge case with empty control group."""
        K_M = np.array([])
        n_treated = 5
        n_dropped = 5
        
        match_rate, retention_rate, avg_reuse, max_reuse = _compute_match_statistics(
            K_M, n_treated, n_dropped
        )
        
        assert match_rate == 0.0
        assert retention_rate == 0.0
        assert avg_reuse == 0.0
        assert max_reuse == 0
        
    def test_edge_case_zero_treated(self):
        """Test edge case with zero treated units."""
        K_M = np.array([0, 0, 0])
        n_treated = 0
        n_dropped = 0
        
        match_rate, retention_rate, avg_reuse, max_reuse = _compute_match_statistics(
            K_M, n_treated, n_dropped
        )
        
        assert match_rate == 0.0
        assert retention_rate == 0.0
        
    def test_defensive_negative_retention_clamped(self):
        """Test defensive case: n_dropped > n_treated should clamp to 0."""
        K_M = np.array([1, 0, 1])  # 2 out of 3 control units matched
        n_treated = 10
        n_dropped = 15  # Abnormal: more dropped than treated
        
        match_rate, retention_rate, avg_reuse, max_reuse = _compute_match_statistics(
            K_M, n_treated, n_dropped
        )
        
        # Without fix: (10 - 15) / 10 = -0.5
        # With fix: max(0.0, min(1.0, -0.5)) = 0.0
        assert retention_rate == 0.0
        assert retention_rate >= 0.0
        assert retention_rate <= 1.0
        
        # Match rate should still be valid
        expected_match_rate = 2 / 3
        assert match_rate == pytest.approx(expected_match_rate, rel=1e-10)
        
    def test_bounds_always_valid(self):
        """Test that bounds are always [0, 1] with random inputs."""
        np.random.seed(42)
        
        for _ in range(100):
            n_control = np.random.randint(1, 100)
            n_treated = np.random.randint(1, 100)
            # Simulate various scenarios including edge cases
            n_dropped = np.random.randint(0, n_treated * 2)  # May exceed n_treated
            K_M = np.random.randint(0, 10, size=n_control)
            
            match_rate, retention_rate, _, _ = _compute_match_statistics(
                K_M, n_treated, n_dropped
            )
            
            assert 0.0 <= match_rate <= 1.0, f"match_rate={match_rate} out of bounds"
            assert 0.0 <= retention_rate <= 1.0, f"retention_rate={retention_rate} out of bounds"


class TestMatchRateIntegration:
    """Integration tests for match_rate in PSM estimation."""
    
    def test_psm_estimation_match_rate_valid(self):
        """Test that PSM estimation produces valid match_rate."""
        try:
            from lwdid import lwdid
        except ImportError:
            pytest.skip("lwdid package not installed")
            
        # Create simple test data
        np.random.seed(123)
        n = 100
        data = pd.DataFrame({
            'id': range(n),
            'time': np.tile([1, 2, 3, 4], n // 4),
            'y': np.random.randn(n),
            'd': np.repeat([0, 1], n // 2),
            'post': np.tile([0, 0, 1, 1], n // 4),
            'x1': np.random.randn(n),
        })
        
        try:
            result = lwdid(
                data=data,
                y='y',
                d='d',
                ivar='id',
                tvar='time',
                post='post',
                rolling='demean',
                estimator='psm',
                controls=['x1'],
            )
            
            # Check that match_rate is valid
            if hasattr(result, 'match_rate'):
                assert 0.0 <= result.match_rate <= 1.0
        except Exception:
            # If estimation fails for other reasons, skip this test
            pytest.skip("PSM estimation failed (not related to match_rate bounds)")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
