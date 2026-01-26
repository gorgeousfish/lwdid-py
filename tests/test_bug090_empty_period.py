"""
BUG-090: Test empty period data handling in estimate_period_effects()

Verifies that when a period has no observations:
1. A clear warning is issued
2. Results for that period are set to NaN
3. Other periods are processed normally
4. The function does not raise unclear regression errors
"""

import numpy as np
import pandas as pd
import pytest
import warnings

from lwdid.estimation import estimate_period_effects


class TestEmptyPeriodHandling:
    """Test estimate_period_effects() handling of empty period data."""

    def test_empty_period_returns_nan_with_warning(self):
        """When a period has no data, should warn and return NaN results."""
        # Create data with missing period t=4 (no observations)
        data = pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'tindex': [1, 2, 3, 1, 2, 3, 1, 2, 3],  # No t=4
            'ydot': [0.5, 0.3, 1.0, -0.2, 0.1, 0.3, 0.0, -0.1, 0.2],
            'd_': [1, 1, 1, 0, 0, 0, 0, 0, 0],
        })
        period_labels = {3: "2003", 4: "2004"}

        # Should warn about empty period t=4
        with pytest.warns(UserWarning, match="Period 2004.*contains no observations"):
            df = estimate_period_effects(
                data=data,
                ydot='ydot',
                d='d_',
                tindex='tindex',
                tpost1=3,
                Tmax=4,
                controls_spec=None,
                vce=None,
                cluster_var=None,
                period_labels=period_labels
            )

        # Should have 2 rows (periods 3 and 4)
        assert len(df) == 2

        # Period 3 (t=3) should have valid results
        row_3 = df[df['tindex'] == 3].iloc[0]
        assert row_3['period'] == "2003"
        assert not np.isnan(row_3['beta'])
        assert not np.isnan(row_3['se'])
        assert row_3['N'] == 3

        # Period 4 (t=4) should have NaN results
        row_4 = df[df['tindex'] == 4].iloc[0]
        assert row_4['period'] == "2004"
        assert np.isnan(row_4['beta'])
        assert np.isnan(row_4['se'])
        assert np.isnan(row_4['ci_lower'])
        assert np.isnan(row_4['ci_upper'])
        assert np.isnan(row_4['tstat'])
        assert np.isnan(row_4['pval'])
        assert row_4['N'] == 0

    def test_all_periods_empty_returns_all_nan(self):
        """When all periods are empty, should return all NaN results."""
        # Create data with no observations for post-treatment periods
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'tindex': [1, 2, 1, 2],  # Only pre-treatment periods
            'ydot': [0.5, 0.3, -0.2, 0.1],
            'd_': [1, 1, 0, 0],
        })
        period_labels = {3: "2003", 4: "2004"}

        # Should warn about both empty periods
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df = estimate_period_effects(
                data=data,
                ydot='ydot',
                d='d_',
                tindex='tindex',
                tpost1=3,
                Tmax=4,
                controls_spec=None,
                vce=None,
                cluster_var=None,
                period_labels=period_labels
            )

            # Should have 2 warnings (one for each empty period)
            empty_period_warnings = [
                warning for warning in w
                if "contains no observations" in str(warning.message)
            ]
            assert len(empty_period_warnings) == 2

        # All results should be NaN
        assert len(df) == 2
        assert all(np.isnan(df['beta']))
        assert all(np.isnan(df['se']))
        assert all(df['N'] == 0)

    def test_normal_data_unaffected(self):
        """Normal data with all periods present should work unchanged."""
        # Create data with all periods present
        data = pd.DataFrame({
            'id': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            'tindex': [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
            'ydot': [0.5, 0.3, 1.0, 1.2, -0.2, 0.1, 0.3, 0.4, 0.0, -0.1, 0.2, 0.5],
            'd_': [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        })
        period_labels = {3: "2003", 4: "2004"}

        # Should NOT warn (no empty periods)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df = estimate_period_effects(
                data=data,
                ydot='ydot',
                d='d_',
                tindex='tindex',
                tpost1=3,
                Tmax=4,
                controls_spec=None,
                vce=None,
                cluster_var=None,
                period_labels=period_labels
            )

            # No empty period warnings
            empty_warnings = [
                warning for warning in w
                if "contains no observations" in str(warning.message)
            ]
            assert len(empty_warnings) == 0

        # All results should be valid
        assert len(df) == 2
        assert not any(np.isnan(df['beta']))
        assert not any(np.isnan(df['se']))
        assert all(df['N'] == 3)

    def test_mixed_empty_and_valid_periods(self):
        """Test mix of empty and valid periods across range."""
        # Create data where t=3 exists, t=4 empty, t=5 exists
        data = pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'tindex': [1, 3, 5, 1, 3, 5, 1, 3, 5],  # t=4 missing
            'ydot': [0.5, 1.0, 1.5, -0.2, 0.3, 0.6, 0.0, 0.2, 0.4],
            'd_': [1, 1, 1, 0, 0, 0, 0, 0, 0],
        })
        period_labels = {3: "2003", 4: "2004", 5: "2005"}

        with pytest.warns(UserWarning, match="Period 2004.*contains no observations"):
            df = estimate_period_effects(
                data=data,
                ydot='ydot',
                d='d_',
                tindex='tindex',
                tpost1=3,
                Tmax=5,
                controls_spec=None,
                vce=None,
                cluster_var=None,
                period_labels=period_labels
            )

        assert len(df) == 3

        # Period 3: valid
        row_3 = df[df['tindex'] == 3].iloc[0]
        assert not np.isnan(row_3['beta'])
        assert row_3['N'] == 3

        # Period 4: empty
        row_4 = df[df['tindex'] == 4].iloc[0]
        assert np.isnan(row_4['beta'])
        assert row_4['N'] == 0

        # Period 5: valid
        row_5 = df[df['tindex'] == 5].iloc[0]
        assert not np.isnan(row_5['beta'])
        assert row_5['N'] == 3

    def test_empty_period_with_vce_robust(self):
        """Empty period handling should work with different VCE types."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3],
            'tindex': [1, 3, 1, 3, 1, 3],  # t=4 missing
            'ydot': [0.5, 1.0, -0.2, 0.3, 0.0, 0.2],
            'd_': [1, 1, 0, 0, 0, 0],
        })
        period_labels = {3: "2003", 4: "2004"}

        for vce in [None, 'robust', 'hc3']:
            with pytest.warns(UserWarning, match="contains no observations"):
                df = estimate_period_effects(
                    data=data,
                    ydot='ydot',
                    d='d_',
                    tindex='tindex',
                    tpost1=3,
                    Tmax=4,
                    controls_spec=None,
                    vce=vce,
                    cluster_var=None,
                    period_labels=period_labels
                )

            # Empty period should still be NaN
            row_4 = df[df['tindex'] == 4].iloc[0]
            assert np.isnan(row_4['beta'])
            assert row_4['N'] == 0

    def test_period_label_fallback_for_unlabeled_period(self):
        """Empty period without label should use string representation."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'tindex': [1, 3, 1, 3],  # t=4 missing, no label
            'ydot': [0.5, 1.0, -0.2, 0.3],
            'd_': [1, 1, 0, 0],
        })
        period_labels = {3: "2003"}  # No label for t=4

        with pytest.warns(UserWarning, match=r"Period 4 \(t=4\) contains no observations"):
            df = estimate_period_effects(
                data=data,
                ydot='ydot',
                d='d_',
                tindex='tindex',
                tpost1=3,
                Tmax=4,
                controls_spec=None,
                vce=None,
                cluster_var=None,
                period_labels=period_labels
            )

        # Empty period should use str(t) as label
        row_4 = df[df['tindex'] == 4].iloc[0]
        assert row_4['period'] == '4'


class TestEmptyPeriodEdgeCases:
    """Edge case tests for empty period handling."""

    def test_single_period_empty(self):
        """Single post-treatment period that is empty."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'tindex': [1, 2, 1, 2],  # Only pre-treatment
            'ydot': [0.5, 0.3, -0.2, 0.1],
            'd_': [1, 1, 0, 0],
        })
        period_labels = {3: "2003"}

        with pytest.warns(UserWarning, match="contains no observations"):
            df = estimate_period_effects(
                data=data,
                ydot='ydot',
                d='d_',
                tindex='tindex',
                tpost1=3,
                Tmax=3,
                controls_spec=None,
                vce=None,
                cluster_var=None,
                period_labels=period_labels
            )

        assert len(df) == 1
        assert np.isnan(df.iloc[0]['beta'])
        assert df.iloc[0]['N'] == 0

    def test_dataframe_structure_preserved(self):
        """Empty period results should have correct DataFrame structure."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'tindex': [1, 2, 1, 2],
            'ydot': [0.5, 0.3, -0.2, 0.1],
            'd_': [1, 1, 0, 0],
        })
        period_labels = {3: "2003"}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = estimate_period_effects(
                data=data,
                ydot='ydot',
                d='d_',
                tindex='tindex',
                tpost1=3,
                Tmax=3,
                controls_spec=None,
                vce=None,
                cluster_var=None,
                period_labels=period_labels
            )

        # Verify all expected columns exist
        expected_cols = ['period', 'tindex', 'beta', 'se', 'ci_lower', 
                         'ci_upper', 'tstat', 'pval', 'N']
        assert list(df.columns) == expected_cols

        # Verify column types
        assert df['period'].dtype == object
        assert df['tindex'].dtype in [np.int64, int]
        assert df['N'].dtype in [np.int64, int]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
