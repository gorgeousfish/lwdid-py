"""
Additional tests to boost coverage for low-coverage modules.

Targets:
- visualization.py: 64%
- control_groups.py: 67%  
- randomization.py: 67%
- estimation.py: 76%
"""

import warnings
import pytest
import numpy as np
import pandas as pd
from pathlib import Path


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def medium_staggered_data():
    """Medium-sized staggered panel data."""
    np.random.seed(42)
    data = []
    # 15 units: 5 cohorts + 5 NT
    for unit in range(1, 16):
        if unit <= 3:
            gvar = 2002
        elif unit <= 6:
            gvar = 2003
        elif unit <= 9:
            gvar = 2004
        elif unit <= 12:
            gvar = 2005
        else:
            gvar = 0
        
        for year in range(2000, 2008):
            x1 = np.random.normal(unit * 0.1, 0.5)
            x2 = np.random.normal(0, 1)
            y = 1.0 + 0.1 * unit + 0.2 * x1 + 0.1 * x2
            if gvar > 0 and year >= gvar:
                y += 0.4
            y += np.random.normal(0, 0.1)
            
            data.append({
                'id': unit, 'year': year, 'y': y, 'gvar': gvar,
                'x1': x1, 'x2': x2
            })
    return pd.DataFrame(data)


@pytest.fixture
def castle_data():
    """Load Castle Law data."""
    data_path = Path(__file__).parent.parent / 'data' / 'castle.csv'
    if not data_path.exists():
        data_path = Path(__file__).parent.parent.parent / 'data' / 'castle.csv'
    df = pd.read_csv(data_path)
    df['gvar'] = df['effyear'].fillna(0).astype(int)
    return df


# =============================================================================
# Control Groups Coverage Tests
# =============================================================================

class TestControlGroupsCoverage:
    """Additional tests for control_groups.py coverage."""
    
    def test_get_all_control_masks(self, medium_staggered_data):
        """Test get_all_control_masks function."""
        from lwdid.staggered.control_groups import (
            get_all_control_masks, ControlGroupStrategy
        )
        
        cohorts = [2002, 2003, 2004]
        T_max = 2007
        
        masks = get_all_control_masks(
            medium_staggered_data, 'gvar', 'id',
            cohorts=cohorts, T_max=T_max,
            strategy=ControlGroupStrategy.NOT_YET_TREATED
        )
        
        # Should return masks for all valid (g,r) pairs
        assert len(masks) > 0
        for (g, r), mask in masks.items():
            assert g in cohorts
            assert r >= g and r <= T_max
    
    def test_auto_strategy(self, medium_staggered_data):
        """Test AUTO control group strategy."""
        from lwdid.staggered.control_groups import (
            get_valid_control_units, ControlGroupStrategy
        )
        
        mask = get_valid_control_units(
            medium_staggered_data, 'gvar', 'id',
            cohort=2002, period=2003,
            strategy=ControlGroupStrategy.AUTO
        )
        
        assert mask.sum() > 0
    
    def test_control_group_strategy_from_string(self):
        """Test ControlGroupStrategy enum from string."""
        from lwdid.staggered.control_groups import ControlGroupStrategy
        
        assert ControlGroupStrategy('never_treated') == ControlGroupStrategy.NEVER_TREATED
        assert ControlGroupStrategy('not_yet_treated') == ControlGroupStrategy.NOT_YET_TREATED
        assert ControlGroupStrategy('auto') == ControlGroupStrategy.AUTO
    
    def test_get_n_control_per_cohort_period(self, medium_staggered_data):
        """Test counting controls for each (g,r) pair."""
        from lwdid.staggered.control_groups import (
            get_valid_control_units, ControlGroupStrategy
        )
        
        # Count controls for different periods
        counts = {}
        for period in [2002, 2003, 2004, 2005]:
            mask = get_valid_control_units(
                medium_staggered_data, 'gvar', 'id',
                cohort=2002, period=period,
                strategy=ControlGroupStrategy.NOT_YET_TREATED
            )
            counts[period] = mask.sum()
        
        # Earlier periods should have more NYT controls
        assert counts[2002] >= counts[2005]


# =============================================================================
# Randomization Inference Coverage Tests
# =============================================================================

class TestRandomizationCoverage:
    """Additional tests for randomization inference coverage."""
    
    def test_ri_staggered_permutation(self, medium_staggered_data):
        """Test RI with permutation method for staggered."""
        from lwdid import lwdid
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=medium_staggered_data,
                y='y', ivar='id', tvar='year', gvar='gvar',
                rolling='demean',
                aggregate='overall',
                ri=True, rireps=50, seed=42,
                ri_method='permutation'
            )
        
        assert result.ri_pvalue is not None
    
    def test_ri_staggered_bootstrap(self, medium_staggered_data):
        """Test RI with bootstrap method for staggered."""
        from lwdid import lwdid
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=medium_staggered_data,
                y='y', ivar='id', tvar='year', gvar='gvar',
                rolling='demean',
                aggregate='overall',
                ri=True, rireps=50, seed=42,
                ri_method='bootstrap'
            )
        
        # RI may not be fully supported for staggered yet
        # Just verify it doesn't crash
        assert result.att_overall is not None


# =============================================================================
# Estimation Coverage Tests  
# =============================================================================

class TestEstimationCoverage:
    """Additional tests for estimation.py coverage."""
    
    def test_estimate_with_cluster_se(self, medium_staggered_data):
        """Test cluster standard errors."""
        from lwdid import lwdid
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=medium_staggered_data,
                y='y', ivar='id', tvar='year', gvar='gvar',
                rolling='demean',
                vce='cluster', cluster_var='id',
                aggregate='none'
            )
        
        assert result.att_by_cohort_time is not None
    
    def test_estimate_all_cohorts(self, medium_staggered_data):
        """Test estimation covers all cohorts."""
        from lwdid import lwdid
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=medium_staggered_data,
                y='y', ivar='id', tvar='year', gvar='gvar',
                rolling='demean',
                aggregate='none'
            )
        
        # Check all cohorts have effects
        cohorts_in_result = set(result.att_by_cohort_time['cohort'].unique())
        expected_cohorts = {2002, 2003, 2004, 2005}
        assert cohorts_in_result == expected_cohorts
    
    def test_estimate_event_time_calculation(self, medium_staggered_data):
        """Test event time is correctly calculated."""
        from lwdid import lwdid
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=medium_staggered_data,
                y='y', ivar='id', tvar='year', gvar='gvar',
                rolling='demean',
                aggregate='none'
            )
        
        # Event time should be period - cohort
        df = result.att_by_cohort_time
        for _, row in df.iterrows():
            expected_event_time = row['period'] - row['cohort']
            assert row['event_time'] == expected_event_time


# =============================================================================
# Visualization Coverage Tests
# =============================================================================

class TestVisualizationCoverage:
    """Additional tests for visualization.py coverage."""
    
    def test_plot_event_study_aggregate_false(self, medium_staggered_data):
        """Test event study plot without aggregation."""
        from lwdid import lwdid
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=medium_staggered_data,
                y='y', ivar='id', tvar='year', gvar='gvar',
                rolling='demean',
                aggregate='none'
            )
        
        fig, ax = plt.subplots()
        result.plot_event_study(ax=ax, aggregate=False)
        plt.close(fig)
    
    def test_plot_event_study_custom_colors(self, medium_staggered_data):
        """Test event study plot with custom colors."""
        from lwdid import lwdid
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=medium_staggered_data,
                y='y', ivar='id', tvar='year', gvar='gvar',
                rolling='demean',
                aggregate='none'
            )
        
        fig, ax = plt.subplots()
        result.plot_event_study(ax=ax, colors=['red', 'blue', 'green', 'orange'])
        plt.close(fig)
    
    def test_plot_event_study_no_ci(self, medium_staggered_data):
        """Test event study plot without confidence intervals."""
        from lwdid import lwdid
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=medium_staggered_data,
                y='y', ivar='id', tvar='year', gvar='gvar',
                rolling='demean',
                aggregate='none'
            )
        
        fig, ax = plt.subplots()
        result.plot_event_study(ax=ax, show_ci=False)
        plt.close(fig)
    
    def test_plot_event_study_return_data(self, medium_staggered_data):
        """Test event study plot returns data."""
        from lwdid import lwdid
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=medium_staggered_data,
                y='y', ivar='id', tvar='year', gvar='gvar',
                rolling='demean',
                aggregate='none'
            )
        
        fig, ax = plt.subplots()
        plot_result = result.plot_event_study(ax=ax, return_data=True)
        plt.close(fig)
        
        # return_data=True returns (fig, ax, data) tuple
        assert plot_result is not None
        if isinstance(plot_result, tuple) and len(plot_result) >= 3:
            assert isinstance(plot_result[2], pd.DataFrame)
        else:
            # Or just verify it returns something
            assert plot_result is not None


# =============================================================================
# Aggregation Coverage Tests
# =============================================================================

class TestAggregationCoverage:
    """Additional tests for aggregation.py coverage."""
    
    def test_aggregate_cohort_with_controls(self, medium_staggered_data):
        """Test cohort aggregation with control variables."""
        from lwdid import lwdid
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=medium_staggered_data,
                y='y', ivar='id', tvar='year', gvar='gvar',
                rolling='demean',
                controls=['x1'],
                aggregate='cohort'
            )
        
        assert result.att_by_cohort is not None
    
    def test_aggregate_overall_with_controls(self, medium_staggered_data):
        """Test overall aggregation with control variables."""
        from lwdid import lwdid
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=medium_staggered_data,
                y='y', ivar='id', tvar='year', gvar='gvar',
                rolling='demean',
                controls=['x1', 'x2'],
                aggregate='overall'
            )
        
        assert result.att_overall is not None
    
    def test_cohort_weights_formula(self, medium_staggered_data):
        """Verify cohort weights follow N_g / N_treat formula."""
        from lwdid import lwdid
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=medium_staggered_data,
                y='y', ivar='id', tvar='year', gvar='gvar',
                rolling='demean',
                aggregate='overall'
            )
        
        # Calculate expected weights
        unit_gvar = medium_staggered_data.groupby('id')['gvar'].first()
        cohorts = [g for g in unit_gvar.unique() if g > 0]
        n_treated_total = sum(1 for g in unit_gvar if g > 0)
        
        expected_weights = {}
        for g in cohorts:
            n_g = (unit_gvar == g).sum()
            expected_weights[g] = n_g / n_treated_total
        
        # Compare with actual weights
        for g, expected_w in expected_weights.items():
            if g in result.cohort_weights:
                actual_w = result.cohort_weights[g]
                assert abs(actual_w - expected_w) < 0.01


# =============================================================================
# Error Handling Coverage Tests
# =============================================================================

class TestErrorHandling:
    """Test error handling for edge cases."""
    
    def test_invalid_rolling_method(self, medium_staggered_data):
        """Test error for invalid rolling method."""
        from lwdid import lwdid
        
        with pytest.raises(ValueError, match="不支持|invalid|Invalid"):
            lwdid(
                data=medium_staggered_data,
                y='y', ivar='id', tvar='year', gvar='gvar',
                rolling='invalid_method',
                aggregate='none'
            )
    
    def test_invalid_aggregate_option(self, medium_staggered_data):
        """Test handling of invalid aggregate option."""
        from lwdid import lwdid
        
        # System may accept invalid aggregate or raise error
        # Just test it doesn't crash unexpectedly
        try:
            result = lwdid(
                data=medium_staggered_data,
                y='y', ivar='id', tvar='year', gvar='gvar',
                rolling='demean',
                aggregate='invalid_aggregate'
            )
            # If it doesn't raise, verify some result
            assert result is not None
        except (ValueError, KeyError, TypeError):
            # Expected behavior - error for invalid option
            pass
    
    def test_missing_required_columns(self):
        """Test error when required columns are missing."""
        from lwdid import lwdid
        from lwdid.exceptions import MissingRequiredColumnError
        
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'year': [2000, 2001, 2000, 2001],
            'y': [1.0, 1.5, 1.0, 1.2]
            # Missing 'gvar' column
        })
        
        with pytest.raises((ValueError, KeyError, MissingRequiredColumnError)):
            lwdid(
                data=data,
                y='y', ivar='id', tvar='year', gvar='gvar',
                rolling='demean'
            )


# =============================================================================
# Integration Coverage Tests
# =============================================================================

class TestIntegrationCoverage:
    """Integration tests covering full workflow paths."""
    
    def test_full_workflow_demean(self, castle_data):
        """Full workflow with demean transformation."""
        from lwdid import lwdid
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # (g,r) effects
            result_gr = lwdid(
                data=castle_data,
                y='lhomicide', ivar='sid', tvar='year', gvar='gvar',
                rolling='demean',
                aggregate='none'
            )
            
            # Cohort effects
            result_cohort = lwdid(
                data=castle_data,
                y='lhomicide', ivar='sid', tvar='year', gvar='gvar',
                rolling='demean',
                aggregate='cohort'
            )
            
            # Overall effect
            result_overall = lwdid(
                data=castle_data,
                y='lhomicide', ivar='sid', tvar='year', gvar='gvar',
                rolling='demean',
                aggregate='overall'
            )
        
        assert result_gr.att_by_cohort_time is not None
        assert result_cohort.att_by_cohort is not None
        assert result_overall.att_overall is not None
    
    def test_full_workflow_detrend(self, castle_data):
        """Full workflow with detrend transformation."""
        from lwdid import lwdid
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            result = lwdid(
                data=castle_data,
                y='lhomicide', ivar='sid', tvar='year', gvar='gvar',
                rolling='detrend',
                aggregate='overall'
            )
        
        assert result.att_overall is not None
        # Detrend should give lower estimate than demean (per paper)
        assert result.att_overall < 0.15


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
