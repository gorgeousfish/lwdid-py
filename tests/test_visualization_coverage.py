"""
Additional tests for visualization.py to boost coverage.
"""

import warnings
import pytest
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


@pytest.fixture
def common_timing_data():
    """Create common timing test data for visualization."""
    np.random.seed(42)
    
    data = []
    for unit in range(1, 11):
        treated = 1 if unit <= 3 else 0
        
        for year in range(2000, 2006):
            tindex = year - 1999  # 1-indexed time
            post = 1 if year >= 2003 else 0
            y = 1.0 + 0.1 * unit + np.random.normal(0, 0.1)
            ydot = y - 0.5  # Simulated transformed outcome
            if treated and post:
                ydot += 0.3
            
            data.append({
                'id': unit,
                'year': year,
                'tindex': tindex,
                'y': y,
                'ydot': ydot,
                'd': treated,
                'post': post
            })
    
    return pd.DataFrame(data)


class TestResolveGid:
    """Tests for _resolve_gid function."""
    
    def test_resolve_numeric_gid(self, common_timing_data):
        """Test resolving numeric gid."""
        from lwdid.visualization import _resolve_gid
        
        result = _resolve_gid(common_timing_data, 'id', 'd', 1)
        assert result == 1
    
    def test_resolve_string_gid_numeric_column(self, common_timing_data):
        """Test resolving string gid when column is numeric."""
        from lwdid.visualization import _resolve_gid
        
        result = _resolve_gid(common_timing_data, 'id', 'd', '1')
        assert result == 1
    
    def test_resolve_gid_not_found(self, common_timing_data):
        """Test error when gid not found."""
        from lwdid.visualization import _resolve_gid
        from lwdid.exceptions import InvalidParameterError
        
        with pytest.raises(InvalidParameterError, match="not found"):
            _resolve_gid(common_timing_data, 'id', 'd', 999)
    
    def test_resolve_gid_not_treated(self, common_timing_data):
        """Test error when gid is not a treated unit."""
        from lwdid.visualization import _resolve_gid
        from lwdid.exceptions import InvalidParameterError
        
        # Unit 5 is control (d=0)
        with pytest.raises(InvalidParameterError, match="not a treated"):
            _resolve_gid(common_timing_data, 'id', 'd', 5)
    
    def test_resolve_gid_with_mapping(self, common_timing_data):
        """Test resolving gid with id_mapping in attrs."""
        from lwdid.visualization import _resolve_gid
        
        # Add id_mapping to attrs
        common_timing_data.attrs['id_mapping'] = {
            'original_to_numeric': {'unit_1': 1, 'unit_2': 2}
        }
        
        result = _resolve_gid(common_timing_data, 'id', 'd', 'unit_1')
        assert result == 1


class TestPreparePlotData:
    """Tests for prepare_plot_data function."""
    
    def test_prepare_plot_data_with_gid(self, common_timing_data):
        """Test prepare_plot_data with specific unit."""
        from lwdid.visualization import prepare_plot_data
        
        period_labels = {i: str(1999 + i) for i in range(1, 7)}
        
        result = prepare_plot_data(
            data=common_timing_data,
            ydot_var='ydot',
            d_var='d',
            tindex_var='tindex',
            ivar_var='id',
            gid=1,
            tpost1=4,  # 2003 is period 4
            Tmax=6,
            period_labels=period_labels
        )
        
        assert 'time' in result
        assert 'control_mean' in result
        assert 'treated_series' in result
        assert 'intervention_point' in result
        assert result['treated_label'] == 'Unit 1'
    
    def test_prepare_plot_data_average(self, common_timing_data):
        """Test prepare_plot_data with treated average."""
        from lwdid.visualization import prepare_plot_data
        
        period_labels = {i: str(1999 + i) for i in range(1, 7)}
        
        result = prepare_plot_data(
            data=common_timing_data,
            ydot_var='ydot',
            d_var='d',
            tindex_var='tindex',
            ivar_var='id',
            gid=None,  # Use average
            tpost1=4,
            Tmax=6,
            period_labels=period_labels
        )
        
        assert result['treated_label'] == 'Treated (Average)'
    
    def test_prepare_plot_data_missing_columns(self, common_timing_data):
        """Test error when required columns are missing."""
        from lwdid.visualization import prepare_plot_data
        from lwdid.exceptions import VisualizationError
        
        with pytest.raises(VisualizationError, match="Missing required"):
            prepare_plot_data(
                data=common_timing_data,
                ydot_var='nonexistent',
                d_var='d',
                tindex_var='tindex',
                ivar_var='id',
                gid=None,
                tpost1=4,
                Tmax=6,
                period_labels={}
            )


class TestPlotResults:
    """Tests for plot_results function."""
    
    def test_plot_results_basic(self, common_timing_data):
        """Test basic plot generation."""
        from lwdid.visualization import prepare_plot_data, plot_results
        
        period_labels = {i: str(1999 + i) for i in range(1, 7)}
        
        plot_data = prepare_plot_data(
            data=common_timing_data,
            ydot_var='ydot',
            d_var='d',
            tindex_var='tindex',
            ivar_var='id',
            gid=None,
            tpost1=4,
            Tmax=6,
            period_labels=period_labels
        )
        
        fig = plot_results(plot_data)
        assert fig is not None
        plt.close(fig)
    
    def test_plot_results_with_options(self, common_timing_data):
        """Test plot with custom options."""
        from lwdid.visualization import prepare_plot_data, plot_results
        
        period_labels = {i: str(1999 + i) for i in range(1, 7)}
        
        plot_data = prepare_plot_data(
            data=common_timing_data,
            ydot_var='ydot',
            d_var='d',
            tindex_var='tindex',
            ivar_var='id',
            gid=1,
            tpost1=4,
            Tmax=6,
            period_labels=period_labels
        )
        
        graph_options = {
            'figsize': (12, 8),
            'title': 'Test Plot',
            'xlabel': 'Time Period',
            'ylabel': 'Effect',
            'legend_loc': 'upper left',
            'dpi': 150
        }
        
        fig = plot_results(plot_data, graph_options=graph_options)
        assert fig is not None
        plt.close(fig)
    
    def test_plot_results_savefig(self, common_timing_data, tmp_path):
        """Test plot saving to file."""
        from lwdid.visualization import prepare_plot_data, plot_results
        
        period_labels = {i: str(1999 + i) for i in range(1, 7)}
        
        plot_data = prepare_plot_data(
            data=common_timing_data,
            ydot_var='ydot',
            d_var='d',
            tindex_var='tindex',
            ivar_var='id',
            gid=None,
            tpost1=4,
            Tmax=6,
            period_labels=period_labels
        )
        
        save_path = tmp_path / 'test_plot.png'
        graph_options = {'savefig': str(save_path)}
        
        fig = plot_results(plot_data, graph_options=graph_options)
        assert save_path.exists()
        plt.close(fig)


class TestVisualizationIntegration:
    """Integration tests for visualization with lwdid."""
    
    def test_lwdid_with_graph(self):
        """Test lwdid with graph=True."""
        from lwdid import lwdid
        
        np.random.seed(42)
        data = []
        for unit in range(1, 6):
            treated = 1 if unit == 1 else 0
            for year in range(2000, 2005):
                post = 1 if year >= 2003 else 0
                y = 1.0 + 0.1 * unit + np.random.normal(0, 0.1)
                if treated and post:
                    y += 0.5
                data.append({
                    'id': unit, 'year': year, 'y': y,
                    'd': treated, 'post': post
                })
        df = pd.DataFrame(data)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=df,
                y='y', d='d', ivar='id', tvar='year', post='post',
                rolling='demean',
                graph=True
            )
        
        assert result is not None
        plt.close('all')
    
    def test_lwdid_with_graph_and_gid(self):
        """Test lwdid with graph=True and specific gid."""
        from lwdid import lwdid
        
        np.random.seed(42)
        data = []
        for unit in range(1, 6):
            treated = 1 if unit == 1 else 0
            for year in range(2000, 2005):
                post = 1 if year >= 2003 else 0
                y = 1.0 + 0.1 * unit + np.random.normal(0, 0.1)
                if treated and post:
                    y += 0.5
                data.append({
                    'id': unit, 'year': year, 'y': y,
                    'd': treated, 'post': post
                })
        df = pd.DataFrame(data)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=df,
                y='y', d='d', ivar='id', tvar='year', post='post',
                rolling='demean',
                graph=True,
                gid=1
            )
        
        assert result is not None
        plt.close('all')


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
