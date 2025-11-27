"""
Visualization module for the lwdid package.

Provides plotting functions for transformed outcomes.
"""

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from .exceptions import InvalidParameterError, VisualizationError


def _resolve_gid(
    data: pd.DataFrame,
    ivar_var: str,
    d_var: str,
    gid: Union[str, int]
) -> int:
    """
    Resolve gid to unit identifier in data
    
    Parameters
    ----------
    data : pd.DataFrame
        Panel data
    ivar_var : str
        Unit identifier
    d_var : str
        Treatment indicator
    gid : str or int
        Unit ID to resolve
    
    Returns
    -------
    int
        Resolved unit ID
    
    Raises
    ------
    InvalidParameterError
        If gid not found or not a treated unit
    """
    original_gid = gid
    mapping = data.attrs.get('id_mapping', None)
    if mapping and 'original_to_numeric' in mapping:
        gid_str = str(gid) if not isinstance(gid, str) else gid
        gid_num = mapping['original_to_numeric'].get(gid_str)
        if gid_num is not None:
            gid_resolved = gid_num
        else:
            gid_to_match = gid
            if isinstance(gid, str) and pd.api.types.is_numeric_dtype(data[ivar_var]):
                try:
                    gid_to_match = pd.to_numeric(gid)
                except (ValueError, TypeError):
                    pass
            elif not isinstance(gid, str) and pd.api.types.is_string_dtype(data[ivar_var]):
                gid_to_match = str(gid)

            mask = (data[ivar_var] == gid_to_match)
            if not mask.any():
                raise InvalidParameterError(f"gid '{original_gid}' not found")
            gid_resolved = data.loc[mask, ivar_var].iloc[0]
    else:
        gid_to_match = gid
        if isinstance(gid, str) and pd.api.types.is_numeric_dtype(data[ivar_var]):
            try:
                gid_to_match = pd.to_numeric(gid)
            except (ValueError, TypeError):
                pass
        elif not isinstance(gid, str) and pd.api.types.is_string_dtype(data[ivar_var]):
            gid_to_match = str(gid)

        mask = (data[ivar_var] == gid_to_match)
        if not mask.any():
            raise InvalidParameterError(f"gid '{original_gid}' not found")
        gid_resolved = data.loc[mask, ivar_var].iloc[0]

    unit_rows = data[data[ivar_var] == gid_resolved]
    if len(unit_rows) == 0:
        raise InvalidParameterError(f"gid '{original_gid}' not found")
    d_max = int(unit_rows[d_var].max())
    if d_max != 1:
        raise InvalidParameterError(f"'{original_gid}' is not a treated unit")

    return gid_resolved


def prepare_plot_data(
    data: pd.DataFrame,
    ydot_var: str,
    d_var: str,
    tindex_var: str,
    ivar_var: str,
    gid: Optional[Union[str, int]],
    tpost1: int,
    Tmax: int,
    period_labels: Dict[int, str],
) -> dict:
    """
    Prepare plot data dictionary
    
    Parameters
    ----------
    data : pd.DataFrame
        Transformed panel data
    ydot_var : str
        Residualized outcome
    d_var : str
        Treatment indicator
    tindex_var : str
        Time index
    ivar_var : str
        Unit identifier
    gid : str or int, optional
        Unit ID to plot (None for treated group average)
    tpost1 : int
        First post-treatment period
    Tmax : int
        Last period
    period_labels : dict
        Mapping tindex → period label
    
    Returns
    -------
    dict
        'time', 'control_mean', 'treated_series', 'intervention_point',
        'treated_label', 'period_labels'
    """
    required = {ydot_var, d_var, tindex_var, ivar_var}
    missing = required - set(data.columns)
    if missing:
        raise VisualizationError(f"Missing required columns: {sorted(missing)}")

    time = list(range(1, int(Tmax) + 1))

    control_mean_series = (
        data[data[d_var] == 0]
        .groupby(tindex_var)[ydot_var]
        .mean()
        .reindex(time)
    )

    if gid is not None:
        gid_resolved = _resolve_gid(data, ivar_var, d_var, gid)
        unit = data[data[ivar_var] == gid_resolved]
        treated_series = (
            unit.set_index(tindex_var)[ydot_var]
            .reindex(time)
        )
        treated_label = f"Unit {gid}"
    else:
        treated_series = (
            data[data[d_var] == 1]
            .groupby(tindex_var)[ydot_var]
            .mean()
            .reindex(time)
        )
        treated_label = "Treated (Average)"

    return {
        'time': time,
        'control_mean': control_mean_series.tolist(),
        'treated_series': treated_series.tolist(),
        'intervention_point': int(tpost1),
        'treated_label': treated_label,
        'period_labels': period_labels,
    }


def plot_results(
    plot_data: dict,
    graph_options: Optional[dict] = None,
):
    """
    Generate plot from prepared data
    
    Parameters
    ----------
    plot_data : dict
        Data from prepare_plot_data()
    graph_options : dict, optional
        'figsize', 'title', 'xlabel', 'ylabel', 'legend_loc', 'dpi', 'savefig'
    
    Returns
    -------
    matplotlib.Figure
        Generated figure
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except Exception as exc:
        raise VisualizationError(
            'Install required dependencies: matplotlib>=3.3.'
        ) from exc

    opts = {
        'figsize': (10, 6),
        'title': None,
        'xlabel': None,
        'ylabel': 'Residualized Outcome',
        'legend_loc': 'best',
        'dpi': 100,
        'savefig': None,
    }
    if graph_options:
        opts.update(graph_options)

    time = plot_data['time']
    ctrl = plot_data['control_mean']
    trt = plot_data['treated_series']
    tpost1 = plot_data['intervention_point']
    tlabel = plot_data['treated_label']
    period_labels = plot_data.get('period_labels', {})

    fig, ax = plt.subplots(figsize=opts['figsize'], dpi=opts['dpi'])

    ax.plot(time, ctrl, linestyle='--', color='blue', linewidth=1.5, label='Control')
    ax.plot(time, trt, linestyle='-', color='red', linewidth=2.0, label=tlabel)
    ax.axvline(x=tpost1, linestyle='--', color='black', linewidth=1.0, alpha=0.7, label='Intervention')

    ax.set_xticks(time)
    ax.set_xticklabels([period_labels.get(t, str(t)) for t in time], rotation=45, ha='right')
    if opts['xlabel'] is not None:
        ax.set_xlabel(opts['xlabel'])
    ax.set_ylabel(opts['ylabel'])
    ax.yaxis.set_major_locator(MaxNLocator(nbins='auto'))

    if opts['title']:
        ax.set_title(opts['title'])

    ax.legend(loc=opts['legend_loc'], frameon=True, shadow=True)
    fig.tight_layout()

    if opts['savefig']:
        fig.savefig(opts['savefig'], dpi=opts['dpi'])

    return fig


