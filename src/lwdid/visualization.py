"""
Visualization utilities for difference-in-differences analysis.

This module provides plotting functions for visualizing transformed outcomes
in panel data difference-in-differences settings. The primary use case is
comparing the trajectory of residualized outcomes between treated units
(or their group average) and the control group mean across time periods.

The visualization functions support both single treated unit analysis and
aggregated treatment group comparisons. Plots display pre-treatment fit
quality and post-intervention treatment effect gaps, with customizable
appearance options.

Notes
-----
Requires matplotlib >= 3.3 for plotting functionality. The module raises
VisualizationError if matplotlib is not installed when plot generation
is requested.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .exceptions import InvalidParameterError, VisualizationError

if TYPE_CHECKING:
    from typing import Any


def _resolve_gid(
    data: pd.DataFrame,
    ivar_var: str,
    d_var: str,
    gid: str | int
) -> int:
    """
    Resolve a user-provided unit identifier to the internal representation.

    Maps user-specified unit identifiers to the internal numeric identifiers
    used in the transformed data. Handles string-to-numeric conversions and
    validates that the resolved identifier corresponds to a treated unit.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data containing the unit identifier column.
    ivar_var : str
        Name of the unit identifier column.
    d_var : str
        Name of the binary treatment indicator column (1 = treated).
    gid : str or int
        User-specified unit identifier to resolve.

    Returns
    -------
    int
        Internal unit identifier corresponding to the input.

    Raises
    ------
    InvalidParameterError
        If the unit identifier is not found in the data or does not
        correspond to a treated unit.
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
    gid: str | int | None,
    tpost1: int,
    Tmax: int,
    period_labels: dict[int, str],
) -> dict[str, Any]:
    """
    Prepare data structures for plotting transformed outcomes.

    Computes control group means and treated unit (or group average) series
    across all time periods. The output dictionary contains all necessary
    data for generating comparative time series plots.

    Parameters
    ----------
    data : pd.DataFrame
        Transformed panel data containing residualized outcomes.
    ydot_var : str
        Name of the column containing the residualized outcome variable
        (unit-specific mean or trend removed).
    d_var : str
        Name of the binary treatment indicator column (1 = treated).
    tindex_var : str
        Name of the time period index column.
    ivar_var : str
        Name of the unit identifier column.
    gid : str, int, or None
        Unit identifier for a specific treated unit to plot. If None,
        computes the average across all treated units.
    tpost1 : int
        First post-treatment time period (intervention point).
    Tmax : int
        Final time period in the panel.
    period_labels : dict of {int: str}
        Mapping from time index values to display labels for the x-axis.

    Returns
    -------
    dict
        Dictionary containing:

        - ``time`` : list of int
            Time period indices from 1 to Tmax.
        - ``control_mean`` : list of float
            Control group mean of the residualized outcome for each period.
        - ``treated_series`` : list of float
            Treated unit or group average of the residualized outcome.
        - ``intervention_point`` : int
            First post-treatment period for the vertical intervention line.
        - ``treated_label`` : str
            Label for the treated series in the plot legend.
        - ``period_labels`` : dict
            Time index to label mapping for x-axis tick labels.

    Raises
    ------
    VisualizationError
        If required columns are missing from the data.
    InvalidParameterError
        If gid is specified but not found or not a treated unit.
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
    plot_data: dict[str, Any],
    graph_options: dict[str, Any] | None = None,
):
    """
    Generate a time series plot comparing treated and control outcomes.

    Creates a matplotlib figure displaying the residualized outcome
    trajectories for treated units (or their average) and control group mean
    across all time periods. A vertical line marks the intervention point.

    Parameters
    ----------
    plot_data : dict
        Data dictionary from :func:`prepare_plot_data` containing time
        indices, outcome series, and labeling information.
    graph_options : dict, optional
        Customization options for the plot appearance:

        - ``figsize`` : tuple of (width, height), default (10, 6)
        - ``title`` : str or None, plot title
        - ``xlabel`` : str or None, x-axis label
        - ``ylabel`` : str, y-axis label, default 'Residualized Outcome'
        - ``legend_loc`` : str, legend position, default 'best'
        - ``dpi`` : int, figure resolution, default 100
        - ``savefig`` : str or None, file path to save the figure

    Returns
    -------
    matplotlib.figure.Figure
        Generated matplotlib figure object.

    Raises
    ------
    VisualizationError
        If matplotlib is not installed.

    See Also
    --------
    prepare_plot_data : Prepare the data dictionary for plotting.
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


