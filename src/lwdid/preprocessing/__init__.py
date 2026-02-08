"""
Preprocessing utilities for repeated cross-sectional data.

This module provides functionality to aggregate repeated cross-sectional data
to panel format for use with lwdid estimation methods. The aggregation follows
the methodology described in Lee & Wooldridge (2026).

Key Functions
-------------
aggregate_to_panel
    Aggregate repeated cross-sectional data to panel format.

Key Classes
-----------
AggregationResult
    Container for aggregation results and metadata.
CellStatistics
    Statistics for individual (unit, period) cells.

Example
-------
>>> from lwdid.preprocessing import aggregate_to_panel
>>> result = aggregate_to_panel(
...     data=survey_data,
...     unit_var='state',
...     time_var='year',
...     outcome_var='income',
...     weight_var='survey_weight',
...     treatment_var='treated',
... )
>>> panel_data = result.panel_data
>>> print(result.summary())
"""

from .aggregation import (
    aggregate_to_panel,
    AggregationResult,
    CellStatistics,
)

__all__ = [
    'aggregate_to_panel',
    'AggregationResult',
    'CellStatistics',
]
