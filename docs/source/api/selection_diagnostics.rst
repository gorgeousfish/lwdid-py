Selection Diagnostics Module (selection_diagnostics)
=====================================================

Diagnostic tools for assessing potential selection bias in unbalanced panel data.

This module helps evaluate whether missing data patterns in unbalanced panels
may compromise the validity of difference-in-differences estimation. The key
assumption is that selection (missing data) may depend on unobserved time-invariant
heterogeneity (which is removed by the rolling transformation), but cannot
systematically depend on outcome shocks in the untreated state.

Overview
--------

The selection mechanism assumption is analogous to the standard fixed effects
assumption: units may be selected based on permanent characteristics, but not
based on time-varying shocks. When this assumption holds, the rolling
transformation removes selection bias along with unit fixed effects.

This module provides:

- **Balance analysis**: Assess how balanced the panel is across units and time
- **Attrition diagnostics**: Identify patterns in unit dropout
- **Missing data classification**: Classify missing patterns as MCAR, MAR, or MNAR
- **Risk assessment**: Evaluate the overall risk of selection bias

Enums
-----

.. autoclass:: lwdid.selection_diagnostics.MissingPattern
   :members:
   :undoc-members:

.. autoclass:: lwdid.selection_diagnostics.SelectionRisk
   :members:
   :undoc-members:

Data Classes
------------

.. autoclass:: lwdid.selection_diagnostics.SelectionDiagnostics
   :members:
   :no-index:

.. autoclass:: lwdid.selection_diagnostics.BalanceStatistics
   :members:
   :no-index:

.. autoclass:: lwdid.selection_diagnostics.AttritionAnalysis
   :members:
   :no-index:

.. autoclass:: lwdid.selection_diagnostics.UnitMissingStats
   :members:
   :no-index:

.. autoclass:: lwdid.selection_diagnostics.SelectionTestResult
   :members:
   :no-index:

Main Functions
--------------

.. autofunction:: lwdid.selection_diagnostics.diagnose_selection_mechanism

.. autofunction:: lwdid.selection_diagnostics.get_unit_missing_stats

.. autofunction:: lwdid.selection_diagnostics.plot_missing_pattern

Example Usage
-------------

.. code-block:: python

   from lwdid import diagnose_selection_mechanism, get_unit_missing_stats
   
   # Run comprehensive selection diagnostics
   diagnostics = diagnose_selection_mechanism(
       data=panel_data,
       ivar='unit',
       tvar='year',
       gvar='first_treat'
   )
   
   # Check risk level
   print(f"Selection risk: {diagnostics.risk_level}")
   print(f"Missing pattern: {diagnostics.missing_pattern}")
   
   # Get per-unit statistics
   unit_stats = get_unit_missing_stats(
       data=panel_data,
       ivar='unit',
       tvar='year'
   )
   
   # Visualize missing patterns
   from lwdid import plot_missing_pattern
   fig, ax = plot_missing_pattern(
       data=panel_data,
       ivar='unit',
       tvar='year'
   )

Interpretation Guide
--------------------

**Risk Levels:**

- **LOW**: Selection mechanism assumption likely holds. Proceed with estimation.
- **MEDIUM**: Some indicators suggest potential issues. Consider using detrending
  and sensitivity analysis.
- **HIGH**: Strong evidence of problematic selection. Results should be
  interpreted with caution.

**Missing Patterns:**

- **MCAR**: Missing Completely At Random. Most benign pattern, no bias expected.
- **MAR**: Missing At Random. Acceptable when controls are included.
- **MNAR**: Missing Not At Random. May violate selection mechanism assumption
  if missingness depends on outcome shocks.

See Also
--------

- :doc:`../methodological_notes` - Theoretical foundations
- :doc:`../user_guide` - Practical guidance on handling unbalanced panels
- :func:`lwdid.lwdid` - Main estimation function with ``balanced_panel`` parameter
