Sensitivity Analysis
====================

.. module:: lwdid.sensitivity

This module provides tools for assessing the robustness of difference-in-differences
estimates to specification choices, implementing the recommendations from
Lee and Wooldridge (2026) and Lee and Wooldridge (2025).

Main Functions
--------------

.. autofunction:: robustness_pre_periods

.. autofunction:: sensitivity_no_anticipation

.. autofunction:: sensitivity_analysis

.. autofunction:: plot_sensitivity

Result Classes
--------------

.. autoclass:: PrePeriodRobustnessResult
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: NoAnticipationSensitivityResult
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ComprehensiveSensitivityResult
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: SpecificationResult
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: AnticipationEstimate
   :members:
   :undoc-members:
   :show-inheritance:

Enumerations
------------

.. autoclass:: RobustnessLevel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: AnticipationDetectionMethod
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Pre-treatment Period Robustness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Assess how estimates change with different numbers of pre-treatment periods::

    from lwdid import robustness_pre_periods

    result = robustness_pre_periods(
        data,
        y='outcome',
        ivar='unit',
        tvar='year',
        gvar='first_treat',
        rolling='detrend',
        pre_period_range=(3, 8),
        verbose=True
    )

    # View summary
    print(result.summary())

    # Visualize results
    fig = result.plot()

    # Access detailed results
    df = result.to_dataframe()
    print(f"Sensitivity ratio: {result.sensitivity_ratio:.1%}")
    print(f"Robustness level: {result.robustness_level.value}")

No-Anticipation Sensitivity
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Test sensitivity to potential anticipation effects::

    from lwdid import sensitivity_no_anticipation

    result = sensitivity_no_anticipation(
        data,
        y='outcome',
        ivar='unit',
        tvar='year',
        gvar='first_treat',
        max_anticipation=3,
        detection_threshold=0.10
    )

    if result.anticipation_detected:
        print(f"Anticipation detected!")
        print(f"Recommended exclusion: {result.recommended_exclusion} periods")
    else:
        print("No significant anticipation effects detected")

Comprehensive Analysis
~~~~~~~~~~~~~~~~~~~~~~

Run multiple sensitivity analyses together::

    from lwdid import sensitivity_analysis

    result = sensitivity_analysis(
        data,
        y='outcome',
        ivar='unit',
        tvar='year',
        gvar='first_treat',
        analyses=['pre_periods', 'anticipation'],
        verbose=True
    )

    # View comprehensive summary
    print(result.summary())

    # Plot all analyses
    result.plot_all()

Using exclude_pre_periods
~~~~~~~~~~~~~~~~~~~~~~~~~

Apply sensitivity analysis findings to main estimation::

    from lwdid import lwdid

    # Based on sensitivity analysis, exclude 2 periods before treatment
    result = lwdid(
        data,
        y='outcome',
        d='d',
        ivar='unit',
        tvar='year',
        post='post',
        rolling='demean',
        exclude_pre_periods=2  # Exclude periods immediately before treatment
    )

    print(result.summary())

Interpreting Results
--------------------

Sensitivity Ratio Thresholds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The sensitivity ratio measures estimate stability across specifications:

- **< 10%**: Highly robust - estimates are stable
- **10-25%**: Moderately robust - some sensitivity, generally acceptable
- **25-50%**: Sensitive - interpret with caution
- **≥ 50%**: Highly sensitive - results depend heavily on specification

Recommendations
~~~~~~~~~~~~~~~

The analysis functions provide automated recommendations based on:

1. **Sensitivity ratio**: How much estimates vary
2. **Sign consistency**: Whether all estimates have the same sign
3. **Significance consistency**: Whether all estimates are statistically significant
4. **Pattern detection**: Whether estimates show systematic trends

When estimates are sensitive, consider:

- Using ``rolling='detrend'`` if heterogeneous trends may be present
- Excluding periods with potential anticipation effects
- Reporting the range of estimates for transparency
- Investigating data quality in specific periods

See Also
--------

- :doc:`../methodological_notes` - Theoretical background
- :doc:`core` - Main estimation function with ``exclude_pre_periods`` parameter
- :doc:`../user_guide` - Comprehensive usage guide
