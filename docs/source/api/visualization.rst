Visualization Module (visualization)
=====================================

The visualization module provides plotting functions for visualizing
difference-in-differences estimation results.

.. automodule:: lwdid.visualization
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

This module implements the plotting utilities used by the ``plot()`` method of
``LWDIDResults``, which create publication-quality visualizations of
transformed outcomes for treated and control groups over time.

**Key features:**

- Residualized outcome trajectories for treated and control groups
- Clear distinction between pre- and post-treatment periods
- Customizable appearance (figure size, titles, labels, colors)
- Automatic handling of time indices and period labels

**API Signature:**

The :meth:`lwdid.LWDIDResults.plot` method accepts two arguments:
``plot(gid=None, graph_options=None)``. Customization options such as
figure size, title, axis labels, legend location, DPI, and output
filename are passed via the ``graph_options`` dictionary (with keys
like ``'figsize'``, ``'title'``, ``'xlabel'``, ``'ylabel'``,
``'legend_loc'``, ``'dpi'``, ``'savefig'``). For advanced styling
(colors, marker sizes, line widths, or custom Matplotlib axes),
apply changes directly to the Matplotlib figure/axes returned by
``results.plot()``.

Plot Components
---------------

The standard plot produced by :meth:`lwdid.LWDIDResults.plot` includes:

1. **Residualized outcomes** over time for treated and control groups
2. **Treatment start indicator** (vertical line separating pre/post periods)
3. **Automatic period labels** on the x-axis based on the underlying time
   variables

Basic Usage
-----------

Simple Plot
~~~~~~~~~~~

.. code-block:: python

   from lwdid import lwdid
   import pandas as pd

   data = pd.read_csv('data.csv')
   results = lwdid(data, 'y', 'd', 'unit', 'year', 'post', 'demean')

   # Create plot
   results.plot()

This creates a default plot with:

- Figure size: (10, 6)
- Title: none by default (can be set via ``graph_options['title']``)
- X-axis label: none by default (can be set via ``graph_options['xlabel']``)
- Y-axis label: "Residualized Outcome" (default, can be overridden via
  ``graph_options['ylabel']``)

Customized Plot
~~~~~~~~~~~~~~~

.. code-block:: python

   results.plot(
       graph_options={
           'figsize': (12, 7),
           'title': 'Impact of Policy on Cigarette Sales',
           'xlabel': 'Year',
           'ylabel': 'Log Cigarette Sales (Difference)',
       }
   )

Advanced Customization
----------------------

Using Matplotlib Axes
~~~~~~~~~~~~~~~~~~~~~

For full control, use the Matplotlib axes object returned by ``results.plot()``:

.. code-block:: python

   import matplotlib.pyplot as plt

   fig = results.plot(graph_options={'figsize': (14, 8)})
   ax = fig.axes[0]

   # Further customization
   ax.set_facecolor('#f0f0f0')
   ax.grid(True, alpha=0.3, linestyle='--')
   ax.legend(['Pre-treatment', 'Post-treatment'], loc='upper left')

   plt.tight_layout()
   plt.savefig('treatment_effects.png', dpi=300)
   plt.show()

Multiple Specifications
~~~~~~~~~~~~~~~~~~~~~~~~

Compare results from different specifications:

.. code-block:: python

   import matplotlib.pyplot as plt

   # Run multiple specifications
   results_demean = lwdid(data, 'y', 'd', 'unit', 'year', 'post', 'demean')
   results_detrend = lwdid(data, 'y', 'd', 'unit', 'year', 'post', 'detrend')

   # Create separate plots for each specification
   fig1 = results_demean.plot(graph_options={'figsize': (8, 6), 'title': 'Demean Transformation'})
   fig2 = results_detrend.plot(graph_options={'figsize': (8, 6), 'title': 'Detrend Transformation'})

   plt.show()

Custom Styling
~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt

   # Set publication style
   plt.style.use('seaborn-v0_8-paper')

   fig = results.plot(
       graph_options={
           'figsize': (10, 6),
           'title': '',  # No title for publication
           'xlabel': 'Year',
           'ylabel': 'ATT (Log Points)',
       }
   )
   ax = fig.axes[0]

   # Add custom annotations
   ax.text(0.02, 0.98, 'Panel A: Main Results',
           transform=ax.transAxes, fontsize=12,
           verticalalignment='top')

   # Adjust layout
   plt.tight_layout()
   plt.savefig('figure1_panel_a.pdf', bbox_inches='tight')

Parallel Trends Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Highlight pre-treatment effects to assess parallel trends:

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np

   fig = results.plot(graph_options={'figsize': (12, 6)})
   ax = fig.axes[0]

   # Highlight pre-treatment region
   # Drop the 'average' row and create a numeric tindex column for convenience.
   period_effects = results.att_by_period.copy()
   period_effects = period_effects[period_effects['period'] != 'average'].copy()
   period_effects['tindex_num'] = period_effects['tindex'].astype(int)

   # Highlight pre-treatment region
   pre_periods = period_effects[period_effects['tindex_num'] < results.tpost1]

   # Add shaded region for pre-treatment
   ax.axvspan(pre_periods['tindex_num'].min() - 0.5,
              results.tpost1 - 0.5,
              alpha=0.1, color='gray', label='Pre-treatment')

   # Add text annotation
   ax.text(pre_periods['tindex_num'].mean(), ax.get_ylim()[1] * 0.9,
           'Parallel Trends Test',
           ha='center', fontsize=11, style='italic')

   ax.legend()
   plt.show()

Event Study Plot (Common Timing)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create an event study plot with time relative to treatment for common timing designs:

.. code-block:: python

   import matplotlib.pyplot as plt

   # Get period effects (drop the 'average' row and convert tindex to numeric)
   period_effects = results.att_by_period.copy()
   period_effects = period_effects[period_effects['period'] != 'average'].copy()
   period_effects['tindex_num'] = period_effects['tindex'].astype(int)

   # Create relative time variable
   period_effects['relative_time'] = period_effects['tindex_num'] - results.tpost1

   # Plot
   fig, ax = plt.subplots(figsize=(12, 6))

   # Pre-treatment
   pre = period_effects[period_effects['relative_time'] < 0]
   ax.errorbar(pre['relative_time'], pre['beta'],
               yerr=1.96*pre['se'], fmt='o-', capsize=5,
               label='Pre-treatment', color='blue')

   # Post-treatment
   post = period_effects[period_effects['relative_time'] >= 0]
   ax.errorbar(post['relative_time'], post['beta'],
               yerr=1.96*post['se'], fmt='s-', capsize=5,
               label='Post-treatment', color='red')

   # Reference lines
   ax.axhline(0, color='black', linestyle='--', alpha=0.5)
   ax.axvline(-0.5, color='gray', linestyle='--', alpha=0.5,
              label='Treatment start')

   ax.set_xlabel('Periods Relative to Treatment')
   ax.set_ylabel('Treatment Effect')
   ax.set_title('Event Study: Treatment Effects Over Time')
   ax.legend()
   ax.grid(alpha=0.3)

   plt.tight_layout()
   plt.show()

Staggered Adoption Visualization
--------------------------------

For staggered adoption designs (where units are treated at different times),
the ``plot_event_study()`` method provides specialized visualization of
dynamic treatment effects across cohorts. This method aggregates cohort-time
specific effects by event time (time relative to treatment) and produces
publication-quality event study diagrams.

Basic Event Study (Staggered)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lwdid import lwdid
   import pandas as pd

   data = pd.read_csv('castle.csv')

   # Estimate staggered effects
   results = lwdid(
       data,
       y='lhomicide',
       ivar='state',
       tvar='year',
       gvar='effyear',
       rolling='demean',
       aggregate='none'  # Required for event study plot
   )

   # Generate event study plot
   fig, ax = results.plot_event_study()

The event study plot displays:

- **X-axis**: Event time (periods relative to treatment, where :math:`e = r - g`)
- **Y-axis**: Treatment effect estimates
- **Points**: Average treatment effects at each event time
- **Error bars or shading**: 95% confidence intervals
- **Reference line**: Horizontal line at zero for visual assessment

Customized Event Study
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   fig, ax = results.plot_event_study(
       title='Castle Doctrine Effect on Homicide Rates',
       ylabel='Effect on Log Homicide Rate',
       xlabel='Years Relative to Adoption',
       include_pre_treatment=True,
       show_ci=True,
       ref_period=0,           # Reference period (normalized to zero if specified)
       aggregation='weighted'  # Weight by cohort size
   )

   fig.savefig('event_study.png', dpi=300, bbox_inches='tight')

Event Study Parameters
~~~~~~~~~~~~~~~~~~~~~~~

The :meth:`lwdid.LWDIDResults.plot_event_study` method accepts the following
parameters:

**Display parameters:**

- ``title``: Plot title (str, optional)
- ``xlabel``: X-axis label (str, default: ``'Event Time'``)
- ``ylabel``: Y-axis label (str, default: ``'Treatment Effect'``)
- ``figsize``: Figure size tuple (default: ``(10, 6)``)

**Data parameters:**

- ``include_pre_treatment``: Include pre-treatment periods (bool, default: ``True``)
- ``ref_period``: Reference period for normalization (int, optional)
- ``aggregation``: Cross-cohort aggregation method (``'mean'`` or ``'weighted'``,
  default: ``'weighted'``)

**Visual parameters:**

- ``show_ci``: Display confidence interval shading (bool, default: ``True``)
- ``ci_alpha``: Confidence interval shading opacity (float, default: ``0.2``)
- ``marker``: Marker style (str, default: ``'o'``)
- ``linestyle``: Line style (str, default: ``'-'``)

**Output parameters:**

- ``return_data``: Also return the aggregated event study DataFrame
  (bool, default: ``False``)

Returning Event Study Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To access the underlying data used in the plot:

.. code-block:: python

   fig, ax, event_df = results.plot_event_study(return_data=True)

   # event_df contains columns:
   # - event_time: periods relative to treatment (e = r - g)
   # - att: average treatment effect at this event time
   # - se: standard error
   # - ci_lower, ci_upper: confidence interval bounds
   # - n_cohorts: number of cohorts contributing to this event time

   print(event_df)

Cohort-Specific Event Studies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For detailed analysis, cohort-specific effects can be visualized manually:

.. code-block:: python

   import matplotlib.pyplot as plt

   # Access cohort-time specific effects
   gt_effects = results.att_by_cohort_time

   # Create event time variable
   gt_effects['event_time'] = gt_effects['period'] - gt_effects['cohort']

   # Plot by cohort
   fig, ax = plt.subplots(figsize=(12, 6))

   for cohort in gt_effects['cohort'].unique():
       cohort_data = gt_effects[gt_effects['cohort'] == cohort]
       ax.plot(cohort_data['event_time'], cohort_data['att'],
               marker='o', label=f'Cohort {cohort}', alpha=0.7)

   ax.axhline(0, color='black', linestyle='--', alpha=0.5)
   ax.axvline(-0.5, color='gray', linestyle='--', alpha=0.3)
   ax.set_xlabel('Event Time')
   ax.set_ylabel('Treatment Effect')
   ax.set_title('Cohort-Specific Treatment Effects')
   ax.legend(title='Treatment Cohort')
   ax.grid(alpha=0.3)

   plt.tight_layout()
   plt.show()

Publication-Ready Staggered Event Study
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt

   plt.style.use('seaborn-v0_8-whitegrid')

   fig, ax = results.plot_event_study(
       title='',
       xlabel='Years Since Policy Adoption',
       ylabel='Effect on Log Homicide Rate',
       figsize=(8, 5),
       show_ci=True,
       ci_alpha=0.15,
       include_pre_treatment=True
   )

   # Add reference line annotation
   ax.annotate('Policy Adoption', xy=(0, 0), xytext=(0.5, 0.05),
               arrowprops=dict(arrowstyle='->', color='gray'),
               fontsize=10, color='gray')

   plt.tight_layout()
   plt.savefig('figure_event_study.pdf', bbox_inches='tight')

Exporting Plots
---------------

Save to File
~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt

   results.plot()

   # Save in multiple formats
   plt.savefig('treatment_effects.png', dpi=300, bbox_inches='tight')
   plt.savefig('treatment_effects.pdf', bbox_inches='tight')
   plt.savefig('treatment_effects.svg', bbox_inches='tight')

High-Resolution for Publication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt

   # Set high DPI
   plt.rcParams['figure.dpi'] = 300
   plt.rcParams['savefig.dpi'] = 300

   results.plot(graph_options={'figsize': (8, 5)})

   plt.savefig('figure1.tiff', dpi=600, bbox_inches='tight')

Plot Parameters Reference
-------------------------

The :meth:`lwdid.LWDIDResults.plot` method accepts two arguments:
``gid`` and ``graph_options``. The plotting options supported via
the ``graph_options`` dictionary are:

**Figure parameters (graph_options keys):**

- ``'figsize'``: Tuple (width, height) in inches, default: ``(10, 6)``
- ``'dpi'``: Figure DPI, default: ``100``

**Labels (graph_options keys):**

- ``'title'``: Plot title, default: ``None`` (no title unless specified)
- ``'xlabel'``: X-axis label, default: ``None`` (no label unless specified)
- ``'ylabel'``: Y-axis label, default: ``"Residualized Outcome"``

**Legend and output (graph_options keys):**

- ``'legend_loc'``: Legend location, default: ``'best'``
- ``'savefig'``: File path to save the figure (no file is saved if this
  key is omitted)

Additional styling options (colors, marker sizes, line widths, or
custom Matplotlib axes) should be applied directly to the Matplotlib
figure and axes returned by ``results.plot()``.

Examples Gallery
----------------

Minimal Plot
~~~~~~~~~~~~

.. code-block:: python

   results.plot()

Publication-Ready Plot
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt

   plt.style.use('seaborn-v0_8-whitegrid')

   results.plot(
       graph_options={
           'figsize': (8, 5),
           'title': '',
           'xlabel': 'Year',
           'ylabel': 'Treatment Effect (Log Points)',
       }
   )

   plt.tight_layout()
   plt.savefig('figure1.pdf', bbox_inches='tight')

Presentation Plot
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt

   plt.rcParams.update({'font.size': 14})

   results.plot(
       graph_options={
           'figsize': (14, 8),
           'title': 'Impact of Policy Intervention on Outcomes',
           'xlabel': 'Year',
           'ylabel': 'Effect Size',
       }
   )

   plt.tight_layout()
   plt.savefig('presentation_slide.png', dpi=150)

See Also
--------

- :class:`lwdid.LWDIDResults` - Results object with plot() method
- :func:`lwdid.lwdid` - Main estimation function
- :doc:`../examples/index` - Complete examples with visualizations
