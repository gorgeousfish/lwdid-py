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

.. note::

   In version 0.1.0, :meth:`lwdid.LWDIDResults.plot` has the signature
   ``plot(gid=None, graph_options=None)``. Customization options such as
   figure size, title, axis labels, legend location, DPI, and output
   filename are passed via the ``graph_options`` dictionary (with keys
   like ``'figsize'``, ``'title'``, ``'xlabel'``, ``'ylabel'``,
   ``'legend_loc'``, ``'dpi'``, ``'savefig'``). Examples in this
   document that show additional keyword arguments (for example,
   ``figsize=...``, ``title=...``, ``xlabel=...``, ``ylabel=...``,
   ``ax=...``, or color/marker options) are conceptual and may require
   adaptation: in version 0.1.0 the only arguments accepted by
   :meth:`lwdid.LWDIDResults.plot` are ``gid`` and ``graph_options``;
   options like figure size, labels, legend location, DPI, and output
   path must be supplied inside ``graph_options``, while styling such as
   colors, marker sizes, line widths, or the use of custom Matplotlib
   axes should be applied directly to the Matplotlib figure/axes
   returned by ``results.plot()``.

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

Event Study Plot
~~~~~~~~~~~~~~~~

Create an event study plot with time relative to treatment:

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

In version 0.1.0, the high-level :meth:`lwdid.LWDIDResults.plot` method
accepts two Python arguments, ``gid`` and ``graph_options``. The
plotting options supported via the ``graph_options`` dictionary are:

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

Other styling options (for example, colors, marker sizes, line widths,
or use of a pre-existing Matplotlib axes object) are not currently
controlled via ``plot()``/``graph_options`` and should instead be
applied directly to the Matplotlib figure and axes returned by
``results.plot()``.

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
