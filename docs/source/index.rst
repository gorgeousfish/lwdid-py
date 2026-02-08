lwdid - Difference-in-Differences with Rolling Transformations
==============================================================

Python implementation of the Lee and Wooldridge difference-in-differences
methods for panel data, supporting both small cross-sectional sample sizes
with exact inference and large-sample settings with asymptotic inference.

The package implements a simple transformation approach that converts panel
DiD estimation into cross-sectional treatment effects problems, enabling
various estimators including regression adjustment, inverse probability
weighting, doubly robust methods, and propensity score matching.

.. image:: https://img.shields.io/badge/python-3.8%2B-blue
   :target: https://www.python.org/
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-AGPL--3.0-blue
   :target: https://github.com/gorgeousfish/lwdid-py/blob/main/LICENSE
   :alt: License

Supported Scenarios
-------------------

**Small-Sample Common Timing** (Lee and Wooldridge, 2026)

For settings with small numbers of treated or control units and common
treatment timing. Under classical linear model assumptions (normality and
homoskedasticity), exact t-based finite-sample inference is available.

**Large-Sample Common Timing** (Lee and Wooldridge, 2025)

For settings with larger cross-sectional samples. Supports heteroskedasticity-
robust (HC0-HC4) and cluster-robust standard errors with asymptotic inference.

**Staggered Adoption** (Lee and Wooldridge, 2025)

For settings where units are treated at different times. Estimates cohort-time
specific ATTs with flexible control group strategies (never-treated or not-yet-
treated units) and multiple aggregation options.

Key Features
------------

- **Small-sample inference**: Exact t-based inference under CLM assumptions
- **Large-sample inference**: HC0-HC4 heteroskedasticity-robust and cluster-robust options
- **Staggered adoption**: Full support for staggered treatment timing with cohort-time effects
- **Four transformation methods**: demean, detrend, demeanq, detrendq
- **Multiple estimators**: Regression adjustment (RA), IPW, IPWRA, PSM
- **Randomization inference**: Bootstrap and permutation-based procedures
- **Period-specific effects**: Separate ATT estimates for each post-treatment period
- **Event study visualization**: Built-in plotting for staggered designs
- **Control variables**: Time-invariant covariates with automatic centering

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install lwdid

Common Timing Example
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   from lwdid import lwdid

   # Load panel data
   data = pd.read_csv('smoking.csv')

   # Estimate ATT with small-sample inference
   results = lwdid(
       data,
       y='lcigsale',      # outcome variable
       d='d',             # treatment indicator
       ivar='state',      # unit identifier
       tvar='year',       # time variable
       post='post',       # post-treatment indicator
       rolling='detrend', # transformation method
   )

   print(results.summary())

Staggered Adoption Example
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Load staggered adoption data
   data = pd.read_csv('castle.csv')

   # Estimate with staggered design
   results = lwdid(
       data,
       y='l_homicide',
       ivar='state',
       tvar='year',
       gvar='effyear',        # first treatment period
       rolling='demean',
       aggregate='overall',   # aggregate to overall effect
       control_group='not_yet_treated',
   )

   print(results.summary())
   results.plot_event_study()

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   user_guide
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Advanced Topics

   methodological_notes

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing

References
----------

Lee, S. J., and Wooldridge, J. M. (2026). Simple Approaches to Inference with
Difference-in-Differences Estimators with Small Cross-Sectional Sample Sizes.
*Available at SSRN 5325686*.

Lee, S. J., and Wooldridge, J. M. (2025). A Simple Transformation Approach to
Difference-in-Differences Estimation for Panel Data.
*Available at SSRN 4516518*.

Indices and Search
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
