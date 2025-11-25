lwdid - Difference-in-Differences Estimator for Small Cross-Sectional Samples
=============================================================================

Python implementation of the Lee and Wooldridge (2025) difference-in-differences
method for panel data with small cross-sectional sample sizes.
Under the classical linear model assumptions with homoskedastic OLS standard
errors, it delivers exact t-based finite-sample inference. Heteroskedasticity-robust
and cluster-robust options are also available for large-sample approximations.

.. image:: https://img.shields.io/badge/python-3.8%2B-blue
   :target: https://www.python.org/
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-AGPL--3.0-blue
   :target: https://github.com/gorgeousfish/lwdid-py/blob/main/LICENSE
   :alt: License

Key Features
------------

- **Small-sample inference**: Designed for settings with small numbers of treated or control units
- **Design assumptions**: Common treatment timing with a binary, time-invariant treatment indicator and treatment persistence; staggered adoption designs are not supported in this version
- **Exact t-based inference**: Available under classical linear model assumptions (normality and homoskedasticity), works best with large time dimensions
- **Four transformation methods**: demean, detrend, demeanq, detrendq
- **Robust standard errors**: HC1/HC3 heteroskedasticity-robust and cluster-robust options
- **Randomization inference**: Bootstrap and permutation-based procedures for testing the sharp null hypothesis using Monte Carlo p-values
- **Period-specific effects**: Separate treatment effect estimates for each post-treatment period
- **Control variables**: Time-invariant covariates with automatic centering

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install lwdid

Basic Example
~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   from lwdid import lwdid

   # Load panel data
   data = pd.read_csv('smoking.csv')

   # Estimate ATT
   results = lwdid(
       data,
       y='lcigsale',      # outcome variable
       d='d',             # treatment indicator
       ivar='state',      # unit identifier
       tvar='year',       # time variable
       post='post',       # post-treatment indicator
       rolling='detrend', # transformation method
   )

   # View results
   print(results.summary())
   print(f"ATT: {results.att:.4f} (SE: {results.se_att:.4f})")

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

   changelog
   contributing

References
----------

Lee, S. J., and Wooldridge, J. M. (2025). Simple Approaches to Inference with
Difference-in-Differences Estimators with Small Cross-Sectional Sample Sizes.
*Available at* `SSRN 5325686 <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5325686>`_.

Authors
-------

Xuanyu Cai, Wenli Xu

Indices and Search
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
