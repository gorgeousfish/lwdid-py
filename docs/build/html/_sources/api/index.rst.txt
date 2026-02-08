API Reference
=============

This section provides complete API documentation for the lwdid package,
automatically generated from source code docstrings.

Core Modules
------------

.. toctree::
   :maxdepth: 2

   core
   results
   staggered
   transformations
   estimation

Inference and Diagnostics
-------------------------

.. toctree::
   :maxdepth: 2

   inference
   clustering_diagnostics
   trend_diagnostics
   selection_diagnostics
   sensitivity

Data Processing
---------------

.. toctree::
   :maxdepth: 2

   preprocessing
   validation
   randomization
   visualization
   exceptions

Quick Index
-----------

**Main API:**

- :func:`lwdid.lwdid` - Main estimation function supporting three scenarios:

  - Small-sample common timing with exact t-based inference (Lee and Wooldridge, 2026)
  - Large-sample common timing with robust standard errors (Lee and Wooldridge, 2025)
  - Staggered adoption with cohort-time effects (Lee and Wooldridge, 2025)

- :class:`lwdid.LWDIDResults` - Results container class

**Diagnostics:**

- :mod:`lwdid.clustering_diagnostics` - Clustering analysis and recommendations
- :mod:`lwdid.trend_diagnostics` - Parallel trends testing and heterogeneous trends
- :mod:`lwdid.selection_diagnostics` - Selection mechanism diagnostics

**Inference:**

- :mod:`lwdid.inference` - Wild cluster bootstrap inference
- :mod:`lwdid.sensitivity` - Sensitivity analysis

**Data Processing:**

- :mod:`lwdid.preprocessing` - Repeated cross-section aggregation
- :mod:`lwdid.staggered` - Staggered DiD module
- :mod:`lwdid.exceptions` - Exception hierarchy

See individual module pages for detailed documentation.
