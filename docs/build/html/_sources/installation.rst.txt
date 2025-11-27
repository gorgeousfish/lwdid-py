Installation Guide
==================

System Requirements
-------------------

**Python Version**

- Python 3.8 or higher
- Python 3.10 or 3.11 recommended

**Dependencies**

lwdid requires the following Python packages:

- numpy >= 1.20
- pandas >= 1.3
- scipy >= 1.7
- statsmodels >= 0.13
- matplotlib >= 3.3
- openpyxl >= 3.1

Installation Methods
--------------------

Install via pip (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest way to install is via pip:

.. code-block:: bash

   pip install lwdid

Install from Source
~~~~~~~~~~~~~~~~~~~

If you want to install the development version or contribute code, clone the repository from GitHub:

.. code-block:: bash

   # Clone repository
   git clone https://github.com/gorgeousfish/lwdid-py.git
   cd lwdid-py

   # Install in development mode
   pip install -e .

Common Issues
-------------

numpy Version Conflicts
~~~~~~~~~~~~~~~~~~~~~~~

If you encounter numpy version conflicts, try:

.. code-block:: bash

   pip install --upgrade numpy pandas

statsmodels Installation Failure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On some systems, statsmodels may require additional dependencies:

.. code-block:: bash

   # macOS
   brew install gcc

   # Ubuntu/Debian
   sudo apt-get install build-essential

Next Steps
----------

After installation, see :doc:`quickstart` to get started.
