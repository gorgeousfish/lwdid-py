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
   git clone https://github.com/gorgeousfish/lwdid-python.git
   cd lwdid-python

   # Install in development mode
   pip install -e .

Development Environment Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you plan to contribute to development, install development dependencies:

.. code-block:: bash

   # Install with test dependencies
   pip install -e ".[dev]"

   # Or manually install development tools
   pip install pytest pytest-cov

Verify Installation
-------------------

After installation, verify by:

.. code-block:: python

   import lwdid
   from lwdid import lwdid as lwdid_func

   # Test basic import
   print("lwdid package imported successfully")

Or run the test suite:

.. code-block:: bash

   pytest tests/

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

Virtual Environment (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is recommended to install in a virtual environment:

.. code-block:: bash

   # Create virtual environment
   python -m venv lwdid_env

   # Activate virtual environment
   # macOS/Linux:
   source lwdid_env/bin/activate
   # Windows:
   lwdid_env\Scripts\activate

   # Install lwdid
   pip install lwdid

Next Steps
----------

After installation, see :doc:`quickstart` to get started.
