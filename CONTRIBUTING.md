# Contributing to lwdid

Thank you for your interest in contributing to **lwdid**! We welcome contributions from the community and are grateful for any help you can provide.

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to F23090105414@cityu.edu.mo.

## How to Contribute

There are many ways to contribute to lwdid:

### Reporting Bugs

If you find a bug, please open an issue on our [GitHub Issues](https://github.com/gorgeousfish/lwdid-py/issues) page. When reporting a bug, please include:

- A clear and descriptive title
- A detailed description of the issue
- Steps to reproduce the behavior
- Expected behavior vs. actual behavior
- Your environment information (Python version, OS, package version)
- Any relevant code snippets or error messages

### Suggesting Enhancements

We welcome suggestions for new features or improvements. Please open an issue with:

- A clear and descriptive title
- A detailed description of the proposed enhancement
- The motivation and use case for the feature
- Any relevant examples or references

### Contributing Code

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/lwdid-py.git
   cd lwdid-py
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes** and commit them with clear messages
5. **Push to your fork** and submit a Pull Request

## Development Setup

### Prerequisites

- Python 3.8 or higher (up to 3.12)
- Git

### Installation for Development

1. Clone the repository:
   ```bash
   git clone https://github.com/gorgeousfish/lwdid-py.git
   cd lwdid-py
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package in development mode with dev dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

### Running Tests

We use pytest for testing. Run the test suite with:

```bash
pytest tests/
```

To run tests with coverage:

```bash
pytest tests/ --cov=src/lwdid --cov-report=html
```

### Building Documentation

Documentation is built using Sphinx. To build locally:

```bash
cd docs
pip install -r requirements.txt
make html
```

The built documentation will be in `docs/build/html/`.

## Code Style Guidelines

### Python Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines
- Use meaningful variable and function names
- Keep functions focused and reasonably sized
- Use type hints where appropriate

### Docstrings

We use NumPy-style docstrings. Every public function, class, and method should have a docstring that includes:

- A brief description
- Parameters with types and descriptions
- Return values with types and descriptions
- Examples (when helpful)

Example:

```python
def estimate_att(data, y, d, ivar, tvar):
    """
    Estimate the average treatment effect on the treated.

    Parameters
    ----------
    data : pandas.DataFrame
        Panel data in long format.
    y : str
        Name of the outcome variable column.
    d : str
        Name of the treatment indicator column.
    ivar : str
        Name of the unit identifier column.
    tvar : str
        Name of the time variable column.

    Returns
    -------
    float
        The estimated ATT.

    Examples
    --------
    >>> result = estimate_att(df, 'outcome', 'treated', 'unit', 'year')
    """
```

### Testing Requirements

- All new features should include tests
- Bug fixes should include a test that would have caught the bug
- Aim for high test coverage on new code
- Tests should be clear and well-documented

## Pull Request Process

1. **Update documentation**: If your changes affect the public API, update the relevant documentation.

2. **Add tests**: Ensure your changes are covered by tests.

3. **Run the test suite**: Make sure all tests pass before submitting:
   ```bash
   pytest tests/
   ```

4. **Write a good PR description**: Include:
   - What changes you made and why
   - Any relevant issue numbers (e.g., "Fixes #123")
   - Any breaking changes or migration notes

5. **Be responsive**: Be prepared to address feedback and make changes if requested.

### PR Checklist

- [ ] Code follows the project's style guidelines
- [ ] Tests have been added/updated as needed
- [ ] Documentation has been updated as needed
- [ ] All tests pass locally
- [ ] Commit messages are clear and descriptive

## Questions and Discussion

- For questions about using lwdid, please open a [GitHub Discussion](https://github.com/gorgeousfish/lwdid-py/discussions) or issue
- For bug reports and feature requests, use [GitHub Issues](https://github.com/gorgeousfish/lwdid-py/issues)

## Recognition

Contributors will be acknowledged in our release notes. We appreciate all contributions, whether they are code, documentation, bug reports, or feature suggestions.

## License

By contributing to lwdid, you agree that your contributions will be licensed under the [AGPL-3.0 License](LICENSE).

---

Thank you for contributing to lwdid! Your efforts help make this package better for everyone in the research community.
