Developer Guide
===============

Welcome to the Burbuja developer guide! This page is intended for users and contributors who wish to help improve Burbuja, add new features, fix bugs, or participate in the community.

.. contents::
	:local:
	:depth: 2

Getting Involved
----------------

The main hub for Burbuja development is the GitHub repository:

  https://github.com/Abrahammc90/Burbuja.git

Here you can:

- Find the latest source code
- Report bugs and request features (see the Issues tab)
- Participate in discussions
- Submit pull requests (PRs) to contribute code or documentation

How to Contribute
-----------------

1. **Fork the repository** on GitHub and clone your fork locally.
2. **Create a new branch** for your feature or bugfix:

	.. code-block:: bash

		git checkout -b my-feature-branch

3. **Make your changes** (code, tests, or docs). Please follow the code style guidelines below.
4. **Test your changes** locally. Add or update tests as needed.
5. **Commit and push** your branch to your fork.
6. **Open a pull request** (PR) on GitHub. Describe your changes clearly and reference any related issues.
7. Participate in code review and address any feedback.

Code Style and Best Practices
-----------------------------

- Follow PEP8 for Python code style.
- Use clear, descriptive variable and function names.
- Write docstrings for all public functions, classes, and modules.
- Add comments to explain complex logic.
- Write tests for new features and bugfixes (see the `tests/` directory for examples).
- Keep pull requests focused and concise.

Testing
-------

Burbuja uses `pytest` for testing. To run the test suite:

.. code-block:: bash

	pytest

Tests are located in the `Burbuja/tests/` directory. Please add tests for any new features or bugfixes.

Reporting Issues and Requesting Features
----------------------------------------

- Use the [GitHub Issues page](https://github.com/Abrahammc90/Burbuja/issues) to report bugs, request features, or ask questions.
- Please provide as much detail as possible, including steps to reproduce bugs, error messages, and your environment (OS, Python version, etc).

Discussions and Community
-------------------------

- Use the [GitHub Discussions page](https://github.com/Abrahammc90/Burbuja/discussions) for general questions, ideas, and community support.
- You can also reach out via issues or pull requests for technical questions.

Documentation
-------------

- Documentation is written in reStructuredText (reST) and located in the `docs/` directory.
- To build the documentation locally:

  .. code-block:: bash

	  cd docs
	  make html

- Please update or add documentation for any new features or changes.

Acknowledgements
----------------

Burbuja is an open-source project and welcomes contributions from the community. Thank you for helping to make Burbuja better!