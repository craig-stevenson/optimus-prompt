# Contributing to Optimus Prompt

First off, thank you for considering contributing to Optimus Prompt! It's people like you that make Optimus Prompt such a great tool.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* Use a clear and descriptive title
* Describe the exact steps which reproduce the problem
* Provide specific examples to demonstrate the steps
* Describe the behavior you observed after following the steps
* Explain which behavior you expected to see instead and why
* Include any error messages or logs

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* Use a clear and descriptive title
* Provide a step-by-step description of the suggested enhancement
* Provide specific examples to demonstrate the steps
* Describe the current behavior and explain which behavior you expected to see instead
* Explain why this enhancement would be useful

### Adding New LLM Providers

To add support for a new LLM provider:

1. Create a new file in `optimus_prompt/providers/`
2. Implement the provider interface defined in `base_provider.py`
3. Add appropriate tests in `tests/test_providers/`
4. Update documentation to reflect the new provider
5. Submit a pull request

## Development Process

1. Fork the repo
2. Create a branch from `main`
3. Make your changes
4. Run the test suite
5. Update documentation if needed
6. Submit a pull request

### Setting Up Development Environment

```bash
# Clone your fork
git clone https://github.com/your-username/optimus-prompt.git
cd optimus-prompt

# Install poetry if you haven't already
pip install poetry

# Install dependencies
poetry install

# Run tests
poetry run pytest
```

### Code Style

We use several tools to maintain code quality:

* `black` for code formatting
* `isort` for import sorting
* `mypy` for type checking
* `ruff` for linting

Before submitting a pull request, please run:

```bash
poetry run black .
poetry run isort .
poetry run mypy .
poetry run ruff .
```

### Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line

## Pull Request Process

1. Update the README.md with details of changes to the interface
2. Update the documentation with details of any new providers or features
3. The PR will be merged once you have the sign-off of at least one maintainer

## Questions?

Feel free to create an issue with the "question" label if you have any questions about contributing.

Thank you for your contributions! ❤️