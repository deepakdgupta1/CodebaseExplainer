# Contributing to CodeHierarchy Explainer

We welcome contributions! Please follow these guidelines.

## Development Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dev dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Install pre-commit hooks (if applicable).

## Testing

Run the test suite with pytest:

```bash
pytest tests/
```

Ensure coverage remains high:

```bash
pytest --cov=src tests/
```

## Code Style

We use `black` for formatting and `mypy` for static type checking.

```bash
black src tests
mypy src
```

## Pull Requests

1. Fork the repo.
2. Create a feature branch.
3. Commit your changes.
4. Push and open a PR.
5. Ensure tests pass.
