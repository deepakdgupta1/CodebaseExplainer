# Testing Guide

## Quick Start

Run the complete test suite with a single command:

```bash
# From project root
python tests/run_tests.py

# Or make it executable and run directly
chmod +x tests/run_tests.py
./tests/run_tests.py
```

## What Gets Tested

The test runner executes three main test suites:

1. **PyTest** - Unit and integration tests with coverage
2. **Flake8** - Code style and PEP 8 compliance
3. **MyPy** - Static type checking

## Output Structure

All test results are saved in the `test_output/` directory:

```
test_output/
├── run_20231209_143022/          # Timestamped run directory
│   ├── test_results.txt          # Detailed pytest output
│   ├── coverage_report.txt       # Coverage summary
│   ├── htmlcov/                  # HTML coverage report
│   │   └── index.html           # Open this in browser
│   ├── junit.xml                 # JUnit XML for CI/CD
│   ├── flake8_report.txt         # Flake8 linting results
│   ├── mypy_report.txt           # MyPy type checking results
│   └── summary.json              # Machine-readable summary
└── latest -> run_20231209_143022/  # Symlink to latest run
```

## Viewing Results

### Quick Summary
The script prints a summary at the end showing pass/fail for each test suite.

### Detailed Results
- **Test Results**: `test_output/latest/test_results.txt`
- **Coverage Report**: Open `test_output/latest/htmlcov/index.html` in a browser
- **Linting Issues**: `test_output/latest/flake8_report.txt`
- **Type Errors**: `test_output/latest/mypy_report.txt`

### JSON Summary
For programmatic access:
```bash
cat test_output/latest/summary.json
```

## Running Individual Test Suites

If you only want to run specific tests:

### PyTest Only
```bash
pytest tests/ -v --cov=src/codehierarchy --cov-report=html
```

### Flake8 Only
```bash
flake8 src/ tests/ --count --statistics
```

### MyPy Only
```bash
mypy src/ tests/
```

## CI/CD Integration

The test runner generates a JUnit XML file (`junit.xml`) compatible with most CI/CD systems:

```yaml
# Example GitHub Actions
- name: Run Tests
  run: python tests/run_tests.py
  
- name: Upload Test Results
  uses: actions/upload-artifact@v3
  with:
    name: test-results
    path: test_output/latest/
```

## Troubleshooting

### Missing Dependencies
If you get import errors, ensure all dev dependencies are installed:
```bash
pip install -e ".[dev]"
```

### Permission Denied
Make the script executable:
```bash
chmod +x tests/run_tests.py
```

### Tests Timeout
The default timeout is 300 seconds (5 minutes). If tests take longer, modify the `timeout` parameter in `run_tests.py`.

## Exit Codes

- `0`: All tests passed
- `1`: One or more test suites failed

This makes it easy to use in scripts:
```bash
if python tests/run_tests.py; then
    echo "All tests passed!"
else
    echo "Tests failed, check test_output/latest/"
fi
```
