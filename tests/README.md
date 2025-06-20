# MLOps Testing Framework

This directory contains the comprehensive testing framework for the MLOps pipeline, designed to ensure code reliability, maintainability, and quality throughout the development lifecycle.

## 🏗️ Testing Architecture

The testing framework follows a multi-layered approach with the following components:

### Test Categories

- **Unit Tests** (`@pytest.mark.unit`): Test individual functions and components in isolation
- **Integration Tests** (`@pytest.mark.integration`): Test end-to-end workflows and component interactions
- **API Tests** (`@pytest.mark.api`): Test API endpoints and HTTP interactions
- **Data Tests** (`@pytest.mark.data`): Test data processing and validation
- **Model Tests** (`@pytest.mark.model`): Test model training and evaluation

### Test Organization

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── run_tests.py             # Test runner script
├── README.md               # This file
├── test_data_loader.py     # Data loading tests
├── test_data_validator.py  # Data validation tests
├── test_preprocessing.py   # Preprocessing tests
├── test_features.py        # Feature engineering tests
├── test_model.py           # Model training tests
├── test_evaluation.py      # Model evaluation tests
├── test_inference.py       # Inference tests
└── test_legacy_main.py     # Legacy main function tests
```

## 🚀 Quick Start

### Running Tests

```bash
# Run all tests
python tests/run_tests.py

# Run unit tests only
python tests/run_tests.py --unit

# Run integration tests only
python tests/run_tests.py --integration

# Run tests with coverage
python tests/run_tests.py --coverage

# Run specific module
python tests/run_tests.py --module test_data_loader

# Run code quality checks
python tests/run_tests.py --quality
```

### Direct pytest Commands

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/ -m unit
pytest tests/ -m integration
pytest tests/ -m data

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_data_loader.py

# Run specific test function
pytest tests/test_data_loader.py::TestDataLoading::test_load_data_source_csv_success
```

## 📋 Test Framework Features

### 1. Test Isolation

- **Path Independence**: Tests use relative paths and temporary directories
- **Configuration Isolation**: Each test has its own configuration fixtures
- **Dependency Mocking**: External dependencies are mocked using `monkeypatch`
- **Data Isolation**: Mock data is generated for each test

### 2. Mock Data Management

The framework provides comprehensive mock data fixtures:

```python
@pytest.fixture
def mock_cancer_data():
    """Create mock cancer dataset for testing."""
    return pd.DataFrame({
        "patient_id": range(1, 101),
        "age": [25 + i % 50 for i in range(100)],
        "gender": ["M" if i % 2 == 0 else "F" for i in range(100)],
        "diagnosis": ["positive" if i % 3 == 0 else "negative" for i in range(100)],
        # ... more features
    })
```

### 3. Test Configuration

Tests use inline configurations to avoid dependency on production config files:

```python
CSV_CONFIG = {
    "raw_path": "mock_data.csv",
    "type": "csv",
    "header": 0,
    "encoding": "utf-8",
}
```

### 4. Error Handling Validation

Tests explicitly validate error conditions:

```python
def test_load_data_source_missing_file(self, tmp_path):
    """Test load_data_source raises DataLoaderError when the file is missing."""
    ds_cfg = CSV_CONFIG.copy()
    ds_cfg["raw_path"] = str(tmp_path / "no.csv")
    
    with pytest.raises(DataLoaderError) as excinfo:
        load_data_source(ds_cfg)
    assert "Data file not found" in str(excinfo.value)
```

## 🔧 Test Development Guidelines

### Writing New Tests

1. **Use Test Classes**: Organize related tests into classes
2. **Add Markers**: Use appropriate pytest markers (`@pytest.mark.unit`, etc.)
3. **Use Fixtures**: Leverage shared fixtures from `conftest.py`
4. **Test Isolation**: Ensure tests don't depend on each other
5. **Mock External Dependencies**: Use `monkeypatch` for external calls

### Example Test Structure

```python
class TestNewFeature:
    """Test new feature functionality."""
    
    @pytest.mark.unit
    def test_feature_success(self, mock_data):
        """Test successful feature execution."""
        result = new_feature(mock_data)
        assert result is not None
        assert len(result) > 0
    
    @pytest.mark.unit
    def test_feature_error_handling(self, mock_data):
        """Test feature error handling."""
        with pytest.raises(ValueError):
            new_feature(None)
    
    @pytest.mark.integration
    def test_feature_integration(self, mock_config):
        """Test feature integration with other components."""
        # Integration test logic
        pass
```

### Test Naming Conventions

- Test functions: `test_<function_name>_<scenario>`
- Test classes: `Test<ComponentName>`
- Test files: `test_<module_name>.py`

## 📊 Coverage Requirements

The testing framework enforces minimum coverage requirements:

- **Overall Coverage**: 80% minimum
- **Critical Paths**: 90% minimum for core pipeline components
- **Error Handling**: 100% coverage for error conditions

### Coverage Reports

Coverage reports are generated in multiple formats:

```bash
# HTML report (view in browser)
pytest tests/ --cov=src --cov-report=html

# Terminal report
pytest tests/ --cov=src --cov-report=term-missing

# XML report (for CI/CD)
pytest tests/ --cov=src --cov-report=xml
```

## 🛠️ Code Quality Tools

The framework integrates with code quality tools:

### Black (Code Formatter)
```bash
# Check formatting
black --check src/ tests/

# Format code
black src/ tests/
```

### Flake8 (Linter)
```bash
# Run linting
flake8 src/ tests/
```

### Running Quality Checks
```bash
python tests/run_tests.py --quality
```

## 🔄 Continuous Integration

The testing framework is designed to integrate with CI/CD pipelines:

### GitHub Actions Example
```yaml
- name: Run Tests
  run: |
    python tests/run_tests.py --coverage
    
- name: Run Quality Checks
  run: |
    python tests/run_tests.py --quality
```

### Pre-commit Hooks
```yaml
repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: python tests/run_tests.py --unit
        language: system
        pass_filenames: false
```

## 📈 Test Metrics

The framework provides test metrics and reporting:

- **Test Execution Time**: Track slow tests with `@pytest.mark.slow`
- **Test Categories**: Analyze test distribution across categories
- **Coverage Trends**: Monitor coverage over time
- **Failure Analysis**: Identify common failure patterns

## 🐛 Debugging Tests

### Common Issues

1. **Import Errors**: Ensure `src/` is in Python path
2. **Path Issues**: Use `tmp_path` fixture for file operations
3. **Mock Data**: Use fixtures for consistent test data
4. **Environment Variables**: Set test environment in `conftest.py`

### Debug Commands

```bash
# Run with verbose output
pytest tests/ -v -s

# Run single test with debugger
pytest tests/test_data_loader.py::TestDataLoading::test_load_data_source_csv_success -s

# Run with print statements
pytest tests/ -s

# Run with maximum verbosity
pytest tests/ -vvv
```

## 📚 Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [Black Documentation](https://black.readthedocs.io/)
- [Flake8 Documentation](https://flake8.pycqa.org/)

## 🤝 Contributing

When adding new tests:

1. Follow the existing test structure and patterns
2. Add appropriate markers and documentation
3. Ensure test isolation and independence
4. Update this README if adding new test categories or tools
5. Run the full test suite before submitting changes

---

**Note**: This testing framework is designed to be human-readable, maintainable, and comprehensive. All tests should be self-documenting and follow consistent patterns for easy understanding and maintenance. 