# Tests for Options Analysis Scripts

Comprehensive unit tests for the stock data generator.

## Test Structure

```
tests/
├── __init__.py                      # Test package init
├── conftest.py                      # Shared fixtures
├── test_stock_data_generator.py     # Basic functionality tests
├── test_controllable_inputs.py      # Controllable inputs tests
└── test_edge_cases.py               # Edge cases and validation tests
```

## Test Coverage

### `test_stock_data_generator.py` (19 test classes, 36 tests)

**TestBasicInitialization**
- Default and custom initialization
- Seed reproducibility
- Different seeds produce different results

**TestDataGeneration**
- DataFrame generation
- Correct number of rows
- Required columns present
- No weekends in data
- Sequential dates

**TestPriceDataValidation**
- Positive prices
- High/Low relationships
- Volume validation

**TestFinancialMetrics**
- Market cap calculation
- P/E ratio calculation
- Drawdown properties
- Sentiment range
- Earnings frequency

**TestStatisticalProperties**
- Reasonable mean returns
- Reasonable volatility
- Volume increases on earnings

**TestSaveAndExport**
- CSV export
- Summary statistics

**TestRegimes**
- Regime definitions
- Regime parameters
- Regime transitions

**TestValuationDrift**
- High PEG drift adjustment
- Low PEG drift adjustment

### `test_controllable_inputs.py` (8 test classes, 28 tests)

**TestTargetPrices**
- Add single/multiple targets
- Target sorting
- Price guided toward target
- Target-guided drift calculation

**TestNewsEvents**
- Add news events
- Sentiment clipping
- News affects sentiment
- News increases volatility
- Multiple events
- Event sorting

**TestVolatilityScheduling**
- Set volatility schedule
- Schedule sorting
- Scheduled volatility is used
- Get scheduled volatility

**TestRegimeScheduling**
- Set regime schedule
- Invalid regime error
- Schedule sorting
- Scheduled regime is used
- Get scheduled regime

**TestClearSchedules**
- Clear all schedules

**TestCombinedSchedules**
- Targets and news together
- All schedules together

### `test_edge_cases.py` (8 test classes, 35 tests)

**TestEdgeCases**
- Minimum days (1 day)
- Large dataset (10 years)
- Extreme prices (very low/high)
- Zero EPS handling
- Negative growth
- Extreme volatility (high/low)
- Zero/perfect correlation
- Zero/negative beta

**TestStartDateHandling**
- Weekend start dates
- Past dates
- Future dates

**TestEarningsHandling**
- Very frequent earnings
- Very infrequent earnings
- No earnings in short period

**TestDataIntegrity**
- No missing values
- No infinite values
- Volume is integer
- Consistent market cap

**TestReproducibility**
- Same seed same output
- Different start dates

**TestParameterValidation**
- Invalid regime error
- Negative shares

**TestExtremeScenarios**
- Continuous bear market
- Continuous bull market
- Extreme volatility spike
- Multiple targets same day

## Running Tests

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run All Tests

```bash
# From project root
pytest tests/

# With verbose output
pytest tests/ -v

# With coverage report
pytest tests/ --cov=data_generator --cov-report=html
```

### Run Specific Test File

```bash
pytest tests/test_stock_data_generator.py -v
pytest tests/test_controllable_inputs.py -v
pytest tests/test_edge_cases.py -v
```

### Run Specific Test Class

```bash
pytest tests/test_stock_data_generator.py::TestBasicInitialization -v
pytest tests/test_controllable_inputs.py::TestTargetPrices -v
```

### Run Specific Test

```bash
pytest tests/test_stock_data_generator.py::TestBasicInitialization::test_seed_reproducibility -v
```

### Run Tests Matching Pattern

```bash
# Run all tests with "target" in name
pytest tests/ -k "target" -v

# Run all tests with "sentiment" or "news"
pytest tests/ -k "sentiment or news" -v
```

## Test Fixtures

Shared fixtures are defined in `conftest.py`:

- `basic_generator` - StockDataGenerator with default test settings
- `start_date` - Fixed start date (2025-01-01)
- `small_dataset_days` - 21 trading days
- `medium_dataset_days` - 63 trading days
- `full_year_days` - 252 trading days

## Writing New Tests

### Test Structure

```python
import pytest
from data_generator import StockDataGenerator

class TestNewFeature:
    """Test description."""

    def test_specific_behavior(self, basic_generator):
        """Test that specific behavior works correctly."""
        # Arrange
        basic_generator.some_setting = value

        # Act
        data = basic_generator.generate(n_days=21)

        # Assert
        assert condition
```

### Best Practices

1. **Use descriptive test names** - Test names should describe what they test
2. **One assertion per test** - Or at least one logical concept
3. **Use fixtures** - Leverage conftest.py fixtures for common setup
4. **Test edge cases** - Include boundary conditions and error cases
5. **Use parametrize** - For testing multiple scenarios:

```python
@pytest.mark.parametrize("volatility,expected_range", [
    (0.10, (0.08, 0.12)),
    (0.30, (0.25, 0.35)),
])
def test_volatility_levels(basic_generator, volatility, expected_range):
    basic_generator.base_volatility = volatility
    data = basic_generator.generate(n_days=63)
    assert expected_range[0] <= data['Volatility'].mean() <= expected_range[1]
```

## Coverage Goals

Current coverage: ~99 tests covering:
- ✅ Basic initialization and configuration
- ✅ Data generation and validation
- ✅ Price data integrity
- ✅ Financial metrics calculation
- ✅ Statistical properties
- ✅ Target price scheduling
- ✅ News event scheduling
- ✅ Volatility scheduling
- ✅ Regime scheduling
- ✅ Combined controllable inputs
- ✅ Edge cases and boundary conditions
- ✅ Error handling
- ✅ Data export

## Continuous Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install -r requirements.txt
    pytest tests/ --cov=data_generator --cov-report=xml
```

## Test Performance

- Full test suite runs in ~10-30 seconds
- Individual test files run in ~3-10 seconds
- Use `-n auto` with pytest-xdist for parallel execution:

```bash
pip install pytest-xdist
pytest tests/ -n auto
```

## Troubleshooting

### Tests Fail with "ModuleNotFoundError"

Make sure you're running from the project root:
```bash
cd /path/to/Options-analysis-scripts
pytest tests/
```

### Random Failures Due to Stochastic Behavior

Some tests may occasionally fail due to the stochastic nature of the generator. Tests use:
- Fixed seeds for reproducibility
- Wide tolerances for statistical tests
- Multiple samples where appropriate

If a test fails intermittently, it may need wider tolerances or more samples.

### Slow Tests

To run only fast tests:
```bash
pytest tests/ -m "not slow"
```

(After marking slow tests with `@pytest.mark.slow`)
