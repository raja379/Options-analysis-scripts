"""
Pytest configuration and shared fixtures
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import data_generator
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_generator import StockDataGenerator


@pytest.fixture
def basic_generator():
    """Create a basic stock data generator with default settings."""
    return StockDataGenerator(
        ticker="TEST",
        initial_price=100.0,
        shares_outstanding=100_000_000,
        initial_eps=5.0,
        initial_revenue_growth=0.15,
        initial_beta=1.0,
        benchmark_correlation=0.7,
        seed=42  # Fixed seed for reproducibility
    )


@pytest.fixture
def start_date():
    """Return a fixed start date for testing."""
    return datetime(2025, 1, 1)


@pytest.fixture
def small_dataset_days():
    """Return a small number of days for quick tests."""
    return 21  # One trading month


@pytest.fixture
def medium_dataset_days():
    """Return a medium number of days for more comprehensive tests."""
    return 63  # One quarter


@pytest.fixture
def full_year_days():
    """Return a full year of trading days."""
    return 252
