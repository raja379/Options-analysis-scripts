"""
Unit tests for StockDataGenerator class

Tests basic functionality, data generation, and statistical properties.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from data_generator import StockDataGenerator


class TestBasicInitialization:
    """Test initialization and basic properties."""

    def test_default_initialization(self):
        """Test generator can be initialized with defaults."""
        generator = StockDataGenerator()
        assert generator.ticker == "SIM"
        assert generator.initial_price == 100.0
        assert generator.base_volatility == 0.20

    def test_custom_initialization(self, basic_generator):
        """Test generator with custom parameters."""
        assert basic_generator.ticker == "TEST"
        assert basic_generator.initial_price == 100.0
        assert basic_generator.initial_eps == 5.0
        assert basic_generator.initial_revenue_growth == 0.15

    def test_seed_reproducibility(self):
        """Test that same seed produces same results."""
        gen1 = StockDataGenerator(seed=42)
        gen2 = StockDataGenerator(seed=42)

        data1 = gen1.generate(n_days=21, start_date=datetime(2025, 1, 1))
        data2 = gen2.generate(n_days=21, start_date=datetime(2025, 1, 1))

        # Prices should be identical
        pd.testing.assert_series_equal(data1['Close'], data2['Close'])

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        gen1 = StockDataGenerator(seed=42)
        gen2 = StockDataGenerator(seed=123)

        data1 = gen1.generate(n_days=21, start_date=datetime(2025, 1, 1))
        data2 = gen2.generate(n_days=21, start_date=datetime(2025, 1, 1))

        # Prices should be different
        assert not data1['Close'].equals(data2['Close'])


class TestDataGeneration:
    """Test data generation functionality."""

    def test_generate_returns_dataframe(self, basic_generator):
        """Test that generate returns a pandas DataFrame."""
        data = basic_generator.generate(n_days=21)
        assert isinstance(data, pd.DataFrame)

    def test_generate_correct_number_of_rows(self, basic_generator, small_dataset_days):
        """Test that correct number of trading days are generated."""
        data = basic_generator.generate(n_days=small_dataset_days)
        assert len(data) == small_dataset_days

    def test_generate_required_columns(self, basic_generator):
        """Test that all required columns are present."""
        data = basic_generator.generate(n_days=21)

        required_columns = [
            'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'MarketCap',
            'EPS_TTM', 'PE', 'RevenueGrowth', 'Beta', 'Volatility',
            'Drawdown', 'DividendYield', 'EarningsFlag', 'NewsSentiment'
        ]

        for col in required_columns:
            assert col in data.columns, f"Missing column: {col}"

    def test_no_weekends_in_data(self, basic_generator, start_date):
        """Test that no weekends are included in the generated data."""
        data = basic_generator.generate(n_days=21, start_date=start_date)

        dates = pd.to_datetime(data['Date'])
        weekdays = dates.dt.dayofweek

        # 5 = Saturday, 6 = Sunday
        assert not any(weekdays >= 5), "Weekend dates found in data"

    def test_dates_are_sequential(self, basic_generator, start_date):
        """Test that dates are sequential (skipping weekends)."""
        data = basic_generator.generate(n_days=21, start_date=start_date)
        dates = pd.to_datetime(data['Date'])

        for i in range(1, len(dates)):
            prev_date = dates.iloc[i-1]
            curr_date = dates.iloc[i]

            # Calculate trading days between dates
            diff_days = (curr_date - prev_date).days

            # Should be 1 trading day (1-3 calendar days depending on weekends)
            assert 1 <= diff_days <= 3, f"Invalid date gap: {diff_days} days"


class TestPriceDataValidation:
    """Test price data validity and constraints."""

    def test_prices_are_positive(self, basic_generator):
        """Test that all prices are positive."""
        data = basic_generator.generate(n_days=63)

        assert all(data['Open'] > 0)
        assert all(data['High'] > 0)
        assert all(data['Low'] > 0)
        assert all(data['Close'] > 0)

    def test_high_low_relationship(self, basic_generator):
        """Test that High >= max(Open, Close) and Low <= min(Open, Close)."""
        data = basic_generator.generate(n_days=63)

        for _, row in data.iterrows():
            assert row['High'] >= row['Open'], f"High < Open"
            assert row['High'] >= row['Close'], f"High < Close"
            assert row['Low'] <= row['Open'], f"Low > Open"
            assert row['Low'] <= row['Close'], f"Low > Close"

    def test_volume_is_positive_integer(self, basic_generator):
        """Test that volume is positive integer."""
        data = basic_generator.generate(n_days=21)

        assert all(data['Volume'] > 0)
        assert all(data['Volume'] == data['Volume'].astype(int))


class TestFinancialMetrics:
    """Test financial metrics calculation."""

    def test_market_cap_calculation(self, basic_generator):
        """Test that market cap = price × shares outstanding."""
        data = basic_generator.generate(n_days=21)

        expected_market_cap = data['Close'] * basic_generator.shares_outstanding

        # Allow small rounding differences
        np.testing.assert_allclose(
            data['MarketCap'],
            expected_market_cap,
            rtol=1e-5
        )

    def test_pe_ratio_calculation(self, basic_generator):
        """Test that P/E = Price / EPS."""
        data = basic_generator.generate(n_days=21)

        # Calculate expected P/E
        expected_pe = data['Close'] / data['EPS_TTM']

        # Allow small rounding differences
        np.testing.assert_allclose(
            data['PE'],
            expected_pe,
            rtol=1e-5
        )

    def test_drawdown_is_non_positive(self, basic_generator):
        """Test that drawdown is always <= 0."""
        data = basic_generator.generate(n_days=63)
        assert all(data['Drawdown'] <= 0)

    def test_drawdown_at_peak_is_zero(self, basic_generator):
        """Test that drawdown is 0 when price is at peak."""
        data = basic_generator.generate(n_days=63)

        # Find the maximum price
        max_price_idx = data['Close'].idxmax()

        # Drawdown at max price should be 0 or very close
        assert abs(data.loc[max_price_idx, 'Drawdown']) < 1e-6

    def test_sentiment_range(self, basic_generator):
        """Test that sentiment is between -1 and 1."""
        data = basic_generator.generate(n_days=63)

        assert all(data['NewsSentiment'] >= -1)
        assert all(data['NewsSentiment'] <= 1)

    def test_earnings_flag_binary(self, basic_generator):
        """Test that earnings flag is 0 or 1."""
        data = basic_generator.generate(n_days=252)

        assert all(data['EarningsFlag'].isin([0, 1]))

    def test_earnings_frequency(self, basic_generator):
        """Test that earnings occur at expected frequency."""
        basic_generator.earnings_frequency = 90
        data = basic_generator.generate(n_days=252)

        earnings_days = data[data['EarningsFlag'] == 1]

        # Should have roughly 3 earnings in a year (252/90 ≈ 2.8)
        assert 2 <= len(earnings_days) <= 4


class TestStatisticalProperties:
    """Test statistical properties of generated data."""

    def test_returns_have_reasonable_mean(self, basic_generator):
        """Test that returns have reasonable mean."""
        data = basic_generator.generate(n_days=252)
        returns = data['Close'].pct_change().dropna()

        # Annualized return should be reasonable (-50% to +100%)
        annualized_return = (1 + returns.mean()) ** 252 - 1
        assert -0.5 <= annualized_return <= 1.0

    def test_returns_have_reasonable_volatility(self, basic_generator):
        """Test that returns have reasonable volatility."""
        data = basic_generator.generate(n_days=252)
        returns = data['Close'].pct_change().dropna()

        # Annualized volatility should match base volatility roughly
        annualized_vol = returns.std() * np.sqrt(252)

        # Allow wide range due to stochastic nature
        assert 0.10 <= annualized_vol <= 0.80

    def test_volume_increases_on_earnings(self, basic_generator):
        """Test that volume is higher on earnings days."""
        basic_generator.earnings_frequency = 90
        data = basic_generator.generate(n_days=252)

        earnings_volume = data[data['EarningsFlag'] == 1]['Volume'].mean()
        normal_volume = data[data['EarningsFlag'] == 0]['Volume'].mean()

        # Earnings volume should be higher
        assert earnings_volume > normal_volume


class TestSaveAndExport:
    """Test data export functionality."""

    def test_save_to_csv(self, basic_generator, tmp_path):
        """Test saving data to CSV file."""
        data = basic_generator.generate(n_days=21)

        # Use temporary path for testing
        csv_path = tmp_path / "test_data.csv"
        basic_generator.save_to_csv(data, str(csv_path))

        # Check file exists
        assert csv_path.exists()

        # Read back and verify
        loaded_data = pd.read_csv(csv_path)
        assert len(loaded_data) == len(data)
        assert list(loaded_data.columns) == list(data.columns)

    def test_summary_statistics(self, basic_generator):
        """Test get_summary_statistics returns expected keys."""
        data = basic_generator.generate(n_days=252)
        stats = basic_generator.get_summary_statistics(data)

        expected_keys = [
            'total_days', 'start_date', 'end_date', 'start_price', 'end_price',
            'total_return', 'annualized_return', 'volatility', 'sharpe_ratio',
            'max_drawdown', 'avg_volume', 'earnings_days', 'avg_pe', 'final_market_cap'
        ]

        for key in expected_keys:
            assert key in stats, f"Missing key: {key}"

    def test_summary_statistics_values(self, basic_generator):
        """Test that summary statistics have reasonable values."""
        data = basic_generator.generate(n_days=252)
        stats = basic_generator.get_summary_statistics(data)

        assert stats['total_days'] == 252
        assert stats['start_price'] > 0
        assert stats['end_price'] > 0
        assert -1 <= stats['total_return'] <= 10  # Allow for extreme cases
        assert 0 <= stats['volatility'] <= 2.0
        assert stats['max_drawdown'] <= 0
        assert stats['avg_volume'] > 0
        assert stats['avg_pe'] > 0
        assert stats['final_market_cap'] > 0


class TestRegimes:
    """Test market regime functionality."""

    def test_regimes_exist(self, basic_generator):
        """Test that regimes are defined."""
        assert 'bull' in basic_generator.regimes
        assert 'neutral' in basic_generator.regimes
        assert 'bear' in basic_generator.regimes

    def test_regime_parameters(self, basic_generator):
        """Test that regime parameters have expected structure."""
        for regime_name, params in basic_generator.regimes.items():
            assert 'drift' in params
            assert 'vol_multiplier' in params
            assert 'transition_prob' in params

            # Check reasonable values
            assert -0.5 <= params['drift'] <= 0.5
            assert 0.1 <= params['vol_multiplier'] <= 3.0
            assert 0 <= params['transition_prob'] <= 0.1

    def test_regime_transition(self, basic_generator):
        """Test regime transition mechanism."""
        # Set current regime
        basic_generator.current_regime = 'neutral'

        # Get transition (may or may not change)
        new_regime = basic_generator._get_regime_transition()

        # Should be one of the valid regimes
        assert new_regime in ['bull', 'neutral', 'bear']


class TestValuationDrift:
    """Test valuation-based drift adjustment."""

    def test_valuation_drift_with_high_peg(self, basic_generator):
        """Test that high PEG ratio reduces drift."""
        # High P/E, low growth = overvalued
        base_drift = 0.10
        current_pe = 50.0
        growth_rate = 0.05  # 5%

        adjusted_drift = basic_generator._calculate_valuation_drift(
            current_pe, growth_rate, base_drift
        )

        # Should reduce drift for overvalued stock
        assert adjusted_drift < base_drift

    def test_valuation_drift_with_low_peg(self, basic_generator):
        """Test that low PEG ratio increases drift."""
        # Low P/E, high growth = undervalued
        base_drift = 0.10
        current_pe = 10.0
        growth_rate = 0.30  # 30%

        adjusted_drift = basic_generator._calculate_valuation_drift(
            current_pe, growth_rate, base_drift
        )

        # Should increase drift for undervalued stock
        assert adjusted_drift > base_drift
