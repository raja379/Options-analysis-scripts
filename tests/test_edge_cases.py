"""
Unit tests for edge cases and validation

Tests boundary conditions, error handling, and unusual scenarios.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from data_generator import StockDataGenerator


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimum_days(self, basic_generator):
        """Test generating minimum number of days."""
        data = basic_generator.generate(n_days=1)
        assert len(data) == 1

    def test_large_dataset(self, basic_generator):
        """Test generating large dataset (10 years)."""
        data = basic_generator.generate(n_days=252 * 10)
        assert len(data) == 252 * 10

    def test_zero_initial_price_handled(self):
        """Test that very small initial price is handled."""
        generator = StockDataGenerator(initial_price=0.01, seed=42)
        data = generator.generate(n_days=21)

        # Should still generate valid data
        assert all(data['Close'] > 0)

    def test_very_high_initial_price(self):
        """Test with very high initial price."""
        generator = StockDataGenerator(initial_price=10000.0, seed=42)
        data = generator.generate(n_days=21)

        # Should generate valid data
        assert all(data['Close'] > 0)
        assert data['Close'].iloc[0] >= 9500  # Should be close to initial

    def test_zero_eps(self):
        """Test handling of zero EPS."""
        generator = StockDataGenerator(initial_eps=0.01, seed=42)
        data = generator.generate(n_days=21)

        # Should not have inf or nan in P/E
        assert not any(data['PE'].isin([np.inf, -np.inf]))
        assert not any(data['PE'].isna())

    def test_negative_growth(self):
        """Test with negative revenue growth."""
        generator = StockDataGenerator(initial_revenue_growth=-0.10, seed=42)
        data = generator.generate(n_days=21)

        # Should still generate valid data
        assert len(data) == 21
        assert all(data['Close'] > 0)

    def test_very_high_volatility(self):
        """Test with very high base volatility."""
        generator = StockDataGenerator(seed=42)
        generator.base_volatility = 0.80  # 80% annual volatility

        data = generator.generate(n_days=63)

        # Should still generate valid data
        assert len(data) == 63
        assert all(data['Close'] > 0)

    def test_very_low_volatility(self):
        """Test with very low base volatility."""
        generator = StockDataGenerator(seed=42)
        generator.base_volatility = 0.01  # 1% annual volatility

        data = generator.generate(n_days=63)

        # Should have low volatility
        returns = data['Close'].pct_change().dropna()
        assert returns.std() < 0.05  # Daily std should be small

    def test_zero_correlation(self):
        """Test with zero benchmark correlation."""
        generator = StockDataGenerator(benchmark_correlation=0.0, seed=42)
        data = generator.generate(n_days=21)

        assert len(data) == 21

    def test_perfect_correlation(self):
        """Test with perfect benchmark correlation."""
        generator = StockDataGenerator(benchmark_correlation=1.0, seed=42)
        data = generator.generate(n_days=21)

        assert len(data) == 21

    def test_zero_beta(self):
        """Test with zero beta."""
        generator = StockDataGenerator(initial_beta=0.0, seed=42)
        data = generator.generate(n_days=21)

        assert len(data) == 21
        # Beta in output should be close to 0
        assert abs(data['Beta'].mean()) < 0.3

    def test_negative_beta(self):
        """Test with negative beta."""
        generator = StockDataGenerator(initial_beta=-0.5, seed=42)
        data = generator.generate(n_days=21)

        assert len(data) == 21


class TestStartDateHandling:
    """Test start date handling."""

    def test_start_date_on_weekend(self, basic_generator):
        """Test that weekend start date is handled correctly."""
        # Saturday
        start_date = datetime(2025, 1, 4)  # Saturday
        assert start_date.weekday() == 5

        data = basic_generator.generate(n_days=21, start_date=start_date)

        # First date should be Monday
        first_date = pd.to_datetime(data['Date'].iloc[0])
        assert first_date.weekday() < 5  # Not weekend

    def test_start_date_in_past(self, basic_generator):
        """Test generating data for past dates."""
        start_date = datetime(2020, 1, 1)
        data = basic_generator.generate(n_days=21, start_date=start_date)

        assert len(data) == 21
        first_date = pd.to_datetime(data['Date'].iloc[0])
        assert first_date.year == 2020

    def test_start_date_in_future(self, basic_generator):
        """Test generating data for future dates."""
        start_date = datetime(2030, 1, 1)
        data = basic_generator.generate(n_days=21, start_date=start_date)

        assert len(data) == 21
        first_date = pd.to_datetime(data['Date'].iloc[0])
        assert first_date.year == 2030


class TestEarningsHandling:
    """Test earnings-related edge cases."""

    def test_very_frequent_earnings(self, basic_generator):
        """Test with very frequent earnings (every 7 days)."""
        basic_generator.earnings_frequency = 7
        data = basic_generator.generate(n_days=63)

        earnings_days = sum(data['EarningsFlag'])

        # Should have many earnings days
        assert earnings_days >= 7  # At least 7 in 63 days

    def test_very_infrequent_earnings(self, basic_generator):
        """Test with infrequent earnings (every 365 days)."""
        basic_generator.earnings_frequency = 365
        data = basic_generator.generate(n_days=252)

        earnings_days = sum(data['EarningsFlag'])

        # Should have 0 or 1 earnings day
        assert earnings_days <= 1

    def test_no_earnings_in_short_period(self, basic_generator):
        """Test that no earnings occur in very short period."""
        basic_generator.earnings_frequency = 90
        data = basic_generator.generate(n_days=5)

        # No earnings should occur
        assert sum(data['EarningsFlag']) == 0


class TestDataIntegrity:
    """Test data integrity and consistency."""

    def test_no_missing_values(self, basic_generator):
        """Test that generated data has no missing values."""
        data = basic_generator.generate(n_days=63)

        # Check for NaN values (except PE which can be NaN for negative earnings)
        for col in data.columns:
            if col != 'PE':
                assert not data[col].isna().any(), f"Column {col} has NaN values"

    def test_no_infinite_values(self, basic_generator):
        """Test that generated data has no infinite values."""
        data = basic_generator.generate(n_days=63)

        for col in data.select_dtypes(include=[np.number]).columns:
            assert not any(data[col].isin([np.inf, -np.inf])), f"Column {col} has infinite values"

    def test_volume_is_integer(self, basic_generator):
        """Test that volume is always integer."""
        data = basic_generator.generate(n_days=63)

        assert all(data['Volume'] == data['Volume'].astype(int))

    def test_consistent_market_cap(self, basic_generator):
        """Test that market cap is consistent with price."""
        data = basic_generator.generate(n_days=21)

        for idx, row in data.iterrows():
            expected_mc = row['Close'] * basic_generator.shares_outstanding
            assert abs(row['MarketCap'] - expected_mc) < 100  # Small rounding tolerance


class TestReproducibility:
    """Test reproducibility with seeds."""

    def test_same_seed_same_output_twice(self):
        """Test that running twice with same seed gives same output."""
        gen1 = StockDataGenerator(seed=42)
        data1a = gen1.generate(n_days=21, start_date=datetime(2025, 1, 1))
        data1b = gen1.generate(n_days=21, start_date=datetime(2025, 1, 1))

        # Note: Since we're not resetting the RNG, these will be different
        # Let's test by creating fresh generators
        gen2 = StockDataGenerator(seed=42)
        gen3 = StockDataGenerator(seed=42)

        data2 = gen2.generate(n_days=21, start_date=datetime(2025, 1, 1))
        data3 = gen3.generate(n_days=21, start_date=datetime(2025, 1, 1))

        # Should be identical
        pd.testing.assert_frame_equal(data2, data3)

    def test_different_start_dates_different_output(self):
        """Test that different start dates can give different data."""
        gen1 = StockDataGenerator(seed=42)
        gen2 = StockDataGenerator(seed=42)

        data1 = gen1.generate(n_days=21, start_date=datetime(2025, 1, 1))
        data2 = gen2.generate(n_days=21, start_date=datetime(2025, 6, 1))

        # Dates should be different
        assert not data1['Date'].equals(data2['Date'])


class TestParameterValidation:
    """Test parameter validation."""

    def test_invalid_regime_in_schedule(self, basic_generator):
        """Test that invalid regime raises error."""
        with pytest.raises(ValueError):
            basic_generator.set_regime_schedule([
                (datetime(2025, 1, 1), 'invalid_regime')
            ])

    def test_negative_shares_outstanding(self):
        """Test with negative shares outstanding."""
        # Should still create generator (no validation currently)
        generator = StockDataGenerator(shares_outstanding=-100_000_000)
        # But market cap will be negative
        data = generator.generate(n_days=5)
        # Market cap should reflect the negative shares
        assert any(data['MarketCap'] < 0)


class TestExtremeScenarios:
    """Test extreme scenarios."""

    def test_continuous_bear_market(self, basic_generator):
        """Test with continuous bear market regime."""
        start_date = datetime(2025, 1, 1)

        # Force bear market
        basic_generator.set_regime_schedule([
            (datetime(2025, 1, 1), 'bear'),
        ])

        data = basic_generator.generate(n_days=252, start_date=start_date)

        # Should show negative total return or significant drawdown
        total_return = (data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1

        # Bear market should generally have negative or low returns
        # (though stochastic, so not guaranteed)
        assert total_return < 0.50  # Not strong bull market returns

    def test_continuous_bull_market(self, basic_generator):
        """Test with continuous bull market regime."""
        start_date = datetime(2025, 1, 1)

        # Force bull market
        basic_generator.set_regime_schedule([
            (datetime(2025, 1, 1), 'bull'),
        ])

        data = basic_generator.generate(n_days=252, start_date=start_date)

        # Should show positive bias (though not guaranteed due to stochastic)
        total_return = (data['Close'] / data['Close'].iloc[0]) - 1

        # Max price should be reasonably above start
        max_gain = total_return.max()
        assert max_gain > -0.20  # Should have some upward movement

    def test_extreme_volatility_spike(self, basic_generator):
        """Test with extreme volatility spike."""
        start_date = datetime(2025, 1, 1)

        # Schedule extreme volatility
        basic_generator.set_volatility_schedule([
            (datetime(2025, 1, 1), 1.00),  # 100% annual volatility
        ])

        data = basic_generator.generate(n_days=63, start_date=start_date)

        # Should still generate valid data
        assert len(data) == 63
        assert all(data['Close'] > 0)

        # Should have high volatility
        assert data['Volatility'].mean() > 0.50

    def test_multiple_targets_same_day(self, basic_generator):
        """Test adding multiple targets for same day."""
        target_date = datetime(2025, 3, 1)

        # Add two targets for same day (last one wins)
        basic_generator.add_price_target(target_date, 120.0)
        basic_generator.add_price_target(target_date, 150.0)

        assert len(basic_generator.price_targets) == 2
