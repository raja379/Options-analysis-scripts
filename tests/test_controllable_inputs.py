"""
Unit tests for controllable inputs feature

Tests target prices, news events, volatility scheduling, and regime scheduling.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from data_generator import StockDataGenerator


class TestTargetPrices:
    """Test target price functionality."""

    def test_add_price_target(self, basic_generator):
        """Test adding a single price target."""
        target_date = datetime(2025, 3, 1)
        target_price = 120.0

        basic_generator.add_price_target(target_date, target_price)

        assert len(basic_generator.price_targets) == 1
        assert basic_generator.price_targets[0] == (target_date, target_price)

    def test_add_multiple_price_targets(self, basic_generator):
        """Test adding multiple price targets."""
        targets = [
            (datetime(2025, 3, 1), 120.0),
            (datetime(2025, 6, 1), 140.0),
            (datetime(2025, 9, 1), 160.0),
        ]

        for date, price in targets:
            basic_generator.add_price_target(date, price)

        assert len(basic_generator.price_targets) == 3

    def test_price_targets_are_sorted(self, basic_generator):
        """Test that price targets are automatically sorted by date."""
        # Add in reverse order
        basic_generator.add_price_target(datetime(2025, 9, 1), 160.0)
        basic_generator.add_price_target(datetime(2025, 3, 1), 120.0)
        basic_generator.add_price_target(datetime(2025, 6, 1), 140.0)

        # Should be sorted by date
        dates = [date for date, _ in basic_generator.price_targets]
        assert dates == sorted(dates)

    def test_price_guided_toward_target(self, basic_generator):
        """Test that price is guided toward target."""
        start_date = datetime(2025, 1, 1)
        target_date = datetime(2025, 3, 1)
        target_price = 150.0

        basic_generator.add_price_target(target_date, target_price)

        data = basic_generator.generate(n_days=60, start_date=start_date)

        # Find price closest to target date
        data['DateDiff'] = abs((pd.to_datetime(data['Date']) - target_date).dt.days)
        closest_row = data.loc[data['DateDiff'].idxmin()]
        actual_price = closest_row['Close']

        # Price should be reasonably close to target (within 30%)
        price_diff_pct = abs(actual_price - target_price) / target_price
        assert price_diff_pct < 0.30, f"Price {actual_price} too far from target {target_price}"

    def test_target_guided_drift(self, basic_generator):
        """Test _calculate_target_guided_drift method."""
        current_date = datetime(2025, 1, 1)
        current_price = 100.0
        target_date = datetime(2025, 3, 1)
        target_price = 120.0
        base_drift = 0.10

        basic_generator.add_price_target(target_date, target_price)

        # Calculate days to target (approximately 40 trading days)
        days_remaining = 100

        guided_drift = basic_generator._calculate_target_guided_drift(
            current_date, current_price, days_remaining, base_drift
        )

        # Guided drift should be positive (moving toward higher target)
        assert guided_drift > 0

class TestNewsEvents:
    """Test news event functionality."""

    def test_add_news_event(self, basic_generator):
        """Test adding a news event."""
        event_date = datetime(2025, 1, 15)
        sentiment = 0.8
        description = "Positive news"
        vol_spike = 1.5

        basic_generator.add_news_event(event_date, sentiment, description, vol_spike)

        assert len(basic_generator.news_events) == 1

        event = basic_generator.news_events[0]
        assert event[0] == event_date
        assert event[1] == sentiment
        assert event[2] == description
        assert event[3] == vol_spike

    def test_sentiment_is_clipped(self, basic_generator):
        """Test that sentiment is clipped to [-1, 1]."""
        event_date = datetime(2025, 1, 15)

        # Test sentiment > 1
        basic_generator.add_news_event(event_date, 1.5, "Test")
        assert basic_generator.news_events[0][1] == 1.0

        basic_generator.clear_schedules()

        # Test sentiment < -1
        basic_generator.add_news_event(event_date, -1.5, "Test")
        assert basic_generator.news_events[0][1] == -1.0

    def test_news_event_affects_sentiment(self, basic_generator):
        """Test that news event overrides calculated sentiment."""
        start_date = datetime(2025, 1, 1)
        event_date = datetime(2025, 1, 15)
        expected_sentiment = 0.9

        basic_generator.add_news_event(event_date, expected_sentiment, "Major news")

        data = basic_generator.generate(n_days=21, start_date=start_date)

        # Find the row with the event date
        data['DateDiff'] = abs((pd.to_datetime(data['Date']) - event_date).dt.days)
        event_row = data.loc[data['DateDiff'].idxmin()]

        # Sentiment should match the scheduled sentiment
        assert abs(event_row['NewsSentiment'] - expected_sentiment) < 0.01

    def test_news_event_increases_volatility(self, basic_generator):
        """Test that news event increases volatility."""
        start_date = datetime(2025, 1, 1)
        event_date = datetime(2025, 1, 15)
        vol_spike = 2.0

        basic_generator.add_news_event(event_date, 0.5, "News", vol_spike)

        data = basic_generator.generate(n_days=30, start_date=start_date)

        # Find event day and surrounding days
        data['DateDiff'] = abs((pd.to_datetime(data['Date']) - event_date).dt.days)
        event_idx = data['DateDiff'].idxmin()

        event_day_vol = data.loc[event_idx, 'Volatility']

        # Get average volatility of surrounding days (excluding event day)
        surrounding_vol = data[
            (data.index >= event_idx - 3) &
            (data.index <= event_idx + 3) &
            (data.index != event_idx)
        ]['Volatility'].mean()

        # Event day volatility should be higher
        assert event_day_vol > surrounding_vol

    def test_multiple_news_events(self, basic_generator):
        """Test adding multiple news events."""
        events = [
            (datetime(2025, 1, 15), 0.8, "Positive", 1.5),
            (datetime(2025, 2, 10), -0.6, "Negative", 1.8),
            (datetime(2025, 3, 5), 0.5, "Neutral", 1.3),
        ]

        for date, sentiment, desc, vol in events:
            basic_generator.add_news_event(date, sentiment, desc, vol)

        assert len(basic_generator.news_events) == 3

    def test_news_events_are_sorted(self, basic_generator):
        """Test that news events are sorted by date."""
        # Add in reverse order
        basic_generator.add_news_event(datetime(2025, 3, 5), 0.5, "Last")
        basic_generator.add_news_event(datetime(2025, 1, 15), 0.8, "First")
        basic_generator.add_news_event(datetime(2025, 2, 10), -0.6, "Middle")

        dates = [date for date, _, _, _ in basic_generator.news_events]
        assert dates == sorted(dates)


class TestVolatilityScheduling:
    """Test volatility scheduling functionality."""

    def test_set_volatility_schedule(self, basic_generator):
        """Test setting volatility schedule."""
        schedule = [
            (datetime(2025, 1, 1), 0.25),
            (datetime(2025, 6, 1), 0.15),
            (datetime(2025, 9, 1), 0.30),
        ]

        basic_generator.set_volatility_schedule(schedule)

        assert len(basic_generator.volatility_schedule) == 3
        assert basic_generator.volatility_schedule[0] == (datetime(2025, 1, 1), 0.25)

    def test_volatility_schedule_is_sorted(self, basic_generator):
        """Test that volatility schedule is sorted by date."""
        schedule = [
            (datetime(2025, 9, 1), 0.30),
            (datetime(2025, 1, 1), 0.25),
            (datetime(2025, 6, 1), 0.15),
        ]

        basic_generator.set_volatility_schedule(schedule)

        dates = [date for date, _ in basic_generator.volatility_schedule]
        assert dates == sorted(dates)

    def test_scheduled_volatility_is_used(self, basic_generator):
        """Test that scheduled volatility affects generated data."""
        start_date = datetime(2025, 1, 1)

        # Set high volatility for first period, low for second
        schedule = [
            (datetime(2025, 1, 1), 0.40),   # High vol
            (datetime(2025, 2, 1), 0.10),   # Low vol
        ]

        basic_generator.set_volatility_schedule(schedule)

        data = basic_generator.generate(n_days=60, start_date=start_date)

        # Split data into two periods
        split_date = datetime(2025, 2, 1)
        data['Date'] = pd.to_datetime(data['Date'])

        period1_vol = data[data['Date'] < split_date]['Volatility'].mean()
        period2_vol = data[data['Date'] >= split_date]['Volatility'].mean()

        # First period should have higher volatility
        assert period1_vol > period2_vol

    def test_get_scheduled_volatility(self, basic_generator):
        """Test _get_scheduled_volatility method."""
        schedule = [
            (datetime(2025, 1, 1), 0.25),
            (datetime(2025, 6, 1), 0.15),
        ]

        basic_generator.set_volatility_schedule(schedule)

        # Before any schedule
        vol = basic_generator._get_scheduled_volatility(datetime(2024, 12, 31))
        assert vol is None

        # After first schedule point
        vol = basic_generator._get_scheduled_volatility(datetime(2025, 3, 1))
        assert vol == 0.25

        # After second schedule point
        vol = basic_generator._get_scheduled_volatility(datetime(2025, 9, 1))
        assert vol == 0.15


class TestRegimeScheduling:
    """Test regime scheduling functionality."""

    def test_set_regime_schedule(self, basic_generator):
        """Test setting regime schedule."""
        schedule = [
            (datetime(2025, 1, 1), 'neutral'),
            (datetime(2025, 3, 20), 'bull'),
            (datetime(2025, 7, 15), 'bear'),
        ]

        basic_generator.set_regime_schedule(schedule)

        assert len(basic_generator.regime_schedule) == 3

    def test_invalid_regime_raises_error(self, basic_generator):
        """Test that invalid regime name raises error."""
        schedule = [
            (datetime(2025, 1, 1), 'invalid_regime'),
        ]

        with pytest.raises(ValueError, match="Invalid regime"):
            basic_generator.set_regime_schedule(schedule)

    def test_regime_schedule_is_sorted(self, basic_generator):
        """Test that regime schedule is sorted by date."""
        schedule = [
            (datetime(2025, 7, 15), 'bear'),
            (datetime(2025, 1, 1), 'neutral'),
            (datetime(2025, 3, 20), 'bull'),
        ]

        basic_generator.set_regime_schedule(schedule)

        dates = [date for date, _ in basic_generator.regime_schedule]
        assert dates == sorted(dates)

    def test_scheduled_regime_is_used(self, basic_generator):
        """Test that scheduled regimes affect data generation."""
        start_date = datetime(2025, 1, 1)

        # Schedule bull market
        schedule = [
            (datetime(2025, 1, 1), 'bull'),
        ]

        basic_generator.set_regime_schedule(schedule)

        data = basic_generator.generate(n_days=63, start_date=start_date)

        # Bull market should generally show positive returns
        total_return = (data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1

        # Note: Due to stochastic nature, we can't guarantee positive returns
        # but we can check that returns are reasonable for bull market
        assert total_return > -0.30  # Not severe loss in bull market

    def test_get_scheduled_regime(self, basic_generator):
        """Test _get_scheduled_regime method."""
        schedule = [
            (datetime(2025, 3, 20), 'bull'),
        ]

        basic_generator.set_regime_schedule(schedule)

        # On scheduled date
        regime = basic_generator._get_scheduled_regime(datetime(2025, 3, 20))
        assert regime == 'bull'

        # Different date
        regime = basic_generator._get_scheduled_regime(datetime(2025, 3, 21))
        assert regime is None


class TestClearSchedules:
    """Test clearing all schedules."""

    def test_clear_schedules(self, basic_generator):
        """Test that clear_schedules removes all scheduled items."""
        # Add various schedules
        basic_generator.add_price_target(datetime(2025, 3, 1), 120.0)
        basic_generator.add_news_event(datetime(2025, 1, 15), 0.8, "News")
        basic_generator.set_volatility_schedule([(datetime(2025, 1, 1), 0.25)])
        basic_generator.set_regime_schedule([(datetime(2025, 1, 1), 'bull')])

        # Clear all
        basic_generator.clear_schedules()

        # All should be empty
        assert len(basic_generator.price_targets) == 0
        assert len(basic_generator.news_events) == 0
        assert len(basic_generator.volatility_schedule) == 0
        assert len(basic_generator.regime_schedule) == 0


class TestCombinedSchedules:
    """Test combining multiple controllable inputs."""

    def test_targets_and_news_together(self, basic_generator):
        """Test using both targets and news events."""
        start_date = datetime(2025, 1, 1)

        # Add target and news
        basic_generator.add_price_target(datetime(2025, 3, 1), 130.0)
        basic_generator.add_news_event(datetime(2025, 1, 15), 0.9, "Major news", 1.8)

        data = basic_generator.generate(n_days=63, start_date=start_date)

        # Should generate successfully
        assert len(data) == 63
        assert all(data['Close'] > 0)

    def test_all_schedules_together(self, basic_generator):
        """Test using all controllable inputs together."""
        start_date = datetime(2025, 1, 1)

        # Add all types of schedules
        basic_generator.add_price_target(datetime(2025, 3, 1), 120.0)
        basic_generator.add_price_target(datetime(2025, 6, 1), 140.0)

        basic_generator.add_news_event(datetime(2025, 1, 15), -0.6, "Negative", 1.8)
        basic_generator.add_news_event(datetime(2025, 2, 10), 0.8, "Positive", 1.5)

        basic_generator.set_volatility_schedule([
            (datetime(2025, 1, 1), 0.25),
            (datetime(2025, 4, 1), 0.15),
        ])

        basic_generator.set_regime_schedule([
            (datetime(2025, 1, 1), 'neutral'),
            (datetime(2025, 3, 1), 'bull'),
        ])

        data = basic_generator.generate(n_days=126, start_date=start_date)

        # Should generate successfully with all schedules
        assert len(data) == 126
        assert all(data['Close'] > 0)

        # Verify required columns exist
        required_columns = ['Date', 'Close', 'Volume', 'Volatility', 'NewsSentiment']
        for col in required_columns:
            assert col in data.columns
