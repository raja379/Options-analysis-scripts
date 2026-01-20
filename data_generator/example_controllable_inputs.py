#!/usr/bin/env python3
"""
Example demonstrating controllable inputs for the StockDataGenerator.

Shows how to:
- Set target prices at specific dates
- Schedule news events with sentiment
- Schedule volatility changes
- Schedule market regime changes
"""

from stock_data_generator import StockDataGenerator
from datetime import datetime, timedelta


def main():
    """Demonstrate controllable inputs feature."""

    print("=" * 80)
    print("Stock Data Generator - Controllable Inputs Example")
    print("=" * 80)

    # Create generator
    print("\nCreating stock generator for 'CONTROLLED' ticker...")
    generator = StockDataGenerator(
        ticker="CONTROLLED",
        initial_price=100.0,
        shares_outstanding=100_000_000,
        initial_eps=4.0,
        initial_revenue_growth=0.12,
        initial_beta=1.1,
        benchmark_correlation=0.75,
        seed=42
    )

    # Define simulation period
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 12, 31)
    n_days = 252  # Trading days in 2025

    print(f"Simulation period: {start_date.date()} to {end_date.date()}")

    # =========================================================================
    # EXAMPLE 1: Target Prices at Specific Dates
    # =========================================================================
    print("\n" + "=" * 80)
    print("Setting Target Prices:")
    print("=" * 80)

    # Define target prices throughout the year
    target_dates = [
        (datetime(2025, 3, 1), 120.0, "Q1 target"),
        (datetime(2025, 6, 1), 140.0, "Q2 target"),
        (datetime(2025, 9, 1), 155.0, "Q3 target"),
        (datetime(2025, 12, 1), 180.0, "Q4 target"),
    ]

    for date, price, description in target_dates:
        generator.add_price_target(date, price)
        print(f"  {date.date()}: ${price:.2f} - {description}")

    # =========================================================================
    # EXAMPLE 2: News Events with Sentiment
    # =========================================================================
    print("\n" + "=" * 80)
    print("Scheduling News Events:")
    print("=" * 80)

    news_events = [
        (datetime(2025, 1, 15), -0.6, "Negative: Regulatory concerns", 1.8),
        (datetime(2025, 2, 10), 0.8, "Positive: Strong product launch", 1.5),
        (datetime(2025, 4, 20), 0.9, "Positive: Partnership announcement", 1.4),
        (datetime(2025, 7, 12), -0.7, "Negative: Executive departure", 2.0),
        (datetime(2025, 9, 5), 0.85, "Positive: Market expansion", 1.3),
        (datetime(2025, 11, 8), -0.4, "Neutral-Negative: Guidance adjustment", 1.6),
    ]

    for date, sentiment, description, vol_spike in news_events:
        generator.add_news_event(date, sentiment, description, vol_spike)
        sentiment_str = f"{sentiment:+.2f}"
        print(f"  {date.date()}: Sentiment={sentiment_str}, VolSpike={vol_spike}x - {description}")

    # =========================================================================
    # EXAMPLE 3: Volatility Schedule
    # =========================================================================
    print("\n" + "=" * 80)
    print("Setting Volatility Schedule:")
    print("=" * 80)

    volatility_schedule = [
        (datetime(2025, 1, 1), 0.25, "High volatility period"),
        (datetime(2025, 3, 15), 0.18, "Volatility decline"),
        (datetime(2025, 6, 1), 0.15, "Low volatility period"),
        (datetime(2025, 8, 1), 0.30, "Volatility spike"),
        (datetime(2025, 10, 1), 0.20, "Normal volatility"),
    ]

    vol_schedule_tuples = [(date, vol) for date, vol, _ in volatility_schedule]
    generator.set_volatility_schedule(vol_schedule_tuples)

    for date, vol, description in volatility_schedule:
        print(f"  {date.date()}: {vol*100:.0f}% - {description}")

    # =========================================================================
    # EXAMPLE 4: Market Regime Schedule
    # =========================================================================
    print("\n" + "=" * 80)
    print("Setting Market Regime Schedule:")
    print("=" * 80)

    regime_schedule = [
        (datetime(2025, 1, 1), 'neutral', "Starting neutral"),
        (datetime(2025, 3, 20), 'bull', "Bull market begins"),
        (datetime(2025, 7, 15), 'bear', "Bear market correction"),
        (datetime(2025, 9, 10), 'neutral', "Return to neutral"),
        (datetime(2025, 11, 1), 'bull', "Year-end rally"),
    ]

    regime_schedule_tuples = [(date, regime) for date, regime, _ in regime_schedule]
    generator.set_regime_schedule(regime_schedule_tuples)

    for date, regime, description in regime_schedule:
        print(f"  {date.date()}: {regime.upper()} - {description}")

    # =========================================================================
    # Generate Data
    # =========================================================================
    print("\n" + "=" * 80)
    print("Generating data...")
    print("=" * 80)

    data = generator.generate(n_days=n_days, start_date=start_date)

    # Save to CSV
    filename = 'controlled_stock_data.csv'
    generator.save_to_csv(data, filename)

    # =========================================================================
    # Analysis and Results
    # =========================================================================
    print("\n" + "=" * 80)
    print("Results Summary:")
    print("=" * 80)

    stats = generator.get_summary_statistics(data)

    print(f"\nOverall Statistics:")
    print(f"  Start Price: ${stats['start_price']:.2f}")
    print(f"  End Price: ${stats['end_price']:.2f}")
    print(f"  Total Return: {stats['total_return']*100:.2f}%")
    print(f"  Annualized Return: {stats['annualized_return']*100:.2f}%")
    print(f"  Volatility: {stats['volatility']*100:.2f}%")
    print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {stats['max_drawdown']*100:.2f}%")

    # Check actual prices near target dates
    print(f"\nTarget Price Achievement:")
    for target_date, target_price, description in target_dates:
        # Find closest date in data
        data['DateDiff'] = abs((pd.to_datetime(data['Date']) - target_date).dt.days)
        closest_row = data.loc[data['DateDiff'].idxmin()]
        actual_price = closest_row['Close']
        diff_pct = ((actual_price - target_price) / target_price) * 100
        print(f"  {target_date.date()}: Target=${target_price:.2f}, Actual=${actual_price:.2f} ({diff_pct:+.1f}%)")

    data = data.drop('DateDiff', axis=1)

    # Show news event impact
    print(f"\nNews Event Impact:")
    for date, sentiment, description, vol_spike in news_events[:3]:  # Show first 3
        # Find date in data
        data['DateDiff'] = abs((pd.to_datetime(data['Date']) - date).dt.days)
        event_row = data.loc[data['DateDiff'].idxmin()]
        price_change = ((event_row['Close'] - event_row['Open']) / event_row['Open']) * 100
        print(f"  {date.date()}: {description}")
        print(f"    Price Change: {price_change:+.2f}%, Volatility: {event_row['Volatility']*100:.1f}%")

    data = data.drop('DateDiff', axis=1)

    # Display sample data around first target
    print(f"\n" + "=" * 80)
    print("Sample Data (around March 1st target):")
    print("=" * 80)

    march_data = data[(data['Date'] >= '2025-02-25') & (data['Date'] <= '2025-03-05')]
    print(march_data[['Date', 'Open', 'Close', 'Volume', 'Volatility', 'NewsSentiment']].to_string(index=False))

    print("\n" + "=" * 80)
    print("Complete! Data saved to:", filename)
    print("=" * 80)


if __name__ == "__main__":
    import pandas as pd
    main()
