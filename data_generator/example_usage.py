#!/usr/bin/env python3
"""
Example usage of the StockDataGenerator class.

Demonstrates how to generate simulated stock data with various configurations.
"""

from stock_data_generator import StockDataGenerator
from datetime import datetime


def main():
    """Generate example stock data and save to CSV."""

    print("=" * 70)
    print("Stock Data Generator - Example Usage")
    print("=" * 70)

    # Example 1: Generate data for a growth stock
    print("\n1. Generating data for a growth stock (high P/E, high growth)...")
    growth_stock = StockDataGenerator(
        ticker="GROWTH",
        initial_price=150.0,
        shares_outstanding=50_000_000,
        initial_eps=3.0,  # High P/E of 50
        initial_revenue_growth=0.35,  # 35% growth
        initial_beta=1.3,
        benchmark_correlation=0.75,
        seed=42
    )

    growth_data = growth_stock.generate(n_days=252 * 2)  # 2 years
    growth_stock.save_to_csv(growth_data, 'growth_stock_data.csv')

    stats = growth_stock.get_summary_statistics(growth_data)
    print("\nGrowth Stock Statistics:")
    print(f"  Start Date: {stats['start_date']}")
    print(f"  End Date: {stats['end_date']}")
    print(f"  Start Price: ${stats['start_price']:.2f}")
    print(f"  End Price: ${stats['end_price']:.2f}")
    print(f"  Total Return: {stats['total_return']*100:.2f}%")
    print(f"  Annualized Return: {stats['annualized_return']*100:.2f}%")
    print(f"  Volatility: {stats['volatility']*100:.2f}%")
    print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {stats['max_drawdown']*100:.2f}%")
    print(f"  Earnings Reports: {stats['earnings_days']}")
    print(f"  Average P/E: {stats['avg_pe']:.2f}")

    # Example 2: Generate data for a value stock
    print("\n" + "=" * 70)
    print("2. Generating data for a value stock (low P/E, moderate growth)...")
    value_stock = StockDataGenerator(
        ticker="VALUE",
        initial_price=50.0,
        shares_outstanding=200_000_000,
        initial_eps=5.0,  # P/E of 10
        initial_revenue_growth=0.08,  # 8% growth
        initial_beta=0.8,
        benchmark_correlation=0.6,
        seed=123
    )

    value_data = value_stock.generate(n_days=252)  # 1 year
    value_stock.save_to_csv(value_data, 'value_stock_data.csv')

    stats = value_stock.get_summary_statistics(value_data)
    print("\nValue Stock Statistics:")
    print(f"  Start Date: {stats['start_date']}")
    print(f"  End Date: {stats['end_date']}")
    print(f"  Start Price: ${stats['start_price']:.2f}")
    print(f"  End Price: ${stats['end_price']:.2f}")
    print(f"  Total Return: {stats['total_return']*100:.2f}%")
    print(f"  Annualized Return: {stats['annualized_return']*100:.2f}%")
    print(f"  Volatility: {stats['volatility']*100:.2f}%")
    print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {stats['max_drawdown']*100:.2f}%")
    print(f"  Earnings Reports: {stats['earnings_days']}")
    print(f"  Average P/E: {stats['avg_pe']:.2f}")

    # Example 3: Generate data for a volatile tech stock
    print("\n" + "=" * 70)
    print("3. Generating data for a volatile tech stock...")
    tech_stock = StockDataGenerator(
        ticker="TECH",
        initial_price=300.0,
        shares_outstanding=30_000_000,
        initial_eps=8.0,  # P/E of 37.5
        initial_revenue_growth=0.25,  # 25% growth
        initial_beta=1.5,
        benchmark_correlation=0.8,
        seed=456
    )
    tech_stock.base_volatility = 0.35  # 35% annualized volatility

    tech_data = tech_stock.generate(n_days=252)  # 1 year
    tech_stock.save_to_csv(tech_data, 'tech_stock_data.csv')

    stats = tech_stock.get_summary_statistics(tech_data)
    print("\nTech Stock Statistics:")
    print(f"  Start Date: {stats['start_date']}")
    print(f"  End Date: {stats['end_date']}")
    print(f"  Start Price: ${stats['start_price']:.2f}")
    print(f"  End Price: ${stats['end_price']:.2f}")
    print(f"  Total Return: {stats['total_return']*100:.2f}%")
    print(f"  Annualized Return: {stats['annualized_return']*100:.2f}%")
    print(f"  Volatility: {stats['volatility']*100:.2f}%")
    print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {stats['max_drawdown']*100:.2f}%")
    print(f"  Earnings Reports: {stats['earnings_days']}")
    print(f"  Average P/E: {stats['avg_pe']:.2f}")

    print("\n" + "=" * 70)
    print("Data generation complete!")
    print("Files created:")
    print("  - growth_stock_data.csv")
    print("  - value_stock_data.csv")
    print("  - tech_stock_data.csv")
    print("=" * 70)

    # Display sample of the data
    print("\nSample data (first 10 rows of growth stock):")
    print(growth_data.head(10).to_string())

    # Display earnings days
    earnings_days = growth_data[growth_data['EarningsFlag'] == 1]
    print(f"\n\nEarnings announcement days (total: {len(earnings_days)}):")
    print(earnings_days[['Date', 'Close', 'Volume', 'Volatility', 'NewsSentiment']].head().to_string())


if __name__ == "__main__":
    main()
