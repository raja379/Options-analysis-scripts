#!/usr/bin/env python3
"""
Integration Test: Realistic Amazon-like Stock Scenario with Controllable Inputs

This test simulates a realistic year for Amazon (AMZN) stock with:
- Quarterly earnings (AWS, Retail, Prime)
- Major business events and announcements
- Market regime changes
- Volatility spikes around events
- Target price progression

The test generates comprehensive visualizations showing the stock's behavior.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_generator import StockDataGenerator


def create_realistic_amazon_stock_scenario():
    """
    Create a realistic scenario for Amazon (AMZN) stock in 2025.

    Company Profile:
    - Ticker: AMZN
    - Initial Price: $140
    - Market Cap: ~$1.46T (10.4B shares)
    - Large-cap e-commerce, cloud computing, AI
    - Mature but still growing (AWS, Retail, Prime)
    """

    print("=" * 80)
    print("INTEGRATION TEST: Realistic Amazon Stock Scenario")
    print("=" * 80)
    print("\nCompany: AMZN - Large-Cap Tech (E-commerce, Cloud, AI)")
    print("Scenario: Full Year 2025 with Major Events")
    print("-" * 80)

    # Initialize generator with Amazon-like parameters
    generator = StockDataGenerator(
        ticker="AMZN",
        initial_price=140.0,
        shares_outstanding=10_400_000_000,  # 10.4B shares = $1.46T market cap
        initial_eps=2.85,  # P/E of ~49 (140/2.85)
        initial_revenue_growth=0.11,  # 11% YoY growth (mature but growing)
        initial_beta=1.15,  # Slightly above market
        benchmark_correlation=0.70,  # Strong market correlation
        seed=42
    )

    # Base volatility for large-cap tech
    generator.base_volatility = 0.28  # 28% annual volatility
    generator.earnings_frequency = 90  # Quarterly earnings

    # =========================================================================
    # DEFINE REALISTIC EVENTS AND TARGETS FOR 2025
    # =========================================================================

    print("\n📅 SCHEDULED EVENTS:")
    print("-" * 80)

    # QUARTER 1: AWS concerns, cost initiatives
    print("\nQ1: AWS Growth Concerns & Cost Reduction")

    # Jan 30: Q4 2024 earnings - AWS growth slows
    generator.add_news_event(
        datetime(2025, 1, 30),
        sentiment=-0.45,
        description="Q4 Earnings: AWS Growth Slows, Retail Strong",
        volatility_spike=1.9
    )
    print("  Jan 30: Q4 Earnings - AWS growth slows (sentiment: -0.45, vol: 1.9x)")

    # Feb 15: Cost-cutting program
    generator.add_news_event(
        datetime(2025, 2, 15),
        sentiment=0.35,
        description="Announces $3B Cost Reduction Program",
        volatility_spike=1.4
    )
    print("  Feb 15: $3B cost reduction program (sentiment: +0.35, vol: 1.4x)")

    # Mar 20: AI announcements
    generator.add_news_event(
        datetime(2025, 3, 20),
        sentiment=0.70,
        description="Major AI Integration in AWS - Bedrock Expansion",
        volatility_spike=1.5
    )
    print("  Mar 20: AI integration in AWS Bedrock (sentiment: +0.70, vol: 1.5x)")

    # Q1 target (conservative growth)
    generator.add_price_target(datetime(2025, 3, 31), 148.0)
    print("  Mar 31: Target Price = $148 (+5.7% from start)")

    # QUARTER 2: AWS recovery & regulatory headwinds
    print("\nQ2: AWS Reacceleration & Prime Day")

    # Apr 24: Q1 earnings beat
    generator.add_news_event(
        datetime(2025, 4, 24),
        sentiment=0.65,
        description="Q1 Earnings Beat - AWS Reaccelerates",
        volatility_spike=1.8
    )
    print("  Apr 24: Q1 earnings beat, AWS reaccelerates (sentiment: +0.65, vol: 1.8x)")

    # May 8: Regulatory concerns
    generator.add_news_event(
        datetime(2025, 5, 8),
        sentiment=-0.55,
        description="FTC Antitrust Investigation Intensifies",
        volatility_spike=1.7
    )
    print("  May 8: FTC antitrust investigation (sentiment: -0.55, vol: 1.7x)")

    # Jun 17: Prime Day success
    generator.add_news_event(
        datetime(2025, 6, 17),
        sentiment=0.50,
        description="Record Prime Day Sales - $14B in 48 Hours",
        volatility_spike=1.3
    )
    print("  Jun 17: Record Prime Day - $14B sales (sentiment: +0.50, vol: 1.3x)")

    # Q2 target (moderate growth)
    generator.add_price_target(datetime(2025, 6, 30), 158.0)
    print("  Jun 30: Target Price = $158 (+12.9% from start)")

    # QUARTER 3: Strong earnings, healthcare expansion
    print("\nQ3: AWS Margin Expansion & Healthcare Growth")

    # Jul 25: Q2 earnings - strong beat
    generator.add_news_event(
        datetime(2025, 7, 25),
        sentiment=0.80,
        description="Q2 Blowout - AWS Margin Expansion",
        volatility_spike=2.0
    )
    print("  Jul 25: Q2 blowout - AWS margin expansion (sentiment: +0.80, vol: 2.0x)")

    # Aug 12: Healthcare expansion
    generator.add_news_event(
        datetime(2025, 8, 12),
        sentiment=0.60,
        description="Amazon Pharmacy Expansion - New Partnerships",
        volatility_spike=1.4
    )
    print("  Aug 12: Pharmacy expansion partnerships (sentiment: +0.60, vol: 1.4x)")

    # Sep 5: AI infrastructure investment
    generator.add_news_event(
        datetime(2025, 9, 5),
        sentiment=0.40,
        description="$10B Investment in AI Data Centers",
        volatility_spike=1.3
    )
    print("  Sep 5: $10B AI data center investment (sentiment: +0.40, vol: 1.3x)")

    # Q3 target (growth continues)
    generator.add_price_target(datetime(2025, 9, 30), 165.0)
    print("  Sep 30: Target Price = $165 (+17.9% from start)")

    # QUARTER 4: Holiday season & year-end rally
    print("\nQ4: Holiday Season & Market Volatility")

    # Oct 15: Tech sector selloff
    generator.add_news_event(
        datetime(2025, 10, 15),
        sentiment=-0.60,
        description="Tech Sector Selloff - Profit Taking",
        volatility_spike=1.9
    )
    print("  Oct 15: Tech sector selloff - profit taking (sentiment: -0.60, vol: 1.9x)")

    # Oct 26: Q3 earnings solid
    generator.add_news_event(
        datetime(2025, 10, 26),
        sentiment=0.55,
        description="Q3 Earnings Solid - Holiday Guidance Strong",
        volatility_spike=1.8
    )
    print("  Oct 26: Q3 solid - strong holiday guidance (sentiment: +0.55, vol: 1.8x)")

    # Nov 28: Black Friday/Cyber Monday
    generator.add_news_event(
        datetime(2025, 11, 28),
        sentiment=0.70,
        description="Record Black Friday/Cyber Monday Sales",
        volatility_spike=1.2
    )
    print("  Nov 28: Record Black Friday/Cyber Monday (sentiment: +0.70, vol: 1.2x)")

    # Dec 15: Holiday season results
    generator.add_news_event(
        datetime(2025, 12, 15),
        sentiment=0.65,
        description="Holiday Season Beats Expectations",
        volatility_spike=1.3
    )
    print("  Dec 15: Holiday season beats expectations (sentiment: +0.65, vol: 1.3x)")

    # Q4 target (strong year-end)
    generator.add_price_target(datetime(2025, 12, 31), 175.0)
    print("  Dec 31: Target Price = $175 (+25.0% from start)")

    # =========================================================================
    # VOLATILITY SCHEDULE - Market conditions
    # =========================================================================

    print("\n📊 VOLATILITY SCHEDULE:")
    print("-" * 80)

    volatility_schedule = [
        (datetime(2025, 1, 1), 0.28),    # Start of year - moderate
        (datetime(2025, 2, 1), 0.32),    # Earnings season volatility
        (datetime(2025, 4, 1), 0.25),    # Post-earnings calm
        (datetime(2025, 7, 1), 0.30),    # Mid-year earnings
        (datetime(2025, 10, 1), 0.35),   # Uncertainty pre-holidays
        (datetime(2025, 11, 1), 0.22),   # Holiday season - more stable
        (datetime(2025, 12, 15), 0.26),  # Year-end positioning
    ]

    generator.set_volatility_schedule(volatility_schedule)

    for date, vol in volatility_schedule:
        print(f"  {date.strftime('%b %d')}: {vol*100:.0f}% annualized volatility")

    # =========================================================================
    # REGIME SCHEDULE - Market sentiment
    # =========================================================================

    print("\n🌐 MARKET REGIME SCHEDULE:")
    print("-" * 80)

    regime_schedule = [
        (datetime(2025, 1, 1), 'neutral'),     # Start neutral
        (datetime(2025, 3, 1), 'bull'),        # Early year rally
        (datetime(2025, 5, 1), 'neutral'),     # Consolidation
        (datetime(2025, 7, 1), 'bull'),        # Summer rally
        (datetime(2025, 10, 1), 'bear'),       # October correction
        (datetime(2025, 11, 15), 'bull'),      # Holiday rally
    ]

    generator.set_regime_schedule(regime_schedule)

    for date, regime in regime_schedule:
        print(f"  {date.strftime('%b %d')}: {regime.upper()} market")

    # =========================================================================
    # GENERATE DATA
    # =========================================================================

    print("\n⚙️  GENERATING DATA...")
    print("-" * 80)

    start_date = datetime(2025, 1, 1)
    data = generator.generate(n_days=252, start_date=start_date)

    print(f"✓ Generated {len(data)} trading days")
    print(f"✓ Date range: {data['Date'].iloc[0]} to {data['Date'].iloc[-1]}")

    # Save to CSV
    filename = 'integration_tests/amzn_realistic_2025.csv'
    generator.save_to_csv(data, filename)
    print(f"✓ Saved to {filename}")

    # Get statistics
    stats = generator.get_summary_statistics(data)

    print("\n📈 SUMMARY STATISTICS:")
    print("-" * 80)
    print(f"Start Price:        ${stats['start_price']:.2f}")
    print(f"End Price:          ${stats['end_price']:.2f}")
    print(f"Total Return:       {stats['total_return']*100:+.2f}%")
    print(f"Annualized Return:  {stats['annualized_return']*100:+.2f}%")
    print(f"Volatility:         {stats['volatility']*100:.1f}%")
    print(f"Sharpe Ratio:       {stats['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:       {stats['max_drawdown']*100:.1f}%")
    print(f"Avg Volume:         {stats['avg_volume']:,.0f}")
    print(f"Earnings Reports:   {stats['earnings_days']}")
    print(f"Final Market Cap:   ${stats['final_market_cap']/1e9:.2f}B")

    return data, generator


def create_visualizations(data, generator):
    """Create comprehensive visualizations of the simulated stock data."""

    print("\n📊 CREATING VISUALIZATIONS...")
    print("-" * 80)

    # Convert date to datetime
    data['Date'] = pd.to_datetime(data['Date'])

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

    # =========================================================================
    # 1. MAIN PRICE CHART with Volume
    # =========================================================================

    ax1 = fig.add_subplot(gs[0:2, :])

    # Plot price
    ax1.plot(data['Date'], data['Close'], linewidth=2, color='#2E86AB', label='Close Price')
    ax1.fill_between(data['Date'], data['Low'], data['High'], alpha=0.2, color='#A23B72', label='Daily Range')

    # Mark earnings days
    earnings_data = data[data['EarningsFlag'] == 1]
    ax1.scatter(earnings_data['Date'], earnings_data['Close'],
                color='gold', s=100, marker='D', zorder=5, label='Earnings', edgecolors='black')

    # Mark major news events (sentiment extremes)
    major_news = data[abs(data['NewsSentiment']) > 0.5]
    positive_news = major_news[major_news['NewsSentiment'] > 0.5]
    negative_news = major_news[major_news['NewsSentiment'] < -0.5]

    ax1.scatter(positive_news['Date'], positive_news['Close'],
                color='green', s=80, marker='^', alpha=0.7, label='Positive News', zorder=4)
    ax1.scatter(negative_news['Date'], negative_news['Close'],
                color='red', s=80, marker='v', alpha=0.7, label='Negative News', zorder=4)

    # Add target price lines
    for target_date, target_price in generator.price_targets:
        ax1.axhline(y=target_price, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax1.text(data['Date'].iloc[-1], target_price, f'  ${target_price}',
                verticalalignment='center', fontsize=9, color='gray')

    ax1.set_title('AMZN Stock Price - 2025 (Integration Test)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('Price ($)', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())

    # Add volume subplot
    ax1_volume = ax1.twinx()
    colors = ['green' if data['Close'].iloc[i] >= data['Open'].iloc[i] else 'red'
              for i in range(len(data))]
    ax1_volume.bar(data['Date'], data['Volume'], alpha=0.3, color=colors, width=0.8)
    ax1_volume.set_ylabel('Volume', fontsize=11)
    ax1_volume.set_ylim(0, data['Volume'].max() * 3)

    # =========================================================================
    # 2. RETURNS DISTRIBUTION
    # =========================================================================

    ax2 = fig.add_subplot(gs[2, 0])

    returns = data['Close'].pct_change().dropna() * 100
    ax2.hist(returns, bins=50, color='#A23B72', alpha=0.7, edgecolor='black')
    ax2.axvline(returns.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {returns.mean():.2f}%')
    ax2.axvline(returns.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {returns.median():.2f}%')
    ax2.set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Daily Return (%)', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # =========================================================================
    # 3. VOLATILITY OVER TIME
    # =========================================================================

    ax3 = fig.add_subplot(gs[2, 1])

    ax3.plot(data['Date'], data['Volatility'] * 100, linewidth=2, color='#F18F01')
    ax3.fill_between(data['Date'], 0, data['Volatility'] * 100, alpha=0.3, color='#F18F01')

    # Mark high volatility periods
    high_vol = data[data['Volatility'] > 0.40]
    ax3.scatter(high_vol['Date'], high_vol['Volatility'] * 100,
                color='red', s=30, alpha=0.6, zorder=5)

    ax3.set_title('Realized Volatility (Annualized)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Date', fontsize=10)
    ax3.set_ylabel('Volatility (%)', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator())

    # =========================================================================
    # 4. SENTIMENT OVER TIME
    # =========================================================================

    ax4 = fig.add_subplot(gs[3, 0])

    # Create sentiment bars
    sentiment_colors = ['green' if s > 0 else 'red' for s in data['NewsSentiment']]
    ax4.bar(data['Date'], data['NewsSentiment'], alpha=0.6, color=sentiment_colors, width=0.8)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_title('News Sentiment Over Time', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Date', fontsize=10)
    ax4.set_ylabel('Sentiment', fontsize=10)
    ax4.set_ylim(-1.1, 1.1)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax4.xaxis.set_major_locator(mdates.MonthLocator())

    # =========================================================================
    # 5. DRAWDOWN CHART
    # =========================================================================

    ax5 = fig.add_subplot(gs[3, 1])

    ax5.fill_between(data['Date'], data['Drawdown'] * 100, 0,
                      color='#C1121F', alpha=0.6)
    ax5.plot(data['Date'], data['Drawdown'] * 100,
             color='#780000', linewidth=2)

    # Mark max drawdown
    max_dd_idx = data['Drawdown'].idxmin()
    max_dd_date = data.loc[max_dd_idx, 'Date']
    max_dd_value = data.loc[max_dd_idx, 'Drawdown'] * 100
    ax5.scatter([max_dd_date], [max_dd_value], color='red', s=100, zorder=5, marker='X')
    ax5.annotate(f'Max DD: {max_dd_value:.1f}%',
                xy=(max_dd_date, max_dd_value),
                xytext=(10, 20), textcoords='offset points',
                bbox=dict(boxstyle='round', fc='yellow', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='red'))

    ax5.set_title('Drawdown from Peak', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Date', fontsize=10)
    ax5.set_ylabel('Drawdown (%)', fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax5.xaxis.set_major_locator(mdates.MonthLocator())

    # =========================================================================
    # SAVE AND SHOW
    # =========================================================================

    plt.suptitle('AMZN Integration Test - Realistic Scenario with Controllable Inputs',
                 fontsize=16, fontweight='bold', y=0.995)

    # Save figure
    output_file = 'integration_tests/amzn_realistic_2025_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to {output_file}")

    # Show figure
    plt.show()
    print("✓ Displaying visualization...")


def main():
    """Run the integration test."""

    # Generate data with realistic scenario
    data, generator = create_realistic_amazon_stock_scenario()

    # Create visualizations
    create_visualizations(data, generator)

    print("\n" + "=" * 80)
    print("INTEGRATION TEST COMPLETE!")
    print("=" * 80)
    print("\nGenerated Files:")
    print("  📄 integration_tests/amzn_realistic_2025.csv")
    print("  📊 integration_tests/amzn_realistic_2025_analysis.png")
    print("\nThe graph should be displayed in a window.")
    print("If running in a headless environment, check the PNG file.")
    print("=" * 80)


if __name__ == "__main__":
    main()
