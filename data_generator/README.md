# Stock Data Generator

A sophisticated simulated stock price data generator with regime-aware stochastic modeling.

## Features

The `StockDataGenerator` class generates realistic simulated stock price data with the following characteristics:

### 1. **Regime-Aware Model**
- Three market regimes: Bull, Neutral, and Bear
- Each regime has distinct drift and volatility characteristics
- Stochastic regime transitions based on configurable probabilities

### 2. **Volatility Clustering**
- GARCH(1,1)-like dynamics for realistic volatility patterns
- Volatility persists over time (high volatility tends to follow high volatility)
- Mean reversion to base volatility level

### 3. **Valuation-Based Drift**
- Drift adjusts based on P/E ratio relative to growth rate (PEG ratio logic)
- Overvalued stocks (high P/E, low growth) drift downward
- Undervalued stocks drift upward

### 4. **Earnings Date Effects**
- Periodic earnings announcements (default: every 90 days)
- Volatility spikes on earnings days
- Volume increases on earnings dates
- Fundamental updates (EPS, revenue growth) on earnings dates

### 5. **Benchmark Correlation**
- Returns are correlated with a benchmark index
- Configurable beta coefficient
- Uses Cholesky decomposition for proper correlation structure

### 6. **Comprehensive Financial Metrics**
- **Price Data**: Open, High, Low, Close
- **Volume & Market Cap**: Trading volume and market capitalization
- **Valuation**: EPS (TTM), P/E ratio
- **Growth**: Revenue growth rate
- **Risk**: Beta, volatility, drawdown
- **Income**: Dividend yield
- **Events**: Earnings announcement flag
- **Sentiment**: News sentiment score

### 7. **Controllable Inputs (NEW!)** 🎯
- **Target Prices**: Set specific price targets for dates (e.g., March 1 = $200, May 1 = $250)
  - Model automatically adjusts drift to guide price toward targets
  - Maintains realistic stochastic behavior while approaching targets
- **News Events**: Schedule news with custom sentiment and volatility impact
  - Sentiment ranges from -1 (very negative) to +1 (very positive)
  - Configurable volatility spikes for each event
  - Automatic volume increases on news days
- **Volatility Schedule**: Set different volatility regimes over time
  - Specify annualized volatility for different periods
  - Smooth transitions between volatility levels
- **Regime Schedule**: Control market regimes (bull/bear/neutral) at specific dates
  - Override stochastic transitions with deterministic regime changes
  - Combine with other controllable inputs for complex scenarios

## Usage

### Basic Example

```python
from data_generator import StockDataGenerator

# Create generator
generator = StockDataGenerator(
    ticker="AAPL",
    initial_price=150.0,
    shares_outstanding=100_000_000,
    initial_eps=5.0,
    initial_revenue_growth=0.15,
    initial_beta=1.0,
    benchmark_correlation=0.7,
    seed=42  # For reproducibility
)

# Generate 1 year of data
data = generator.generate(n_days=252)

# Save to CSV
generator.save_to_csv(data, 'stock_data.csv')

# Get summary statistics
stats = generator.get_summary_statistics(data)
print(f"Total Return: {stats['total_return']*100:.2f}%")
print(f"Volatility: {stats['volatility']*100:.2f}%")
print(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
```

### Advanced Configuration

```python
# Create a volatile growth stock
tech_stock = StockDataGenerator(
    ticker="TECH",
    initial_price=300.0,
    shares_outstanding=50_000_000,
    initial_eps=8.0,
    initial_revenue_growth=0.30,  # 30% growth
    initial_beta=1.5,  # High beta
    benchmark_correlation=0.8,
    seed=123
)

# Increase base volatility
tech_stock.base_volatility = 0.40  # 40% annualized

# Adjust earnings frequency
tech_stock.earnings_frequency = 90  # Quarterly

# Generate 2 years of data
data = tech_stock.generate(n_days=252 * 2)
```

### Controllable Inputs - Target Prices & Events

Control specific aspects of the simulation with scheduled targets and events:

```python
from datetime import datetime

# Create generator
generator = StockDataGenerator(
    ticker="CONTROLLED",
    initial_price=100.0,
    seed=42
)

# Set target prices - model will guide price toward these targets
generator.add_price_target(datetime(2025, 3, 1), 120.0)   # March target
generator.add_price_target(datetime(2025, 6, 1), 140.0)   # June target
generator.add_price_target(datetime(2025, 9, 1), 155.0)   # September target
generator.add_price_target(datetime(2025, 12, 1), 180.0)  # December target

# Schedule news events with sentiment and volatility impact
generator.add_news_event(
    date=datetime(2025, 1, 15),
    sentiment=-0.6,  # Negative news
    description="Regulatory concerns",
    volatility_spike=1.8  # 1.8x volatility multiplier
)

generator.add_news_event(
    date=datetime(2025, 2, 10),
    sentiment=0.8,  # Positive news
    description="Strong product launch",
    volatility_spike=1.5
)

# Set volatility schedule
volatility_schedule = [
    (datetime(2025, 1, 1), 0.25),   # Start with 25% vol
    (datetime(2025, 6, 1), 0.15),   # Drop to 15% vol
    (datetime(2025, 9, 1), 0.30),   # Spike to 30% vol
]
generator.set_volatility_schedule(volatility_schedule)

# Set market regime schedule
regime_schedule = [
    (datetime(2025, 1, 1), 'neutral'),
    (datetime(2025, 3, 20), 'bull'),
    (datetime(2025, 7, 15), 'bear'),
    (datetime(2025, 9, 10), 'neutral'),
]
generator.set_regime_schedule(regime_schedule)

# Generate data with all controllable inputs
data = generator.generate(n_days=252, start_date=datetime(2025, 1, 1))

# Results will show:
# - Price trending toward targets (while maintaining realism)
# - Sharp moves and high volume on news event days
# - Volatility changes at scheduled dates
# - Regime-specific behavior at scheduled dates
```

**Key Features:**
- **Target Guidance**: Price is guided toward targets using drift adjustment
- **News Impact**: Sentiment affects price direction, volatility spikes on news days
- **Realistic Behavior**: Maintains stochastic properties despite guidance
- **Volume Response**: Trading volume increases on news/event days

See `example_controllable_inputs.py` for a complete demonstration.

## Output Format

The generated DataFrame includes the following columns:

| Column | Description |
|--------|-------------|
| `Date` | Trading date |
| `Open` | Opening price |
| `High` | Daily high price |
| `Low` | Daily low price |
| `Close` | Closing price |
| `Volume` | Trading volume |
| `MarketCap` | Market capitalization (Price × Shares Outstanding) |
| `EPS_TTM` | Earnings per share (trailing twelve months) |
| `PE` | Price-to-earnings ratio |
| `RevenueGrowth` | Revenue growth rate |
| `Beta` | Beta coefficient vs benchmark |
| `Volatility` | Annualized volatility |
| `Drawdown` | Drawdown from running maximum |
| `DividendYield` | Dividend yield |
| `EarningsFlag` | 1 if earnings day, 0 otherwise |
| `NewsSentiment` | News sentiment score (-1 to 1) |

## CSV Export

Data is exported in CSV format suitable for backtesting:

```csv
Date,Open,High,Low,Close,Volume,MarketCap,EPS_TTM,PE,RevenueGrowth,Beta,Volatility,Drawdown,DividendYield,EarningsFlag,NewsSentiment
2024-01-02,150.0,151.5,149.2,150.8,1000000,15080000000,5.0,30.16,0.15,1.0,0.20,-0.0053,0.02,0,0.25
...
```

## Parameters

### Constructor Parameters

- `ticker` (str): Stock ticker symbol (default: "SIM")
- `initial_price` (float): Starting stock price (default: 100.0)
- `shares_outstanding` (float): Number of shares outstanding (default: 100,000,000)
- `initial_eps` (float): Initial earnings per share (default: 5.0)
- `initial_revenue_growth` (float): Initial revenue growth rate (default: 0.15)
- `initial_beta` (float): Beta coefficient vs benchmark (default: 1.0)
- `benchmark_correlation` (float): Correlation with benchmark index (default: 0.7)
- `seed` (int, optional): Random seed for reproducibility

### Configurable Attributes

After creating the generator, you can adjust:

- `base_volatility`: Base annualized volatility (default: 0.20)
- `earnings_frequency`: Days between earnings (default: 90)
- `earnings_vol_spike`: Volatility multiplier on earnings days (default: 2.5)
- `regimes`: Dictionary of market regime parameters

### Controllable Input Methods

- `add_price_target(date, target_price)`: Add a target price for a specific date
- `add_news_event(date, sentiment, description, volatility_spike)`: Schedule a news event
- `set_volatility_schedule(schedule)`: Set a list of (date, volatility) tuples
- `set_regime_schedule(schedule)`: Set a list of (date, regime_name) tuples
- `clear_schedules()`: Clear all scheduled events and targets

## Examples

### Basic Example
See `example_usage.py` for complete examples including:
- Growth stock simulation (high P/E, high growth)
- Value stock simulation (low P/E, moderate growth)
- Volatile tech stock simulation (high beta, high volatility)

```bash
cd data_generator
python3 example_usage.py
```

### Controllable Inputs Example
See `example_controllable_inputs.py` for advanced scenarios with:
- Target price scheduling (March=$120, June=$140, etc.)
- News events with custom sentiment and volatility impact
- Volatility regime changes throughout the year
- Market regime scheduling (bull/bear/neutral transitions)

```bash
cd data_generator
python3 example_controllable_inputs.py
```

## Technical Details

### Volatility Model

The volatility follows a GARCH(1,1)-like process:

```
σ²(t) = ω·σ²_base + α·r²(t-1) + β·σ²(t-1)
```

Where:
- ω = mean reversion speed (0.08)
- α = shock sensitivity (0.02)
- β = volatility persistence (0.90)
- r(t-1) = previous return

### Price Process

Daily returns are generated as:

```
r(t) = μ(t)/252 + β·r_benchmark(t) + ε(t)
```

Where:
- μ(t) = regime-dependent drift adjusted for valuation
- β = beta coefficient
- ε(t) = idiosyncratic shock with time-varying volatility

### Regime Dynamics

Market regimes switch stochastically:
- **Bull**: 15% annual drift, 0.8× volatility
- **Neutral**: 8% annual drift, 1.0× volatility
- **Bear**: -10% annual drift, 1.5× volatility

## Requirements

- Python 3.7+
- numpy >= 1.24.0
- pandas >= 2.0.0

## License

MIT License
