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

## Examples

See `example_usage.py` for complete examples including:
- Growth stock simulation
- Value stock simulation
- Volatile tech stock simulation

Run the examples:

```bash
cd data_generator
python3 example_usage.py
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
