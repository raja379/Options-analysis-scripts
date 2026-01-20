# Integration Tests

End-to-end integration tests for the Stock Data Generator with realistic scenarios and visualizations.

## Overview

Integration tests validate the complete workflow of the data generator including:
- Controllable inputs (targets, news, volatility, regimes)
- Data generation over extended periods
- Statistical properties of generated data
- Visualization capabilities
- CSV export functionality

## Test: Realistic Tech Stock Scenario

### File: `test_realistic_scenario.py`

A comprehensive integration test simulating a full year (2025) for a mid-cap tech company stock.

**Company Profile:**
- **Ticker**: TECHX
- **Type**: Growing SaaS company
- **Initial Price**: $85
- **Market Cap**: ~$8.5B
- **Volatility**: 35% annualized (tech stock)
- **Beta**: 1.35 (high beta)

### Scenario Timeline

#### Q1 2025: Market Correction & Guidance Cut
- **Jan 15**: Q4 Earnings miss, guidance cut (-0.65 sentiment, 2.2x vol spike)
- **Feb 20**: Activist investor takes position (-0.40 sentiment, 1.6x vol)
- **Target**: $72 by Mar 31 (15% down)

#### Q2 2025: Recovery & Product Launch
- **Apr 17**: Q1 earnings beat (+0.55 sentiment, 1.8x vol)
- **May 12**: AI product launch, strong reception (+0.85 sentiment, 1.5x vol)
- **Target**: $88 by Jun 30 (recovery above start)

#### Q3 2025: Strong Growth & Partnership
- **Jul 21**: Q2 blowout earnings (+0.90 sentiment, 2.0x vol)
- **Aug 5**: Strategic partnership with Fortune 100 (+0.75 sentiment, 1.4x vol)
- **Target**: $105 by Sep 30 (23% up)

#### Q4 2025: Volatility & Year-End Rally
- **Oct 3**: Market correction (-0.50 sentiment, 1.9x vol)
- **Oct 23**: Q3 earnings solid (+0.60 sentiment, 1.7x vol)
- **Nov 18**: Insider buying (+0.45 sentiment, 1.2x vol)
- **Target**: $115 by Dec 31 (35% up)

### Controllable Inputs Used

#### Price Targets (4 quarterly targets)
```python
Mar 31: $72.00
Jun 30: $88.00
Sep 30: $105.00
Dec 31: $115.00
```

#### News Events (10 major events)
- 3 negative events (earnings miss, activist, correction)
- 7 positive events (beats, launches, partnerships)
- Sentiment range: -0.65 to +0.90
- Volatility spikes: 1.2x to 2.2x

#### Volatility Schedule (6 periods)
```python
Jan 1:  35% → Feb 15: 45% → Apr 1: 30%
Jul 1:  25% → Oct 1:  40% → Nov 15: 28%
```

#### Market Regimes (6 regime changes)
```python
Jan 1:  neutral → Feb 1:  bear   → Apr 15: neutral
Jun 1:  bull    → Oct 1:  bear   → Nov 1:  bull
```

## Running the Integration Test

### Prerequisites

Install required packages:
```bash
pip install -r requirements.txt
```

This includes:
- numpy
- pandas
- matplotlib (for visualizations)
- pytest (optional, for test framework)

### Run the Test

```bash
# From project root
python3 integration_tests/test_realistic_scenario.py
```

Or with explicit matplotlib backend:
```bash
MPLBACKEND=Agg python3 integration_tests/test_realistic_scenario.py
```

### Expected Output

**Console Output:**
- Event schedule listing
- Volatility schedule
- Regime schedule
- Generation progress
- Summary statistics

**Generated Files:**
1. `techx_realistic_2025.csv` - Full dataset with all metrics
2. `techx_realistic_2025_analysis.png` - Comprehensive visualization

## Visualizations

The test generates a comprehensive 6-panel visualization:

### 1. **Main Price Chart** (Top, full width)
- Close price line
- Daily high/low range (shaded)
- Earnings events (gold diamonds)
- Positive news events (green triangles)
- Negative news events (red triangles)
- Target price lines (gray dashed)
- Volume bars (background)

### 2. **Daily Returns Distribution** (Middle left)
- Histogram of daily returns
- Mean and median lines
- Shows return distribution characteristics

### 3. **Realized Volatility** (Middle right)
- Annualized volatility over time
- Shows volatility clustering
- Highlights high-volatility periods (red dots)

### 4. **News Sentiment** (Bottom left)
- Sentiment score over time (-1 to +1)
- Green bars for positive sentiment
- Red bars for negative sentiment

### 5. **Drawdown from Peak** (Bottom right)
- Maximum drawdown visualization
- Shows peak-to-trough declines
- Annotates maximum drawdown point

## Example Output

**Summary Statistics:**
```
Start Price:        $89.73
End Price:          $95.23
Total Return:       +6.13%
Annualized Return:  +6.13%
Volatility:         42.8%
Sharpe Ratio:       0.34
Max Drawdown:       -28.5%
Avg Volume:         1,245,678
Earnings Reports:   4
Final Market Cap:   $9.52B
```

## Customizing the Test

### Modify the Scenario

Edit `test_realistic_scenario.py` to:

1. **Change company parameters:**
```python
generator = StockDataGenerator(
    ticker="MYSTOCK",
    initial_price=150.0,
    shares_outstanding=50_000_000,
    # ... other parameters
)
```

2. **Add/modify news events:**
```python
generator.add_news_event(
    datetime(2025, 3, 15),
    sentiment=0.8,
    description="Major product release",
    volatility_spike=1.6
)
```

3. **Adjust targets:**
```python
generator.add_price_target(datetime(2025, 6, 30), 200.0)
```

4. **Change volatility schedule:**
```python
generator.set_volatility_schedule([
    (datetime(2025, 1, 1), 0.20),
    (datetime(2025, 7, 1), 0.40),
])
```

5. **Modify regimes:**
```python
generator.set_regime_schedule([
    (datetime(2025, 1, 1), 'bull'),
    (datetime(2025, 6, 1), 'bear'),
])
```

### Create Additional Tests

Add new integration test scripts for different scenarios:
- **Value stock**: Low volatility, stable growth
- **Penny stock**: Very high volatility, explosive moves
- **Dividend stock**: Stable, predictable behavior
- **Turnaround story**: Bear → bull transition
- **Market crash scenario**: Sharp decline with recovery

## Interpreting Results

### Successful Test Characteristics

✅ **Data Quality:**
- No missing values
- Proper date sequencing (no weekends)
- High/Low relationships maintained
- Positive prices and volumes

✅ **Controllable Inputs:**
- News events visible in sentiment chart
- Volatility changes evident in volatility chart
- Price trending toward targets
- Regime changes affecting price action

✅ **Statistical Properties:**
- Returns distribution approximately normal
- Volatility clustering visible
- Drawdowns recover over time
- Volume spikes on news/earnings

### Common Issues

❌ **Price crashes to near-zero:**
- Too many negative events
- Excessive volatility
- Conflicting targets
- Solution: Adjust sentiment balance and volatility

❌ **Unrealistic returns:**
- Target prices too aggressive
- Regime settings too extreme
- Solution: Use more conservative targets

❌ **Flat price action:**
- Insufficient volatility
- No events scheduled
- Solution: Add more volatility or events

## Integration with Testing Framework

To run as part of pytest suite:

```bash
pytest integration_tests/ -v -s
```

Note: Use `-s` flag to see console output including the visualization message.

## Continuous Integration

For CI/CD pipelines without display:

```yaml
- name: Run integration tests
  run: |
    MPLBACKEND=Agg python3 integration_tests/test_realistic_scenario.py
```

The test will generate PNG files that can be saved as artifacts.

## Performance

- **Generation Time**: ~2-5 seconds for 252 days
- **Visualization Time**: ~1-2 seconds
- **Total Test Time**: ~3-7 seconds

## Future Enhancements

Potential additions to integration tests:
- Multiple stock comparison
- Portfolio-level scenarios
- Options pricing integration
- Risk metrics calculation
- Backtesting framework integration
- Interactive dashboards
- Real-time streaming simulation

## Support

For issues or questions about integration tests:
1. Check that matplotlib is installed
2. Verify all controllable inputs are valid
3. Review console output for errors
4. Examine generated CSV for data quality
5. Check PNG visualization for expected patterns
