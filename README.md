# Options Analysis Scripts

Python scripts for options analysis.

## Setup

### Prerequisites
- Python 3.11 or higher

### Installation

1. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Test the environment with the hello world script:
```bash
python3 hello_world.py
```

## Modules

### Data Generator (`data_generator/`)

Sophisticated simulated stock price data generator with regime-aware stochastic modeling.

**Features:**
- Volatility clustering (GARCH-like dynamics)
- Valuation-based drift adjustment (P/E vs growth)
- Earnings date volatility spikes
- Benchmark correlation
- Comprehensive financial metrics (EPS, P/E, Beta, Revenue Growth, etc.)

**Quick Start:**
```python
from data_generator import StockDataGenerator

# Create and generate data
generator = StockDataGenerator(
    ticker="AAPL",
    initial_price=150.0,
    initial_eps=5.0,
    seed=42
)

data = generator.generate(n_days=252)  # 1 year
generator.save_to_csv(data, 'stock_data.csv')
```

See [data_generator/README.md](data_generator/README.md) for detailed documentation.

**Example:**
```bash
cd data_generator
python3 example_usage.py
```

## Scripts

- `hello_world.py` - Simple test script to verify Python environment
