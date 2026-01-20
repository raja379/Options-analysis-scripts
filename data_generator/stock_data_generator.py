"""
Stock Data Generator with Regime-Aware Stochastic Model

Generates realistic simulated stock price data with:
- Volatility clustering (GARCH-like behavior)
- Valuation-based drift adjustment (P/E vs growth)
- Earnings date volatility spikes
- Benchmark correlation
- Market cap dynamics
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple


class StockDataGenerator:
    """
    Generate simulated daily stock price data with realistic financial characteristics.

    The model incorporates:
    - Regime switching (bull, bear, neutral markets)
    - Volatility clustering with GARCH-like dynamics
    - Mean reversion in valuation metrics
    - Earnings announcement effects
    - Correlation with benchmark index
    """

    def __init__(
        self,
        ticker: str = "SIM",
        initial_price: float = 100.0,
        shares_outstanding: float = 100_000_000,
        initial_eps: float = 5.0,
        initial_revenue_growth: float = 0.15,
        initial_beta: float = 1.0,
        benchmark_correlation: float = 0.7,
        seed: Optional[int] = None
    ):
        """
        Initialize the stock data generator.

        Args:
            ticker: Stock ticker symbol
            initial_price: Starting stock price
            shares_outstanding: Number of shares outstanding
            initial_eps: Initial earnings per share (TTM)
            initial_revenue_growth: Initial revenue growth rate
            initial_beta: Beta coefficient vs benchmark
            benchmark_correlation: Correlation with benchmark index
            seed: Random seed for reproducibility
        """
        self.ticker = ticker
        self.initial_price = initial_price
        self.shares_outstanding = shares_outstanding
        self.initial_eps = initial_eps
        self.initial_revenue_growth = initial_revenue_growth
        self.initial_beta = initial_beta
        self.benchmark_correlation = benchmark_correlation

        if seed is not None:
            np.random.seed(seed)

        # Market regime parameters
        self.regimes = {
            'bull': {'drift': 0.15, 'vol_multiplier': 0.8, 'transition_prob': 0.02},
            'neutral': {'drift': 0.08, 'vol_multiplier': 1.0, 'transition_prob': 0.03},
            'bear': {'drift': -0.10, 'vol_multiplier': 1.5, 'transition_prob': 0.02}
        }
        self.current_regime = 'neutral'

        # Volatility parameters (GARCH-like)
        self.base_volatility = 0.20  # 20% annualized
        self.vol_persistence = 0.90  # How much previous volatility carries forward
        self.vol_mean_reversion = 0.08  # Speed of mean reversion
        self.shock_sensitivity = 0.02  # Sensitivity to return shocks

        # Earnings parameters
        self.earnings_frequency = 90  # Days between earnings
        self.earnings_vol_spike = 2.5  # Volatility multiplier on earnings days

    def _get_regime_transition(self) -> str:
        """Determine if regime should transition based on probabilities."""
        transition_prob = self.regimes[self.current_regime]['transition_prob']

        if np.random.random() < transition_prob:
            # Transition to a different regime
            other_regimes = [r for r in self.regimes.keys() if r != self.current_regime]
            return np.random.choice(other_regimes)

        return self.current_regime

    def _calculate_valuation_drift(
        self,
        current_pe: float,
        growth_rate: float,
        base_drift: float
    ) -> float:
        """
        Adjust drift based on valuation metrics.

        Uses PEG ratio logic: overvalued stocks (high P/E, low growth)
        drift down, undervalued stocks drift up.
        """
        # Calculate PEG ratio (P/E / growth%)
        growth_pct = growth_rate * 100
        if growth_pct <= 0:
            growth_pct = 1.0  # Avoid division by zero

        peg_ratio = current_pe / growth_pct

        # Adjust drift based on valuation
        # PEG < 1: undervalued, increase drift
        # PEG > 2: overvalued, decrease drift
        if peg_ratio < 1.0:
            valuation_adjustment = 0.05 * (1.0 - peg_ratio)
        elif peg_ratio > 2.0:
            valuation_adjustment = -0.03 * (peg_ratio - 2.0)
        else:
            valuation_adjustment = 0.0

        return base_drift + valuation_adjustment

    def _update_volatility(
        self,
        current_vol: float,
        recent_return: float,
        is_earnings_day: bool
    ) -> float:
        """
        Update volatility with clustering and earnings effects.

        Uses GARCH(1,1)-like dynamics with mean reversion.
        """
        return self._update_volatility_with_regime(
            current_vol, recent_return, is_earnings_day, self.base_volatility
        )

    def _update_volatility_with_regime(
        self,
        current_vol: float,
        recent_return: float,
        is_earnings_day: bool,
        base_vol_annual: float
    ) -> float:
        """
        Update volatility with clustering, regime, and earnings effects.

        Uses GARCH(1,1)-like dynamics with mean reversion.

        Args:
            current_vol: Current daily volatility
            recent_return: Most recent return
            is_earnings_day: Whether it's an earnings announcement day
            base_vol_annual: Annual base volatility (regime-adjusted)
        """
        # Convert base volatility to daily
        daily_base_vol = base_vol_annual / np.sqrt(252)

        # Squared returns for GARCH dynamics
        shock_squared = recent_return ** 2

        # GARCH(1,1) update: variance = omega + alpha*shock^2 + beta*variance
        variance = (
            self.vol_mean_reversion * (daily_base_vol ** 2) +
            self.shock_sensitivity * shock_squared +
            self.vol_persistence * (current_vol ** 2)
        )

        new_vol = np.sqrt(variance)

        # Bound volatility to prevent explosion
        min_vol = daily_base_vol * 0.5
        max_vol = daily_base_vol * 3.0
        new_vol = np.clip(new_vol, min_vol, max_vol)

        # Spike volatility on earnings days
        if is_earnings_day:
            earnings_spike = min(self.earnings_vol_spike, 2.0)
            new_vol = min(new_vol * earnings_spike, max_vol * 1.2)

        return new_vol

    def _generate_benchmark_returns(self, n_days: int) -> np.ndarray:
        """Generate correlated benchmark index returns."""
        # Benchmark has lower volatility and steady drift
        benchmark_vol = 0.15
        benchmark_drift = 0.10

        daily_drift = benchmark_drift / 252
        daily_vol = benchmark_vol / np.sqrt(252)

        returns = np.random.normal(daily_drift, daily_vol, n_days)
        return returns

    def _generate_correlated_shock(
        self,
        benchmark_return: float,
        idiosyncratic_vol: float
    ) -> float:
        """
        Generate stock return correlated with benchmark.

        Uses Cholesky decomposition for correlation structure.
        """
        # Decompose into systematic and idiosyncratic components
        systematic_component = self.initial_beta * benchmark_return

        # Idiosyncratic shock (uncorrelated)
        correlation_matrix = np.array([
            [1.0, self.benchmark_correlation],
            [self.benchmark_correlation, 1.0]
        ])

        # Generate correlated random variables
        independent_shocks = np.random.normal(0, 1, 2)
        cholesky = np.linalg.cholesky(correlation_matrix)
        correlated_shocks = cholesky @ independent_shocks

        # Stock return = beta * benchmark + idiosyncratic
        stock_shock = correlated_shocks[1] * idiosyncratic_vol

        return systematic_component + stock_shock

    def _calculate_high_low(
        self,
        open_price: float,
        close_price: float,
        volatility: float
    ) -> Tuple[float, float]:
        """Calculate realistic high and low prices for the day."""
        # Intraday volatility is typically higher
        intraday_vol = volatility * 1.5

        # High and low based on open and close
        price_range = abs(close_price - open_price)
        additional_range = open_price * intraday_vol * np.random.uniform(0.5, 1.5)

        high = max(open_price, close_price) + additional_range * np.random.uniform(0.3, 0.7)
        low = min(open_price, close_price) - additional_range * np.random.uniform(0.3, 0.7)

        # Ensure high >= max(open, close) and low <= min(open, close)
        high = max(high, open_price, close_price)
        low = min(low, open_price, close_price)

        return high, low

    def _calculate_volume(
        self,
        base_volume: float,
        price_change: float,
        volatility: float,
        is_earnings_day: bool
    ) -> int:
        """Calculate trading volume with realistic dynamics."""
        # Volume increases with volatility and price changes
        vol_multiplier = 1.0 + volatility * 2.0
        price_change_multiplier = 1.0 + abs(price_change) * 3.0

        # Earnings days have higher volume
        earnings_multiplier = 2.0 if is_earnings_day else 1.0

        # Add random noise
        noise = np.random.lognormal(0, 0.3)

        volume = base_volume * vol_multiplier * price_change_multiplier * earnings_multiplier * noise

        return int(volume)

    def _update_fundamentals(
        self,
        day: int,
        is_earnings_day: bool,
        current_eps: float,
        current_growth: float
    ) -> Tuple[float, float]:
        """Update EPS and revenue growth with realistic dynamics."""
        if is_earnings_day:
            # Quarterly earnings update
            # Growth trends with some noise
            growth_change = np.random.normal(0, 0.02)
            new_growth = current_growth + growth_change
            new_growth = np.clip(new_growth, -0.3, 0.5)  # Realistic bounds

            # EPS grows with revenue (with some lag and variation)
            eps_growth = new_growth * np.random.uniform(0.8, 1.2)
            new_eps = current_eps * (1 + eps_growth / 4)  # Quarterly growth

            return new_eps, new_growth
        else:
            # Gradual updates between earnings
            return current_eps, current_growth

    def _calculate_news_sentiment(self, price_change: float, is_earnings_day: bool) -> float:
        """
        Generate news sentiment score (-1 to 1).

        Correlated with price changes but with noise.
        """
        # Base sentiment from price change
        base_sentiment = np.tanh(price_change * 10)  # Scale and bound

        # Add noise
        noise = np.random.normal(0, 0.2)
        sentiment = base_sentiment + noise

        # Earnings days have more extreme sentiment
        if is_earnings_day:
            sentiment *= 1.5

        return np.clip(sentiment, -1, 1)

    def generate(
        self,
        n_days: int = 252,
        start_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Generate simulated stock price data.

        Args:
            n_days: Number of trading days to simulate
            start_date: Starting date (defaults to today - n_days)

        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume, MarketCap,
            EPS_TTM, PE, RevenueGrowth, Beta, Volatility, Drawdown, DividendYield,
            EarningsFlag, NewsSentiment
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=int(n_days * 1.4))  # Account for weekends

        # Initialize arrays
        dates = []
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []

        # Initialize state variables
        current_price = self.initial_price
        current_vol = self.base_volatility / np.sqrt(252)  # Daily volatility
        current_eps = self.initial_eps
        current_growth = self.initial_revenue_growth
        base_volume = 1_000_000

        # Track for drawdown calculation
        running_max = current_price

        # Generate benchmark returns
        benchmark_returns = self._generate_benchmark_returns(n_days)

        # Storage for calculated metrics
        eps_values = []
        pe_values = []
        growth_values = []
        beta_values = []
        vol_values = []
        drawdown_values = []
        dividend_yields = []
        earnings_flags = []
        sentiment_values = []
        market_caps = []

        current_date = start_date
        day_count = 0

        while len(closes) < n_days:
            # Skip weekends
            if current_date.weekday() >= 5:
                current_date += timedelta(days=1)
                continue

            # Check if it's an earnings day
            is_earnings_day = (day_count % self.earnings_frequency == 0) and day_count > 0

            # Update market regime
            self.current_regime = self._get_regime_transition()
            regime_params = self.regimes[self.current_regime]

            # Calculate P/E ratio
            current_pe = current_price / current_eps if current_eps > 0 else 20

            # Calculate drift with valuation adjustment
            annual_drift = self._calculate_valuation_drift(
                current_pe, current_growth, regime_params['drift']
            )
            daily_drift = annual_drift / 252

            # Update volatility with clustering
            recent_return = 0 if len(closes) < 2 else (closes[-1] - closes[-2]) / closes[-2]

            # Adjust base volatility for current regime
            regime_adjusted_base_vol = self.base_volatility * regime_params['vol_multiplier']

            # Update volatility with regime-adjusted base
            current_vol = self._update_volatility_with_regime(
                current_vol, recent_return, is_earnings_day, regime_adjusted_base_vol
            )

            # Generate correlated return
            shock = self._generate_correlated_shock(benchmark_returns[day_count], current_vol)
            daily_return = daily_drift + shock

            # Update price
            open_price = current_price
            close_price = open_price * (1 + daily_return)
            close_price = max(0.01, close_price)  # Prevent negative prices

            # Calculate high and low
            high, low = self._calculate_high_low(open_price, close_price, current_vol)

            # Calculate volume
            price_change = (close_price - open_price) / open_price
            volume = self._calculate_volume(base_volume, price_change, current_vol, is_earnings_day)

            # Update fundamentals
            current_eps, current_growth = self._update_fundamentals(
                day_count, is_earnings_day, current_eps, current_growth
            )

            # Calculate metrics
            pe_ratio = close_price / current_eps if current_eps > 0 else np.nan
            market_cap = close_price * self.shares_outstanding

            # Update running max for drawdown
            running_max = max(running_max, close_price)
            drawdown = (close_price - running_max) / running_max

            # Dividend yield (stable with small variations)
            base_dividend_yield = 0.02
            dividend_yield = base_dividend_yield + np.random.normal(0, 0.002)
            dividend_yield = max(0, dividend_yield)

            # Beta (slowly varying around initial value)
            beta = self.initial_beta + np.random.normal(0, 0.05)

            # Annualized volatility for output
            annual_vol = current_vol * np.sqrt(252)

            # News sentiment
            sentiment = self._calculate_news_sentiment(price_change, is_earnings_day)

            # Store values
            dates.append(current_date)
            opens.append(round(open_price, 2))
            highs.append(round(high, 2))
            lows.append(round(low, 2))
            closes.append(round(close_price, 2))
            volumes.append(volume)
            market_caps.append(round(market_cap, 2))
            eps_values.append(round(current_eps, 2))
            pe_values.append(round(pe_ratio, 2))
            growth_values.append(round(current_growth, 4))
            beta_values.append(round(beta, 3))
            vol_values.append(round(annual_vol, 4))
            drawdown_values.append(round(drawdown, 4))
            dividend_yields.append(round(dividend_yield, 4))
            earnings_flags.append(1 if is_earnings_day else 0)
            sentiment_values.append(round(sentiment, 3))

            # Update for next iteration
            current_price = close_price
            current_date += timedelta(days=1)
            day_count += 1

        # Create DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': closes,
            'Volume': volumes,
            'MarketCap': market_caps,
            'EPS_TTM': eps_values,
            'PE': pe_values,
            'RevenueGrowth': growth_values,
            'Beta': beta_values,
            'Volatility': vol_values,
            'Drawdown': drawdown_values,
            'DividendYield': dividend_yields,
            'EarningsFlag': earnings_flags,
            'NewsSentiment': sentiment_values
        })

        return df

    def save_to_csv(self, df: pd.DataFrame, filename: str) -> None:
        """Save generated data to CSV file."""
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

    def get_summary_statistics(self, df: pd.DataFrame) -> dict:
        """Calculate summary statistics for the generated data."""
        returns = df['Close'].pct_change().dropna()

        stats = {
            'total_days': len(df),
            'start_date': df['Date'].iloc[0],
            'end_date': df['Date'].iloc[-1],
            'start_price': df['Close'].iloc[0],
            'end_price': df['Close'].iloc[-1],
            'total_return': (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1),
            'annualized_return': (df['Close'].iloc[-1] / df['Close'].iloc[0]) ** (252 / len(df)) - 1,
            'volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0,
            'max_drawdown': df['Drawdown'].min(),
            'avg_volume': df['Volume'].mean(),
            'earnings_days': df['EarningsFlag'].sum(),
            'avg_pe': df['PE'].mean(),
            'final_market_cap': df['MarketCap'].iloc[-1]
        }

        return stats
