import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import config
from data.fetch_data import get_spy_data


def forecast() -> dict:
    """Monte Carlo simulation using Geometric Brownian Motion"""
    df = get_spy_data()
    returns = df['Returns'].dropna()
    
    # Estimate drift and volatility from historical returns
    mu = returns.mean() * config.TRADING_DAYS_PER_YEAR
    sigma = returns.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)
    
    # Simulation parameters
    S0 = df['Close'].iloc[-1]
    forecast_dates = pd.date_range(config.FORECAST_START_DATE, config.FORECAST_END_DATE, freq='B', tz=df.index.tz)
    T = len(forecast_dates) / config.TRADING_DAYS_PER_YEAR
    dt = 1 / config.TRADING_DAYS_PER_YEAR
    n_steps = len(forecast_dates)
    
    np.random.seed(config.RANDOM_SEED)
    
    # Vectorized GBM: dS = mu*S*dt + sigma*S*dW
    Z = np.random.standard_normal((config.MONTE_CARLO_PATHS, n_steps))
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    
    # Cumulative sum and exponential to get price paths
    price_paths = S0 * np.exp(np.cumsum(drift + diffusion, axis=1))
    
    # Calculate mean and percentiles across paths
    mean_path = price_paths.mean(axis=0)
    lower_bound = np.percentile(price_paths, (1 - config.CONFIDENCE_LEVEL) / 2 * 100, axis=0)
    upper_bound = np.percentile(price_paths, (1 + config.CONFIDENCE_LEVEL) / 2 * 100, axis=0)
    
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecast': mean_path,
        'Lower': lower_bound,
        'Upper': upper_bound
    }).set_index('Date')
    
    return {
        'name': 'Monte Carlo (GBM)',
        'forecast_df': forecast_df,
        'end_2026_target': forecast_df['Forecast'].iloc[-1],
        'confidence_interval': (forecast_df['Lower'].iloc[-1], forecast_df['Upper'].iloc[-1])
    }


if __name__ == "__main__":
    result = forecast()
    print(f"{result['name']}: End-2026 Target = ${result['end_2026_target']:.2f}")
    print(f"95% CI: [${result['confidence_interval'][0]:.2f}, ${result['confidence_interval'][1]:.2f}]")