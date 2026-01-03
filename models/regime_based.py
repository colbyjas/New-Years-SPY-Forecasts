import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import config
from data.fetch_data import get_spy_data


def forecast() -> dict:
    """Markov regime-switching model for market state dynamics"""
    df = get_spy_data()
    returns = df['Returns'].dropna()
    
    # Fit 2-regime switching model
    model = MarkovRegression(
        returns,
        k_regimes=config.N_REGIMES,
        switching_variance=True
    ).fit()
    
    # Use smoothed probabilities to identify regimes
    probs = model.smoothed_marginal_probabilities
    
    # Calculate regime-specific return statistics
    regime_0_mask = probs.iloc[:, 0] > 0.5
    regime_1_mask = probs.iloc[:, 1] > 0.5
    
    # Get returns for each regime
    regime_0_returns = returns[regime_0_mask]
    regime_1_returns = returns[regime_1_mask]
    
    # Calculate mean and std for each regime
    mu_0 = regime_0_returns.mean() * config.TRADING_DAYS_PER_YEAR
    mu_1 = regime_1_returns.mean() * config.TRADING_DAYS_PER_YEAR
    vol_0 = regime_0_returns.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)
    vol_1 = regime_1_returns.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)
    
    # Weight by final regime probabilities
    final_probs = probs.iloc[-1].values
    expected_return = final_probs[0] * mu_0 + final_probs[1] * mu_1
    expected_vol = np.sqrt(final_probs[0] * vol_0**2 + final_probs[1] * vol_1**2)
    
    # Generate forecast dates
    forecast_dates = pd.date_range(config.FORECAST_START_DATE, config.FORECAST_END_DATE, freq='B', tz=df.index.tz)
    n_periods = len(forecast_dates)
    
    # Project forward using expected return
    S0 = df['Close'].iloc[-1]
    days_forward = np.arange(1, n_periods + 1)
    forecast_prices = S0 * np.exp(expected_return * days_forward / config.TRADING_DAYS_PER_YEAR)
    
    # Confidence intervals using expected volatility
    z_score = 1.96
    time_factor = np.sqrt(days_forward / config.TRADING_DAYS_PER_YEAR)
    lower = forecast_prices * np.exp(-z_score * expected_vol * time_factor)
    upper = forecast_prices * np.exp(z_score * expected_vol * time_factor)
    
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecast': forecast_prices,
        'Lower': lower,
        'Upper': upper
    }).set_index('Date')
    
    return {
        'name': 'Regime Switching',
        'forecast_df': forecast_df,
        'end_2026_target': forecast_df['Forecast'].iloc[-1],
        'confidence_interval': (forecast_df['Lower'].iloc[-1], forecast_df['Upper'].iloc[-1])
    }


if __name__ == "__main__":
    result = forecast()
    print(f"{result['name']}: End-2026 Target = ${result['end_2026_target']:.2f}")
    print(f"95% CI: [${result['confidence_interval'][0]:.2f}, ${result['confidence_interval'][1]:.2f}]")