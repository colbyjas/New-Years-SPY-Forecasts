import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import config
from data.fetch_data import get_spy_data


def forecast() -> dict:
    """ARIMA + GARCH time series forecast"""
    df = get_spy_data()
    returns = df['Returns'].dropna()
    
    # Fit ARIMA(1,0,1) to returns for mean dynamics
    arima = ARIMA(returns, order=(1, 0, 1)).fit()
    
    # Fit GARCH(1,1) to returns for volatility dynamics
    garch = arch_model(returns, vol='Garch', p=1, q=1).fit(disp='off')
    
    # Forecast returns
    n_periods = len(pd.date_range(config.FORECAST_START_DATE, config.FORECAST_END_DATE, freq='B'))
    arima_forecast = arima.forecast(steps=n_periods)
    garch_forecast = garch.forecast(horizon=n_periods)
    
    # Convert return forecasts to price forecasts
    forecast_returns = arima_forecast.values
    
    # Extract GARCH variance forecast - handle different shapes
    var_forecast = garch_forecast.variance.values
    if var_forecast.ndim == 2:
        # 2D array: [n_simulations, horizon], take last simulation
        forecast_vol = np.sqrt(var_forecast[-1, :])
    else:
        # 1D array: just horizon
        forecast_vol = np.sqrt(var_forecast)
    
    start_price = df['Close'].iloc[-1]
    forecast_prices = [start_price]
    
    for ret in forecast_returns:
        forecast_prices.append(forecast_prices[-1] * np.exp(ret))
    
    forecast_prices = np.array(forecast_prices[1:])
    
    # Confidence intervals using GARCH volatility forecast
    z_score = 1.96
    # Use average volatility over horizon with moderate time scaling
    avg_vol = forecast_vol.mean()
    time_steps = np.arange(1, len(forecast_prices) + 1)
    # More conservative scaling: sqrt of normalized time
    vol_scaling = np.sqrt(time_steps / config.TRADING_DAYS_PER_YEAR)
    lower = forecast_prices * np.exp(-z_score * avg_vol * vol_scaling)
    upper = forecast_prices * np.exp(z_score * avg_vol * vol_scaling)
    
    forecast_dates = pd.date_range(config.FORECAST_START_DATE, config.FORECAST_END_DATE, freq='B', tz=df.index.tz)
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecast': forecast_prices,
        'Lower': lower,
        'Upper': upper
    }).set_index('Date')
    
    return {
        'name': 'ARIMA-GARCH',
        'forecast_df': forecast_df,
        'end_2026_target': forecast_df['Forecast'].iloc[-1],
        'confidence_interval': (forecast_df['Lower'].iloc[-1], forecast_df['Upper'].iloc[-1])
    }


if __name__ == "__main__":
    result = forecast()
    print(f"{result['name']}: End-2026 Target = ${result['end_2026_target']:.2f}")
    print(f"95% CI: [${result['confidence_interval'][0]:.2f}, ${result['confidence_interval'][1]:.2f}]")