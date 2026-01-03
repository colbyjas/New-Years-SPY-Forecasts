import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import config
from data.fetch_data import get_spy_data


def forecast() -> dict:
    """Holt-Winters exponential smoothing with trend and seasonality"""
    df = get_spy_data()
    
    # Fit Holt-Winters model (additive trend, no seasonality for simplicity)
    model = ExponentialSmoothing(
        df['Close'],
        trend='add',
        seasonal=None,
        initialization_method='estimated'
    ).fit()
    
    # Generate forecast
    forecast_dates = pd.date_range(config.FORECAST_START_DATE, config.FORECAST_END_DATE, freq='B', tz=df.index.tz)
    n_periods = len(forecast_dates)
    
    forecast_result = model.forecast(steps=n_periods)
    
    # Calculate prediction intervals using model's residuals
    residuals = df['Close'] - model.fittedvalues
    std_error = residuals.std()
    z_score = 1.96
    
    # Expanding uncertainty over time (simple approach)
    time_factor = np.sqrt(np.arange(1, n_periods + 1))
    lower = forecast_result.values - z_score * std_error * time_factor
    upper = forecast_result.values + z_score * std_error * time_factor
    
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecast': forecast_result.values,
        'Lower': lower,
        'Upper': upper
    }).set_index('Date')
    
    return {
        'name': 'Exponential Smoothing',
        'forecast_df': forecast_df,
        'end_2026_target': forecast_df['Forecast'].iloc[-1],
        'confidence_interval': (forecast_df['Lower'].iloc[-1], forecast_df['Upper'].iloc[-1])
    }


if __name__ == "__main__":
    result = forecast()
    print(f"{result['name']}: End-2026 Target = ${result['end_2026_target']:.2f}")
    print(f"95% CI: [${result['confidence_interval'][0]:.2f}, ${result['confidence_interval'][1]:.2f}]")