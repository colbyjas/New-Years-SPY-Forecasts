import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import config
from data.fetch_data import get_spy_data


def forecast() -> dict:
    """Linear trend extrapolation via OLS regression"""
    df = get_spy_data()
    
    # Prepare X (days since start) and y (prices)
    df['days'] = (df.index - df.index[0]).days
    X = df['days'].values.reshape(-1, 1)
    y = df['Close'].values
    
    # Fit linear model
    model = LinearRegression().fit(X, y)
    
    # Generate forecast dates (match timezone)
    forecast_dates = pd.date_range(
        start=config.FORECAST_START_DATE,
        end=config.FORECAST_END_DATE,
        freq='B',
        tz=df.index.tz
    )
    
    # Predict on future dates
    future_days = (forecast_dates - df.index[0]).days.values.reshape(-1, 1)
    forecast_prices_raw = model.predict(future_days)
    
    # Adjust to start from last observed price (maintain slope but shift intercept)
    last_actual_price = df['Close'].iloc[-1]
    last_predicted_price = model.predict([[df['days'].iloc[-1]]])[0]
    adjustment = last_actual_price - last_predicted_price
    forecast_prices = forecast_prices_raw + adjustment
    
    # Calculate prediction intervals using residual std
    residuals = y - model.predict(X)
    std_error = np.std(residuals)
    z_score = 1.96  # 95% confidence
    
    # More conservative expanding uncertainty (sqrt of time normalized by data length)
    n_train = len(X)
    time_steps = np.arange(1, len(forecast_prices) + 1)
    # Scale uncertainty growth proportionally to avoid explosion
    expanding_factor = np.sqrt(1 + time_steps / n_train)
    expanding_std = std_error * expanding_factor
    
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecast': forecast_prices,
        'Lower': forecast_prices - z_score * expanding_std,
        'Upper': forecast_prices + z_score * expanding_std
    }).set_index('Date')
    
    return {
        'name': 'Linear Trend',
        'forecast_df': forecast_df,
        'end_2026_target': forecast_df['Forecast'].iloc[-1],
        'confidence_interval': (forecast_df['Lower'].iloc[-1], forecast_df['Upper'].iloc[-1])
    }


if __name__ == "__main__":
    result = forecast()
    print(f"{result['name']}: End-2026 Target = ${result['end_2026_target']:.2f}")
    print(f"95% CI: [${result['confidence_interval'][0]:.2f}, ${result['confidence_interval'][1]:.2f}]")