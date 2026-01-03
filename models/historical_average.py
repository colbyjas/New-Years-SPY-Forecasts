import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import config
from data.fetch_data import get_spy_data


def forecast() -> dict:
    """Historical average return projection"""
    df = get_spy_data()
    
    # Calculate annualized return
    total_days = (df.index[-1] - df.index[0]).days
    years = total_days / 365.25
    total_return = df['Close'].iloc[-1] / df['Close'].iloc[0] - 1
    annualized_return = (1 + total_return) ** (1 / years) - 1
    
    # Generate forecast dates (match timezone)
    forecast_dates = pd.date_range(
        start=config.FORECAST_START_DATE,
        end=config.FORECAST_END_DATE,
        freq='B',
        tz=df.index.tz
    )
    
    # Project forward using compound growth
    start_price = df['Close'].iloc[-1]
    days_from_start = (forecast_dates - df.index[-1]).days
    forecast_prices = start_price * (1 + annualized_return) ** (days_from_start / 365.25)
    
    # Add confidence intervals based on historical volatility
    returns = df['Returns'].dropna()
    annual_vol = returns.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)
    z_score = 1.96
    
    # Time-scaled volatility for expanding uncertainty
    time_in_years = days_from_start / 365.25
    vol_scaling = np.sqrt(time_in_years)
    lower = forecast_prices * np.exp(-z_score * annual_vol * vol_scaling)
    upper = forecast_prices * np.exp(z_score * annual_vol * vol_scaling)
    
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecast': forecast_prices,
        'Lower': lower,
        'Upper': upper
    }).set_index('Date')
    
    return {
        'name': 'Historical Average',
        'forecast_df': forecast_df,
        'end_2026_target': forecast_df['Forecast'].iloc[-1],
        'confidence_interval': (forecast_df['Lower'].iloc[-1], forecast_df['Upper'].iloc[-1])
    }


if __name__ == "__main__":
    result = forecast()
    print(f"{result['name']}: End-2026 Target = ${result['end_2026_target']:.2f}")
    print(f"95% CI: [${result['confidence_interval'][0]:.2f}, ${result['confidence_interval'][1]:.2f}]")