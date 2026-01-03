import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import yfinance as yf
import pandas as pd
import numpy as np
import config


def get_spy_data(start: str = config.DATA_START_DATE, end: str = config.TRAIN_END_DATE) -> pd.DataFrame:
    """Fetches and cleans SPY data with log returns in a vectorized pipeline."""
    df = yf.Ticker(config.TICKER).history(start=start, end=end)
    if df.empty:
        raise ValueError(f"No data for {config.TICKER}")
    
    # Method chaining: clean and calculate in one pass
    df = (df.loc[~df.index.duplicated()]
            .sort_index()
            .ffill()
            .bfill())
    
    # Vectorized return calculation
    df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    return df.dropna()


if __name__ == "__main__":
    data = get_spy_data()
    print(f"Fetched {len(data)} rows | Range: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"Total Return: {(data['Close'].iloc[-1]/data['Close'].iloc[0] - 1):.2%}")