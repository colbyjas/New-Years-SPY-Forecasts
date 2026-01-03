"""Configuration for SPY 2026 Forecast Project"""

import datetime as dt

# Data parameters
TICKER = 'SPY'
DATA_START_DATE = '2015-01-01'
TRAIN_END_DATE = '2025-12-31'
FORECAST_START_DATE = '2026-01-01'
FORECAST_END_DATE = '2026-12-31'

# Model parameters
RANDOM_SEED = 42
MONTE_CARLO_PATHS = 10000
CONFIDENCE_LEVEL = 0.95
TRADING_DAYS_PER_YEAR = 252
N_REGIMES = 2

# Visualization parameters
DASHBOARD_FIGSIZE = (18, 12)
SUBPLOT_FIGSIZE = (10, 6)
HISTORICAL_COLOR = '#2E86AB'
FORECAST_COLOR = '#A23B72'
CI_COLOR = '#F18F01'
CI_ALPHA = 0.2
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
SAVE_DPI = 300

# Output directories
OUTPUT_DIR = 'outputs'
FIGURES_DIR = f'{OUTPUT_DIR}/figures'
RESULTS_DIR = f'{OUTPUT_DIR}/results'