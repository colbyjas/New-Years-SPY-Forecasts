import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict
import config
from data.fetch_data import get_spy_data


def plot_single_model(result: dict, historical_data: pd.DataFrame, save_path: str = None):
    """Plot single model forecast with historical data and confidence intervals"""
    fig, ax = plt.subplots(figsize=config.SUBPLOT_FIGSIZE)
    
    # Plot historical prices
    ax.plot(historical_data.index, historical_data['Close'], 
            color=config.HISTORICAL_COLOR, linewidth=2, label='Historical')
    
    # Plot forecast
    forecast_df = result['forecast_df']
    ax.plot(forecast_df.index, forecast_df['Forecast'],
            color=config.HISTORICAL_COLOR, linewidth=2, label='Forecast', linestyle='--')
    
    # Plot confidence interval if available
    if result['confidence_interval'] is not None and 'Lower' in forecast_df.columns:
        ax.fill_between(forecast_df.index, forecast_df['Lower'], forecast_df['Upper'],
                        color=config.CI_COLOR, alpha=config.CI_ALPHA, label='95% CI')
    
    # Formatting
    ax.set_title(f"{result['name']} - End 2026: ${result['end_2026_target']:.2f}", 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('SPY Price ($)', fontsize=11)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.SAVE_DPI, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_dashboard(results: List[dict], save_path: str = None):
    """Create 2x3 dashboard showing all model forecasts"""
    fig, axes = plt.subplots(2, 3, figsize=config.DASHBOARD_FIGSIZE)
    axes = axes.flatten()
    
    # Get historical data
    historical_data = get_spy_data()
    
    for idx, result in enumerate(results):
        ax = axes[idx]
        
        # Plot historical
        ax.plot(historical_data.index, historical_data['Close'],
               color=config.HISTORICAL_COLOR, linewidth=1.5, alpha=0.8)
        
        # Plot forecast
        forecast_df = result['forecast_df']
        ax.plot(forecast_df.index, forecast_df['Forecast'],
               color=config.HISTORICAL_COLOR, linewidth=2, linestyle='--')
        
        # Add CI if available
        if result['confidence_interval'] is not None and 'Lower' in forecast_df.columns:
            ax.fill_between(forecast_df.index, forecast_df['Lower'], forecast_df['Upper'],
                           color=config.CI_COLOR, alpha=config.CI_ALPHA)
        
        # Formatting
        ax.set_title(f"{result['name']}\nTarget: ${result['end_2026_target']:.2f}",
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.suptitle('SPY End-2026 Price Forecasts: Method Comparison', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.SAVE_DPI, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_summary_table(results: List[dict]) -> pd.DataFrame:
    """Create summary table of all forecasts"""
    summary_data = []
    
    for result in results:
        row = {
            'Method': result['name'],
            'End-2026 Target': f"${result['end_2026_target']:.2f}",
        }
        
        if result['confidence_interval'] is not None:
            lower, upper = result['confidence_interval']
            row['95% CI Lower'] = f"${lower:.2f}"
            row['95% CI Upper'] = f"${upper:.2f}"
        else:
            row['95% CI Lower'] = 'N/A'
            row['95% CI Upper'] = 'N/A'
        
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    
    # Add average row
    targets = [r['end_2026_target'] for r in results]
    avg_row = {
        'Method': 'Average',
        'End-2026 Target': f"${np.mean(targets):.2f}",
        '95% CI Lower': '-',
        '95% CI Upper': '-'
    }
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    
    return df


def plot_comparison_bar(results: List[dict], save_path: str = None):
    """Create bar chart comparing end-2026 targets across methods"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = [r['name'] for r in results]
    targets = [r['end_2026_target'] for r in results]
    
    bars = ax.bar(methods, targets, color=config.HISTORICAL_COLOR, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'${height:.2f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add average line
    avg_target = np.mean(targets)
    ax.axhline(avg_target, color='red', linestyle='--', linewidth=2, 
              label=f'Average: ${avg_target:.2f}')
    
    ax.set_ylabel('End-2026 Price Target ($)', fontsize=12)
    ax.set_title('SPY End-2026 Price Targets by Forecasting Method', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.SAVE_DPI, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # Quick test - import models directly
    import sys
    from pathlib import Path
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))
    sys.path.insert(0, str(parent_dir / 'models'))
    
    from historical_average import forecast as historical_average
    from linear_trend import forecast as linear_trend
    from arima_garch import forecast as arima_garch
    from monte_carlo import forecast as monte_carlo
    from exponential_smoothing import forecast as exponential_smoothing
    from regime_based import forecast as regime_based
    
    ALL_MODELS = [
        historical_average,
        linear_trend,
        arima_garch,
        monte_carlo,
        exponential_smoothing,
        regime_based
    ]
    
    print("Generating all forecasts...")
    results = [model() for model in ALL_MODELS]
    
    print("\nSummary Table:")
    summary = create_summary_table(results)
    print(summary.to_string(index=False))
    
    print("\nGenerating visualizations...")
    historical = get_spy_data()
    
    # Create figures dir if needed
    Path(config.FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    
    # Individual plots
    for result in results:
        filename = result['name'].lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
        plot_single_model(result, historical, 
                         save_path=f"{config.FIGURES_DIR}/{filename}.png")
    
    # Dashboard
    create_dashboard(results, save_path=f"{config.FIGURES_DIR}/dashboard.png")
    
    # Comparison bar chart
    plot_comparison_bar(results, save_path=f"{config.FIGURES_DIR}/comparison.png")
    
    print(f"\nAll figures saved to {config.FIGURES_DIR}/")