"""
Main script for SPY 2026 Forecast Project
Runs all models, generates visualizations, and saves outputs
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add models directory to path using absolute paths
project_root = Path(__file__).resolve().parent
models_dir = project_root / 'models'
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(models_dir))

import config
from data.fetch_data import get_spy_data
from historical_average import forecast as historical_average
from linear_trend import forecast as linear_trend
from arima_garch import forecast as arima_garch
from monte_carlo import forecast as monte_carlo
from exponential_smoothing import forecast as exponential_smoothing
from regime_based import forecast as regime_based
from visualization.plots import (
    plot_single_model, 
    create_dashboard, 
    create_summary_table,
    plot_comparison_bar
)

ALL_MODELS = [
    historical_average,
    linear_trend,
    arima_garch,
    monte_carlo,
    exponential_smoothing,
    regime_based
]


def create_output_dirs():
    """Create output directories if they don't exist"""
    Path(config.OUTPUT_DIR).mkdir(exist_ok=True)
    Path(config.FIGURES_DIR).mkdir(exist_ok=True)
    Path(config.RESULTS_DIR).mkdir(exist_ok=True)


def run_all_models():
    """Execute all forecasting models and return results"""
    print("=" * 60)
    print("SPY 2026 FORECAST - Running All Models")
    print("=" * 60)
    
    results = []
    for model in ALL_MODELS:
        try:
            print(f"\nRunning {model.__name__.replace('_', ' ').title()}...")
            result = model()
            results.append(result)
            print(f"  ✓ {result['name']}: ${result['end_2026_target']:.2f}")
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
    
    return results


def generate_visualizations(results):
    """Generate all plots and save to outputs"""
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    
    historical = get_spy_data()
    
    # Individual model plots
    print("\nCreating individual model plots...")
    for result in results:
        filename = result['name'].lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
        save_path = f"{config.FIGURES_DIR}/{filename}.png"
        plot_single_model(result, historical, save_path=save_path)
        print(f"  ✓ {result['name']} plot saved")
    
    # Dashboard
    print("\nCreating dashboard...")
    create_dashboard(results, save_path=f"{config.FIGURES_DIR}/dashboard.png")
    print("  ✓ Dashboard saved")
    
    # Comparison bar chart
    print("\nCreating comparison chart...")
    plot_comparison_bar(results, save_path=f"{config.FIGURES_DIR}/comparison.png")
    print("  ✓ Comparison chart saved")


def save_summary_table(results):
    """Generate and save summary table"""
    print("\n" + "=" * 60)
    print("Summary Table")
    print("=" * 60)
    
    summary = create_summary_table(results)
    print("\n" + summary.to_string(index=False))
    
    # Save to CSV
    summary.to_csv(f"{config.RESULTS_DIR}/forecast_summary.csv", index=False)
    print(f"\n✓ Summary table saved to {config.RESULTS_DIR}/forecast_summary.csv")


def main():
    """Main execution pipeline"""
    # Setup
    create_output_dirs()
    
    # Run models
    results = run_all_models()
    
    if not results:
        print("\n✗ No models completed successfully!")
        return
    
    # Generate outputs
    generate_visualizations(results)
    save_summary_table(results)
    
    # Final summary
    targets = [r['end_2026_target'] for r in results]
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Models Run: {len(results)}")
    print(f"Average Target: ${sum(targets)/len(targets):.2f}")
    print(f"Range: ${min(targets):.2f} - ${max(targets):.2f}")
    print(f"Spread: ${max(targets) - min(targets):.2f}")
    print("\n✓ All outputs saved to outputs/ directory")


if __name__ == "__main__":
    main()