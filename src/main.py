"""
Main entry point for the Stock Risk Engine pipeline.
"""

# main.py
from src.ingestion import DataIngestor
from src.database import create_medallion_schema, run_silver_and_gold_views, update_risk_inference, update_silver_risk_features, update_risk_metrics, get_universe_tickers_from_config, get_spotlight_tickers_from_config
from src.maintenance import archive_old_data
from src.setup_db import create_medallion_schema
from src.app_visualizer import plot_stock_risk, plot_stock_risk_with_panic, plot_correlation_heatmap
from src.config import DATABASE_PATH, REPORT_DIR
from src.app_visualizer2 import run_beta_drift_forecast_report
from src.app_visualizer3 import run_risk_performance_report
import os
import yaml
import argparse

def main():

    parser = argparse.ArgumentParser(description="Run the Stock Risk Engine pipeline.")
    parser.add_argument("--dockermode", action="store_true", help="Run in Docker mode.")
    args = parser.parse_args()

    print(f"Arguments received: {args.dockermode}") 
    docker_mode = args.dockermode

    print("--- Starting Stock Risk Engine ---")

    print(f"Running in {'Docker' if docker_mode else 'Local'} mode.")

    # 1. Initialize the database schema
    if docker_mode:
        print("Initializing database schema in Docker mode...")
        create_medallion_schema(initial_setup=True)
    
    # 2. Ingest Raw Data (Sourcing from tickers.yml inside the module)
    ingestor = DataIngestor()
    ingestor.run_bronze_ingestion()
    
    # 3. Build Analytical Views (SQL-based transformations)
    run_silver_and_gold_views()

    # 4. Update Silver Risk Features
    update_silver_risk_features()

    # 5. Update Risk Metrics
    update_risk_metrics()

    # 6. Update Risk Inference Table
    update_risk_inference()

    # 7. Cleanup & Archive
    archive_old_data()

    print("--- Pipeline Complete ---")

    # 8. Generate Visual Reports

    tickers = get_universe_tickers_from_config()  # Access the tickers list stored during ingestion
    print(f"Generating reports for tickers: {tickers}")

    for ticker in tickers:
        if ticker.startswith("^"):  # Skip indices for individual stock reports
            continue
        else:
            print(f"Generating report for {ticker}...")
            plot_stock_risk(ticker)
            plot_stock_risk_with_panic(ticker)

    plot_correlation_heatmap()

    spotlight_tickers = get_spotlight_tickers_from_config()
    print(f"Generating detailed reports for spotlight tickers: {spotlight_tickers}")    

    run_beta_drift_forecast_report(tickers=spotlight_tickers)
    run_risk_performance_report()

if __name__ == "__main__":
    main()
