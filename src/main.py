"""
Main entry point for the Stock Risk Engine pipeline.
"""

# main.py
from src.services.ingestion import DataIngestor
from src.services.database import (
    create_medallion_schema, 
    run_silver_and_gold_views, 
    update_risk_inference, 
    update_silver_risk_features, 
    update_risk_metrics, 
    get_universe_tickers_from_config, 
    get_spotlight_tickers_from_config
)
from src.services.maintenance import archive_old_data
from src.utils.config import DATABASE_PATH, REPORT_DIR
from src.services.reporting import ReportGenerator, SectorEnricher

import sys
import os
from pathlib import Path
import yaml
import argparse
import pandas as pd
import logging

# Setup logging for the ingestion pipeline
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# This ensures the 'src' directory is in the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():

    parser = argparse.ArgumentParser(description="Run the Stock Risk Engine pipeline.")
    parser.add_argument("--dockermode", action="store_true", help="Run in Docker mode.")
    args = parser.parse_args()

    logger.info(f"Arguments received: {args.dockermode}")
    docker_mode = args.dockermode

    logger.info("--- Starting Stock Risk Engine ---")

    logger.info(f"Running in {'Docker' if docker_mode else 'Local'} mode.")

    # 1. Initialize the database schema
    if docker_mode:
        logger.info("Initializing database schema in Docker mode...")
        create_medallion_schema(initial_setup=True)
    else:
        logger.info("Initializing database schema in Local mode...")
        create_medallion_schema(initial_setup=False)

        tickers = get_universe_tickers_from_config()  # Access the tickers list stored during ingestion
        logger.info(f"Updating Sector and Industry metadata for tickers: {tickers}")

    # Initialize the SectorEnricher and update metadata for each ticker
    enricher = SectorEnricher()
    for ticker in tickers:
        if ticker.startswith("^"):  # Skip indices for individual stock reports
            continue
        else:
            enricher.get_metadata(ticker)
    
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

    logger.info("--- Pipeline Complete ---")

    # 8. Generate Visual Reports
    repgen = ReportGenerator()
    repgen.set_byebass_validate(True)  # Bypass validation for testing purposes

    tickers = get_universe_tickers_from_config()  # Access the tickers list stored during ingestion
    logger.info(f"Generating reports for tickers: {tickers}")

    for ticker in tickers:
        if ticker.startswith("^"):  # Skip indices for individual stock reports
            continue
        else:
            logger.info(f"Generating report for {ticker}...")
            repgen.plot_stock_risk(ticker)
            repgen.plot_stock_risk_with_panic(ticker)
            var_summary = repgen.get_var_risk_summary(ticker)
            logger.info(f"VaR Summary for {ticker}: {var_summary}")

    repgen.plot_correlation_heatmap()
    repgen.plot_beta_drift_forecast_report()
    repgen.plot_risk_performance_report()
    repgen.plot_risk_summary_matrix()
    sector_summary = repgen.get_sector_summary()
    logger.info(f"Sector Summary: {sector_summary}")

    repgen.backfill_phase_iv_backtests()

    logger.info("All reports generated and saved to disk.")

if __name__ == "__main__":
    main()
