"""
Main entry point for the Stock Risk Engine pipeline.
"""

# main.py
from src.ingestion import DataIngestor
from src.database import create_medallion_schema, run_silver_and_gold_views, update_risk_inference, update_silver_risk_features, update_risk_metrics
from src.maintenance import archive_old_data
from src.setup_db import create_medallion_schema
from src.app_visualizer import plot_stock_risk, plot_stock_risk_with_panic, plot_correlation_heatmap
from src.config import DATABASE_PATH, REPORT_DIR
from src.app_visualizer2 import run_beta_drift_forecast_report
from src.app_visualizer3 import run_risk_performance_report
#from src.transformations import run_silver_and_gold_views
#from src.maintenance import archive_old_data

def main():
    print("--- Starting Stock Risk Engine ---")

    # Initialize the database schema
    create_medallion_schema(initial_setup=True)
    
    # 1. Ingest Raw Data (Sourcing from tickers.yml inside the module)
    ingestor = DataIngestor()
    ingestor.run_bronze_ingestion()
    
    # 2. Build Analytical Views (SQL-based transformations)
    run_silver_and_gold_views()

    # 3. Update Silver Risk Features
    update_silver_risk_features()

    # 4. Update Risk Metrics
    update_risk_metrics()

    # 5. Update Risk Inference Table
    update_risk_inference()

    # 6. Cleanup & Archive
    archive_old_data()

    print("--- Pipeline Complete ---")

    plot_stock_risk("NVDA")
    plot_stock_risk_with_panic("NVDA")
    plot_correlation_heatmap()
    run_beta_drift_forecast_report()
    run_risk_performance_report()

if __name__ == "__main__":
    main()
