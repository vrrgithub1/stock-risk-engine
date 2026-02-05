"""
Main entry point for the Stock Risk Engine pipeline.
"""

# main.py
from src.ingestion import DataIngestor
from src.database import create_medallion_schema, run_silver_and_gold_views, update_risk_inference, update_silver_risk_features, update_risk_metrics
from src.maintenance import archive_old_data
from src.setup_db import create_medallion_schema

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

if __name__ == "__main__":
    main()
