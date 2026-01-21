# main.py
from src.ingestion import DataIngestor
from src.database import create_medallion_schema, run_silver_and_gold_views
from src.maintenance import archive_old_data

#from src.transformations import run_silver_and_gold_views
#from src.maintenance import archive_old_data

def main():
    print("--- Starting Stock Risk Engine ---")
    
    # 1. Ingest Raw Data (Sourcing from tickers.yml inside the module)
    ingestor = DataIngestor()
    ingestor.run_bronze_ingestion()
    
    # 2. Build Analytical Views (SQL-based transformations)
    run_silver_and_gold_views()
    
    # 3. Cleanup & Archive
    archive_old_data()

    print("--- Pipeline Complete ---")

if __name__ == "__main__":
    main()
