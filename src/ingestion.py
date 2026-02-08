"""
Ingestion module for fetching and storing stock market data.
"""

import yfinance as yf
import pandas as pd
import sqlite3
import logging
from datetime import datetime
import yaml
import os
from src.config import DATABASE_PATH, TICKERS_YAML_PATH, REPORT_DIR, SQL_DIR

TICKERS_YAML_PATH = TICKERS_YAML_PATH
DATABASE_PATH = DATABASE_PATH

# Setup logging for the ingestion pipeline
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestor:
    """
    Class to handle data ingestion from Yahoo Finance to SQLite Bronze layer.
    """
    tickers_yaml_path = TICKERS_YAML_PATH
    tickers = []
    
    def __init__(self, db_conn=None):
        if db_conn is None:
            self.db_conn = sqlite3.connect(DATABASE_PATH)
        else:
            self.db_conn = db_conn

    def fetch_stock_data(self, tickers, start_date, end_date):
        """
        Pulls raw OHLCV data from Yahoo Finance.
        """
        logger.info(f"Fetching market data for: {tickers}")
        try:
            # group_by='column' ensures we get a clean multi-index or flat DF
            df = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)
            return df
        except Exception as e:
            logger.error(f"Error fetching yfinance data: {e}")
            return None

    def fetch_macro_indicator(self, tickers, start_date, end_date):
        """
        Pulls Macro data (e.g., ^TNX for 10Y Yield).
        """
        logger.info(f"Fetching macro indicator: {tickers}")
        return yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)

    def save_to_bronze(self, df, table_name="bronze_price_history"):
        """
        Saves the raw dataframe to the SQLite Bronze layer.
        """
        df_flat = df.stack(level=0).reset_index()
        df_flat.rename(columns={'level_1': 'ticker'}, inplace=True)
        df_flat['Date'] = df_flat['Date'].dt.strftime('%Y-%m-%d')
        df_flat.columns = [c.lower().replace(' ', '_') for c in df_flat.columns]
        df_flat['adj_close'] = df_flat['close']

        print(df_flat.head())

        with self.db_conn as conn:
            df_flat.to_sql(
                'bronze_price_history', 
                conn, 
                if_exists='append', 
                index=False,
                chunksize=500
            )

#        df.to_sql(table_name, self.db_conn, if_exists='append', index=True)
        logger.info(f"Successfully saved data to {table_name}")

    def cleanup_duplicates(self, table_name="bronze_price_history"):
        """
        Removes duplicate entries from the Bronze layer.
        """
        logger.info("Cleaning up duplicates in Bronze layer.")
        query = f"""
        DELETE FROM {table_name}
        WHERE rowid NOT IN (
            SELECT rowid 
            FROM (
                SELECT rowid, 
                    ROW_NUMBER() OVER (PARTITION BY date, ticker ORDER BY rowid DESC) as rn
                FROM {table_name}
            )
            WHERE rn = 1
        )
        """
        with self.db_conn as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            conn.commit()
        logger.info("Duplicate cleanup complete.")

    def get_tickers_from_config(config_path=str("config/tickers.yml")):
        """
        Reads the tickers.yml file and returns a flat list of all unique symbols.
        """
        if not os.path.exists(config_path):
            # Fallback if the file is missing
            print(f"Warning: {config_path} not found. using default list.")
            return ["NVDA", "TSLA", "^GSPC"]

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        
        # Extract all symbols from the different categories
        all_tickers = []
        for category in config:
            if category not in ["universe_tickers", "spotlight_tickers"]:  # Only pull from market indicators, not universe/spotlight lists
                all_tickers.extend(config[category])
        
        # Return unique values only (in case you listed a ticker twice)
        return list(set(all_tickers))
    
    def run_bronze_ingestion(self):
        """
        Main function to run the Bronze ingestion pipeline.
        """
        tickers = DataIngestor.get_tickers_from_config()
        self.tickers = tickers  # Store for later use in main.py
        start_date = "2024-01-01"
        end_date = datetime.today().strftime('%Y-%m-%d')

        # Fetch data
        df = self.fetch_stock_data(tickers, start_date, end_date)
        if df is not None:
            # Save to Bronze
            self.save_to_bronze(df)
            # Cleanup duplicates
            self.cleanup_duplicates()
        else:
            logger.error("No data fetched; skipping save and cleanup.")

# Example usage for tomorrow:
# ingestor = DataIngestor(your_sqlite_conn)
# data = ingestor.fetch_stock_data(['NVDA', 'XOM'], '2024-01-01', '2026-01-19')
