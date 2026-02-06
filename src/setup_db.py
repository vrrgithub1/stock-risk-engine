"""
Database setup module for creating the Medallion Schema (Bronze, Silver, Gold layers).
"""

import sqlite3
import logging
from src.config import DATABASE_PATH

# Configure logging for an audit trail
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_medallion_schema(db_path=DATABASE_PATH, initial_setup=False):
    """
    Creates the medallion schema in the SQLite database.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Bronze: Raw History

    if initial_setup:
        cursor.execute("""
                DROP TABLE IF EXISTS bronze_price_history
            """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bronze_price_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL, 
            adj_close REAL, volume INTEGER,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Gold: Risk Metrics

    cursor.execute("""
        DROP TABLE IF EXISTS gold_risk_metrics
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS gold_risk_metrics (
            ticker TEXT,
            date DATE,
            actual_beta_130d REAL,
            actual_vol_30d REAL,
            actual_return_5d REAL,
            vix_regime INTEGER,
            PRIMARY KEY (ticker, date)
        )
    """)
    
    # Gold: Risk Inference Table

    if initial_setup:
        cursor.execute("""
            DROP TABLE IF EXISTS model_feature_store
        """)
        cursor.execute("""
            DROP TABLE IF EXISTS silver_risk_features
        """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS silver_risk_features (
            ticker TEXT,
            date DATE,
            -- Lagged Features (What the model sees)
            feat_rolling_vol_30d FLOAT, 
            feat_rolling_beta_130d FLOAT,
            feat_cumulative_return_5d FLOAT,
            feat_market_regime_vix FLOAT,
            -- The Target Variable (What the model tries to guess)
            target_beta_drift_5d FLOAT, 
            PRIMARY KEY (ticker, date)
        )
    """)

    if initial_setup:
        cursor.execute("""
            DROP TABLE IF EXISTS model_predictions_audit
        """)
        cursor.execute("""
            DROP TABLE IF EXISTS gold_risk_inference
        """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS gold_risk_inference (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ticker TEXT,
            forecast_date DATE,
            base_beta_130d FLOAT,
            predicted_drift FLOAT,
            predicted_beta_final FLOAT,
            model_version TEXT,
            actual_beta_realized FLOAT NULL,
            prediction_error FLOAT NULL
        )
    """)

    conn.commit()
    conn.close()
    print("Database initialized at " + str(db_path))

if __name__ == "__main__":
    create_medallion_schema( initial_setup=True)

