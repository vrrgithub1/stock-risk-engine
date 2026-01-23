import sqlite3
import logging
from config import DATABASE_PATH

# Configure logging for an audit trail
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_medallion_schema(db_name=DATABASE_PATH):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    try:
        logging.info("Initializing Medallion Schema...")

        # --- BRONZE LAYER (Raw ODS) ---
        # Immutable landing zone for raw API ticks
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

        # --- SILVER LAYER (Cleansed / Curated) ---
        # A view that handles Data Quality (DQ) - Forward filling and logic returns
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS silver_clean_returns AS
            SELECT 
                ticker,
                date,
                adj_close,
                (adj_close / LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date) - 1) AS daily_return
            FROM bronze_price_history
        """)

        # --- GOLD LAYER (Business Intelligence / Risk) ---
        # Normalized table for point-in-time risk metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS gold_risk_metrics (
                ticker TEXT,
                calculation_date TEXT,
                metric_type TEXT, -- 'beta', 'sharpe', 'sortino', 'liquidity'
                period_years INTEGER,
                value REAL,
                PRIMARY KEY (ticker, calculation_date, metric_type, period_years)
            )
        """)

        # Reporting View: The "Wide" Dashboard
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS gold_v_risk_dashboard AS
            SELECT 
                ticker,
                calculation_date,
                MAX(CASE WHEN metric_type = 'beta' AND period_years = 2 THEN value END) AS beta_2y,
                MAX(CASE WHEN metric_type = 'beta' AND period_years = 5 THEN value END) AS beta_5y,
                MAX(CASE WHEN metric_type = 'sortino' THEN value END) AS sortino_ratio,
                MAX(CASE WHEN metric_type = 'liquidity' THEN value END) AS days_to_liquidate
            FROM gold_risk_metrics
            GROUP BY 1, 2
        """)

        conn.commit()
        logging.info("Schema created successfully (Bronze/Silver/Gold).")

    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    create_medallion_schema()

