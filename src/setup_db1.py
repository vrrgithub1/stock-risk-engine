import sqlite3
import logging

# Configure logging for an audit trail
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def update_medallion_schema(db_name="data/stock_risk_vault.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    try:
        logging.info("Initializing Medallion Schema...")

        # --- SILVER LAYER (Cleansed / Curated) ---
        # A view that fetches only stock prices (excludes macro indicators)
        cursor.execute("""
            CREATE VIEW silver_stock_prices AS
            SELECT date, ticker, adj_close, volume
            FROM bronze_price_history
            WHERE ticker NOT LIKE '^%'
        """)

        # --- SILVER LAYER (Cleansed / Curated) ---
        # A view that fetches only macro prices (excludes stock prices)
        cursor.execute("""
            CREATE VIEW silver_macro_indicators AS
            SELECT date, ticker AS indicator_name, adj_close AS value
            FROM bronze_price_history
            WHERE ticker LIKE '^%'
        """)

        # --- SILVER LAYER (Cleansed / Curated) ---
        # A view that fetches all price history with deduplication
        cursor.execute("""
            CREATE VIEW silver_price_history_clean AS
            WITH Deduplicated AS (
                SELECT 
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY date, ticker 
                        ORDER BY rowid DESC  -- Keeps the most recent entry if duplicates exist
                    ) as row_num
                FROM bronze_price_history
            )
            SELECT 
                date, 
                ticker, 
                adj_close, 
                volume
            FROM Deduplicated
            WHERE row_num = 1
        """)


        # --- SILVER LAYER (Cleansed / Curated) ---
        # A view that fetches all price history with deduplication
        cursor.execute("""
            CREATE VIEW silver_returns AS
            SELECT 
                date,
                ticker,
                adj_close,
                -- Calculate (Price_Today / Price_Yesterday) - 1
                (adj_close / LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date) - 1) AS daily_return
            FROM bronze_price_history
        """)

        conn.commit()
        logging.info("Schema created successfully (Bronze/Silver/Gold).")

    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    update_medallion_schema()
