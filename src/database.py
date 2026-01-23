import sqlite3
import logging
from config import DATABASE_PATH

DATABASE_PATH = DATABASE_PATH
ANALYTICS_LAYER_SQL_PATH = "sql/init_analytics_layer.sql"

def create_medallion_schema(db_path=DATABASE_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Bronze: Raw History
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
        CREATE TABLE IF NOT EXISTS gold_risk_metrics (
            ticker TEXT, calculation_date TEXT, metric_type TEXT, 
            period_years INTEGER, value REAL,
            PRIMARY KEY (ticker, calculation_date, metric_type, period_years)
        )
    """)
    
    conn.commit()
    conn.close()
    print("Database initialized at " + db_path)

def run_silver_and_gold_views(db_path=DATABASE_PATH, sql_path=ANALYTICS_LAYER_SQL_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    with open(sql_path, 'r') as sql_file: 
        sql_script = sql_file.read()
        cursor.executescript(sql_script)   

    conn.commit()
    conn.close()
    print("Analytical views created in database at " + db_path)

if __name__ == "__main__":
    create_medallion_schema()
