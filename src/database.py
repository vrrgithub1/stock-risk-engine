import sqlite3
import logging

def create_medallion_schema(db_path="data/stock_risk.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Bronze: Raw History
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bronze_price_history (
            ticker TEXT, date TEXT, adj_close REAL, volume INTEGER
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

def run_silver_and_gold_views(db_path="data/stock_risk.db", sql_path="sql/init_analytics_layer.sql"):
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
