#!/bin/bash

# 1. Create Folder Structure
mkdir -p data/bronze src config docs

# 2. Create requirements.txt
cat <<EOF > requirements.txt
yfinance>=1.0.0
pandas>=2.2.0
numpy>=2.1.0
scipy>=1.15.0
SQLAlchemy>=2.0.0
pandas-flavor>=0.6.0
matplotlib>=3.10.0
seaborn>=0.13.0
PyYAML>=6.0.0
pandas-datareader>=0.10.0
statsmodels>=0.14.0
EOF

# 3. Create the Database Setup Script
cat <<EOF > src/database.py
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

if __name__ == "__main__":
    create_medallion_schema()
EOF

# 4. Create empty __init__ for imports
touch src/__init__.py

echo "Project 'stock-risk-engine' initialized successfully!"
