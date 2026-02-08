# config.py
"""
Configuration file for the Stock Risk Engine Database.
""" 
import os
from pathlib import Path

# Define the base directory of the application
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = Path(__file__).resolve().parent.parent


# Data Paths
DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "stock_risk_vault.db"

# YAML Config Path
CONFIG_DIR = BASE_DIR / "config"
TICKERS_YAML_PATH = CONFIG_DIR / "tickers.yml"

# Report Paths (Used by your Plotly logic)
REPORT_DIR = BASE_DIR / "reports"

SQL_DIR = BASE_DIR / "sql"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(SQL_DIR, exist_ok=True)


# Market Settings
REGIME_THRESHOLD_STRESS = 20.0
REGIME_THRESHOLD_QUIET = 12.0