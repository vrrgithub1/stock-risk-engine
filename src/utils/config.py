# config.py
"""
Configuration file for the Stock Risk Engine Database.
""" 
import os
import yaml
from pathlib import Path

# Define the base directory of the application
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = Path(__file__).resolve().parent.parent.parent


# Data Paths
DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "stock_risk_vault.db"

# YAML Config Path
CONFIG_DIR = BASE_DIR / "config"
TICKERS_YAML_PATH = CONFIG_DIR / "tickers.yml"
STRESS_CONFIG_PATH = CONFIG_DIR / "stress_config.yml"

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

def get_tickers_from_yaml():
    """
    Load tickers from the tickers.yml configuration file.
    """
    with open(TICKERS_YAML_PATH, 'r') as file:
        config = yaml.safe_load(file)
    return config.get('equities', [])

def get_benchmark_tickers_from_yaml():
    """
    Load benchmark tickers from the tickers.yml configuration file.
    """
    with open(TICKERS_YAML_PATH, 'r') as file:
        config = yaml.safe_load(file)
    return config.get('benchmarks', [])

def get_index_tickers_from_yaml():
    """
    Load index tickers from the tickers.yml configuration file.
    """
    with open(TICKERS_YAML_PATH, 'r') as file:
        config = yaml.safe_load(file)
    return config.get('indicators', [])
