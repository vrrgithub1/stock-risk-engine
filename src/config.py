# config.py
"""
Configuration file for the Stock Risk Engine Database.
""" 
import os

# Define the base directory of the application
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = ""

# Define the full path for the SQLite database file
# 'stock_risk_vault.db' will be created in the same directory as config.py
# DATABASE_PATH = os.path.join(BASE_DIR, 'stock_risk_vault.db')

# Another option: store in a data subdirectory
DATA_DIR = os.path.join(BASE_DIR, 'data')
DATABASE_PATH = os.path.join(DATA_DIR, 'stock_risk_vault.db')
