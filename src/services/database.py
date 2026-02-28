"""
Database initialization and schema creation for the medallion architecture.
"""
import sqlite3
import logging
import math
import pandas as pd
import numpy as np
#from turtle import pd

# Setup logging for the ingestion pipeline
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import yaml
from src.utils.config import DATABASE_PATH, TICKERS_YAML_PATH, REPORT_DIR, SQL_DIR
from src.core.var_engine import VarEngine

def safe_sqrt(x):
    if x is None or x < 0:
        return 0.0  # Or return None
    return math.sqrt(x)

def safe_pow(x, y):
    if x is None:
        return 0.0
    return pow(x, y)

def safe_log(x):
    if x is None or x <= 0:
        return 0.0
    return math.log(x)

def safe_exp(x):
    if x is None:
        return 0.0
    return math.exp(x)


DATABASE_PATH = DATABASE_PATH
ANALYTICS_LAYER_SQL_PATH = SQL_DIR / "init_analytics_layer.sql"
TICKERS_YAML_PATH = TICKERS_YAML_PATH
REPORT_DIR = REPORT_DIR


def get_universe_tickers_from_config(config_path=TICKERS_YAML_PATH):
    """
    Reads the tickers.yml file and returns a flat list of all unique symbols.
    """

    logger.info(f"Reading tickers from config at: {config_path}")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    tickers = []
    for category in config:
        if category in ["universe_tickers"]:
            tickers.extend(config[category])
    
    logger.info(f"Extracted universe tickers: {tickers}")
    return sorted(list(set(tickers)))



def get_spotlight_tickers_from_config(config_path=TICKERS_YAML_PATH):
    """
    Reads the tickers.yml file and returns a flat list of all unique symbols.
    """
    logger.info(f"Reading tickers from config at: {config_path}")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    
    tickers = []
    for category in config:
        if category in ["spotlight_tickers"]:
            tickers.extend(config[category])
    
    logger.info(f"Extracted spotlight tickers: {tickers}")
    return sorted(list(set(tickers)))


def create_medallion_schema(db_path=DATABASE_PATH, initial_setup=False):
    """
    Creates the medallion schema in the SQLite database.
    """

    logger.info(f"Creating medallion schema in database at: {db_path} (initial_setup={initial_setup})")
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
    logger.info("Bronze layer table 'bronze_price_history' created successfully.")

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
    logger.info("Gold layer table 'gold_risk_metrics' created successfully.")

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
    logger.info("Silver layer table 'silver_risk_features' created successfully.")

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
    logger.info("Gold layer table 'gold_risk_inference' created successfully.")

    logger.info("Creating gold_risk_var_summary table...")
    if initial_setup:
        cursor.execute("""
            DROP TABLE IF EXISTS gold_risk_var_summary
        """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS gold_risk_var_summary (
            ticker TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            historical_var FLOAT,
            parametric_var FLOAT,
            monte_carlo_var FLOAT,
            display_text TEXT,
            forecast_date DATE,
            PRIMARY KEY (ticker, timestamp)
        )
    """)
    logger.info("Gold layer table 'gold_risk_var_summary' created successfully.")

    ########  Phase V: Adding new columns sector and industry to gold_risk_inference and gold_risk_metrics tables ########

    logger.info("Creating ticker_ref table...")
    if initial_setup:
        cursor.execute("""
            DROP TABLE IF EXISTS ticker_ref
        """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ticker_ref (
            ticker VARCHAR(10) PRIMARY KEY,
            sector VARCHAR(100),
            industry VARCHAR(100),
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    logger.info("Successfully created ticker_ref table.")

    logger.info("Creating gold_risk_backtesting table...")
    if initial_setup:
        cursor.execute("""
            DROP TABLE IF EXISTS gold_risk_backtesting
        """)    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS gold_risk_backtesting (
            ticker VARCHAR(10),
            forecast_date DATE,
            predicted_var_95 FLOAT,
            actual_return FLOAT,
            is_violation BOOLEAN,
            PRIMARY KEY (ticker, forecast_date)
        )
    """)
    logger.info("Successfully created gold_risk_backtesting table.")
        

    conn.commit()
    conn.close()
    logger.info(f"Successfully initialized database at: {db_path}") 


def run_silver_and_gold_views(db_path=DATABASE_PATH, sql_path=ANALYTICS_LAYER_SQL_PATH):
    """
    Runs the SQL script to create silver and gold views.
    """
    logger.info(f"Running SQL script to create silver and gold views from: {sql_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    with open(sql_path, 'r') as sql_file: 
        sql_script = sql_file.read()
        cursor.executescript(sql_script)   

    conn.commit()
    conn.close()
    logger.info(f"Successfully created silver and gold views in database at: {db_path}")


def update_silver_risk_features(db_path=DATABASE_PATH):
    """
    Updates the gold risk inference table in the SQLite database.
    """
    logger.info(f"Updating silver risk features in database at: {db_path}") 

    conn = sqlite3.connect(db_path)
    conn.create_function("SQRT", 1, safe_sqrt)
    conn.create_function("POWER", 2, safe_pow)
    conn.create_function("LOG", 1, safe_log)
    conn.create_function("EXP", 1, safe_exp)    

    cursor = conn.cursor()
    
    logger.info("Calculating new risk features and deleting duplicates from silver_risk_features table.")

    # Bronze: Raw History
    cursor.execute("""
        WITH risk_features_new AS
        (
            SELECT 
                b.ticker,
                b.date,
                -- Lagged features (Inputs)
                srv.annualized_volatility_30d as feat_rolling_vol_30d,
                grbd.beta_30d as feat_rolling_beta_130d,
                cr.cumulative_return_5d as feat_cumulative_return_5d,
                v.market_regime_vix as feat_market_regime_vix,
                -- ... add other features from your Phase II doc ...
                -- Target (Label)
                b.beta_30d_drift_5d as target_beta_drift_5d
            FROM gold_beta_30d_drift_5d b
            JOIN silver_rolling_volatility srv ON srv.date = b.date AND srv.ticker = b.ticker 
            JOIN gold_rolling_beta_30d grbd ON grbd.date = b.date AND grbd.ticker = b.ticker
            JOIN gold_cum_return_5d cr ON b.ticker = cr.ticker AND b.date = cr.date
            JOIN gold_market_regime_vix v ON b.date = v.date
        )
        DELETE FROM silver_risk_features
        WHERE (ticker, date) IN
        (
            SELECT DISTINCT fmna.ticker, fmna.date
            FROM risk_features_new fmna
        )
    """)
    
    logger.info("Inserting new risk features into silver_risk_features table.")

    # Gold: Risk Metrics
    cursor.execute("""
        WITH risk_features_new AS
        (
            SELECT 
                b.ticker,
                b.date,
                -- Lagged features (Inputs)
                srv.annualized_volatility_30d as feat_rolling_vol_30d,
                grbd.beta_30d as feat_rolling_beta_130d,
                cr.cumulative_return_5d as feat_cumulative_return_5d,
                v.market_regime_vix as feat_market_regime_vix,
                -- ... add other features from your Phase II doc ...
                -- Target (Label)
                b.beta_30d_drift_5d as target_beta_drift_5d
            FROM gold_beta_30d_drift_5d b
            JOIN silver_rolling_volatility srv ON srv.date = b.date AND srv.ticker = b.ticker 
            JOIN gold_rolling_beta_30d grbd ON grbd.date = b.date AND grbd.ticker = b.ticker
            JOIN gold_cum_return_5d cr ON b.ticker = cr.ticker AND b.date = cr.date
            JOIN gold_market_regime_vix v ON b.date = v.date
        )
        INSERT INTO silver_risk_features (ticker, date, feat_rolling_vol_30d, feat_rolling_beta_130d, feat_cumulative_return_5d, feat_market_regime_vix, target_beta_drift_5d)
        SELECT mfs.ticker, mfs.date, mfs.feat_rolling_vol_30d, mfs.feat_rolling_beta_130d, mfs.feat_cumulative_return_5d, mfs.feat_market_regime_vix, mfs.target_beta_drift_5d
        FROM risk_features_new mfs
    """)
    
    conn.commit()
    conn.close()
    logger.info(f"Silver risk features updated successfully in database at: {db_path}")

def update_risk_metrics(db_path=DATABASE_PATH):
    """
    Updates the gold risk metrics table in the SQLite database.
    """
    logger.info(f"Updating gold risk metrics in database at: {db_path}")    

    conn = sqlite3.connect(db_path)
    conn.create_function("SQRT", 1, safe_sqrt)
    conn.create_function("POWER", 2, safe_pow)
    conn.create_function("LOG", 1, safe_log)
    conn.create_function("EXP", 1, safe_exp)    

    cursor = conn.cursor()
    
    # Gold: Risk Metrics
    logger.info("Calculating new risk metrics and inserting into gold_risk_metrics table.")

    cursor.execute("""
            INSERT INTO gold_risk_metrics (
                ticker, 
                date, 
                actual_beta_130d, 
                actual_vol_30d, 
                actual_return_5d,
                vix_regime
            )
            SELECT 
                ticker, 
                date, 
                feat_rolling_beta_130d, 
                feat_rolling_vol_30d,
                feat_cumulative_return_5d,
                feat_market_regime_vix
            FROM silver_risk_features
            WHERE date <= (
                -- Subquery to find the date that is exactly 5 business days ago
                SELECT date FROM (SELECT DISTINCT date FROM silver_risk_features ORDER BY date DESC LIMIT 1 OFFSET 5)
            )
            AND NOT EXISTS (
                -- Ensure we don't duplicate records
                SELECT 1 FROM gold_risk_metrics g 
                WHERE g.ticker = silver_risk_features.ticker 
                AND g.date = silver_risk_features.date
            )
    """)
    conn.commit()
    conn.close()
    logger.info(f"Gold risk metrics updated successfully in database at: {db_path}")
                   

def update_risk_inference(db_path=DATABASE_PATH):
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from src.utils.config import DATABASE_PATH
    import sqlite3

    logger.info(f"Updating risk inference in database at: {db_path}")
    # Connect to the SQLite database
    conn = sqlite3.connect(DATABASE_PATH)
    conn.create_function("SQRT", 1, safe_sqrt)
    conn.create_function("POWER", 2, safe_pow)
    conn.create_function("LOG", 1, safe_log)
    conn.create_function("EXP", 1, safe_exp)    

    # 1. Load Data with safety check
    try:
        df = pd.read_sql("SELECT * FROM silver_risk_features", conn)
    except Exception as e:
        logger.warning(f"⚠️ Skipping Inference: Table 'silver_risk_features' not ready. {e}")
        conn.close()
        return

    if df.empty:
        logger.warning("⚠️ Skipping Risk Inference: No data found in silver_risk_features.")
        conn.close()
        return
    
    # 2. Split into Training (Known Outcomes) and Inference (The 46 NULLs)
    train_df = df[df['target_beta_drift_5d'].notnull()].copy()
    inference_df = df[df['target_beta_drift_5d'].isnull()].copy()

    # 3. Define Features and Target
    features = ['feat_rolling_vol_30d', 'feat_rolling_beta_130d', 
                'feat_cumulative_return_5d', 'feat_market_regime_vix']
    X_train = train_df[features]
    y_train = train_df['target_beta_drift_5d']

    # 4. Train the Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    if X_train is None or len(X_train) == 0:
        logger.warning("⚠️ Skipping Risk Inference update: Not enough data in Silver/Gold views to train model.")
        return
    
    model.fit(X_train, y_train)

    if not inference_df.empty:
            inference_df['predicted_beta_drift_5d'] = model.predict(inference_df[features])
            
            # Prepare Audit Table Data
            audit_df = inference_df[['ticker', 'date', 'feat_rolling_beta_130d', 'predicted_beta_drift_5d']].copy()
            audit_df['predicted_beta_final'] = audit_df['feat_rolling_beta_130d'] + audit_df['predicted_beta_drift_5d']
            audit_df['model_version'] = 'RF_v1.0_202602'
            audit_df['prediction_timestamp'] = pd.Timestamp.now()
            
            audit_df.columns = ['ticker', 'forecast_date', 'base_beta_130d', 'predicted_drift', 
                                'predicted_beta_final', 'model_version', 'prediction_timestamp']

            audit_df.to_sql('gold_risk_inference', con=conn, if_exists='append', index=False)
            logger.info(f"Successfully preserved {len(audit_df)} predictions in Gold Layer.")

    conn.commit()
    #conn.close()    
    
    # 5. Update the 'actual_beta_realized' by joining with your Gold/Fact table
    # This logic assumes you have a 'fact_stock_metrics' table with the real data
    cursor = conn.cursor()
    
    logger.info("Updating 'actual_beta_realized' in gold_risk_inference table based on gold_rolling_beta_30d data.")

    cursor.execute("""
        WITH beta_5dlead AS (
            SELECT 
                grbd.ticker,
                grbd.date,
                LEAD(grbd.date, 5, NULL) OVER ( PARTITION BY grbd.ticker ORDER BY grbd.date ) AS Lead5d_date,
                grbd.beta_30d
            FROM gold_rolling_beta_30d grbd 
        )
        UPDATE gold_risk_inference AS gri
        SET actual_beta_realized = (
            SELECT g.beta_30d
            FROM beta_5dlead g
            WHERE g.ticker = gri.ticker 
            AND g.Lead5d_date = gri.forecast_date
            AND g.Lead5d_date IS NOT NULL
        )
        WHERE actual_beta_realized IS NULL;
    """)
    
    logger.info("Successfully updated 'actual_beta_realized' for predictions where data is available.")
    # 2. Calculate the 'prediction_error'
    # Error = Actual - Predicted
    logger.info("Calculating 'prediction_error' for gold_risk_inference table.")

    cursor.execute("""
        UPDATE gold_risk_inference
        SET prediction_error = actual_beta_realized - predicted_beta_final
        WHERE actual_beta_realized IS NOT NULL;
    """)
    
    conn.commit()
    conn.close()
    logger.info(f"Risk inference updated successfully in database at: {db_path}")   

if __name__ == "__main__":
    create_medallion_schema(initial_setup=False)
    logger.info("Database schema created successfully.")
