-- ====================================================================
-- 1. SILVER LAYER: CLEANING & RETURNS
-- ====================================================================

-- View to deduplicate and standardize column names
DROP VIEW IF EXISTS silver_price_history_clean;
CREATE VIEW silver_price_history_clean AS
WITH Deduplicated AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY date, ticker ORDER BY rowid DESC) as rn
    FROM bronze_price_history
)
SELECT date, ticker, adj_close, volume
FROM Deduplicated
WHERE rn = 1;

-- View to calculate daily percentage returns
DROP VIEW IF EXISTS silver_returns;
CREATE VIEW silver_returns AS
SELECT 
    date,
    ticker,
    adj_close,
    (adj_close / LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date) - 1) AS daily_return
FROM silver_price_history_clean;

-- ====================================================================
-- 2. SILVER LAYER: VOLATILITY MATH
-- ====================================================================

-- View for 30-Day Rolling Annualized Volatility
DROP VIEW IF EXISTS silver_rolling_volatility;
CREATE VIEW silver_rolling_volatility AS
SELECT 
    date,
    ticker,
    SQRT(
        (SUM(daily_return * daily_return) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) 
        - (POWER(SUM(daily_return) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW), 2) / 30.0)) / 29.0
    ) * SQRT(252) AS annualized_volatility_30d
FROM silver_returns
WHERE daily_return IS NOT NULL;

-- ====================================================================
-- 3. GOLD LAYER: MARKET SENSITIVITY (BETA)
-- ====================================================================

-- View for Rolling Beta against the S&P 500 (^GSPC)
DROP VIEW IF EXISTS gold_rolling_beta_30d;
CREATE VIEW gold_rolling_beta_30d AS
WITH market_data AS (
    SELECT date, daily_return AS mkt_return FROM silver_returns WHERE ticker = '^GSPC'
),
joined AS (
    SELECT s.date, s.ticker, s.daily_return AS stk_return, m.mkt_return
    FROM silver_returns s
    JOIN market_data m ON s.date = m.date
    WHERE s.ticker != '^GSPC'
)
SELECT 
    date,
    ticker,
    (SUM(stk_return * mkt_return) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) - 
    (SUM(stk_return) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) * SUM(mkt_return) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) / 30.0)) / 
    (SUM(mkt_return * mkt_return) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) - 
    (POWER(SUM(mkt_return) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW), 2) / 30.0)) AS beta_30d
FROM joined;

-- Add this to your Gold Layer
DROP VIEW IF EXISTS gold_max_drawdown;
CREATE VIEW gold_max_drawdown AS
WITH Peaks AS (
    SELECT 
        date, 
        ticker, 
        adj_close,
        MAX(adj_close) OVER (PARTITION BY ticker ORDER BY date) as peak_price
    FROM silver_price_history_clean
),
Drawdowns AS (
    SELECT 
        *,
        ((adj_close - peak_price) / peak_price) * 100 as drawdown_pct
    FROM Peaks
)
SELECT 
    ticker,
    MIN(drawdown_pct) as max_drawdown_pct,
    MAX(peak_price) as cycle_high,
    MIN(adj_close) as cycle_low
FROM Drawdowns
GROUP BY ticker;
