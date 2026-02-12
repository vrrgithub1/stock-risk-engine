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

-- ====================================================================
-- 2. SILVER LAYER: DAILY PERCENT OF RETURNS
-- ====================================================================

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
-- 3. SILVER LAYER: 30-DAY ROLLING ANNUALIZED VOLATILITY
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
-- 4. GOLD LAYER: MARKET SENSITIVITY (BETA)
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


-- ====================================================================
-- 5. GOLD LAYER: MAX DRAWDOWN VIEW
-- ====================================================================

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


-- ====================================================================
-- Phase 2: Additional Gold Layer Views
-- ====================================================================


-- ====================================================================
-- 6. Gold Layer - Beta Drift over 5 Days
-- ====================================================================


DROP VIEW IF EXISTS gold_beta_30d_drift_5d;

CREATE VIEW gold_beta_30d_drift_5d
AS
SELECT 
	grbd.date 
,	grbd.ticker 
,	grbd.beta_30d 
,	LEAD(grbd.beta_30d, 5) OVER (
		PARTITION BY grbd.ticker 
		ORDER BY grbd.date 
	) AS beta_30d_5d_ahead
,	LEAD(grbd.beta_30d, 5) OVER (
		PARTITION BY grbd.ticker 
		ORDER BY grbd.date 
	) - grbd.beta_30d AS beta_30d_drift_5d
FROM gold_rolling_beta_30d grbd 
WHERE grbd.beta_30d IS NOT NULL 
;


-- ====================================================================
-- 7. Gold Layer - Cumulative Return over 5 Days
-- ====================================================================

DROP VIEW IF EXISTS gold_cum_return_5d;


CREATE VIEW gold_cum_return_5d
AS
SELECT 
	sr.date 
,	sr.ticker 
,	sr.daily_return 
,	EXP(
		SUM( LOG(1+sr.daily_return) ) OVER (
			PARTITION BY sr.ticker 
			ORDER BY sr.date 
			ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
		)
	) - 1 AS cumulative_return_5d
FROM silver_returns sr 
WHERE sr.daily_return IS NOT NULL
;


-- ====================================================================
-- 8. Gold Layer - Market Regime based on VIX Levels
-- ====================================================================

DROP VIEW IF EXISTS gold_market_regime_vix;

CREATE VIEW gold_market_regime_vix
AS
SELECT 
	sr.date 
,	sr.adj_close 
,	CASE 
		WHEN sr.adj_close < 15 THEN  0
		WHEN sr.adj_close >= 15 AND sr.adj_close <= 25 THEN 1
		WHEN sr.adj_close > 25 THEN 2
		ELSE 3
	END AS market_regime_vix
FROM silver_returns sr 
WHERE sr.adj_close IS NOT NULL
AND sr.ticker = '^VIX'
;


-- ====================================================================
-- 9. Gold Layer - Rolling 30-Day Average Volume
-- ====================================================================

DROP VIEW IF EXISTS gold_rolling_vol_30d;

CREATE VIEW gold_rolling_vol_30d
AS
SELECT 
	ssp.date 
,	ssp.ticker 
,	ssp.volume 
,	AVG(ssp.volume) OVER (
		PARTITION BY ssp.ticker 
		ORDER BY ssp.date 
		ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
	) AS rolling_volume_30d
FROM silver_stock_prices ssp 
WHERE ssp.volume IS NOT NULL 
;

DROP VIEW IF EXISTS gold_recent_risk_inference;

CREATE VIEW gold_recent_risk_inference
AS
SELECT 
   gri3.ticker 
  ,gri3.forecast_date 
  ,gri3.prediction_id 
  ,gri3.prediction_timestamp 
  ,gri3.base_beta_130d 
  ,gri3.predicted_drift 
  ,gri3.predicted_beta_final 
  ,gri3.model_version 
  ,gri3.actual_beta_realized 
  ,gri3.prediction_error 
FROM 
(
SELECT gri.*, ROW_NUMBER() OVER (PARTITION BY gri.ticker ORDER BY gri.prediction_id DESC) as RN 
FROM gold_risk_inference gri 
WHERE gri.forecast_date = (SELECT MAX(gri2.forecast_date) FROM gold_risk_inference gri2 )
) gri3 
WHERE gri3.RN = 1
;

DROP VIEW IF EXISTS gold_recent_risk_var_summary;

CREATE VIEW gold_recent_risk_var_summary 
AS
SELECT 
	grvs3.ticker,
	grvs3."timestamp",
	grvs3.historical_Var,
	grvs3.parametric_var,
	grvs3.monte_carlo_var,
	grvs3.display_text
FROM
(
SELECT grvs.*, ROW_NUMBER() OVER (PARTITION BY grvs.ticker ORDER BY grvs."timestamp" DESC ) AS RN
FROM gold_risk_var_summary grvs 
WHERE STRFTIME('%Y%m%d', grvs."timestamp") = (SELECT MAX(STRFTIME('%Y%m%d', grvs2."timestamp")) FROM gold_risk_var_summary grvs2 )
) grvs3 
WHERE grvs3.RN = 1
;


-- ====================================================================
-- END OF FILE
-- ====================================================================

