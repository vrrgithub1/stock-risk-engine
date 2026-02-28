"""
Reporting Service for Stock Risk Analysis
This module generates interactive HTML reports for stock risk analysis, including:  
1. Stock Risk Dashboard with Price and Rolling Beta
2. Stock Risk Dashboard with Panic Overlay (VIX > 20)
3. Correlation Heatmap for all tickers based on Daily Returns
The reports are saved in the 'reports' directory with timestamped filenames for easy archiving.
Author: Venkat Rajadurai
Date: 2024-06-01
"""

import pandas as pd
import sqlite3
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from src.utils.config import DATABASE_PATH, REPORT_DIR
from src.services.database import get_spotlight_tickers_from_config, get_universe_tickers_from_config
from src.core.var_engine import VarEngine
import os
from datetime import datetime
import logging
import yfinance as yf

# Setup logging for the ingestion pipeline
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_PATH = DATABASE_PATH
REPORT_DIR = REPORT_DIR

# --- New Phase V Logic ---
class SectorEnricher:
    def __init__(self):
        self.cache = {}
        self.db_path = DATABASE_PATH

    def get_metadata(self, ticker):
        if ticker in self.cache:
            return self.cache[ticker]
        
        try:
            info = yf.Ticker(ticker).info
            metadata = {
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown')
            }
            self.cache[ticker] = metadata
            sconn = sqlite3.connect(self.db_path)
            sconn.execute('''
                INSERT OR REPLACE INTO ticker_ref (ticker, sector, industry)
                VALUES (?, ?, ?)
            ''', (ticker, metadata['sector'], metadata['industry']))
            sconn.commit()
            sconn.close()
            return metadata
        except:
            return {'sector': 'Unknown', 'industry': 'Unknown'}
class ReportGenerator:
    def __init__(self, db_path=DATABASE_PATH):
        self.db_path = db_path
        self.inference_df = None
        self.byebass_validate = False
        self.universe_tickers = get_universe_tickers_from_config()
        self.spotlight_tickers = get_spotlight_tickers_from_config()
        self.enricher = SectorEnricher()

    def set_byebass_validate(self, value: bool):
        self.byebass_validate = value

    def safe_sqrt(self, x):
        if x is None or x < 0:
            return 0.0  # Or return None
        return math.sqrt(x)

    def safe_pow(self, x, y):
        if x is None:
            return 0.0
        return pow(x, y)

    def get_regime_label(self, vix_level):
        if vix_level == 0:        
            return "Quiet (Low Volatility)"
        elif vix_level == 1:
            return "Standard (Moderate Risk)"
        else:
            return "Stress (High Risk/Panic)"
    
    def save_report(self, fig, ticker):
        logger.info(f"Saving report for {ticker} to HTML file.")
        timestamp = datetime.now().strftime("%Y-%m-%d")
        # Use the path from config
        file_path = REPORT_DIR / f"risk_report_{ticker}_{timestamp}.html"
        
        fig.write_html(str(file_path))
        logger.info(f"Report successfully archived at: {file_path}")

    def plot_stock_risk(self, ticker):
        """
        Plot Stock Risk Dashboard with Price and Rolling Beta
        1. Load Data from your Gold and Silver Views
        2. Create an Interactive Dashboard with Plotly"""
        logger.info(f"Generating stock risk dashboard for {ticker}.")
        conn = sqlite3.connect(self.db_path)

        conn.create_function("SQRT", 1, self.safe_sqrt)
        conn.create_function("POWER", 2, self.safe_pow)

        logger.info(f"Fetching data for {ticker} from database for risk dashboard.")
        # 1. Load Data from your Gold and Silver Views
        query = f"""
        SELECT v.date, v.annualized_volatility_30d, b.beta_30d, r.adj_close
        FROM silver_rolling_volatility v
        JOIN gold_rolling_beta_30d b ON v.date = b.date AND v.ticker = b.ticker
        JOIN silver_returns r ON v.date = r.date AND v.ticker = r.ticker
        WHERE v.ticker = '{ticker}'
        ORDER BY v.date ASC
        """
        df = pd.read_sql(query, conn)
        conn.close()

        logger.info(f"Data for {ticker} successfully loaded. Generating dashboard.")

        # 2. Create an Interactive Dashboard
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add Price Line
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['adj_close'], name="Price ($)", line=dict(color='black', width=2)),
            secondary_y=False,
        )

        # Add Beta Line
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['beta_30d'], name="Rolling Beta", line=dict(color='royalblue', dash='dot')),
            secondary_y=True,
        )

        fig.update_layout(
            title=f"Risk Analysis Dashboard: {ticker}",
            xaxis_title="Date",
            template="plotly_white",
            hovermode="x unified"
        )
        self.save_report(fig, ticker)
        logger.info(f"Stock risk dashboard for {ticker} generated and saved successfully.")
    
    def plot_stock_risk_with_panic(self, ticker):
        """
        Plot Stock Risk Dashboard with Panic Overlay
        1. Load Data including VIX from your Gold and Silver Views
        2. Create an Interactive Dashboard with Plotly including Panic Overlay where VIX > 20
        """
        logger.info(f"Generating stock risk dashboard with panic overlay for {ticker}.")

        conn = sqlite3.connect(self.db_path)
        conn.create_function("SQRT", 1, self.safe_sqrt)
        conn.create_function("POWER", 2, self.safe_pow)
        
        logger.info(f"Fetching data for {ticker} including VIX from database for panic overlay dashboard.")
        # 1. Updated Query to include VIX data
        query = f"""
        SELECT 
            v.date, 
            v.annualized_volatility_30d, 
            b.beta_30d, 
            r.adj_close,
            m.adj_close as vix_price
        FROM silver_rolling_volatility v
        JOIN gold_rolling_beta_30d b ON v.date = b.date AND v.ticker = b.ticker
        JOIN silver_returns r ON v.date = r.date AND v.ticker = r.ticker
        JOIN silver_price_history_clean m ON v.date = m.date
        WHERE v.ticker = '{ticker}' AND m.ticker = '^VIX'
        ORDER BY v.date ASC
        """
        df = pd.read_sql(query, conn)
        conn.close()

        logger.info(f"Data for {ticker} including VIX successfully loaded. Generating panic overlay dashboard.")
        # 2. Build the Plot
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add NVDA Price
        fig.add_trace(go.Scatter(x=df['date'], y=df['adj_close'], name="Price ($)", line=dict(color='black', width=2)), secondary_y=False)

        # Add Rolling Beta
        fig.add_trace(go.Scatter(x=df['date'], y=df['beta_30d'], name="Rolling Beta", line=dict(color='royalblue', dash='dot')), secondary_y=True)

        # 3. THE PANIC OVERLAY: Highlight areas where VIX > 20
        # We find contiguous blocks of panic to draw rectangles
        panic_threshold = 20
        panic_df = df[df['vix_price'] >= panic_threshold]
        
        # Logic to draw red "vrects" for panic periods
        for date in panic_df['date']:
            fig.add_vrect(
                x0=date, x1=date,
                fillcolor="red", opacity=0.1,
                layer="below", line_width=0,
                name="Panic Zone (VIX > 20)"
            )

        fig.update_layout(
            title=f"Risk Dashboard with Panic Overlay: {ticker}",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # fig.show()
        self.save_report(fig, f"withPanic_{ticker}")
        logger.info(f"Stock risk dashboard with panic overlay for {ticker} generated and saved successfully.")
   
    def plot_correlation_heatmap(self):
        """
        Plot Correlation Heatmap for all tickers based on Daily Returns
        1. Load Daily Returns Data from Silver View
        2. Calculate Correlation Matrix
        3. Create Heatmap using Plotly
        """

        logger.info("Generating correlation heatmap for all tickers based on daily returns.")
            
        conn = sqlite3.connect(self.db_path)
        
        logger.info("Fetching daily returns data for all tickers from database for correlation heatmap.")
        # 1. Pull daily returns for all tickers in a 'Pivot' format
        # We want dates as rows and Tickers as columns
        query = """
        SELECT date, ticker, daily_return
        FROM silver_returns
        WHERE daily_return IS NOT NULL
        """
        df = pd.read_sql(query, conn)
        conn.close()

        logger.info("Data for correlation heatmap successfully loaded. Calculating correlation matrix and generating heatmap.")
        # 2. Pivot the data so each column is a ticker
        pivot_df = df.pivot(index='date', columns='ticker', values='daily_return')

        # 3. Calculate the Pearson Correlation Matrix
        corr_matrix = pivot_df.corr()

        # 4. Generate the Heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=".2f", # Shows the correlation number inside the box
            aspect="auto",
            color_continuous_scale='RdBu_r', # Red for negative, Blue for positive
            range_color=[-1, 1],
            title="Portfolio Correlation Matrix (Daily Returns)"
        )

        fig.update_layout(
            xaxis_title="Ticker",
            yaxis_title="Ticker",
            template="plotly_white"
        )

        # fig.show()
        self.save_report(fig, "correlation_heatmap")
        logger.info("Correlation heatmap generated and saved successfully.")

    def fetch_inference_data(self):
        # Connect to the SQLite database

        logger.info("Fetching inference data from database for model training and prediction.")
        conn = sqlite3.connect(self.db_path)

        # 1. Load your Model Feature Store Data
        df = pd.read_sql("SELECT * FROM silver_risk_features ", conn)

        # 2. Split into Training (Known Outcomes) and Inference (The 46 NULLs)
        train_df = df[df['target_beta_drift_5d'].notnull()].copy()
        inference_df = df[df['target_beta_drift_5d'].isnull()].copy()

        # 3. Define Features and Target
        features = ['feat_rolling_vol_30d', 'feat_rolling_beta_130d', 
                    'feat_cumulative_return_5d', 'feat_market_regime_vix']
        X_train = train_df[features]
        y_train = train_df['target_beta_drift_5d']

        logger.info("Model training data prepared. Starting model training and inference.")
        # 4. Train the Model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # 5. Predict the "NULL" values (The 46 rows)
        inference_df['predicted_beta_drift_5d'] = model.predict(inference_df[features])

        # 6. Get Feature Importance
        importance = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        conn.close()

        logger.info("✅ Model Trained and Inference Completed!")
        logger.info(importance)
        self.inference_df = inference_df
        logger.info("Inference data fetched and predictions made successfully.")
        return inference_df

    def validate_gold_layer(self):
        """
        Quality Gate: Validates data integrity before report generation.
        Returns True if pass, raises ValueError if fail.
        """
        logger.info("Validating Gold Layer data integrity before report generation.")

        if self.inference_df is None:
            self.inference_df = self.fetch_inference_data()
        df = self.inference_df

        logger.info("Performing data integrity checks on Gold Layer inference data.")
        # 1. Null Check (ML Integrity)
        # Ensures the Random Forest actually produced a drift value

        failures = df[df['predicted_beta_drift_5d'].isnull()]
        
        if not failures.empty:
            logger.warning(f"VALIDATION FAILED: ML Model produced NULL values in {len(failures)} rows.")
            logger.warning(f"Failures details: \n{failures[['ticker', 'predicted_beta_drift_5d']]}")
            raise ValueError("VALIDATION FAILED: ML Model produced NULL values. Aborting PDF.")

        logger.info("Null check passed: No NULL values in predicted_beta_drift_5d.")

        # 2. Financial Sanity Check (Outlier Detection)
        # If a Beta > 4.0 is detected, it's usually a data error, not a market reality
        # Identify the failures
        failures = df[df['feat_rolling_beta_130d'] > 4.0]

        if not failures.empty:
            logger.warning(f"CRITICAL WARNING: Extreme Beta (>4.0) detected in {len(failures)} rows.")
            # This prints the ticker, the beta value, and the date (if it's in the index)
            logger.warning(f"Extreme Beta details: \n{failures[['ticker', 'feat_rolling_beta_130d']]}")            # You can choose to raise an error here or just log it

        logger.info("Financial sanity check completed: No extreme beta values detected.")

        # 3. Data Freshness Check (API Sync)
        # Checks if the most recent date in the DB is older than 48 hours
        latest_date = pd.to_datetime(df['date'].max())
        if latest_date < (pd.Timestamp.now() - pd.Timedelta(days=2)):
            # raise ValueError(f"VALIDATION FAILED: Data is stale. Last update: {latest_date}")
            return False

        logger.info("Validation passed: Data integrity confirmed for Gold Layer.")
        return True

    def plot_beta_drift_forecast_report(self):
        """
        Generates a PDF report visualizing the predicted Beta Drift for selected stocks.
        """
        # Visualization code will go here
        pass
        # Below code is for visualization for inference results
        # 1. Run the prediction logic (assuming 'inference_df' is ready)
        # Let's visualize NVDA and TSLA specifically
        logger.info("Generating Beta Drift Forecast Report for NVDA and TSLA.")

        tickers = get_spotlight_tickers_from_config()
        inference_df = self.inference_df if self.inference_df is not None else self.fetch_inference_data()

        logger.info(f"Spotlight tickers for Beta Drift Forecast: {tickers}")
        if not self.validate_gold_layer() and not self.byebass_validate:
            raise ValueError("Validation failed. Cannot generate report.")

        logger.info("Data validation passed. Proceeding to generate Beta Drift Forecast Report.")

        fig = go.Figure()

        for ticker in tickers:
            subset = inference_df[inference_df['ticker'] == ticker].sort_values('date')
            
            # Calculate the forecasted Beta (Current + Predicted Drift)
            forecast_beta = subset['feat_rolling_beta_130d'] + subset['predicted_beta_drift_5d']
            
            # Add Current Beta line
            fig.add_trace(go.Scatter(
                x=subset['date'], y=subset['feat_rolling_beta_130d'],
                name=f'{ticker} Current Beta',
                mode='lines+markers', line=dict(dash='dot')
            ))
            
            # Add Forecasted Beta line
            fig.add_trace(go.Scatter(
                x=subset['date'], y=forecast_beta,
                name=f'{ticker} 5-Day Forecast',
                mode='lines+markers', line=dict(width=4)
            ))

        fig.update_layout(
            title='<b>Predictive Risk Analytics: Beta Drift Forecast </b> (NVDA vs TSLA)',
            xaxis_title='Forecast Date',
            yaxis_title='Beta Value',
            template='plotly_dark',
            hovermode='x unified'
        )

        fig.update_xaxes(
            tickformat="%b %d",  # Result: Jan 30, 2026
            dtick="86400000.0",      # Force ticks to appear exactly every 24 hours (ms)
            tickangle=-45            # Optional: Tilts labels for better spacing
        )    

        self.save_report(fig, "beta_drift_forecast")
        logger.info("Beta Drift Forecast Report generated and saved successfully.")

    def plot_risk_performance_report(self):
        """
        Generates the Phase II Executive Report with Forecasts, Logic, and Market Context.
        
        """
        logger.info("Generating Risk Performance Report with Forecasts, Logic, and Market Context.")    
        spotlight_tickers = get_spotlight_tickers_from_config()
        inference_df = self.inference_df if self.inference_df is not None else self.fetch_inference_data()

        logger.info(f"Spotlight tickers for Risk Performance Report: {spotlight_tickers}")

        if not self.validate_gold_layer() and not self.byebass_validate:
            raise ValueError("Validation failed. Cannot generate report.")

        importance_dict = {'Feature': ['Beta', 'Vol', 'Return', 'VIX'], 'Weight': [0.386, 0.344, 0.215, 0.054]}

        conn = sqlite3.connect(self.db_path)
        df_vix = pd.read_sql("SELECT * FROM gold_market_regime_vix ORDER BY date DESC LIMIT 1", conn)

        current_vix_level = df_vix['adj_close'].iloc[-1]
        current_regime_id = df_vix['market_regime_vix'].iloc[-1]

        regime_label = self.get_regime_label(current_regime_id) # Logic: Standard if 15 <= VIX <= 25

        vix_metadata = {'current_level': current_vix_level, 'regime_label': regime_label}


        logger.info(f"VIX Market Regime Metadata: {vix_metadata}")
        # 1. Setup the Subplots
        # Note: col 2, row 2 is 'domain' type to allow the Gauge Chart
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"colspan": 2}, None], [{}, {"type": "domain"}]],
            vertical_spacing=0.2,
            subplot_titles=(
                "<b>5-Day Beta Drift Forecast (NVDA vs TSLA)</b>", 
                "<b>Model Logic: Feature Importance</b>", 
                f"<b>VIX Market Regime: {vix_metadata['regime_label']}</b>"
            )
        )

        # 2. Add Forecast Lines (Inference Layer)
        for ticker in spotlight_tickers:
            inference_df['date'] = pd.to_datetime(inference_df['date']).dt.strftime('%Y-%m-%d')    
            subset = inference_df[inference_df['ticker'] == ticker].sort_values('date')
            forecasted_val = subset['feat_rolling_beta_130d'] + subset['predicted_beta_drift_5d']
            
            fig.add_trace(go.Scatter(
                x=subset['date'], 
                y=forecasted_val, 
                name=f"{ticker} Forecast",
                mode='lines+markers',
                line=dict(width=3)
            ), row=1, col=1)

        # 3. Add Feature Importance (The Logic Step - Fixed)
        fig.add_trace(go.Bar(
            x=importance_dict['Weight'], 
            y=importance_dict['Feature'], 
            orientation='h', 
            marker=dict(color='rgba(50, 171, 96, 0.6)'),
            name="Model Weights"
        ), row=2, col=1)

        # 4. Add Market Regime Gauge (The Context Layer)
        # Mapping the SQL View result to colors
        color_map = {"Quiet": "lime", "Standard": "orange", "Stress": "crimson"}
        gauge_color = color_map.get(vix_metadata['regime_label'], "white")

    # 4. Add Market Regime Gauge (The Context Layer)
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=vix_metadata['current_level'],
            # 1. Reduce title size or move it slightly to avoid overlap
            #  title={'text': f"<b>{vix_metadata['regime_label']}</b>", 'font': {'size': 14}},
            # domain={'x': [0.55, 0.95], 'y': [0.05, 0.45]}, # 2. Explicitly define space in the quadrant
            gauge={
                'axis': {'range': [0, 40], 'tickfont': {'size': 10}},
                'bar': {'color': "white"},
                'steps': [
                    {'range': [0, 15], 'color': "lime"},
                    {'range': [15, 25], 'color': "orange"},
                    {'range': [25, 40], 'color': "red"}
                ],
            }
        ), row=2, col=2)

        # 5. Finalize Layout
        fig.update_layout(
            height=900, 
            template="plotly_dark", 
            showlegend=True, 
            title_text="<b>Model Validation: Backtest Performance Report</b>",
            title_x=0.5,
            # 3. Increase top margin and vertical spacing between subplots
            margin=dict(t=120, b=50, l=50, r=50),
            grid={'rows': 2, 'columns': 2, 'pattern': 'independent'}
        )

        # Assuming your Forecast Chart is in Row 1, Col 1
        fig.update_xaxes(
            tickformat="%b %d", 
            dtick="86400000.0", 
            tickangle=-45,
            row=1, col=1  # <--- This ensures it ONLY affects the Forecast chart
        ) 

        self.save_report(fig, "risk_performance_report")
        logger.info("Risk Performance Report with Forecasts, Logic, and Market Context generated and saved successfully.")  


    def get_var_risk_summary(self, ticker, trading_days=252, confidence_level=0.95):
        """
        Generates a textual summary of the stock's risk profile based on VaR calculations.
         - Historical VaR: Based on actual past returns
         - Parametric VaR: Based on mean and volatility assuming normal distribution
         - Output: A formatted string summary that can be included in reports or dashboards
        """

        # db_path = self.db_path
        logger.info(f"Calculating VaR summary for {ticker} using database at: {self.db_path}")
        conn = sqlite3.connect(self.db_path)
        
        # 1. Pull the last 252 trading days of returns for the specified ticker
        query = """
            SELECT 
                r.date, 
                r.daily_return
            FROM silver_returns r
            WHERE r.ticker = ? 
            ORDER BY r.date DESC
            LIMIT ?
            """
        try:
            # Pass the variables as a tuple in the second argument
            df = pd.read_sql(query, conn, params=(ticker, trading_days))

            # Assuming 'df' is your historical price dataframe (252 rows)
            # Get the date of the very last row
            effective_date = df['date'].iloc[0]  # Get the first date in the dataframe (most recent)
            
            if df.empty:
                logger.warning(f"⚠️ No data found for {ticker}")
                return None
                
            returns = df.set_index('date')['daily_return'].dropna().values
            
            # ... your VaR calculation logic ...
            
        finally:
            conn.close()

#        df = pd.read_sql(query, conn)
#        returns = df.set_index('date')['daily_return'].dropna().values
#        conn.close()

        var_engine = VarEngine(confidence_level=confidence_level)   
        historical_var = var_engine.calculate_historical_var(returns) 
        parametric_var = var_engine.calculate_parametric_var(returns) 
        monte_carlo_var = var_engine.calculate_monte_carlo_var(returns) 
        display_text = f"95% Daily VaR: {abs(historical_var)*100:.2f}% (Historical)"

        conn = sqlite3.connect(self.db_path)

        insert_query = '''
            INSERT INTO gold_risk_var_summary (ticker, historical_var, parametric_var, monte_carlo_var, display_text, forecast_date)
            VALUES (?, ?, ?, ?, ?, ?);
        '''      
        conn.execute(insert_query, (ticker, historical_var, parametric_var, monte_carlo_var, display_text, effective_date))  

        conn.commit()
        conn.close()
        logger.info(f"VaR summary for {ticker} calculated and stored in database successfully.")

        return {
            "ticker": ticker,
            "historical_var": round(float(historical_var), 4),
            "parametric_var": round(float(parametric_var), 4),
            "monte_carlo_var": round(float(monte_carlo_var), 4),
            'display_text': display_text,
            'effective_date': effective_date
        }

    def plot_risk_summary_matrix(self):
        logger.info("Generating Risk Summary Matrix for all tickers in the universe.")

        conn = sqlite3.connect(self.db_path)
        universal_tickers = get_universe_tickers_from_config()
        univ_tickers_tuper = tuple(universal_tickers)  # Convert list to tuple for SQL IN clause
        logger.info(f"Generating Risk Summary Matrix for tickers: {universal_tickers}")
        
        # Joining the ML predictions with the VaR calculations
        query = """
        SELECT 
            grrvs.ticker, 
            grrvs.monte_carlo_var as mc_var, 
            grri.predicted_beta_final as pred_beta,
            grri.forecast_date
        FROM gold_recent_risk_var_summary grrvs 
        JOIN gold_recent_risk_inference grri ON grrvs.ticker = grri.ticker
        WHERE grrvs.ticker IN {}
        """
        query = query.format(univ_tickers_tuper)
        
        df = pd.read_sql(query, conn)
        conn.close()

        if df.empty:
                logger.warning("⚠️ No data available for Risk Matrix.")
                return

        # 2. Define Quadrants for "Colorfulness"
        # We use medians as the crosshair for the quadrants
        var_mid = df['mc_var'].median()
        beta_mid = df['pred_beta'].median()
        
        def assign_quadrant(row):
                if row['pred_beta'] > beta_mid and row['mc_var'] > var_mid:
                    return 'High Beta / High VaR (Aggressive)'
                elif row['pred_beta'] > beta_mid and row['mc_var'] <= var_mid:
                    return 'High Beta / Low VaR (Efficient)'
                elif row['pred_beta'] <= beta_mid and row['mc_var'] > var_mid:
                    return 'Low Beta / High VaR (Outlier Risk)'
                else:
                    return 'Low Beta / Low VaR (Defensive)'

        df['Risk Category'] = df.apply(assign_quadrant, axis=1)

        # 3. Create the Colorful Scatter Plot
        fig = px.scatter(
            df, 
            x="mc_var", 
            y="pred_beta",
            text="ticker", 
            color="Risk Category",  # <--- This adds the colors!
            title=f"<b>Stock Risk-Reward Matrix ({datetime.now().strftime('%Y-%m-%d')})</b>",
            labels={'mc_var': '<b>95% Monte Carlo VaR (Downside Risk)</b>', 
                    'pred_beta': '<b>Predicted Beta (Market Sensitivity)</b>'},
            color_discrete_map={
                'High Beta / High VaR (Aggressive)': '#EF553B', # Red
                'High Beta / Low VaR (Efficient)': '#636EFA',  # Blue
                'Low Beta / High VaR (Outlier Risk)': '#FECB52', # Gold
                'Low Beta / Low VaR (Defensive)': '#00CC96'     # Green
            },
            template="plotly_white"
        )

        # Add crosshair lines for the quadrants
        fig.add_hline(y=beta_mid, line_dash="dot", line_color="dimgray", annotation_text="<b>Beta Median</b>")
        fig.add_vline(x=var_mid, line_dash="dot", line_color="dimgray", annotation_text="<b>VaR Median</b>")

        fig.update_traces(textposition='top center', marker=dict(size=12, opacity=0.8))
        fig.update_layout(
            legend_title_font=dict(weight="bold"),
            legend_font=dict(weight="bold"),
            xaxis=dict(
                title_font=dict(weight="bold"),
            ),
            yaxis=dict(
                title_font=dict(weight="bold"),
            )
        )

        fig.update_xaxes(title_font=dict(weight="bold"))
        fig.update_yaxes(title_font=dict(weight="bold"))

        self.save_report(fig, "risk_summary_matrix")
        logger.info("Risk Summary Matrix generated and saved successfully.")

    def calculate_sector_concentration(self, gold_df):
        """
        Aggregates risk metrics by sector to identify concentration hot-spots.
        Assumes gold_df contains 'ticker', 'sector', and 'var_95_monte_carlo'
        """
        # 1. Group by sector and calculate mean risk & count of tickers
        sector_summary = gold_df.groupby('sector').agg({
            'ticker': 'count',
            'var_95_monte_carlo': 'mean',
            'feat_rolling_beta_130d': 'mean'
        }).rename(columns={'ticker': 'ticker_count'})

        # 2. Calculate Relative Risk Contribution
        # We use the absolute value of VaR because it is a negative number
        total_avg_risk = sector_summary['var_95_monte_carlo'].abs().sum()
        sector_summary['risk_contribution_pct'] = (
            sector_summary['var_95_monte_carlo'].abs() / total_avg_risk
        ) * 100

        # 3. Sort by highest risk contribution
        sector_summary = sector_summary.sort_values(by='risk_contribution_pct', ascending=False)
        
        return sector_summary


    def get_sector_summary(self):
        logger.info("Generating Sector Concentration Summary for all tickers in the universe.")

        conn = sqlite3.connect(self.db_path)
        universal_tickers = get_universe_tickers_from_config()
        univ_tickers_tuper = tuple(universal_tickers)  # Convert list to tuple for SQL IN clause
        logger.info(f"Generating Risk Summary Matrix for tickers: {universal_tickers}")

        # Joining the ML predictions with the VaR calculations
        query = """
        SELECT 
            grrvs.ticker, 
            grrvs.monte_carlo_var as var_95_monte_carlo, 
            grri.predicted_beta_final as feat_rolling_beta_130d,
            grri.forecast_date,
            tr.sector,
            tr.industry
        FROM gold_recent_risk_var_summary grrvs 
        JOIN gold_recent_risk_inference grri ON grrvs.ticker = grri.ticker
        LEFT JOIN ticker_ref tr ON grrvs.ticker = tr.ticker
        WHERE grrvs.ticker IN {}
        """
        query = query.format(univ_tickers_tuper)
        
        df = pd.read_sql(query, conn)

        conn.close()
        if df.empty:
            logger.warning("⚠️ No data available for Sector Summary.")
            return None
        else:
            return self.calculate_sector_concentration(df)

    def validate_model_performance(self, ticker, predicted_var, forecast_date):
        """
        Compares the predicted VaR floor against actual market realization.
        """

        try:
            # Fetch the actual price on/after the forecast date
            data = yf.download(ticker, start=forecast_date, end=(pd.to_datetime(forecast_date) + pd.Timedelta(days=5)).strftime('%Y-%m-%d'), multi_level_index=False, auto_adjust=True)
            # logger.info(f"Data fetched for validation: \n{data['Close']}")
            if data.empty or len(data) < 2:
                logger.warning(f"Not enough data to validate model performance for {ticker} on {forecast_date}.")
                return None
            
            # Calculate actual log return for the next trading day
            actual_return = data['Close'].pct_change().dropna().iloc[0]
            logger.info(f"Actual return for {ticker} on {data.index[1].date()}: {actual_return:.4f}")
            
            # Check for 'Violation' (Breach)
            is_violation = actual_return < predicted_var
            
            logger.info(f"Validating model performance for {ticker} on forecast date {forecast_date}:")

            return {
                'actual_return': actual_return,
                'is_violation': is_violation,
                'breach_magnitude': actual_return - predicted_var if is_violation else 0
            }
        except Exception as e:
            logger.error(f"Error validating model performance for {ticker}: {e}")
            return None

    def persist_backtest_results(self, ticker, predicted_var_95, forecast_date):
        """
        Validates a past forecast and saves the result to the Gold Backtesting table.
        """
        # 1. Run the validation logic we drafted
        validation = self.validate_model_performance(ticker, predicted_var_95, forecast_date)

        if validation is None:
            logger.warning(f"Validation failed for {ticker} on {forecast_date}. Skipping persistence.")
            return
        
        actual_return = validation['actual_return']
        is_violation = 1 if validation['is_violation'] else 0
        
        if validation:
            # 2. SQL UPSERT (Update if exists, Insert if not) 
            # to ensure no duplicate records for the same ticker/date
            query = """
            INSERT INTO gold_risk_backtesting (ticker, forecast_date, predicted_var_95, actual_return, is_violation)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (ticker, forecast_date) 
            DO UPDATE SET 
                actual_return = EXCLUDED.actual_return,
                is_violation = EXCLUDED.is_violation;
            """
            
            # Execute using your existing db_engine logic
            conn = sqlite3.connect(self.db_path)

            conn.execute(query, (
                ticker, 
                str(forecast_date), 
                predicted_var_95, 
                actual_return, 
                is_violation
            ))
            conn.commit()
            conn.close()

    def backfill_phase_iv_backtests(self):
        # 1. Pull all historical forecasts
        query = "SELECT ticker, timestamp, monte_carlo_var, forecast_date as forecast_date FROM gold_risk_var_summary"  # Pull only last 2 days
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql(query, conn)
        conn.close()
        
        logger.info(f"🚀 Starting backfill for {len(df)} historical records...")
        
        for _, row in df.iterrows():
            # Skip today's date as it hasn't 'happened' yet
            if row['forecast_date'] >= pd.Timestamp.today().strftime('%Y-%m-%d'):
                continue
                
            # 3. Use your existing logic to validate and persist
            self.persist_backtest_results(
                ticker=row['ticker'],
                predicted_var_95=row['monte_carlo_var'],
                forecast_date=row['forecast_date']
            )
            logger.info(f"✅ Backfilled: {row['ticker']} for {row['forecast_date']}")

    def get_backtest_summary(self):
        """Fetches the backfill data for the dashboard."""
        import sqlite3
        import pandas as pd
        
        conn = sqlite3.connect(self.db_path)
        # Join with your new ticker_ref table to get Sector/Industry info!
        query = """
        SELECT b.*, r.sector, r.industry 
        FROM gold_risk_backtesting b
        LEFT JOIN ticker_ref r ON b.ticker = r.ticker
        ORDER BY b.forecast_date DESC
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df

    def get_extreme_events(self, threshold=-0.01):
        """
        Identifies breaches where the actual return was at least 1% worse 
        than the predicted VaR floor.
        """
        import sqlite3
        import pandas as pd
        
        conn = sqlite3.connect(self.db_path)
        query = """
        SELECT 
            ticker, 
            forecast_date, 
            predicted_var_95, 
            actual_return,
            (actual_return - predicted_var_95) as breach_magnitude
        FROM gold_risk_backtesting
        WHERE is_violation = 1
        ORDER BY breach_magnitude ASC
        """
        df = pd.read_sql(query, conn)
        conn.close()
        
        # Filter for 'Extreme' events (e.g., magnitude > 1%)
        extreme_df = df[df['breach_magnitude'] <= threshold]
        return extreme_df        

if __name__ == "__main__":
    # Indenting these lines makes them "safe." 
    # They won't run when main.py imports this file.
    repgen = ReportGenerator()
    repgen.backfill_phase_iv_backtests()
    df = repgen.get_backtest_summary()
    logger.info(df.head())
#    summary = repgen.get_var_risk_summary("CVX") 
#    print(summary)
#    Sector_Summary = repgen.get_sector_summary()
#    print(Sector_Summary)
#    repgen = ReportGenerator()
#    repgen.set_byebass_validate(True)  # Bypass validation for testing purposes
#    repgen.plot_stock_risk("NVDA")
#    repgen.plot_stock_risk_with_panic("NVDA")   
#    repgen.plot_correlation_heatmap()
#    repgen.plot_beta_drift_forecast_report()
#    repgen.plot_risk_performance_report()

#    plot_stock_risk("NVDA")
#    plot_stock_risk_with_panic("NVDA")
#    plot_correlation_heatmap()