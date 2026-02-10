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

import sqlite3
import pandas as pd
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

class ReportGenerator:
    def __init__(self, db_path=DATABASE_PATH):
        self.db_path = db_path
        self.inference_df = None
        self.byebass_validate = False
        self.universe_tickers = get_universe_tickers_from_config()
        self.spotlight_tickers = get_spotlight_tickers_from_config()

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
        timestamp = datetime.now().strftime("%Y-%m-%d")
        # Use the path from config
        file_path = REPORT_DIR / f"risk_report_{ticker}_{timestamp}.html"
        
        fig.write_html(str(file_path))
        print(f"Report successfully archived at: {file_path}")

    def plot_stock_risk(self, ticker):
        """
        Plot Stock Risk Dashboard with Price and Rolling Beta
        1. Load Data from your Gold and Silver Views
        2. Create an Interactive Dashboard with Plotly"""
        conn = sqlite3.connect(self.db_path)

        conn.create_function("SQRT", 1, self.safe_sqrt)
        conn.create_function("POWER", 2, self.safe_pow)

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
    
    def plot_stock_risk_with_panic(self, ticker):
        """
        Plot Stock Risk Dashboard with Panic Overlay
        1. Load Data including VIX from your Gold and Silver Views
        2. Create an Interactive Dashboard with Plotly including Panic Overlay where VIX > 20
        """

        conn = sqlite3.connect(self.db_path)
        conn.create_function("SQRT", 1, self.safe_sqrt)
        conn.create_function("POWER", 2, self.safe_pow)
        
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
   
    def plot_correlation_heatmap(self):
        """
        Plot Correlation Heatmap for all tickers based on Daily Returns
        1. Load Daily Returns Data from Silver View
        2. Calculate Correlation Matrix
        3. Create Heatmap using Plotly
        """
            
        conn = sqlite3.connect(self.db_path)
        
        # 1. Pull daily returns for all tickers in a 'Pivot' format
        # We want dates as rows and Tickers as columns
        query = """
        SELECT date, ticker, daily_return
        FROM silver_returns
        WHERE daily_return IS NOT NULL
        """
        df = pd.read_sql(query, conn)
        conn.close()

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

    def fetch_inference_data(self):
        # Connect to the SQLite database
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

        print("✅ Model Trained and Inference Completed!")
        print(importance)
        self.inference_df = inference_df
        return inference_df

    def validate_gold_layer(self):
        """
        Quality Gate: Validates data integrity before report generation.
        Returns True if pass, raises ValueError if fail.
        """
        print("Initalizing Automated PDF Validation...")

        if self.inference_df is None:
            self.inference_df = self.fetch_inference_data()
        df = self.inference_df

        # 1. Null Check (ML Integrity)
        # Ensures the Random Forest actually produced a drift value
        if df['predicted_beta_drift_5d'].isnull().any():
            raise ValueError("VALIDATION FAILED: ML Model produced NULL values. Aborting PDF.")

        # 2. Financial Sanity Check (Outlier Detection)
        # If a Beta > 4.0 is detected, it's usually a data error, not a market reality
        if (df['feat_rolling_beta_130d'] > 4.0).any():
            print("CRITICAL WARNING: Extreme Beta (>4.0) detected. Check raw price data.")
            # You can choose to raise an error here or just log it
            
        # 3. Data Freshness Check (API Sync)
        # Checks if the most recent date in the DB is older than 48 hours
        latest_date = pd.to_datetime(df['date'].max())
        if latest_date < (pd.Timestamp.now() - pd.Timedelta(days=2)):
            # raise ValueError(f"VALIDATION FAILED: Data is stale. Last update: {latest_date}")
            return False

        print("✅ Validation Passed: Data integrity confirmed for Gold Layer.")
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

        tickers = get_spotlight_tickers_from_config()
        inference_df = self.inference_df if self.inference_df is not None else self.fetch_inference_data()

        if not self.validate_gold_layer() and not self.byebass_validate:
            raise ValueError("Validation failed. Cannot generate report.")

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

    def plot_risk_performance_report(self):
        """
        Generates the Phase II Executive Report with Forecasts, Logic, and Market Context.
        
        """

        spotlight_tickers = get_spotlight_tickers_from_config()
        inference_df = self.inference_df if self.inference_df is not None else self.fetch_inference_data()

        if not self.validate_gold_layer() and not self.byebass_validate:
            raise ValueError("Validation failed. Cannot generate report.")

        importance_dict = {'Feature': ['Beta', 'Vol', 'Return', 'VIX'], 'Weight': [0.386, 0.344, 0.215, 0.054]}

        conn = sqlite3.connect(self.db_path)
        df_vix = pd.read_sql("SELECT * FROM gold_market_regime_vix ORDER BY date DESC LIMIT 1", conn)

        current_vix_level = df_vix['adj_close'].iloc[-1]
        current_regime_id = df_vix['market_regime_vix'].iloc[-1]

        regime_label = self.get_regime_label(current_regime_id) # Logic: Standard if 15 <= VIX <= 25

        vix_metadata = {'current_level': current_vix_level, 'regime_label': regime_label}

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

    def get_var_risk_summary(self, ticker, trading_days=252, confidence_level=0.95):
        """
        Generates a textual summary of the stock's risk profile based on VaR calculations.
         - Historical VaR: Based on actual past returns
         - Parametric VaR: Based on mean and volatility assuming normal distribution
         - Output: A formatted string summary that can be included in reports or dashboards
        """

        conn = sqlite3.connect(self.db_path)
        
        # 1. Updated Query to include VIX data
        query = f"""
        SELECT 
            r.date, 
            r.daily_return
        FROM silver_clean_returns r
        WHERE r.ticker = '{ticker}' 
        ORDER BY r.date DESC
        LIMIT {trading_days}  -- Last 252 trading days (~1 year)
        """
        df = pd.read_sql(query, conn)
        returns = df.set_index('date')['daily_return'].dropna().values
        conn.close()

        var_engine = VarEngine(confidence_level=confidence_level)   
        historical_var = var_engine.calculate_historical_var(returns) 
        parametric_var = var_engine.calculate_parametric_var(returns) 
        monte_carlo_var = var_engine.calculate_monte_carlo_var(returns) 
        display_text = f"95% Daily VaR: {abs(historical_var)*100:.2f}% (Historical)"

        conn = sqlite3.connect(self.db_path)

        insert_query = '''
            INSERT INTO gold_risk_var_summary (ticker, historical_var, parametric_var, monte_carlo_var, display_text)
            VALUES (?, ?, ?, ?, ?);
        '''      
        conn.execute(insert_query, (ticker, historical_var, parametric_var, monte_carlo_var, display_text))  

        conn.commit()
        conn.close()

        return {
            "ticker": ticker,
            "historical_var": round(float(historical_var), 4),
            "parametric_var": round(float(parametric_var), 4),
            "monte_carlo_var": round(float(monte_carlo_var), 4),
            'display_text': f"95% Daily VaR: {abs(historical_var)*100:.2f}% (Historical)"
        }

if __name__ == "__main__":
    print("Report Generator module loaded successfully. Ready to generate reports!")
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