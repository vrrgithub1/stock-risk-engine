"""
App Visualizer Module: Functions to create interactive dashboards and visualizations for stock risk analysis.
"""


import sqlite3
import pandas as pd
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from src.config import DATABASE_PATH, REPORT_DIR
import os
from datetime import datetime


def safe_sqrt(x):
    if x is None or x < 0:
        return 0.0  # Or return None
    return math.sqrt(x)

def safe_pow(x, y):
    if x is None:
        return 0.0
    return pow(x, y)

def save_report(fig, ticker):
    timestamp = datetime.now().strftime("%Y-%m-%d")
    # Use the path from config
    file_path = REPORT_DIR / f"risk_report_{ticker}_{timestamp}.html"
    
    fig.write_html(str(file_path))
    print(f"Report successfully archived at: {file_path}")

def plot_stock_risk(ticker, db_path=DATABASE_PATH):
    """
    Plot Stock Risk Dashboard with Price and Rolling Beta
     1. Load Data from your Gold and Silver Views
     2. Create an Interactive Dashboard with Plotly"""
    conn = sqlite3.connect(db_path)

    conn.create_function("SQRT", 1, safe_sqrt)
    conn.create_function("POWER", 2, safe_pow)

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
    save_report(fig, ticker)
    # fig.show()

    
    # Save as Static Image (Requires 'kaleido' library in requirements.txt)
    # fig.write_image(f"{report_dir}/risk_report_{timestamp}.png")
    
    print(f"Report saved to {report_dir}/risk_report_{timestamp}.html")
    
def plot_stock_risk_with_panic(ticker, db_path=DATABASE_PATH):
    """
    Plot Stock Risk Dashboard with Panic Overlay
     1. Load Data including VIX from your Gold and Silver Views
     2. Create an Interactive Dashboard with Plotly including Panic Overlay where VIX > 20
    """

    conn = sqlite3.connect(db_path)
    conn.create_function("SQRT", 1, safe_sqrt)
    conn.create_function("POWER", 2, safe_pow)
    
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
    save_report(fig, ticker)
   
def plot_correlation_heatmap(db_path=DATABASE_PATH):
    """
    Plot Correlation Heatmap for all tickers based on Daily Returns
     1. Load Daily Returns Data from Silver View
     2. Calculate Correlation Matrix
     3. Create Heatmap using Plotly
    """
        
    conn = sqlite3.connect(db_path)
    
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
    save_report(fig, "correlation_heatmap")

# Try it out for NVDA!
if __name__ == "__main__":
    plot_stock_risk("NVDA")
    plot_stock_risk_with_panic("NVDA")
    plot_correlation_heatmap()