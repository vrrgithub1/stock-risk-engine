import pandas as pd
import yfinance as yf
import yaml
import sqlite3
from datetime import datetime

def calculate_stress_metrics(ticker, start_date, end_date, market_proxy="^GSPC"):
    # Download historical data for the stress period
    data = yf.download([ticker, market_proxy], start=start_date, end=end_date)['Close']
    
    if data.empty or ticker not in data:
        return None

    returns = data.pct_change().dropna()
    
    # 1. Historical Max Drawdown during the period
    cumulative_returns = (1 + returns[ticker]).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min()
    
    # 2. Correlation to Market during stress
    correlation = returns[ticker].corr(returns[market_proxy])
    
    # 3. Simulated Stress VaR 
    # (Applying the historical volatility of the crash to current prices)
    stress_vol = returns[ticker].std()
    simulated_stress_var = -(1.645 * stress_vol) # 95% Confidence simplified for stress
    
    return {
        'max_drawdown': float(max_drawdown),
        'correlation': float(correlation),
        'stress_var': float(simulated_stress_var)
    }

def run_phase_vi_stress_test(db_path, config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Get your current ticker list from your existing silver/gold layers
    conn = sqlite3.connect(db_path)
    tickers = pd.read_sql("SELECT DISTINCT ticker FROM silver_returns", conn)['ticker'].tolist()
    
    results = []
    run_date = datetime.now().strftime('%Y-%m-%d')
    
    for event in config['historical_events']:
        print(f"Running Scenario: {event['name']}")
        for ticker in tickers:
            metrics = calculate_stress_metrics(ticker, event['start'], event['end'], config['market_proxy'])
            
            if metrics:
                results.append((
                    event['name'],
                    ticker,
                    run_date,
                    metrics['max_drawdown'],
                    metrics['stress_var'],
                    metrics['correlation']
                ))

    # Bulk Insert into your new table
    cursor = conn.cursor()
    cursor.executemany("""
        INSERT OR REPLACE INTO gold_stress_scenarios 
        (scenario_name, ticker, run_date, historical_max_drawdown, simulated_stress_var, correlation_to_market)
        VALUES (?, ?, ?, ?, ?, ?)
    """, results)
    
    conn.commit()
    conn.close()
    print(f"Successfully injected {len(results)} stress records into Gold Layer.")
    