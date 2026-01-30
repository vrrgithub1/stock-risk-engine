import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from config import DATABASE_PATH
import sqlite3


def fetch_inference_data(db_path = DATABASE_PATH):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)

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
    return inference_df


def generate_phase2_risk_report(inference_df, importance_data, vix_metadata):
    """
    Generates the Phase II Executive Report with Forecasts, Logic, and Market Context.
    
    :param inference_df: Dataframe containing gold_risk_inference records
    :param importance_data: Dict with 'Feature' and 'Weight' keys
    :param vix_metadata: Dict with 'current_level' and 'regime_label'
    """
    
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
    for ticker in ['NVDA', 'TSLA']:
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
        x=importance_data['Weight'], 
        y=importance_data['Feature'], 
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
    
    return fig

def get_regime_label(vix_level):
    if vix_level == 0:        
        return "Quiet (Low Volatility)"
    elif vix_level == 1:
        return "Standard (Moderate Risk)"
    else:
        return "Stress (High Risk/Panic)"
    
# --- EXAMPLE USAGE ---

df_inference = fetch_inference_data()
print(df_inference)
importance_dict = {'Feature': ['Beta', 'Vol', 'Return', 'VIX'], 'Weight': [0.386, 0.344, 0.215, 0.054]}
conn = sqlite3.connect(DATABASE_PATH)
df_vix = pd.read_sql("SELECT * FROM gold_market_regime_vix ORDER BY date DESC LIMIT 1", conn)

current_vix_level = df_vix['adj_close'].iloc[-1]
current_regime_id = df_vix['market_regime_vix'].iloc[-1]

regime_label = get_regime_label(current_regime_id) # Logic: Standard if 15 <= VIX <= 25

vix_info = {'current_level': current_vix_level, 'regime_label': regime_label}
fig = generate_phase2_risk_report(df_inference, importance_dict, vix_info)
fig.show()
