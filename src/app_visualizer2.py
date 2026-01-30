import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from config import DATABASE_PATH
import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect(DATABASE_PATH)

# 1. Load your Model Feature Store Data
df = pd.read_sql("SELECT * FROM silver_risk_features", conn)

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

print("✅ Model Trained and Inference Completed!")
print(importance)

# Below code is for visualization for inference results
# 1. Run the prediction logic (assuming 'inference_df' is ready)
# Let's visualize NVDA and TSLA specifically
tickers = ['NVDA', 'TSLA']
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
fig.show()
