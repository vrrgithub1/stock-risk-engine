import streamlit as st
import plotly.graph_objects as go
from src.services.reporting import ReportGenerator # Import your generator

# Initialize your reporting service
report_gen = ReportGenerator()

st.set_page_config(page_title="Risk Command Center", layout="wide")

# 1. Fetch Data via the Backend
df = report_gen.get_backtest_summary()

st.title("🛡️ Institutional Risk Command Center")
st.markdown(f"**Data Status:** Monitoring {df['ticker'].nunique()} tickers across {df['sector'].nunique()} sectors.")

# 2. Logic for the Health Gauge
total_forecasts = len(df)
total_violations = df['is_violation'].sum()
violation_rate = (total_violations / total_forecasts) * 100

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Model Health Score")
    
    # Determine Status for the caption
    if violation_rate <= 5.0:
        status_color = "green"
        summary_text = "✅ **PASS**: Model is calibrated correctly."
    elif violation_rate <= 10.0:
        status_color = "orange"
        summary_text = "⚠️ **WARNING**: Model is slightly aggressive."
    else:
        status_color = "red"
        summary_text = "🚨 **FAIL**: Model requires recalibration."

    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = violation_rate,
        number = {'suffix': "%", 'font': {'size': 40}},
        gauge = {
            'axis': {'range': [0, 15], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "white"},
            'steps': [
                {'range': [0, 5], 'color': "#2ecc71"},
                {'range': [5, 10], 'color': "#f1c40f"},
                {'range': [10, 15], 'color': "#e74c3c"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 5 # This is your target 95% confidence limit
            }
        }
    ))
    fig_gauge.update_layout(height=300, margin=dict(l=30, r=30, t=30, b=0))
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Add the Caption
    st.markdown(summary_text)
    st.caption(f"Target: < 5% violation rate (95% Confidence). Current realization is {violation_rate:.2f}%.")
    
with col2:
    st.subheader("VaR Breach Timeline")
    
    # Sort data for time-series consistency
    df_timeline = df.sort_values('forecast_date')
    
    fig_timeline = go.Figure()

    # 1. Add the Actual Returns as Bars
    fig_timeline.add_trace(go.Bar(
        x=df_timeline['forecast_date'],
        y=df_timeline['actual_return'],
        name="Actual Return",
        marker_color=['#e74c3c' if v else '#3498db' for v in df_timeline['is_violation']]
    ))

    # 2. Add the VaR Floor as a Line
    fig_timeline.add_trace(go.Scatter(
        x=df_timeline['forecast_date'],
        y=df_timeline['predicted_var_95'],
        mode='lines',
        name="95% VaR Floor",
        line=dict(color='orange', width=2, dash='dot')
    ))

    fig_timeline.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=20, b=20),
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

# --- Extreme Event Tracker Section ---
st.divider()
st.subheader("🚨 Extreme Event Tracker (Exception Attribution)")

extreme_events = report_gen.get_extreme_events(threshold=-0.005) # 0.5% threshold

if not extreme_events.empty:
    st.warning(f"Found {len(extreme_events)} events where the crash exceeded the VaR floor by more than 0.5%.")
    
    # Format the dataframe for display
    display_df = extreme_events.copy()
    display_df['breach_magnitude'] = display_df['breach_magnitude'].map("{:.2%}".format)
    display_df['actual_return'] = display_df['actual_return'].map("{:.2%}".format)
    display_df['predicted_var_95'] = display_df['predicted_var_95'].map("{:.2%}".format)

    st.dataframe(
        display_df,
        column_config={
            "ticker": "Asset",
            "forecast_date": "Event Date",
            "predicted_var_95": "Model Floor",
            "actual_return": "Actual Crash",
            "breach_magnitude": st.column_config.TextColumn(
                "Excess Loss",
                help="How much worse the return was compared to the VaR prediction"
            )
        },
        hide_index=True,
        use_container_width=True
    )
else:
    st.success("No extreme breaches detected. Model tails are well-contained.")
