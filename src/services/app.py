import streamlit as st
import plotly.graph_objects as go
from src.services.reporting import ReportGenerator # Import your generator
import pandas as pd
import sqlite3
from src.utils.config import (
    DATABASE_PATH, 
    get_benchmark_tickers_from_yaml,
    get_index_tickers_from_yaml,
    get_tickers_from_yaml
)

def render_what_if_analysis(db_path, selected_scenario, shock, multiplier):
    st.subheader("🧪 'What-If' Hypothetical Impact")
    st.markdown(f"### 🔮 Predictive Simulation: {selected_scenario.replace('_', ' ')}")
    st.info(f"**Scenario Logic:** Applying **{selected_scenario}** correlations to a **{shock}%** market drop.")

    pf_tickers = get_tickers_from_yaml()
    benchmark_tickers = get_benchmark_tickers_from_yaml()
    index_tickers = get_index_tickers_from_yaml()
    print( type(pf_tickers), type(benchmark_tickers), type(index_tickers)) # Debugging line to check types
    print(f"DEBUG: Tickers from YAML - PF: {pf_tickers}, Benchmarks: {benchmark_tickers}, Indices: {index_tickers}")

    conn = sqlite3.connect(db_path)
    
    # 1. Fetch Stress Data
    df_stress = pd.read_sql("""
        SELECT * 
        FROM gold_stress_scenarios
        WHERE run_date = (SELECT MAX(run_date) FROM gold_stress_scenarios)
    """, conn)
    
    conn.close()

    # Filter for the selected historical regime to get relevant correlations
# 1. Filter and CLEAN the data first
    # This removes tickers that don't have correlation data for the selected period (e.g., TSLA in 2008)

    allowed_tickers_list = pf_tickers + index_tickers
    not_allowed_tickers_list = benchmark_tickers # Assuming you want to exclude benchmarks from the 'What-If' analysis
    
    filter_condition = df_stress['ticker'].isin(allowed_tickers_list) & ~df_stress['ticker'].isin(not_allowed_tickers_list)
    df_filtered = df_stress[filter_condition]

    active_data = df_filtered[
        (df_filtered['scenario_name'] == selected_scenario) & 
        (df_filtered['correlation_to_market'].notna()) &
        (df_filtered['ticker'].isin(allowed_tickers_list))
    ].copy()

    if active_data.empty:
        st.warning(f"No correlation data available for the {selected_scenario} period.")
        return

    # 2. Calculate Projected Loss
    active_data['projected_loss'] = (shock / 100) * active_data['correlation_to_market'] * multiplier
    
    # 3. Create columns ONLY for the valid rows we have
    num_assets = len(active_data)
    cols = st.columns(num_assets)
    
    # 4. Use .reset_index() to ensure 'i' always matches the column index correctly
    for i, row in active_data.reset_index(drop=True).iterrows():
        with cols[i]:
            st.metric(
                label=f"{row['ticker']} Impact", 
                value=f"{row['projected_loss']:.2%}",
                delta=f"Corr: {row['correlation_to_market']:.2f}",
                delta_color="inverse"
            )

    # Bar Chart for Impact
    fig = go.Figure(go.Bar(
        x=active_data['ticker'],
        y=active_data['projected_loss'],
        marker_color='orange',
        text=[f"{x:.1%}" for x in active_data['projected_loss']],
        textposition='auto',
    ))
    fig.update_layout(
        title="Projected Asset Devaluation",
        yaxis_title="Estimated Return (%)",
        template="plotly_dark",
        yaxis=dict(range=[min(active_data['projected_loss'])*1.2, 0])
    )
    st.plotly_chart(fig, use_container_width=True)


def render_stress_tab(db_path, selected_scenario):
    st.header("📊 Phase VI: Macro Stress Scenarios")
    st.subheader("📜 Historical Regime Analysis")
    st.info("Showing actual performance during selected historical crash.")    
    st.markdown("Comparing current portfolio risk against historical 'Black Swan' events.")

    conn = sqlite3.connect(db_path)
    
    # 1. Fetch Stress Data
    df_stress = pd.read_sql("""
        SELECT * 
        FROM gold_stress_scenarios
        WHERE run_date = (SELECT MAX(run_date) FROM gold_stress_scenarios)
    """, conn)
    
    # 2. Fetch Latest Normal VaR (from Phase V results) for comparison
    # Assuming your backtesting or risk_metrics table has the latest 95% VaR
    df_normal = pd.read_sql("""
        SELECT ticker, predicted_var_95 as normal_var 
        FROM gold_risk_backtesting
        WHERE forecast_date = (SELECT MAX(forecast_date) FROM gold_risk_backtesting)
    """, conn)
    
    conn.close()

    # Scenario Selection
#    scenarios = df_stress['scenario_name'].unique()
#    selected_scenario = st.selectbox("Select Stress Scenario", scenarios)

    # Filter data for selected scenario
    scenario_data = df_stress[df_stress['scenario_name'] == selected_scenario].merge(df_normal, on='ticker')

    # Display Key Metric Cards
    avg_drawdown = scenario_data['historical_max_drawdown'].mean()
    st.columns(3)[0].metric("Avg. Scenario Drawdown", f"{avg_drawdown:.2%}", delta_color="inverse")

    # --- Comparison Table ---
    st.subheader(f"Risk Comparison: {selected_scenario.replace('_', ' ')}")
    
    # Calculate the 'Stress Gap'
    scenario_data['stress_gap'] = scenario_data['simulated_stress_var'] - scenario_data['normal_var']
    
    display_df = scenario_data[[
        'ticker', 'normal_var', 'simulated_stress_var', 'stress_gap', 'historical_max_drawdown'
    ]].copy()
    
    # Format for display
    st.dataframe(display_df.style.format({
        'normal_var': '{:.2%}',
        'simulated_stress_var': '{:.2%}',
        'stress_gap': '{:.2%}',
        'historical_max_drawdown': '{:.2%}'
    }), column_config={
        'ticker': { "label": "Asset", "alignment": "left" },
        'normal_var': { "label": "Normal VaR", "alignment": "right" },
        'simulated_stress_var': { "label": "Stress VaR", "alignment": "right" },
        'stress_gap': { "label": "Stress Gap", "alignment": "right" },
        'historical_max_drawdown': { "label": "Historical Max Drawdown", "alignment": "right" }
    }, hide_index=True, use_container_width=True)

    # --- Stress Gap Visualization ---
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=scenario_data['ticker'],
        y=scenario_data['normal_var'],
        name='Normal VaR (Phase V)',
        marker_color='royalblue'
    ))
    
    fig.add_trace(go.Bar(
        x=scenario_data['ticker'],
        y=scenario_data['simulated_stress_var'],
        name='Stress VaR (Phase VI)',
        marker_color='crimson'
    ))

    fig.update_layout(
        title=f"Normal vs. Stress VaR: {selected_scenario}",
        barmode='group',
        yaxis_title="Downside Risk (%)",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)

    st.info("💡 **Insight:** A larger 'Stress Gap' (the difference between bars) indicates an asset that is highly sensitive to market shocks despite appearing stable in normal conditions.")

#    st.divider()

#    st.subheader("🔮 Predictive 'What-If' Simulation")
#    st.write("Adjust the sidebar sliders to simulate a custom shock based on historical correlations.")
#    render_what_if_analysis(df_stress, selected_scenario, market_shock, vol_multiplier)

def render_model_card():
    st.header("⚖️ AI Governance: Model Card (v6.0.0)")
    st.info("This documentation aligns with NIST AI RMF and ISO/IEC 42001 transparency standards.")

    # --- SECTION 1: MODEL DETAILS ---
    with st.expander("📝 1. Model Details & Intended Use", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Model Type:** Monte Carlo VaR Simulation  
            **Version:** 6.0.0 (Phase VI Certified)  
            **Primary Use:** Portfolio Downside Risk Estimation
            """)
        with col2:
            st.markdown("""
            **Developer:** Finance Professional (21+ yrs Exp)  
            **Framework:** Python / SQL / Streamlit  
            **Status:** Production-Ready / Validated
            """)

    # --- SECTION 2: DATA AUDIT ---
    with st.expander("📂 2. Data Provenance & Integrity"):
        st.markdown("""
        - **Source:** Yahoo Finance API (Bronze Layer)
        - **Pipeline:** Medallion Architecture (Bronze -> Silver -> Gold)
        - **History:** 130-day trailing lookback with daily refresh.
        - **Integrity Check:** Automated outlier detection and missing value imputation for ticker data.
        """)

    # --- SECTION 3: QUANTITATIVE VALIDATION ---
    with st.expander("🎯 3. Model Health & Performance"):
        # You can pull these dynamically if you have them in variables
        st.markdown(f"""
        - **Confidence Level:** 95%
        - **Certification Metric:** Violation Rate < 5.0%
        - **Current Performance:** ✅ **2.63%** (Based on Phase V Backtest)
        - **Validation Method:** Daily realized return comparison vs. predicted floor.
        """)

    # --- SECTION 4: STRESS & LIMITATIONS ---
    with st.expander("⚠️ 4. Risks, Limitations & Mitigations"):
        st.warning("**Limitation:** Standard VaR assumes 'normal' market distributions and may underrepresent tail-risk.")
        st.success("**Mitigation:** Integrated Phase VI Macro Stress scenarios (COVID, GFC, Tech Bubble) to provide a non-linear safety margin.")
        st.markdown("""
        - **Out of Scope:** High-frequency trading, illiquid 'Penny' stocks, and cryptocurrencies.
        - **Human-in-the-Loop:** Requires qualitative review of 'What-If' scenarios for strategic capital allocation.
        """)

    st.caption("Last Governance Audit: March 2026")

# Initialize your reporting service
report_gen = ReportGenerator()

st.set_page_config(page_title="Risk Command Center", layout="wide")

# Sidebar - Global Filters
st.sidebar.header("🌍 Market Regime Selection")


conn = sqlite3.connect(DATABASE_PATH)

# 1. Fetch Stress Data
df_stress = pd.read_sql("""
    SELECT * 
    FROM gold_stress_scenarios
    WHERE run_date = (SELECT MAX(run_date) FROM gold_stress_scenarios)
""", conn)

conn.close()

scenarios = df_stress['scenario_name'].unique()
selected_scenario = st.sidebar.selectbox("Select Historical Stress Event", scenarios)

st.sidebar.divider()

# Sidebar - Hypothetical Controls
st.sidebar.header("🕹️ Hypothetical Shock Controls")
market_shock = st.sidebar.slider("Market Crash Scenario (%)", min_value=-30, max_value=0, value=-10, step=-1)
vol_multiplier = st.sidebar.slider("Volatility Multiplier", min_value=1.0, max_value=3.0, value=1.5, step=0.1)

# 1. Fetch Data via the Backend
df = report_gen.get_backtest_summary()
print(df.head()) # Debugging line to check the structure of the DataFrame

st.title("🛡️ Institutional Risk Command Center")
st.markdown(f"**Data Status:** Monitoring {df['ticker'].nunique()} tickers across {df['sector'].nunique()} sectors.")

# 2. Logic for the Health Gauge
total_forecasts = len(df)
total_violations = df['is_violation'].sum()
violation_rate = (total_violations / total_forecasts) * 100

tab1, tab2, tab3, tab4 = st.tabs(["⚖️ Model Card", "🛡️ Phase V: Model Health", "📉 Breach Timeline", "🔥 Phase VI: Macro Stress"])

with tab1:
    render_model_card()

with tab2:
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
        df_timeline['forecast_date'] = pd.to_datetime(df_timeline['forecast_date'])

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

with tab3:
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

with tab4:
    sub_tab1, sub_tab2 = st.tabs(["📜 Historical Audit", "🔮 Predictive Simulation"])
    with sub_tab1:
        render_stress_tab(DATABASE_PATH, selected_scenario)
    with sub_tab2:
        st.write("Adjust the sidebar sliders to simulate a custom shock based on historical correlations.")
        render_what_if_analysis(DATABASE_PATH, selected_scenario, market_shock, vol_multiplier)

