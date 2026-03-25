# 🛠️ Project Evolution & Technical Milestones

This document tracks the iterative development of the Stock Risk Engine, from initial data ingestion to institutional-grade model validation.

## ✅ Phase VI: Macro Stress Testing & AI Governance (March 2026)

*Goal: Implement systemic risk simulations and align with NIST AI RMF standards.*

* **Stress Simulation Engine:** Developed a 'What-If' module using historical correlation matrices (GFC 2008, COVID 2020, Tech Bubble 2000).
* **Predictive Risk Mapping:** Integrated "Beta Shift" logic to account for correlation convergence during liquidity crises.
* **AI Governance Layer:**
* * Implemented **Model Cards** for transparency.
* * Aligned documentation with **ISO/IEC 42001** and **NIST AI RMF**.
* * Automated **Violation Rate** monitoring (Current: 2.47%).
* **UI/UX Refinement:** Moved global parameters to the sidebar and implemented sub-tab navigation for "Historical Audit" vs. "Predictive Simulation."

## **Phase V: Model Validation & Governance**

* **Goal:** Certify model accuracy.
* **Outcome:** Achieved 3.28% violation rate.
* **Innovation:** Migrated backtesting to the internal Silver Layer for zero-lag reporting.

## **Phase IV: The Gold Layer & Visualization**

* **Goal:** Curate data for high-performance reporting.
* **Outcome:** Built a Streamlit dashboard with 10,000-path Monte Carlo simulations.

## **Phase III: Risk Engineering Core**

* **Goal:** Quantify downside risk.
* **Outcome:** Automated 95% VaR calculations using a 130-day trailing window.

## **Phase II: The Silver Layer (Feature Engineering)**

* **Goal:** Transform raw prices into actionable metrics.
* **Outcome:** Developed automated pipelines for Rolling Volatility and Beta.

## **Phase I: The Bronze Layer (Data Ingestion)**

* **Goal:** Build the foundation.
* **Outcome:** Designed a resilient SQLite schema and `yfinance` scraper.
