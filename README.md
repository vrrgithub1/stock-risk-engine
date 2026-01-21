# Stock Risk Engine

A professional-grade financial data pipeline built to analyze volatility and market sensitivity (β) using a Medallion Architecture on SQLite.

## Overview

The Stock Risk Engine is a comprehensive financial analytics platform that ingests stock market data, calculates key risk metrics, and provides insights into portfolio volatility and market correlations. The system implements a structured data lake approach ensuring data lineage and mathematical integrity at every layer.

### Architecture Diagram

The following is the Draw.io XML representation of the Stock Risk Engine architecture:

```xml
<mxfile host="app.diagrams.net" agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36" version="29.3.4">
  <diagram name="Page-1" id="Q5_Dq8RWVIdcXN7gq1a4">
    <mxGraphModel dx="566" dy="577" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="850" pageHeight="1100" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        <mxCell id="dTgfwxhhdpq2pmFGcOVE-1" parent="1" style="rounded=1;whiteSpace=wrap;html=1;align=center;verticalAlign=top;fontStyle=1" value="SQLite" vertex="1">
          <mxGeometry height="320" width="190" x="190" y="60" as="geometry" />
        </mxCell>
        <mxCell id="7qELfcKDKXbZDzLM8vZQ-9" parent="1" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#6d8764;fontColor=#ffffff;strokeColor=#3A5431;fontStyle=1" value="Bronze: Raw SQL" vertex="1">
          <mxGeometry height="60" width="120" x="220" y="100" as="geometry" />
        </mxCell>
        <mxCell id="7qELfcKDKXbZDzLM8vZQ-10" parent="1" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#6d8764;fontColor=#ffffff;strokeColor=#3A5431;fontStyle=1" value="Silver: Cleaned SQL" vertex="1">
          <mxGeometry height="60" width="120" x="220" y="190" as="geometry" />
        </mxCell>
        <mxCell id="7qELfcKDKXbZDzLM8vZQ-12" parent="1" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#6d8764;fontColor=#ffffff;strokeColor=#3A5431;fontStyle=1" value="Gold: Analytics View" vertex="1">
          <mxGeometry height="60" width="120" x="220" y="290" as="geometry" />
        </mxCell>
        <mxCell id="7qELfcKDKXbZDzLM8vZQ-17" edge="1" parent="1" source="7qELfcKDKXbZDzLM8vZQ-9" style="endArrow=classic;html=1;rounded=0;exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;" target="7qELfcKDKXbZDzLM8vZQ-10" value="">
          <mxGeometry height="50" relative="1" width="50" as="geometry">
            <mxPoint x="120" y="250" as="sourcePoint" />
            <mxPoint x="170" y="200" as="targetPoint" />
          </mxGeometry>
        </mxCell>
        <mxCell id="7qELfcKDKXbZDzLM8vZQ-18" edge="1" parent="1" source="7qELfcKDKXbZDzLM8vZQ-10" style="endArrow=classic;html=1;rounded=0;exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;" target="7qELfcKDKXbZDzLM8vZQ-12" value="">
          <mxGeometry height="50" relative="1" width="50" as="geometry">
            <mxPoint x="290" y="270" as="sourcePoint" />
            <mxPoint x="290" y="300" as="targetPoint" />
          </mxGeometry>
        </mxCell>
        <mxCell id="7qELfcKDKXbZDzLM8vZQ-8" parent="1" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#0050ef;fontColor=#ffffff;strokeColor=#001DBC;fontStyle=1" value="yfinance API" vertex="1">
          <mxGeometry height="60" width="120" x="30" y="100" as="geometry" />
        </mxCell>
        <mxCell id="7qELfcKDKXbZDzLM8vZQ-13" parent="1" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#ffcd28;strokeColor=#d79b00;gradientColor=#ffa500;fontStyle=1" value="Plotly Dashboard" vertex="1">
          <mxGeometry height="60" width="120" x="416" y="290" as="geometry" />
        </mxCell>
        <mxCell id="7qELfcKDKXbZDzLM8vZQ-16" edge="1" parent="1" source="7qELfcKDKXbZDzLM8vZQ-8" style="endArrow=classic;html=1;rounded=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;exitX=1;exitY=0.5;exitDx=0;exitDy=0;" target="7qELfcKDKXbZDzLM8vZQ-9" value="">
          <mxGeometry height="50" relative="1" width="50" as="geometry">
            <mxPoint x="150" y="140" as="sourcePoint" />
            <mxPoint x="200" y="90" as="targetPoint" />
          </mxGeometry>
        </mxCell>
        <mxCell id="7qELfcKDKXbZDzLM8vZQ-19" edge="1" parent="1" source="7qELfcKDKXbZDzLM8vZQ-12" style="endArrow=classic;html=1;rounded=0;exitX=1;exitY=0.5;exitDx=0;exitDy=0;" value="">
          <mxGeometry height="50" relative="1" width="50" as="geometry">
            <mxPoint x="290" y="370" as="sourcePoint" />
            <mxPoint x="420" y="320" as="targetPoint" />
          </mxGeometry>
        </mxCell>
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
```




## 🚀 Getting Started

**1. Clone the repo:** `git clone <your-repo-url>`<br>
**2. Setup Conda:** `conda env create -f environment.yml`<br>
**3. Configure Tickers:** Edit `config/tickers.yml` to track your preferred assets.<br>
**4. Run Pipeline:** `python main.py` (This builds the Bronze/Silver/Gold layers).<br>
**5. View Dashboard:** `python src/visualizer.py`<br>

## Architecture Overview

This project implements a Medallion Architecture for financial data processing:

* **Bronze Layer (Raw):** Immutable ledger of raw yfinance ingestion. Includes OHLCV data for equities and key macro indicators (Treasury Yields, VIX, S&P 500)
* **Silver Layer (Cleansed):** Deduplicated time-series data with standardized return calculations and rolling volatility metrics.
* **Gold Layer (Analytics):** High-value business logic including Rolling Beta calculations and Portfolio Stress Testing models.

## Key Quantitative Features

### 1. Rolling Volatility
Calculates the 30-day annualized standard deviation of returns. This helps identify "Volatility Regimes" where a stock's risk profile shifts independently of the market.

σₐₙₙᵤₐₗ = σₔₐᵢₗᵧ × √252

### 2. Rolling Market Beta (β)
Measures the systematic risk of an asset in relation to the S&P 500.
- β > 1: High sensitivity (Aggressive)
- β < 1: Low sensitivity (Defensive)

### 3. Historical Stress Testing
A simulation engine that identifies the "Maximum 5-Day Drawdown" for a custom-weighted portfolio, providing a realistic view of tail risk during historical market shocks.

## Tech Stack

* **Language:** Python 3.x
* **Database:** SQLite (File-based, serverless architecture)
* **Data Source:** Yahoo Finance API (yfinance)
* **Libraries:** Pandas, NumPy, SQLAlchemy, SciPy, StatsModels, Matplotlib, Seaborn

## Project Structure

```
stock-risk-engine/
├── environment.yml         # Conda environment configuration
├── init_project.sh         # Project initialization script
├── main.py                 # Main entry point for running the pipeline
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── config/
│   └── tickers.yml         # Configuration for stock tickers
├── data/
│   └── bronze/             # Raw data storage location
├── docs/                   # Documentation (empty)
├── sql/
│   └── init_analytics_layer.sql  # SQL script for analytics layer
└── src/
    ├── __init__.py         # Package initialization
    ├── app_visualizer.py   # Visualization module for risk metrics
    ├── database.py         # Database schema creation
    ├── ingestion.py        # DataIngestor class for fetching and saving data
    ├── maintenance.py      # Maintenance tasks (e.g., archiving)
    ├── setup_db.py         # Database setup utilities
    └── setup_db1.py        # Alternative database setup
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Installation Steps

1. **Clone or download the project**
   ```bash
   cd /path/to/your/projects
   # Assuming you have the project folder
   cd stock-risk-engine
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize the project structure** (optional, if starting fresh)
   ```bash
   chmod +x init_project.sh
   ./init_project.sh
   ```

4. **Set up the database**
   ```bash
   python src/setup_db.py
   ```

## Usage

### Data Ingestion

Run the main pipeline script to fetch stock data from Yahoo Finance and process it:

```bash
python main.py
```

This script will:
- Fetch data for predefined stocks (NVDA, TSLA, XOM, CVX, PG)
- Fetch macro indicators (^TNX, ^IRX, ^GSPC, ^IXIC, ^VIX)
- Save data to the bronze layer in SQLite
- Clean up duplicate entries
- Build analytical views for silver and gold layers
- Perform maintenance tasks like archiving old data

### Custom Data Ingestion

You can modify `main.py` or the ingestion logic in `src/ingestion.py` to fetch data for different stocks or date ranges:

```python
from src.ingestion import DataIngestor
import sqlite3

# Connect to database
conn = sqlite3.connect("data/stock_risk_vault.db")
ingestor = DataIngestor(conn)

# Fetch custom stocks
data = ingestor.fetch_stock_data(['AAPL', 'GOOGL'], '2023-01-01', '2024-01-01')
ingestor.save_to_bronze(data)

# Clean up duplicates
ingestor.cleanup_duplicates()
```

## Data Dictionary

### Bronze Layer Tables

#### Table: bronze_price_history
| Column | Data Type | Description |
|:--------|:----------|:------------|
| ticker | TEXT | Company ticker symbol |
| date | TEXT | Business or stock trade date (YYYY-MM-DD) |
| open | REAL | Opening trade price for the day |
| high | REAL | Maximum trade price for the day |
| low | REAL | Minimum trade price for the day |
| close | REAL | Closing trade price for the day |
| adj_close | REAL | Adjusted closing trade price for the day |
| volume | INTEGER | Trade volume for the day |

### Gold Layer Tables

#### Table: gold_risk_metrics
| Column | Data Type | Description |
|:--------|:----------|:------------|
| ticker | TEXT | Company ticker symbol |
| calculation_date | TEXT | Date when the metric was calculated |
| metric_type | TEXT | Type of risk metric (e.g., 'volatility', 'beta') |
| period_years | INTEGER | Lookback period in years |
| value | REAL | Calculated metric value |

## Configuration

The project uses configuration files in the `config/` directory. The `tickers.yml` file contains the list of stock tickers and macro indicators to be ingested. Configuration is handled programmatically, with room for expansion to YAML-based settings.

## Development

### Adding New Risk Metrics

1. Extend the database schema in `src/database.py`
2. Add calculation logic in a new module under `src/`
3. Update the ingestion pipeline as needed

### Testing

Currently, the project does not have automated tests. Manual testing can be performed by:
- Running the ingestion scripts
- Verifying data in the SQLite database
- Checking calculated metrics manually

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request


## 📈 Key Research Findings (Jan 2024 - Jan 2026)
| Ticker | Max Drawdown | Avg 30D Beta | Risk Category |
|:-------|:-------------|:-------------|:--------------|
| TSLA   | -53.76%      | 1.50         | Aggressive    |
| NVDA   | -21.45%      | 1.96         | High-Growth   |
| ^GSPC  | -18.90%      | 1.00         | Benchmark     |
| PG     | -8.12%       | -0.78        | Defensive     |

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. It should not be used for actual investment decisions without proper validation and professional financial advice. Past performance does not guarantee future results.









