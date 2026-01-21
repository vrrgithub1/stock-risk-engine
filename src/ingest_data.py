import sqlite3
from ingestion import DataIngestor
from datetime import date

today = date.today()
end_date = today.strftime("%Y-%m-%d")
start_date = "2024-01-01"

db_name="data/stock_risk_vault.db"
conn = sqlite3.connect(db_name)
ingestor = DataIngestor(conn)

data = ingestor.fetch_stock_data(['NVDA', 'TSLA', 'XOM', 'CVX', 'PG'], start_date, end_date)
ingestor.save_to_bronze(data)


mi_data = ingestor.fetch_macro_indicator(['^TNX', '^IRX', '^GSPC', '^IXIC', '^VIX' ], start_date, end_date)
ingestor.save_to_bronze(mi_data)

ingestor.cleanup_duplicates()



