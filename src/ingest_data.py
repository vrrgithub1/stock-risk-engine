import sqlite3
from ingestion import DataIngestor

db_name="data/stock_risk_vault.db"
conn = sqlite3.connect(db_name)
ingestor = DataIngestor(conn)

data = ingestor.fetch_stock_data(['NVDA', 'TSLA', 'XOM', 'CVX', 'PG'], '2024-01-01', '2026-01-19')
ingestor.save_to_bronze(data)


mi_data = ingestor.fetch_macro_indicator(['^TNX', '^IRX', '^GSPC', '^IXIC', '^VIX' ], '2024-01-01', '2026-01-19')
ingestor.save_to_bronze(mi_data)

ingestor.cleanup_duplicates()



