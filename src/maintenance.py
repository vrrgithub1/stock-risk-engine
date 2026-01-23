import sqlite3
from datetime import datetime, timedelta
from config import DATABASE_PATH

def archive_old_data(days_to_keep=1095, db_path=DATABASE_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Calculate the cutoff date
    cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).strftime('%Y-%m-%d')
    print(f"Archiving data older than {cutoff_date}...")

    try:
        # 1. Create the Historical Table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS historical_price_archive AS 
            SELECT * FROM bronze_price_history WHERE 1=0
        """)

        # 2. Move data: Copy to archive, then delete from main
        cursor.execute("""
            INSERT INTO historical_price_archive 
            SELECT * FROM bronze_price_history WHERE date < ?
        """, (cutoff_date,))
        
        cursor.execute("DELETE FROM bronze_price_history WHERE date < ?", (cutoff_date,))

        # 3. VACUUM to shrink the database file and recover space
        conn.commit()
        conn.execute("VACUUM")
        print("Archive complete. Database optimized.")
        
    except Exception as e:
        print(f"Error during cleanup: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    archive_old_data()
    