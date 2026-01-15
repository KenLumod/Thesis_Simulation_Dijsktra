import sqlite3
import sys

def list_tables(gpkg_path):
    print(f"Inspecting: {gpkg_path}")
    try:
        conn = sqlite3.connect(gpkg_path)
        tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        print("Tables found:")
        for t in tables:
            print(f"- {t}")
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_tables("C:/Users/aerod/OneDrive/Desktop/Thesis/GPKG_Files/road_cells_split.gpkg")
