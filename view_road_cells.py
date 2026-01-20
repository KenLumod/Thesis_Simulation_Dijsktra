import sqlite3
import pandas as pd
import os

# Set display options to show all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 20)

def inspect_gpkg_tables(gpkg_path):
    if not os.path.exists(gpkg_path):
        print(f"Error: File '{gpkg_path}' not found.")
        return

    print(f"Inspecting GeoPackage: {gpkg_path}")
    print("="*80)

    try:
        conn = sqlite3.connect(gpkg_path)
        
        # Tables to inspect based on user request
        # "spawners" and "bldig exits" usually refer to building_exits in this dataset
        target_tables = {
            'University Exits': 'university_exits',
            'Building Exits (Spawners)': 'building_exits',
            'Road Cells': 'road_cells'
        }

        for label, table_name in target_tables.items():
            print(f"\n[{label} - Table: {table_name}]")
            print("-" * 80)
            
            # Check if table exists
            query_check = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
            if not conn.execute(query_check).fetchone():
                print(f"Table '{table_name}' not found.")
                continue

            # Get column info
            query_info = f"PRAGMA table_info([{table_name}])"
            columns_info = conn.execute(query_info).fetchall()
            col_names = [col[1] for col in columns_info]
            
            print(f"Columns: {col_names}")
            
            # Get data
            query_data = f"SELECT * FROM [{table_name}]"
            df = pd.read_sql_query(query_data, conn)
            
            # Display stats
            print(f"Row Count: {len(df)}")
            print("\nSample Data (First 5 rows):")
            print(df.head())
            
            if 'spawn_rate' in df.columns:
                print("\nSpawn Rate Stats:")
                print(df['spawn_rate'].describe())

            print("\n")

        conn.close()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Check if a file path was provided as a command line argument
    if len(sys.argv) > 1:
        gpkg_file = sys.argv[1]
    else:
        # Default options to search for
        gpkg_dir = Path(__file__).parent / "GPKG_Files"
        possible_files = [
            gpkg_dir / "csu_map.gpkg",
            gpkg_dir / "xu-road-cells.gpkg",
            gpkg_dir / "road_cells_split.gpkg"
        ]
        
        gpkg_file = None
        for p in possible_files:
            if p.exists():
                gpkg_file = str(p)
                break
        
        if gpkg_file is None and gpkg_dir.exists():
            # Fallback to any gpkg found
            found = list(gpkg_dir.glob("*.gpkg"))
            if found:
                gpkg_file = str(found[0])

    if not gpkg_file or not os.path.exists(gpkg_file):
        print(f"❌ Error: No valid GPKG file found!")
        print(f"Searched for: {gpkg_file}")
        print("Usage: python view_road_cells.py [path_to_gpkg]")
    else:
        print(f"Using GPKG file: {gpkg_file}")
        print("Exporting data to Excel...")
        
        try:
            conn = sqlite3.connect(gpkg_file)
            output_excel = "gpkg_data_split.xlsx"
            
            target_tables = {
                'University/Campus Exits': 'campus_exits',
                'Building Exits': 'building_exits',
                'Road Cells': 'road_cells(10x6m)', # Fallback or specific name
                'Road Cells (Alt)': 'road_cells' # Try generic name too
            }
            
            # Since we iterate over target_tables, we need to be careful about not duplicating
            # Let's inspect what tables are actually in the DB first to match the best one?
            # Or just let it skip missing ones.
            
            with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
                data_found = False
                for label, table_name in target_tables.items():
                    # Check if table exists
                    query_check = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
                    if not conn.execute(query_check).fetchone():
                         # Don't spam warnings for alternate names
                        # print(f"⚠️ Table '{table_name}' ({label}) not found - skipping.")
                        continue

                    # Get data
                    query_data = f"SELECT * FROM [{table_name}]"
                    df = pd.read_sql_query(query_data, conn)
                    
                    # specific clean up for geometry if needed, converting bytes to string for excel
                    if 'geom' in df.columns:
                        df['geom'] = df['geom'].apply(lambda x: str(x) if isinstance(x, bytes) else x)
                    
                    # Write to sheet
                    sheet_name = table_name[:31] # Excel sheet limit
                    
                    # Avoid duplicate sheet names if both 'road_cells' and 'road_cells(10x6m)' exist (unlikely but safe)
                    if sheet_name in writer.book.sheetnames:
                         continue
                         
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"✅ Exported '{table_name}' to sheet '{sheet_name}' ({len(df)} rows)")
                    data_found = True
                
                if data_found:
                    print(f"\n🎉 Successfully saved to: {os.path.abspath(output_excel)}")
                else:
                    print("\n❌ No matching tables found to export.")

            conn.close()
            
        except Exception as e:
            print(f"❌ Error during export: {e}")
            # If openpyxl missing, suggest installation
            if "No module named 'openpyxl'" in str(e):
                print("Try running: pip install openpyxl")
