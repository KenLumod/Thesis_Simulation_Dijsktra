import pandas as pd
import os

target_file = r"c:\Users\aerod\OneDrive\Desktop\Thesis_Simulation_Dijsktra\Dijkstra_SafeZones\simulation_results_dijkstra_2.xlsx"

if not os.path.exists(target_file):
    print(f"❌ File NOT FOUND: {target_file}")
    exit()

print(f"Reading: {target_file}")
try:
    df = pd.read_excel(target_file, sheet_name='Run Comparisons')
    print("\n--- ALL ROWS ---")
    print(df[['Run', 'Remaining Agents', 'Total Casualties', 'Total Evacuated']].to_string())
    
    print("\n--- SEARCHING FOR AVERAGE ---")
    # mimicking comparison script logic
    df['Run'] = df['Run'].astype(str)
    avg_row = df[df['Run'] == 'AVERAGE']
    
    if not avg_row.empty:
        print("✅ Found AVERAGE Row:")
        print(avg_row.iloc[0])
    else:
        print("❌ No AVERAGE row found. Calculated Mean:")
        print(df.mean(numeric_only=True))

except Exception as e:
    print(f"Error reading file: {e}")
