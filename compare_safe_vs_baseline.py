import pandas as pd
import os

# Paths (Updated for Safe vs Baseline comparison)
# Assuming run from root "Thesis_Simulation_Dijsktra" folder or adjusting relative paths
files = {
    'Baseline': r"Baseline/simulation_results_base_2.xlsx",
    'Enhanced': r"Dijkstra_SafeZones/simulation_results_dijkstra_2.xlsx"
}

# Adjust paths to absolute if needed based on CWD
base_dir = os.path.dirname(os.path.abspath(__file__))
for name, p in files.items():
    if not os.path.isabs(p):
        files[name] = os.path.join(base_dir, p)

output_file = "comparison_safe_vs_baseline.txt"

def load_data(name, path):
    if not os.path.exists(path):
        print(f"Warning: {name} file not found at {path}")
        return None
    try:
        data = {}
        # 1. Scalars
        try:
            df_runs = pd.read_excel(path, sheet_name='Run Comparisons')
            # Look for AVERAGE row 
            # Note: The scripts sometimes save 'Run' column as integer or 'AVERAGE' string
            # Force conversion to string to be safe
            df_runs['Run'] = df_runs['Run'].astype(str)
            avg_row = df_runs[df_runs['Run'] == 'AVERAGE']
            
            if not avg_row.empty:
                data['scalars'] = avg_row.iloc[0]
            else:
                # Fallback: if only one run or no average row, take mean
                # Exclude non-numeric columns if any
                data['scalars'] = df_runs.mean(numeric_only=True)
        except:
             # Fallback for single run files which might put scalars in Summary sheet
             try:
                 data['scalars'] = pd.read_excel(path, sheet_name='Summary').iloc[0]
             except:
                 data['scalars'] = None

        # 2. Time Series
        try:
            data['time'] = pd.read_excel(path, sheet_name='Avg Time Series')
        except:
            data['time'] = None
        
        # 3. Spatial
        try:
            data['spatial'] = pd.read_excel(path, sheet_name='Avg Spatial Dist')
        except:
            data['spatial'] = None
            
        # 4. Exits
        try:
            data['exits'] = pd.read_excel(path, sheet_name='Avg Exit Usage')
        except:
            data['exits'] = None
            
        return data
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

# Enhanced load_data to get metadata
def load_data_with_meta(name, path):
    data = load_data(name, path)
    if not data: return None
    
    try:
        # Re-read key sheets for metadata
        df_runs = pd.read_excel(path, sheet_name='Run Comparisons')
        # Check for numeric runs
        numeric_runs = df_runs[pd.to_numeric(df_runs['Run'], errors='coerce').notnull()]
        data['iterations'] = len(numeric_runs)
        
        # Agents (Sum ofavgs)
        s = data['scalars']
        if s is not None:
             ev = s.get('Total Evacuated', 0)
             rem = s.get('Remaining Agents', 0)
             cas = s.get('Total Casualties', 0)
             data['agent_count'] = int(round(ev + rem + cas))
        else:
             data['agent_count'] = "?"
             
        # Duration
        if data['time'] is not None and 'Time (s)' in data['time']:
             data['duration'] = int(data['time']['Time (s)'].max())
        else:
             data['duration'] = "?"
             
    except Exception as e:
        print(f"Meta error {name}: {e}")
        data['iterations'] = "?"
        data['agent_count'] = "?"
        data['duration'] = "?"
        
    return data

# Load models
models = {}
for name, path in files.items():
    print(f"Loading {name} from {path}...")
    res = load_data_with_meta(name, path)
    if res: models[name] = res

with open(output_file, "w", encoding="utf-8") as f:
    if not models:
        msg = "No simulation data loaded. Please run the simulations first."
        print(msg)
        f.write(msg + "\n")
    else:
        # Title
        f.write("================================================================================\n")
        f.write("                  SAFE ZONE VS BASELINE COMPARISON REPORT                       \n")
        f.write("================================================================================\n")
        
        # METADATA HEADER
        first_m = list(models.values())[0]
        iters = first_m.get('iterations', '?')
        agents = first_m.get('agent_count', '?')
        dur = first_m.get('duration', '?')
        
        f.write(f"CONFIGURATION: {iters} Iterations | {agents} Agents | {dur} Seconds Max\n")
        f.write("================================================================================\n\n")
        
        model_names = list(models.keys())
        # Sort to ensure Baseline is left (comparison reference) if present
        if 'Baseline' in model_names:
            model_names.sort(key=lambda x: 0 if x == 'Baseline' else 1)
        
        # 1. SCALAR METRICS
        f.write("1. OVERALL PERFORMANCE METRICS (RSME / Efficiency)\n")
        f.write("-" * 120 + "\n")
        # Header
        header = f"{'Metric':<30} | " + " | ".join([f"{m:<15}" for m in model_names]) + " | Interpretation"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        
        check_metrics = [
            ('Avg Exit Flow Rate', 'Avg Exit Flow Rate', 'Higher is better (evacuation speed)'),
            ('Remaining Agents', 'Remaining Agents', 'Lower is better (clearance)'),
            ('Total Evacuated', 'Total Evacuated', 'Higher is better'),
            ('Total Casualties', 'Total Casualties', 'Lower is better (safety)'),
            ('SEP', 'SEP', 'SEP'), 
            ('Avg Density (p/m2)', 'Avg Density', 'Lower is better (congestion)'),
            ('Avg Velocity (m/s)', 'Avg Velocity', 'Higher is better (mobility)'),
            ('Peak Density (p/m2)', 'Peak Density', 'Lower is better (crush risk)'),
            ('Total Evacuation Time (s)', 'Total Time', 'Lower is better'),
        ]
        
        for m_key, m_display, interp in check_metrics:
            if m_key == 'SEP':
                 f.write("\n")
                 continue
            
            row_str = f"{m_display:<30} | "
            for m in model_names:
                val = "N/A"
                if models[m]['scalars'] is not None:
                    # Try exact key, then partial match
                    s_data = models[m]['scalars']
                    if m_key in s_data:
                        val = s_data[m_key]
                    else:
                        # Fuzzy search keys
                        for k in s_data.keys():
                            if m_key in str(k) or m_display in str(k):
                                val = s_data[k]
                                break
                                
                if isinstance(val, (int, float)):
                    row_str += f"{val:<15.2f} | "
                else:
                    row_str += f"{str(val):<15} | "
            
            f.write(f"{row_str}{interp}\n")
        f.write("\n")

        # 2. EXIT USAGE
        f.write("2. EXIT USAGE DISTRIBUTION (Load Balancing)\n")
        f.write("-" * 80 + "\n")
        
        # Collect all Exit IDs
        all_exits = set()
        for m in model_names:
            if models[m]['exits'] is not None and 'Exit ID' in models[m]['exits'].columns:
                all_exits.update(models[m]['exits']['Exit ID'].unique())
        
        sorted_exits = sorted(list(all_exits))
        if sorted_exits:
            header = f"{'Exit ID':<10} | " + " | ".join([f"{m + ' Usage':<15}" for m in model_names])
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            
            for eid in sorted_exits:
                row_str = f"{int(eid):<10} | "
                for m in model_names:
                    val = 0.0
                    if models[m]['exits'] is not None:
                        df = models[m]['exits']
                        match = df[df['Exit ID'] == eid]
                        if not match.empty:
                            # Try 'Exit Usage Distribution' or similar
                            try:
                                val = match.iloc[0]['Exit Usage Distribution']
                            except:
                                # Startswith 'Avg Exit Usage'
                                col = [c for c in df.columns if 'Usage' in c][0]
                                val = match.iloc[0][col]
                    row_str += f"{val:<15.1f} | "
                f.write(row_str + "\n")
        else:
            f.write("No Exit Usage Data.\n")
        f.write("\n")

        # 3. SPATIAL DISTRIBUTION (Top 5 Cells)
        f.write("3. SPATIAL DISTRIBUTION (Top 5 Casualty/Blockage Zones)\n")
        f.write("-" * 80 + "\n")
        
        for m in model_names:
            f.write(f"--- {m} Top Hazard Zones ---\n")
            if models[m]['spatial'] is not None:
                # Look for columns
                df = models[m]['spatial']
                # Sort by Casualties descending
                cas_col = [c for c in df.columns if 'Casualties' in c][0]
                
                top = df.sort_values(by=cas_col, ascending=False).head(5)
                f.write(f"{'Cell ID':<10} | {'Casualties':<15}\n")
                for _, row in top.iterrows():
                    val = row[cas_col]
                    if val > 0:
                        f.write(f"{int(row['Cell ID']):<10} | {val:<15.1f}\n")
                    else:
                        break # Stop if 0
            else:
                f.write("No spatial data.\n")
            f.write("\n")
            
        print(f"Comparison logic updated. Generated output in: {output_file}")
