"""
MCA Evacuation Simulation (Pure Dijkstra Version) (Batch/Iteration Mode)
Based on "Simulation method of urban evacuation based on mesoscopic cellular automata"
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from shapely.geometry import Point, LineString
import warnings
import argparse
import tkinter as tk
from tkinter import simpledialog, messagebox
import pandas as pd # Ensure pandas is available for export
# Custom Solver
import Dijkstra

warnings.filterwarnings('ignore')

class MCASimulation:
    def __init__(self, gpkg_path):
        self.gpkg_path = gpkg_path
        self.road_cells = None
        self.blocked_cells = [] # Track IDs for visualization
        self.exits = None
        self.spawns = None
        self.graph = None
        self.directions = {} # cell_index -> next_cell_index
        
        # Simulation Parameters (from paper/gpkg)
        self.V_FREE = 1.5      # m/s
        self.RHO_MAX = 5.0     # p/m^2
        self.DT = 1.0          # time step (s)
        self.CELL_WIDTH = 6.0  # m (approx default)
        
        # We will load these from GPKG now
        self.capacities = {}   # cell_id -> SAFE capacity (int) - from GPKG
        self.max_capacities = {} # cell_id -> PHYSICAL capacity (Area * 5)
        self.cell_areas = {}   # cell_id -> area (float)
        
        # Stampede Logic
        self.STAMPEDE_DENSITY = 3.5 # p/m^2 (Lowered from 4.5 to increase sensitivity)
        self.DEATH_RATE = 0.1 # 10% per second if overcrowded
        
        # State
        self.population = {} # cell_index -> count
        self.casualties = 0
        self.casualties_per_cell = {} # cell_id -> total_deaths
        self.time_step = 0
        self.history = []
        self.casualty_history = []
        self.per_cell_casualty_history = [] # LIST of DICTS: [ {cell_id: death_count}, ... ]
        
        # Exit Analysis
        self.exit_status = {} # exit_id -> 'OPEN'/'CLOSED'
        self.exit_usage = {}  # exit_id -> count (accumulated)
        self.exit_usage_history = [] # list of dicts
        self.road_to_exit = {} # road_cell_id -> exit_id
        
        # Viz Data
        self.cell_centroids = {}
        self.flow_vectors = {} # cell_idx -> (u, v)

    def load_data(self):
        print(f"Loading data from {self.gpkg_path}...")
        import fiona
        try:
            layers = fiona.listlayers(self.gpkg_path)
            
            # Road Cells
            layer_name = 'road_cells'
            if layer_name not in layers: layer_name = layers[0]
            
            self.road_cells = gpd.read_file(self.gpkg_path, layer=layer_name)
            
            # Ensure ID
            if 'fid' not in self.road_cells.columns:
                self.road_cells['id'] = self.road_cells.index
            else:
                self.road_cells['id'] = self.road_cells['fid']
                
            # LOAD CAPACITIES & AREA
            print("Loading properties from GPKG columns...")
            for idx, row in self.road_cells.iterrows():
                cell_id = row['id']
                
                # Area first (needed for fallback)
                if 'cell_area' in row:
                    area = float(row['cell_area'])
                    self.cell_areas[cell_id] = area
                else:
                    area = row.geometry.area
                    self.cell_areas[cell_id] = area

                # SAFE Capacity (from GPKG)
                if 'capacity' in row:
                    self.capacities[cell_id] = float(row['capacity'])
                else:
                    self.capacities[cell_id] = area * 2.0 # Default safe density ~2 p/m2
                
                # PHYSICAL Capacity (Area * 5.0) - Allows overcrowding
                self.max_capacities[cell_id] = area * self.RHO_MAX

            print(f"Loaded {len(self.road_cells)} road cells. Max Capacity: {max(self.capacities.values())}")
            
            # Spawns (Building Exits)
            if 'building_exits' in layers:
                self.spawns = gpd.read_file(self.gpkg_path, layer='building_exits')
                print(f"Loaded {len(self.spawns)} spawn points (building exits).")
            
            # Exits (Sinks)
            target_layer = 'university_exits'
            if 'campus_exits' in layers: 
                target_layer = 'campus_exits'
            elif 'University_Exits' in layers: 
                target_layer = 'University_Exits'
            
            if target_layer in layers:
                self.exits = gpd.read_file(self.gpkg_path, layer=target_layer)
            elif 'safe_zones' in layers:
                self.exits = gpd.read_file(self.gpkg_path, layer='safe_zones')
                
            if self.exits is not None:
                print(f"Loaded {len(self.exits)} sink nodes (exits) from layer '{target_layer if target_layer in layers else 'safe_zones'}'.")

        except Exception as e:
            print(f"Error loading layers: {e}")
            return

    
    def analyze_specific_cell(self, target_id=88):
        if target_id not in self.road_cells['id'].values:
            print(f"Cell {target_id} not found.")
            return

        print(f"\n=== FORENSIC ANALYSIS: CELL {target_id} ===")
        # Get row
        row = self.road_cells[self.road_cells['id'] == target_id].iloc[0]
        
        # Geometry
        area = self.cell_areas.get(target_id, 0)
        cap = self.max_capacities.get(target_id, 0)
        safe_cap = self.capacities.get(target_id, 0)
        
        print(f"  Geometry: {row.geometry.geom_type}")
        print(f"  Area: {area:.2f} m2")
        print(f"  Physical Max Load: {cap:.1f} agents (5 p/m2)")
        print(f"  Visual Safe Load: {safe_cap:.1f} agents")
        
        # Connectivity
        if target_id in self.graph:
            neighbors = list(self.graph.neighbors(target_id))
            print(f"  Neighbors: {neighbors}")
            print(f"  Degree: {len(neighbors)}")
        
        # Exit Connectivity
        exit_id = self.road_to_exit.get(target_id)
        if exit_id is not None:
             print(f"  Direct Connection to Exit: {exit_id} (This is an Exit Node!)")
        else:
             print(f"  Distance to Exit: (Calculated during simulation)")
             
        print("========================================\n")

    def build_graph(self):
        print("Building connectivity graph...")
        buffered_cells = self.road_cells.copy()
        buffered_cells.geometry = buffered_cells.geometry.buffer(0.1)
        joins = gpd.sjoin(buffered_cells, buffered_cells, how='inner', predicate='intersects')
        
        self.graph = nx.Graph()
        for idx, row in self.road_cells.iterrows():
            cid = row['id']
            self.graph.add_node(cid, geometry=row.geometry)
            self.population[cid] = 0.0
            self.cell_centroids[cid] = row.geometry.centroid
            # Initialize tracker
            self.casualties_per_cell[cid] = 0.0
            
        for idx_left, row in joins.iterrows():
            id_l = self.road_cells.iloc[idx_left]['id']
            id_r = self.road_cells.iloc[row['index_right']]['id']
            
            if id_l != id_r:
                # Calculate Geometric Distance (Euclidean)
                c_l = self.cell_centroids[id_l]
                c_r = self.cell_centroids[id_r]
                dist = c_l.distance(c_r)
                
                self.graph.add_edge(id_l, id_r, weight=dist)

    def apply_scenario_blockages(self, blocked_ids):
        """
        Removes specified cells from the graph to simulate blockage/fire.
        """
        if not blocked_ids: return

        print(f"\n!!! APPLYING SCENARIO: BLOCKING {len(blocked_ids)} CELLS !!!")
        for bid in blocked_ids:
            if bid in self.graph:
                self.graph.remove_node(bid)
                self.max_capacities[bid] = 0.0
                self.blocked_cells.append(bid)
                print(f"  - BLOCKED Cell {bid} (Removed from Graph)")
            else:
                print(f"  - WARNING: Cell {bid} not found or already removed.")
        print("!!! SCENARIO APPLIED: Agents will re-route around these cells. !!!\n")

    def compute_flow_directions(self):
        print("Computing flow directions (Pure Dijkstra Mode)...")
        # 1. Map Road Cells to Exits
        exit_indices = []
        
        if self.exits is not None:
            # Initialize all exits
            for idx, row in self.exits.iterrows():
                self.exit_status[idx] = 'CLOSED'
                self.exit_usage[idx] = 0.0
                
                # Check specific intersection for this exit
                # Using 5.0m buffer
                exit_geo = row.geometry.buffer(5.0)
                connected_cells = []
                
                for r_idx, cell in self.road_cells.iterrows():
                    if exit_geo.intersects(cell.geometry):
                        cid = cell['id']
                        connected_cells.append(cid)
                        # Map road cell to this Exit ID
                        # Note: If cell connects to multiple, last one wins (acceptable approx)
                        self.road_to_exit[cid] = idx
                
                if connected_cells:
                    self.exit_status[idx] = 'OPEN'
                    exit_indices.extend(connected_cells)
                    print(f"DEBUG: Exit {idx} is OPEN (Connected to {len(connected_cells)} cells)")
                else:
                    self.exit_status[idx] = 'CLOSED'
                    print(f"DEBUG: Exit {idx} is CLOSED")

        if not exit_indices:
            # Fallback
            print("WARNING: No exits connected. Using fallback (West bounds).")
            min_x = self.road_cells.bounds.minx.min()
            ids = self.road_cells[self.road_cells.bounds.minx < min_x + 10]['id'].tolist()
            exit_indices = ids
            # Assign fallback ID
            for i in ids: self.road_to_exit[i] = 999 

        # Store exit_indices for reference
        self.exits_ids = exit_indices

        # ------------------------------------------------------------------------
        # DIJKSTRA SOLVER CALL (Replaces ACO)
        # ------------------------------------------------------------------------
        print(f"Using Dijkstra Solver from {len(exit_indices)} exit nodes...")
        distances, history = Dijkstra.calculate_dijkstra_field(self.graph, exit_indices)
        
        # Derive Flow Directions (Gradient Descent)
        self.directions = {}
        for node in self.graph.nodes:
            if node in exit_indices:
                self.directions[node] = None 
                continue
            
            best_neighbor = None
            min_dist = distances.get(node, float('inf'))
            current_best_dist = min_dist
            
            for neighbor in self.graph.neighbors(node):
                d = distances.get(neighbor, float('inf'))
                if d < current_best_dist:
                    current_best_dist = d
                    best_neighbor = neighbor
            
            self.directions[node] = best_neighbor
        
        # Pre-calc flow vectors
        self.flow_vectors = {}
        for idx in self.graph.nodes:
            target_idx = self.directions.get(idx)
            if target_idx is not None and target_idx in self.cell_centroids:
                start = self.cell_centroids[idx]
                end = self.cell_centroids[target_idx]
                dx = end.x - start.x
                dy = end.y - start.y
                norm = np.hypot(dx, dy)
                if norm > 0:
                    dx /= norm
                    dy /= norm
                self.flow_vectors[idx] = (dx, dy)
            else:
                self.flow_vectors[idx] = (0, 0)
    
    def initialize_population(self, total_agents=5000):
        self.total_agents_init = total_agents
        print(f"Initializing {total_agents} agents...")
        nodes = list(self.graph.nodes)
        
        # Prefer spawning near 'spawns'
        if self.spawns is not None:
            source_ids = []
            # Widen buffer to 5.0 to prevent initial overcrowding
            spawn_buffer = self.spawns.buffer(5.0)
            for idx, cell in self.road_cells.iterrows():
                if spawn_buffer.intersects(cell.geometry).any():
                    source_ids.append(cell['id'])
            
            if source_ids:
                # Randomize distribution
                weights = np.random.random(len(source_ids)) 
                weights /= weights.sum()
                
                counts = (weights * total_agents).astype(int)
                diff = total_agents - counts.sum()
                for i in range(diff):
                    counts[i] += 1
                
                for i, cid in enumerate(source_ids):
                    self.population[cid] += counts[i]
            else:
                 for _ in range(total_agents):
                    self.population[np.random.choice(nodes)] += 1
        else:
            for _ in range(total_agents):
                self.population[np.random.choice(nodes)] += 1
        
        self.history = [] # RESET
        self.history.append(self.population.copy())
        
        self.casualty_history = [] # RESET
        self.casualty_history.append(0)
        
        self.exit_usage_history = [] # RESET
        self.exit_usage_history.append(self.exit_usage.copy())
        
        self.per_cell_casualty_history = [] # RESET
        self.per_cell_casualty_history.append(self.casualties_per_cell.copy())

    def step(self):
        new_population = self.population.copy()
        total_deaths_this_step = 0
        
        # 1. Check for Stampedes (Apply Grace Period of 10 steps)
        if self.time_step > 10:
            for cid, count in self.population.items():
                area = self.cell_areas.get(cid, 60.0)
                rho = count / area if area > 0 else 0
                
                if rho > self.STAMPEDE_DENSITY:
                    deaths = count * self.DEATH_RATE
                    new_population[cid] -= deaths
                    total_deaths_this_step += deaths
                    # Track location
                    self.casualties_per_cell[cid] = self.casualties_per_cell.get(cid, 0) + deaths
        
        self.casualties += total_deaths_this_step

        # 2. Movement
        for cid in self.graph.nodes:
            current_pop = new_population[cid]
            if current_pop <= 0: continue
                
            target_id = self.directions.get(cid)
            if target_id is None:
                # Sink or stuck
                if cid in self.directions: 
                     # Absorb flow at sink
                     area = self.cell_areas.get(cid, 60.0)
                     width = self.CELL_WIDTH # Approximate width if not in GPKG
                     
                     flow_out = (self.V_FREE * width * self.DT * self.RHO_MAX)
                     actual_out = min(new_population[cid], flow_out)
                     
                     new_population[cid] = max(0, new_population[cid] - actual_out)
                     
                     # Track Exit Usage
                     exit_id = self.road_to_exit.get(cid)
                     if exit_id is not None:
                         self.exit_usage[exit_id] += actual_out
                continue
            
            # Flow Calculation
            area = self.cell_areas.get(cid, 60.0)
            # Use MAX/PHYSICAL capacity for flow limit allows overcrowding
            cap = self.max_capacities.get(cid, 300.0)
            
            rho_i = current_pop / area if area > 0 else 0
            v_i = self.V_FREE * np.exp(-rho_i / self.RHO_MAX)
            
            # Q = rho * v * w * dt
            # Use fixed width approx or derive from area/len
            q_out = rho_i * v_i * self.CELL_WIDTH * self.DT
            
            # Target Limit: Use PHYSICAL capacity here too
            target_cap = self.max_capacities.get(target_id, 300.0)
            pop_target = new_population[target_id]
            available_capacity = target_cap - pop_target
            
            actual_flow = min(q_out, available_capacity)
            actual_flow = max(0, actual_flow)
            actual_flow = min(actual_flow, current_pop)
            
            new_population[cid] -= actual_flow
            new_population[target_id] += actual_flow
            
        self.population = new_population
        self.time_step += 1
        return sum(self.population.values())

    def export_to_excel(self, filename="simulation_results_dijkstra_2.xlsx"):
        print(f"Exporting results to {filename}...")
        
        # 1. Time Series Data (Step-by-Step)
        time_data = []
        
        # Pre-calculate cell parameters for speed calc
        cell_params = {}
        for idx, row in self.road_cells.iterrows():
            cid = row['id']
            area = self.cell_areas.get(cid, 60.0)
            cell_params[cid] = area

        for t, (pop_map, cas_count) in enumerate(zip(self.history, self.casualty_history)):
            # Global Stats
            total_alive = sum(pop_map.values())
            
            # Avg Density & Speed Estimate
            total_weighted_speed = 0
            total_density = 0
            occupied_cells = 0
            
            max_rho = 0
            
            for cid, count in pop_map.items():
                if count > 0:
                    area = cell_params.get(cid, 60.0)
                    rho = count / area
                    
                    # Fundamental Diagram (Greenshields)
                    # v = v_free * (1 - rho/rho_max)
                    # Clamped at 0
                    v_ratio = max(0.0, 1.0 - (rho / self.RHO_MAX))
                    speed = self.V_FREE * v_ratio
                    
                    total_weighted_speed += speed * count
                    total_density += rho
                    occupied_cells += 1
                    
                    if rho > max_rho: max_rho = rho
            
            avg_speed = (total_weighted_speed / total_alive) if total_alive > 0 else 0
            avg_density = (total_density / occupied_cells) if occupied_cells > 0 else 0
            
            # Exit Flow Rate (Agents per step)
            # Flow = Current Usage - Previous Usage
            current_usage = self.exit_usage_history[t] if t < len(self.exit_usage_history) else self.exit_usage
            if t > 0:
                prev_usage = self.exit_usage_history[t-1]
                flow_step = sum(current_usage.values()) - sum(prev_usage.values())
            else:
                flow_step = 0
                
            time_data.append({
                'Time (s)': t * self.DT,
                'Exit Flow Rate': int(flow_step), # Round to int
                'Remaining Evacuees': int(total_alive), # Round to int
                'Density Evolution': avg_density,
                'Velocity of Evacuees': avg_speed,
                # Extra internal metrics kept for utility but renamed if needed
                'Peak Density': max_rho, 
                'Cumulative Casualties': int(cas_count) # Round to int
            })
            
        df_time = pd.DataFrame(time_data)
        
        # 2. Spatial Distribution (Cell Analysis)
        cell_data = []
        for cid in self.road_cells['id']:
            final_deaths = self.casualties_per_cell.get(cid, 0)
            area = self.cell_areas.get(cid, 60.0)
            cap = self.max_capacities.get(cid, 0)
            
            cell_data.append({
                'Cell ID': cid,
                'Spatial Distribution (Casualties)': int(final_deaths), # Round to int
                'Area': area,
                'Status': 'BLOCKED' if cid in self.blocked_cells else 'OPEN'
            })
        df_cells = pd.DataFrame(cell_data).sort_values(by='Spatial Distribution (Casualties)', ascending=False)
        
        # 3. Exit Usage Distribution
        exit_data = []
        for eid, count in self.exit_usage.items():
            status = self.exit_status.get(eid, 'UNKNOWN')
            exit_data.append({
                'Exit ID': eid,
                'Exit Usage Distribution': int(count), # Round to int
                'Status': status
            })
        df_exits = pd.DataFrame(exit_data)
        
        # 4. Total Evacuation Time (Summary)
        # Find first step where population is 0, or use max time
        final_time = (len(self.history) - 1) * self.DT
        if self.history[-1] and sum(self.history[-1].values()) == 0:
             # Find exact step it became 0
             for t, pop in enumerate(self.history):
                 if sum(pop.values()) == 0:
                     final_time = t * self.DT
                     break
        
        df_summary = pd.DataFrame([{
            'Total Evacuation Time': final_time,
            'Total Casualties': int(self.casualties) # Round to int
        }])

        # WRITE TO EXCEL
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                df_time.to_excel(writer, sheet_name='Time Metrics', index=False)
                df_cells.to_excel(writer, sheet_name='Spatial Distribution', index=False)
                df_exits.to_excel(writer, sheet_name='Exit Usage', index=False)
                df_summary.to_excel(writer, sheet_name='Summary', index=False)
            print(f"✅ Simulation results saved to {filename}")
        except Exception as e:
            print(f"❌ Failed to save Excel: {e}")

    def run(self, steps=100):
        print(f"Starting simulation for {steps} steps...")
        # Note: History reset is handled in initialize_population now, but let's be safe
        # If user calls run() directly without re-init?
        # But run() relies on init.
        # Let's trust initialize_population to do the reset before run.
        
        for t in range(steps):
            total = self.step()
            self.history.append(self.population.copy())
            self.casualty_history.append(self.casualties)
            self.per_cell_casualty_history.append(self.casualties_per_cell.copy())
            self.exit_usage_history.append(self.exit_usage.copy())
            
            if t % 10 == 0:
                print(f"Step {t}: Agents: {total:.0f} | Dead: {self.casualties:.0f}")
            if total < 1:
                break
        
        # FINAL REPORT
        print("\n" + "="*40)
        print(f"=== CASUALTY REPORT (Total: {self.casualties:.1f}) ===")
        
        # 1. Full Breakdown
        print("Breakdown by Cell:")
        sorted_cells = sorted(self.casualties_per_cell.items(), key=lambda x: x[1], reverse=True)
        count = 0
        others_count = 0
        others_sum = 0
        
        for cid, deaths in sorted_cells:
            if deaths > 0:
                # Analyze Node Properties
                area = self.cell_areas.get(cid, 0)
                cap = self.max_capacities.get(cid, 0)
                degree = len(list(self.graph.neighbors(cid))) if cid in self.graph else 0
                
                print(f"  Cell {cid:>4}: {deaths:>6.1f} deaths | Area: {area:>5.1f}m2 | MaxCap: {cap:>5.1f} | Neighbors: {degree}")
                count += 1
        
        if count == 0:
            print("  No casualties reported.")
        print("="*40 + "\n")
        
        # Don't auto-export in loop. Main will handle it.
        # self.export_to_excel()

    def generate_results_dataframes(self):
        # 1. Time Series Data (Step-by-Step)
        time_data = []
        
        # Pre-calculate cell parameters for speed calc
        cell_params = {}
        for idx, row in self.road_cells.iterrows():
            cid = row['id']
            area = self.cell_areas.get(cid, 60.0)
            cell_params[cid] = area

        for t, (pop_map, cas_count) in enumerate(zip(self.history, self.casualty_history)):
            # Global Stats
            total_alive = sum(pop_map.values())
            
            # Avg Density & Speed Estimate
            total_weighted_speed = 0
            total_density = 0
            occupied_cells = 0
            
            max_rho = 0
            
            for cid, count in pop_map.items():
                if count > 0:
                    area = cell_params.get(cid, 60.0)
                    rho = count / area
                    
                    v_ratio = max(0.0, 1.0 - (rho / self.RHO_MAX))
                    speed = self.V_FREE * v_ratio
                    
                    total_weighted_speed += speed * count
                    total_density += rho
                    occupied_cells += 1
                    
                    if rho > max_rho: max_rho = rho
            
            avg_speed = (total_weighted_speed / total_alive) if total_alive > 0 else 0
            avg_density = (total_density / occupied_cells) if occupied_cells > 0 else 0
            
            # Exit Flow Rate (Agents per step)
            current_usage = self.exit_usage_history[t] if t < len(self.exit_usage_history) else self.exit_usage
            if t > 0:
                prev_usage = self.exit_usage_history[t-1]
                flow_step = sum(current_usage.values()) - sum(prev_usage.values())
            else:
                flow_step = 0
                
            time_data.append({
                'Time (s)': t * self.DT,
                'Exit Flow Rate': int(flow_step),
                'Remaining Evacuees': int(total_alive),
                'Density Evolution': avg_density,
                'Velocity of Evacuees': avg_speed,
                'Peak Density': max_rho, 
                'Cumulative Casualties': int(cas_count)
            })
            
        df_time = pd.DataFrame(time_data)
        
        # 2. Spatial Distribution
        cell_data = []
        for cid in self.road_cells['id']:
            final_deaths = self.casualties_per_cell.get(cid, 0)
            area = self.cell_areas.get(cid, 60.0)
            cap = self.max_capacities.get(cid, 0)
            
            cell_data.append({
                'Cell ID': cid,
                'Spatial Distribution (Casualties)': int(final_deaths),
                'Area': area,
                'Status': 'BLOCKED' if cid in self.blocked_cells else 'OPEN'
            })
        df_cells = pd.DataFrame(cell_data).sort_values(by='Spatial Distribution (Casualties)', ascending=False)
        
        # 3. Exit Usage
        exit_data = []
        for eid, count in self.exit_usage.items():
            status = self.exit_status.get(eid, 'UNKNOWN')
            exit_data.append({
                'Exit ID': eid,
                'Exit Usage Distribution': int(count),
                'Status': status
            })
        df_exits = pd.DataFrame(exit_data)
        
        # 4. Summary
        final_time = (len(self.history) - 1) * self.DT
        if self.history[-1] and sum(self.history[-1].values()) == 0:
             for t, pop in enumerate(self.history):
                 if sum(pop.values()) == 0:
                     final_time = t * self.DT
                     break
        
        df_summary = pd.DataFrame([{
            'Total Evacuation Time': final_time,
            'Total Casualties': int(self.casualties)
        }])
        
        return df_time, df_cells, df_exits, df_summary
                
    def animate_results(self):
        print("Preparing animation (Dummy for Headless)...")
        # Headless mode doesn't animate, but we keep the method for class compatibility if needed
        pass

def get_config_from_terminal():
    print("\n=== MCA Simulation (Iterative Batch Run) ===")
    
    # Defaults
    d_agents = 5000
    d_steps = 300
    d_iters = 5
    
    try:
        a_str = input(f"Total Agents [{d_agents}]: ").strip()
        agents = int(a_str) if a_str else d_agents
        
        s_str = input(f"Duration (Steps) [{d_steps}]: ").strip()
        steps = int(s_str) if s_str else d_steps
        
        i_str = input(f"Iterations [{d_iters}]: ").strip()
        iterations = int(i_str) if i_str else d_iters
        
        b_str = input("Blocked Cells (IDs, space separated) []: ").strip()
        block = [int(x) for x in b_str.split()] if b_str else []
        
        return {'agents': agents, 'steps': steps, 'iterations': iterations, 'block': block}
        
    except ValueError:
        print("Invalid input. Using defaults.")
        return {'agents': d_agents, 'steps': d_steps, 'iterations': d_iters, 'block': []}

def main():
    config = get_config_from_terminal()
    print(f"\nStarting Batch Run with config: {config}")
    
    sim_iterations = config['iterations']
    
    # Store per-run summaries
    run_summaries = []
    
    # Store bulky data for "Average Curve" calculation
    all_time_metrics = []
    all_spatial_metrics = []
    all_exit_metrics = []
    
    # DATA PATH
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    gpkg_path = os.path.join(base_dir, "..", "GPKG_Files", "cmu-map.gpkg")
    
    for i in range(sim_iterations):
        print(f"\n{'='*20}\nRUN {i+1}/{sim_iterations}\n{'='*20}")
        
        sim = MCASimulation(gpkg_path)
        sim.load_data()
        sim.build_graph()
        sim.apply_scenario_blockages(config['block'])
        sim.compute_flow_directions()
        sim.initialize_population(total_agents=config['agents'])
        
        # Run Headless
        sim.run(steps=config['steps'])
        
        # Capture Data
        df_t, df_s, df_e, df_sum = sim.generate_results_dataframes()
        
        # 1. Compute Scalar Metrics for this Run
        # Means over time (excluding time=0 if needed, but mean including 0 is fine for "session average")
        mean_flow = df_t['Exit Flow Rate'].mean()
        mean_density = df_t['Density Evolution'].mean()
        mean_velocity = df_t['Velocity of Evacuees'].mean()
        peak_density = df_t['Peak Density'].max()
        
        # End-of-Run Status
        remaining_agents = df_t['Remaining Evacuees'].iloc[-1]
        total_evacuated = df_e['Exit Usage Distribution'].sum()
        
        total_time = df_sum['Total Evacuation Time'].iloc[0]
        total_casualties = df_sum['Total Casualties'].iloc[0]
        
        # Append to summary list
        run_summaries.append({
            'Run': i + 1,
            'Avg Exit Flow Rate': float(f"{mean_flow:.2f}"),
            'Avg Density (p/m2)': float(f"{mean_density:.2f}"),
            'Avg Velocity (m/s)': float(f"{mean_velocity:.2f}"),
            'Peak Density (p/m2)': float(f"{peak_density:.2f}"),
            'Total Casualties': int(total_casualties),
            'Total Evacuated': int(total_evacuated),
            'Remaining Agents': int(remaining_agents),
            'Total Evacuation Time (s)': float(total_time)
        })
        
        # Store for global averaging of the curves
        df_t['Run'] = i + 1
        all_time_metrics.append(df_t)
        
        # Store for global averaging of Spatial/Exit
        all_spatial_metrics.append(df_s)
        all_exit_metrics.append(df_e)
        
        
    print(f"\n{'='*40}")
    print("COMPUTING AGGREGATE STATS...")
    print(f"{'='*40}")

    # 1. Run Comparison Table
    df_runs = pd.DataFrame(run_summaries)
    
    # Calculate Average of Runs
    avg_row = df_runs.mean(numeric_only=True)
    avg_row['Run'] = 'AVERAGE' 
    # Cast integers for logic
    avg_row['Total Casualties'] = int(avg_row['Total Casualties'])
    avg_row['Total Evacuated'] = int(avg_row['Total Evacuated'])
    avg_row['Remaining Agents'] = int(avg_row['Remaining Agents'])
    
    # Append Average Row safely
    df_runs_final = pd.concat([df_runs, pd.DataFrame([avg_row])], ignore_index=True)

    # USER REQUEST: Add column titles below the average row for readability
    header_row = {col: col for col in df_runs.columns}
    df_runs_final = pd.concat([df_runs_final, pd.DataFrame([header_row])], ignore_index=True)
    
    # 2. Average Time Curve (Optional but requested "average for each metric")
    # This gives the "Average Flow vs Time" graph data
    full_time = pd.concat(all_time_metrics)
    avg_time_curve = full_time.groupby('Time (s)').mean(numeric_only=True).reset_index()
    avg_time_curve['Run'] = 'AVERAGE_CURVE'
    
    # 3. Average Spatial Distribution
    full_spatial = pd.concat(all_spatial_metrics)
    # Group by Cell ID and Average the numeric columns
    avg_spatial = full_spatial.groupby('Cell ID').mean(numeric_only=True).reset_index()
    avg_spatial = avg_spatial.sort_values(by='Spatial Distribution (Casualties)', ascending=False)
    
    # 4. Average Exit Usage
    full_exits = pd.concat(all_exit_metrics)
    avg_exits = full_exits.groupby('Exit ID').mean(numeric_only=True).reset_index()
    
    # SAVE
    # User Request: Save to same folder as script
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, "simulation_results_dijkstra_2.xlsx")
    
    try:
        # STYLING
        def color_columns(s):
            # s is a Series (column)
            # Define colors
            c_gray = 'background-color: #e0e0e0'
            c_red = 'background-color: #ffcccb'
            c_green = 'background-color: #d0f0c0'
            c_blue = 'background-color: #add8e6'
            c_yellow = 'background-color: #ffffe0'
            
            if s.name == 'Run':
                return [c_gray] * len(s)
            elif 'Casualties' in s.name or 'Remaining' in s.name:
                return [c_red] * len(s)
            elif 'Evacuated' in s.name or 'Flow' in s.name:
                return [c_green] * len(s) # Positive flow/evac
            elif 'Density' in s.name or 'Velocity' in s.name:
                return [c_blue] * len(s) # Physics
            elif 'Time' in s.name:
                return [c_yellow] * len(s)
            return [''] * len(s)

        # Apply style
        try:
            styled_df = df_runs_final.style.apply(color_columns, axis=0)
            
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Sheet 1: The Iteration Table (User's primary request) - STYLED
                styled_df.to_excel(writer, sheet_name='Run Comparisons', index=False)
                # Sheet 2: The Average Curve
                avg_time_curve.to_excel(writer, sheet_name='Avg Time Series', index=False)
                # Sheet 3: Avg Spatial
                avg_spatial.to_excel(writer, sheet_name='Avg Spatial Dist', index=False)
                # Sheet 4: Avg Exit Usage
                avg_exits.to_excel(writer, sheet_name='Avg Exit Usage', index=False)
                
            print(f"✅ AVERAGED Results saved to {filename}")
            print("Report contains 'Run Comparisons' (Colored Columns) and 'Avg Time Series'.")
            
        except ImportError as e:
            # Fallback if jinja2 missing
            print(f"⚠️ Styling failed (missing dependency): {e}. Saving unstyled version...")
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                df_runs_final.to_excel(writer, sheet_name='Run Comparisons', index=False)
                avg_time_curve.to_excel(writer, sheet_name='Avg Time Series', index=False)
            print(f"✅ AVERAGED Results saved to {filename} (Unstyled)")

        except Exception as e:
            print(f"⚠️ Styling/Saving failed: {e}. Attempting unstyled save...")
            # Emergency Fallback
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                df_runs_final.to_excel(writer, sheet_name='Run Comparisons', index=False)
                avg_time_curve.to_excel(writer, sheet_name='Avg Time Series', index=False)
            print(f"✅ AVERAGED Results saved to {filename} (Unstyled)")

    except Exception as e:
        print(f"❌ Failed to save Excel: {e}")
        try:
            df_runs_final.to_csv("simulation_results_backup.csv", index=False)
            print("⚠️ Saved backup to simulation_results_backup.csv")
        except:
            pass

    print("Batch Run Complete. (Pure Headless Mode)")
    # Keep console open
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()
