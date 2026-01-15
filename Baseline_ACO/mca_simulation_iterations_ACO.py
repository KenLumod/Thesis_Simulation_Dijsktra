"""
MCA Evacuation Simulation
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
        self.STAMPEDE_DENSITY = 4.0 # p/m^2 (Lowered from 4.5 to increase sensitivity)
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
        # ... (unchanged code for load_data) ...
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
            
            # Incoming Flow Potential?
            # We can't know flow until run, but we can see degree
            print(f"  Degree: {len(neighbors)}")
        
        # Exit Connectivity
        exit_id = self.road_to_exit.get(target_id)
        if exit_id is not None:
             print(f"  Direct Connection to Exit: {exit_id} (This is an Exit Node!)")
        else:
             print(f"  Distance to Exit: (Calculated during simulation)")
             
        print("========================================\n")

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
            
            # Incoming Flow Potential?
            # We can't know flow until run, but we can see degree
            print(f"  Degree: {len(neighbors)}")
        
        # Exit Connectivity
        exit_id = self.road_to_exit.get(target_id)
        if exit_id is not None:
             print(f"  Direct Connection to Exit: {exit_id} (This is an Exit Node!)")
        else:
             print(f"  Distance to Exit: (Calculated during simulation)")
             
        print("========================================\n")

    def build_graph(self):
        # ... (unchanged code for build_graph) ...
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
                self.graph.add_edge(id_l, id_r)
        
            if id_l != id_r:
                self.graph.add_edge(id_l, id_r)

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
        print("Computing flow directions...")
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

        if self.exits is not None:
             # ... (Exit Mapping Removed for Brevity as it was above)
             pass 

        # Store exit_indices as self.exits_ids for ACO logic
        self.exits_ids = exit_indices

        print("Computing flow directions (ACO Mode)...")
        try:
             from aco_logic import run_aco_pathfinding
        except ImportError:
             from Baseline_ACO.aco_logic import run_aco_pathfinding
        
        # Filter Exits that are actually in graph
        valid_exits = [e for e in self.exits_ids if e in self.graph]
        if not valid_exits:
             print("❌ No reachable exits for ACO!")
             return

        # Run ACO shared logic
        self.directions, self.pheromones = run_aco_pathfinding(
            self.graph, valid_exits, self.road_cells, max_iterations=2000, n_ants=200
        )
        
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

    def export_to_excel(self, filename="simulation_results_aco_2.xlsx"):
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
        print("Preparing animation...")
        from matplotlib.widgets import Slider, Button
        
        # Adjust layout to make room for slider and buttons
        fig = plt.figure(figsize=(14, 10))
        
        # 1. Main Map Axes (Centered Manually)
        # [left, bottom, width, height]
        # We give it a centered box.
        ax = fig.add_axes([0.1, 0.2, 0.8, 0.75])
        
        # 2. Colorbar Axes (Manual placement on the right)
        cax = fig.add_axes([0.92, 0.25, 0.02, 0.6])
        
        # Markers
        if self.spawns is not None:
             self.spawns.plot(ax=ax, color='blue', marker='o', markersize=30, label='Spawn Points', zorder=5)
        if self.exits is not None:
             self.exits.plot(ax=ax, color='lime', marker='X', markersize=150, edgecolor='black', linewidth=2, label='Exits', zorder=10)
        
        # SCENARIO VISUALIZATION: Blocked Cells
        if self.blocked_cells:
            blocked_geom = self.road_cells[self.road_cells['id'].isin(self.blocked_cells)]
            if not blocked_geom.empty:
                blocked_geom.plot(ax=ax, color='black', alpha=0.7, zorder=6, label='Blocked/Destroyed')
                # Add cross hatch
                blocked_geom.plot(ax=ax, color='none', edgecolor='red', hatch='XX', zorder=7)
             
        ax.legend(loc='upper right')
        ax.set_title("USTP Evacuation (Per-Cell Capacity Analysis)", fontsize=14, fontweight='bold')
        ax.axis('off')
        ax.set_aspect('equal', anchor='C') # Force content to center of the box

        info_text = ax.text(0.02, 0.9, "", transform=ax.transAxes, fontsize=11,
                           bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'))
        
        selected_cell_text = ax.text(0.02, 0.82, "Selection: None", transform=ax.transAxes, fontsize=10,
                                     verticalalignment='top', bbox=dict(facecolor='ivory', alpha=0.9, boxstyle='round,pad=0.5'))

        # Heatmap (Variable: Occupancy OR Casualties)
        # Default: Occupancy (Turbo)
        self.road_cells.plot(ax=ax, color='white', edgecolor='lightgray', linewidth=0.5)
        self.road_cells.plot(ax=ax, column='id', cmap='turbo', alpha=0.8, vmin=0, vmax=1.0)
        collection = ax.collections[-1]
        
        sm = plt.cm.ScalarMappable(cmap='turbo', norm=plt.Normalize(vmin=0, vmax=1.0))
        sm._A = []
        # Use simple colorbar on dedicated axes
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label('Occupancy Ratio (Population / Capacity)')

        self.quiver = None
        self.selected_cell_id = None
        self.selected_exit_id = None
        self.current_frame = 0
        self.view_mode = 'occupancy' # 'occupancy' or 'casualties'

        # Helper to update selection text
        def update_selection_text(frame_idx):
            if self.selected_exit_id is not None:
                # Show Exit Info
                eid = self.selected_exit_id
                status = self.exit_status.get(eid, 'UNKNOWN')
                
                # Get usage from history if available (for precise playback)
                # or current state if live
                if frame_idx < len(self.exit_usage_history):
                    usage_dict = self.exit_usage_history[frame_idx]
                    count = usage_dict.get(eid, 0)
                else:
                    count = self.exit_usage.get(eid, 0)

                bg_color = 'lime' if status == 'OPEN' else 'salmon'
                
                selected_cell_text.set_text(
                    f"EXIT NODE\n"
                    f"ID: {eid}\n"
                    f"Status: {status}\n"
                    f"Evacuated: {int(count)}"
                )
                selected_cell_text.set_backgroundcolor(bg_color)
                return

            if self.selected_cell_id is None:
                selected_cell_text.set_text("Selection: None")
                selected_cell_text.set_backgroundcolor('ivory')
                return
            
            # Reset color for normal cells
            selected_cell_text.set_backgroundcolor('ivory')
            cid = self.selected_cell_id
            if self.view_mode == 'occupancy':
                pop_data = self.history[frame_idx]
                p = pop_data.get(cid, 0)
                # ... (cap logic) ... same as before
                c = self.capacities.get(cid, 1)
                
                ratio = p / c if c > 0 else 0
                status = "OK"
                if ratio > 0.5: status = "BUSY"
                if ratio >= 1.0: status = "OVERCROWDED"
                
                selected_cell_text.set_text(
                    f"Cell ID: {cid}\n"
                    f"Pop: {p:.0f} / Cap: {c:.0f}\n"
                    f"Occupancy: {ratio*100:.1f}%\n"
                    f"Status: {status}"
                )
            else:
                # Casualty Mode
                # Use history if available
                if frame_idx < len(self.per_cell_casualty_history):
                    d = self.per_cell_casualty_history[frame_idx].get(cid, 0)
                else:
                    d = self.casualties_per_cell.get(cid, 0)
                    
                selected_cell_text.set_text(
                    f"Cell ID: {cid}\n"
                    f"Total Casualties: {d:.1f}\n"
                    f"(Historical Accumulation)"
                )

        def update(frame):
            self.current_frame = frame
            pop_data = self.history[frame]
            casualties = self.casualty_history[frame]
            
            # 1. Update Map based on Mode
            if self.view_mode == 'occupancy':
                ratios = []
                for idx in self.road_cells['id']:
                     p = pop_data.get(idx, 0)
                     c = self.capacities.get(idx, 1.0)
                     r = p / c if c > 0 else 0
                     ratios.append(r)
                collection.set_array(np.array(ratios))
                collection.set_cmap('turbo')
                collection.set_clim(0, 1.0)
                
                # Quiver (Show in occupancy mode)
                if self.quiver: self.quiver.remove()
                xq, yq, uq, vq = [], [], [], []
                for idx, count in pop_data.items():
                    if count > 1.0:
                         centroid = self.cell_centroids.get(idx)
                         vec = self.flow_vectors.get(idx)
                         if centroid and vec and vec != (0,0):
                             xq.append(centroid.x)
                             yq.append(centroid.y)
                             uq.append(vec[0])
                             vq.append(vec[1])
                
                if xq:
                    self.quiver = ax.quiver(xq, yq, uq, vq, scale=30, width=0.003, color='black', alpha=0.6, zorder=6)
                else:
                    self.quiver = None
                    
            else:
                # Casualty Mode (Dynamic Map of deaths up to this frame)
                if frame < len(self.per_cell_casualty_history):
                    current_deaths = self.per_cell_casualty_history[frame]
                else:
                    current_deaths = self.casualties_per_cell
                    
                deaths = [current_deaths.get(idx, 0) for idx in self.road_cells['id']]
                collection.set_array(np.array(deaths))
                collection.set_cmap('Reds')
                # USER REQUEST: Any death = RED
                # Clamp at 1.0. So 1 death = 1.0 (Max Red).
                collection.set_clim(0, 1) 
                
                # Hide Quiver
                if self.quiver: 
                    self.quiver.remove()
                    self.quiver = None

            # 3. Counters
            current_agents = sum(pop_data.values())
            evacuated_count = self.total_agents_init - int(current_agents) - int(casualties)
            info_text.set_text(
                f"⏱️ Time: {frame}s\n"
                f"👥 Alive: {int(current_agents)}\n"
                f"💀 Casualties: {int(casualties)}\n"
                f"✅ Evacuated: {evacuated_count}"
            )
            
            # 4. Selection
            update_selection_text(frame)
            
            return collection, info_text, selected_cell_text

        # --- Controls ---
        
        # --- Controls Layout (Centered) ---
        
        # 1. Slider (Top Row)
        # Position: Left=0.25, Bottom=0.1, Width=0.5, Height=0.03
        ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03], facecolor='lightgoldenrodyellow')
        self.speed_slider = Slider(ax_slider, 'Speed', 10, 1000, valinit=100, valstep=10)
        
        def update_speed(val):
            self.anim.event_source.interval = val
        self.speed_slider.on_changed(update_speed)
        
        # 2. Buttons (Bottom Row)
        # Pause Button (Left Side)
        ax_pause = plt.axes([0.35, 0.04, 0.1, 0.04])
        self.btn_pause = Button(ax_pause, 'Pause', color='lightgray', hovercolor='gray')
        
        # View Toggle Button (Right Side)
        ax_btn = plt.axes([0.55, 0.04, 0.15, 0.04])
        self.btn_casualty = Button(ax_btn, 'Show Casualties', color='salmon', hovercolor='red')
        
        def toggle_view(event):
            if self.view_mode == 'occupancy':
                self.view_mode = 'casualties'
                self.btn_casualty.label.set_text('Show Live Flow')
                self.btn_casualty.color = 'lightblue'
                self.btn_casualty.hovercolor = 'blue'
                ax.set_title("Total Casualties Heatmap (Any Red = >0 Deaths)", fontsize=14, fontweight='bold')
                cbar.set_label('Total Casualties (Threshold = 1)')
            else:
                self.view_mode = 'occupancy'
                self.btn_casualty.label.set_text('Show Casualties')
                self.btn_casualty.color = 'salmon'
                self.btn_casualty.hovercolor = 'red'
                ax.set_title("USTP Evacuation (Per-Cell Capacity Analysis)", fontsize=14, fontweight='bold')
                cbar.set_label('Occupancy Ratio (Population / Capacity)')
            
            # Force re-draw
            update(self.current_frame)
            fig.canvas.draw_idle()
            
        self.btn_casualty.on_clicked(toggle_view)

        def toggle_pause(event):
            if self.anim.event_source:
                if self.is_paused:
                    self.anim.event_source.start()
                    self.btn_pause.label.set_text('Pause')
                    self.is_paused = False
                else:
                    self.anim.event_source.stop()
                    self.btn_pause.label.set_text('Play')
                    self.is_paused = True
        
        self.is_paused = False
        self.btn_pause.on_clicked(toggle_pause)

        # Zoom
        def on_scroll(event):
            if event.inaxes != ax: return
            base_scale = 1.1
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()
            # ... existing zoom logic ...
            xdata = event.xdata
            ydata = event.ydata
            
            if event.button == 'up':
                scale_factor = 1/base_scale
            elif event.button == 'down':
                scale_factor = base_scale
            else:
                scale_factor = 1
                
            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
            
            rel_x = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
            rel_y = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])
            
            ax.set_xlim([xdata - new_width * (1-rel_x), xdata + new_width * (rel_x)])
            ax.set_ylim([ydata - new_height * (1-rel_y), ydata + new_height * (rel_y)])
            fig.canvas.draw_idle()

        # Pan (Middle Click)
        self.pan_start = None
        
        def on_press(event):
            if event.button == 2: # Middle click
                self.pan_start = (event.xdata, event.ydata)
            
            # Left click is for inspection (handled by on_click currently, but we can merge or keep separate)
            # If we separate press/release, we should verify 'on_click' logic
            if event.button == 1 and event.inaxes == ax:
                 on_click_inspect(event)

        def on_release(event):
            if event.button == 2:
                self.pan_start = None

        def on_motion(event):
            if self.pan_start is None or event.inaxes != ax: return
            
            dx = event.xdata - self.pan_start[0]
            dy = event.ydata - self.pan_start[1]
            
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()
            
            ax.set_xlim(cur_xlim[0] - dx, cur_xlim[1] - dx)
            ax.set_ylim(cur_ylim[0] - dy, cur_ylim[1] - dy)
            fig.canvas.draw_idle()

        # Rename original on_click to on_click_inspect to distinguish
        def on_click_inspect(event):
            if event.inaxes != ax: return
            click_pt = Point(event.xdata, event.ydata)
            
            # 1. Check Exits First
            nearest_exit = None
            min_exit_dist = float('inf')
            
            if self.exits is not None:
                for idx, row in self.exits.iterrows():
                    # Check centroid distance or geometry contains
                    # Geometry might be point or polygon
                    dist = row.geometry.distance(click_pt)
                    if dist < min_exit_dist:
                        min_exit_dist = dist
                        nearest_exit = idx
            
            if self.exits is not None and min_exit_dist < 5.0:
                 self.selected_exit_id = nearest_exit
                 self.selected_cell_id = None
                 print(f"Selected Exit: {nearest_exit}")
                 update_selection_text(self.current_frame)
                 fig.canvas.draw_idle()
                 return

            # 2. Check Road Cells
            min_dist = float('inf')
            nearest = None
            for idx, pt in self.cell_centroids.items():
                d = pt.distance(click_pt)
                if d < min_dist:
                    min_dist = d
                    nearest = idx
            
            if min_dist < 20.0:
                self.selected_cell_id = nearest
                self.selected_exit_id = None
                print(f"Selected Cell: {nearest}")
                # Force update text immediately using current frame
                update_selection_text(self.current_frame)
                fig.canvas.draw_idle()

        # Connect events
        fig.canvas.mpl_connect('button_press_event', on_press)
        fig.canvas.mpl_connect('button_release_event', on_release)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        fig.canvas.mpl_connect('scroll_event', on_scroll)
        
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
    gpkg_path = os.path.join(base_dir, "..", "GPKG_Files", "road_cells_split.gpkg")
    
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
    filename = os.path.join(script_dir, "simulation_results_aco_2.xlsx")
    
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
        # Even the fallback failed or something else is wrong
        # Try to dump CSV at least?
        try:
            df_runs_final.to_csv("simulation_results_backup.csv", index=False)
            print("⚠️ Saved backup to simulation_results_backup.csv")
        except:
            pass

    print("Batch Run Complete. (Pure Headless Mode)")

if __name__ == "__main__":
    main()
