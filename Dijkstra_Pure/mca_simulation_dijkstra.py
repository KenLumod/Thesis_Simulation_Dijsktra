"""
MCA Evacuation Simulation (Pure Dijkstra V2 - Strict ACO Template Match)
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
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
config_path = BASE_DIR.parent / "GPKG_Files" / "config.txt"

with open(config_path) as f:
    line = f.read().strip()
    themap = line.split("=", 1)[1].strip().strip('"')
print("Start simulation", themap)

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
        self.STAMPEDE_DENSITY = 3.5 # p/m^2 
        self.DEATH_RATE = 0.1 # 10% per second if overcrowded
        
        # State
        self.population = {} # cell_index -> count
        self.casualties = 0
        self.casualties_per_cell = {} # cell_id -> total_deaths
        self.time_step = 0
        self.history = []
        self.casualty_history = []
        self.per_cell_casualty_history = [] # LIST of DICTS
        
        # Exit Analysis
        self.exit_status = {} # exit_id -> 'OPEN'/'CLOSED'
        self.exit_usage = {}  # exit_id -> count (accumulated)
        self.exit_usage_history = [] # list of dicts
        self.road_to_exit = {} # road_cell_id -> exit_id
        
        # Viz Data
        self.cell_centroids = {}
        self.flow_vectors = {} # cell_idx -> (u, v)

        # DIJKSTRA Specific
        self.dijkstra_distances = {} # Final Distance Map

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
                
                # Add weighted edge for True Dijkstra
                self.graph.add_edge(id_l, id_r, weight=dist)

    def apply_scenario_blockages(self, blocked_ids):
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
                        self.road_to_exit[cid] = idx
                
                if connected_cells:
                    self.exit_status[idx] = 'OPEN'
                    exit_indices.extend(connected_cells)
                    print(f"DEBUG: Exit {idx} is OPEN (Connected to {len(connected_cells)} cells)")
                else:
                    self.exit_status[idx] = 'CLOSED'
                    print(f"DEBUG: Exit {idx} is CLOSED")

        if not exit_indices:
            print("WARNING: No exits connected. Using fallback (West bounds).")
            min_x = self.road_cells.bounds.minx.min()
            ids = self.road_cells[self.road_cells.bounds.minx < min_x + 10]['id'].tolist()
            exit_indices = ids
            for i in ids: self.road_to_exit[i] = 999 

        self.exits_ids = exit_indices
        valid_exits = [e for e in self.exits_ids if e in self.graph]
        
        if not valid_exits:
             print("❌ No reachable exits for Dijkstra!")
             return

        # ------------------------------------------------------------------------
        # DIJKSTRA SOLVER CALL (Replaces ACO)
        # ------------------------------------------------------------------------
        print(f"Using Dijkstra Solver from {len(exit_indices)} exit nodes...")
        # Capture History for Animation (Phase 1 = Floodfill)
        self.dijkstra_distances, raw_visited = Dijkstra.calculate_dijkstra_field(
            self.graph, valid_exits
        )
        
        # Post-process for Animation Frames (Chunking)
        self.dijkstra_history = []
        visited_so_far = set()
        chunk_size = 10 # Animate 10 nodes per frame for smooth floodfill
        
        for i in range(0, len(raw_visited), chunk_size):
            chunk = raw_visited[i : i+chunk_size]
            for node, dist in chunk:
                visited_so_far.add(node)
                
            self.dijkstra_history.append({
                'visited': visited_so_far.copy()
            })
            
        # Ensure final state is captured
        if not self.dijkstra_history or len(visited_so_far) < len(raw_visited):
             for node, dist in raw_visited:
                 visited_so_far.add(node)
             self.dijkstra_history.append({'visited': visited_so_far.copy()})
        
        # Derive Flow Directions (Gradient Descent)
        self.directions = {}
        for node in self.graph.nodes:
            if node in exit_indices:
                self.directions[node] = None
                continue
            
            best_neighbor = None
            min_dist = self.dijkstra_distances.get(node, float('inf'))
            current_best_dist = min_dist
            
            for neighbor in self.graph.neighbors(node):
                d = self.dijkstra_distances.get(neighbor, float('inf'))
                if d < current_best_dist:
                    current_best_dist = d
                    best_neighbor = neighbor
            
            self.directions[node] = best_neighbor
        
        # Pre-calc flow vectors (Standard Visualization)
        self.flow_vectors = {}  # For Agents (Occupancy)
        self.global_flow_vectors = {} # For Static Field Viz
        
        for idx in self.graph.nodes:
            target_idx = self.directions.get(idx)
            dx, dy = 0, 0
            if target_idx is not None and target_idx in self.cell_centroids:
                start = self.cell_centroids[idx]
                end = self.cell_centroids[target_idx]
                dx_raw = end.x - start.x
                dy_raw = end.y - start.y
                norm = np.hypot(dx_raw, dy_raw)
                if norm > 0:
                    dx = dx_raw / norm
                    dy = dy_raw / norm
            
            # Populate
            self.flow_vectors[idx] = (dx, dy)
            self.global_flow_vectors[idx] = (dx, dy)
    
    def initialize_population(self, total_agents=5000):
        self.total_agents_init = total_agents
        print(f"Initializing {total_agents} agents...")
        nodes = list(self.graph.nodes)
        
        # Prefer spawning near 'spawns'
        if self.spawns is not None:
            source_ids = []
            spawn_buffer = self.spawns.buffer(5.0)
            for idx, cell in self.road_cells.iterrows():
                if spawn_buffer.intersects(cell.geometry).any():
                    source_ids.append(cell['id'])
            
            if source_ids:
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
        
        self.history = []
        self.history.append(self.population.copy())
        
        self.casualty_history = []
        self.casualty_history.append(0)
        
        self.exit_usage_history = []
        self.exit_usage_history.append(self.exit_usage.copy())
        
        self.per_cell_casualty_history = []
        self.per_cell_casualty_history.append(self.casualties_per_cell.copy())

    def step(self):
        new_population = self.population.copy()
        total_deaths_this_step = 0
        
        # 1. Check for Stampedes
        if self.time_step > 10:
            for cid, count in self.population.items():
                area = self.cell_areas.get(cid, 60.0)
                rho = count / area if area > 0 else 0
                
                if rho > self.STAMPEDE_DENSITY:
                    deaths = count * self.DEATH_RATE
                    new_population[cid] -= deaths
                    total_deaths_this_step += deaths
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
                     width = self.CELL_WIDTH
                     
                     flow_out = (self.V_FREE * width * self.DT * self.RHO_MAX)
                     actual_out = min(new_population[cid], flow_out)
                     
                     new_population[cid] = max(0, new_population[cid] - actual_out)
                     
                     if self.road_to_exit.get(cid) is not None:
                         self.exit_usage[self.road_to_exit.get(cid)] += actual_out
                continue
            
            # Flow Calculation
            area = self.cell_areas.get(cid, 60.0)
            cap = self.max_capacities.get(cid, 300.0)
            
            rho_i = current_pop / area if area > 0 else 0
            v_i = self.V_FREE * np.exp(-rho_i / self.RHO_MAX)
            
            q_out = rho_i * v_i * self.CELL_WIDTH * self.DT
            
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

    def export_to_excel(self, filename="simulation_results_dijkstra_1.xlsx"):
        print(f"Exporting results to {filename}...")
        
        time_data = []
        cell_params = {}
        for idx, row in self.road_cells.iterrows():
            cid = row['id']
            cell_params[cid] = self.cell_areas.get(cid, 60.0)

        for t, (pop_map, cas_count) in enumerate(zip(self.history, self.casualty_history)):
            total_alive = sum(pop_map.values())
            
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
        
        cell_data = []
        for cid in self.road_cells['id']:
            final_deaths = self.casualties_per_cell.get(cid, 0)
            area = self.cell_areas.get(cid, 60.0)
            
            cell_data.append({
                'Cell ID': cid,
                'Spatial Distribution (Casualties)': int(final_deaths),
                'Area': area,
                'Status': 'BLOCKED' if cid in self.blocked_cells else 'OPEN'
            })
        df_cells = pd.DataFrame(cell_data).sort_values(by='Spatial Distribution (Casualties)', ascending=False)
        
        exit_data = []
        for eid, count in self.exit_usage.items():
            status = self.exit_status.get(eid, 'UNKNOWN')
            exit_data.append({
                'Exit ID': eid,
                'Exit Usage Distribution': int(count),
                'Status': status
            })
        df_exits = pd.DataFrame(exit_data)
        
        final_time = (len(self.history) - 1) * self.DT
        if self.history[-1] and sum(self.history[-1].values()) == 0:
             for t, pop in enumerate(self.history):
                if sum(pop.values()) == 0:
                     final_time = t * self.DT
                     break
        
        df_summary = pd.DataFrame([{
            'Total Evacuation Time': final_time,
            'Total Casualties': int(self.casualties), # Round to int
            'Total Evacuated': int(sum(self.exit_usage.values())),
            'Remaining Agents': int(sum(self.history[-1].values()))
        }])

        try:
            def color_columns(s):
                c_gray, c_red = 'background-color: #e0e0e0', 'background-color: #ffcccb'
                c_green, c_blue = 'background-color: #d0f0c0', 'background-color: #add8e6'
                c_yellow = 'background-color: #ffffe0'
                
                name = str(s.name)
                if 'Casualties' in name or 'Remaining' in name or 'Deaths' in name: return [c_red] * len(s)
                elif 'Evacuated' in name or 'Flow' in name or 'Usage' in name: return [c_green] * len(s)
                elif 'Density' in name or 'Velocity' in name or 'Area' in name or 'Speed' in name: return [c_blue] * len(s)
                elif 'Time' in name: return [c_yellow] * len(s)
                return [''] * len(s)

            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                try:
                    df_time.style.apply(color_columns, axis=0).to_excel(writer, sheet_name='Time Metrics', index=False)
                    df_cells.sort_values(by='Spatial Distribution (Casualties)', ascending=False).style.apply(color_columns, axis=0).to_excel(writer, sheet_name='Spatial Distribution', index=False)
                    df_exits.style.apply(color_columns, axis=0).to_excel(writer, sheet_name='Exit Usage', index=False)
                    df_summary.style.apply(color_columns, axis=0).to_excel(writer, sheet_name='Summary', index=False)
                    print(f"✅ Simulation results saved to {filename} (Colored)")
                except Exception:
                    df_time.to_excel(writer, sheet_name='Time Metrics', index=False)
                    df_cells.to_excel(writer, sheet_name='Spatial Distribution', index=False)
                    df_exits.to_excel(writer, sheet_name='Exit Usage', index=False)
                    df_summary.to_excel(writer, sheet_name='Summary', index=False)
                    print(f"✅ Simulation results saved to {filename} (Unstyled)")

        except Exception as e:
            print(f"❌ Failed to save Excel: {e}")

    def run(self, steps=100):
        print(f"Starting simulation for {steps} steps...")
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
        
        print("\n" + "="*40)
        print(f"=== CASUALTY REPORT (Total: {self.casualties:.1f}) ===")
        print("Breakdown by Cell:")
        sorted_cells = sorted(self.casualties_per_cell.items(), key=lambda x: x[1], reverse=True)
        count = 0
        for cid, deaths in sorted_cells:
            if deaths > 0:
                area = self.cell_areas.get(cid, 0)
                cap = self.max_capacities.get(cid, 0)
                degree = len(list(self.graph.neighbors(cid))) if cid in self.graph else 0
                print(f"  Cell {cid:>4}: {deaths:>6.1f} deaths | Area: {area:>5.1f}m2 | MaxCap: {cap:>5.1f} | Neighbors: {degree}")
                count += 1
        
        if count == 0: print("  No casualties reported.")
        print("="*40 + "\n")
        
        # Export Data
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.export_to_excel(filename=os.path.join(script_dir, "simulation_results_dijkstra_1.xlsx"))
                
    def animate_results(self):
        print("Preparing animation...")
        from matplotlib.widgets import Slider, Button
        
        fig = plt.figure(figsize=(14, 10))
        
        ax = fig.add_axes([0.1, 0.2, 0.8, 0.75])
        cax = fig.add_axes([0.92, 0.25, 0.02, 0.6])
        
        if self.spawns is not None:
             self.spawns.plot(ax=ax, color='blue', marker='o', markersize=30, label='Spawn Points', zorder=5)
        if self.exits is not None:
             self.exits.plot(ax=ax, color='lime', marker='X', markersize=150, edgecolor='black', linewidth=2, label='Exits', zorder=10)
        
        if self.blocked_cells:
            blocked_geom = self.road_cells[self.road_cells['id'].isin(self.blocked_cells)]
            if not blocked_geom.empty:
                blocked_geom.plot(ax=ax, color='black', alpha=0.7, zorder=6, label='Blocked/Destroyed')
                blocked_geom.plot(ax=ax, color='none', edgecolor='red', hatch='XX', zorder=7)
             
        ax.legend(loc='upper right')
        ax.set_title("USTP Evacuation (Per-Cell Capacity Analysis)", fontsize=14, fontweight='bold')
        ax.axis('off')
        ax.set_aspect('equal', anchor='C') 

        info_text = ax.text(0.02, 0.9, "", transform=ax.transAxes, fontsize=11,
                           bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'))
        
        selected_cell_text = ax.text(0.02, 0.82, "Selection: None", transform=ax.transAxes, fontsize=10,
                                     verticalalignment='top', bbox=dict(facecolor='ivory', alpha=0.9, boxstyle='round,pad=0.5'))

        self.road_cells.plot(ax=ax, color='white', edgecolor='lightgray', linewidth=0.5)
        self.road_cells.plot(ax=ax, column='id', cmap='turbo', alpha=0.8, vmin=0, vmax=1.0)
        collection = ax.collections[-1]
        
        sm = plt.cm.ScalarMappable(cmap='turbo', norm=plt.Normalize(vmin=0, vmax=1.0))
        sm._A = []
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label('Occupancy Ratio (Population / Capacity)')

        self.quiver = None
        self.selected_cell_id = None
        self.selected_exit_id = None
        self.current_frame = 0
        self.view_mode = 'occupancy' 
        self.remaining_artist = None 

        def update_selection_text(frame_idx):
            if self.selected_exit_id is not None:
                eid = self.selected_exit_id
                status = self.exit_status.get(eid, 'UNKNOWN')
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
            
            selected_cell_text.set_backgroundcolor('ivory')
            cid = self.selected_cell_id
            
            # PHASE 1 Check (Dijkstra Floodfill)
            if hasattr(self, 'dijkstra_history') and frame_idx < len(self.dijkstra_history):
                 step_data = self.dijkstra_history[frame_idx]
                 visited = step_data['visited']
                 dist = self.dijkstra_distances.get(cid, float('inf'))
                 
                 is_visited = "YES" if cid in visited else "NO"
                 dist_str = f"{dist:.1f}m" if dist != float('inf') else "INF"
                 
                 selected_cell_text.set_text(
                    f"Cell ID: {cid}\n"
                    f"Floodfill Phase\n"
                    f"Visited: {is_visited}\n"
                    f"Dist Exp: {dist_str}"
                 )
                 return

            # PHASE 2: Adjust Frame
            sim_frame = frame_idx
            if hasattr(self, 'dijkstra_history'):
                 sim_frame = frame_idx - len(self.dijkstra_history)
                 
            if self.view_mode == 'occupancy':
                pop_data = self.history[sim_frame]
                p = pop_data.get(cid, 0)
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
            elif self.view_mode == 'dijkstra':
                # Show Static Dist
                d = self.dijkstra_distances.get(cid, float('inf'))
                d_str = f"{d:.1f}" if d != float('inf') else "Unreachable"
                selected_cell_text.set_text(
                     f"Cell ID: {cid}\n"
                     f"Exit Dist: {d_str}m\n"
                )
            else:
                if sim_frame < len(self.per_cell_casualty_history):
                    d = self.per_cell_casualty_history[sim_frame].get(cid, 0)
                else:
                    d = self.casualties_per_cell.get(cid, 0)
                    
                selected_cell_text.set_text(
                    f"Cell ID: {cid}\n"
                    f"Total Casualties: {d:.1f}\n"
                    f"(Historical Accumulation)"
                )

        def update(frame):
            self.current_frame = frame
            
            # --- PHASE 1: DIJKSTRA FORMATION (PROGRESSIVE HEATMAP) ---
            if hasattr(self, 'dijkstra_history') and frame < len(self.dijkstra_history):
                step_data = self.dijkstra_history[frame]
                visited_nodes = step_data['visited']
                
                # Calculate Max Dist for Normalization (Do once ideally, but cheap enough here)
                max_d = 0
                for d in self.dijkstra_distances.values():
                    if d != float('inf'): max_d = max(max_d, d)
                
                # Progressive Heatmap: 
                # Unvisited = 0 (Dark)
                # Visited = (1.0 - dist/max) (Bright=Close, Red=Far)
                ratios = []
                for idx in self.road_cells['id']:
                     if idx in visited_nodes:
                         d = self.dijkstra_distances.get(idx, float('inf'))
                         if d != float('inf') and max_d > 0:
                             ratios.append(1.0 - (d / max_d))
                         else:
                             ratios.append(0.0)
                     else:
                         ratios.append(0.0) # Masked later? Or just 0
                
                collection.set_array(np.array(ratios))
                collection.set_cmap('magma') 
                collection.set_clim(0, 1.0)
                
                info_text.set_text(
                    f"🌊 DIJKSTRA WAVEFRONT\n"
                    f"Step: {frame * 10}\n" 
                    f"Nodes Scanned: {len(visited_nodes)}\n"
                    f"Mapping Distances..."
                )
                
                ax.set_title(f"Phase 1: Dijkstra Map Formation (Distance Wave)", fontsize=14, fontweight='bold')
                return collection, info_text

            # --- PHASE 2: EVACUATION ---
            # Correct frame mapping
            sim_frame = frame
            if hasattr(self, 'dijkstra_history'):
                 sim_frame = frame - len(self.dijkstra_history)

            pop_data = self.history[sim_frame]
            casualties = self.casualty_history[sim_frame]
            
            # --- VIEW MODES ---
            
            # 1. OCCUPANCY (Default)
            if self.view_mode == 'occupancy':
                 ax.set_title("Phase 2: Evacuation (Occupancy)", fontsize=14, fontweight='bold')
                 collection.set_cmap('turbo')
                 
                 ratios = []
                 for idx in self.road_cells['id']:
                      p = pop_data.get(idx, 0)
                      c = self.capacities.get(idx, 1.0)
                      r = p / c if c > 0 else 0
                      ratios.append(r)
                 collection.set_array(np.array(ratios))
                 collection.set_clim(0, 1.0)
                 
                 # Dynamic Flow Arrows (Only where agents are)
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

            # 2. DIJKSTRA STATIC FIELD (The "Whole Map" View)
            elif self.view_mode == 'dijkstra':
                 ax.set_title("Static Dijkstra Field (Heatmap: Distance | Arrows: Gradient)", fontsize=14, fontweight='bold')
                 
                 # Show Distance Map
                 d_vals = []
                 max_d = 0
                 for idx in self.road_cells['id']:
                      d = self.dijkstra_distances.get(idx, float('inf'))
                      if d != float('inf'): max_d = max(max_d, d)
                      d_vals.append(d)
                 
                 # Normalize (Inverse: Close=Bright, Far=Dark/Red)
                 norm_vals = []
                 for d in d_vals:
                     if d == float('inf'): norm_vals.append(0)
                     else: 
                         # Invert so Exits are Brightest
                         val = 1.0 - (d / max_d) if max_d > 0 else 0
                         norm_vals.append(val)
                 
                 collection.set_array(np.array(norm_vals))
                 collection.set_cmap('magma') 
                 collection.set_clim(0, 1.0)
                 
                 # STATIC GLOBAL FLOW ARROWS (Quiver) - Shows "where flow is going"
                 if self.quiver: self.quiver.remove()
                 
                 # Subsample arrows to avoid clutter if too many cells
                 xq, yq, uq, vq = [], [], [], []
                 count = 0
                 for idx, (u, v) in self.global_flow_vectors.items():
                     # Plot every cell's arrow? Or subsample?
                     # Let's plot ALL for now, but maybe use thin arrows
                     centroid = self.cell_centroids.get(idx)
                     if centroid and (u != 0 or v != 0):
                         xq.append(centroid.x)
                         yq.append(centroid.y)
                         uq.append(u)
                         vq.append(v)
                 
                 if xq:
                     # White arrows for visibility on Magma
                     self.quiver = ax.quiver(xq, yq, uq, vq, scale=30, width=0.003, color='white', alpha=0.5, zorder=6)
                 else:
                     self.quiver = None

            # 3. CASUALTIES
            else:
                if self.quiver: 
                     self.quiver.remove()
                     self.quiver = None
                
                # Title
                ax.set_title("Phase 2: Evacuation (Casualties)", fontsize=14, fontweight='bold')

                if sim_frame < len(self.per_cell_casualty_history):
                    current_deaths = self.per_cell_casualty_history[sim_frame]
                else:
                    current_deaths = self.casualties_per_cell
                    
                deaths = [current_deaths.get(idx, 0) for idx in self.road_cells['id']]
                collection.set_array(np.array(deaths))
                collection.set_cmap('Reds')
                collection.set_clim(0, 1) 

            # Common Artifacts (Stars for stranded)
            if self.view_mode != 'dijkstra':
                 if self.remaining_artist:
                    try: self.remaining_artist.remove()
                    except: pass 
                    self.remaining_artist = None
 
                 if sim_frame == len(self.history) - 1:
                    remaining_ids = [k for k, v in pop_data.items() if v > 0.1]
                    if remaining_ids:
                        rem_geom = self.road_cells[self.road_cells['id'].isin(remaining_ids)]
                        if not rem_geom.empty:
                            centroids = rem_geom.geometry.centroid
                            centroids.plot(ax=ax, color='#ff00ff', marker='*', markersize=150, zorder=8, edgecolor='black', linewidth=1)
                            self.remaining_artist = ax.collections[-1]
            else:
                 if self.remaining_artist: 
                     try: self.remaining_artist.remove()
                     except: pass
                     self.remaining_artist = None

            current_agents = sum(pop_data.values())
            evacuated_count = self.total_agents_init - int(current_agents) - int(casualties)
            
            status_line = ""
            if frame == len(self.history) - 1 and current_agents > 0:
                status_line = f"\n⚠️ FINAL: {int(current_agents)} Agents Stranded (Magenta)"

            info_text.set_text(
                f"⏱️ Time: {frame}s\n"
                f"👥 Alive: {int(current_agents)}\n"
                f"💀 Casualties: {int(casualties)}\n"
                f"✅ Evacuated: {evacuated_count}"
                f"{status_line}"
            )
            
            update_selection_text(frame)
            return collection, info_text, selected_cell_text

        ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03], facecolor='lightgoldenrodyellow')
        self.speed_slider = Slider(ax_slider, 'Speed', 10, 1000, valinit=100, valstep=10)
        
        def update_speed(val):
            self.anim.event_source.interval = val
        self.speed_slider.on_changed(update_speed)
        
        ax_pause = plt.axes([0.35, 0.04, 0.1, 0.04])
        self.btn_pause = Button(ax_pause, 'Pause', color='lightgray', hovercolor='gray')
        
        ax_btn = plt.axes([0.50, 0.04, 0.12, 0.04])
        self.btn_casualty = Button(ax_btn, 'Casualties', color='salmon', hovercolor='red')
        
        # 3rd Button: DIJKSTRA
        ax_btn_aco = plt.axes([0.65, 0.04, 0.12, 0.04])
        self.btn_dijkstra = Button(ax_btn_aco, 'Show Dijkstra', color='lightblue', hovercolor='cyan')

        def set_view_mode(mode):
            self.view_mode = mode
            self.btn_casualty.color = 'salmon'
            self.btn_casualty.label.set_text('Casualties')
            self.btn_dijkstra.color = 'lightblue'
            
            if mode == 'casualties':
                self.btn_casualty.color = 'red' 
                ax.set_title("Total Casualties Heatmap", fontsize=14, fontweight='bold')
                cbar.set_label('Total Casualties')
            elif mode == 'dijkstra':
                self.btn_dijkstra.color = 'cyan' 
                ax.set_title("Static Dijkstra Distance Field", fontsize=14, fontweight='bold')
                cbar.set_label('Proximity to Exit (Bright=Close)')
            else: 
                ax.set_title("USTP Evacuation (Per-Cell Capacity Analysis)", fontsize=14, fontweight='bold')
                cbar.set_label('Occupancy Ratio (Population / Capacity)')
            
            update(self.current_frame)
            fig.canvas.draw_idle()

        def toggle_casualty(event):
            if self.view_mode == 'casualties':
                set_view_mode('occupancy')
            else:
                set_view_mode('casualties')
                
        def toggle_dijkstra(event):
            if self.view_mode == 'dijkstra':
                set_view_mode('occupancy')
            else:
                set_view_mode('dijkstra')
            
        self.btn_casualty.on_clicked(toggle_casualty)
        self.btn_dijkstra.on_clicked(toggle_dijkstra)

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

        def on_scroll(event):
            if event.inaxes != ax: return
            base_scale = 1.1
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()
            xdata = event.xdata
            ydata = event.ydata
            
            if event.button == 'up': scale_factor = 1/base_scale
            elif event.button == 'down': scale_factor = base_scale
            else: scale_factor = 1
                
            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
            
            rel_x = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
            rel_y = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])
            
            ax.set_xlim([xdata - new_width * (1-rel_x), xdata + new_width * (rel_x)])
            ax.set_ylim([ydata - new_height * (1-rel_y), ydata + new_height * (rel_y)])
            fig.canvas.draw_idle()

        self.pan_start = None
        def on_press(event):
            if event.button == 2: self.pan_start = (event.xdata, event.ydata)
            if event.button == 1 and event.inaxes == ax: on_click_inspect(event)

        def on_release(event):
            if event.button == 2: self.pan_start = None

        def on_motion(event):
            if self.pan_start is None or event.inaxes != ax: return
            dx = event.xdata - self.pan_start[0]
            dy = event.ydata - self.pan_start[1]
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()
            ax.set_xlim(cur_xlim[0] - dx, cur_xlim[1] - dx)
            ax.set_ylim(cur_ylim[0] - dy, cur_ylim[1] - dy)
            fig.canvas.draw_idle()

        def on_click_inspect(event):
            if event.inaxes != ax: return
            click_pt = Point(event.xdata, event.ydata)
            
            nearest_exit = None
            min_exit_dist = float('inf')
            
            if self.exits is not None:
                for idx, row in self.exits.iterrows():
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
                update_selection_text(self.current_frame)
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect('button_press_event', on_press)
        fig.canvas.mpl_connect('button_release_event', on_release)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        fig.canvas.mpl_connect('scroll_event', on_scroll)
        
        total_frames = len(self.history)
        if hasattr(self, 'dijkstra_history'):
             total_frames += len(self.dijkstra_history)

        self.anim = FuncAnimation(fig, update, frames=total_frames, interval=100, blit=False, repeat=False)
        plt.show()

def show_launcher():
    root = tk.Tk()
    root.title("MCA Simulation Launcher")
    
    bg_color = "#f0f0f0"
    root.configure(bg=bg_color)
    
    window_width = 450
    window_height = 350
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x_c = int((screen_width/2) - (window_width/2))
    y_c = int((screen_height/2) - (window_height/2))
    root.geometry(f"{window_width}x{window_height}+{x_c}+{y_c}")
    
    config = {'block': [], 'agents': 5000, 'steps': 300}
    
    lbl_header = tk.Label(root, text="Simulation Configuration", font=("Segoe UI", 16, "bold"), bg=bg_color)
    lbl_header.pack(pady=15)
    
    frame_form = tk.Frame(root, bg=bg_color)
    frame_form.pack(pady=10, padx=20, fill="x")
    
    tk.Label(frame_form, text="Total Agents:", font=("Segoe UI", 10), bg=bg_color, anchor="w").grid(row=0, column=0, sticky="w", pady=5)
    ent_agents = tk.Entry(frame_form, font=("Segoe UI", 10))
    ent_agents.insert(0, "5000")
    ent_agents.grid(row=0, column=1, sticky="e", padx=5)

    tk.Label(frame_form, text="Duration (Steps/Secs):", font=("Segoe UI", 10), bg=bg_color, anchor="w").grid(row=1, column=0, sticky="w", pady=5)
    ent_steps = tk.Entry(frame_form, font=("Segoe UI", 10))
    ent_steps.insert(0, "500")
    ent_steps.grid(row=1, column=1, sticky="e", padx=5)

    tk.Label(frame_form, text="Blocked Cells (IDs):", font=("Segoe UI", 10), bg=bg_color, anchor="w").grid(row=2, column=0, sticky="w", pady=5)
    ent_block = tk.Entry(frame_form, font=("Segoe UI", 10))
    ent_block.grid(row=2, column=1, sticky="e", padx=5)
    
    tk.Label(frame_form, text="(Space separated, e.g. '88 78')", font=("Segoe UI", 8, "italic"), bg=bg_color, fg="gray").grid(row=3, column=1, sticky="e")

    def on_start():
        try:
            config['agents'] = int(ent_agents.get())
            config['steps'] = int(ent_steps.get())
            block_str = ent_block.get().strip()
            if block_str:
                config['block'] = [int(x) for x in block_str.split()]
            root.destroy()
        except ValueError:
            messagebox.showerror("Error", "Invalid numeric input!")
    
    btn_start = tk.Button(root, text="START SIMULATION", command=on_start, 
                          bg="#4CAF50", fg="white", font=("Segoe UI", 12, "bold"), 
                          relief="flat", padx=20, pady=5, cursor="hand2")
    btn_start.pack(pady=25)
    
    root.mainloop()
    return config

def main():
    config = show_launcher()
    
    print(f"Starting with config: {config}")

    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    gpkg_path = os.path.join(base_dir, "..", "GPKG_Files", themap)

    sim = MCASimulation(gpkg_path)
    sim.load_data()
    sim.build_graph()
    
    if config['block']:
        sim.apply_scenario_blockages(config['block'])
        
    sim.compute_flow_directions()
    
    if not config['block']:
        sim.analyze_specific_cell(88) 
    
    sim.initialize_population(total_agents=config['agents'])
    sim.run(steps=config['steps'])
    sim.animate_results()

if __name__ == "__main__":
    main()
