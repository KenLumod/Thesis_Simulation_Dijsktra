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
        self.STAMPEDE_DENSITY = 3.5 # p/m^2 (Lowered from 4.0) (Lowered from 4.5 to increase sensitivity)
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
        
        # ----------------------------------------------------------------
        # HAZARD-AWARE DIJKSTRA PARAMETERS
        # ----------------------------------------------------------------
        # 1. Weights (Influence on Routing)
        self.W_FIRE = 1.0
        self.W_SMOKE = 0.8
        self.W_DENSITY = 0.6
        self.W_DEBRIS = 0.4
        self.W_TERRAIN = 0.2
        self.W_TEMP_OBJ = 0.2
        
        # 2. Dynamics
        self.HAZARD_COST_MULT = 10.0
        self.REROUTE_THRESHOLD = 0.7
        self.REROUTE_COOLDOWN = 5
        self.last_reroute_time = -999
        
        # 3. State
        self.penalties = {} 
        self.current_max_penalty = 0.0
        self.burnt_cells = set() # Track cells that have peaked/decayed

    def load_data(self):
        print(f"Loading data from {self.gpkg_path}...")
        import fiona
        try:
            layers = fiona.listlayers(self.gpkg_path)
            
            # Road Cells
            layer_name = 'road_cells'
            if layer_name not in layers: layer_name = layers[0]
            
            self.road_cells = gpd.read_file(self.gpkg_path, layer=layer_name)
            
            # AUTO-FIX: Reproject if Lat/Lon (Geographic)
            if self.road_cells.crs and self.road_cells.crs.is_geographic:
                print("⚠️ Map is unprojected (Lat/Lon). Auto-converting to EPSG:3857 (Meters)...")
                self.road_cells = self.road_cells.to_crs(epsg=3857)
            
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
                if 'cell_area' in row and row['cell_area'] > 0.1:
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
                print(f"Loaded {len(self.exits)} sink nodes (exits) from layer '{target_layer}'.")
            
            # Safe Zones (New)
            self.safe_zones = None
            if 'safe_zones' in layers:
                self.safe_zones = gpd.read_file(self.gpkg_path, layer='safe_zones')
                print(f"Loaded {len(self.safe_zones)} Safe Zones from layer 'safe_zones'.")
            else:
                print("WARNING: 'safe_zones' layer not found!")

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
            
            # Initialize Penalties
            self.penalties[cid] = {
                'fire': 0.0, 'smoke': 0.0, 'debris': 0.0, 
                'terrain': 0.0, 'temp': 0.0
            }
            
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
        print("Computing flow directions (SAFE ZONE MODE)...")
        
        # 1. Map Road Cells to Exits
        exit_indices = []
        if self.exits is not None:
            for idx, row in self.exits.iterrows():
                self.exit_status[idx] = 'CLOSED'
                self.exit_usage[idx] = 0.0
                exit_geo = row.geometry.buffer(5.0)
                connected = []
                for r_idx, cell in self.road_cells.iterrows():
                    if exit_geo.intersects(cell.geometry):
                        cid = cell['id']
                        connected.append(cid)
                        self.road_to_exit[cid] = idx
                
                if connected:
                    self.exit_status[idx] = 'OPEN'
                    exit_indices.extend(connected)

        if not exit_indices:
            print("WARNING: No exits connected. Using fallback.")
            min_x = self.road_cells.bounds.minx.min()
            ids = self.road_cells[self.road_cells.bounds.minx < min_x + 10]['id'].tolist()
            exit_indices = ids
            for i in ids: self.road_to_exit[i] = 999 

        self.exits_ids = exit_indices
        valid_exits = [e for e in self.exits_ids if e in self.graph]
        
        # 2. Map Road Cells to Safe Zones & Boost Capacity
        safe_indices = []
        if self.safe_zones is not None:
             sz_buf = self.safe_zones.copy()
             sz_buf.geometry = sz_buf.geometry.buffer(2.0)
             
             for idx, cell in self.road_cells.iterrows():
                 if sz_buf.intersects(cell.geometry).any():
                     cid = cell['id']
                     safe_indices.append(cid)
                     # Boost Capacity
                     if cid in self.max_capacities:
                         self.max_capacities[cid] *= 4.0
                         self.capacities[cid] *= 4.0 # Visual safe cap too
        
        valid_safe = [s for s in safe_indices if s in self.graph]
        self.safe_zone_cells = set(valid_safe)
        print(f"Mapped {len(valid_safe)} Safe Zone cells. Capacity Boosted x4.")

        if not valid_exits:
             print("❌ No reachable exits!")
             return

        # ------------------------------------------------------------------------
        # STAGE 1: Dijkstra from EXITS (To know dist from SafeZone to Exit)
        # ------------------------------------------------------------------------
        print("DEBUG: Checking Exit Connectivity (BFS)...")
        for ex in valid_exits:
            reachable = 0
            q = [ex]
            seen = {ex}
            while q:
                curr = q.pop(0)
                reachable += 1
                for n in self.graph.neighbors(curr):
                    if n not in seen:
                        seen.add(n)
                        q.append(n)
            
            fid = self.road_to_exit.get(ex, 'UNK')
            status = "ISOLATED/BLOCKED" if reachable < 5 else f"OK ({reachable} nodes)"
            print(f"  Exit Node {ex} (FID {fid}): {status}")

        print("DEBUG: Running Forensic Distance Audit (Safe Zone -> Per Exit)...")
        fid_exits = {}
        for ex in valid_exits:
            fid = self.road_to_exit.get(ex, 'UNK')
            if fid not in fid_exits: fid_exits[fid] = []
            fid_exits[fid].append(ex)
        
        sz_audit = {sz: {} for sz in valid_safe}
        
        for fid, nodes in fid_exits.items():
            d_field, _ = Dijkstra.calculate_dijkstra_field(self.graph, nodes)
            for sz in valid_safe:
                dist = d_field.get(sz, float('inf'))
                sz_audit[sz][fid] = dist
                
        print("\n=== SAFE ZONE DISTANCE AUDIT ===")
        print(f"{'SZ_ID':<8} | {'BEST_EXIT (FID)':<20} | {'DISTANCES (m)'}")
        for sz in valid_safe:
            best_fid = None
            best_d = float('inf')
            dist_strs = []
            for fid, d in sz_audit[sz].items():
                d_str = f"{d:.1f}" if d != float('inf') else "INF"
                dist_strs.append(f"FID{fid}: {d_str}")
                if d < best_d:
                    best_d = d
                    best_fid = fid
            
            print(f"{sz:<8} | {f'FID {best_fid} ({best_d:.1f}m)':<20} | {', '.join(dist_strs)}")
        print("================================\n")

        print("Stage 1: Computing Separate Dijkstra Fields for Each Exit...")
        
        self.exit_fields = {}
        
        for fid, nodes in fid_exits.items():
            print(f"  > Computing Field for Exit FID {fid} (Nodes: {nodes})...")
            d, v = Dijkstra.calculate_dijkstra_field(self.graph, nodes)
            self.exit_fields[fid] = {'distances': d, 'visited': v}
            
        self.dist_to_exit = {}
        all_visited = [] 
        
        for fid in self.exit_fields:
            all_visited.extend(self.exit_fields[fid]['visited'])
            d_map = self.exit_fields[fid]['distances']
            for n, dist in d_map.items():
                cur = self.dist_to_exit.get(n, float('inf'))
                if dist < cur:
                    self.dist_to_exit[n] = dist
        
        # ------------------------------------------------------------------------
        # TRACE PATHS & ASSIGN SAFE ZONES (FORCED LOAD BALANCING)
        # ------------------------------------------------------------------------
        self.safe_paths = []
        self.safe_path_nodes = []
        print("\n=== SAFE ZONE ASSIGNMENT (ROUND ROBIN BALANCING) ===")
        
        exit_counts = {fid: 0 for fid in fid_exits}
        available_fids = sorted(list(fid_exits.keys()))
        
        for i, sz in enumerate(valid_safe):
            assigned_fid = None
            assigned_dist = float('inf')
            
            start_index = i % len(available_fids)
            
            for offset in range(len(available_fids)):
                idx = (start_index + offset) % len(available_fids)
                fid = available_fids[idx]
                
                data = self.exit_fields[fid]
                d = data['distances'].get(sz, float('inf'))
                
                if d != float('inf'):
                    assigned_fid = fid
                    assigned_dist = d
                    break 
            
            if assigned_fid is None:
                print(f"  ⚠️ SZ {sz} -> Unreachable from ANY exit!")
                continue
                
            best_fid = assigned_fid
            best_dist = assigned_dist
            exit_counts[best_fid] += 1
            
            # TRACE PATH
            target_field = self.exit_fields[best_fid]['distances']
            target_exits = fid_exits[best_fid]
            
            path = [sz]
            curr = sz
            dist = best_dist
            
            for _ in range(300):
                if curr in target_exits: break
                
                best_n = None
                best_d = dist
                
                for n in self.graph.neighbors(curr):
                    d = target_field.get(n, float('inf')) 
                    if d < best_d:
                        best_d = d
                        best_n = n
                        
                if best_n is not None:
                    curr = best_n
                    dist = best_d
                    path.append(curr)
                else: break
            
            if len(path) > 1 and path[-1] in target_exits:
                 self.safe_path_nodes.append(path)
                 print(f"  SZ {sz} -> FORCED to Exit FID {best_fid} (Dist: {best_dist:.1f}m)")
            else:
                 print(f"  ⚠️ SZ {sz} -> Path Trace Failed for FID {best_fid}")

        print(f"Assignment Summary: {exit_counts}")
        
        # ------------------------------------------------------------------------
        # STAGE 2: Dijkstra from SAFE ZONES
        # ------------------------------------------------------------------------
        print("Stage 2: Computing Path via Safe Zones...")
        
        initial_costs = {}
        for sz in valid_safe:
            d = self.dist_to_exit.get(sz, float('inf'))
            if d != float('inf'):
                initial_costs[sz] = d
            else:
                initial_costs[sz] = float('inf') 
        
        if not valid_safe:
            print("⚠️ NO SAFE ZONES MAPPED! Reverting to direct Exit routing.")
            self.dijkstra_distances = self.dist_to_exit
        else:
            self.dijkstra_distances, _ = Dijkstra.calculate_dijkstra_field(
                self.graph, valid_safe, initial_costs=initial_costs
            )

        # Derive Flow Directions
        self.directions = {}
        for node in self.graph.nodes:
            if node in exit_indices:
                self.directions[node] = None
                continue
                
            # Hybrid Logic
            d_safe = self.dijkstra_distances.get(node, float('inf'))
            if d_safe != float('inf'):
                 source_field = self.dijkstra_distances
            else:
                 source_field = self.dist_to_exit
            
            min_dist = source_field.get(node, float('inf'))
            current_best = min_dist
            best_neighbor = None
            
            for neighbor in self.graph.neighbors(node):
                d = source_field.get(neighbor, float('inf'))
                if d < current_best:
                    current_best = d
                    best_neighbor = neighbor
            
            self.directions[node] = best_neighbor
        
        # OVERRIDE: BRIDGE SAFE ZONES TO EXITS
        if hasattr(self, 'safe_path_nodes'):
            print("Applying Safe Zone Path Overrides...")
            for path in self.safe_path_nodes:
                for i in range(len(path) - 1):
                    self.directions[path[i]] = path[i+1]
        
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



    def calculate_composite_score(self, cid, rho, fire_val):
        """
        Computes clamped composite penalty score for a cell.
        S = min(0.95, weighted_sum)
        """
        p = self.penalties.get(cid)
        if not p: return 0.0

        val_density = min(1.0, rho / self.STAMPEDE_DENSITY) if self.STAMPEDE_DENSITY > 0 else 0
        
        val_fire = min(1.0, p['fire'])
        val_smoke = min(1.0, p['smoke'])
        val_debris = min(1.0, p['debris'])
        val_terrain = min(1.0, p['terrain'])
        val_temp = min(1.0, p['temp'])
        
        w_sum = (val_fire * self.W_FIRE) + \
                (val_smoke * self.W_SMOKE) + \
                (val_density * self.W_DENSITY) + \
                (val_debris * self.W_DEBRIS) + \
                (val_terrain * self.W_TERRAIN) + \
                (val_temp * self.W_TEMP_OBJ)
                
        return min(0.95, w_sum)

    def update_penalties(self):
        """
        Decay dynamics for transient penalties.
        Fire/Terrain do NOT decay here (managed by event/static).
        """
        # Decay Rates
        DECAY_SMOKE = 0.03  # Faster Decay (was 0.01)
        DECAY_DEBRIS = 0.01 # Faster Decay (was 0.005)
        DECAY_TEMP = 0.1
        
        for cid, p_dict in self.penalties.items():
            # Decay
            if p_dict['smoke'] > 0:
                p_dict['smoke'] = max(0.0, p_dict['smoke'] - DECAY_SMOKE)
            if p_dict['debris'] > 0:
                p_dict['debris'] = max(0.0, p_dict['debris'] - DECAY_DEBRIS)
            if p_dict['temp'] > 0:
                p_dict['temp'] = max(0.0, p_dict['temp'] - DECAY_TEMP)
            
            # Fire Logic (Simple Propagation with Lifecycle)
            import random
            if p_dict['fire'] > 0:
                if cid in self.burnt_cells:
                    # DECAY PHASE
                    # User Request: "Decay is WAY too slow" -> Increased to 0.03
                    p_dict['fire'] = max(0.0, p_dict['fire'] - 0.03)
                    if p_dict['fire'] == 0:
                        self.burnt_cells.remove(cid)
                else:
                    # GROWTH PHASE
                    if p_dict['fire'] < 1.0:
                        p_dict['fire'] = min(1.0, p_dict['fire'] + 0.01) # Slower Growth (was 0.05)
                    else:
                        # Reached Peak -> Start Burnout (Switch to Decay)
                        # Increased burnout chance to 10% (So it actually dies out)
                        if random.random() < 0.10: 
                             self.burnt_cells.add(cid)
                
                # ADJUSTED: Threshold 0.2, Chance 5% (Very Slow Spread, contained)
                if p_dict['fire'] > 0.2 and cid not in self.burnt_cells:
                    neighbors = list(self.graph.neighbors(cid))
                    for n in neighbors:
                         # 5% chance to ignite neighbor (reliable spread, but slow growth)
                         if self.penalties[n]['fire'] == 0 and random.random() < 0.05:
                             self.penalties[n]['fire'] = 0.1 # Start VERY small (needs ~10 steps to become dangerous)
                             # Also add smoke
                             self.penalties[n]['smoke'] = 0.6

    def step(self):
        # 0. Update Dynamics
        self.update_penalties()


        new_population = self.population.copy()
        total_deaths_this_step = 0
        
        # 1. Check for Stampedes (Apply Grace Period of 10 steps)
        if self.time_step > 10:
            for cid, count in self.population.items():
                # SAFE ZONE IMMUNITY: No casualties in Safe Zones/Refuge Areas
                if hasattr(self, 'safe_zone_cells') and cid in self.safe_zone_cells:
                    continue

                area = self.cell_areas.get(cid, 60.0)
                rho = count / area if area > 0 else 0
                
                if rho > self.STAMPEDE_DENSITY:
                    deaths = count * self.DEATH_RATE
                    new_population[cid] -= deaths
                    total_deaths_this_step += deaths
                    # Track location
                    self.casualties_per_cell[cid] = self.casualties_per_cell.get(cid, 0) + deaths
        
        self.casualties += total_deaths_this_step

        self.casualties += total_deaths_this_step

        # 1.5 HAZARD-AWARE REROUTE CHECK
        # --------------------------------------------------------
        max_occupied_penalty = 0.0
        
        for cid, count in new_population.items():
            if count > 0:
                area = self.cell_areas.get(cid, 60.0)
                rho = count / area
                fire_val = self.penalties[cid]['fire']
                
                score = self.calculate_composite_score(cid, rho, fire_val)
                if score > max_occupied_penalty:
                    max_occupied_penalty = score
        
        self.current_max_penalty = max_occupied_penalty
        
        if (max_occupied_penalty > self.REROUTE_THRESHOLD) and \
           (self.time_step - self.last_reroute_time > self.REROUTE_COOLDOWN):
            
            # UPDATE GRAPH WEIGHTS
            for u, v, data in self.graph.edges(data=True):
                p_u = self.penalties[u]
                p_v = self.penalties[v]
                
                pop_u = new_population.get(u, 0)
                pop_v = new_population.get(v, 0)
                area_u = self.cell_areas.get(u, 60.0)
                rho_u = pop_u / area_u if area_u > 0 else 0.0
                
                area_v = self.cell_areas.get(v, 60.0)
                rho_v = pop_v / area_v if area_v > 0 else 0.0
                
                score_u = self.calculate_composite_score(u, rho_u, p_u['fire'])
                score_v = self.calculate_composite_score(v, rho_v, p_v['fire'])
                
                edge_penalty = max(score_u, score_v)
                
                # Base Distance (Geometry)
                base_dist = data.get('weight_original', data['weight'])
                if 'weight_original' not in data:
                    data['weight_original'] = base_dist
                
                new_weight = base_dist * (1.0 + (edge_penalty * self.HAZARD_COST_MULT))
                
                self.graph[u][v]['weight'] = new_weight
            
            # RE-RUN DIJKSTRA (LOCAL REPAIR)
            # Find the center of the hazard (cell with max penalty)
            best_center = None
            best_score = -1
            
            for cid in self.graph.nodes:
                p_dict = self.penalties[cid]
                count = new_population.get(cid, 0)
                area = self.cell_areas.get(cid, 60.0)
                rho = count / area if area > 0 else 0.0
                s = self.calculate_composite_score(cid, rho, p_dict['fire'])
                if s > best_score:
                    best_score = s
                    best_center = cid
            
            if best_center is not None and best_score > 0.3:
                # LOCAL REPAIR via ANCHORS
                RADIUS = 500.0 
                
                target_field = self.dijkstra_distances
                
                updates = Dijkstra.calculate_dijkstra_repair(
                    self.graph, best_center, RADIUS, target_field
                )
                
                # Apply updates
                for n, new_d in updates.items():
                    target_field[n] = new_d
                
            else:
                # Fallback to Global
                initial_costs = {}
                for sz in self.safe_zone_cells:
                    d = self.dist_to_exit.get(sz, float('inf'))
                    initial_costs[sz] = d
                    
                if self.safe_zone_cells:
                    self.dijkstra_distances, _ = Dijkstra.calculate_dijkstra_field(
                        self.graph, self.safe_zone_cells, initial_costs=initial_costs
                    )
            
            # Re-derive directions (PARTIAL UPDATE OPTIMIZATION)
            nodes_to_update = self.graph.nodes
            if best_center is not None:
                # Optimized: Only update directions for nodes in Radius + Buffer
                nodes_to_update = []
                c_pt = self.graph.nodes[best_center]['geometry'].centroid
                UP_RAD = RADIUS + 20.0
                for n in self.graph.nodes:
                    if self.graph.nodes[n]['geometry'].centroid.distance(c_pt) <= UP_RAD:
                        nodes_to_update.append(n)
            
            for node in nodes_to_update:
                if node in self.exits_ids:
                    self.directions[node] = None
                    continue
                
                d_safe = self.dijkstra_distances.get(node, float('inf'))
                source_field = self.dijkstra_distances if d_safe != float('inf') else self.dist_to_exit
                
                current_best = source_field.get(node, float('inf'))
                best_neighbor = None
                
                for neighbor in self.graph.neighbors(node):
                    d = source_field.get(neighbor, float('inf'))
                    # Dijkstra field ALREADY accounts for weight.
                    if d < current_best:
                        current_best = d
                        best_neighbor = neighbor
                
                self.directions[node] = best_neighbor

            # Re-apply Safe Zone Overrides
            if hasattr(self, 'safe_path_nodes'):
                for path in self.safe_path_nodes:
                    for i in range(len(path) - 1):
                         self.directions[path[i]] = path[i+1]
                         
            self.last_reroute_time = self.time_step

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

    def run(self, steps=100):
        print(f"Starting simulation for {steps} steps...")
        # Note: History reset is handled in initialize_population now
        
        # Randomize Hazard Start
        import random
        fire_start_time = random.randint(30, 40)
        fire_locs = []
        
        # Pick 2 Random Cells explicitly NOT near Exts
        valid_candidates = []
        for cid in self.graph.nodes:
            # Check dist to nearest exit
            # We can use our computed dist_to_exit dict
            d = self.dist_to_exit.get(cid, 0)
            if d > 60.0: # At least 60m away from exit
                valid_candidates.append(cid)
        
        if valid_candidates:
             # Pick ONLY 1 start location
             fire_locs = random.sample(valid_candidates, 1)
             print(f"-> Hazard Plan: Fire at Cells {fire_locs} at Step {fire_start_time}")
        else:
             print("!!! WARNING: No valid fire locations found! Fire will NOT start. Check distance map.")

        for t in range(steps):
            # INJECT HAZARD (RANDOMIZED)
            if t == fire_start_time:
                print(f"!!! HAZARD INJECTION: IGINTING FIRE AT {fire_locs} !!!")
                for fid in fire_locs:
                    if fid in self.penalties: 
                        self.penalties[fid]['fire'] = random.uniform(0.7, 1.0)
            
            total = self.step()
            self.history.append(self.population.copy())
            self.casualty_history.append(self.casualties)
            self.per_cell_casualty_history.append(self.casualties_per_cell.copy())
            self.exit_usage_history.append(self.exit_usage.copy())
            
            if t % 50 == 0:
                print(f"Step {t}: Agents: {total:.0f} | Dead: {self.casualties:.0f}")
            if total < 1:
                break
        
        # FINAL REPORT
        print("\n" + "="*40)
        print(f"=== CASUALTY REPORT (Total: {self.casualties:.1f}) ===")
        print("="*40 + "\n")

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

def get_config_from_terminal():
    print("\n=== MCA Simulation (Safe Zone + Dijkstra Iterative) ===")
    
    # Defaults
    d_agents = 5000
    d_steps = 300
    d_iters = 5
    
    # Try parsing CLI args first
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int)
    parser.add_argument('--agents', type=int)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--block', type=int, nargs='*')
    args, unknown = parser.parse_known_args()
    
    if args.iters or args.agents:
        # If CLI args are present, use them and skip interactive input
        return {
            'agents': args.agents if args.agents else d_agents,
            'steps': args.steps if args.steps else d_steps,
            'iterations': args.iters if args.iters else d_iters,
            'block': args.block if args.block else []
        }

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
    gpkg_path = os.path.join(base_dir, "..", "GPKG_Files", themap)
    
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
        mean_flow = df_t['Exit Flow Rate'].mean()
        mean_density = df_t['Density Evolution'].mean()
        mean_velocity = df_t['Velocity of Evacuees'].mean()
        peak_density = df_t['Peak Density'].max()
        
        remaining_agents = df_t['Remaining Evacuees'].iloc[-1]
        total_evacuated = df_e['Exit Usage Distribution'].sum()
        
        total_time = df_sum['Total Evacuation Time'].iloc[0]
        total_casualties = df_sum['Total Casualties'].iloc[0]
        
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
        
        # Store for global averaging
        df_t['Run'] = i + 1
        all_time_metrics.append(df_t)
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
    avg_row['Total Casualties'] = int(avg_row['Total Casualties'])
    avg_row['Total Evacuated'] = int(avg_row['Total Evacuated'])
    avg_row['Remaining Agents'] = int(avg_row['Remaining Agents'])
    
    df_runs_final = pd.concat([df_runs, pd.DataFrame([avg_row])], ignore_index=True)

    # USER REQUEST: Add column titles below the average row
    header_row = {col: col for col in df_runs.columns}
    df_runs_final = pd.concat([df_runs_final, pd.DataFrame([header_row])], ignore_index=True)

    # 2. Average Time Curve
    full_time = pd.concat(all_time_metrics)
    avg_time_curve = full_time.groupby('Time (s)').mean(numeric_only=True).reset_index()
    avg_time_curve['Run'] = 'AVERAGE_CURVE'
    
    # 3. Average Spatial Distribution
    full_spatial = pd.concat(all_spatial_metrics)
    avg_spatial = full_spatial.groupby('Cell ID').mean(numeric_only=True).reset_index()
    avg_spatial = avg_spatial.sort_values(by='Spatial Distribution (Casualties)', ascending=False)
    
    # 4. Average Exit Usage
    full_exits = pd.concat(all_exit_metrics)
    avg_exits = full_exits.groupby('Exit ID').mean(numeric_only=True).reset_index()
    
    # SAVE
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, "simulation_results_dijkstra_2.xlsx")
    
    try:
        # STYLING
        def color_columns(s):
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
                return [c_green] * len(s)
            elif 'Density' in s.name or 'Velocity' in s.name:
                return [c_blue] * len(s)
            elif 'Time' in s.name:
                return [c_yellow] * len(s)
            return [''] * len(s)

        try:
            styled_df = df_runs_final.style.apply(color_columns, axis=0)
            
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                styled_df.to_excel(writer, sheet_name='Run Comparisons', index=False)
                avg_time_curve.to_excel(writer, sheet_name='Avg Time Series', index=False)
                avg_spatial.to_excel(writer, sheet_name='Avg Spatial Dist', index=False)
                avg_exits.to_excel(writer, sheet_name='Avg Exit Usage', index=False)
                
            print(f"✅ AVERAGED Results saved to {filename}")
            
        except ImportError:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                df_runs_final.to_excel(writer, sheet_name='Run Comparisons', index=False)
                avg_time_curve.to_excel(writer, sheet_name='Avg Time Series', index=False)
                avg_spatial.to_excel(writer, sheet_name='Avg Spatial Dist', index=False)
                avg_exits.to_excel(writer, sheet_name='Avg Exit Usage', index=False)
            print(f"✅ AVERAGED Results saved to {filename} (Unstyled)")

    except Exception as e:
        print(f"❌ Failed to save Excel: {e}")

    print("Batch Run Complete.")

if __name__ == "__main__":
    main()
