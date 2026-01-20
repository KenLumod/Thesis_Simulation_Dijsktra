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
        self.STAMPEDE_DENSITY = 3.5 # p/m^2 (Lowered from 4.0) 
        self.DEATH_RATE = 0.1 # 10% per second if overcrowded
        
        # State
        self.population = {} # cell_index -> count
        self.casualties = 0
        self.casualties_per_cell = {} # cell_id -> total_deaths
        self.time_step = 0
        self.time_step = 0
        self.history = []
        self.casualty_history = []
        self.per_cell_casualty_history = [] # LIST of DICTS
        self.penalty_history = []  # NEW: Store composite scores per frame
        self.distance_history = [] # NEW: Store Dijkstra Field per frame (Dynamic)
        
        # Exit Analysis
        
        # Exit Analysis
        self.exit_status = {} # exit_id -> 'OPEN'/'CLOSED'
        self.exit_usage = {}  # exit_id -> count (accumulated)
        self.exit_usage_history = [] # list of dicts
        self.road_to_exit = {} # road_cell_id -> exit_id
        
        # Viz Data
        self.cell_centroids = {}
        self.flow_vectors = {} # cell_idx -> (u, v)
        self.flow_history = [] # List of dicts: time_step -> {(u,v): count}

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
        self.HAZARD_COST_MULT = 50.0   # Cost = Dist * (1 + Score * MULT)
        self.REROUTE_THRESHOLD = 0.7   # Max penalty occupied cell > 0.7 -> Reroute
        self.REROUTE_COOLDOWN = 5      # Steps between reroutes
        self.last_reroute_time = -999  # Allow immediate first visual update
        
        # 3. State
        # {cell_id: {'fire':0.0, 'smoke':0.0, 'debris':0.0, 'terrain':0.0, 'temp':0.0}}
        self.penalties = {} 
        self.current_max_penalty = 0.0 # Track for GUI/Logic
        self.burnt_cells = set() # Track cells that have peaked and are now decaying


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
        print(f"DEBUG: Active Exit Nodes (Source for Dijkstra): {valid_exits}")
        print(f"DEBUG: Original Exit Mapping: {[self.road_to_exit.get(n, 'UNK') for n in valid_exits]}")
        
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
        # ------------------------------------------------------------------------
        # DEBUG: CHECK EXIT CONNECTIVITY
        # ------------------------------------------------------------------------
        print("DEBUG: Checking Exit Connectivity (BFS)...")
        for ex in valid_exits:
            reachable = 0
            # Quick BFS to count component size
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

        # ------------------------------------------------------------------------
        # DEBUG: FORENSIC DISTANCE AUDIT
        # ------------------------------------------------------------------------
        print("DEBUG: Running Forensic Distance Audit (Safe Zone -> Per Exit)...")
        # Group exits by FID
        fid_exits = {}
        for ex in valid_exits:
            fid = self.road_to_exit.get(ex, 'UNK')
            if fid not in fid_exits: fid_exits[fid] = []
            fid_exits[fid].append(ex)
        
        # Calculate individual fields
        sz_audit = {sz: {} for sz in valid_safe}
        
        for fid, nodes in fid_exits.items():
            # Run Dijkstra for this SPECIFIC Exit Group
            # Note: We run on the FULL graph to measure true distance
            d_field, _ = Dijkstra.calculate_dijkstra_field(self.graph, nodes)
            for sz in valid_safe:
                dist = d_field.get(sz, float('inf'))
                sz_audit[sz][fid] = dist
                
        # Print Table
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

        # ------------------------------------------------------------------------
        # ------------------------------------------------------------------------
        # STAGE 1: Dijkstra from EXITS (Multi-Field: Per Exit Group)
        # ------------------------------------------------------------------------
        print("Stage 1: Computing Separate Dijkstra Fields for Each Exit...")
        
        # 1. Group Exits by FID
        fid_exits = {}
        for ex in valid_exits:
            fid = self.road_to_exit.get(ex, 'UNK')
            if fid not in fid_exits: fid_exits[fid] = []
            fid_exits[fid].append(ex)
            
        # 2. Calculate Field for each FID
        self.exit_fields = {} # {fid: {'distances': d, 'visited': v}}
        
        for fid, nodes in fid_exits.items():
            print(f"  > Computing Field for Exit FID {fid} (Nodes: {nodes})...")
            d, v = Dijkstra.calculate_dijkstra_field(self.graph, nodes)
            self.exit_fields[fid] = {'distances': d, 'visited': v}
            
        # 3. Create Composite "Nearest Exit" Field (for general visualization logic)
        self.dist_to_exit = {}
        all_visited = [] # Flattened for Phase 1 animation if needed
        
        # Merge visited for animation (interleaved?)
        # Let's just flatten them for simplicity
        for fid in self.exit_fields:
            all_visited.extend(self.exit_fields[fid]['visited'])
            
            d_map = self.exit_fields[fid]['distances']
            for n, dist in d_map.items():
                cur = self.dist_to_exit.get(n, float('inf'))
                if dist < cur:
                    self.dist_to_exit[n] = dist
        
        raw_visited_1 = all_visited # Keep compatible var name

        # ------------------------------------------------------------------------
        # TRACE PATHS & ASSIGN SAFE ZONES (FORCED LOAD BALANCING)
        # ------------------------------------------------------------------------
        self.safe_paths = []
        self.safe_path_nodes = []
        print("\n=== SAFE ZONE ASSIGNMENT (ROUND ROBIN BALANCING) ===")
        
        # Keep track of assignments for balancing
        exit_counts = {fid: 0 for fid in fid_exits}
        available_fids = sorted(list(fid_exits.keys())) # Ensure deterministic order
        
        for i, sz in enumerate(valid_safe):
            # Round Robin Selection: Cycle through available FIDs
            # But only accept if reachable (dist != inf)
            
            assigned_fid = None
            assigned_dist = float('inf')
            
            # Try to assign to the "Next" FID in sequence
            start_index = i % len(available_fids)
            
            # Check FIDs starting from the round-robin index, wrapping around
            for offset in range(len(available_fids)):
                idx = (start_index + offset) % len(available_fids)
                fid = available_fids[idx]
                
                data = self.exit_fields[fid]
                d = data['distances'].get(sz, float('inf'))
                
                if d != float('inf'):
                    assigned_fid = fid
                    assigned_dist = d
                    break # Found a valid exit for this turn
            
            if assigned_fid is None:
                print(f"  ⚠️ SZ {sz} -> Unreachable from ANY exit!")
                continue
                
            best_fid = assigned_fid # Use the forced assignment
            best_dist = assigned_dist
            
            exit_counts[best_fid] += 1
            
            # TRACE PATH using the SPECIFIC FIELD of best_fid
            target_field = self.exit_fields[best_fid]['distances']
            target_exits = fid_exits[best_fid]
            
            path = [sz]
            curr = sz
            dist = best_dist
            
            # Gradient Descent on TARGET FIELD
            for _ in range(300): # Increased limit for longer paths
                if curr in target_exits: break
                
                best_n = None
                best_d = dist
                
                # Hill climbing logic (Gradient Descent)
                # Note: Since we are forcing suboptimal exits, the path might be long.
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
            
            # Store Valid Path
            if len(path) > 1 and path[-1] in target_exits:
                 self.safe_path_nodes.append(path)
                 
                 # visual
                 coords = []
                 for n in path:
                     if n in self.cell_centroids:
                         pt = self.cell_centroids[n]
                         coords.append((pt.x, pt.y))
                 self.safe_paths.append(coords)
                 print(f"  SZ {sz} -> FORCED to Exit FID {best_fid} (Dist: {best_dist:.1f}m)")
            else:
                 print(f"  ⚠️ SZ {sz} -> Path Trace Failed for FID {best_fid}")

        print(f"Assignment Summary: {exit_counts}")
        
        # ------------------------------------------------------------------------
        # STAGE 2: Dijkstra from SAFE ZONES (Initialized with Exit Dist)
        # ------------------------------------------------------------------------
        print("Stage 2: Computing Path via Safe Zones...")
        
        # Prepare Initial Costs for Safe Zones
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
            raw_visited_2 = []
        else:
            # Run Dijkstra from SAFE ZONES, carrying over the cost to exit
            self.dijkstra_distances, raw_visited_2 = Dijkstra.calculate_dijkstra_field(
                self.graph, valid_safe, initial_costs=initial_costs
            )

        # ------------------------------------------------------------------------
        # BUILD ANIMATION HISTORY (Sequential Stages)
        # ------------------------------------------------------------------------
        # Map IDs to Source FID for Coloring
        self.node_source_fid = {}
        for fid, data in self.exit_fields.items():
             for n, d in data['distances'].items():
                 cur = self.dist_to_exit.get(n, float('inf'))
                 if abs(d - cur) < 1e-4: self.node_source_fid[n] = fid

        # Define Colors (HSV)
        import matplotlib.colors as mcolors
        unique_fids = sorted(self.exit_fields.keys())
        # Assign hues:
        fid_hues = {fid: (i * (1.0/max(1, len(unique_fids)))) % 1.0 for i, fid in enumerate(unique_fids)}

        self.dijkstra_history = []
        chunk_size = 20
        
        # Phase A: Exit Field Wave (Multi-Color)
        visited_1 = set()
        for i in range(0, len(raw_visited_1), chunk_size):
            chunk = raw_visited_1[i : i+chunk_size]
            for node, dist in chunk: visited_1.add(node)
            
            # Compute Color Array for this Frame
            fcolors = []
            for _, row in self.road_cells.iterrows():
                cid = row['id']
                if cid in visited_1:
                    fid = self.node_source_fid.get(cid, 0)
                    hue = fid_hues.get(fid, 0.0)
                    dist = self.dist_to_exit.get(cid, 0)
                    # Intensity fades with distance (Value)
                    # Max dist is often ~300m. 
                    val = max(0.4, 1.0 - (dist / 250.0)) 
                    fcolors.append(mcolors.hsv_to_rgb((hue, 0.9, val)))
                else:
                    fcolors.append((1,1,1,1)) # White background
            
            self.dijkstra_history.append({
                'visited': visited_1.copy(),
                'phase': 'EXIT_FIELD',
                'distances': self.dist_to_exit,
                'facecolors': fcolors
            })
            
        # Optional Pause Frame
        for _ in range(5):
             self.dijkstra_history.append({
                'visited': visited_1.copy(),
                'phase': 'EXIT_FIELD',
                'distances': self.dist_to_exit
            })
            
        # Phase B: Safe Zone Field Wave (Magma/Hot)
        # Start fresh or overlay? User wants to see "connecting".
        # Let's start fresh to show the new gradient source.
        visited_2 = set()
        for i in range(0, len(raw_visited_2), chunk_size):
            chunk = raw_visited_2[i : i+chunk_size]
            for node, dist in chunk: visited_2.add(node)
            self.dijkstra_history.append({
                'visited': visited_2.copy(),
                'phase': 'SAFE_FIELD',
                'distances': self.dijkstra_distances
            })
        
        # Ensure final frame exists
        if not self.dijkstra_history: 
             self.dijkstra_history.append({'visited': set(), 'phase':'SAFE_FIELD', 'distances':self.dijkstra_distances})
        
        # Derive Flow Directions (Gradient Descent on FINAL field)
        self.directions = {}
        for node in self.graph.nodes:
            # If node is an Exit, it has no target (sink)
            if node in exit_indices:
                self.directions[node] = None
                continue
                
            # HYBRID NAVIGATION: Prefer Safe Zone, Fallback to Direct Exit
            # This handles cases where a road network is isolated from Safe Zones but has an Exit.
            d_safe = self.dijkstra_distances.get(node, float('inf'))
            
            if d_safe != float('inf'):
                 source_field = self.dijkstra_distances
            else:
                 # Override: Use Exit Field if Safe Zone is unreachable
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
        
        # ------------------------------------------------------------------------
        # OVERRIDE: BRIDGE SAFE ZONES TO EXITS
        # ------------------------------------------------------------------------
        # Force agents on the Safe Zone Path to flow towards the Exit.
        if hasattr(self, 'safe_path_nodes'):
            print("Applying Safe Zone Path Overrides...")
            override_count = 0
            for path in self.safe_path_nodes:
                # Path: [SZ, n1, n2, ..., Exit]
                # We want directions: SZ->n1, n1->n2, ..., PreExit->Exit
                for i in range(len(path) - 1):
                    u_node = path[i]
                    v_node = path[i+1]
                    # Enforce direction u -> v
                    if self.directions.get(u_node) != v_node:
                         self.directions[u_node] = v_node
                         override_count += 1
            print(f"Overridden {override_count} direction vectors to enforce Safe Zone -> Exit flow.")

        # Compute Stable Flow Vectors for Visualization (Baseline Logic)
        self.flow_vectors = {}
        self.global_flow_vectors = {}
        
        for idx in self.graph.nodes:
            target_idx = self.directions.get(idx)
            dx, dy = 0, 0
            if target_idx is not None and target_idx in self.cell_centroids:
                if idx in self.cell_centroids:
                    start = self.cell_centroids[idx]
                    end = self.cell_centroids[target_idx]
                    dx_raw = end.x - start.x
                    dy_raw = end.y - start.y
                    norm = np.hypot(dx_raw, dy_raw)
                    if norm > 0:
                        dx = dx_raw / norm
                        dy = dy_raw / norm
            
            self.flow_vectors[idx] = (dx, dy)
            self.global_flow_vectors[idx] = (dx, dy)
    
    def initialize_population(self, total_agents=5000):
        self.total_agents_init = total_agents
        print(f"Initializing {total_agents} agents...")
        nodes = list(self.graph.nodes)
        
        # Prefer spawning near 'spawns' (BASELINE LOGIC PORT)
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
                 print("  Warning: No road cells found intersecting spawn buffer. Randomizing.")
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


    def calculate_composite_score(self, cid, rho, fire_val):
        """
        Computes clamped composite penalty score for a cell.
        S = min(0.95, weighted_sum)
        """
        p = self.penalties.get(cid)
        if not p: return 0.0

        # Note: fire_val passed in to ensure sync with external fire logic if any
        # For now, we assume self.penalties['fire'] is updated via update_penalties or visually
        
        # 1. Normalize Components to [0, 1]
        # Density: > 4.0 is panic (1.0), 0 is 0.
        val_density = min(1.0, rho / self.STAMPEDE_DENSITY) if self.STAMPEDE_DENSITY > 0 else 0
        
        val_fire = min(1.0, p['fire'])
        val_smoke = min(1.0, p['smoke'])
        val_debris = min(1.0, p['debris'])
        val_terrain = min(1.0, p['terrain'])
        val_temp = min(1.0, p['temp'])
        
        # 2. Weighted Sum
        w_sum = (val_fire * self.W_FIRE) + \
                (val_smoke * self.W_SMOKE) + \
                (val_density * self.W_DENSITY) + \
                (val_debris * self.W_DEBRIS) + \
                (val_terrain * self.W_TERRAIN) + \
                (val_temp * self.W_TEMP_OBJ)
                
        # 3. Clamp Final Score
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
                             self.penalties[n]['fire'] = 0.2 # Start VERY small (needs ~10 steps to become dangerous)
                             # Also add smoke
                             self.penalties[n]['smoke'] = 0.6
                             
    def trigger_random_events(self):
        """
        Simulate secondary disasters (Collapse, Obstructions)
        """
        import random
        # 0.5% chance per second for a random collapse somewhere
        if random.random() < 0.005:
            # Pick a random victim cell
            candidates = list(self.graph.nodes)
            if not candidates: return
            
            target = random.choice(candidates)
            
            # Immunity for Safe Zones / Exits (Don't block the finish line)
            if hasattr(self, 'safe_zone_cells') and target in self.safe_zone_cells:
                return
            if self.road_to_exit.get(target) is not None:
                return
                
            print(f"!!! EVENT: STRUCTURAL COLLAPSE at Cell {target} !!!")
            # Set high penalties
            self.penalties[target]['debris'] = 0.9
            self.penalties[target]['temp'] = 0.8 # Obstruction
            # (These will decay naturally via update_penalties)

    def step(self):
        # 0. Check for New Random Events
        self.trigger_random_events()
        
        # 1. Update Dynamics (Decay, Fire Spread)
        self.update_penalties()


        new_population = self.population.copy()
        current_step_flows = {} # (u, v) -> count
        total_deaths_this_step = 0
        
        # 1. Check for Stampedes
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
                    self.casualties_per_cell[cid] = self.casualties_per_cell.get(cid, 0) + deaths
        
        self.casualties += total_deaths_this_step
        
        # 1.5 HAZARD-AWARE REROUTE CHECK
        # --------------------------------------------------------
        # Optimization: Check max penalty ONLY for occupied cells
        max_occupied_penalty = 0.0
        
        # We need density for calculation
        for cid, count in new_population.items():
            if count > 0:
                area = self.cell_areas.get(cid, 60.0)
                rho = count / area
                
                # Calculate current score
                # Assume fire is manual for now or in p_dict
                fire_val = self.penalties[cid]['fire']
                
                score = self.calculate_composite_score(cid, rho, fire_val)
                if score > max_occupied_penalty:
                    max_occupied_penalty = score
        
        self.current_max_penalty = max_occupied_penalty
        
        # TRIGGER: Threshold + Cooldown
        if (max_occupied_penalty > self.REROUTE_THRESHOLD) and \
           (self.time_step - self.last_reroute_time > self.REROUTE_COOLDOWN):
            
            print(f"REROUTE TRIGGERED (Step {self.time_step}): Max Penalty {max_occupied_penalty:.2f} > {self.REROUTE_THRESHOLD}")
            
            # UPDATE GRAPH WEIGHTS
            for u, v, data in self.graph.edges(data=True):
                # We need to penalize the TARGET node (v) usually in directed, 
                # but this is undirected graph used for pathing.
                # Average penalty of u and v? Or specific?
                # Dijkstra.py uses 'weight'.
                
                # Conservative: Max penalty of u or v
                p_u = self.penalties[u]
                p_v = self.penalties[v]
                
                # Get simplistic density for weight calc (using prev step pop)
                # Get simplistic density for weight calc (using prev step pop)
                pop_u = new_population.get(u, 0)
                pop_v = new_population.get(v, 0)
                
                area_u = self.cell_areas.get(u, 60.0)
                rho_u = pop_u / area_u if area_u > 0 else 0.0
                
                area_v = self.cell_areas.get(v, 60.0)
                rho_v = pop_v / area_v if area_v > 0 else 0.0
                
                score_u = self.calculate_composite_score(u, rho_u, p_u['fire'])
                score_v = self.calculate_composite_score(v, rho_v, p_v['fire'])
                
                # Edge penalty is max of nodes
                edge_penalty = max(score_u, score_v)
                
                # Base Distance (Geometry)
                base_dist = data.get('weight_original', data['weight'])
                # Store original once
                if 'weight_original' not in data:
                    data['weight_original'] = base_dist
                
                # FORMULA: weight = dist * (1 + Score * MULT)
                new_weight = base_dist * (1.0 + (edge_penalty * self.HAZARD_COST_MULT))
                
                self.graph[u][v]['weight'] = new_weight
            
            # RE-RUN DIJKSTRA (LOCAL REPAIR)
            # Find the center of the hazard (cell with max penalty)
            # (We already found current_max_penalty, but let's finding the ID)
            hazard_center = None
            max_p = -1
            for cid in new_population:
                if count > 0: # Check only occupied? Or all? 
                    # We need the source of the penalty, which might be empty now but was high.
                    # Let's iterate self.penalties for the true source
                    pass 
            
            # Better: Iterate all penalties to find the "Epicenter"
            # (Simplification: Use the node with highest combined score)
            best_center = None
            best_score = -1
            
            # Optimization: Only check nodes with non-zero dynamic penalties
            # (Fire/Density/Smoke).
            # If we don't have a specific center, we can fall back to global recompute.
            # But the logic implies we found a max_occupied_penalty > 0.7.
            # So let's find that specific cell again.
            
            for cid in self.graph.nodes:
                p_dict = self.penalties[cid]
                # Quick score check
                # We need rho.
                count = new_population.get(cid, 0)
                area = self.cell_areas.get(cid, 60.0)
                rho = count / area if area > 0 else 0.0
                s = self.calculate_composite_score(cid, rho, p_dict['fire'])
                if s > best_score:
                    best_score = s
                    best_center = cid
            
            if best_center is not None and best_score > 0.3:
                # LOCAL REPAIR via ANCHORS
                # Radius? Increased to 500m to catch distant junctions
                RADIUS = 500.0 
                # Identify which Field to repair? 
                # Stage 2 uses 'self.dijkstra_distances' (from Safe Zones)
                # If d_safe is INF, it uses 'self.dist_to_exit' (fallback).
                # We mainly repair 'self.dijkstra_distances' (Safe Zone Flow).
                
                print(f"  > Running Anchor-Based Local Repair around Cell {best_center} (R={RADIUS}m)...")
                
                # We need the current dist map. 
                # NOTE: self.dijkstra_distances might be incomplete if fallback was used.
                # Let's ensure we are repairing the active field.
                
                target_field = self.dijkstra_distances
                
                updates = Dijkstra.calculate_dijkstra_repair(
                    self.graph, best_center, RADIUS, target_field,
                    sources=self.safe_zone_cells  # CRITICAL: Re-seed Safe Zones if inside radius
                )
                
                # Apply updates
                for n, new_d in updates.items():
                    target_field[n] = new_d
                    
                print(f"  > Local Repair Complete. Updated {len(updates)} nodes.")
                
            else:
                # Fallback to Global if no clear center (shouldn't happen with trigger)
                print("  > Warning: No clear hazard center. Running Global Recompute.")
                initial_costs = {}
                for sz in self.safe_zone_cells:
                    d = self.dist_to_exit.get(sz, float('inf'))
                    initial_costs[sz] = d
                    
                if self.safe_zone_cells:
                    self.dijkstra_distances, _ = Dijkstra.calculate_dijkstra_field(
                        self.graph, self.safe_zone_cells, initial_costs=initial_costs
                    )
            
            # Re-derive directions (PARTIAL UPDATE OPTIMIZATION)
            # Only update nodes that were in the 'updates' set (plus neighbors?)
            # Conservative: Update all, or update 'updates'.
            # If we only update 'updates', we might miss neighbors flowing INTO updates.
            # Safe Strategy: Update all nodes in affected zone + Rim.
            
            # For simplicity in this step, let's just re-derive directions for ALL 
            # (Calculation is cheap compared to Dijkstra).
            # Or use the Radius again.
            
            nodes_to_update = self.graph.nodes
            if best_center is not None:
                # Optimized: Only update directions for nodes in Radius + Buffer
                nodes_to_update = []
                c_pt = self.graph.nodes[best_center]['geometry'].centroid
                UP_RAD = RADIUS + 20.0 # Buffer
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


            # Re-apply Safe Zone Overrides (They are static corridors)
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
            
            # Base Speed (Density Dependent)
            v_i = self.V_FREE * np.exp(-rho_i / self.RHO_MAX)
            
            # --- HAZARD IMPEDANCE ---
            # Calculate Environmental Penalty (Fire + Smoke + Debris + Obstruction)
            # We ignore rho here because 'v_i' already accounts for density above.
            p_dict = self.penalties.get(cid, {})
            env_penalty = p_dict.get('fire', 0) + p_dict.get('debris', 0) + p_dict.get('smoke', 0)*0.5 + p_dict.get('temp_obj', 0)
            
            # Impedance Factor: Speed *= (1.0 - (Penalty * 0.55))
            # User Request: "Do like 55%" (Interpret as 55% Slowdown Max)
            # If Penalty=1.0, Speed=45% (They can squeeze through)
            impedance = max(0.1, 1.0 - (env_penalty * 0.55))
            
            # Removed Hard Blockage logic (User found it too deadly)
            
            v_i *= impedance
            
            q_out = rho_i * v_i * self.CELL_WIDTH * self.DT
            
            target_cap = self.max_capacities.get(target_id, 300.0)
            pop_target = new_population[target_id]
            available_capacity = target_cap - pop_target
            
            actual_flow = min(q_out, available_capacity)
            actual_flow = max(0, actual_flow)
            actual_flow = min(actual_flow, current_pop)
            
            new_population[cid] -= actual_flow
            new_population[target_id] += actual_flow
            
            # Log Flow
            if actual_flow > 0:
                current_step_flows[(cid, target_id)] = current_step_flows.get((cid, target_id), 0) + actual_flow

        self.population = new_population
        self.flow_history.append(current_step_flows)
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
            
            # Add Penalty Stats if available
            if hasattr(self, 'penalty_history') and t < len(self.penalty_history):
                # penalty_history[t] is {cid: composite_score}
                # Filter for non-zero entries to show actual hazard presence
                p_scores = [v for v in self.penalty_history[t].values() if v > 0.0]
                
                time_data[-1]['Active Hazard Cells'] = len(p_scores)
                time_data[-1]['Total Hazard Intensity'] = sum(p_scores)
            else:
                time_data[-1]['Active Hazard Cells'] = 0
                time_data[-1]['Total Hazard Intensity'] = 0.0

        df_time = pd.DataFrame(time_data)
        
        # 5. Flow Logs (New Sheet)
        # Convert list of dicts to DataFrame
        flow_records = []
        for t, flows in enumerate(self.flow_history):
            for (u, v), count in flows.items():
                flow_records.append({
                    'Step': t,
                    'Time (s)': t * self.DT,
                    'From Cell': u,
                    'To Cell': v,
                    'Agent Count': float(f"{count:.2f}")
                })
        df_flow = pd.DataFrame(flow_records)
        
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
                    df_flow.to_excel(writer, sheet_name='Node Flow Logs', index=False) 
                    df_summary.style.apply(color_columns, axis=0).to_excel(writer, sheet_name='Summary', index=False)
                    print(f"✅ Simulation results saved to {filename} (Colored)")
                except Exception:
                    df_time.to_excel(writer, sheet_name='Time Metrics', index=False)
                    df_cells.to_excel(writer, sheet_name='Spatial Distribution', index=False)
                    df_exits.to_excel(writer, sheet_name='Exit Usage', index=False)
                    df_flow.to_excel(writer, sheet_name='Node Flow Logs', index=False)
                    df_summary.to_excel(writer, sheet_name='Summary', index=False)
                    print(f"✅ Simulation results saved to {filename} (Unstyled)")

        except Exception as e:
            print(f"❌ Failed to save Excel: {e}")

    def run(self, steps=100):
        print(f"Starting simulation for {steps} steps...")
        
        # Record Initial State (Step 0)
        # We need initial penalties/distances
        initial_penalties = {}
        for cid in self.graph.nodes:
            # Re-calc initial scores
            count = self.population.get(cid, 0)
            area = self.cell_areas.get(cid, 60.0)
            rho = count / area if area > 0 else 0.0
            p_dict = self.penalties[cid]
            initial_penalties[cid] = self.calculate_composite_score(cid, rho, p_dict['fire'])
            
        self.penalty_history.append(initial_penalties)
        self.distance_history.append(self.dijkstra_distances.copy())
        
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
        
        print(f"DEBUG: Found {len(valid_candidates)} potential fire locations (>60m from exits).")

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
            
            # ERA: Record Dynamic Fields
            # 1. Penalties (HAZARD ONLY for Viz - Exclude Density)
            current_penalties = {}
            for cid in self.graph.nodes:
                p_dict = self.penalties[cid]
                # Pass rho=0 to ignore density component in the score for visualization
                # This ensures we see FIRE/SMOKE/DEBRIS, not just crowd flow.
                current_penalties[cid] = self.calculate_composite_score(cid, 0.0, p_dict['fire'])
            self.penalty_history.append(current_penalties)
            
            # 2. Distances (Only changed if reroute happened, but store ref or copy?)
            # Copy is safer for replay. Memory heavy? 
            # 500 steps * 300 cells * float = Small.
            self.distance_history.append(self.dijkstra_distances.copy())
            
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
             # Match Dijkstra Pure Style: Solid Blue, zorder=5
             self.spawns.plot(ax=ax, color='blue', marker='o', markersize=30, label='Spawn Points', zorder=5)
        if self.safe_zones is not None:
             self.safe_zones.plot(ax=ax, color='lime', marker='*', markersize=150, edgecolor='black', linewidth=1, label='Safe Zones', zorder=9)
        if self.exits is not None:
             self.exits.plot(ax=ax, color='red', marker='X', markersize=150, edgecolor='black', linewidth=2, label='Exits', zorder=10)
        
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
        
        # Visualization for Safe Zone Connections (Yellow Dashed Lines)
        from matplotlib.collections import LineCollection
        self.lc_safe_paths = LineCollection([], colors='yellow', linestyles='dashed', linewidths=2.5, zorder=20, alpha=0.8)
        ax.add_collection(self.lc_safe_paths) 
        
        # Visualization for Casualties (Markers)
        # We use a scatter plot to show 'X' marks where deaths occur
        self.scat_casualties = ax.scatter([], [], marker='x', s=100, color='red', linewidth=2.0, zorder=30) 
        # NEW: Scatter for Penalties (Orange Squares)
        self.scat_penalties = ax.scatter([], [], marker='s', s=120, color='orange', alpha=0.6, zorder=35, label='Hazards') 

        def update_selection_text(frame_idx):
            # Correct for Phase 1 Offset
            # Correct for Phase 1 Offset
            sim_frame = frame_idx
            if sim_frame >= len(self.history):
                 sim_frame = len(self.history) - 1

            if self.selected_exit_id is not None:
                eid = self.selected_exit_id
                status = self.exit_status.get(eid, 'UNKNOWN')
                
                count = 0
                if sim_frame < len(self.exit_usage_history):
                    usage_dict = self.exit_usage_history[sim_frame]
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

            # PHASE 2: Adjust Frame (Already computed as sim_frame at start)
            
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
                
                selected_cell_text.set_text(
                    f"Cell ID: {cid}\n"
                    f"Pop: {p:.0f} / Cap: {c:.0f}\n"
                    f"Occupancy: {ratio*100:.1f}%\n"
                    f"Status: {status}"
                )
            elif self.view_mode == 'dijkstra':
                # Show Dynamic Dist
                d_map = self.dijkstra_distances 
                if sim_frame < len(self.distance_history):
                    d_map = self.distance_history[sim_frame]
                
                d = d_map.get(cid, float('inf'))
                d_str = f"{d:.1f}" if d != float('inf') else "Unreachable"
                selected_cell_text.set_text(
                     f"Cell ID: {cid}\n"
                     f"Exit Dist: {d_str}m\n"
                     f"(Dynamic @ T={sim_frame})"
                )
            elif self.view_mode == 'penalties':
                # Show Penalty Score
                score = 0
                if sim_frame < len(self.penalty_history):
                    score = self.penalty_history[sim_frame].get(cid, 0)
                
                selected_cell_text.set_text(
                    f"Cell ID: {cid}\n"
                    f"Penalty Score: {score:.3f}\n"
                    f"(Max 0.95)"
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
                phase = step_data.get('phase', 'UNKNOWN')
                active_dists = step_data.get('distances', self.dijkstra_distances)
                
                # Dynamic Colormap & Title
                cmap_name = 'magma'
                title_text = "Phase 1: Dijkstra Map Formation"
                
                # Normalize based on Active Map MAX (Calculated BEFORE branching)
                max_d = 0
                for d in active_dists.values():
                     if d != float('inf'): max_d = max(max_d, d)

                if 'facecolors' in step_data:
                     # Multi-Color Mode (Phase 1A)
                     title_text = "Phase 1A: Computing Exit Field (Colored by Source Exit)"
                     collection.set_array(None) 
                     collection.set_facecolors(step_data['facecolors'])
                else:
                    if phase == 'EXIT_FIELD':
                        cmap_name = 'summer' # Greenish
                        title_text = "Phase 1A: Computing Exit Field (Exits -> Safe Zones)"
                    elif phase == 'SAFE_FIELD':
                        cmap_name = 'autumn_r' # Red-Orange
                        title_text = "Phase 1B: Computing Safe Zone Field (Safe Zones -> Agents)"
                    
                    ratios = []
                    for idx in self.road_cells['id']:
                         if idx in visited_nodes:
                             d = active_dists.get(idx, float('inf'))
                             if d != float('inf') and max_d > 0:
                                 ratios.append(1.0 - (d / max_d)) 
                             else:
                                 ratios.append(0.0)
                         else:
                             ratios.append(0.0)
                    
                    collection.set_array(np.array(ratios))
                    collection.set_cmap(cmap_name) 
                    collection.set_clim(0, 1.0)
                
                info_text.set_text(
                    f"🌊 {phase}\n"
                    f"Nodes Scanned: {len(visited_nodes)}\n"
                    f"Max Dist: {max_d:.1f}m"
                )
                
                ax.set_title(title_text, fontsize=14, fontweight='bold')
                return collection, info_text

            # --- PHASE 2: EVACUATION ---
            if hasattr(self, 'lc_safe_paths') and hasattr(self, 'safe_paths'):
                 if self.view_mode == 'dijkstra':
                     self.lc_safe_paths.set_segments(self.safe_paths)
                 else:
                     self.lc_safe_paths.set_segments([])

            # Correct frame mapping
            sim_frame = frame
            
            # SUBTRACT Dijkstra Pre-Roll Frames
            if hasattr(self, 'dijkstra_history'):
                 sim_frame = frame - len(self.dijkstra_history)

            # Safety clamps
            if sim_frame < 0: sim_frame = 0
            if sim_frame >= len(self.history):
                sim_frame = len(self.history) - 1

            pop_data = self.history[sim_frame]
            casualties = self.casualty_history[sim_frame]
            
            # --- VIEW MODES ---
            
            # 1. OCCUPANCY (Default)
            if self.view_mode == 'occupancy':
                 ax.set_title("Phase 2: Evacuation (Occupancy)", fontsize=14, fontweight='bold')
                 # Reset Scatter visibility
                 self.scat_casualties.set_visible(False)
                 self.scat_penalties.set_visible(False)

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
                 
                 # Get Dynamic Field for this frame
                 d_map = self.dijkstra_distances # Default
                 if hasattr(self, 'distance_history') and sim_frame < len(self.distance_history):
                     d_map = self.distance_history[sim_frame]
                 
                 for idx, count in pop_data.items():
                     if count > 1.0:
                          centroid = self.cell_centroids.get(idx)
                          # FIX: Use Pre-Computed Stable Flow Vectors
                          vec = self.flow_vectors.get(idx)
                          if vec and vec != (0,0):
                             xq.append(centroid.x)
                             yq.append(centroid.y)
                             uq.append(vec[0])
                             vq.append(vec[1])
                               
                 if xq:
                     self.quiver = ax.quiver(xq, yq, uq, vq, scale=30, width=0.003, color='black', alpha=0.6, zorder=6)
                 else:
                     self.quiver = None
 
            # 3. STATIC/DYNAMIC DIJKSTRA FIELD
            elif self.view_mode == 'dijkstra':
                 ax.set_title(f"Dynamic Dijkstra Field (Time: {sim_frame}s)", fontsize=14, fontweight='bold')
                 self.scat_casualties.set_visible(False)
                 self.scat_penalties.set_visible(False)
                 
                 # Fetch Historical Map
                 d_map = self.dijkstra_distances
                 if sim_frame is not None and sim_frame < len(self.distance_history):
                     d_map = self.distance_history[sim_frame]
                 
                 # Show Distance Map
                 d_vals = []
                 max_d = 0
                 for idx in self.road_cells['id']:
                      d = d_map.get(idx, float('inf'))
                      if d != float('inf'): max_d = max(max_d, d)
                      d_vals.append(d)
                 
                 # Normalize (Inverse)
                 norm_vals = []
                 for d in d_vals:
                     if d == float('inf'): norm_vals.append(0)
                     else: 
                         val = 1.0 - (d / max_d) if max_d > 0 else 0
                         norm_vals.append(val)
                 
                 collection.set_array(np.array(norm_vals))
                 
                 # COLOR CYCLING: Change theme every 100s to show "New Phase"
                 cycle_idx = int(sim_frame // 100)
                 cmaps = ['magma', 'viridis', 'plasma', 'cividis', 'inferno']
                 acc_cmap = cmaps[cycle_idx % len(cmaps)]
                 
                 collection.set_cmap(acc_cmap) 
                 collection.set_clim(0, 1.0)
                 collection.changed() # FORCE UPDATE
                 
                 ax.set_title(f"Dynamic Dijkstra Field (Time: {sim_frame}s) | Phase {cycle_idx} ({acc_cmap})", fontsize=14, fontweight='bold')
                 
                 if self.quiver: 
                     self.quiver.remove()
                 
                 # DYNAMIC ARROWS: Compute new Flow Direction based on d_map
                 xq, yq, uq, vq = [], [], [], []
                 
                 # Use Flow Vectors if available, else Dynamic
                 for idx in self.graph.nodes:
                     if idx in self.flow_vectors and self.flow_vectors[idx] != (0,0):
                         c = self.cell_centroids.get(idx)
                         v = self.flow_vectors[idx]
                         xq.append(c.x); yq.append(c.y)
                         uq.append(v[0]); vq.append(v[1])
                 
                 if xq:
                     self.quiver = ax.quiver(xq, yq, uq, vq, scale=30, width=0.003, color='black', alpha=0.6, zorder=6)
                 else:
                     self.quiver = None

            # 4. PENALTIES HEATMAP (NEW)
            # 4. PENALTIES HEATMAP (NEW) -> MODIFIED TO SCATTER MARKERS
            elif self.view_mode == 'penalties':
                 ax.set_title(f"Hazard Penalties (Orange X) (T={sim_frame}s)", fontsize=14, fontweight='bold')
                 self.scat_casualties.set_visible(False)
                 if self.quiver: self.quiver.remove(); self.quiver = None
                 
                 # Clear Background Heatmap
                 collection.set_array(None)
                 collection.set_facecolors('whitesmoke')
                 
                 # Fetch Data
                 p_map = {}
                 if sim_frame < len(self.penalty_history):
                     p_map = self.penalty_history[sim_frame]
                
                 # Filter for active hazards
                 x_vals, y_vals, sizes = [], [], []
                 for idx, val in p_map.items():
                     if val > 0.01: # Threshold to filter clutter
                         pt = self.cell_centroids.get(idx)
                         if pt:
                             x_vals.append(pt.x)
                             y_vals.append(pt.y)
                             # Size scales with penalty? or fixed?
                             sizes.append(100 + val * 100)
                             
                 if x_vals:
                     self.scat_penalties.set_offsets(np.column_stack([x_vals, y_vals]))
                     self.scat_penalties.set_sizes(sizes)
                     self.scat_penalties.set_visible(True)
                 else:
                     self.scat_penalties.set_visible(False)

            # 5. CASUALTIES HEATMAP/MARKERS

            # 3. CASUALTIES
            # 3. CASUALTIES HEATMAP
            # 3. CASUALTIES (MARKERS)
            elif self.view_mode == 'casualties':
                 if self.quiver: 
                     self.quiver.remove()
                     self.quiver = None
                 
                 ax.set_title("Phase 2: Casualty Locations (Red Markers)", fontsize=14, fontweight='bold')
                 self.scat_penalties.set_visible(False)
                 
                 # 1. Reset Road Colors to Neutral (so we can see the network)
                 collection.set_array(None)
                 collection.set_facecolors('whitesmoke')
                 collection.set_edgecolors('lightgray')
                 
                 # 2. Get Casualty Data
                 current_casualties = {}
                 if sim_frame < len(self.per_cell_casualty_history):
                      current_casualties = self.per_cell_casualty_history[sim_frame]
                 else:
                      current_casualties = self.casualties_per_cell
                 
                 # Robustness Check (Fallback)
                 max_val = 0
                 for c in current_casualties.values(): max_val = max(max_val, c)
                 if max_val == 0 and self.casualties > 0 and sim_frame > (len(self.history) * 0.9):
                      current_casualties = self.casualties_per_cell

                 # 3. Filter for Markers
                 x_vals, y_vals, sizes = [], [], []
                 for cid, count in current_casualties.items():
                     if count > 0:
                         pt = self.cell_centroids.get(cid)
                         if pt:
                             x_vals.append(pt.x)
                             y_vals.append(pt.y)
                             # Size scales with casualty count (min 50, max 300)
                             sizes.append(min(300, 50 + count * 10))
                 
                 # 4. Update Scatter Plot
                 if x_vals:
                     self.scat_casualties.set_offsets(np.column_stack([x_vals, y_vals]))
                     self.scat_casualties.set_sizes(sizes)
                     self.scat_casualties.set_visible(True)
                 else:
                     self.scat_casualties.set_visible(False)
 

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
            if hasattr(self, 'anim') and self.anim and self.anim.event_source:
                self.anim.event_source.interval = val
        self.speed_slider.on_changed(update_speed)
        
        ax_pause = plt.axes([0.35, 0.04, 0.1, 0.04])
        self.btn_pause = Button(ax_pause, 'Pause', color='lightgray', hovercolor='gray')
        
        ax_btn = plt.axes([0.50, 0.04, 0.12, 0.04])
        self.btn_casualty = Button(ax_btn, 'Casualties', color='salmon', hovercolor='red')
        
        # 3rd Button: DIJKSTRA
        # 3rd Button: DIJKSTRA
        ax_btn_dijk = plt.axes([0.65, 0.04, 0.12, 0.04])
        self.btn_dijkstra = Button(ax_btn_dijk, 'Dijkstra', color='lightblue', hovercolor='cyan')

        # 4th Button: PENALTIES (NEW)
        ax_btn_pen = plt.axes([0.78, 0.04, 0.12, 0.04])
        self.btn_penalty = Button(ax_btn_pen, 'Penalties', color='#FFA500', hovercolor='#FFD700')

        def set_view_mode(mode):
            self.view_mode = mode
            self.btn_casualty.color = 'salmon'
            self.btn_dijkstra.color = 'lightblue'
            self.btn_penalty.color = '#FFA500' # Orange
            
            if mode == 'casualties':
                self.btn_casualty.color = 'red' 
                ax.set_title("Total Casualties Heatmap", fontsize=14, fontweight='bold')
                cbar.set_label('Total Casualties')
                
            elif mode == 'dijkstra':
                self.btn_dijkstra.color = 'cyan' 
                ax.set_title("Dynamic Dijkstra Field", fontsize=14, fontweight='bold')
                cbar.set_label('Proximity/Safety (Bright=Safe)')
                
            elif mode == 'penalties':
                self.btn_penalty.color = '#FF4500' # Dark Orange
                ax.set_title("Hazard Penalties", fontsize=14, fontweight='bold')
                cbar.set_label('Penalty Score (0.0 - 1.0)')
                
            else: 
                ax.set_title("USTP Evacuation (Per-Cell Capacity Analysis)", fontsize=14, fontweight='bold')
                cbar.set_label('Occupancy Ratio (Population / Capacity)')
            
            update(self.current_frame)
            fig.canvas.draw_idle()

        def toggle_casualty(event):
            if self.view_mode == 'casualties': set_view_mode('occupancy')
            else: set_view_mode('casualties')
                
        def toggle_dijkstra(event):
            if self.view_mode == 'dijkstra': set_view_mode('occupancy')
            else: set_view_mode('dijkstra')
            
        def toggle_penalty(event):
            if self.view_mode == 'penalties': set_view_mode('occupancy')
            else: set_view_mode('penalties')
            
        self.btn_casualty.on_clicked(toggle_casualty)
        self.btn_dijkstra.on_clicked(toggle_dijkstra)
        self.btn_penalty.on_clicked(toggle_penalty)

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
            if event.xdata is None or event.ydata is None: return
            
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
        total_frames = len(self.history)
        # (Removed garbage offset logic)

        self.anim = FuncAnimation(fig, update, frames=total_frames, interval=100, blit=False, repeat=False)
        
        # Center Window
        try:
            mgr = plt.get_current_fig_manager()
            # TkAgg Backend Check
            if hasattr(mgr, 'window'):
                window = mgr.window
                screen_w = window.winfo_screenwidth()
                screen_h = window.winfo_screenheight()
                # Estimate size (14x10 inches * 100 dpi) = 1400x1000
                win_w, win_h = 1400, 1000
                x = max(0, int((screen_w - win_w) / 2))
                y = max(0, int((screen_h - win_h) / 2))
                window.wm_geometry(f"+{x}+{y}")
        except Exception:
            pass

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
    gpkg_path = os.path.join(base_dir, "..", "GPKG_Files", themap) #road_cells_split.gpkg | xu-road-cells.gpkg | csu_map.gpkg | cmu-map.gpkg

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
