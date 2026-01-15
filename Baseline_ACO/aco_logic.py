import numpy as np
import random
import time

class Ant:
    def __init__(self, start_node, graph):
        self.current_node = start_node
        self.path = [start_node]
        self.visited = {start_node}
        self.graph = graph
        self.finished = False

    def move(self, pheromones, alpha=1.0, beta=1.0):
        if self.finished: return

        neighbors = list(self.graph.neighbors(self.current_node))
        # Filter unvisited to preventing looping immediately, 
        # but in ACO we might allow loops? 
        # For evacuation pathfinding (Reverse), we want to explore OUTWARD.
        # So we prefer unvisited nodes.
        
        valid_neighbors = [n for n in neighbors if n not in self.visited]
        
        if not valid_neighbors:
            # Dead end or surrounded by visited
            self.finished = True
            return

        # Probabilistic Choice
        # P(n) ~ (Pheromone(u,n)^alpha) * (Heuristic(n)^beta)
        # Heuristic: Maybe degree? Or just uniform for exploration?
        # Initial exploration: Uniform is fine.
        
        # In Reverse ACO:
        # Pheromone on edge (u, v) means "This way leads to Exit".
        # But we are exploring FROM Exit. So we want to mostly EXPLORE (Random).
        # We deposit pheromone pointing BACK to where we came from.
        
        # Simple exploration: Random Walk with bias towards open space?
        next_node = random.choice(valid_neighbors)
        
        self.path.append(next_node)
        self.visited.add(next_node)
        self.current_node = next_node

def run_aco_pathfinding(graph, exits_ids, road_cells, max_iterations=1000, n_ants=100, record_history=False):
    """
    Runs a 'Reverse' ACO process.
    ...
    Args:
        record_history (bool): If True, returns a list of state snapshots for animation.
    
    Returns:
        flow_directions (dict): {cell_id: target_neighbor_id}
        pheromones (dict): {(from, to): intensity}
        history (list): [ {'ants': [node_ids], 'pheromones': {copy}} ] (Optional)
    """
    print(f"🐜 Starting ACO Pathfinding ({n_ants} ants, {max_iterations} steps)...")
    
    # Initialize Pheromones (Edge -> Intensity)
    pheromones = {} 
    ants = []
    
    # History Container
    history_frames = []
    
    # Spawn initial batch
    for _ in range(n_ants):
        start = random.choice(exits_ids)
        ants.append(Ant(start, graph))
        
    start_time = time.time()
    
    for step in range(max_iterations):
        active_ants = [a for a in ants if not a.finished]
        if not active_ants:
             for _ in range(n_ants):
                start = random.choice(exits_ids)
                ants.append(Ant(start, graph))
             continue
             
        for ant in active_ants:
            curr = ant.current_node
            ant.move(pheromones)
            new_pos = ant.current_node
            
            if new_pos != curr:
                edge = (new_pos, curr)
                strength = 100.0 / (len(ant.path) + 1)
                pheromones[edge] = pheromones.get(edge, 0.0) + strength
        
        # Record History (Every 10 steps to safe RAM/Time)
        if record_history and step % 10 == 0:
            snapshot = {
                'ants': [a.current_node for a in active_ants],
                # Deep copy of pheromones might be heavy?
                # Just copy edge keys or values?
                # For viz, we need values.
                'pheromones': pheromones.copy()
            }
            history_frames.append(snapshot)
                
    # Evaporation (One shot at end or iterative?)
    # ... (rest unchanged)
    
    # Build Flow Directions... (same as before)
    flow_directions = {}
    
    nodes = list(graph.nodes)
    for u in nodes:
        if u in exits_ids:
            flow_directions[u] = None # At exit
            continue
            
        neighbors = list(graph.neighbors(u))
        best_n = None
        max_pher = -1.0
        
        for v in neighbors:
            p = pheromones.get((u, v), 0.0)
            if p > max_pher:
                max_pher = p
                best_n = v
        
        if max_pher > 0:
            flow_directions[u] = best_n
        else:
             flow_directions[u] = None

    print(f"🐜 ACO Complete in {time.time() - start_time:.2f}s. Covered {len(flow_directions)} nodes.")
    
    if record_history:
        return flow_directions, pheromones, history_frames
    return flow_directions, pheromones
