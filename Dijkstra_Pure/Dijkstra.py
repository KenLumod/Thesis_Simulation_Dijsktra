
import networkx as nx
import heapq

def calculate_dijkstra_field(G, exit_nodes):
    """
    Calculates the distance from every node in the graph G to the NEAREST exit node
    using a Reverse Dijkstra (Multi-Source Dijkstra) approach.
    
    Args:
        G (nx.Graph): The networkx graph representing the road network.
                      Edges must have a 'weight' attribute (distance).
        exit_nodes (list): A list of node IDs that are designated as exits.
        
    Returns:
        tuple: (distances_dict, visited_history_list)
               distances: {node_id: distance_float}
               visited_history: [(node_id, distance_float), ...] in order of exploration
    """
    
    # Initialize distances to infinity (or simply track visited)
    distances = {node: float('inf') for node in G.nodes()}
    
    # Priority Queue: (distance, node)
    pq = []
    
    # Initialize all exits with distance 0 and push to PQ
    for exit_node in exit_nodes:
        if exit_node in G:
            distances[exit_node] = 0.0
            heapq.heappush(pq, (0.0, exit_node))
            
    visited_order = [] # List of (node_id, distance) in order of settlement
    
    while pq:
        current_dist, u = heapq.heappop(pq)
        
        # If we found a shorter path to u already, skip
        if current_dist > distances[u]:
            continue
        
        visited_order.append((u, current_dist))
        
        # Explore neighbors
        for v in G.neighbors(u):
            # Get edge weight (default 1.0 if not set, though we usually work with unweighted grid-like, 
            # ideally use distance if available)
            # For our road cells, we used graph edge weights? 
            # In build_graph, we usually just added edges.
            # Let's assume Unit weight or Distance based on centroid?
            # ACO used 1.0 usually. Let's stick to 1.0 for flood fill consistency unless strict distance needed.
            # Actually, to be "Better", let's check if 'weight' exists.
            
            weight = G[u][v].get('weight', 1.0) 
            
            new_dist = current_dist + weight
            
            # Relaxation step
            if new_dist < distances[v]:
                distances[v] = new_dist
                heapq.heappush(pq, (new_dist, v))
                
    return distances, visited_order
