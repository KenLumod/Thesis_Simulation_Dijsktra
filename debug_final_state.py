from Baseline.mca_simulation_original import MCASimulation
import os

# Setup
base_dir = os.path.dirname(os.path.abspath(__file__))
gpkg_path = os.path.join(base_dir, "GPKG_Files", "road_cells_split.gpkg")

print(f"Loading from {gpkg_path}...")
sim = MCASimulation(gpkg_path)
sim.load_data()
sim.build_graph()
sim.compute_flow_directions()

# Initialize 5000 agents (Standard)
sim.initialize_population(total_agents=5000)

# Run for 500 steps (Standard)
print("Running simulation...")
sim.run(steps=500)

# Inspect Final State
final_pop = sim.history[-1]
remaining = {k:v for k,v in final_pop.items() if v > 0.1}

print("\n=== FINAL STATE ANALYSIS ===")
print(f"Total Agents Left: {sum(final_pop.values()):.2f}")
print(f"Count of Occupied Cells: {len(remaining)}")
print("Occupied Cells (ID: Count):")
for k, v in remaining.items():
    print(f"  Cell {k}: {v:.2f}")

if len(remaining) > 200:
    print("⚠️  WARNING: It seems MOST cells are occupied. This matches the 'whole roads' issue.")
else:
    print("✅  Normal occupation. If visualization shows 'whole roads', it is a plotting issue.")
