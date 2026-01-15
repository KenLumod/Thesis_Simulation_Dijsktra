# USTP Evacuation Simulation Project - Documentation

This repository contains the simulation code, data, and analysis tools for the USTP Evacuation Thesis project. The project compares different pathfinding algorithms (Baseline MCA, Dijkstra, ACO) for crowd evacuation simulation.

## 📂 Project Structure

### 1. 🧬 Simulation Algorithms
Each algorithm variant is isolated in its own directory containing specific logic and iteration scripts.

*   **`Baseline/`**
    *   `mca_simulation_iterations.py`: The standard MCA (Multi-Criteria Analysis) simulation runner.
    *   `mca_simulation_original.py`: Original single-run version of the baseline.
    *   `simulation_results_*.xlsx`: Output logs from baseline runs.

*   **`Baseline_ACO/`**
    *   `mca_simulation_iterations_ACO.py`: MCA simulation enhanced with Ant Colony Optimization.
    *   `aco_logic.py`: Helper module containing the specific ACO path update logic.
    *   `simulation_results_aco_*.xlsx`: Output logs from ACO runs.

*   **`Dijkstra_Pure/`**
    *   `mca_simulation_iterations_dijkstra.py`: MCA simulation using purely Dijkstra's algorithm for pathfinding.
    *   `Dijkstra.py`: Reusable Dijkstra pathfinding implementation.
    *   `simulation_results_dijkstra_*.xlsx`: Output logs from Dijkstra runs.

### 2. 🗺️ Data Management (`GPKG_Files` & `Backups`)
Evacuation environment data is stored in GeoPackage format.

*   **`Backups/road_cells_split.gpkg`** (**PRIMARY - COMPLETE**)
    *   Contains the **complete dataset** used for full simulations.
    *   **Layers**:
        *   `road_cells(10x6m)`: The 10m x 6m road grid.
        *   `campus_exits`: The 3 main university exits.
        *   `building_exits`: Spawner locations from buildings.
        *   `safe_zones`: Evacuation destination zones.
    *   *Role*: Use this file for accurate visualization and simulation.

*   **`GPKG_Files/xu-road-cells.gpkg`**
    *   Contains an incomplete or older version of the map.
    *   **Missing**: Does NOT contain `campus_exits`.
    *   *Role*: Legacy or specific testing only.

### 3. 🛠️ Analysis & Visualization Tools (Root Directory)
Scripts created to inspect, verify, and compare simulation data.

#### Interactive Map Viewer (`view_map.py`)
A graphical tool to explore the GPKG layers.
*   **Usage**: `python view_map.py`
*   **Role**: Visual confirmation of map geometry, exit locations, and road layout. Use this to verify `road_cells_split.gpkg`.

#### Data Inspector (`view_road_cells.py`)
Extracts attributes from GPKG files to Excel for tabular inspection.
*   **Usage**: `python view_road_cells.py`
*   **Output**: Generates `gpkg_data_split.xlsx`.
*   **Role**: Checking spawn rates, node attributes, and verifying table counts.

#### Comparison Scripts
*   `compare_results_full.py`: Aggregates and compares metrics (time, casualties) across the different algorithm results directories.

---

## 🚀 How to Run

### 1. Visualization
To see the map and confirm data:
```bash
python view_map.py
```

### 2. Running Simulations
Navigate to the specific algorithm folder and run the iteration script:

**Baseline:**
```bash
cd Baseline
python mca_simulation_iterations.py
```

**Dijkstra:**
```bash
cd Dijkstra_Pure
python mca_simulation_iterations_dijkstra.py
```

**ACO:**
```bash
cd Baseline_ACO
python mca_simulation_iterations_ACO.py
```

---
## 📦 Dependencies
*   `geopandas`, `matplotlib`, `fiona`: For mapping tools.
*   `pandas`, `openpyxl`: For result logging and Excel export.
*   `networkx`: For graph theory operations (Dijkstra).
