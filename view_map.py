"""
Interactive GPKG Map Visualizer
This script creates an interactive map with zoom, pan, and layer toggle capabilities.
"""

import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Button
import geopandas as gpd
import warnings
warnings.filterwarnings('ignore')

class InteractiveGPKGMap:
    def __init__(self, gpkg_path):
        self.gpkg_path = gpkg_path
        self.layers_data = {}
        self.layer_artists = {}
        self.layer_visible = {}
        
        # Define colors and styles
        self.layer_styles = {
            'Ghost_Bldgs.gpkg': {
                'color': '#FFB6C1', 'edgecolor': '#FF1493', 'alpha': 0.6,
                'linewidth': 0.5, 'label': 'Buildings', 'zorder': 1
            },
            'road_cells(10x6m)': {
                'color': '#4169E1', 'linewidth': 2, 'alpha': 0.8,
                'label': 'Road Network', 'zorder': 2
            },
            'building_exits': {
                'color': '#32CD32', 'marker': 'o', 'markersize': 60,
                'alpha': 0.9, 'label': 'Building Exits', 'zorder': 4,
                'edgecolor': 'darkgreen', 'linewidth': 1
            },
            'safe_zones': {
                'color': '#FFD700', 'marker': 's', 'markersize': 100,
                'alpha': 0.9, 'label': 'Safe Zones', 'zorder': 5,
                'edgecolor': '#FF8C00', 'linewidth': 2
            },
            'campus_exits': {
                'color': '#FF4500', 'marker': '^', 'markersize': 120,
                'alpha': 0.9, 'label': 'Campus Exits', 'zorder': 6,
                'edgecolor': 'darkred', 'linewidth': 2
            }
        }
    
    def load_layers(self):
        """Load all layers from the GeoPackage."""
        print("=" * 80)
        print(f"📍 Loading GeoPackage: {self.gpkg_path}")
        print("=" * 80)
        
        try:
            import fiona
            layers = fiona.listlayers(self.gpkg_path)
            print(f"\n📋 Found {len(layers)} layers: {', '.join(layers)}\n")
            
            for layer_name in layers:
                try:
                    # Skip internal tables
                    if layer_name.startswith('rtree_') or layer_name.startswith('gpkg_'):
                        continue
                    
                    # Also skip sqlite_sequence
                    if layer_name == 'sqlite_sequence':
                        continue
                        
                    print(f"📊 Loading layer: {layer_name}...", end=" ")
                    gdf = gpd.read_file(self.gpkg_path, layer=layer_name)
                    
                    if len(gdf) == 0:
                        print("⚠️  Empty, skipping")
                        continue
                    
                    self.layers_data[layer_name] = gdf
                    self.layer_visible[layer_name] = True
                    print(f"✅ {len(gdf)} features")
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
                    continue
        except Exception as e:
            print(f"❌ Critical error loading layers: {e}")

    def create_map(self):
        """Create the interactive map."""
        if not self.layers_data:
            print("❌ No data loaded to map!")
            return

        # Create figure with space for controls
        self.fig = plt.figure(figsize=(14, 10))
        
        # Main map axis
        self.ax = plt.subplot2grid((1, 6), (0, 0), colspan=5)
        
        # Control panel axis (for checkboxes)
        self.ax_controls = plt.subplot2grid((1, 6), (0, 5))
        self.ax_controls.axis('off')
        
        # Plot all layers
        self.plot_layers()
        
        # Setup controls
        self.setup_controls()
        
        # Customize the plot
        self.ax.set_xlabel('Easting (m)', fontsize=10, fontweight='bold')
        self.ax.set_ylabel('Northing (m)', fontsize=10, fontweight='bold')
        self.ax.set_title('Evacuation Map - Interactive Viewer\n(Scroll to zoom, Drag to pan)', 
                         fontsize=12, fontweight='bold')
        
        # Add grid
        self.ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        self.ax.set_aspect('equal')
        
        self.enable_scroll_zoom()
        
        plt.tight_layout()
    
    def plot_layers(self):
        """Plot all loaded layers."""
        for layer_name, gdf in self.layers_data.items():
            style = self.layer_styles.get(layer_name, {
                'color': '#808080', 'alpha': 0.5, 'label': layer_name
            })
            
            if gdf.geometry.empty:
                continue

            geom_type = gdf.geometry.geom_type.iloc[0]
            artist = None
            
            if geom_type in ['Point', 'MultiPoint']:
                artist = gdf.plot(
                    ax=self.ax,
                    color=style.get('color'),
                    marker=style.get('marker', 'o'),
                    markersize=style.get('markersize', 50),
                    alpha=style.get('alpha', 0.8),
                    edgecolor=style.get('edgecolor', 'black'),
                    linewidth=style.get('linewidth', 1),
                    zorder=style.get('zorder', 3),
                    label=style.get('label', layer_name)
                ).collections[-1]
                
            elif geom_type in ['LineString', 'MultiLineString']:
                current_collections = len(self.ax.collections)
                gdf.plot(
                    ax=self.ax,
                    color=style.get('color'),
                    linewidth=style.get('linewidth', 1),
                    alpha=style.get('alpha', 0.8),
                    zorder=style.get('zorder', 2),
                    label=style.get('label', layer_name)
                )
                if len(self.ax.collections) > current_collections:
                    artist = self.ax.collections[-1]
                
            elif geom_type in ['Polygon', 'MultiPolygon']:
                current_collections = len(self.ax.collections)
                gdf.plot(
                    ax=self.ax,
                    facecolor=style.get('color'),
                    edgecolor=style.get('edgecolor', 'black'),
                    linewidth=style.get('linewidth', 0.5),
                    alpha=style.get('alpha', 0.6),
                    zorder=style.get('zorder', 1),
                    label=style.get('label', layer_name)
                )
                if len(self.ax.collections) > current_collections:
                    artist = self.ax.collections[-1]
            
            # Store the artist for toggling visibility
            if artist:
                self.layer_artists[layer_name] = artist
    
    def setup_controls(self):
        """Setup interactive controls."""
        active_layers = [name for name in self.layers_data.keys() if name in self.layer_artists]
        
        if not active_layers:
            return

        labels = [self.layer_styles.get(name, {}).get('label', name) for name in active_layers]
        visibility = [True] * len(active_layers)
        
        check_ax = plt.axes([0.82, 0.4, 0.15, 0.4])
        self.check = CheckButtons(check_ax, labels, visibility)
        
        self.label_to_layer = {
            self.layer_styles.get(name, {}).get('label', name): name 
            for name in active_layers
        }
        
        self.check.on_clicked(self.toggle_layer)
        
        info_text = (
            "Mouse Wheel: Zoom\n"
            "pan: Click & Drag\n"
        )
        self.ax_controls.text(0.1, 0.85, info_text, fontsize=9, 
                             verticalalignment='top', transform=self.ax_controls.transAxes)

    def toggle_layer(self, label):
        layer_name = self.label_to_layer.get(label)
        if layer_name and layer_name in self.layer_artists:
            artist = self.layer_artists[layer_name]
            artist.set_visible(not artist.get_visible())
            self.fig.canvas.draw_idle()
    
    def enable_scroll_zoom(self):
        def on_scroll(event):
            if event.inaxes != self.ax: return
            cur_xlim = self.ax.get_xlim()
            cur_ylim = self.ax.get_ylim()
            xdata = event.xdata
            ydata = event.ydata
            if xdata is None or ydata is None: return
            
            scale_factor = 0.8 if event.button == 'up' else 1.2
            
            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
            
            relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
            rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
            
            self.ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
            self.ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
            self.fig.canvas.draw_idle()
        
        self.fig.canvas.mpl_connect('scroll_event', on_scroll)

    def show(self):
        print("\n🗺️  Opening map window...")
        plt.show()

def main():
    # Check if a file path was provided as a command line argument
    if len(sys.argv) > 1:
        gpkg_path = sys.argv[1]
    else:
        # Default options to search for
        gpkg_dir = Path(__file__).parent / "GPKG_Files"
        possible_files = [
            gpkg_dir / "usep-map.gpkg", 
            gpkg_dir / "csu_map.gpkg", 
            gpkg_dir / "xu-road-cells.gpkg",
            gpkg_dir / "road_cells_split.gpkg"
        ]
        
        gpkg_path = None
        for p in possible_files:
            if p.exists():
                gpkg_path = str(p)
                break
        
        if gpkg_path is None and gpkg_dir.exists():
            # Fallback to any gpkg found
            found = list(gpkg_dir.glob("*.gpkg"))
            if found:
                gpkg_path = str(found[0])

    if not gpkg_path or not os.path.exists(gpkg_path):
        print(f"❌ Error: No valid GPKG file found!")
        print(f"Searched for: {gpkg_path}")
        print("Usage: python view_map.py [path_to_gpkg]")
        return
    
    map_viewer = InteractiveGPKGMap(gpkg_path)
    map_viewer.load_layers()
    map_viewer.create_map()
    map_viewer.show()

if __name__ == "__main__":
    main()
