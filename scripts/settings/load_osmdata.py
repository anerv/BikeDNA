# This data is prepared in the 1a_initialize_osm notebook

import osmnx as ox
import geopandas as gpd

exec(open("../settings/yaml_variables.py").read())
exec(open("../settings/paths.py").read())

# Load simplified and non-simplified graphs
osm_graph = ox.load_graphml(
    osm_graph_fp
)

osm_graph_simplified = ox.load_graphml(
    osm_graph_simplified_fp
)

print("OSM graphs loaded successfully!")

# Load grid
osm_grid = gpd.read_file(osm_grid_fp)
grid_ids = osm_grid.grid_id.to_list()

# Load saved edged and nodes
osm_nodes = gpd.read_file(osm_nodes_fp)
osm_nodes.set_index("osmid", inplace=True)

osm_edges = gpd.read_file(osm_edges_fp)
osm_edges.set_index(["u", "v", "key"], inplace=True)

osm_edges_simplified = gpd.read_file(osm_edges_simplified_fp)
osm_edges_simplified.set_index(["u", "v", "key"], inplace=True)

osm_nodes_simplified = gpd.read_file(osm_nodes_simplified_fp)
osm_nodes_simplified.set_index("osmid", inplace=True)
osm_nodes_simplified["osmid"] = osm_nodes_simplified.index

osm_edges_joined = gpd.read_file(osm_edges_joined_fp)

osm_nodes_joined = gpd.read_file(osm_nodes_joined_fp)

osm_edges_simp_joined = gpd.read_file(osm_edges_simplified_joined_fp)

osm_nodes_simp_joined = gpd.read_file(osm_nodes_simplified_joined_fp)

print("OSM data loaded successfully!")
