# This data is prepared in the 2a_initialize_ref notebook

import osmnx as ox
import geopandas as gpd

exec(open("../settings/yaml_variables.py").read())
exec(open("../settings/paths.py").read())

# Load simplified and non-simplified graphs
ref_graph = ox.load_graphml(ref_graph_fp)

ref_graph_simplified = ox.load_graphml(ref_graph_simplified_fp)

print("Reference graphs loaded successfully!")

# Load grid
ref_grid = gpd.read_file(ref_grid_fp)
grid_ids = ref_grid.grid_id.to_list()

# # Load saved edged and nodes
ref_nodes = gpd.read_file(ref_nodes_fp)
ref_nodes.set_index("osmid", inplace=True)

ref_edges = gpd.read_file(ref_edges_fp)
ref_edges.set_index(["u", "v", "key"], inplace=True)

ref_edges_simplified = gpd.read_file(ref_edges_simplified_fp)
ref_edges_simplified.set_index(["u", "v", "key"], inplace=True)

ref_nodes_simplified = gpd.read_file(ref_nodes_simplified_fp)
ref_nodes_simplified.set_index("osmid", inplace=True)

ref_edges_joined = gpd.read_file(ref_edges_joined_fp)

ref_nodes_joined = gpd.read_file(ref_nodes_joined_fp)

ref_edges_simp_joined = gpd.read_file(ref_edges_simplified_joined_fp)

ref_nodes_simp_joined = gpd.read_file(ref_nodes_simplified_joined_fp)

print("Reference data loaded successfully!")
