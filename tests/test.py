#%%
import geopandas as gpd
import pandas as pd
import os.path
import osmnx as ox
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon, Point, MultiLineString
import math

from src import evaluation_functions as ef

from src import matching_functions as mf
from src import graph_functions as gf

#%%
###################### TESTS FOR EVALUATION FUNCTIONS #############################


# Test for create_grid_geometry
ext = [(0, 0), (0, 10), (10, 10), (10, 0)]
interior = [(4, 4), (6, 4), (6, 6), (4, 6)]
poly = Polygon(ext, [interior])
gdf = gpd.GeoDataFrame(geometry=[poly])
grid = ef.create_grid_geometry(gdf, 1)

assert len(grid) == (10 * 10) - (2 * 2)
assert len(grid.geom_type.unique()) == 1
assert grid.geom_type.unique()[0] == "Polygon"
assert grid.loc[0, "geometry"].area == 1


# Test simplify bicycle tags
queries = {
    "centerline_false_bidirectional_true": [
        "highway == 'cycleway' & (oneway=='no' or oneway_bicycle=='no')",
        "highway == 'track' & bicycle == 'designated' & (oneway=='no' or oneway_bicycle =='no')",
        "highway == 'path' & bicycle == 'designated' & (oneway=='no' or oneway_bicycle =='no')",
    ],
    "centerline_false_bidirectional_false": [
        "highway == 'cycleway' & (oneway !='no' or oneway_bicycle != 'no')",
        "highway == 'track' & bicycle == 'designated' & (oneway !='no' or oneway_bicycle !='no')",
        "highway == 'path' & bicycle == 'designated' & (oneway !='no' or oneway_bicycle !='no')",
    ],
    "centerline_true_bidirectional_true": [
        "cycleway_left in ['lane','track','opposite_lane','opposite_track','designated','crossing'] and (cycleway_right in ['no','none','separate'] or cycleway_right.isnull() or cycleway_right not in ['lane','track','opposite_lane','opposite_track','designated','crossing']) and oneway_bicycle =='no'",
        "cycleway_right in ['lane','track','opposite_lane','opposite_track','designated','crossing'] and (cycleway_left in ['no','none','separate'] or cycleway_left.isnull() or cycleway_left not in ['lane','track','opposite_lane','opposite_track','designated','crossing']) and oneway_bicycle =='no'",
        "cycleway in ['lane','track','opposite_lane','opposite_track','designated','crossing'] and (oneway_bicycle == 'no' or oneway_bicycle.isnull())",
        "cycleway_both in ['lane','track','opposite_lane','opposite_track','designated','crossing'] and (oneway_bicycle == 'no' or oneway_bicycle.isnull())",
        "cycleway_left in ['lane','track','opposite_lane','opposite_track','designated','crossing'] and cycleway_right in ['lane','track','opposite_lane','opposite_track','designated','crossing']",
    ],
    "centerline_true_bidirectional_false": [
        "cycleway_left in ['lane','track','opposite_lane','opposite_track','designated','crossing'] and (cycleway_right in ['no','none','separate'] or cycleway_right.isnull()  or cycleway_right not in ['lane','track','opposite_lane','opposite_track','designated','crossing']) and oneway_bicycle !='no'",
        "cycleway_right in ['lane','track','opposite_lane','opposite_track','designated','crossing'] and (cycleway_left in ['no','none','separate'] or cycleway_left.isnull()  or cycleway_left not in ['lane','track','opposite_lane','opposite_track','designated','crossing']) and oneway_bicycle != 'no'",
        "cycleway in ['lane','track','opposite_lane','opposite_track','designated','crossing'] and oneway_bicycle == 'yes'",
        "cycleway_both in ['lane','track','opposite_lane','opposite_track','designated','crossing'] and oneway_bicycle == 'yes'",
    ],
}

# Test simplify cycling tags
l1 = LineString([[1, 1], [10, 10]])
l2 = LineString([[2, 1], [6, 10]])
l3 = LineString([[10, 10], [10, 20]])
l4 = LineString([[11, 9], [5, 20]])
l5 = LineString([[1, 12], [4, 12]])

lines = [l1, l2, l3, l4, l5]
d = {
    "highway": ["cycleway", "primary", "secondary", "path", "track"],
    "bicycle_infrastructure": ["yes", "yes", "yes", "yes", "yes"],
    "cycleway": [np.nan, "track", np.nan, np.nan, "no"],
    "cycleway_both": [np.nan, np.nan, "lane", "track", np.nan],
    "bicycle_road": [0, 0, 0, 0, "yes"],
    "bicycle": [np.nan, np.nan, np.nan, "designated", "designated"],
    "cycleway_right": [np.nan, np.nan, np.nan, np.nan, np.nan],
    "cycleway_left": [np.nan, np.nan, np.nan, np.nan, np.nan],
    "oneway": ["no", "yes", np.nan, "yes", np.nan],
    "oneway_bicycle": [np.nan, "yes", np.nan, "no", np.nan],
    "geometry": lines,
}
edges = gpd.GeoDataFrame(d)
edges["length"] = edges.geometry.length


edges = ef.simplify_bicycle_tags(edges, queries)

assert (
    edges.loc[0, "bicycle_bidirectional"] == True
    and edges.loc[0, "bicycle_geometries"] == "true_geometries"
)
assert (
    edges.loc[1, "bicycle_bidirectional"] == False
    and edges.loc[1, "bicycle_geometries"] == "centerline"
)
assert (
    edges.loc[2, "bicycle_bidirectional"] == True
    and edges.loc[2, "bicycle_geometries"] == "centerline"
)
assert (
    edges.loc[3, "bicycle_bidirectional"] == True
    and edges.loc[3, "bicycle_geometries"] == "true_geometries"
)
assert (
    edges.loc[4, "bicycle_bidirectional"] == False
    and edges.loc[4, "bicycle_geometries"] == "true_geometries"
)


# Test measure_infrastructure_length
l1 = LineString([[1, 1], [10, 10]])
l2 = LineString([[2, 1], [6, 10]])
l3 = LineString([[10, 10], [10, 20]])
l4 = LineString([[11, 9], [5, 20]])
l5 = LineString([[1, 12], [4, 12]])

lines = [l1, l2, l3, l4, l5]
d = {
    "cycling_infrastructure": ["yes", "yes", "yes", "yes", "no"],
    "cycling_bidirectional": [True, False, False, True, False],
    "cycling_geometries": [
        "true_geometries",
        "true_geometries",
        "centerline",
        "centerline",
        "centerline",
    ],
    "geometry": lines,
}
edges = gpd.GeoDataFrame(d)
edges["length"] = edges.geometry.length


edges["infrastructure_length"] = edges.apply(
    lambda x: ef.measure_infrastructure_length(
        edge=x.geometry,
        geometry_type=x.cycling_geometries,
        bidirectional=x.cycling_bidirectional,
        bicycle_infrastructure=x.cycling_infrastructure,
    ),
    axis=1,
)


assert edges.loc[0, "infrastructure_length"] == edges.loc[0, "length"] * 2
assert edges.loc[1, "infrastructure_length"] == edges.loc[1, "length"]
assert edges.loc[2, "infrastructure_length"] == edges.loc[2, "length"]
assert edges.loc[3, "infrastructure_length"] == edges.loc[3, "length"] * 2
assert pd.isnull(edges.loc[4, "infrastructure_length"]) == True


# Test define_protected_unprotected
l1 = LineString([[1, 1], [10, 10]])
l2 = LineString([[2, 1], [6, 10]])
l3 = LineString([[10, 10], [10, 20]])
l4 = LineString([[11, 9], [5, 20]])
l5 = LineString([[1, 12], [4, 12]])

lines = [l1, l2, l3, l4, l5]
d = {
    "highway": ["cycleway", "primary", "secondary", "path", "track"],
    "cycleway": [np.nan, "track", np.nan, "lane", "no"],
    "cycleway_both": [np.nan, np.nan, "shared_lane", "track", np.nan],
    "bicycle_road": [0, 0, 0, 0, "yes"],
    "cycleway_right": [np.nan, np.nan, np.nan, np.nan, np.nan],
    "cycleway_left": [np.nan, np.nan, np.nan, np.nan, np.nan],
    "geometry": lines,
}
edges = gpd.GeoDataFrame(d)
edges["length"] = edges.geometry.length


queries = {
    "protected": [
        "highway == 'cycleway'",
        "cycleway in ['track','opposite_track']",
        "cycleway_left in ['track','opposite_track']",
        "cycleway_right in ['track','opposite_track']",
        "cycleway_both in ['track','opposite_track']",
    ],
    "unprotected": [
        "cycleway in ['lane','opposite_lane','shared_lane','crossing']",
        "cycleway_left in ['lane','opposite_lane','shared_lane','crossing']",
        "cycleway_right in ['lane','opposite_lane','shared_lane','crossing']",
        "cycleway_both in ['lane','opposite_lane','shared_lane','crossing']",
        "bicycle_road == 'yes'",
    ],
    "unknown": [
        "cycleway in ['designated']",
        "cycleway_left in ['designated']",
        "cycleway_right in ['designated']",
        "cycleway_both in ['designated']",
    ],
}

edges = ef.define_protected_unprotected(edges, queries)

assert edges.loc[0, "protected"] == "protected"
assert edges.loc[1, "protected"] == "protected"
assert edges.loc[2, "protected"] == "unprotected"
assert edges.loc[3, "protected"] == "mixed"
assert edges.loc[4, "protected"] == "unprotected"


# Test get_dangling_nodes
edges = gpd.read_file("../tests/test_data/edges.gpkg")
nodes = gpd.read_file("../tests/test_data/nodes.gpkg")
nodes.set_index("osmid", inplace=True)

d_nodes = ef.get_dangling_nodes(edges, nodes)
assert len(d_nodes) == 9
assert type(d_nodes) == gpd.geodataframe.GeoDataFrame


# Test count features in grid
ext = [(0, 0), (0, 10), (10, 10), (10, 0)]
interior = [(4, 4), (6, 4), (6, 6), (4, 6)]
poly = Polygon(ext, [interior])
gdf = gpd.GeoDataFrame(geometry=[poly])
grid = ef.create_grid_geometry(gdf, 1)
grid["grid_id"] = grid.index

points = gpd.read_file("../tests/test_data/random_points.gpkg")
points_joined = gpd.overlay(points, grid, how="intersection")

test_count = ef.count_features_in_grid(points_joined, "points")

assert test_count.loc[0, "count_points"] == 2
assert test_count.loc[18, "count_points"] == 2
assert test_count.loc[28, "count_points"] == 1


# Test length features in grid
ext = [(0, 0), (0, 10), (10, 10), (10, 0)]
poly = Polygon(ext)
gdf = gpd.GeoDataFrame(geometry=[poly])
grid = ef.create_grid_geometry(gdf, 1)
grid["grid_id"] = grid.index

l1 = LineString([[0.5, 0], [0.5, 1]])
l2 = LineString([[1, 1], [4, 4]])
l3 = LineString([[8.5, 0], [8.5, 10]])

lines = [
    l1,
    l2,
    l3,
]
d = {"id": [1, 2, 3], "geometry": lines}
edges = gpd.GeoDataFrame(d)
edges["length"] = edges.geometry.length

edges_joined = gpd.overlay(edges, grid, how="intersection", keep_geom_type=True)

test_len = ef.length_features_in_grid(edges_joined, "edges")

assert test_len.loc[0, "length_edges"] == 1
assert round(test_len.loc[2, "length_edges"], 2) == 1.41
assert round(test_len.loc[2, "length_edges"], 2) == 1.41

assert round(test_len.length_edges.sum(), 2) == round(1 + 3 * (1.414214) + 10, 2)


# Test length of features in grid
ext = [(0, 0), (0, 10), (10, 10), (10, 0)]
poly = Polygon(ext)
gdf = gpd.GeoDataFrame(geometry=[poly])
grid = ef.create_grid_geometry(gdf, 1)
grid["grid_id"] = grid.index

# Test length_of_features_in_grid
line_gdf = gpd.read_file("../tests/test_data/random_lines.gpkg", driver="GPKG")
lines_joined = gpd.overlay(line_gdf, grid, how="intersection", keep_geom_type=True)

test_length = ef.length_of_features_in_grid(lines_joined, "lines")

assert round(test_length.loc[0, "length_lines"], 2) == 1.41
assert round(test_length.loc[1, "length_lines"], 2) == 2.83


# Test compute_network_density
G = nx.MultiDiGraph()

# In grid cell one
l1 = LineString([[0, 0], [10, 10]])
l2 = LineString([[2, 1], [6, 10]])

# in grid cell two
l3 = LineString([[10, 10], [10, 20]])

# in grid cell four
l4 = LineString([[11, 11], [30, 30]])

G.add_node(1, x=0, y=0)
G.add_node(2, x=10, y=10)
G.add_node(3, x=2, y=1)
G.add_node(4, x=6, y=10)
G.add_node(5, x=10, y=20)
G.add_node(6, x=11, y=11)
G.add_node(7, x=30, y=30)

# add length and osmid just for the functions to work
G.add_edge(
    1,
    2,
    0,
    length=10,
    osmid=np.random.randint(1, 999999),
    infrastructure_length=20,
    geometry=l1,
)
G.add_edge(
    3,
    4,
    0,
    length=10,
    osmid=np.random.randint(1, 999999),
    infrastructure_length=10.7,
    geometry=l2,
)
G.add_edge(
    2,
    5,
    0,
    length=10,
    osmid=np.random.randint(1, 999999),
    infrastructure_length=13,
    geometry=l3,
)
G.add_edge(
    6,
    7,
    0,
    length=10,
    osmid=np.random.randint(1, 999999),
    infrastructure_length=25,
    geometry=l4,
)

G.graph["crs"] = "EPSG:25832"

nodes, edges = ox.graph_to_gdfs(G)
polygon = Polygon([(0, 0), (40, 0), (40, 40), (0, 40)])
poly_gdf = gpd.GeoDataFrame(geometry=[polygon])
poly_gdf = poly_gdf.set_crs("EPSG:25832")
grid = ef.create_grid_geometry(poly_gdf, 10)
grid = grid.set_crs("EPSG:25832")

edge_density, node_density, dangling_node_density = ef.compute_network_density(
    (edges, nodes), grid.unary_union.area, return_dangling_nodes=True
)
assert edge_density == 42937.5
assert node_density == 4375.0
assert dangling_node_density == 3750.0

results_dict = {}
grid_ids = grid.grid_id.to_list()

edges_j = gpd.overlay(edges, grid, how="intersection", keep_geom_type=True)
nodes_j = gpd.overlay(nodes, grid, how="intersection", keep_geom_type=True)

data = (edges_j, nodes_j)
[
    ef.run_grid_analysis(
        grid_id,
        data,
        results_dict,
        ef.compute_network_density,
        grid["geometry"].loc[grid.grid_id == grid_id].area.values[0],
    )
    for grid_id in grid_ids
]

results_df = pd.DataFrame.from_dict(results_dict, orient="index")
results_df.reset_index(inplace=True)
results_df.rename(
    columns={"index": "grid_id", 0: "edge_density", 1: "node_density"}, inplace=True
)

results = results_df.edge_density.to_list()
results.sort()
res = [r for r in results if r > 0]
assert res == [130000.0, 250000.0, 307000.0, 380000.0]


# Test get_component_edges
# Add geometries to edges
G = nx.MultiDiGraph()

# Geometries
l1 = LineString([[1, 1], [10, 10]])
l2 = LineString([[2, 1], [6, 10]])
l3 = LineString([[10, 10], [10, 20]])
l4 = LineString([[11, 9], [5, 20]])
l5 = LineString([[1, 12], [4, 12]])
l6 = LineString([[11, 9], [5, 20]])

# One component
G.add_node(1, x=10, y=10)
G.add_node(2, x=20, y=20)
G.add_node(3, x=25, y=30)
G.add_node(4, x=25, y=40)
G.add_node(5, x=24, y=40)

# add length and osmid just for the functions to work
G.add_edge(1, 2, 0, length=10, osmid=np.random.randint(1, 999999), geometry=l1)
G.add_edge(2, 3, 0, length=10, osmid=np.random.randint(1, 999999), geometry=l2)
G.add_edge(3, 4, 0, length=10, osmid=np.random.randint(1, 999999), geometry=l3)
G.add_edge(1, 5, 0, length=10, osmid=np.random.randint(1, 999999), geometry=l2)
G.add_edge(5, 1, 1, length=10, osmid=np.random.randint(1, 999999), geometry=l5)

# Second component
G.add_node(6, x=50, y=50)
G.add_node(7, x=47, y=47)
G.add_node(8, x=53, y=50)
G.add_node(9, x=45, y=60)
G.add_node(10, x=44, y=60)

# add length and osmid just for the functions to work
G.add_edge(6, 7, 0, length=10, osmid=np.random.randint(1, 999999), geometry=l6)
G.add_edge(7, 8, 0, length=10, osmid=np.random.randint(1, 999999), geometry=l1)
G.add_edge(8, 9, 0, length=30, osmid=np.random.randint(1, 999999), geometry=l3)
G.add_edge(9, 10, 0, length=17, osmid=np.random.randint(1, 999999), geometry=l4)

G.graph["crs"] = "epsg:25832"
components = ef.return_components(G)
component_edges = ef.get_component_edges(components, "EPSG:25832")
assert len(component_edges) == 9
assert type(component_edges) == gpd.geodataframe.GeoDataFrame
assert "component_id" in component_edges.columns
assert list(component_edges.component_id.values) == [0, 0, 0, 0, 0, 1, 1, 1, 1]


# Test check_intersection
l1 = LineString([[1, 1], [10, 10]])
l2 = LineString([[2, 1], [6, 10]])
l3 = LineString([[10, 10], [10, 20]])
l4 = LineString([[11, 9], [5, 20]])
l5 = LineString([[1, 12], [4, 12]])

lines = [l1, l2, l3, l4, l5]
d = {
    "bridge": ["yes", "no", None, "no", None],
    "tunnel": ["no", "no", None, None, None],
    "geometry": lines,
}
edges = gpd.GeoDataFrame(d)

edges["intersection_issues"] = edges.apply(
    lambda x: ef.check_intersection(row=x, gdf=edges, print_check=False), axis=1
)

count_intersection_issues = len(
    edges.loc[(edges.intersection_issues.notna()) & edges.intersection_issues > 0]
)

assert count_intersection_issues == 2
assert edges.loc[2, "intersection_issues"] == 1
assert edges.loc[3, "intersection_issues"] == 1


# Test find_missing_intersection
l1 = LineString([[1, 1], [11, 11]])
l2 = LineString([[2, 1], [6, 10]])
l3 = LineString([[10, 10], [10, 20]])
l4 = LineString([[11, 9], [5, 20]])
l5 = LineString([[1, 12], [4, 12]])
l6 = LineString([[6, 20], [10, 20]])

lines = [l1, l2, l3, l4, l5, l6]
d = {
    "id": [1, 2, 3, 4, 5, 6],
    "bridge": ["yes", "no", None, "no", None, "no"],
    "tunnel": ["no", "no", None, None, None, None],
    "geometry": lines,
}
edges = gpd.GeoDataFrame(d)

missing_nodes_edge_ids, edges_with_missing_nodes = ef.find_missing_intersections(
    edges, "id"
)

assert len(missing_nodes_edge_ids) == 2
assert missing_nodes_edge_ids == [3, 4] or missing_nodes_edge_ids == [4, 3]
assert len(edges_with_missing_nodes) == 2
assert edges_with_missing_nodes.id.to_list() == [3, 4]


# Test incompatible tags
l1 = LineString([[1, 1], [10, 10]])
l2 = LineString([[2, 1], [6, 10]])
l3 = LineString([[10, 10], [10, 20]])
l4 = LineString([[11, 9], [5, 20]])
l5 = LineString([[1, 12], [4, 12]])

lines = [l1, l2, l3, l4, l5]
d = {
    "cycling": ["yes", "no", None, "yes", None],
    "car": ["no", "no", None, "yes", None],
    "geometry": lines,
}
edges = gpd.GeoDataFrame(d)

dict = {
    "cycling": {"yes": [["bicycle", "no"], ["bicycle", "dismount"], ["car", "yes"]]}
}

incomp_tags_results = ef.check_incompatible_tags(edges, dict)
assert incomp_tags_results["cycling/car"] == 1


# Test existing tags
l1 = LineString([[1, 1], [10, 10]])
l2 = LineString([[2, 1], [6, 10]])
l3 = LineString([[10, 10], [10, 20]])
l4 = LineString([[11, 9], [5, 20]])
l5 = LineString([[1, 12], [4, 12]])

lines = [l1, l2, l3, l4, l5]
d = {
    "cycleway_width": [np.nan, 2, 2, 1, np.nan],
    "width": [1, np.nan, 2, 1, 0],
    "surface": ["paved", np.nan, np.nan, "gravel", np.nan],
    "bicycle_geometries": [
        "true_geometries",
        "true_geometries",
        "centerline",
        "centerline",
        "centerline",
    ],
    "geometry": lines,
}
edges = gpd.GeoDataFrame(d)

dict = {
    "surface": {
        "true_geometries": ["surface", "cycleway_surface"],
        "centerline": ["cycleway_surface"],
    },
    "width": {
        "true_geometries": [
            "width",
            "cycleway_width",
            "cycleway_left_width",
            "cycleway_right_width",
            "cycleway_both_width",
        ],
        "centerline": [
            "cycleway_width",
            "cycleway_left_width",
            "cycleway_right_width",
            "cycleway_both_width",
        ],
    },
    "speedlimit": {"all": ["maxspeed"]},
    "lit": {"all": ["lit"]},
}

existing_tags_results = ef.analyze_existing_tags(edges, dict)

assert existing_tags_results["surface"]["count"] == 1
assert round(existing_tags_results["surface"]["length"]) == 13
assert existing_tags_results["width"]["count"] == 4
assert round(existing_tags_results["width"]["length"]) == 35


# Test return components
G = nx.MultiDiGraph()
# One component
G.add_node(1, x=10, y=10)
G.add_node(2, x=20, y=20)
G.add_node(3, x=25, y=30)
G.add_node(4, x=25, y=40)
G.add_node(5, x=24, y=40)

# add length and osmid just for the functions to work
G.add_edge(1, 2, 0, length=10, osmid=np.random.randint(1, 999999))
G.add_edge(2, 3, 0, length=10, osmid=np.random.randint(1, 999999))
G.add_edge(3, 4, 0, length=10, osmid=np.random.randint(1, 999999))
G.add_edge(1, 5, 0, length=10, osmid=np.random.randint(1, 999999))
G.add_edge(5, 1, 1, length=10, osmid=np.random.randint(1, 999999))

# Second component
G.add_node(6, x=50, y=50)
G.add_node(7, x=47, y=47)
G.add_node(8, x=53, y=50)
G.add_node(9, x=45, y=60)
G.add_node(10, x=44, y=60)

# add length and osmid just for the functions to work
G.add_edge(6, 7, 0, length=10, osmid=np.random.randint(1, 999999))
G.add_edge(7, 8, 0, length=10, osmid=np.random.randint(1, 999999))
G.add_edge(8, 9, 0, length=30, osmid=np.random.randint(1, 999999))
G.add_edge(9, 10, 0, length=17, osmid=np.random.randint(1, 999999))

G.graph["crs"] = "epsg:25832"

components = ef.return_components(G)

assert len(components) == 2
# Check that the same graph type is returned
assert type(components[0]) == nx.MultiDiGraph
# Check that the expected nodes are returned
assert list(components[0].nodes) == [1, 2, 3, 4, 5]
# Check that the expected number of edges are returned
assert len(components[0].edges) == 5
assert len(components[1].edges) == 4


# Test component_lengths
G = nx.MultiDiGraph()
# One component
G.add_node(1, x=10, y=10)
G.add_node(2, x=20, y=20)
G.add_node(3, x=25, y=30)
G.add_node(4, x=25, y=40)
G.add_node(5, x=24, y=40)

# add length and osmid just for the functions to work
G.add_edge(1, 2, 0, length=10, osmid=np.random.randint(1, 999999))
G.add_edge(2, 3, 0, length=10, osmid=np.random.randint(1, 999999))
G.add_edge(3, 4, 0, length=10, osmid=np.random.randint(1, 999999))
G.add_edge(1, 5, 0, length=10, osmid=np.random.randint(1, 999999))

# Second component
G.add_node(6, x=50, y=50)
G.add_node(7, x=47, y=47)
G.add_node(8, x=53, y=50)
G.add_node(9, x=45, y=60)
G.add_node(10, x=44, y=60)

# add length and osmid just for the functions to work
G.add_edge(6, 7, 0, length=10, osmid=np.random.randint(1, 999999))
G.add_edge(7, 8, 0, length=10, osmid=np.random.randint(1, 999999))
G.add_edge(8, 9, 0, length=30, osmid=np.random.randint(1, 999999))
G.add_edge(9, 10, 0, length=17, osmid=np.random.randint(1, 999999))

G.graph["crs"] = "epsg:25832"
nodes, edges = ox.graph_to_gdfs(G)

components = ef.return_components(G)
test_c_lengths = ef.component_lengths(components)

assert test_c_lengths.loc[0, "component_length"] == 40
assert test_c_lengths.loc[1, "component_length"] == 67


import networkx as nx

# Test find_adjacent_components
G = nx.MultiDiGraph()
# One component
G.add_node(1, x=10, y=10)
G.add_node(2, x=20, y=20)
G.add_node(3, x=25, y=30)
G.add_node(4, x=25, y=40)
G.add_node(5, x=24, y=40)

# add length and osmid just for the functions to work
G.add_edge(1, 2, 0, length=10, osmid=1)
G.add_edge(2, 3, 0, length=10, osmid=2)
G.add_edge(3, 4, 0, length=10, osmid=3)
G.add_edge(1, 5, 0, length=10, osmid=4)

# Second component
G.add_node(6, x=50, y=50)
G.add_node(7, x=47, y=47)
G.add_node(8, x=53, y=50)
G.add_node(9, x=45, y=60)
G.add_node(10, x=44, y=60)

G.add_edge(6, 7, 0, length=10, osmid=5)
G.add_edge(7, 8, 0, length=10, osmid=6)
G.add_edge(8, 9, 0, length=30, osmid=7)
G.add_edge(9, 10, 0, length=17, osmid=8)

# Third component
G.add_node(11, x=53, y=55)
G.add_node(12, x=70, y=70)
G.add_node(13, x=80, y=85)
G.add_node(14, x=75, y=85)

G.add_edge(11, 12, 0, length=10, osmid=9)
G.add_edge(12, 13, 0, length=10, osmid=10)
G.add_edge(13, 14, 0, length=30, osmid=11)

G.graph["crs"] = "EPSG:25832"
nodes, edges = ox.graph_to_gdfs(G)

components = ef.return_components(G)
adj_comps = ef.find_adjacent_components(
    components, edge_id="osmid", buffer_dist=5, crs="EPSG:25832"
)

# Check that the expected components are considered adjacent
assert adj_comps[0]["osmid_left"] in [9, 7]
assert adj_comps[0]["osmid_right"] in [9, 7]
assert len(adj_comps) == 1


# Test assign_component_id
G = nx.MultiDiGraph()
# One component
G.add_node(1, x=10, y=10)
G.add_node(2, x=20, y=20)
G.add_node(3, x=25, y=30)
G.add_node(4, x=25, y=40)
G.add_node(5, x=24, y=40)

# add length and osmid just for the functions to work
G.add_edge(1, 2, 0, length=10, osmid=np.random.randint(1, 999999))
G.add_edge(2, 3, 0, length=10, osmid=np.random.randint(1, 999999))
G.add_edge(3, 4, 0, length=10, osmid=np.random.randint(1, 999999))
G.add_edge(1, 5, 0, length=10, osmid=np.random.randint(1, 999999))

# Second component
G.add_node(6, x=50, y=50)
G.add_node(7, x=47, y=47)
G.add_node(8, x=53, y=50)
G.add_node(9, x=45, y=60)
G.add_node(10, x=44, y=60)

G.add_edge(6, 7, 0, length=10, osmid=np.random.randint(1, 999999))
G.add_edge(7, 8, 0, length=10, osmid=np.random.randint(1, 999999))
G.add_edge(8, 9, 0, length=30, osmid=np.random.randint(1, 999999))
G.add_edge(9, 10, 0, length=17, osmid=np.random.randint(1, 999999))

# Third component
G.add_node(11, x=53, y=55)
G.add_node(12, x=70, y=70)
G.add_node(13, x=80, y=85)
G.add_node(14, x=75, y=85)

G.add_edge(11, 12, 0, length=10, osmid=np.random.randint(1, 999999))
G.add_edge(12, 13, 0, length=10, osmid=np.random.randint(1, 999999))
G.add_edge(13, 14, 0, length=30, osmid=np.random.randint(1, 999999))

G.graph["crs"] = "EPSG:25832"
nodes, edges = ox.graph_to_gdfs(G)
edges["edge_id"] = edges["osmid"]

components = ef.return_components(G)
edges_comp_ids, comp_dict = ef.assign_component_id(
    components, edges, edge_id_col="osmid"
)

assert len(edges_comp_ids) == 11
assert len(comp_dict) == 3
assert list(comp_dict.keys()) == [0, 1, 2]
assert edges_comp_ids[0:4]["component"].unique()[0] == 0
assert edges_comp_ids[4:8]["component"].unique()[0] == 1
assert edges_comp_ids[8:10]["component"].unique()[0] == 2


# Test assign_component_id_to_grid
G = nx.MultiDiGraph()
# One component
G.add_node(1, x=10, y=10)
G.add_node(2, x=20, y=20)
G.add_node(3, x=25, y=30)
G.add_node(4, x=25, y=40)
G.add_node(5, x=24, y=40)

# add length and osmid just for the functions to work
G.add_edge(1, 2, 0, length=10, osmid=np.random.randint(1, 999999))
G.add_edge(2, 3, 0, length=10, osmid=np.random.randint(1, 999999))
G.add_edge(3, 4, 0, length=10, osmid=np.random.randint(1, 999999))
G.add_edge(1, 5, 0, length=10, osmid=np.random.randint(1, 999999))

# Second component
G.add_node(6, x=50, y=50)
G.add_node(7, x=47, y=47)
G.add_node(8, x=53, y=50)
G.add_node(9, x=45, y=60)
G.add_node(10, x=44, y=60)

G.add_edge(6, 7, 0, length=10, osmid=np.random.randint(1, 999999))
G.add_edge(7, 8, 0, length=10, osmid=np.random.randint(1, 999999))
G.add_edge(8, 9, 0, length=30, osmid=np.random.randint(1, 999999))
G.add_edge(9, 10, 0, length=17, osmid=np.random.randint(1, 999999))

# Third component
G.add_node(11, x=53, y=55)
G.add_node(12, x=70, y=70)
G.add_node(13, x=80, y=85)
G.add_node(14, x=75, y=85)

G.add_edge(11, 12, 0, length=10, osmid=np.random.randint(1, 999999))
G.add_edge(12, 13, 0, length=10, osmid=np.random.randint(1, 999999))
G.add_edge(13, 14, 0, length=30, osmid=np.random.randint(1, 999999))

G.graph["crs"] = "EPSG:25832"
nodes, edges = ox.graph_to_gdfs(G)
edges["edge_id"] = edges["osmid"]

components = ef.return_components(G)
edges_comp_ids, comp_dict = ef.assign_component_id(
    components, edges, edge_id_col="osmid"
)

# Create test grid and joined data
grid = gpd.read_file("../tests/test_data/grid_component_test.gpkg", driver="GPKG")
edges_joined = gpd.overlay(edges, grid, how="intersection", keep_geom_type=True)

test_id_to_grid = ef.assign_component_id_to_grid(
    edges=edges,
    edges_joined_to_grids=edges_joined,
    components=components,
    grid=grid,
    prefix="osm",
    edge_id_col="osmid",
)

assert len(test_id_to_grid) == len(grid)
assert test_id_to_grid.loc[5, "component_ids_osm"][0] == 0
assert test_id_to_grid.loc[7, "component_ids_osm"][0] == 0
assert test_id_to_grid.loc[14, "component_ids_osm"][0] == 0
assert test_id_to_grid.loc[26, "component_ids_osm"][0] == 1
assert test_id_to_grid.loc[27, "component_ids_osm"][0] == 1
assert test_id_to_grid.loc[28, "component_ids_osm"][0] == 1
assert test_id_to_grid.loc[35, "component_ids_osm"][0] == 1
assert test_id_to_grid.loc[35, "component_ids_osm"][1] == 2
assert test_id_to_grid.loc[48, "component_ids_osm"][0] == 2
assert test_id_to_grid.loc[49, "component_ids_osm"][0] == 2


# Test count_component_cell_reach
G = nx.MultiDiGraph()
# One component
G.add_node(1, x=10, y=10)
G.add_node(2, x=20, y=20)
G.add_node(3, x=25, y=30)
G.add_node(4, x=25, y=40)
G.add_node(5, x=24, y=40)

# add length and osmid just for the functions to work
G.add_edge(1, 2, 0, length=10, osmid=np.random.randint(1, 999999))
G.add_edge(2, 3, 0, length=10, osmid=np.random.randint(1, 999999))
G.add_edge(3, 4, 0, length=10, osmid=np.random.randint(1, 999999))
G.add_edge(1, 5, 0, length=10, osmid=np.random.randint(1, 999999))

# Second component
G.add_node(6, x=50, y=50)
G.add_node(7, x=47, y=47)
G.add_node(8, x=53, y=50)
G.add_node(9, x=45, y=60)
G.add_node(10, x=44, y=60)

G.add_edge(6, 7, 0, length=10, osmid=np.random.randint(1, 999999))
G.add_edge(7, 8, 0, length=10, osmid=np.random.randint(1, 999999))
G.add_edge(8, 9, 0, length=30, osmid=np.random.randint(1, 999999))
G.add_edge(9, 10, 0, length=17, osmid=np.random.randint(1, 999999))

# Third component
G.add_node(11, x=53, y=55)
G.add_node(12, x=70, y=70)
G.add_node(13, x=80, y=85)
G.add_node(14, x=75, y=85)

G.add_edge(11, 12, 0, length=10, osmid=np.random.randint(1, 999999))
G.add_edge(12, 13, 0, length=10, osmid=np.random.randint(1, 999999))
G.add_edge(13, 14, 0, length=30, osmid=np.random.randint(1, 999999))

G.graph["crs"] = "EPSG:25832"
nodes, edges = ox.graph_to_gdfs(G)
edges["edge_id"] = edges["osmid"]

components = ef.return_components(G)

components_df = ef.component_lengths(components)

# Create test grid and joined data
grid = gpd.read_file("../tests/test_data/grid_component_test.gpkg", driver="GPKG")

edges_joined = gpd.overlay(edges, grid, how="intersection", keep_geom_type=True)
grid = ef.assign_component_id_to_grid(
    edges=edges,
    edges_joined_to_grids=edges_joined,
    components=components,
    grid=grid,
    prefix="osm",
    edge_id_col="osmid",
)

test_comp_cell_reach = ef.count_component_cell_reach(
    components_df, grid, "component_ids_osm"
)

assert len(test_comp_cell_reach) == len(components)
assert list(test_comp_cell_reach.keys()) == components_df.index.to_list()
assert test_comp_cell_reach[0] == 6
assert test_comp_cell_reach[1] == 4
assert test_comp_cell_reach[2] == 6


# Test find_overshoots function
G = nx.MultiDiGraph()  # construct the graph
G.add_node(1, x=10, y=10)
G.add_node(2, x=20, y=20)
G.add_node(3, x=25, y=30)
G.add_node(4, x=25, y=28)
G.add_node(5, x=20, y=15)

# add length and osmid just for the osmnx function to work
G.add_edge(1, 2, 0, length=10, osmid=np.random.randint(1, 999999))
G.add_edge(2, 3, 0, length=10, osmid=np.random.randint(1, 999999))
G.add_edge(3, 4, 0, length=2, osmid=np.random.randint(1, 999999))
G.add_edge(2, 5, 0, length=5, osmid=np.random.randint(1, 999999))

G.graph["crs"] = "epsg:25832"
nodes, edges = ox.graph_to_gdfs(G)
edges["length"] = edges.geometry.length
dn_nodes = ef.get_dangling_nodes(edges, nodes)

overshoots_2 = ef.find_overshoots(dn_nodes, edges, 2)
overshoots_5 = ef.find_overshoots(dn_nodes, edges, 5)

assert len(overshoots_2) == 1
assert len(overshoots_5) == 2
assert overshoots_2["u"].values[0] == 3
assert overshoots_2["v"].values[0] == 4
assert overshoots_5["u"].values[0] == 2
assert overshoots_5["v"].values[0] == 5
assert overshoots_5["u"].values[1] == 3
assert overshoots_5["v"].values[1] == 4


# Test find_undershoots function
G = nx.MultiDiGraph()  # construct the graph
G.add_node(1, x=1, y=1)
G.add_node(2, x=1, y=20)
G.add_node(3, x=1, y=30)
G.add_node(4, x=10, y=20)
G.add_node(5, x=20, y=20)
G.add_node(6, x=12, y=18)
G.add_node(7, x=12, y=1)
G.add_node(8, x=5, y=18)
G.add_node(9, x=5, y=1)
G.add_node(10, x=20, y=22)


# add length and osmid just for the osmnx function to work
G.add_edge(1, 2, 0, length=10, osmid=12)
G.add_edge(2, 3, 0, length=10, osmid=23)
G.add_edge(2, 4, 0, length=5, osmid=24)
G.add_edge(4, 5, 0, length=2, osmid=45)
G.add_edge(4, 6, 0, length=2, osmid=46)
G.add_edge(6, 7, 0, length=2, osmid=67)
G.add_edge(8, 9, 0, length=2, osmid=89)
G.add_edge(5, 10, 0, length=2, osmid=510)

G.graph["crs"] = "epsg:25832"

nodes, edges = ox.graph_to_gdfs(G)
edges["length"] = edges.geometry.length
edges["edge_id"] = edges.osmid
dangling_nodes = ef.get_dangling_nodes(edges, nodes)

undershoot_dict_3, undershoot_nodes_3 = ef.find_undershoots(
    dangling_nodes, edges, 3, "edge_id"
)

undershoot_dict_5, undershoot_nodes_5 = ef.find_undershoots(
    dangling_nodes, edges, 5, "edge_id"
)

assert len(undershoot_dict_3) == 1
assert len(undershoot_dict_5) == 3
assert list(undershoot_dict_3.keys()) == [8]
assert list(undershoot_dict_3.values()) == [[24]]

assert list(undershoot_dict_5.keys()) == [1, 8, 9]
assert list(undershoot_dict_5.values()) == [[89], [12, 24, 23], [12]]


print("All tests of evaluation functions passed!")
#%%
###################### TESTS FOR MATCHING FUNCTIONS #############################

# Test merge multiline function
line1 = LineString([[1, 0], [10, 0]])
line2 = LineString([[10, 0], [12, 0]])
multiline = MultiLineString([line1, line2])
geoms = [line1, line2, multiline]
test_gdf = gpd.GeoDataFrame(geometry=geoms)

test_gdf["geometry"] = test_gdf["geometry"].apply(lambda x: mf._merge_multiline(x))

assert test_gdf.geometry.geom_type.unique()[0] == "LineString"


# Test get_angle function
linestring1 = LineString([[0, 0], [10, 10]])
linestring2 = LineString([[0, 0], [10, 0]])
linestring3 = LineString([[10, 0], [0, 0]])

angle1 = mf._get_angle(linestring1, linestring2)

angle2 = mf._get_angle(linestring2, linestring1)

angle3 = mf._get_angle(linestring1, linestring3)

assert round(angle1, 5) == round(angle2, 5) == round(angle3, 5), "Angle test failed!"


# Test get_hausdorff_dist function
line1 = LineString([[1, 1], [10, 10]])
line2 = LineString([[2, 1], [4, 3]])
line3 = LineString([[4, 3], [2, 1]])

h1 = mf._get_hausdorff_dist(line1, line2)
h2 = mf._get_hausdorff_dist(line2, line1)
h3 = mf._get_hausdorff_dist(line1, line3)

h4 = LineString([[1, 1], [10, 1]])
h5 = LineString([[10, 1], [20, 1]])
h4 = mf._get_hausdorff_dist(h4, h5)

assert h1 == h2 == h3, "Hausdorff distance test failed!"
assert h4 == 10, "Hausdorff distance test failed!"


# Test overlay buffer matches function

ref = gpd.read_file("../tests/test_data/geodk_small_test.gpkg")
osm = gpd.read_file("../tests/test_data/osm_small_test.gpkg")

fot_id = 1095203923
index = ref.loc[ref.fot_id == fot_id].index.values[0]
correct_osm_matches_id = [17463, 17466, 17467, 17472, 17473, 58393, 58394]

buffer_matches = mf.overlay_buffer(
    reference_data=ref, osm_data=osm, dist=10, ref_id_col="fot_id", osm_id_col="osmid"
)

assert ["fot_id", "matches_id", "count"] == buffer_matches.columns.to_list()

assert type(buffer_matches) == gpd.geodataframe.GeoDataFrame

if len(buffer_matches) > 1:
    for b in buffer_matches["matches_id"].loc[index]:
        assert b in correct_osm_matches_id

    assert len(correct_osm_matches_id) == len(buffer_matches["matches_id"].loc[index])

else:
    for b in buffer_matches["matches_id"].loc[0]:
        assert b in correct_osm_matches_id

    assert len(correct_osm_matches_id) == len(buffer_matches["matches_id"].loc[0])


# Tests for get_segments function
test_line = LineString([[0, 0], [53, 0]])
segment_length = 8

lines = mf._get_segments(test_line, segment_length)

assert len(lines.geoms) == round(test_line.length / segment_length)

for l in lines.geoms:
    assert l.geom_type == "LineString"

for i in range(len(lines.geoms) - 1):
    l = lines.geoms[i]
    assert l.length == segment_length


# Test create segment gdf function
ref = gpd.read_file("../tests/test_data/geodk_small_test.gpkg")
seg_length = 5
test_segments = mf.create_segment_gdf(org_gdf=ref, segment_length=seg_length)
types = list(set(test_segments.geometry.geom_type))

assert types[0] == "LineString"
assert len(types) == 1

grouped = test_segments.groupby("fot_id")
for n, g in grouped:
    if len(g) > 1:
        for _, row in g.iterrows():
            assert (
                round(row.geometry.length, 1) <= seg_length * 1.35
            )  # A bit higher test value due to Shapely imprecision issues
            assert round(row.geometry.length, 1) >= seg_length / 3


# Test find best match function
ref = gpd.read_file("../tests/test_data/geodk_small_test.gpkg")
osm = gpd.read_file("../tests/test_data/osm_small_test.gpkg")

ref_segments = mf.create_segment_gdf(ref, segment_length=5)
osm_segments = mf.create_segment_gdf(osm, segment_length=5)

osm_segments.set_crs("EPSG:25832", inplace=True)
ref_segments.set_crs("EPSG:25832", inplace=True)
ref_segments.rename(columns={"seg_id": "seg_id_ref"}, inplace=True)


buffer_matches = mf.overlay_buffer(
    osm_data=osm_segments,
    reference_data=ref_segments,
    ref_id_col="seg_id_ref",
    osm_id_col="seg_id",
    dist=10,
)

matched_data = ref_segments.loc[buffer_matches.index].copy(deep=True)

matched_data["match"] = matched_data.apply(
    lambda x: mf._find_best_match(
        buffer_matches,
        ref_index=x.name,
        osm_edges=osm_segments,
        reference_edge=x["geometry"],
        angular_threshold=20,
        hausdorff_threshold=12,
    ),
    axis=1,
)

# Key is ref segments index, value is osm segment index
test_values = {13: 114, 14: 115, 15: 72, 44: 133, 12: 113, 22: 113, 23: 112}

for key, value in test_values.items():

    test_match = matched_data.loc[key, "match"]

    assert test_match == value, "Unexpected match!"


# Test find_matches_from_buffer function
ref_segments = gpd.read_file("../tests/test_data/ref_subset_segments.gpkg")
osm_segments = gpd.read_file("../tests/test_data/osm_subset_segments.gpkg")

osm_segments.set_crs("EPSG:25832", inplace=True)
ref_segments.set_crs("EPSG:25832", inplace=True)
ref_segments.rename(columns={"seg_id": "seg_id_ref"}, inplace=True)

buffer_matches = mf.overlay_buffer(
    osm_data=osm_segments,
    reference_data=ref_segments,
    ref_id_col="seg_id_ref",
    osm_id_col="seg_id",
    dist=10,
)
final_matches = mf.find_matches_from_buffer(
    buffer_matches=buffer_matches,
    osm_edges=osm_segments,
    reference_data=ref_segments,
    angular_threshold=20,
    hausdorff_threshold=15,
)

# Key is ref segment id, value is osm segment id
test_values = {
    1013: 1114,
    1014: 1115,
    1015: 1072,
    1044: 1133,
    1012: 1113,
    1022: 1113,
    1023: 1112,
}

for key, value in test_values.items():

    test_match = final_matches.loc[final_matches.seg_id_ref == key][
        "matches_id"
    ].values[0]

    assert test_match == value, "Unexpected match!"


# Test _find_matches_from_group
list1 = [1, 2, 3, 4, 5, 1, 2, 3, 5, 5]
list2 = [21, 45, 56, 78, 49, 77, 2, 44, 13, 6]
df = pd.DataFrame(data={"col1": list1, "col2": list2})

grouped = df.groupby("col1")
test1 = mf._find_matches_from_group(5, grouped, "col2")
assert test1 == [49, 13, 6]

test2 = mf._find_matches_from_group(2, grouped, "col2")
assert test2 == [45, 2]


print("All tests of matching functions passed!")
#%%
###################### TESTS FOR GRAPH FUNCTIONS #############################

# Test create osmnx graph function
test_data = gpd.read_file("../tests/test_data/geodk_test.gpkg")

test_graph = gf.create_osmnx_graph(test_data)

assert test_graph.is_directed() == True, "Failed test for create osmnx graph"

assert type(test_graph) == nx.classes.multidigraph.MultiDiGraph

nodes, edges = ox.graph_to_gdfs(test_graph)

assert len(test_data) == len(edges), "Failed test for create osmnx graph"

assert nodes.index.name == "osmid"

assert edges.index.names == ["u", "v", "key"]


# Test for unzip_linestrings

l1 = LineString([[1, 1], [5, 5], [10, 10], [15, 15]])
l2 = LineString([[2, 1], [6, 10], [200, 346]])
l3 = LineString([[10, 10], [10, 20], [52, 47]])

lines = [
    l1,
    l2,
    l3,
]
d = {
    "highway": ["cycleway", "primary", "secondary"],
    "edge_id": [1, 2, 3],
    "geometry": lines,
}

org_gdf = gpd.GeoDataFrame(d)

test = gf.unzip_linestrings(org_gdf, "edge_id")

assert org_gdf.crs == test.crs

for c in org_gdf.columns:
    assert c in test.columns

assert test.geometry.geom_type.unique()[0] == "LineString"

assert test.edge_id.to_list() == [1, 1, 1, 2, 2, 3, 3]
assert test.highway.to_list() == [
    "cycleway",
    "cycleway",
    "cycleway",
    "primary",
    "primary",
    "secondary",
    "secondary",
]
assert test.loc[0, "geometry"] == LineString([[1, 1], [5, 5]])
assert test.loc[2, "geometry"] == LineString([[10, 10], [15, 15]])
assert test.loc[6, "geometry"] == LineString([[10, 20], [52, 47]])


# Test find_parallel_edges
l1 = LineString([[1, 1], [10, 10]])
l2 = LineString([[2, 1], [6, 10]])
l3 = LineString([[10, 10], [10, 20]])
l4 = LineString([[11, 9], [5, 20]])
l5 = LineString([[1, 12], [4, 12]])
l6 = LineString([[11, 9], [5, 20]])

lines = [l1, l2, l3, l4, l5, l6]

# Correct, key values should not be modified
u = [1, 2, 3, 4, 2, 1]
v = [2, 3, 4, 1, 3, 2]
key = [0, 0, 0, 0, 1, 1]

d = {"u": u, "v": v, "key": key, "geometry": lines}
edges = gpd.GeoDataFrame(d)

edges_test = gf.find_parallel_edges(edges)

assert list(edges_test["key"].values) == key
assert len(edges) == len(edges_test)


# Incorrect, key values should be modified
u = [1, 2, 3, 4, 2, 1]
v = [2, 3, 4, 1, 3, 2]
key = [0, 0, 0, 0, 0, 0]

d = {"u": u, "v": v, "key": key, "geometry": lines}
edges = gpd.GeoDataFrame(d)
# edges.set_index(['u','v','key'],inplace=True)

edges_test = gf.find_parallel_edges(edges)

assert list(edges_test["key"].values) != key
k = list(edges_test["key"].values)
assert k[-1] == 1
assert k[-2] == 1
assert len(edges) == len(edges_test)

assert list(edges_test["key"].values) != key
assert len(edges) == len(edges_test)


# Incorrect, key values should be modified
l7 = LineString([[11, 9], [5, 20]])

lines.append(l7)

u = [1, 2, 3, 4, 2, 1, 1]
v = [2, 3, 4, 1, 3, 2, 2]
key = [0, 0, 0, 0, 0, 0, 0]

d = {"u": u, "v": v, "key": key, "geometry": lines}
edges = gpd.GeoDataFrame(d)

edges_test = gf.find_parallel_edges(edges)

assert list(edges_test["key"].values) != key
k = list(edges_test["key"].values)
assert k[-1] == 2
assert k[-2] == 1
assert k[-3] == 1

assert len(edges) == len(edges_test)

# Incorrect, key values should be modified
# Test that (u,v) is not treated as equal to (v,u)

u = [1, 2, 3, 4, 2, 1, 2]
v = [2, 3, 4, 1, 3, 2, 1]
key = [0, 0, 0, 0, 0, 0, 0]

d = {"u": u, "v": v, "key": key, "geometry": lines}
edges = gpd.GeoDataFrame(d)

edges_test = gf.find_parallel_edges(edges)

assert list(edges_test["key"].values) != key
k = list(edges_test["key"].values)
assert k[-1] == 0
assert k[-2] == 1
assert k[-3] == 1

assert len(edges) == len(edges_test)

print("All tests of graph functions passed!")
# %%
