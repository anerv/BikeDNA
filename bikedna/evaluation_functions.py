"""
This script contain all functions used for evaluating and describing network completeness and structure.
For functions used specifically for the feature matching, see matching_functions.py
"""

import geopandas as gpd
import pandas as pd
import os.path
import osmnx as ox
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def merge_results(grid, results_df, how):

    """
    Merges a dataframe with analysis results with the grid gdf.
    Checks if columns already have been merged - if they have, existing columns are dropped from the grid.

    Arguments:
        grid (gdf): grid data
        results_df (df): dataframe with results
        how (str): merge method. E.g. 'left' or 'inner'.

    Returns:
        grid (gdf): grid gdf with results columns merged
    """

    merge_cols = results_df.columns.to_list()
    merge_cols.remove("grid_id")

    for c in merge_cols:
        if c in grid.columns:
            grid.drop(c, axis=1, inplace=True)
            # print('Existing column dropped!')

    grid = grid.merge(results_df, on="grid_id", how=how)

    return grid


# def find_pct_diff(row, osm_col, ref_col):

#     """
#     Small helper function for computing rounded pct difference between two values.

#     Arguments:
#         row (row in pandas df): Row with values to be compared
#         osm_col, ref_col (str): Names of columns in rows to be compared

#     Returns:
#         bicycle_edges (gdf): Same dataframe with fixed key-values
#     """

#     if row.isnull().values.any() == True:

#         pass

#     else:
#         pct_diff = round(
#             (row[osm_col] - row[ref_col]) / ((row[osm_col] + row[ref_col]) / 2) * 100, 2
#         )

#         return pct_diff


def create_grid_geometry(gdf, cell_size):

    """
    Creates a geodataframe with grid cells covering the area specificed by the input gdf

    Arguments:
        gdf (gdf): geodataframe with a polygon/polygons defining the study area
        cell_size (numeric): width of the grid cells in units used by gdf crs

    Returns:
        grid (gdf): gdf with grid cells in same crs as input data
    """

    geometry = gdf["geometry"].unary_union
    geometry_cut = ox.utils_geo._quadrat_cut_geometry(geometry, quadrat_width=cell_size)

    grid = gpd.GeoDataFrame(geometry=[geometry_cut], crs=gdf.crs)

    grid = grid.explode(index_parts=False, ignore_index=True)

    # Create arbitraty grid id col
    grid["grid_id"] = grid.index

    return grid


def get_graph_area(nodes, study_area_polygon, crs):

    """
    Compute the size of the area covered by a graph (based on the convex hull)

    Arguments:
        nodes (gdf): geodataframe with graph nodes
        study_area_polygon (gdf): polygon defining the study area
        crs (str): CRS in format recognised by geopandas

    Returns:
        area (numeric): area in unit determined by crs (should be projected)
    """

    poly = nodes.unary_union.convex_hull  # Use convex hull for area computation
    poly_gdf = gpd.GeoDataFrame()
    poly_gdf.at[0, "geometry"] = poly
    poly_gdf = poly_gdf.set_crs(crs)

    area = poly_gdf.clip(study_area_polygon).area.values[
        0
    ]  # Clip in case convex hull goes beyond study area

    return area


def simplify_bicycle_tags(osm_edges, queries):

    """
    Function for creating two columns in gdf containing linestrings/network edges
    with bicycle infrastructure from OSM, indicating whether the bicycle infrastructure is
    bidirectional and whether it is mapped as true geometries or center lines.

    Does not take into account when there are differing types of bicycle infrastructure in both sides
    OBS! Some features might query as True for seemingly incompatible combinations

    Arguments:
        osm_edges (gdf): geodataframe with linestrings with bicycle infrastructure from OSM
        queries (dict): dictionary with queries used to define bicycle infrastructure as centerline mappings and bidirectional

    Returns:
        osm_edges (gdf): same gdf + two new columns
    """

    osm_edges["bicycle_bidirectional"] = None
    osm_edges["bicycle_geometries"] = None

    # The order of the queries matter: To account for instances where highway=cycleway and cycleway=some value indicating bicycle infrastructure, the queries classifying based on highway should be run lastest
    for c in queries["centerline_true_bidirectional_true"]:
        ox_filtered = osm_edges.query(c)
        osm_edges.loc[ox_filtered.index, "bicycle_bidirectional"] = True
        osm_edges.loc[ox_filtered.index, "bicycle_geometries"] = "centerline"

    for c in queries["centerline_true_bidirectional_false"]:
        ox_filtered = osm_edges.query(c, engine="python")
        osm_edges.loc[ox_filtered.index, "bicycle_bidirectional"] = False
        osm_edges.loc[ox_filtered.index, "bicycle_geometries"] = "centerline"

    for c in queries["centerline_false_bidirectional_false"]:
        ox_filtered = osm_edges.query(c)
        osm_edges.loc[ox_filtered.index, "bicycle_bidirectional"] = False
        osm_edges.loc[ox_filtered.index, "bicycle_geometries"] = "true_geometries"

    for c in queries["centerline_false_bidirectional_true"]:
        ox_filtered = osm_edges.query(c)
        osm_edges.loc[ox_filtered.index, "bicycle_bidirectional"] = True
        osm_edges.loc[ox_filtered.index, "bicycle_geometries"] = "true_geometries"

    # Assert that bicycle bidirectional and bicycle geometries have been filled out for all where bicycle infrastructure is yes!
    assert (
        len(
            osm_edges.query(
                "bicycle_infrastructure =='yes' & (bicycle_bidirectional.isnull() or bicycle_geometries.isnull())"
            )
        )
        == 0
    ), "Not all bicycle infrastructure has been classified!"

    return osm_edges



def define_protected_unprotected(bicycle_edges, classifying_dictionary):

    """
    Function for classifying rows in gdf containing linestrings/network edges
    with bicycle infrastructure from OSM as either protected or unprotected.
    Input dictionary with queries used in classification should have keys corresponding to types of protection level.
    Each key should contain a list of queries

    Arguments:
        bicycle_edges (gdf): geodataframe with linestrings with bicycle infrastructure from OSM
        classifying_dictionary (gdf): dictionary with queries used in classification.

    Returns:
        bicycle_edges (gdf): same gdf +  new column 'protected'
    """

    bicycle_edges["protected"] = None

    for type, queries in classifying_dictionary.items():

        # Check if there already is a value for protected
        not_classified = bicycle_edges[bicycle_edges.protected.isna()]
        already_classified = bicycle_edges[bicycle_edges.protected.notna()]

        for q in queries:

            filtered_not_classified = not_classified.query(q)
            filtered_already_classified = already_classified.query(q)

            bicycle_edges.loc[filtered_not_classified.index, "protected"] = type
            bicycle_edges.loc[filtered_already_classified.index, "protected"] = "mixed"

    # Assert that bicycle bidirectional and bicycle geometries have been filled out for all where bicycle infrastructure is yes!
    assert (
        len(bicycle_edges.query("protected.isnull()")) == 0
    ), "Not all bicycle infrastructure has been classified!"

    return bicycle_edges


def measure_infrastructure_length(
    edge, geometry_type, bidirectional, bicycle_infrastructure
):

    """
    Measure the infrastructure length of edges with bicycle infrastructure.
    If an edge represents a bidirectional lane/path or infrasstructure on both sides on a street,
    the infrastructure is set to two times the geometric length.
    If onesided/oneway, infrastructure length == geometric length.
    ...

    Arguments:
        edge (LineString): geometry of infrastructure
        geometry_type (str): variable used to determine if two-way or not.
                            Can be either a variable for whole dataset OR name of column with the variable
        bidirectional (str): variable used to determine if two-way or not.
                            Can be either a variable for whole dataset OR name of column with the variable
        bicycle_infrastructure: variable used to define bicycle infrastructure if datasets included non-bicycle infrastructure edges.
                              Can be either a variable for whole dataset OR name of column with the variable

    Returns:
        infrastructure_length (numeric): length of infrastructure
    """
    edge_length = edge.length

    if (
        bicycle_infrastructure == "yes"
        and geometry_type == "true_geometries"
        and bidirectional == True
    ):
        infrastructure_length = edge_length * 2

    elif (
        bicycle_infrastructure == "yes"
        and geometry_type == "true_geometries"
        and bidirectional == False
    ):
        infrastructure_length = edge_length

    elif (
        bicycle_infrastructure == "yes"
        and geometry_type == "centerline"
        and bidirectional == True
    ):
        infrastructure_length = edge_length * 2

    elif (
        bicycle_infrastructure == "yes"
        and geometry_type == "centerline"
        and bidirectional == False
    ):
        infrastructure_length = edge_length

    elif bicycle_infrastructure == "yes" and (
        geometry_type is None or bidirectional is None
    ):
        print("Missing information when calculating true infrastructure length!")
        infrastructure_length = None

    else:
        infrastructure_length = None

    return infrastructure_length


def analyze_existing_tags(gdf, dict):

    """
    Analyse the extent of existing/missing tags in a gdf with data from OSM based on custom dictionary.
    Custom dictionary should be a nested dict with top keys indicating the general attribute to be analysed.
    Next layer in the dictionary should have the keys true_geometries, centerline or all.
    These keys should contain lists of specific OSM tags to be checked.
    For example, when looking at surface, the surface of a bikelane mapped as centerline can be found under the tag
    'cycleway_surface', while it for a bikelane mapped as true geometry can be found under the osm tag 'surface'.

    Note that all ':' in osm tags have been replaced with '_'
    Look at the projects README for further explanation/illustration of concepts.
    ...

    Arguments:
        edges (gdf): data to be checked for missing tags
        dict (dictionary): dictionary defining the tags to be checked.

    Returns:
        results (dictionary): dictionary with the number and length of features with a given tag
    """

    cols = gdf.columns.to_list()

    results = {}

    for attribute, sub_dict in dict.items():

        attr_results = {}

        attr_results["count"] = 0
        attr_results["length"] = 0
        count_not_na = None
        len_not_na = None

        for geom_type, tags in sub_dict.items():

            tags = [t for t in tags if t in cols]

            if geom_type == "true_geometries":

                subset = gdf.loc[gdf.bicycle_geometries == "true_geometries"]

            elif geom_type == "centerline":

                subset = gdf.loc[gdf.bicycle_geometries == "centerline"]

            elif geom_type == "all":

                subset = gdf

            if len(tags) == 1:

                count_not_na = len(subset.loc[subset[tags[0]].notna()])
                len_not_na = subset.loc[subset[tags[0]].notna()].geometry.length.sum()

            elif len(tags) > 1:

                count_not_na = len(subset[subset[tags].notna().any(axis=1)])
                len_not_na = subset.loc[subset[tags[0]].notna()].geometry.length.sum()

            else:
                count_not_na = 0
                len_not_na = 0

            attr_results["count"] += count_not_na
            attr_results["length"] += len_not_na

        results[attribute] = attr_results

    return results


def check_incompatible_tags(edges, incompatible_tags_dictionary, store_edge_ids=False):

    """
    Check incompatible tags in gdf with data from OSM.
    Input dictionary should be a nested dictionary, with the top level keys indicating columns to be analysed,
    and the sub-dicts keys indicating the column values to be analysed.
    Each sub-dict key should refer to a list or tupple with first a column to compare to and secondly a column value
    that is incompatible with the value defined in the second layer key.

    Arguments:
        edges (gdf): gdf with data to be analysed
        incompatible_tags_dictionary (dict): dictionary with tag combinations to analyse
        store_edge_ids (True/False): setting for whether to return ids of edges w. incompatible tags

    Returns:
        results (dict): dictionary with the count of incompatible tags for each type
    """

    cols = edges.columns.to_list()
    results = {}

    for tag, subdict in incompatible_tags_dictionary.items():

        for value, combinations in subdict.items():

            for c in combinations:

                if c[0] in cols and store_edge_ids == False:
                    results[tag + "/" + c[0]] = 0
                    count = len(
                        edges.loc[(edges[tag] == value) & (edges[c[0]] == c[1])]
                    )
                    results[tag + "/" + c[0]] += count
                elif c[0] in cols and store_edge_ids == True:
                    results[tag + "/" + c[0]] = list(
                        edges["edge_id"].loc[
                            (edges[tag] == value) & (edges[c[0]] == c[1])
                        ]
                    )

    return results


def check_intersection(row, gdf, print_check=False):

    """
    Detects topological errors in gdf with edges from OSM data.
    If two edges are intersecting (i.e. no node at intersection) and neither is tagged as a bridge or a tunnel,
    it is considered an error in the data.

    Arguments:
        row (row): row currently analysed
        gdf (gdf): gdf with other edges to check for intersections iwth
        print_check: setting for whether to print when a missing intersection node is detected

    Returns:
        count (int): number of intersection issues for each row
    """

    intersection = gdf[gdf.crosses(row.geometry)]

    if (
        len(intersection) > 0
        and (pd.isnull(row.bridge) == True or row.bridge == "no")
        and (pd.isnull(row.tunnel) == True or row.bridge == "no")
    ):

        count = 0

        for _, r in intersection.iterrows():

            if (pd.isnull(r.bridge) == True or r.bridge == "no") and (
                pd.isnull(r.tunnel) == True or r.bridge == "no"
            ):

                if print_check:
                    print("Found problem!")

                count += 1

        if count:
            if count > 0:
                return count


def check_crossing(row, gdf):

    """
    Detects whether a row from a gdf with line geomtries intersects with any of the geometries in a gdf

    Arguments:
        row (row): row currently analysed
        gdf (gdf): gdf with other edges/geometries to check for intersections iwth

    Returns:
        count (int): number of intersection issues for each row
    """

    intersection = gdf[gdf.crosses(row.geometry)]

    intersection_issues_count = len(intersection)

    return intersection_issues_count


def find_missing_intersections(edges, edge_id_col, return_edges=True):

    """
    Detects topological errors in gdf with edges from OSM data.
    If two edges are intersecting (i.e. no node at intersection) and neither is tagged as a bridge or a tunnel,
    it is considered an error in the data.
    """

    # Don't include tunnels or bridges
    edges_subset = edges.loc[
        ~(
            edges.tunnel.isin(
                ["yes", "Yes", True, "passage", "building_passage", "movable"]
            )
            | edges.bridge.isin(
                ["yes", "Yes", True, "passage", "building_passage", "movable"]
            )
        )
    ].copy()

    edges_subset["intersection_issues"] = edges_subset.apply(
        lambda x: check_crossing(row=x, gdf=edges_subset), axis=1
    )

    missing_nodes = list(
        edges_subset.loc[
            (edges_subset.intersection_issues.notna())
            & edges_subset.intersection_issues
            > 0
        ][edge_id_col].values
    )

    edges_missing_node = edges_subset.loc[
        (edges_subset.intersection_issues.notna()) & edges_subset.intersection_issues
        > 0
    ]

    if return_edges:
        return missing_nodes, edges_missing_node

    else:
        return missing_nodes


def compute_alpha_beta_gamma(edges, nodes, G, planar=True):

    """
    Computes alpha, beta and gamma for a network

    Arguments:
        edges (gdf): network edges
        nodes (gdf): network nodes
        G (networkx graph): network for computing number of components
        planar: whether network is (approx.) planar or not

    Returns:
        alpha, beta, gamma (float): values for each metric

    """

    e = len(edges)
    v = len(nodes)

    assert edges.geom_type.unique()[0] == "LineString"
    assert nodes.geom_type.unique()[0] == "Point"

    p = nx.number_connected_components(G)

    if planar:
        alpha = (e - v + p) / (2 * v - 5)

    else:
        alpha = (e - v) / ((v * (v - 1) / 2) - (v - 1))

    assert alpha >= 0 and alpha <= 1

    beta = e / v

    if beta > 3:
        print("Unusually high beta value!")

    if planar:
        gamma = e / (3 * (v - 2))

    else:
        gamma = e / ((v * (v - 1)) / 2)

    assert gamma >= 0 and gamma <= 1

    return alpha, beta, gamma


def compute_edge_node_ratio(data_tuple):

    """
    Computes the ratio between edges and nodes in a network.

    Arguments:
        data_tuple (tuple): tuple with edges and nodes

    Returns:
        ratio (float): the ratio between edges and nodes

    """

    edges, nodes = data_tuple

    e = len(edges)
    v = len(nodes)

    if len(nodes) > 0 and len(edges) > 0:

        assert edges.geom_type.unique()[0] in ["LineString", "MultiLineString"]
        assert nodes.geom_type.unique()[0] == "Point"

        # Compute alpha # between 0 and 1
        ratio = e / v

        return ratio


def return_components(graph):

    """
    Return all connected components as list of individual graphs.
    If the graph is directed, an undirected version of the graph is used.

    Arguments:
        graph (gdf): networkx graph

    Returns:
        graphs (list): list with all connected components as networkx graphs

    """

    if nx.is_directed(graph) == True:

        G_un = ox.get_undirected(graph)

        graphs = [graph.subgraph(c).copy() for c in nx.connected_components(G_un)]
        # print('{} graphs have been generated from the connected components'.format(len(graphs)))

    else:

        graphs = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
        # print('{} graphs have been generated from the connected components'.format(len(graphs)))

    return graphs


def component_lengths(components):

    """
    Return the length of all components

    Arguments:
        components (list): list of networkx graphs

    Returns:
        components_df (df): dataframe with length of components and the component id
                            (component id = position in input list with components)

    """

    components_length = {}

    for i, c in enumerate(components):

        c_length = 0

        for (_, _, l) in c.edges(data="length"):

            c_length += l

        components_length[i] = c_length

    components_df = pd.DataFrame.from_dict(components_length, orient="index")

    components_df.rename(columns={0: "component_length"}, inplace=True)

    return components_df


def get_dangling_nodes(network_edges, network_nodes):

    """
    Return all dangling (degree one) nodes in a network
    Assumes an undirected network - if two parallel nodes representing a two-way street end in the same
    node, it will not be considered dangling.

    Arguments:
        network edges (gdf): gdf with network edges indexed by their start and end nodes
        network nodes (gdf): gdf with network nodes

    Returns:
        dangling_nodes (gdf): geodataframe with all dangling nodes
    """

    edges = network_edges.copy(deep=True)
    nodes = network_nodes.copy(deep=True)

    if "u" not in edges.columns:

        all_node_occurences = (
            edges.reset_index().u.to_list() + edges.reset_index().v.to_list()
        )

    else:

        all_node_occurences = edges.u.to_list() + edges.v.to_list()

    dead_ends = [x for x in all_node_occurences if all_node_occurences.count(x) == 1]

    dangling_nodes = nodes[nodes.index.isin(dead_ends)]

    return dangling_nodes


def count_features_in_grid(joined_data, label):

    """
    Count the number of features in each grid cell based on a dataset already joined to the grid
    (i.e. having a column with reference to intersecting grid cell ids)

    Arguments:
        joined_data (gdf): gdf with network edges indexed by their start and end nodes
        label (str): name of joined_data for naming column with feature count

    Returns:
        count_df (df): dataframe with the count of all features in each grid cell, indexed by grid cell id
    """

    count_features_in_grid = {}
    grouped = joined_data.groupby("grid_id")

    for name, group in grouped:
        count_features_in_grid[name] = len(group)

    count_df = pd.DataFrame.from_dict(count_features_in_grid, orient="index")
    count_df.reset_index(inplace=True)
    count_df.rename(columns={"index": "grid_id", 0: f"count_{label}"}, inplace=True)

    return count_df


def length_features_in_grid(joined_data, label):

    """
    Count the length of the features in each grid cell based on a dataset already joined to the grid
    (i.e. having a column with reference to intersecting grid cell ids)

    Arguments:
        joined_data (gdf): gdf with network edges indexed by their start and end nodes
        label (str): name of joined_data for naming column with feature length

    Returns:
        len_df (df): dataframe with the length of all features in each grid cell, indexed by grid cell id
    """

    len_features_in_grid = {}
    grouped = joined_data.groupby("grid_id")

    for name, group in grouped:
        len_features_in_grid[name] = group.geometry.length.sum()

    len_df = pd.DataFrame.from_dict(len_features_in_grid, orient="index")
    len_df.reset_index(inplace=True)
    len_df.rename(columns={"index": "grid_id", 0: f"length_{label}"}, inplace=True)

    return len_df


def length_of_features_in_grid(joined_data, label):

    """
    Compute the length of all features of one dataset for each grid cell.
    Assumes that the data have already been joined to the grid (all rows must have a reference to the intersecting grid cell id)
    OBS! Returns geometric length, not infrastructure length

    Arguments:
        joined_data (gdf): gdf with network edges indexed by their start and end nodes
        label (str): name of joined_data for naming column with feature count

    Returns:
        len_df (df): dataframe with the length of all features in each grid cell, indexed by grid cell id
    """

    features_in_grid_length = {}
    grouped = joined_data.groupby("grid_id")

    for name, group in grouped:
        features_in_grid_length[name] = group.geometry.length.sum()

    len_df = pd.DataFrame.from_dict(features_in_grid_length, orient="index")
    len_df.reset_index(inplace=True)
    len_df.rename(columns={"index": "grid_id", 0: f"length_{label}"}, inplace=True)

    return len_df


def compute_network_density(data_tuple, area, return_dangling_nodes=False):

    """
    Compute the network density (edge density, node density and optionally, dangling ndoe density)
    Uses infrastructure length, not geometric length.
    Requires a column with infrastructure length in edges dataset.
    The function is written to be used to compute local network density on a grid cell level.

    Arguments:
        data_tuple (tuple): tuple with gdfs with network edges and nodes
        area (numeric): area of study area in square meters
        return_dangling_nodes (boolean): Set to True if the density of the dangling nodes should be returned

    Returns:
        edge_density, node_density (dangling_node_density) (numeric): network densities
    """

    area = area / 1000000

    edges, nodes = data_tuple

    assert "infrastructure_length" in edges.columns

    if len(edges) > 0:

        edge_density = edges.infrastructure_length.sum() / area

    else:
        edge_density = 0

    if len(nodes) > 0:

        node_density = len(nodes) / area

    else:
        node_density = 0

    if return_dangling_nodes and len(nodes) > 0:

        dangling_nodes = get_dangling_nodes(edges, nodes)

        dangling_node_density = len(dangling_nodes) / area

        return edge_density, node_density, dangling_node_density

    else:
        return edge_density, node_density


def find_adjacent_components(components, edge_id, buffer_dist, crs):

    """
    Find edges in different (unconnected) components that are within a specified distance from each other.

    Arguments:
        components (list): list with network components (networkx graphs)
        edge_id (str): name of column with unique edge id
        buffer_dist (numeric): max distance for which edges in different components are considered 'adjacent'
        crs (str): crs to use when computing distances between edges

    Returns:
        all_results (dict): dictionary with the ids for all edge pairs identifying as overlapping based on their buffers, and the centroid of the buffer intersection
    """

    edge_list = []

    for i, c in enumerate(components):

        if len(c.edges) > 0:

            edges = ox.graph_to_gdfs(c, nodes=False)

            edges["component"] = i

            edge_list.append(edges)

    component_edges = pd.concat(edge_list)

    component_edges = component_edges.set_crs(crs)

    # Buffer component edges and find overlapping buffers
    component_edges_buffer = component_edges.copy()
    component_edges_buffer.geometry = component_edges_buffer.geometry.buffer(
        buffer_dist / 2
    )

    component_buffer_sjoin = gpd.sjoin(component_edges_buffer, component_edges_buffer)
    # Drop matches between edges on the same component
    intersecting_buffer_components = component_buffer_sjoin.loc[
        component_buffer_sjoin.component_left != component_buffer_sjoin.component_right
    ].copy()

    left_ids = intersecting_buffer_components[edge_id + "_left"].to_list()
    right_ids = intersecting_buffer_components[edge_id + "_right"].to_list()

    adjacent_edges_all = list(zip(left_ids, right_ids))
    adjacent_edges_all = [set(a) for a in adjacent_edges_all]

    # Remove duplicaties
    adjacent_edges = []
    for item in adjacent_edges_all:
        if item not in adjacent_edges:
            adjacent_edges.append(item)

    adjacent_edges = [tuple(a) for a in adjacent_edges]

    all_results = {}

    for i in range(len(adjacent_edges)):
        intersection = gpd.overlay(
            component_edges_buffer.loc[
                component_edges_buffer[edge_id] == adjacent_edges[i][0]
            ],
            component_edges_buffer.loc[
                component_edges_buffer[edge_id] == adjacent_edges[i][1]
            ],
            how="intersection",
        )

        geom = intersection.geometry.centroid.values[0]

        results = {}
        results[edge_id + "_left"] = adjacent_edges[i][0]
        results[edge_id + "_right"] = adjacent_edges[i][1]
        results["geometry"] = geom

        all_results[i] = results

    return all_results


def assign_component_id(components, edges, edge_id_col):

    """
    Assign a component id to all edges

    Arguments:
        components (list): list with network components (networkx graphs)
        edge_id_col (str): name of column with unique edge id
        edges (gdf): edges from network used to generate components

    Returns:
        edges (gdf): edges with a new column 'component' referencing the component the edge is part of
        components_dict (dict): dictionary with all components indexed by the component ids used in the edge component column
    """

    components_dict = {}
    edge_list = []

    org_edge_len = len(edges)

    for i, c in enumerate(components):

        if len(c.edges) < 1:
            print("Empty component")

        elif len(c.edges) > 0:

            c_edges = ox.graph_to_gdfs(c, nodes=False)

            c_edges["component"] = i

            edge_list.append(c_edges)

            components_dict[i] = c

    component_edges = pd.concat(edge_list)

    component_edges[edge_id_col] = component_edges[edge_id_col].astype(int)

    joined_edges = edges.merge(
        component_edges[["component", edge_id_col]], on=edge_id_col, how="left"
    )

    assert org_edge_len == len(joined_edges), "Some edges have been dropped!"

    assert (
        len(joined_edges.loc[joined_edges.component.isna()]) == 0
    ), "Not all edges have a component ID"

    return joined_edges, components_dict


def assign_component_id_to_grid(
    edges, edges_joined_to_grids, components, grid, prefix, edge_id_col
):

    """
    Join component id to intersecting grid cells.

    Arguments:
        simplified_edges (gdf): network edges
        edges_joined_to_grid (gdf): network edges joined with grid (must have a column 'grid_id' identifying intersecting grid cell)
        components (list): list of network components (networkx graphs)
        grid (gdf): grid used for the local analysis
        prefix (str): label to identify the network type (e.g. 'OSM' or 'reference')
        edge_id_col (str): name of column with unique edge id

    Returns:
        grid (gdf): grid gdf with column ('component_ids_'+prefix) identifying components in each cell
    """

    org_grid_len = len(grid)

    edges, _ = assign_component_id(components, edges, edge_id_col)

    edges_joined_to_grids = edges_joined_to_grids.merge(
        edges[["component", "edge_id"]], on="edge_id", how="right"
    )

    grouped = edges_joined_to_grids.groupby("grid_id")

    grid_components = {}

    for g_id, data in grouped:
        comp_ids = list(set(data.component))

        grid_components[g_id] = comp_ids

    grid_comp_df = pd.DataFrame()
    grid_comp_df["component_ids_" + prefix] = None

    for key, val in grid_components.items():
        grid_comp_df.at[key, "component_ids_" + prefix] = val

    grid_comp_df.reset_index(inplace=True)
    grid_comp_df.rename(columns={"index": "grid_id"}, inplace=True)

    grid = grid.merge(grid_comp_df, on="grid_id", how="left")

    assert len(grid) == org_grid_len

    return grid


def run_grid_analysis(grid_id, data, results_dict, func, *args, **kwargs):

    """
    Run a function for each cell in a spatial grid.
    This works for functions which returns a list, dict, value etc. - but not for functions that return a dataframe, since the meta-function does not return anything
    There will be some over-counting comparen to non-grid analysis due to edges being clipped by the grid cell boundaries

    Arguments:
        grid_id (undefined): unique identifier for grid cell
        data (gdf/df/tuple): data to be used in the analysis. Each row must have a key referencing the intersecting grid cell's id.
        results_dict (dict): dictionary to store results in
        func (function): name of function to run
        *args (undefined): additional parameters
        **kwargs (undefined): additional parameters

    Returns:
        None
    """

    if type(data) == tuple:

        edges, nodes = data

        grid_edges = edges.loc[edges.grid_id == grid_id]
        grid_nodes = nodes.loc[nodes.grid_id == grid_id]

        if len(grid_edges) > 0 or len(grid_nodes) > 0:

            grid_data = (grid_edges, grid_nodes)

            result = func(grid_data, *args, *kwargs)

            results_dict[grid_id] = result

            return result

        else:
            pass

    else:
        grid_data = data.loc[data.grid_id == grid_id]

        if len(grid_data) > 0:

            result = func(grid_data, *args, *kwargs)

            results_dict[grid_id] = result

            return result

        else:
            pass


def count_component_cell_reach(components_df, grid, component_id_col_name):

    """
    Count the number of cells in a grid reached/intersected by each component.

    Arguments:
        components_df (df): dataframe returned by component_lengths function (df with length of components and the component id)
        component_id_col_name (str): name of column with component ids in components_df

    Returns:
        component_cell_count (df): dataframe with count of reachable cells for each component
    """

    component_ids = components_df.index.to_list()

    component_cell_count = {}

    for c_id in component_ids:

        selector = [str(c_id)]

        col_name_str = component_id_col_name + "_str"
        grid[col_name_str] = grid[component_id_col_name].apply(
            lambda x: [str(i) for i in x] if type(x) == list else x
        )

        mask = grid[col_name_str].apply(
            lambda x: any(item for item in selector if item in x)
            if type(x) == list
            else False
        )

        selection = grid[mask]

        component_cell_count[c_id] = len(selection)

    grid.drop(col_name_str, axis=1, inplace=True)

    return component_cell_count


def count_cells_reached(component_lists, component_cell_count_dict):

    """
    Count the number of cells in a grid reachable from each grid cell, based on the network components intersected by that grid cell.

    Arguments:
        component_lists (list): list with network components as networkx graphs
        component_cell_count_dict (dict): dictionary with the count of cells reached by each component

    Returns:
        cell_count (int): count of reachable cells
    """

    cell_count = sum([component_cell_count_dict.get(c) for c in component_lists])

    # Subtract one since this count also includes the cell itself?
    # cell_count = cell_count - 1

    return cell_count


def find_overshoots(
    dangling_nodes, edges, length_tolerance, return_overshoot_edges=True
):

    """
    Find overshoots in a network based on specified tolerance.

    Arguments:
        dangling_nodes (gdf): gdf with dangling nodes in network
        edges (gdf): gdf with network edges
        length_tolerance (numeric): threshold for when an edge is considered an overshoot
        return_undershoot_edges (boolean): Set to False if the edges identified as overshoots should not be returned.
                                            If it is set to False, the index of the edges will be returned instead.

    Returns:
        overshoot_edges (gdf): gdf with edges identified as overshoots
        overshoot_ix (list): list with index values of edges identified as overshoots
    """

    # Get index of all dangling nodes
    dn_index = dangling_nodes.index.to_list()

    if "u" not in edges.columns:

        u_list = edges.reset_index().u.to_list()
        v_list = edges.reset_index().v.to_list()

        edges["u"] = u_list
        edges["v"] = v_list

    # Get edges connected to a dangling node
    subset_edges = edges.loc[edges.u.isin(dn_index) | edges.v.isin(dn_index)]

    # Select dangling node edges with a length below threshold (overshoot edges)
    overshoots = subset_edges.loc[subset_edges.length <= length_tolerance].copy()

    # If both u and v are in dangling nodes, remove from overshoots
    short_edges = overshoots.loc[
        overshoots.u.isin(dn_index) & overshoots.v.isin(dn_index)
    ]
    short_edges_ix = short_edges.index

    overshoots.drop(short_edges_ix, inplace=True)

    # Get index of overshoot edges
    overshoot_ix = overshoots.index.to_list()

    if return_overshoot_edges:

        return overshoots

    else:
        return overshoot_ix


def find_undershoots(
    dangling_nodes, edges, length_tolerance, edge_id_col, return_undershoot_nodes=True
):

    """
    Find undershoots in a network, based on specified tolerance.

    Arguments:
        dangling_nodes (gdf): gdf with dangling nodes in network
        edges (gdf): gdf with network edges
        length_tolerance (numeric): threshold for when a node is considered an undershoot
        edge_id_col (str): name of column with unique edge id in edges gdf
        return_undershoot_nodes (boolean): Set to False if the nodes identified as undershoots should not be returned

    Returns:
        undershoots (dict): dictionary with nodes identified as undershoots.
        undershoot_nodes (gdf): node identified as undershoots. Dictionary key is osmid, values are edges that nodes (potentially) should be snapped to
    """

    # For each node in dangling nodes, get all edges within specified distance
    dangling_nodes["osmid"] = dangling_nodes.index
    buffered_nodes = dangling_nodes[["osmid", "geometry"]].copy(deep=True)
    buffered_nodes["geometry"] = buffered_nodes.geometry.buffer(length_tolerance)

    if "v" not in edges.columns:

        u_list = edges.reset_index().u.to_list()
        v_list = edges.reset_index().v.to_list()

        edges["u"] = u_list
        edges["v"] = v_list

    joined = buffered_nodes.sjoin(edges, how="left")

    grouped = joined.groupby("osmid_left")

    undershoots = {}

    for name, group in grouped:

        # Check if there are edges within the threshold distance that are not connected to the node
        if (
            len(
                group.loc[(group.osmid_left != group.u) & (group.osmid_left != group.v)]
            )
            > 0
        ):

            group.reset_index(inplace=True)

            # Get all edges connected to the node
            connected_edges = edges.loc[(edges.u == name) | (edges.v == name)]

            # Get their other node
            connected_edges_nodes = list(
                set(connected_edges.u.to_list() + connected_edges.v.to_list())
            )

            connected_edges_nodes.remove(name)

            # Get adjacent edges and remove them from potential undershoot issues
            adjacent_edges = group.loc[
                (group.u.isin(connected_edges_nodes))
                | (group.v.isin(connected_edges_nodes))
            ]
            group.drop(adjacent_edges.index, inplace=True)

            # Remove edges connected to the node
            group = group.loc[
                (group.osmid_left != group.u) & (group.osmid_left != group.v)
            ]

            if len(group) > 0:
                unconnected_edge_ids = group[edge_id_col].to_list()

                undershoots[name] = unconnected_edge_ids

    if return_undershoot_nodes:
        return (
            undershoots,
            dangling_nodes.loc[dangling_nodes.osmid.isin(undershoots.keys())],
        )

    else:
        return undershoots


def get_component_edges(components, crs):

    """
    Get all edges from one or more network components, including a reference to their component's id

    Arguments:
        components (list): list of components as networkx graphs
        crs (str): crs for the graph data

    Returns:
        edges (gdf): edges from all components with the column 'component_id' identifying the component they belonged to.
        (id is based on position in list of components)
    """

    comp_ids = []
    edge_geometries = []

    for i, c in enumerate(components):

        if len(c.edges) > 0:

            attr = nx.get_edge_attributes(c, "geometry")

            geoms = list(attr.values())
            edge_geometries = edge_geometries + geoms

            ids = [i] * len(geoms)

            comp_ids = comp_ids + ids

    assert len(comp_ids) == len(edge_geometries)

    edges = gpd.GeoDataFrame(
        data={"component_id": comp_ids}, geometry=edge_geometries, crs=crs
    )

    return edges


def _node_degrees_as_df(graph):

    """
    Return the degree for all nodes in a networkx graph as a pandas dataframe.

    Arguments:
        graph (networkx graph): graph to compute node degree for

    Returns:
        degree_df (df): dataframe indexed by node id
    """

    degrees = {}

    for node, degree in graph.degree():
        degrees[node] = degree

    degree_df = pd.DataFrame.from_dict(degrees, orient="index", columns=["degree"])

    return degree_df


def get_average_node_degree(graph, joined_nodes, prefix=None):

    """
    Return the average node degree for each grid cell.
    Nodes must have a column 'grid_id' referencing the id of their respective grid cell.

    Arguments:
        graph (networkx graph): graph   to compute node degree for
        joined_nodes (gdf): network nodes
        prefix (str): optional prefix to add to column names

    Returns:
        ave_deg (df): dataframe with columns with grid id and with the average degree of all nodes in the grid cell
    """

    degree_df = _node_degrees_as_df(graph)

    merged_degrees = (
        joined_nodes[["osmid", "grid_id"]]
        .set_index("osmid")
        .merge(degree_df, left_index=True, right_index=True, how="left")
    )

    ave_deg = merged_degrees.groupby("grid_id").mean().round(decimals=3)

    ave_deg.reset_index(inplace=True)
    ave_deg.rename({"degree": "ave_degree"}, axis=1, inplace=True)

    if prefix:
        ave_deg.rename({"ave_degree": prefix + "_ave_degree"}, axis=1, inplace=True)

    return ave_deg
