"""
The functions defined below are used for creating creating and modifying networkx graphs using the osmnx format for indexing edges and nodes
"""

import pandas as pd
import geopandas as gpd
from shapely.ops import linemerge
from shapely.geometry import LineString
import momepy
import osmnx as ox


def update_key_values(bicycle_graph):
    
    """
    Make sure that no edges exist with key=1 in the multiindex, without a corresponding key=0 (this can occur due to how the bicycle graph is created).
    If key=1 edges without a parallel corresponding edges are found, the key value is set to 0.
    If this is not done, some OSMnx function will not function correctly.

    Arguments:
        bicycle_graph (networkx/osmnx graph): graph to check for edges with wrong key-value.

    Returns:
        updated_bicycle_graph (networkx/osmnx graph): updated graph with no key=1 edges without a corresponding key=0 edge
    """

    bicycle_edges = ox.graph_to_gdfs(bicycle_graph, nodes=False)

    # Unfreeze graph
    bicycle_graph_unfrozen = bicycle_graph.copy()

    try:
        # Find edges that have only key 1, but no key 0
        key1edges = bicycle_edges.loc[:, :, 1].index
        edges_to_key0 = []
        for edge in key1edges:
            if 0 not in bicycle_edges.loc[edge].index.get_level_values(0):
                edges_to_key0.append(edge)

        # For each of these edges,
        for e in edges_to_key0:
            # Create an edge copy with key 0 (but same attributes)
            # Then delete the key 1 edge
            myedgeattrs = bicycle_graph_unfrozen.edges[e[0], e[1], 1]
            bicycle_graph_unfrozen.add_edge(e[0], e[1], 0)
            bicycle_graph_unfrozen.edges[e[0], e[1], 0].update(myedgeattrs)
            bicycle_graph_unfrozen.remove_edge(e[0], e[1], 1)

        # Derive gdfs once more, from updated graph
        bicycle_edges = ox.graph_to_gdfs(bicycle_graph_unfrozen, nodes=False)

        # Run once more for assertion:

        # Find edges that have only key 1, but no key 0
        key1edges = bicycle_edges.loc[:, :, 1].index
        edges_to_key0 = []
        for edge in key1edges:
            if 0 not in bicycle_edges.loc[edge].index.get_level_values(0):
                edges_to_key0.append(edge)
        assert len(edges_to_key0) == 0

        # Set the bicycle graph to be the unfrozen & updated one:
        del bicycle_graph
        updated_bicycle_graph = bicycle_graph_unfrozen
        del bicycle_graph_unfrozen

        return updated_bicycle_graph

    except KeyError:
        print("No update necessary. Returning original bicycle graph.")

        return bicycle_graph


def clean_col_names(df):

    """
    Remove upper-case letters and : from data with OSM tags
    Special characters like ':' can for example break with pd.query function

    Arguments:
        df (df/gdf): dataframe/geodataframe with OSM tag data

    Returns:
        df (df/gdf): the same dataframe with updated column names
    """

    df.columns = df.columns.str.lower()

    df_cols = df.columns.to_list()

    new_cols = [c.replace(":", "_") for c in df_cols]

    df.columns = new_cols

    return df


def unzip_linestrings(org_gdf, edge_id_col):

    """
    Splits lines into their smallest possible line geometry, so each line only is defined by the start and end coordinate.
    Used to convert reference data to a similar data structure as used by osnmnx

    Arguments:
        org_gdf (gdf): gdf with original linestring/multilinestring data
        edge_id_col (str): name of column in org_gdf with unique edge id

    Returns:
        new_gdf (gdf): gdf with smallest possible linestring, with the same attributes as the original
    """

    gdf = org_gdf.copy()

    gdf["geometry"] = gdf["geometry"].apply(
        lambda x: linemerge(x) if x.geom_type == "MultiLineString" else x
    )

    # helper column: list of points
    gdf["points"] = gdf.apply(lambda x: [c for c in x.geometry.coords], axis=1)
    gdf["edges"] = gdf.apply(
        lambda x: [LineString(e) for e in zip(x.points, x.points[1:])], axis=1
    )
    edgelist = [item for sublist in gdf["edges"] for item in sublist]

    gdf[edge_id_col] = gdf.apply(lambda x: len(x.edges) * [x[edge_id_col]], axis=1)

    edgeid_list = [item for sublist in gdf[edge_id_col] for item in sublist]

    new_gdf = gpd.GeoDataFrame(
        {"geometry": edgelist, edge_id_col: edgeid_list}, crs=org_gdf.crs
    )

    new_gdf["new_edge_id"] = new_gdf.index  # Create random but unique edge id!
    assert len(new_gdf) == len(new_gdf.new_edge_id.unique())

    new_gdf = new_gdf.merge(
        org_gdf.drop("geometry", axis=1), how="left", on=edge_id_col
    )

    return new_gdf


def create_osmnx_graph(gdf):

    """
    Function for  converting a geodataframe with LineStrings to a NetworkX graph object (MultiDiGraph), which follows the data structure required by OSMnx.
    (I.e. Nodes indexed by osmid, nodes contain columns with x and y coordinates, edges is multiindexed by u, v, key).
    Converts MultiLineStrings to LineStrings - assumes that there are no gaps between the lines in the MultiLineString

    OBS! Current version does not fix issues with topology.

    Arguments:
        gdf (gdf): The data to be converted to a graph format
        directed (bool): Whether the resulting graph should be directed or not. Directionality is based on the order of the coordinates.

    Returns:
        G_ox (NetworkX MultiDiGraph object): The original data in a NetworkX graph format
    """

    gdf["geometry"] = gdf["geometry"].apply(
        lambda x: linemerge(x) if x.geom_type == "MultiLineString" else x
    )

    # If Multilines cannot be merged do to gaps, use explode
    geom_types = gdf.geom_type.to_list()
    # unique_geom_types = set(geom_types)

    if "MultiLineString" in geom_types:
        gdf = gdf.explode(index_parts=False)

    G = momepy.gdf_to_nx(gdf, approach="primal", directed=True)

    nodes, edges = momepy.nx_to_gdf(G)

    # Create columns and index as required by OSMnx
    # index_length = len(str(nodes['nodeID'].iloc[-1].item()))
    nodes["osmid"] = nodes[
        "nodeID"
    ]  # .apply(lambda x: create_node_index(x, index_length))

    # Create x y coordinate columns
    nodes["x"] = nodes.geometry.x
    nodes["y"] = nodes.geometry.y

    edges["u"] = nodes["osmid"].loc[edges.node_start].values
    edges["v"] = nodes["osmid"].loc[edges.node_end].values

    nodes.set_index("osmid", inplace=True)

    edges["length"] = edges.geometry.length  # Length is required by some functions

    edges["key"] = 0

    edges = find_parallel_edges(edges)

    # Create multiindex in u v key format
    edges = edges.set_index(["u", "v", "key"])

    # For ox simplification to work, edge geometries must be dropped. Edge geometries is defined by their start and end node
    # edges.drop(['geometry'], axis=1, inplace=True) # Not required by new simplification function

    G_ox = ox.graph_from_gdfs(nodes, edges)

    return G_ox


def find_parallel_edges(edges):

    """
    Check for parallel edges in a pandas DataFrame with edges, including columns u with start node index and v with end node index.
    If two edges have the same u-v pair, the column 'key' is updated to ensure that the u-v-key combination can uniquely identify an edge.
    Note that (u,v) is not considered parallel to (v,u)

    Arguments:
        edges (gdf): network edges

    Returns:
        edges (gdf): edges with updated key index
    """

    # Find edges with duplicate node pairs
    parallel = edges[edges.duplicated(subset=["u", "v"])]

    edges.loc[parallel.index, "key"] = 1  # Set keys to 1

    k = 1

    while len(edges[edges.duplicated(subset=["u", "v", "key"])]) > 0:

        k += 1

        parallel = edges[edges.duplicated(subset=["u", "v", "key"])]

        edges.loc[parallel.index, "key"] = k  # Set keys to 1

    assert (
        len(edges[edges.duplicated(subset=["u", "v", "key"])]) == 0
    ), "Edges not uniquely indexed by u,v,key!"

    return edges


def create_node_index(x, index_length):

    """
    Create unique id or index value of specific length based on another shorter column

    Arguments:
        x (undefined): the value to base the new id on (e.g. the index)
        index_length (int): the desired length of the id value

    Returns:
        x (str): the original id padded with zeroes to reach the required length of the index value
    """

    x = str(x)
    x = x.zfill(index_length)

    assert len(x) == index_length

    return x
