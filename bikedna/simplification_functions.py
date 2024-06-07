# -*- coding: utf-8 -*-
"""
New functions and original and modified functions from OSMnx in order to
simplify the graph by removing interstitial nodes and by going from a
multidigraph to a graph.
"""

import itertools
import numpy as np
import geopandas as gpd
import shapely
from shapely.geometry import LineString
from shapely.geometry import Point
import networkx as nx
from haversine import haversine
from haversine import Unit


# New function
def multidigraph_to_graph(G, attributes=None,
                          verbose=False, debug=False):
    """
    Transform a MultiDiGraph into an undirected Graph by first removing
    the direction, making a MultiGraph, and then making it a Graph by
    making sure that there is no multiple edges : for self-loop we create
    two nodes within the geometry of the edge, for node with multiple
    edges we create one node within each geometry of the edge. We avoid
    to merge directed edges without the same arbitrary attributes.

    Parameters
    ----------
    G : networkx.classes.multidigraph.MultiDiGraph
        MultiDiGraph we want to transform.
    attributes : list or str or number, optional
        Key to the attributes we want to discriminate.
        The default is None.
    verbose : bool, optional
        If True, give the number of self-loop and multiple path found,
        and the nodes with multiple straight edge between
        (sign of bad OSM practice). The default is False.
    debug : bool, optional
        If True, return a dictionary with every osmid and geometry of
        self-loop and multiple path. The default is False.

    Returns
    -------
    G : networkx.classes.graph.Graph
        Undirected graph made from the initial MultiDiGraph.
    debug_dict : dict
        Dictionary of every osmid and geometry of self-loop and multiple
        path to return only if debug is True.

    """
    if verbose is True:
        self_loop_count = 0
        multiple_path_count = dict()
    if debug is True:
        debug_dict = dict()
        debug_dict['self-loop'] = []
        debug_dict['multiple-path'] = []
    G = get_undirected(G, attributes=attributes) # make it undirected
    initial_node_list = list(G.nodes()) # to avoid issue with changing
    for node in initial_node_list: # number of node during the process
        neighbors = np.transpose(list(G.edges(node)))[1]
        if node in neighbors: # then self_loop, need 2 artifical nodes
            for k in list(G.get_edge_data(node, node).keys()):
                if verbose is True:
                    self_loop_count +=1
                if debug is True:
                    debug_dict['self-loop'].append([
                        G.edges[node, node, k]['osmid'],
                        G.edges[node, node, k]['geometry']])
                G = _solve_self_loop(G, node, k)
        for neigh in neighbors:
            if G.number_of_edges(node, neigh) > 1: #then multiple path, need 1
                if verbose is True:
                    if G.number_of_edges(node, neigh) in multiple_path_count:
                        multiple_path_count[
                            G.number_of_edges(node, neigh)] += 1
                    else:
                        multiple_path_count[
                            G.number_of_edges(node, neigh)] = 1
                if debug is True:
                    for k in list(G.get_edge_data(node, neigh).keys()):
                        debug_dict['multiple-path'].append([
                            G.edges[node, neigh, k]['osmid'],
                            G.edges[node, neigh, k]['geometry']])
                G = _solve_multiple_path(G, node, neigh, verbose=verbose)
    G = nx.Graph(G) #if no multiple edges, simply change the type
    if verbose is True:
        print("""
              Number of self-loop found : {} \n
              Number of multiple path between nodes found : {}
              """.format(self_loop_count, multiple_path_count)
              )
    if debug is True:
        return G, debug_dict
    return G


# New function
def _solve_self_loop(G, node, key):
    """
    Transform a loop where a node is connected to itself by adding two 
    nodes in the geometry of the loop, in order to make it simple
    (no multiple edges)

    Parameters
    ----------
    G : networkx.MultiGraph
        MultiGraph we want to transform.
    node : int
        Node's ID where there is a self-loop.
    key : int
        Key of the edge, needed because the graph is a MultiGraph

    Returns
    -------
    G : networkx.MultiGraph
        MultiGraph with the self-loop resolved.

    """
    edge_attributes = dict(G.edges[node, node, key]) # take attributes
    geom = list(edge_attributes['geometry'].coords[:])
    edge_attributes.pop('geometry') # remove geometry
    edge_attributes.pop('length') # remove length
    G.remove_edge(node, node, key)
    f_num = node + 1 # find unique ID not already in the graph
    while f_num in G.nodes():
        f_num += 1
    s_num = f_num + 1
    while s_num in G.nodes():
        s_num += 1
    # TODO : add street_count to nodes ?
    # Add nodes as the first and last point in the LineString geometry
    # if we don't count the original node of the self-loop
    G.add_node(f_num, x=geom[1][0], y=geom[1][1])
    G.add_node(s_num, x=geom[-2][0], y=geom[-2][1])
    # Connect them with edges keeping the attributes and having in total
    # the same geometry as before
    G.add_edge(node, f_num, key=0, **edge_attributes,
               geometry=LineString(geom[:2]),
               length=_get_length(G, node, f_num))
    G.add_edge(node, s_num, key=0, **edge_attributes,
               geometry=LineString(geom[-2:]),
               length=_get_length(G, node, s_num))
    G.add_edge(f_num, s_num, key=0, **edge_attributes,
               geometry=LineString(geom[1:-1]),
               length=_get_length(G, s_num, f_num))
    return G

# New function
def _solve_multiple_path(G, node, other_node, verbose=False):
    """
    Transform multiple paths between nodes by adding artifical nodes on every
    path but one, in order to make it simple (no multiple edges)

    Parameters
    ----------
    G : networkx.classes.multidigraph.MultiDiGraph
        MultiDiGraph we want to transform.
    node : int
        First node's ID.
    other_node : int
        Second node's ID.
    verbose : bool, optional
        If True, give  the nodes with multiple straight edge between
        (sign of bad OSM practice) and their keys. The default is False.

    Returns
    -------
    G : networkx.classes.multidigraph.MultiDiGraph
        MultiDiGraph with the multiple path issue solved.

    """
    # for every path but one, to add as little number of node as needed
    count = 0
    straigth_key = []
    initial_n_edges = G.number_of_edges(node, other_node)
    initial_key_list = list(G.get_edge_data(node, other_node).keys())
    for i in initial_key_list:
        if count == initial_n_edges - 1:
            break
        elif len(list(
                G.edges[node, other_node, i]['geometry'].coords[:]
                )) > 2:
            # take attributes
            edge_attributes = dict(G.edges[node, other_node, i])
            geom = list(edge_attributes['geometry'].coords[:])
            edge_attributes.pop('geometry') #remove geometry
            edge_attributes.pop('length') # remove length
            G.remove_edge(node, other_node, i)
            p_num = node + 1 # find ID that is not already in the graph
            while p_num in G.nodes():
                p_num += 1
            # add node as the first point of the geometry
            G.add_node(p_num, x=geom[1][0], y=geom[1][1])
            # Connect it with edges keeping the attributes and having in total
            # the same geometry as before
            G.add_edge(node, p_num, key=0, **edge_attributes,
                       geometry = LineString(geom[:2]),
                       length=_get_length(G, node, p_num))
            G.add_edge(p_num, other_node, key=0, **edge_attributes,
                       geometry=LineString(geom[1:]),
                       length=_get_length(G, other_node, p_num))
            count += 1
        else: #if straight line
            straigth_key.append(i)
    if count < G.number_of_edges(node, other_node) - 1:
        if verbose is True:
            print("""
                  Multiple straight path between node {} and {} 
                  at the keys {}
                  """.format(node, other_node, straigth_key))
        while count < G.number_of_edges(node, other_node) - 1:
            f_key = straigth_key[0]
            edge_attributes = dict(G.edges[node, other_node, f_key])
            geom = list(edge_attributes['geometry'].coords[:])
            edge_attributes.pop('geometry') #remove geometry
            edge_attributes.pop('length') # remove length
            mid_x = (geom[0][0]+geom[1][0]) / 2. # take middle coordinates
            mid_y = (geom[0][1]+geom[1][1]) / 2.
            geom.insert(1, (mid_x, mid_y)) # insert it into the geometry
            G.remove_edge(node, other_node, f_key)
            p_num = node + 1# find ID that is not already in the graph
            while p_num in G.nodes():
                p_num += 1
            # add node as the first point of the geometry
            G.add_node(p_num, x=geom[1][0], y=geom[1][1])
            # Connect it with edges keeping the attributes and having in total
            # the same geometry as before
            G.add_edge(node, p_num, key=0, **edge_attributes,
                       geometry=LineString(geom[:2]),
                       length=_get_length(G, node, p_num))
            G.add_edge(p_num, other_node, key=0, **edge_attributes,
                       geometry=LineString(geom[1:]),
                       length=_get_length(G, other_node, p_num))
            straigth_key.remove(f_key)
            count += 1
    return G

# New function
def _get_length(G, f_node, s_node):
    """Return the haversine length in meters between two nodes like OSM."""
    f_point = [G.nodes[f_node]['y'], G.nodes[f_node]['x']] #[lat, lon]
    s_point = [G.nodes[s_node]['y'], G.nodes[s_node]['x']]
    return round(haversine(f_point, s_point, unit=Unit.METERS), 3)

# Modified function
def get_undirected(G, attributes=None):
    """
    Convert MultiDiGraph to undirected MultiGraph.

    Maintains parallel edges only if their geometries or other selected
    attributes differ. Note: see also `get_digraph` to convert 
    MultiDiGraph to DiGraph.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph

    Returns
    -------
    networkx.MultiGraph
    """
    # make a copy to not mutate original graph object caller passed in
    G = G.copy()

    for u, v, d in G.edges(data=True):
        # add geometry if missing, to compare parallel edges' geometries
        if "geometry" not in d:
            point_u = (G.nodes[u]["x"], G.nodes[u]["y"])
            point_v = (G.nodes[v]["x"], G.nodes[v]["y"])
            d["geometry"] = LineString([point_u, point_v])

    # increment parallel edges' keys so we don't retain only one edge of sets
    # of true parallel edges when we convert from MultiDiGraph to MultiGraph
    G = _update_edge_keys(G)

    # convert MultiDiGraph to MultiGraph, retaining edges in both directions
    # of parallel edges and self-loops for now
    H = nx.MultiGraph(**G.graph)
    H.add_nodes_from(G.nodes(data=True))
    H.add_edges_from(G.edges(keys=True, data=True))

    # the previous operation added all directed edges from G as undirected
    # edges in H. we now have duplicate edges for every bidirectional parallel
    # edge or self-loop. so, look through the edges and remove any duplicates.
    duplicate_edges = set()
    for u1, v1, key1, data1 in H.edges(keys=True, data=True):

        # if we haven't already flagged this edge as a duplicate
        if (u1, v1, key1) not in duplicate_edges:

            # look at every other edge between u and v, one at a time
            for key2 in H[u1][v1]:

                # don't compare this edge to itself
                if key1 != key2:

                    # compare the first edge's data to the second's
                    # if they match up, flag the duplicate for removal
                    data2 = H.edges[u1, v1, key2]
                    if _is_duplicate_edge(data1, data2, attributes=attributes):
                        duplicate_edges.add((u1, v1, key2))

    H.remove_edges_from(duplicate_edges)
    return H

# Same function
def _update_edge_keys(G):
    """
    Increment key of one edge of parallel edges that differ in geometry.

    For example, two streets from u to v that bow away from each other as
    separate streets, rather than opposite direction edges of a single street.
    Increment one of these edge's keys so that they do not match across u, v,
    k or v, u, k so we can add both to an undirected MultiGraph.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph

    Returns
    -------
    G : networkx.MultiDiGraph
    """
    # identify all the edges that are duplicates based on a sorted combination
    # of their origin, destination, and key. that is, edge uv will match edge vu
    # as a duplicate, but only if they have the same key
    edges = graph_to_gdfs(G, nodes=False, fill_edge_geometry=False)
    edges["uvk"] = ["_".join(sorted([str(u), str(v)]) + [str(k)]) for u, v, k in edges.index]
    mask = edges["uvk"].duplicated(keep=False)
    dupes = edges[mask].dropna(subset=["geometry"])

    different_streets = []
    groups = dupes[["geometry", "uvk"]].groupby("uvk")

    # for each group of duplicate edges
    for _, group in groups:

        # for each pair of edges within this group
        for geom1, geom2 in itertools.combinations(group["geometry"], 2):

            # if they don't have the same geometry, flag them as different
            # streets: flag edge uvk, but not edge vuk, otherwise we would
            # increment both their keys and they'll still duplicate each other
            if not _is_same_geometry(geom1, geom2):
                different_streets.append(group.index[0])

    # for each unique different street, increment its key to make it unique
    for u, v, k in set(different_streets):
        new_key = max(list(G[u][v]) + list(G[v][u])) + 1
        G.add_edge(u, v, key=new_key, **G.get_edge_data(u, v, k))
        G.remove_edge(u, v, key=k)

    return G

# Same function
def graph_to_gdfs(G, nodes=True, edges=True, node_geometry=True,
                  fill_edge_geometry=True):
    """
    Convert a MultiDiGraph to node and/or edge GeoDataFrames.

    This function is the inverse of `graph_from_gdfs`.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    nodes : bool
        if True, convert graph nodes to a GeoDataFrame and return it
    edges : bool
        if True, convert graph edges to a GeoDataFrame and return it
    node_geometry : bool
        if True, create a geometry column from node x and y attributes
    fill_edge_geometry : bool
        if True, fill in missing edge geometry fields using nodes u and v

    Returns
    -------
    geopandas.GeoDataFrame or tuple
        gdf_nodes or gdf_edges or tuple of (gdf_nodes, gdf_edges). gdf_nodes
        is indexed by osmid and gdf_edges is multi-indexed by u, v, key
        following normal MultiDiGraph structure.
    """
    crs = G.graph["crs"]

    if nodes:

        if not G.nodes:  # pragma: no cover
            raise ValueError("graph contains no nodes")

        nodes, data = zip(*G.nodes(data=True))

        if node_geometry:
            # convert node x/y attributes to Points for geometry column
            geom = (Point(d["x"], d["y"]) for d in data)
            gdf_nodes = gpd.GeoDataFrame(data, index=nodes, crs=crs,
                                         geometry=list(geom))
        else:
            gdf_nodes = gpd.GeoDataFrame(data, index=nodes)

        gdf_nodes.index.rename("osmid", inplace=True)

    if edges:

        if not G.edges:  # pragma: no cover
            raise ValueError("graph contains no edges")

        u, v, k, data = zip(*G.edges(keys=True, data=True))

        if fill_edge_geometry:

            # subroutine to get geometry for every edge: if edge already has
            # geometry return it, otherwise create it using the incident nodes
            x_lookup = nx.get_node_attributes(G, "x")
            y_lookup = nx.get_node_attributes(G, "y")

            def make_geom(u, v, data, x=x_lookup, y=y_lookup):
                if "geometry" in data:
                    return data["geometry"]
                else:
                    return LineString((Point((x[u], y[u])),
                                       Point((x[v], y[v]))))

            geom = map(make_geom, u, v, data)
            gdf_edges = gpd.GeoDataFrame(data, crs=crs, geometry=list(geom))

        else:
            gdf_edges = gpd.GeoDataFrame(data)
            if "geometry" not in gdf_edges.columns:
                # if no edges have a geometry attribute, create null column
                gdf_edges["geometry"] = np.nan
            gdf_edges.set_geometry("geometry")
            gdf_edges.crs = crs

        # add u, v, key attributes as index
        gdf_edges["u"] = u
        gdf_edges["v"] = v
        gdf_edges["key"] = k
        gdf_edges.set_index(["u", "v", "key"], inplace=True)

    if nodes and edges:
        return gdf_nodes, gdf_edges
    elif nodes:
        return gdf_nodes
    elif edges:
        return gdf_edges
    else:  # pragma: no cover
        raise ValueError("you must request nodes or edges or both")

# Modified function
def _is_duplicate_edge(data1, data2, attributes=None):
    """
    Check if two graph edge data dicts have the same osmid and geometry.

    Parameters
    ----------
    data1: dict
        the first edge's data
    data2 : dict
        the second edge's data

    Returns
    -------
    is_dupe : bool
    """
    is_dupe = False

    # if either edge's osmid contains multiple values (due to simplification)
    # compare them as sets to see if they contain the same values
    osmid1 = set(data1["osmid"]) if isinstance(data1["osmid"], list) else data1["osmid"]
    osmid2 = set(data2["osmid"]) if isinstance(data2["osmid"], list) else data2["osmid"]

    # if they contain the same osmid or set of osmids (due to simplification)
    if osmid1 == osmid2:

        # if both edges have geometry attributes and they match each other
        if ("geometry" in data1) and ("geometry" in data2):
            if _is_same_geometry(data1["geometry"], data2["geometry"]):
                is_dupe = True

        # if neither edge has a geometry attribute
        elif ("geometry" not in data1) and ("geometry" not in data2):
            is_dupe = True

        # if one edge has geometry attribute but the other doesn't:
        # not dupes
        else:
            pass

        if attributes is None:
            pass
        elif isinstance(attributes, list):
            for attr in attributes:
                if data1[attr] != data2[attr]:
                    is_dupe = False
                else:
                    pass
        else:
            if data1[attributes] != data2[attributes]:
                is_dupe = False

    return is_dupe

# Same function
def _is_same_geometry(ls1, ls2):
    """
    Determine if two LineString geometries are the same (in either direction).

    Check both the normal and reversed orders of their constituent points.

    Parameters
    ----------
    ls1 : shapely.geometry.LineString
        the first LineString geometry
    ls2 : shapely.geometry.LineString
        the second LineString geometry

    Returns
    -------
    bool
    """
    # extract coordinates from each LineString geometry
    geom1 = [tuple(coords) for coords in ls1.xy]
    geom2 = [tuple(coords) for coords in ls2.xy]

    # reverse the first LineString's coordinates' direction
    geom1_r = [tuple(reversed(coords)) for coords in ls1.xy]

    # if first geometry matches second in either direction, return True
    return geom1 == geom2 or geom1_r == geom2


# Modified function
def simplify_graph(G, attributes=None, strict=True, remove_rings=True):
    """
    Simplify a graph's topology by removing interstitial nodes.

    Simplifies graph topology by removing all nodes that are not intersections
    or dead-ends. Create an edge directly between the end points that
    encapsulate them, but retain the geometry of the original edges, saved as
    a new `geometry` attribute on the new edge. Note that only simplified
    edges receive a `geometry` attribute. Some of the resulting consolidated
    edges may comprise multiple OSM ways, and if so, their multiple attribute
    values are stored as a list.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    strict : bool
        if False, allow nodes to be end points even if they fail all other
        rules but have incident edges with different OSM IDs. Lets you keep
        nodes at elbow two-way intersections, but sometimes individual blocks
        have multiple OSM IDs within them too.
    remove_rings : bool
        if True, remove isolated self-contained rings that have no endpoints

    Returns
    -------
    G : networkx.MultiDiGraph
        topologically simplified graph, with a new `geometry` attribute on
        each simplified edge
    """
    if "simplified" in G.graph and G.graph["simplified"]:
        raise Exception("This graph has already been simplified, cannot simplify it again.")

    # define edge segment attributes to sum upon edge simplification
    attrs_to_sum = {"length", "travel_time"}

    # make a copy to not mutate original graph object caller passed in
    G = G.copy()
    all_nodes_to_remove = []
    all_edges_to_add = []

    # generate each path that needs to be simplified
    for path in _get_paths_to_simplify(G, attributes=attributes,
                                       strict=strict):

        # add the interstitial edges we're removing to a list so we can retain
        # their spatial geometry
        path_attributes = dict()
        for u, v in zip(path[:-1], path[1:]):

            # there should rarely be multiple edges between interstitial nodes
            # usually happens if OSM has duplicate ways digitized for just one
            # street... we will keep only one of the edges (see below)

            # get edge between these nodes: if multiple edges exist between
            # them (see above), we retain only one in the simplified graph
            edge_data = G.edges[u, v, 0]
            for attr in edge_data:
                if attr in path_attributes:
                    # if this key already exists in the dict, append it to the
                    # value list
                    path_attributes[attr].append(edge_data[attr])
                else:
                    # if this key doesn't already exist, set the value to a list
                    # containing the one value
                    path_attributes[attr] = [edge_data[attr]]

        # consolidate the path's edge segments' attribute values
        for attr in path_attributes:
            if attr in attrs_to_sum:
                # if this attribute must be summed, sum it now
                path_attributes[attr] = sum(path_attributes[attr])
            elif len(set(path_attributes[attr])) == 1:
                # if there's only 1 unique value in this attribute list,
                # consolidate it to the single value (the zero-th):
                path_attributes[attr] = path_attributes[attr][0]
            else:
                # otherwise, if there are multiple values, keep one of each
                path_attributes[attr] = list(set(path_attributes[attr]))

        # construct the new consolidated edge's geometry for this path
        path_attributes["geometry"] = LineString(
            [Point((G.nodes[node]["x"], G.nodes[node]["y"])) for node in path]
        )

        # add the nodes and edges to their lists for processing at the end
        all_nodes_to_remove.extend(path[1:-1])
        all_edges_to_add.append(
            {"origin": path[0], "destination": path[-1],
             "attr_dict": path_attributes}
        )

    # for each edge to add in the list we assembled, create a new edge between
    # the origin and destination
    for edge in all_edges_to_add:
        G.add_edge(edge["origin"], edge["destination"], **edge["attr_dict"])

    # finally remove all the interstitial nodes between the new edges
    G.remove_nodes_from(set(all_nodes_to_remove))

    if remove_rings:
        # remove any connected components that form a self-contained ring
        # without any endpoints
        wccs = nx.weakly_connected_components(G)
        nodes_in_rings = set()
        for wcc in wccs:
            if not any(_is_endpoint(G, n) for n in wcc):
                nodes_in_rings.update(wcc)
        G.remove_nodes_from(nodes_in_rings)

    # mark graph as having been simplified
    G.graph["simplified"] = True
    return G


# Modified function
def _get_paths_to_simplify(G, attributes=None, strict=True):
    """
    Generate all the paths to be simplified between endpoint nodes.

    The path is ordered from the first endpoint, through the interstitial
    nodes, to the second endpoint.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    attributes : list
        key of the attributes we should discriminate
    strict : bool
        if False, allow nodes to be end points even if they fail all other
        rules but have edges with different OSM IDs

    Yields
    ------
    path_to_simplify : list
    """
    # first identify all the nodes that are endpoints
    endpoints = set([n for n in G.nodes if _is_endpoint(G, n,
                                                        attributes=attributes,
                                                        strict=strict)])

    # for each endpoint node, look at each of its successor nodes
    for endpoint in endpoints:
        for successor in G.successors(endpoint):
            if successor not in endpoints:
                # if endpoint node's successor is not an endpoint, build path
                # from the endpoint node, through the successor, and on to the
                # next endpoint node
                yield _build_path(G, endpoint, successor, endpoints)

# Modified function
def _is_endpoint(G, node, attributes=None, strict=True):
    """
    Is node a true endpoint of an edge.

    Return True if the node is a "real" endpoint of an edge in the network,
    otherwise False. OSM data includes lots of nodes that exist only as points
    to help streets bend around curves. An end point is a node that either:
    1) is its own neighbor, ie, it self-loops.
    2) or, has no incoming edges or no outgoing edges, ie, all its incident
    edges point inward or all its incident edges point outward.
    3) or, it does not have exactly two neighbors and degree of 2 or 4.
    4) or, if strict mode is false, if its edges have different OSM IDs.
    5) or, if attributes is not None, we take either every value of the
    given list or the given value and test whether every edges connected to
    the node have the same attribute.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    node : int
        the node to examine
    attributes : list
        key of the attributes we should discriminate
    strict : bool
        if False, allow nodes to be end points even if they fail all other
        rules but have edges with different OSM IDs

    Returns
    -------
    bool
    """
    neighbors = set(list(G.predecessors(node)) + list(G.successors(node)))
    n = len(neighbors)
    d = G.degree(node)

    # rule 1
    if node in neighbors:
        # if the node appears in its list of neighbors, it self-loops
        # this is always an endpoint.
        return True

    # rule 2
    elif G.out_degree(node) == 0 or G.in_degree(node) == 0:
        # if node has no incoming edges or no outgoing edges, it is an endpoint
        return True

    # rule 3
    elif not (n == 2 and (d == 2 or d == 4)):
        # else, if it does NOT have 2 neighbors AND either 2 or 4 directed
        # edges, it is an endpoint. either it has 1 or 3+ neighbors, in which
        # case it is a dead-end or an intersection of multiple streets or it
        # has 2 neighbors but 3 degree (indicating a change from oneway to
        # twoway) or more than 4 degree (indicating a parallel edge) and thus
        # is an endpoint
        return True

    # rule 4
    elif not strict:
        # non-strict mode: do its incident edges have different OSM IDs?
        osmids = []

        # add all the edge OSM IDs for incoming edges
        for u in G.predecessors(node):
            for key in G[u][node]:
                osmids.append(G.edges[u, node, key]["osmid"])

        # add all the edge OSM IDs for outgoing edges
        for v in G.successors(node):
            for key in G[node][v]:
                osmids.append(G.edges[node, v, key]["osmid"])

        # if there is more than 1 OSM ID in the list of edge OSM IDs then it is
        # an endpoint, if not, it isn't
        return len(set(osmids)) > 1

    # if none of the preceding rules returned true, then it is not an endpoint
    # except if the attributes is not None and the values are different
    else:
        if attributes is None:
            return False
        else:
            if isinstance(attributes, list):
                for attr in attributes:
                    for pre in list(G.predecessors(node)):
                        for suc in list(G.successors(node)):
                            if (G.edges[pre, node, 0][attr]) == (
                                    G.edges[node, suc, 0][attr]):
                                pass
                            else:
                                return True
            else:
                for pre in list(G.predecessors(node)):
                    for suc in list(G.successors(node)):
                        if (G.edges[pre, node, 0][attributes]) == (
                                G.edges[node, suc, 0][attributes]):
                            pass
                        else:
                            return True
            return False

# Same function
def _build_path(G, endpoint, endpoint_successor, endpoints):
    """
    Build a path of nodes from one endpoint node to next endpoint node.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    endpoint : int
        the endpoint node from which to start the path
    endpoint_successor : int
        the successor of endpoint through which the path to the next endpoint
        will be built
    endpoints : set
        the set of all nodes in the graph that are endpoints

    Returns
    -------
    path : list
        the first and last items in the resulting path list are endpoint
        nodes, and all other items are interstitial nodes that can be removed
        subsequently
    """
    # start building path from endpoint node through its successor
    path = [endpoint, endpoint_successor]

    # for each successor of the endpoint's successor
    for successor in G.successors(endpoint_successor):
        if successor not in path:
            # if this successor is already in the path, ignore it, otherwise
            # add it to the path
            path.append(successor)
            while successor not in endpoints:
                # find successors (of current successor) not in path
                successors = [n for n in G.successors(successor) if n not in path]

                # 99%+ of the time there will be only 1 successor: add to path
                if len(successors) == 1:
                    successor = successors[0]
                    path.append(successor)

                # handle relatively rare cases or OSM digitization quirks
                elif len(successors) == 0:
                    if endpoint in G.successors(successor):
                        # we have come to the end of a self-looping edge, so
                        # add first node to end of path to close it and return
                        return path + [endpoint]
                    else:  # pragma: no cover
                        # this can happen due to OSM digitization error where
                        # a one-way street turns into a two-way here, but
                        # duplicate incoming one-way edges are present
                        return path
                else:  # pragma: no cover
                    # if successor has >1 successors, then successor must have
                    # been an endpoint because you can go in 2 new directions.
                    # this should never occur in practice
                    raise Exception(f"Unexpected simplify pattern failed near {successor}")

            # if this successor is an endpoint, we've completed the path
            return path

    # if endpoint_successor has no successors not already in the path, return
    # the current path: this is usually due to a digitization quirk on OSM
    return path


# Modified function
def momepy_simplify_graph(G, attributes=None,
                          strict=True, remove_rings=True):
    """
    Same as simplify_graph, but geometry is not taken into account in the same
    way : here it can take into account places where a geometry attribute
    already exist for edges.
    Simplify a graph's topology by removing interstitial nodes.

    Simplifies graph topology by removing all nodes that are not intersections
    or dead-ends. Create an edge directly between the end points that
    encapsulate them, but retain the geometry of the original edges, saved as
    a new `geometry` attribute on the new edge. Note that only simplified
    edges receive a `geometry` attribute. Some of the resulting consolidated
    edges may comprise multiple OSM ways, and if so, their multiple attribute
    values are stored as a list.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    strict : bool
        if False, allow nodes to be end points even if they fail all other
        rules but have incident edges with different OSM IDs. Lets you keep
        nodes at elbow two-way intersections, but sometimes individual blocks
        have multiple OSM IDs within them too.
    remove_rings : bool
        if True, remove isolated self-contained rings that have no endpoints

    Returns
    -------
    G : networkx.MultiDiGraph
        topologically simplified graph, with a new `geometry` attribute on
        each simplified edge
    """
    if "simplified" in G.graph and G.graph["simplified"]:
        raise Exception("This graph has already been simplified, cannot simplify it again.")

    # define edge segment attributes to sum upon edge simplification
    attrs_to_sum = {"length", "travel_time"}

    # make a copy to not mutate original graph object caller passed in
    G = G.copy()
    all_nodes_to_remove = []
    all_edges_to_add = []
    

    # generate each path that needs to be simplified
    for path in _get_paths_to_simplify(G, attributes=attributes,
                                       strict=strict):
        # add the interstitial edges we're removing to a list so we can retain
        # their spatial geometry
        path_attributes = dict()
        geometry_batch = []
        for u, v in zip(path[:-1], path[1:]):

            # there should rarely be multiple edges between interstitial nodes
            # usually happens if OSM has duplicate ways digitized for just one
            # street... we will keep only one of the edges (see below)

            # get edge between these nodes: if multiple edges exist between
            # them (see above), we retain only one in the simplified graph
            edge_data = G.edges[u, v, 0]
            geometry_batch.append(edge_data['geometry'])
            for attr in edge_data:
                if attr == 'geometry':
                    pass
                if attr in path_attributes:
                    # if this key already exists in the dict, append it to the
                    # value list
                    path_attributes[attr].append(edge_data[attr])
                else:
                    # if this key doesn't already exist, set the value to a list
                    # containing the one value
                    path_attributes[attr] = [edge_data[attr]]

        # consolidate the path's edge segments' attribute values
        for attr in path_attributes:
            # we want to make a flat list to be able to hash it
            temp = path_attributes[attr]
            for i in range(len(temp)):
                if isinstance(temp[i],list):
                    pass
                else:
                    temp[i] = [temp[i]]
            temp = [item for sublist in temp for item in sublist]
            path_attributes[attr] = temp
            if attr in attrs_to_sum:
                # if this attribute must be summed, sum it now
                path_attributes[attr] = sum(path_attributes[attr])
            elif attr == 'geometry':
                pass
            elif len(set(path_attributes[attr])) == 1:
                # if there's only 1 unique value in this attribute list,
                # consolidate it to the single value (the zero-th):
                path_attributes[attr] = path_attributes[attr][0]
            else:
                # otherwise, if there are multiple values, keep one of each
                path_attributes[attr] = list(set(path_attributes[attr]))
                
        # construct the geometry and sum the lengths of the segments
        multi_line = shapely.geometry.MultiLineString(geometry_batch)
        path_attributes["geometry"] = shapely.ops.linemerge(multi_line)

        # add the nodes and edges to their lists for processing at the end
        all_nodes_to_remove.extend(path[1:-1])
        all_edges_to_add.append(
            {"origin": path[0], "destination": path[-1],
             "attr_dict": path_attributes}
        )
 
    # for each edge to add in the list we assembled, create a new edge between
    # the origin and destination
    for edge in all_edges_to_add:
        G.add_edge(edge["origin"], edge["destination"], **edge["attr_dict"])

    # finally remove all the interstitial nodes between the new edges
    G.remove_nodes_from(set(all_nodes_to_remove))

    if remove_rings:
        # remove any connected components that form a self-contained ring
        # without any endpoints
        wccs = nx.weakly_connected_components(G)
        nodes_in_rings = set()
        for wcc in wccs:
            if not any(_is_endpoint(G, n) for n in wcc):
                nodes_in_rings.update(wcc)
        G.remove_nodes_from(nodes_in_rings)

    # mark graph as having been simplified
    G.graph["simplified"] = True
    return G
