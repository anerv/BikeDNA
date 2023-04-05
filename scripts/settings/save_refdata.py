exec(open("../settings/yaml_variables.py").read())
exec(open("../settings/paths.py").read())

assert len(ref_nodes) == len(ref_nodes.nodeID.unique())
assert len(ref_edges) == len(ref_edges.edge_id.unique())
assert len(ref_edges_simplified) == len(ref_edges_simplified.edge_id.unique())
assert len(ref_nodes_simplified) == len(ref_nodes_simplified.nodeID.unique())

assert "infrastructure_length" in ref_edges_simplified.columns
assert "length" in ref_edges.columns

ref_nodes.to_file(ref_nodes_fp, index=True, overwrite="yes")

ref_edges.to_file(ref_edges_fp, index=True, overwrite="yes")

ref_nodes_simplified.to_file(ref_nodes_simplified_fp, index=True, overwrite="yes")

cols = [
    "edge_id",
    reference_id_col,
    "osmid",
    "length",
    "infrastructure_length",
    "protected",
    "multiedge",
    "from",
    "to",
    reference_geometries,
    bicycle_bidirectional,
    "geometry",
]
keep_cols = [c for c in cols if c in ref_edges_simplified.columns]

ref_edges_simplified[reference_id_col] = ref_edges_simplified[reference_id_col].astype(
    str
)

ref_edges_simplified["osmid"] = ref_edges_simplified["osmid"].astype(str)

if bicycle_bidirectional in ref_edges_simplified.columns:
    ref_edges_simplified[bicycle_bidirectional] = ref_edges_simplified[
        bicycle_bidirectional
    ].astype(str)

if reference_geometries in ref_edges_simplified.columns:
    ref_edges_simplified[reference_geometries] = ref_edges_simplified[
        reference_geometries
    ].astype(str)

ref_edges_simplified = ref_edges_simplified[keep_cols]

ref_edges_simplified.to_file(ref_edges_simplified_fp, index=True, overwrite="yes")

ref_nodes_joined.to_file(ref_nodes_joined_fp, index=True, overwrite="yes")

ref_edges_joined.to_file(ref_edges_joined_fp, index=True, overwrite="yes")

ref_nodes_simp_joined.to_file(
    ref_nodes_simplified_joined_fp, index=True, overwrite="yes"
)

cols = [
    "edge_id",
    reference_id_col,
    "osmid",
    "length",
    "infrastructure_length",
    "protected",
    "multiedge",
    "from",
    "to",
    reference_geometries,
    bicycle_bidirectional,
    "grid_id",
    "u",
    "v",
    "key",
    "geometry",
]
keep_cols = [c for c in cols if c in ref_edges_simp_joined.columns]

ref_edges_simp_joined[reference_id_col] = ref_edges_simp_joined[
    reference_id_col
].astype(str)
ref_edges_simp_joined["osmid"] = ref_edges_simp_joined["osmid"].astype(str)

if bicycle_bidirectional in ref_edges_simp_joined.columns:
    ref_edges_simp_joined[bicycle_bidirectional] = ref_edges_simp_joined[
        bicycle_bidirectional
    ].astype(str)

if reference_geometries in ref_edges_simp_joined.columns:
    ref_edges_simp_joined[reference_geometries] = ref_edges_simp_joined[
        reference_geometries
    ].astype(str)

ref_edges_simp_joined = ref_edges_simp_joined[keep_cols]

ref_edges_simp_joined.to_file(
    ref_edges_simplified_joined_fp, index=True, overwrite="yes"
)


print(f"{reference_name} nodes and edges saved successfully!")

ox.save_graphml(graph_ref, ref_graph_fp)
ox.save_graphml(graph_ref_simplified, ref_graph_simplified_fp)

print(f"{reference_name} networks saved successfully!")

# Export grid
grid.to_file(ref_grid_fp)
print("Reference grid saved successfully!")
