import geopandas as gpd
import pickle
import json

exec(open("../settings/yaml_variables.py").read())
exec(open("../settings/paths.py").read())

# Load and merge grids with local intrinsic results

# ref_grid = gpd.read_file(ref_grid_fp)

# osm_grid = gpd.read_file(osm_grid_fp)

# Save grid with results
with open(
    f"../../results/OSM/{study_area}/data/grid_results_intrinsic.pickle", "rb"
) as fp:
    osm_intrinsic_grid = pickle.load(fp)

with open(
    f"../../results/REFERENCE/{study_area}/data/grid_results_intrinsic.pickle", "rb"
) as fp:
    ref_intrinsic_grid = pickle.load(fp)

ref_intrinsic_grid.drop("geometry", axis=1, inplace=True)

grid = osm_intrinsic_grid.merge(
    ref_intrinsic_grid, on="grid_id", suffixes=("_osm", "_ref")
)

assert len(grid) == len(osm_intrinsic_grid) == len(ref_intrinsic_grid)

grid_ids = grid.grid_id.to_list()

# Load JSON files with results of intrinsic results

osm_intrinsic_file = open(
    f"../../results/OSM/{study_area}/data/intrinsic_analysis.json"
)

osm_intrinsic_results = json.load(osm_intrinsic_file)

ref_intrinsic_file = open(
    f"../../results/REFERENCE/{study_area}/data/intrinsic_analysis.json"
)

ref_intrinsic_results = json.load(ref_intrinsic_file)


print("Results from intrinsic analyses loaded successfully!")
