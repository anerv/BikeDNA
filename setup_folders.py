# Run this file while in the main folder

import os
import yaml

with open(r"config.yml") as file:

    parsed_yaml_file = yaml.load(file, Loader=yaml.FullLoader)

    study_area = parsed_yaml_file["study_area"]

# Create folder structure for data
compare_data_path = "data/COMPARE/"
osm_data_path = "data/OSM/"
ref_data_path = "data/REFERENCE/"
polygon_data_path = "data/study_area_polygon"

compare_results_path = "results/COMPARE/"
osm_results_path = "results/OSM/"
ref_results_path = "results/REFERENCE/"
exports_path = "exports/"

paths = [
    osm_data_path,
    ref_data_path,
    compare_data_path,
    osm_results_path,
    ref_results_path,
    compare_results_path,
]

# make "data" and "results" superior folders with osm/ref/comp subfolders
subfolders = ["data", "results"]
for folder in subfolders:
    if not os.path.exists(folder):
        os.mkdir(folder)
for folder in paths:
    if not os.path.exists(folder):
        os.mkdir(folder)

# Create folders for study area polygon
if not os.path.exists(polygon_data_path):
    os.mkdir(polygon_data_path)

sa_poly_folder = "data/study_area_polygon/" + study_area
if not os.path.exists(sa_poly_folder):
    os.mkdir(sa_poly_folder)
    print("Successfully created folder " + sa_poly_folder)

# make subfolders with study area name for storing workflow raw/processed data and exports
for path in paths:
    sa_folder = path + study_area + "/"

    if not os.path.exists(sa_folder):
        os.mkdir(sa_folder)
        print("Successfully created folder " + sa_folder)

# Create folders for raw data
for path in paths[1:2]:
    raw_path = path + study_area + "/raw/"

    if not os.path.exists(raw_path):
        os.makedirs(raw_path)
        print("Successfully created folder " + raw_path)

# Create folders for processed data
for path in paths[0:3]:
    process_path = path + study_area + "/processed/"

    if not os.path.exists(process_path):
        os.makedirs(process_path)
        print("Successfully created folder " + process_path)


# make subfolders with study area name for storing results
for path in paths[3:6]:

    sub_folders = ["/maps_static/", "/maps_interactive/", "/plots/", "/data/"]

    for s in sub_folders:
        result_path = path + study_area + s

        if not os.path.exists(result_path):
            os.makedirs(result_path)
            print("Successfully created folder " + result_path)

# make subfolders with study area name for storing exports
sub_folders = ["/html/", "/pdf/"]
for s in sub_folders:
    export_path = exports_path + study_area + s

    if not os.path.exists(export_path):
        os.makedirs(export_path)
        print("Successfully created folder " + export_path)
