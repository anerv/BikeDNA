import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

exec(open("../settings/plotting.py").read())
exec(open("../settings/yaml_variables.py").read())

cell_hover = {"selector": "td:hover", "props": [("background-color", "#ffffb3")]}

row_hover = {"selector": "tr:hover", "props": [("background-color", "#eff7fa")]}

caption = {
    "selector": "caption",
    "props": "caption-side: top; text-align:center; font-weight: bold; font-size:20px; color: white;",
}

cell_style = {"selector": "td", "props": "text-align: center; font-weight: bold; "}


#####

# Styling setting for reference results
index_name_ref = {
    "selector": ".index_name",
    "props": f"color:{pdict['ref_base']}; font-size:14px;",
}

columns_ref = {
    "selector": "th",
    "props": f"color: {pdict['ref_base']}; font-size:14px;",
}


def format_ref_style(styler):
    styler.set_caption(f"Intrinsic Quality Metrics - {reference_name} data")
    caption_here = caption.copy()
    caption_here["props"] += "background-color: " + pdict["ref_base"] + ";"
    styler.format(precision=0, na_rep=" - ", thousands=",")
    styler.format(
        formatter={" ": lambda x: f"{str(round(x))}%"},
        subset=pd.IndexSlice["Largest component's share of network length", :],
    )
    styler.set_table_styles(
        [cell_hover, row_hover, columns_ref, caption_here, index_name_ref, cell_style],
        overwrite=False,
    )
    styler.applymap_index(
        lambda v: f"color:{pdict['ref_base']}; font-size:14px;",
        axis=0,
    )

    return styler


#####

# Styling setting for osm results
index_name_osm = {
    "selector": ".index_name",
    "props": f"color:{pdict['osm_base']}; font-size:14px;",
}

columns_osm = {
    "selector": "th",
    "props": f"color: {pdict['osm_base']}; font-size:14px;",
}


def format_osm_style(styler):
    styler.set_caption("Intrinsic Quality Metrics - OSM data")
    caption_here = caption.copy()
    caption_here["props"] += "background-color: " + pdict["osm_base"] + ";"
    styler.format(precision=0, na_rep=" - ", thousands=",")
    styler.format(
        formatter={" ": lambda x: f"{str(round(x))}%"},
        subset=pd.IndexSlice["Largest component's share of network length", :],
    )
    styler.set_table_styles(
        [cell_hover, row_hover, columns_osm, caption_here, index_name_osm, cell_style],
        overwrite=False,
    )
    styler.applymap_index(
        lambda v: f"color:{pdict['osm_base']}; font-size:14px;",
        axis=0,
    )

    return styler


####

# Styling setting for matching results
index_name_match = {
    "selector": ".index_name",
    "props": f"color:{pdict['compare_base']}; font-size:14px;",
}

columns_match = {
    "selector": "th",
    "props": f"color: {pdict['compare_base']}; font-size:14px;",
}

pct_rows = [
    "Percent matched edges",
    "Percent matched edges",
    "Local min of % matched edges",
    "Local max of % matched edges",
    "Local average of % matched edges",
]


def format_matched_style(styler):
    styler.set_caption("Feature Matching Results")
    caption_here = caption.copy()
    caption_here["props"] += "background-color: " + pdict["compare_base"] + ";"
    styler.format(precision=0, na_rep=" - ", thousands=",")
    styler.format(
        "{:,.0f}%",
        subset=pd.IndexSlice[pct_rows, :],
    )
    styler.set_table_styles(
        [
            cell_hover,
            row_hover,
            columns_match,
            caption_here,
            index_name_match,
            cell_style,
        ],
        overwrite=False,
    )
    styler.applymap_index(
        lambda v: f"color:{pdict['compare_base']}; font-size:14px;",
        axis=0,
    )

    return styler


#####

# Styling settings for extrinsic results

index_name_extrinsic = {
    "selector": ".index_name",
    "props": f"color: {pdict['compare_base']}; font-size:14px;",
}

columns_extrinsic = {
    "selector": "th",
    "props": f"color: {pdict['compare_base']}; font-size:14px;",
}


def format_extrinsic_style(styler):
    styler.set_caption("Extrinsic Quality Comparison")
    caption_here = caption.copy()
    caption_here["props"] += "background-color: " + pdict["compare_base"] + ";"
    styler.format(
        precision=0,
        na_rep=" - ",
        thousands=",",
        formatter={
            "Percent difference": lambda x: f"{x:,.0f}%",  # f"{str(x)} %",
        },
    )
    styler.format(
        precision=2,
        subset=pd.IndexSlice[["Alpha", "Beta", "Gamma"], :],
        formatter={
            "Percent difference": lambda x: f"{x:,.0f}%",  # f"{str(x)} %",
        },
    )
    styler.format(
        "{:,.0f}%",
        precision=0,
        subset=pd.IndexSlice[
            "Largest component's share of network length", ["OSM", reference_name]
        ],
    )
    styler.set_table_styles(
        [
            cell_hover,
            row_hover,
            columns_extrinsic,
            caption_here,
            index_name_extrinsic,
            cell_style,
        ],
        overwrite=False,
    )
    styler.applymap_index(
        lambda v: f"color:{pdict['compare_base']}; font-size:20px;",
        axis=0,
    )

    return styler


#####

# # Styling setting for completeness results
# index_name_completeness = {
#     "selector": ".index_name",
#     "props": "color:white; font-weight:bold; background-color: orange; font-size:1.3em;",
# }

# columns_completeness = {
#     "selector": "th",
#     "props": "background-color: orange; color: white; font-weight:bold; font-size:1.3em;",
# }


# def format_completeness_style(styler):
#     styler.set_caption("Network Completeness Quality Metrics")
#     styler.format(
#         precision=2,
#         na_rep=" - ",
#         thousands=",",
#         formatter={
#             "pct_difference": lambda x: f"{str(x)} %",
#             "normalised_values_pct_difference": lambda x: f"{str(x)} %",
#         },
#     )
#     styler.set_table_styles(
#         [
#             cell_hover,
#             row_hover,
#             columns_completeness,
#             caption,
#             index_name_completeness,
#             cell_style,
#         ],
#         overwrite=False,
#     )
#     styler.applymap_index(
#         lambda v: "color:white; font-style: italic; font-weight:bold; background-color: orange; font-size:1em;",
#         axis=0,
#     )
#     styler.applymap(
#         style_pct_value_completeness,
#         osm_bigger="color:blue;",
#         osm_smaller="color:orange;",
#     )

#     return styler


# #####

# # Styling settings for topology results
# index_name_topo = {
#     "selector": ".index_name",
#     "props": "color:white; font-weight:bold; background-color: purple; font-size:1.3em;",
# }

# columns_topo = {
#     "selector": "th",
#     "props": "background-color: purple; color: white; font-weight:bold; font-size:1.3em;",
# }

# high_bad_topo = [
#     "dangling_node_count",
#     "dangling_node_density_sqkm",
#     "component_count",
#     "component_gaps",
#     "count_overshoots",
#     "count_undershoots",
# ]
# high_good_topo = ["largest_cc_pct_size", "largest_cc_length_km"]

# topo_slice_inverse = high_bad_topo, [
#     "pct_difference",
#     "normalised_values_pct_difference",
# ]
# topo_slice = high_good_topo, ["pct_difference", "normalised_values_pct_difference"]


# def format_topology_style(styler):
#     styler.set_caption("Network Topology Quality Metrics")
#     styler.format(
#         precision=2,
#         na_rep=" - ",
#         thousands=",",
#         formatter={
#             "pct_difference": lambda x: f"{str(x)} %",
#             "normalised_values_pct_difference": lambda x: f"{str(x)} %",
#         },
#     )
#     styler.set_table_styles(
#         [cell_hover, row_hover, columns_topo, caption, index_name_topo, cell_style],
#         overwrite=False,
#     )
#     styler.applymap_index(
#         lambda v: "color:white; font-style: italic; font-weight:bold; background-color: purple; font-size:1em;",
#         axis=0,
#     )
#     styler.applymap(
#         style_pct_value,
#         osm_better="color:blue;",
#         osm_worse="color:orange;",
#         subset=topo_slice,
#     )
#     styler.applymap(
#         style_pct_value_inversed,
#         osm_better="color:blue;",
#         osm_worse="color:orange;",
#         subset=topo_slice_inverse,
#     )

#     return styler


# def style_pct_value(v, osm_better="color:blue;", osm_worse="color:green;"):

#     """
#     Helper function for styling the dataframe with results for data topology.

#     Arguments:
#         v (numeric: value in cell to be styled
#         osm_better (str): color to use if v is above zero
#         osm_worse (str): color to use if v is smaller than zero

#     Returns:
#         osm_better (str): color
#         osm_worse (str): color
#     """

#     if v > 0:
#         return osm_better
#     elif v < 0:
#         return osm_worse
#     else:
#         None


# def style_pct_value_inversed(v, osm_better="color:blue;", osm_worse="color:green;"):

#     """
#     Helper function for styling the dataframe with results for data topology.

#     Arguments:
#         v (numeric: value in cell to be styled
#         osm_better (str): color to use if v is above zero
#         osm_worse (str): color to use if v is smaller than zero

#     Returns:
#         osm_better (str): color
#         osm_worse (str): color
#     """

#     if v > 0:
#         return osm_worse
#     elif v < 0:
#         return osm_better
#     else:
#         None


# def style_pct_value_completeness(
#     v, osm_bigger="color:blue;", osm_smaller="color:green;"
# ):

#     """
#     Helper function for styling the dataframe with results for data completeness.

#     Arguments:
#         v (numeric: value in cell to be styled
#         osm_bigger (str): color to use if v is above zero
#         osm_smaller (str): color to use if v is smaller than zero

#     Returns:
#         osm_bigger (str): color
#         osm_smaller (str): color
#     """

#     if v > 0:
#         return osm_bigger
#     elif v < 0:
#         return osm_smaller
#     else:
#         None
