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
    "Percent matched segments",
    "Percent of matched network length",
    "Local min of % matched segments",
    "Local max of % matched segments",
    "Local average of % matched segments",
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
