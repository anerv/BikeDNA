import matplotlib as mpl
from matplotlib import cm, colors

import matplotlib_inline.backend_inline


def set_renderer(f="svg"):
    matplotlib_inline.backend_inline.set_matplotlib_formats(f)


# Plot and map renderers
# Change renderer_map to svg to get crisp maps with the full vector data.
# Do this only for small areas (sub-city) due to html/pdf size explosion!
renderer_map = "png"
renderer_plot = "svg"

# Plot parameters
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["xtick.minor.visible"] = False
mpl.rcParams["xtick.major.size"] = 0
mpl.rcParams["xtick.labelbottom"] = True
mpl.rcParams["ytick.major.size"] = 3
mpl.rcParams["font.size"] = 10
mpl.rcParams["figure.titlesize"] = 10
mpl.rcParams["legend.title_fontsize"] = 10
mpl.rcParams["legend.fontsize"] = 9
# mpl.rcParams["figure.labelsize"] = 10 # use if figure.titlesize does not work?
mpl.rcParams["axes.labelsize"] = 10
mpl.rcParams["xtick.labelsize"] = 9
mpl.rcParams["ytick.labelsize"] = 9
mpl.rcParams["hatch.linewidth"] = 0.5


def col_to_rgb(col):
    """Return list of 3 rgb values from col (can be hex, name,..)
    Example: colname_to_rgb(purple)
    """
    import matplotlib.colors as mcols

    return list(mcols.to_rgb(col))


def convert_cmap_to_hex(cmap_name, n=None):

    if n is None:
        cmap = cm.get_cmap(cmap_name)

    else:
        cmap = cm.get_cmap(cmap_name, n)

    hex_codes = []

    for i in range(cmap.N):

        hex_codes.append(mpl.colors.rgb2hex(cmap(i)))

    return hex_codes


# Exact colors used
pink_green_cmap = convert_cmap_to_hex("PiYG", 10)
pink = pink_green_cmap[1]
green = pink_green_cmap[-2]

orange_cmap = convert_cmap_to_hex("Oranges", 10)
orange = orange_cmap[5]
light_orange = orange_cmap[4]
dark_orange = orange_cmap[8]

purple_cmap = convert_cmap_to_hex("Purples", 10)
purple = purple_cmap[6]
light_purple = purple_cmap[4]
dark_purple = purple_cmap[8]


blue_cmap = convert_cmap_to_hex("Blues", 10)
blue = blue_cmap[6]
light_blue = blue_cmap[4]
dark_blue = blue_cmap[8]

red_cmap = convert_cmap_to_hex("Reds", 10)
red = red_cmap[6]
light_red = red_cmap[5]
dark_red = red_cmap[8]


# pdict for plotting styles
pdict = {
    # grid; polygon; base barplots
    "base": "black",
    "base2": "grey",
    "compare_base": "black",  # "dimgray",
    # osm network in geopandas and folium plots
    "osm_base": purple,  # base: for nodes and edges
    "osm_emp": dark_purple,  # emphasis: for dangling nodes, component issues, etc.
    "osm_emp2": light_purple,  # emphasis 2: for 2-fold distinctions e.g. over/undershoots
    "osm_contrast": convert_cmap_to_hex("winter", 10)[1],
    "osm_contrast2": convert_cmap_to_hex("winter", 10)[-1],
    # reference network in geopandas and folium plots
    "ref_base": orange,  # base: for nodes and edges
    "ref_emp": dark_orange,  # emphasis: for dangling nodes, component issues, etc.
    "ref_emp2": light_orange,  # emphasis 2: for 2-fold distinctions e.g. over/undershoots
    "ref_contrast": convert_cmap_to_hex("RdYlBu", 10)[1],
    "ref_contrast2": convert_cmap_to_hex("autumn", 10)[-2],
    # colormaps for grid cell plots
    "pos": "Blues",  # Positive values (but not percentages)
    "neg": "Reds",  # Negative/Missing/Unmatched values
    "diff": "RdBu",  # for osm-ref difference plots (alternatives: "PiYG", "PRGn", "PuOr")
    "seq": "YlGnBu",  # for sequential plots where low should not be white (usually percentages)
    # alpha (transparency) values (alternatives: PuRd, RdPu, PbBuGn)
    "alpha_back": 0.5,  # for unicolor plots with relevant background
    "alpha_bar": 0.7,  # for partially overlapping stats barplots
    "alpha_grid": 0.9,  # for multicolor/divcolor gridplots
    "alpha_nodata": 0.3,  # for no data patches
    # linewidths (base, emphasis, emphasis2)
    "line_base": 1,
    "line_emp": 3,
    "line_emp2": 5,
    "line_nodata": 0.3,
    # widths for bar plots; single: for 1 value, double: for 2 values comparison
    "bar_single": 0.4,
    "bar_double": 0.75,
    # marker sizes (base, emphasis)
    "mark_base": 2,
    "mark_emp": 6,
    # list of colors for differing tagging patterns
    "basecols": convert_cmap_to_hex("tab20"),
    # for segment matching: matched vs unmatched features
    "match": blue,
    "nomatch": light_red,
    # for segment matching: semistransparent segment matches plot
    "osm_seg": light_purple,
    "osm_alpha": 0.7,
    "osm_weight": 4,
    "ref_seg": light_orange,
    "ref_alpha": 0.7,
    "ref_weight": 6,
    "mat_seg": blue,  # "#4dac26",
    "mat_alpha": 1,
    "mat_weight": 3,
    # Colors of no-data grid cell patches
    "nodata": "grey",
    "nodata_osm": "grey",  
    "nodata_ref": "grey",  
    "nodata_face": "none",
    "nodata_edge": "grey",
    "nodata_hatch": "//",
    # GLOBAL SETTINGS FOR PLOTS
    "dpi": 300,  # resolution
    # matplotlib figure size for map plots of study area
    "fsmap": (13, 7.3125),
    # size for bar plots
    "fsbar": (8, 8),
    "fsbar_small": (4, 3.5),
    "fsbar_short": (6, 3),
    "fsbar_sub": (4, 3),  # size per subplot
}

# patches for geopandas plots legend of "no data"
import matplotlib.patches as mpatches

nodata_patch = mpatches.Patch(
    facecolor=pdict["nodata_face"],
    edgecolor=pdict["nodata_edge"],
    linewidth=0.3,
    label="No data",
    hatch=pdict["nodata_hatch"],
    alpha=pdict["alpha_nodata"],
)


incompatible_true_patch = mpatches.Patch(
    facecolor=dark_blue,
    edgecolor=dark_blue,
    label="Incompatible tag combinations",
    alpha=pdict["alpha_grid"],
)

incompatible_false_patch = mpatches.Patch(
    facecolor=light_blue,
    edgecolor=light_blue,
    label="No incompatible tag combinations",
    alpha=pdict["alpha_grid"],
)

import contextily as cx

cx_tile_1 = cx.providers.CartoDB.Voyager
cx_tile_2 = cx.providers.CartoDB.Positron
