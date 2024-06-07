### FUNCTIONS FOR FOLIUM PLOTTING
import folium
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import contextily as cx
from collections import Counter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython.display import Image, HTML, display

exec(open("../settings/yaml_variables.py").read())
exec(open("../settings/plotting.py").read())
exec(open("../settings/tiledict.py").read())
exec(open("../settings/yaml_variables.py").read())


def make_foliumplot(feature_groups, layers_dict, center_gdf, center_crs, attr=None):

    """
    Creates a folium plot from a list of already generated feature groups,
    centered around the centroid of the center_gdf.

    Parameters
    ----------
    feature_groups : list
        List of folium FeatureGroup objects to display on the map in desired order
    layers_dict : dict
        Dictionary of folium TileLayers to include in the map
    center_gdf : geopandas GeoDataFrame
        GeoDataFrame with shapely Point objects as geometries; its centroid will be used for map centering.
    center_crs: epsg crs
        Coordinate system of the center_gdf.
    attr:
        Attribution for vector map data
    Returns
    ----------
    folium map object
    """

    # FIND CENTER (RELATIVE TO NODES) AND CONVERT TO EPSG 4326 FOR FOLIUM PLOTTING
    centergdf = gpd.GeoDataFrame(geometry=center_gdf.dissolve().centroid)
    centergdf.set_crs(center_crs)
    centergdf = centergdf.to_crs("EPSG:4326")
    mycenter = (centergdf["geometry"][0].y, centergdf["geometry"][0].x)

    # CREATE MAP OBJECT
    if attr is not None:

        m = folium.Map(location=mycenter, zoom_start=13, tiles=None, attr=attr)

    else:
        m = folium.Map(location=mycenter, zoom_start=13, tiles=None)

    # ADD TILE LAYERS
    for key in layers_dict.keys():
        layers_dict[key].add_to(m)

    # ADD FEATURE GROUPS
    for fg in feature_groups:
        fg.add_to(m)

    # ADD LAYER CONTROL
    folium.LayerControl().add_to(m)

    return m


def make_edgefeaturegroup(gdf, myweight, mycolor, nametag, show_edges=True, myalpha=1):
    """
    Parameters
    ----------
    gdf : geopandas GeoDataFrame
        geodataframe containing the edges to be plotted as LineStrings in the geometry column.
    myweight : int
        numerical value - weight of plotted edges
    mycolor : str
        color of plotted edges (can be hex code)
    nametag : str
        feature group name to be displayed in the legend
    show_edges : bool
        for display of edges upon map generation, default is true
    Returns
    ----------
    folium FeatureGroup object
    """

    #### convert to espg 4326 for folium plotting
    gdf = gdf.to_crs("epsg:4326")

    locs = []  # initialize list to store coordinates

    for geom in gdf["geometry"]:  # for each of the linestrings,
        my_locs = [
            (c[1], c[0]) for c in geom.coords
        ]  # extract locations as list points
        locs.append(my_locs)  # add to list of coordinates for this feature group

    # make a polyline containing all edges
    my_line = folium.PolyLine(
        locations=locs, weight=myweight, color=mycolor, opacity=myalpha
    )

    # make a feature group
    fg_es = folium.FeatureGroup(name=nametag, show=show_edges)

    # add the polyline to the feature group
    my_line.add_to(fg_es)

    return fg_es


def make_nodefeaturegroup(gdf, mysize, mycolor, nametag, show_nodes=True):
    """
    Creates a feature group ready to be added to a folium map object from a geodataframe of points.

    Parameters
    ----------
    gdf : geopandas GeoDataFrame
        GeoDataFrame containing the nodes to be plotted as Points in the geometry column.
    myweight : int
        weight of plotted edges
    mycolor : str
        (can be hex code) - color of plotted edges
    nametag : str
        feature group name to be displayed in the legend
    show_edges : bool
        for display of edges upon map generation, default is true
    Returns
    ----------
    folium FeatureGroup object
    """

    #### convert to espg 4326 for folium plotting
    gdf = gdf.to_crs("epsg:4326")

    fg_no = folium.FeatureGroup(name=nametag, show=show_nodes)

    for geom in gdf["geometry"]:

        folium.Circle(
            location=(geom.y, geom.x),
            radius=mysize,
            color=mycolor,
            opacity=1,
            fill_color=mycolor,
            fill_opacity=1,
        ).add_to(fg_no)

    return fg_no


def make_markerfeaturegroup(gdf, nametag="Show markers", show_markers=False):
    """
    Parameters
    ----------
    gdf : geopandas GeoDataFrame
        geodataframe containing the geometries which map markers should be plotted on.
    nametag : str
        feature group name to be displayed in the legend
    show_edges : bool
        for display of markers upon map generation, default is false
    Returns
    ----------
    folium FeatureGroup object
    """

    #### convert to espg 4326 for folium plotting
    gdf = gdf.to_crs("epsg:4326")

    locs = []  # initialize list to store coordinates

    for geom in gdf["geometry"]:  # for each of the linestrings,
        my_locs = [
            (c[1], c[0]) for c in geom.coords
        ]  # extract locations as list points
        locs.append(my_locs[0])  # add to list of coordinates for this feature group

    # make a feature group
    fg_ms = folium.FeatureGroup(name=nametag, show=show_markers)

    for loc in locs:
        folium.Marker(loc).add_to(fg_ms)

    return fg_ms


def save_fig(fig, filepath, dpi=pdict["dpi"], plot_res=plot_res):

    if plot_res == "high":
        fig.savefig(filepath + ".svg", dpi=dpi)

    elif plot_res == "low":
        fig.savefig(filepath + ".png", dpi=dpi)

    else:
        print(
            "Please provide a valid input for the image resolution! Valin inputs are 'low' and 'high'"
        )


def plot_grid_results(
    grid,
    plot_cols,
    plot_titles,
    filepaths,
    cmaps,
    alpha,
    cx_tile,
    no_data_cols,
    na_facecolor=pdict["nodata_face"],
    na_edgecolor=pdict["nodata_edge"],
    na_linewidth=pdict["line_nodata"],
    na_hatch=pdict["nodata_hatch"],
    na_alpha=pdict["alpha_nodata"],
    na_legend=nodata_patch,
    figsize=pdict["fsmap"],
    dpi=pdict["dpi"],
    crs=study_crs,
    legend=True,
    set_axis_off=True,
    legend_loc="upper left",
    use_norm=False,
    norm_min=None,
    norm_max=None,
    # formats=["png", "svg"],
    plot_res=plot_res,
    attr=None,
):

    """
    Make multiple choropleth maps of e.g. grid with analysis results based on a list of geodataframe columns to be plotted
    and save them in separate files
    Arguments:
        grid (gdf): geodataframe with polygons to be plotted
        plot_cols (list): list of column names (strings) to be plotted
        plot_titles (list): list of strings to be used as plot titles
        cmaps (list): list of color maps
        alpha(numeric): value between 0-1 for setting the transparency of the plots
        cx_tile(cx tileprovider): name of contextily tile to be used for base map
        no_data_cols(list): list of column names used for generating no data layer in each plot
        na_facecolor(string): name of color used for the no data layer fill
        na_edegcolor(string): name of color used for the no data layer outline
        na_hatch: hatch pattern used for no data layer
        na_alpha (numeric): value between 0-1 for setting the transparency of the plots
        na_linewidth (numeric): width of edge lines of no data grid cells
        na_legend(matplotlib Patch): patch to be used for the no data layer in the legend
        figsize(tuple): size of each plot
        dpi(numeric): resolution of saved plots
        crs (string): name of crs used for the grid (to set correct crs of basemap)
        legend (bool): True if a legend/colorbar should be plotted
        set_axis_off (bool): True if axis ticks and values should be omitted
        legend_loc (string): Position of map legend (see matplotlib doc for valid entries)
        use_norm (bool): True if colormap should be defined based on provided min and max values
        norm_min(numeric): min value to use for norming color map
        norm_max(numeric): max value to use for norming color map
        #formats (list): list of file formats
        attr (string): optional attribution

    Returns:
        None
    """

    if use_norm is True:
        assert norm_min is not None, print("Please provide a value for norm_min")
        assert norm_max is not None, print("Please provide a value for norm_max")

    for i, c in enumerate(plot_cols):

        fig, ax = plt.subplots(1, figsize=figsize)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3.5%", pad="1%")

        if use_norm is True:

            cbnorm = colors.Normalize(vmin=norm_min[i], vmax=norm_max[i])

            grid.plot(
                cax=cax,
                ax=ax,
                column=c,
                legend=legend,
                alpha=alpha,
                norm=cbnorm,
                cmap=cmaps[i],
            )

        else:
            grid.plot(
                cax=cax,
                ax=ax,
                column=c,
                legend=legend,
                alpha=alpha,
                cmap=cmaps[i],
            )
        cx.add_basemap(ax=ax, crs=crs, source=cx_tile)

        if attr is not None:
            cx.add_attribution(ax=ax, text="(C) " + attr)
            txt = ax.texts[-1]
            txt.set_position([1, 0.00])
            txt.set_ha("right")
            txt.set_va("bottom")

        ax.set_title(plot_titles[i])

        if set_axis_off:
            ax.set_axis_off()

        # add patches in grid cells with no data on edges
        if type(no_data_cols[i]) == tuple:

            grid[
                (grid[no_data_cols[i][0]].isnull())
                & (grid[no_data_cols[i][1]].isnull())
            ].plot(
                ax=ax,
                facecolor=na_facecolor,
                edgecolor=na_edgecolor,
                linewidth=na_linewidth,
                hatch=na_hatch,
                alpha=na_alpha,
            )

        else:
            grid[grid[no_data_cols[i]].isnull()].plot(
                ax=ax,
                facecolor=na_facecolor,
                edgecolor=na_edgecolor,
                linewidth=na_linewidth,
                hatch=na_hatch,
                alpha=na_alpha,
            )

        ax.legend(handles=[na_legend], loc=legend_loc)

        if plot_res == "high":
            fig.savefig(filepaths[i] + ".svg", dpi=dpi)
        else:
            fig.savefig(filepaths[i] + ".png", dpi=dpi)

        # for f in formats:
        #     fig.savefig(filepaths[i] + "." + f, dpi=dpi)


def plot_multiple_grid_results(
    grid,
    plot_cols,
    plot_titles,
    filepath,
    cmap,
    alpha,
    cx_tile,
    no_data_cols,
    na_facecolor=pdict["nodata_face"],
    na_edgecolor=pdict["nodata_edge"],
    na_linewidth=pdict["line_nodata"],
    na_hatch=pdict["nodata_hatch"],
    na_alpha=pdict["alpha_nodata"],
    na_legend=nodata_patch,
    figsize=pdict["fsmap"],
    dpi=pdict["dpi"],
    crs=study_crs,
    legend=True,
    set_axis_off=True,
    legend_loc="upper left",
    use_norm=False,
    norm_min=None,
    norm_max=None,
    wspace=0.12,
):

    """
    Make multiple choropleth maps of e.g. grid with analysis results based on a list of geodataframe columns to be plotted
    normed to the same max value (equal color bar scales!)
    and save them in one single file (displayed in subplots side by side)

    Arguments:
        grid (gdf): geodataframe with polygons to be plotted
        plot_cols (list): list of column names (strings) to be plotted
        plot_titles (list): list of strings to be used as plot titles
        filepath (str): filepath to save the image
        cmap: color map to be used for all plots
        alpha(numeric): value between 0-1 for setting the transparency of the plots
        cx_tile(cx tileprovider): name of contextily tile to be used for base map
        no_data_cols(list): list of column names used for generating no data layer in each plot
        na_facecolor(string): name of color used for the no data layer fill
        na_edegcolor(string): name of color used for the no data layer outline
        na_linewidth (numeric): width of edge lines of no data grid cells
        na_hatch: hatch pattern used for no data layer
        na_alpha (numeric): value between 0-1 for setting the transparency of the plots
        na_legend(matplotlib Patch): patch to be used for the no data layer in the legend
        figsize(tuple): size of each plot
        dpi(numeric): resolution of saved plots
        crs (string): name of crs used for the grid (to set correct crs of basemap)
        legend (bool): True if a legend/colorbar should be plotted
        set_axis_off (bool): True if axis ticks and values should be omitted
        legend_loc (string): Position of map legend (see matplotlib doc for valid entries)
        use_norm (bool): True if colormap should be defined based on provided min and max values
        norm_min(numeric): min value to use for norming color map
        norm_max(numeric): max value to use for norming color map
        wspace (float): relative horizontal space between sub plots. If None, will use default.


    Returns:
        None
    """

    if use_norm is True:
        assert norm_min is not None, print("Please provide a value for norm_min")
        assert norm_max is not None, print("Please provide a value for norm_max")

    fig, ax = plt.subplots(1, len(plot_cols), figsize=figsize)

    for i, c in enumerate(plot_cols):

        if use_norm is True:

            cbnorm = colors.Normalize(vmin=norm_min, vmax=norm_max)

            grid.plot(
                ax=ax[i],
                column=c,
                legend=legend,
                alpha=alpha,
                norm=cbnorm,
                cmap=cmap,
            )

        else:
            grid.plot(
                ax=ax[i],
                column=c,
                legend=legend,
                alpha=alpha,
                cmap=cmap,
            )
        cx.add_basemap(ax=ax[i], crs=crs, source=cx_tile)
        ax[i].set_title(plot_titles[i])

        if set_axis_off:
            ax[i].set_axis_off()

        # add patches in grid cells with no data on edges
        if type(no_data_cols[i]) == tuple:

            grid[
                (grid[no_data_cols[i][0]].isnull())
                & (grid[no_data_cols[i][1]].isnull())
            ].plot(
                ax=ax[i],
                facecolor=na_facecolor,
                edgecolor=na_edgecolor,
                linewidth=na_linewidth,
                hatch=na_hatch,
                alpha=na_alpha,
            )

        else:
            grid[grid[no_data_cols[i]].isnull()].plot(
                ax=ax[i],
                facecolor=na_facecolor,
                edgecolor=na_edgecolor,
                linewidth=na_linewidth,
                hatch=na_hatch,
                alpha=na_alpha,
            )

        ax[i].legend(handles=[na_legend], loc=legend_loc)

    # add equally scaled colorbars to all plots
    for myax in ax:
        divider = make_axes_locatable(myax)
        cax = divider.append_axes("right", size="3.5%", pad="1%")
        plt.colorbar(
            cax=cax,
            mappable=cm.ScalarMappable(norm=cbnorm, cmap=cmap),
        )

    if wspace is not None:
        fig.subplots_adjust(wspace=wspace)

    fig.savefig(filepath, dpi=dpi)


def compute_folium_bounds(gdf):

    gdf_wgs84 = gdf.to_crs("EPSG:4326")

    gdf_wgs84["Lat"] = gdf_wgs84.geometry.y
    gdf_wgs84["Long"] = gdf_wgs84.geometry.x
    sw = gdf_wgs84[["Lat", "Long"]].min().values.tolist()
    ne = gdf_wgs84[["Lat", "Long"]].max().values.tolist()

    return [sw, ne]


def plot_saved_maps(filepaths, figsize=pdict["fsmap"], alpha=None, plot_res=plot_res):

    """
    Helper function for printing saved plots/maps/images (up to two maps plotted side by side)

    Arguments:
        filepaths(list): list of filepaths of images to be plotted
        figsize(tuple): figsize
        alpha(list): list of len(filepaths) with values between 0-1 for setting the image transparency

    Returns:
        None
    """

    assert len(filepaths) <= 2, print(
        "This function cam plot max two images at a time!"
    )

    if plot_res == "low":

        filepaths = [f + ".png" for f in filepaths]

        fig = plt.figure(figsize=figsize)

        for i, f in enumerate(filepaths):

            img = plt.imread(f)
            ax = fig.add_subplot(1, 2, i + 1)

            if alpha is not None:

                plt.imshow(img, alpha=alpha[i])

            else:
                plt.imshow(img)

            ax.set_axis_off()

        fig.subplots_adjust(wspace=0)

    elif plot_res == "high":

        filepaths = [f + ".svg" for f in filepaths]

        filepaths.reverse()

        html_string = "<div class='row'></div>"

        for i, f in enumerate(filepaths):

            if alpha is None:
                img_html = "<img src='" + f + "'style='width:49%'> </img>"
                html_string = html_string[:17] + img_html + html_string[17:]

            else:
                if alpha[i] != 0:
                    img_html = "<img src='" + f + "'style='width:49%'> </img>"
                    html_string = html_string[:17] + img_html + html_string[17:]

        display(HTML(html_string))


def compare_print_network_length(osm_length, ref_length):

    print(f"Length of the OSM data set: {osm_length/1000:.2f} km")
    print(f"Length of the reference data set: {ref_length/1000:.2f} km")

    diff = ref_length - osm_length
    percent_diff = abs(ref_length - osm_length) / osm_length * 100

    print("\n")

    if diff > 0:
        print(
            f"The reference data set is {abs(diff)/1000:.2f} km longer than the OSM data set."
        )
        print(
            f"The reference data set is {percent_diff:.2f}% longer than the OSM data set."
        )

    elif diff < 0:
        print(
            f"The reference data set is {abs(diff)/1000:.2f} km shorter than the OSM data set."
        )
        print(
            f"The reference data set is {percent_diff:.2f}% shorter than the OSM data set."
        )

    elif diff == 0:
        print("The OSM and reference data sets have the same length.")


# def compare_print_network_length(osm_length, ref_length):

#     print(f"Length of the OSM data set: {osm_length/1000:.2f} km")
#     print(f"Length of the reference data set: {ref_length/1000:.2f} km")

#     h = max([ref_length, osm_length])
#     l = min([ref_length, osm_length])

#     diff = h - l

#     percent_diff = (osm_length - ref_length) / osm_length * 100

#     if ref_length > osm_length:
#         comparison = "shorter"
#     elif osm_length > ref_length:
#         comparison = "longer"

#     print("\n")

#     print(
#         f"The OSM data set is {diff/1000:.2f} km {comparison} than the reference data set."
#     )
#     print(
#         f"The OSM data set is {percent_diff:.2f}% {comparison} than the reference data set."
#     )


def print_node_sequence_diff(degree_sequence_before, degree_sequence_after, name):

    """
    Helper function for printing the node degree counts before and after network simplification.

    Arguments:
        degree_sequence_before(list): sorted list with node degrees from non-simplified graph
        degree_sequence_after(dict): sorted list with node degrees from simplified graph

    Returns:
        None
    """

    before = dict(Counter(degree_sequence_before))
    after = dict(Counter(degree_sequence_after))

    print(f"Before the network simplification the {name} graph had:")

    for k, v in before.items():
        print(f"- {v} node(s) with degree {k}")

    print("\n")

    print(f"After the network simplification the {name} graph had:")

    for k, v in after.items():
        print(f"- {v} node(s) with degree {k}")

    print("\n")


def print_network_densities(density_dictionary, data_label):

    """
    Helper function for printing the network densities

    Arguments:
        density_dictionary (dict): dictionary with results of computation of network densities
        data_label(string): name of dataset

    Returns:
        None
    """

    edge_density = density_dictionary["network_density"]["edge_density_m_sqkm"]
    node_density = density_dictionary["network_density"]["node_density_count_sqkm"]
    dangling_node_density = density_dictionary["network_density"][
        "dangling_node_density_count_sqkm"
    ]

    print(f"In the {data_label} data, there are:")
    print(f" - {edge_density:.2f} meters of cycling infrastructure per km2.")
    print(f" - {node_density:.2f} nodes in the cycling network per km2.")
    print(
        f" - {dangling_node_density:.2f} dangling nodes in the cycling network per km2."
    )

    print("\n")


def make_bar_plot(
    data,
    bar_labels,
    y_label,
    x_positions,
    title,
    bar_colors,
    filepath,
    alpha=pdict["alpha_bar"],
    figsize=pdict["fsbar"],
    bar_width=pdict["bar_double"],
    dpi=pdict["dpi"],
    ylim=None  # ,
    # formats=["png", "svg"],
):

    """
    Make a bar plot using matplotlib.

    Arguments:
        data (list): list of values to be plotted
        bar_labels (list): list of labels for x-axis/bars
        y_label (string): label for the y-axis
        x_positions (list): list of positions on x-axis where ticks and labels should be placed
        title (string): title of plot
        bar_colors (list): list of colors to be used for bars. Must be same length as data.
        filepath (string): Filepath where plot will be saved
        alpha (numeric): value between 0-1 used to set bar transparency
        figsize (tuple): size of the plot
        bar_width (numeric): width of each bar
        dpi (numeric): resolution of the saved plot
        ylim (numeric): upper limit for y-axis
        formats (list): list of file formats

    Returns:
        fig (matplotlib figure): the plot figure
    """

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for i, d in enumerate(data):
        ax.bar(x_positions[i], d, width=bar_width, alpha=alpha, color=bar_colors[i])

    ax.set_title(title)
    ax.set_xticks(x_positions, bar_labels)
    ax.set_ylabel(y_label)
    if ylim is not None:
        ax.set_ylim([0, ylim])

    # for f in formats:
    #     fig.savefig(filepath + "." + f, dpi=dpi)

    if plot_res == "high":
        fig.savefig(filepath + ".svg", dpi=dpi)
    else:
        fig.savefig(filepath + ".png", dpi=dpi)

    return fig


def make_bar_plot_side(
    x_axis,
    data_osm,
    data_ref,
    legend_labels,
    title,
    x_ticks,
    x_labels,
    x_label,
    y_label,
    filepath,
    bar_colors,
    width=pdict["bar_single"],
    alpha=pdict["alpha_bar"],
    figsize=pdict["fsbar_small"],
    dpi=pdict["dpi"]  # ,
    # formats=["png", "svg"],
):

    """
    Make a bar subplot using matplotlib where two datasets with corresponding values are plotted side by side.

    Arguments:
        x_axis (list): list of positions on x-axis. Expected input is len(x_axis) == number of values to be plotted
        data_osm (list): values to be plotted
        data_ref (list): values to be plotted
        legend_labels (list): list of legend labels for the bars (one for each dataset)
        title (string): title of plot
        x_ticks (list): list of x-tick locations
        x_labels (list): list of tick labesl
        y_label (string): label for the y-axis
        filepath (string): Filepath where plot will be saved
        bar_colors (list): list of colors to be used for bars. Expects one color for each dataset.
        width (numeric): width of each bar
        alpha (numeric): value between 0-1 used to set bar transparency
        figsize (tuple): size of the plot
        dpi (numeric): resolution of the saved plot
        formats (list): list of file formats

    Returns:
        fig (matplotlib figure): the plot figure
    """

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.bar(
        x=x_axis[0],
        height=data_osm,
        label=legend_labels[0],
        width=width,
        alpha=alpha,
        color=bar_colors[0],
    )
    ax.bar(
        x=x_axis[1],
        height=data_ref,
        label=legend_labels[1],
        width=width,
        alpha=alpha,
        color=bar_colors[1],
    )
    ax.set_xticks(x_ticks, x_labels)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()

    # for f in formats:
    #     fig.savefig(filepath + "." + f, dpi=dpi)

    if plot_res == "high":

        fig.savefig(filepath + ".svg", dpi=dpi)
    else:
        fig.savefig(filepath + ".png", dpi=dpi)

    return fig


def make_bar_subplots(
    subplot_data,
    nrows,
    ncols,
    bar_labels,
    y_label,
    x_positions,
    title,
    bar_colors,
    filepath,
    alpha=pdict["alpha_bar"],
    figsize=pdict["fsbar_sub"],
    bar_width=pdict["bar_double"],
    dpi=pdict["dpi"],
    # formats=["png", "svg"],
    ylim=None,
    wspace=None,
):

    """
    Make a bar plot with several subplots using matplotlib

    Arguments:
        subplot_data (list): nested list with values to be plotted
        nrows (int): number of rows in subplot
        ncols (int): number of cols in subplot
        bar_labels (list of lists): labels for x-axis
        y_label (string): label for the y-axis
        x_positions (list of lists): list of positions on x-axis. Expected input is len(x_axis) == number of values to be plotted (len of nested list)
        title (string): title of plot
        bar_colors (list): list of colors to be used for bars. Expects a list with the same length as the longest nested list in subplot_data
        filepath (string): Filepath where plot will be saved
        alpha (numeric): value between 0-1 used to set bar transparency
        figsize (tuple): size of the plot
        bar_width (numeric): width of bars
        dpi (numeric): resolution of the saved plot
        formats (list): list of file formats
        ylim (numeric): upper limit for all y-axis
        wspace (float): relative horizontal space between sub plots. If None, will use default.

    Returns:
        fig (matplotlib figure): the plot figure
    """

    figsize = (figsize[0] * ncols, figsize[1] * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    axes = axes.flatten()
    for i, data in enumerate(subplot_data):
        for z, d in enumerate(data):
            axes[i].bar(
                x_positions[i][z], d, width=bar_width, alpha=alpha, color=bar_colors[z]
            )

        axes[i].set_ylabel(y_label[i])
        axes[i].set_xticks(x_positions[i], bar_labels[i])
        axes[i].set_title(title[i])
        if ylim is not None:
            axes[i].set_ylim([0, ylim])

    if wspace is not None:
        fig.subplots_adjust(wspace=wspace)

    # for f in formats:
    #     fig.savefig(filepath + "." + f, dpi=dpi)

    if plot_res == "high":
        fig.savefig(filepath + ".svg", dpi=dpi)
    else:
        fig.savefig(filepath + ".png", dpi=dpi)

    return fig
