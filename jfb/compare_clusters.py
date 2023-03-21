# compare clusters with classified data.
# get composition of each cluster

from copern_data import plot_classified_data, get_copern_xarray, get_feature_dict
from cluster_sat_data import run_clustering, fetch_data
import numpy as np
from plotly.subplots import make_subplots
import rioxarray
import plotly.graph_objects as go

def plot_side_by_side(lat, lon, n_clusters):
    copern_xarray = get_copern_xarray(lat, lon)

    cluster_fig, cluster_xarray = run_clustering(lat=49.2827, lon=-123.120, n_clusters_range=[n_clusters], copern_xarray=copern_xarray)

    classified_fig = plot_classified_data(copern_xarray)

    fig = make_subplots(cols=2)
    fig.add_trace(cluster_fig["data"][0], row=1, col=1)
    fig.add_trace(classified_fig["data"][0], row=1, col=2)

    fig.show()

    calc_cluster_composition(copern_xarray.squeeze(), cluster_xarray)

def calc_cluster_composition(copern_xarray, sat_array):
    feature_dict = get_feature_dict(copern_xarray)

    n_sat = len(np.unique(sat_array))
    cats = np.unique(copern_xarray.values)
    n_cats = len(cats)

    composition_dict = {}
    for i in range(n_sat):
        composition_dict["cluster_" + str(i)] = {}
        inds = sat_array == i
        n_cluster_values = np.sum(inds)
        for j in cats:
            percent_comp = np.sum(copern_xarray.values[inds] == j)/n_cluster_values
            composition_dict["cluster_" + str(i)][feature_dict[j]] = percent_comp

    for i in range(n_sat):
        percent_comps = composition_dict["cluster_" + str(i)]

        fig = go.Figure(data=[go.Pie(labels=list(percent_comps.keys()), values=list(percent_comps.values()))])
        fig.show()





