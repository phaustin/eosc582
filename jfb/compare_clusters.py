# compare clusters with classified data.
# get composition of each cluster

from copern_data import plot_classified_data, get_copern_xarray, get_feature_dict
from cluster_sat_data import run_clustering, fetch_data
import numpy as np
from plotly.subplots import make_subplots
import rioxarray
import plotly.graph_objects as go
import plotly.express as px
import json

def plot_side_by_side(lat, lon, date, size=0.25, algorithm="knn", n_clusters=3, bands="all"):
    # need to pass copern_array to clustering to align the projections
    copern_xarray = get_copern_xarray(lat, lon, size)
    cluster_fig, cluster_xarray = run_clustering(lat=lat, lon=lon, date=date, algorithm=algorithm, n_clusters=n_clusters, bands=bands, xarray=copern_xarray)

    classified_fig = plot_classified_data(copern_xarray)

    fig = make_subplots(cols=2)
    fig.add_trace(cluster_fig["data"][0], row=1, col=1)
    fig.add_trace(classified_fig["data"][0], row=1, col=2)
    
    _, cat_to_mean = get_feature_dict(copern_xarray)
    fig.update_coloraxes(
        colorscale=px.colors.qualitative.Dark24,
        colorbar_tickvals=list(cat_to_mean.keys()),
        colorbar_ticktext=list(cat_to_mean.values()),
        colorbar_tickmode="array",
        )

    fig.show()

    composition_dict = calc_cluster_composition(copern_xarray.squeeze(), cluster_xarray)

    #plot pie plots
    for i in range(n_sat):
        percent_comps = composition_dict["cluster_" + str(i)]

        fig = go.Figure(data=[go.Pie(labels=list(percent_comps.keys()), values=list(percent_comps.values()))])
        fig.show()

def save_cluster(lat, lon, date, size=0.25, algorithm="knn", n_clusters=3, bands="all"):
    # need to pass copern_array to clustering to align the projections
    copern_xarray = get_copern_xarray(lat, lon, size)
    cluster_fig, cluster_xarray = run_clustering(lat=lat, lon=lon, date=date, algorithm=algorithm, n_clusters=n_clusters, bands=bands, xarray=copern_xarray)
    composition_dict = calc_cluster_composition(copern_xarray.squeeze(), cluster_xarray)

    cluster_dict = {
        "lat": lat,
        "lon": lon,
        "date": date,
        "size": size,
        "algorithm": algorithm,
        "n_clusters": n_clusters,
        "bands": bands,
        "composition_dict": composition_dict
    }

    # add dict to json
    with open("cluster_data.json",'r+') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data["data"].append(cluster_dict)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent = 4)

def calc_cluster_composition(copern_xarray, sat_array):
    feature_dict, _ = get_feature_dict(copern_xarray)

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

    return composition_dict





