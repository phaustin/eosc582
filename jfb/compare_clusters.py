# compare clusters with classified data.
# get composition of each cluster

from copern_data import plot_classified_data
from cluster_sat_data import run_clustering
import numpy as np


def plot_side_by_side(lat, lon, n_clusters):
    run_clustering([n_clusters])

    lat_range = np.array([0.25, -0.25]) + lat
    lon_range = np.array([-0.25, 0.25]) + lon
    plot_classified_data(lat_range, lon_range)
    