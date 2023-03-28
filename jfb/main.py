from compare_clusters import plot_side_by_side, save_cluster
from cluster_sat_data import fetch_data
import rioxarray
import numpy as np
import pylab
import plotly.express as px
from copern_data import plot_classified_data, get_copern_xarray, get_feature_dict
from plotly.subplots import make_subplots

# TODO: 
#use fmask
#change scene from list
#change fetch data to get min cloud cover
#need to match years (also time of year will affect clusters)
#flip maps

# test different locations
# save dict of settings
# test diff band numbers and combinations, test different algorithms, diff cluster amounts

#bands = ["B07", "B06", "B04"]
#save_cluster(lat=49.2827, lon=-123.120, date="2015-06-01/2015-06-30", size=0.10, algorithm="knn", n_clusters=3, bands="all")
save_cluster(lat=49.2827, lon=-123.120, date="2015-06-01/2015-06-30", size=0.10, algorithm="knn", n_clusters=4, bands="all")
save_cluster(lat=49.2827, lon=-123.120, date="2015-06-01/2015-06-30", size=0.10, algorithm="knn", n_clusters=5, bands="all")
save_cluster(lat=49.2827, lon=-123.120, date="2015-06-01/2015-06-30", size=0.10, algorithm="knn", n_clusters=6, bands="all")
