from compare_clusters import plot_side_by_side
from cluster_sat_data import fetch_data
import rioxarray
import numpy as np
from fix_projection import test_plotting
import pylab
import plotly.express as px
from copern_data import plot_classified_data, get_copern_xarray, get_feature_dict
from plotly.subplots import make_subplots

# TODO: use fmask; change scene from list, change fetch data to get min cloud cover
# figure out a good way to plot comparison with subplots
#need to match years (also time of year will affect clusters)



plot_side_by_side(lat=49.2827, lon=-123.120, n_clusters=3)
