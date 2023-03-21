from compare_clusters import plot_side_by_side
from cluster_sat_data import fetch_data
import rioxarray
import numpy as np
from fix_projection import test_plotting
import pylab
import plotly.express as px

# TODO: use fmask; change scene from list, change fetch data to get min cloud cover
# figure out a good way to plot comparison with subplots
#need to match years (also time of year will affect clusters)
"""
#colors = px.colors.qualitative.Plotly
colors = []
cm = pylab.get_cmap('gist_rainbow')
for i in range(21):
    colors.append(cm(1.*i/21)[0:-1])

print(type(colors[0]))

print(px.colors.qualitative.Dark24)
"""

plot_side_by_side(lat=49.2827, lon=-123.120, n_clusters=3)
