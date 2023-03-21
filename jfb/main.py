from compare_clusters import plot_side_by_side
from cluster_sat_data import fetch_data
from copern_data import plot_classified_data_v2
import rioxarray
import numpy as np
from fix_projection import test_plotting

# TODO: use fmask; change scene from list, change fetch data to get min cloud cover
# figure out a good way to plot comparison with subplots
#need to match years (also time of year will affect clusters)


#plot_side_by_side(lat=49.2827, lon=-123.120, n_clusters=4)

"""
lat=49.2827
lon=-123.120
lat_range = np.array([0.25, -0.25]) + lat
lon_range = np.array([-0.25, 0.25]) + lon
plot_classified_data_v2(lat_range, lon_range)
"""

test_plotting()