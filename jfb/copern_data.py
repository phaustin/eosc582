import rioxarray
import cartopy
from matplotlib import pyplot as plt
import matplotlib as mpl
from copy import copy
from pyresample.utils.cartopy import Projection
import numpy as np
import pylab
import plotly.express as px

def get_copern_xarray(lat, lon):
    lat_range = np.array([0.25, -0.25]) + lat
    lon_range = np.array([-0.25, 0.25]) + lon

    copern_tif  = "./data/copernicus/PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif"
    copern_xarray = rioxarray.open_rasterio(copern_tif)
    copern_xarray = copern_xarray.sel(y=slice(lat_range[0], lat_range[1]), x=slice(lon_range[0], lon_range[1]))

    #scene = fetch_data(lat=49.2827, lon=-123.120, date="2015-06-01/2015-06-30")

    #sat_xarray_match = sat_xarray.rio.reproject_match(copern_xarray)

    return copern_xarray

def get_feature_dict(band_xarray):
    flag_meanings = band_xarray.flag_meanings.split(", ")
    flag_values = list(map(int, band_xarray.flag_values.split(", ")))

    n_cats = len(flag_values)
    cat = np.arange(n_cats)

    val_to_cat = dict(zip(flag_values, cat))
    cat_to_mean = dict(zip(cat, flag_meanings))
    val_to_mean = dict(zip(flag_values, flag_meanings))
    return val_to_mean

def plot_classified_data(band_xarray):
    #the_tif  = "./data/copernicus/PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif"
    #band_xarray = rioxarray.open_rasterio(the_tif)

    flag_meanings = band_xarray.flag_meanings.split(", ")
    flag_values = list(map(int, band_xarray.flag_values.split(", ")))

    n_cats = len(flag_values)
    cat = np.arange(n_cats)

    val_to_cat = dict(zip(flag_values, cat))
    cat_to_mean = dict(zip(cat, flag_meanings))

    #band_xarray = band_xarray.sel(y=slice(lat_range[0], lat_range[1]), x=slice(lon_range[0], lon_range[1]))

    out = band_xarray.rio.transform()

    wv_raster = band_xarray.data
    wv_raster = wv_raster.squeeze()

    #rewrite the classification ints
    vals = wv_raster
    vals_shape = vals.shape
    vals = list(vals.flatten())

    sort_key = [val_to_cat[v] for v in vals]
    wv_raster = np.reshape(sort_key, vals_shape)


    nrows, ncols = wv_raster.shape

    colors = px.colors.qualitative.Dark24
    #[]
    """
    cm = pylab.get_cmap('gist_rainbow')
    for i in range(n_cats):
        colors.append(cm(1.*i/n_cats)[0:-1])
    """

    fig = px.imshow(wv_raster, color_continuous_scale=colors)
    
    

    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    colors = []
    cm = pylab.get_cmap('gist_rainbow')
    for i in range(n_cats):
        colors.append(cm(1.*i/n_cats))

    cmap = mpl.colors.ListedColormap(colors)
    
    cs = ax.imshow(
        wv_raster,
        alpha=0.8,
        cmap=cmap,
    )

    formatter = mpl.ticker.FuncFormatter(lambda c, loc: cat_to_mean[c])
    plt.colorbar(cs, ticks=cat, format=formatter)
    plt.show()
    """
    return fig

    
