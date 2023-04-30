# Read in classified data.

import rioxarray
import cartopy
from matplotlib import pyplot as plt
import matplotlib as mpl
from copy import copy
from pyresample.utils.cartopy import Projection
import numpy as np
import pylab
import plotly.express as px

def get_copern_xarray(lat, lon, size):
    lat_range = np.array([size, -size]) + lat
    lon_range = np.array([-size, size]) + lon

    copern_tif  = "./data/copernicus/PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif"
    copern_xarray = rioxarray.open_rasterio(copern_tif)
    copern_xarray = copern_xarray.sel(y=slice(lat_range[0], lat_range[1]), x=slice(lon_range[0], lon_range[1]))

    return copern_xarray

def get_feature_dict(band_xarray):
    flag_meanings = band_xarray.flag_meanings.split(", ")
    flag_values = list(map(int, band_xarray.flag_values.split(", ")))

    n_cats = len(flag_values)
    cat = np.arange(n_cats)

    val_to_cat = dict(zip(flag_values, cat))
    cat_to_mean = dict(zip(cat, flag_meanings))
    val_to_mean = dict(zip(flag_values, flag_meanings))
    return val_to_mean, cat_to_mean

def plot_classified_data(band_xarray):
    flag_meanings = band_xarray.flag_meanings.split(", ")
    flag_values = list(map(int, band_xarray.flag_values.split(", ")))

    n_cats = len(flag_values)
    cat = np.arange(n_cats)

    val_to_cat = dict(zip(flag_values, cat))
    cat_to_mean = dict(zip(cat, flag_meanings))

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

    fig = px.imshow(wv_raster, color_continuous_scale=colors)
    
    return fig

    
