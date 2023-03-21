import rioxarray
import cartopy
from matplotlib import pyplot as plt
import matplotlib as mpl
from copy import copy
from pyresample.utils.cartopy import Projection
import numpy as np
import pylab


#def plot_classified_data(band_xarray, lat_range, lon_range):
def plot_classified_data(lat_range, lon_range):
    the_tif  = "./data/copernicus/PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif"
    band_xarray = rioxarray.open_rasterio(the_tif)

    flag_meanings = band_xarray.flag_meanings.split(", ")
    flag_values = list(map(int, band_xarray.flag_values.split(", ")))

    n_cats = len(flag_values)
    cat = np.arange(n_cats)

    val_to_cat = dict(zip(flag_values, cat))
    cat_to_mean = dict(zip(cat, flag_meanings))

    band_xarray = band_xarray.sel(y=slice(lat_range[0], lat_range[1]), x=slice(lon_range[0], lon_range[1]))

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
    ll_x, ll_y = band_xarray.rio.transform()*(0,nrows+1)
    ur_x, ur_y = band_xarray.rio.transform()*(ncols+1,0)
    extent = (ll_x,ur_x, ll_y, ur_y)

    cartopy_crs = Projection(band_xarray.rio.crs, bounds=extent)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={"projection": cartopy_crs})
    ax.gridlines(linewidth=2)
    ax.add_feature(cartopy.feature.GSHHSFeature(scale="coarse", levels=[1, 2, 3]))
    ax.set_extent(cartopy_crs.bounds, cartopy_crs)

    #norm = mpl.colors.BoundaryNorm(cat, cmap.N, extend='both')

    #prop_cycle = plt.rcParams['axes.prop_cycle']
    #colors = prop_cycle.by_key()['color']

    colors = []
    cm = pylab.get_cmap('gist_rainbow')
    for i in range(n_cats):
        colors.append(cm(1.*i/n_cats))

    cmap = mpl.colors.ListedColormap(colors)

    cs = ax.imshow(
        wv_raster,
        transform=cartopy_crs,
        extent=extent,
        origin="upper",
        alpha=0.8,
        cmap=cmap,
        #norm=norm,
    )

    formatter = mpl.ticker.FuncFormatter(lambda c, loc: cat_to_mean[c])
    plt.colorbar(cs, ticks=cat, format=formatter)
    plt.show()




def plot_classified_data_v2(lat_range, lon_range):
    the_tif  = "./data/copernicus/PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif"
    band_xarray = rioxarray.open_rasterio(the_tif)

    flag_meanings = band_xarray.flag_meanings.split(", ")
    flag_values = list(map(int, band_xarray.flag_values.split(", ")))

    n_cats = len(flag_values)
    cat = np.arange(n_cats)

    val_to_cat = dict(zip(flag_values, cat))
    cat_to_mean = dict(zip(cat, flag_meanings))

    band_xarray = band_xarray.sel(y=slice(lat_range[0], lat_range[1]), x=slice(lon_range[0], lon_range[1]))

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

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    #norm = mpl.colors.BoundaryNorm(cat, cmap.N, extend='both')

    #prop_cycle = plt.rcParams['axes.prop_cycle']
    #colors = prop_cycle.by_key()['color']

    colors = []
    cm = pylab.get_cmap('gist_rainbow')
    for i in range(n_cats):
        colors.append(cm(1.*i/n_cats))

    cmap = mpl.colors.ListedColormap(colors)

    cs = ax.imshow(
        wv_raster,
        alpha=0.8,
        cmap=cmap,
        #norm=norm,
    )

    formatter = mpl.ticker.FuncFormatter(lambda c, loc: cat_to_mean[c])
    plt.colorbar(cs, ticks=cat, format=formatter)
    plt.show()

#canada
#plot_classified_data(band_xarray, [83, 42], [-141, -53])

#BC
#plot_classified_data(band_xarray, [60, 48], [-139, -114])

#tiny #-123.120, 49.2827
#plot_classified_data(band_xarray, [49, 48], [-123, -122])

#print(band_xarray.attrs)