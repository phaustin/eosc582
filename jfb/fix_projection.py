#https://corteva.github.io/rioxarray/stable/examples/reproject_match.html

from cluster_sat_data import fetch_data
import matplotlib.pyplot as plt
import numpy as np
import rioxarray
from pyproj import Proj

def print_raster(raster):
    print(
        f"shape: {raster.rio.shape}\n"
        f"resolution: {raster.rio.resolution()}\n"
        f"bounds: {raster.rio.bounds()}\n"
        f"CRS: {raster.rio.crs}\n"
    )

def test_plotting():
    scene = fetch_data(lat=49.2827, lon=-123.120, date="2015-06-01/2015-06-30")
    sat_xarray = rioxarray.open_rasterio(scene.assets['B05'].href)

    lat=49.2827
    lon=-123.120
    lat_range = np.array([0.25, -0.25]) + lat
    lon_range = np.array([-0.25, 0.25]) + lon

    copern_tif  = "./data/copernicus/PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif"
    copern_xarray = rioxarray.open_rasterio(copern_tif)
    copern_xarray = copern_xarray.sel(y=slice(lat_range[0], lat_range[1]), x=slice(lon_range[0], lon_range[1]))

    sat_xarray_match = sat_xarray.rio.reproject_match(copern_xarray)
    
    print(sat_xarray_match.coords)


    out = sat_xarray_match.rio.transform()

    wv_raster = sat_xarray_match.data
    wv_raster = wv_raster.squeeze()

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    cs = ax.imshow(wv_raster)

    plt.show()