# Save the satellite data as a .npy that will be used by the cnn notebook.

from shapely.geometry import Point
from pystac_client import Client
import numpy as np
import rioxarray

import os
os.environ["GDAL_HTTP_COOKIEFILE"] = "./cookies.txt"
os.environ["GDAL_HTTP_COOKIEJAR"] = "./cookies.txt"


def get_copern_xarray(lat, lon, size):
    lat_range = np.array([size, -size]) + lat
    lon_range = np.array([-size, size]) + lon

    copern_tif  = "./data/copernicus/PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif"
    copern_xarray = rioxarray.open_rasterio(copern_tif)
    copern_xarray = copern_xarray.sel(y=slice(lat_range[0], lat_range[1]), x=slice(lon_range[0], lon_range[1]))

    return copern_xarray


def fetch_sat_data(lat, lon, date):
    point = Point(lon, lat)

    # connect to the STAC endpoint
    cmr_api_url = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD"
    client = Client.open(cmr_api_url)

    search = client.search(
        collections=["HLSL30.v2.0"],
        intersects=point,
        datetime= date
    )

    items = search.get_all_items()

    min_cloud_cover = 100
    for index, the_scene in enumerate(items):
        cloud_cover = the_scene.properties["eo:cloud_cover"]
        if cloud_cover < min_cloud_cover:
            scene = items[index]
            min_cloud_cover = cloud_cover

    return scene

def get_raster_stack(lat, lon, date, colors, copern_xarray):
    scene = fetch_sat_data(lat=lat, lon=lon, date=date)

    if colors == "all":
        colors = [idx for idx in scene.assets.keys() if idx[0]=="B"]
    elif colors == "false_colour":
        colors = ["B06", "B05", "B04"]
    fmask = rioxarray.open_rasterio(scene.assets["Fmask"].href, masked=True)
    fmask_match = (fmask).rio.reproject_match(copern_xarray)

    band_list = []
    for color in colors:
        print(color)
        band = rioxarray.open_rasterio(scene.assets[color].href, masked=True)
        #https://corteva.github.io/rioxarray/stable/examples/reproject_match.html
        band_match = band.rio.reproject_match(copern_xarray)

        raster = (band_match*fmask_match).squeeze()*band_match.scale_factor

        band_list.append(np.nan_to_num(raster))

    bands_stack = np.stack(band_list, axis=0)

    return bands_stack


lat = 49.2827
lon = -123.120
size = 0.25
#size = 0.05
band_combinations = ["all", "false_colour"]
colors = band_combinations[0]
#colors = band_combinations[1]
# jan, apr, july, oct
date_combinations = ["2019-01-01/2019-01-31", "2019-04-01/2019-04-30", "2019-07-01/2019-07-31", "2019-10-01/2019-10-31"]
date = date_combinations[2]

# Ground Truth
y_data = np.squeeze(get_copern_xarray(lat, lon, size))

# Data
arr_st = get_raster_stack(lat, lon, date, colors, y_data)

np.save("arr_st.npy", arr_st)
