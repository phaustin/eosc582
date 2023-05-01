# Fetch and cluster satellite data.
#https://ml-gis-service.com/index.php/2020/10/14/data-science-unsupervised-classification-of-satellite-images-with-k-means-algorithm/

import os
import numpy as np
import rasterio as rio
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from skimage.exposure import equalize_adapthist
import matplotlib.pyplot as plt
import rioxarray
from shapely.geometry import Point
from pystac_client import Client
import plotly.express as px
from sklearn.cluster import DBSCAN

import os
os.environ["GDAL_HTTP_COOKIEFILE"] = "./cookies.txt"
os.environ["GDAL_HTTP_COOKIEJAR"] = "./cookies.txt"


def fetch_data(lat, lon, date):
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

class ClusteredBands:
    
    def __init__(self, scene):
        self.scene = scene
        self.model_input = None
        self.width = 0
        self.height = 0
        self.depth = 0
        self.n_clusters = None
        self.model = None
        self.predicted_raster = None
        self.s_score = None
        self.inertia_score = None
    
    def set_raster_stack(self, colors, copern_xarray):
        fmask = rioxarray.open_rasterio(self.scene.assets["Fmask"].href, masked=True)
        fmask_match = (fmask).rio.reproject_match(copern_xarray)

        band_list = []
        for color in colors:
            band = rioxarray.open_rasterio(self.scene.assets[color].href, masked=True)
            #https://corteva.github.io/rioxarray/stable/examples/reproject_match.html
            band_match = band.rio.reproject_match(copern_xarray)

            raster = (band_match*fmask_match).squeeze()*band_match.scale_factor

            band_list.append(np.nan_to_num(raster))
        bands_stack = np.dstack(band_list)
        
        # Prepare model input from bands stack
        self.width, self.height, self.depth = bands_stack.shape
        self.model_input = bands_stack.reshape(self.width * self.height, self.depth)
            
            
    def build_models(self, algorithm, n_clusters):
        self.n_clusters = n_clusters
        
        if algorithm == "knn":
            kmeans = KMeans(n_clusters=n_clusters)
            y_pred = kmeans.fit_predict(self.model_input)            
            self.model = kmeans
            self.s_score = self._calc_s_score(y_pred)
            self.inertia_val = kmeans.inertia_
        elif algorithm == "dbscan":
            dbscan = DBSCAN()
            dbscan.fit_predict(self.model_input)
            y_pred = dbscan.labels_
            self.model = dbscan
        
        self.predicted_raster = np.reshape(y_pred, (self.width, self.height))

    def _calc_s_score(self, labels):
        s_score = silhouette_score(self.model_input, labels, sample_size=1000)
        return s_score
        
    def show_clustered(self):
        title = 'Number of clusters: ' + str(self.n_clusters)
        image = self.predicted_raster
        fig = px.imshow(image)
        return fig, image


def run_clustering(lat, lon, date, algorithm, n_clusters, bands, xarray):
    scene = fetch_data(lat=lat, lon=lon, date=date)

    if bands == "all":
        bands = [idx for idx in scene.assets.keys() if idx[0]=="B"]
    elif bands == "false_colour":
        bands = ["B06", "B05", "B04"]

    clustered_models = ClusteredBands(scene)
    clustered_models.set_raster_stack(bands, xarray)

    clustered_models.build_models(algorithm, n_clusters)
    cluster_figure, image = clustered_models.show_clustered()

    return cluster_figure, image