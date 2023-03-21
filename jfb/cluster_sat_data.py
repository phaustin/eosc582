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

    scene = items[1]

    return scene


def show_rgb(scenes_list, red="B04", green="B03", blue="B02"):
    stack = []
    
    colors = [red, green, blue]
    #colors = ['B0' + str(x) for x in colors]
    for scene in scenes_list:
        for color in colors:
            june14_band = rioxarray.open_rasterio(scene.assets[color].href, masked=True)
            june14_raster = june14_band.squeeze()
            june14_raster = june14_raster*june14_band.scale_factor
            stack.append(june14_raster)
    
    stack = np.dstack(stack)
    
    for i in range(0, 3):
        stack[:, :, i] = equalize_adapthist(stack[:, :, i], clip_limit=0.025)
        
    fig = plt.figure(figsize=(15,15))
    plt.axis('off')
    
    #print(stack)
    
    plt.imshow(stack)
    return fig
#show_rgb(complete_dataset)

class ClusteredBands:
    
    def __init__(self, scene):
        self.rasters = [scene]
        self.model_input = None
        self.width = 0
        self.height = 0
        self.depth = 0
        self.no_of_ranges = None
        self.models = None
        self.predicted_rasters = None
        self.s_scores = []
        self.inertia_scores = []
    
    def set_raster_stack(self, colors):
        band_list = []
        for image in self.rasters:
            for color in colors:
                june14_band = rioxarray.open_rasterio(image.assets[color].href, masked=True)
                june14_raster = june14_band.squeeze()
                june14_raster = june14_raster*june14_band.scale_factor

                band = june14_raster
                band = np.nan_to_num(band)
                band_list.append(band)
        bands_stack = np.dstack(band_list)
        
        # Prepare model input from bands stack
        self.width, self.height, self.depth = bands_stack.shape
        self.model_input = bands_stack.reshape(self.width * self.height, self.depth)
            
            
    def build_models(self, no_of_clusters_range):
        self.no_of_ranges = no_of_clusters_range
        models = []
        predicted = []
        inertia_vals = []
        s_scores = []
        for n_clust in no_of_clusters_range:
            kmeans = KMeans(n_clusters=n_clust)
            y_pred = kmeans.fit_predict(self.model_input)
            
            # Append model
            models.append(kmeans)
            
            # Calculate metrics
            s_scores.append(self._calc_s_score(y_pred))
            inertia_vals.append(kmeans.inertia_)
            
            # Append output image (classified)
            quantized_raster = np.reshape(y_pred, (self.width, self.height))
            predicted.append(quantized_raster)
            
        # Update class parameters
        self.models = models
        self.predicted_rasters = predicted
        self.s_scores = s_scores
        self.inertia_scores = inertia_vals
        
    def _calc_s_score(self, labels):
        s_score = silhouette_score(self.model_input, labels, sample_size=1000)
        return s_score
        
    def show_clustered(self):
        for idx, no_of_clust in enumerate(self.no_of_ranges):
            title = 'Number of clusters: ' + str(no_of_clust)
            image = self.predicted_rasters[idx]
            plt.figure(figsize = (15,15))
            plt.axis('off')
            plt.title(title)
            plt.imshow(image, cmap='Accent')
            plt.colorbar()
            plt.show()
            
    def show_inertia(self):
        plt.figure(figsize = (10,10))
        plt.title('Inertia of the models')
        plt.plot(self.no_of_ranges, self.inertia_scores)
        plt.show()
        
    def show_silhouette_scores(self):
        plt.figure(figsize = (10,10))
        plt.title('Silhouette scores')
        plt.plot(self.no_of_ranges, self.s_scores)
        plt.show


def run_clustering(n_clusters_range):
    scene = fetch_data(lat=49.2827, lon=-123.120, date="2015-06-01/2015-06-30")

    bands = [idx for idx in scene.assets.keys() if idx[0]=="B"] #all bands
    #bands = ["B07", "B06", "B04"]

    clustered_models = ClusteredBands(scene)
    clustered_models.set_raster_stack(bands)

    clustered_models.build_models(n_clusters_range)
    cluster_figures = clustered_models.show_clustered(ax)
    #clustered_models.show_inertia()
    #clustered_models.show_silhouette_scores()

    return cluster_figures