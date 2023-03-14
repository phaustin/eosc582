---
jupytext:
  text_representation:
    extension: .md
    format_name: markdown
    format_version: 0.13
    jupytext_version: 1.14.5
---

# E582 project notes


## Jill:

* Title: Satellite detection of atmospheric rivers
* AR definition:  width&lt; 1000 km, length> 2000 km, precipitable water > 2 cm, east-west orientation
* Possible technique:
    * Threshold a water vapor retrieval so that it registers 1 when pixel > 2cm 0 otherwise
    * Connect threshold pixels using connected components code
    * Check shape/orientation
    * See if wind fields correlate?
    * References:  See folder
    * Links: 
      * [https://pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/](https://pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/)

      * [https://automaticaddison.com/how-to-determine-the-orientation-of-an-object-using-opencv/](https://automaticaddison.com/how-to-determine-the-orientation-of-an-object-using-opencv/)
        
* Data: GOES 17/18 (GOES West) total precipitable water
  * goes2go package
    * https://blaylockbk.github.io/goes2go/_build/html/user_guide/notebooks/field-of-view_ABI.html
  * using stac catalog
    * https://gist.github.com/scottyhq/f8af457488050d8de369da9733acb4e7
  * GOES on AWS
    * https://registry.opendata.aws/noaa-goes/
    * GOES 18 total precipitable water (moved to GOES west on Jan. 4, 2023)
      * https://www.nesdis.noaa.gov/news/earth-orbit-noaas-goes-18-now-goes-west
      * https://noaa-goes18.s3.amazonaws.com/index.html#ABI-L2-TPWC/
    * GOES 17  total precipitable water content (offline on Jan 4, 2023)
      * https://noaa-goes16.s3.amazonaws.com/index.html#ABI-L2-TPWC/


## Jamie:

* Title: surface classification for Landsat/Sentinel images
* Use some kind of machine learning technique to train a classifier, using the Corrine Land Cover dataset as ground truth
* Dataset:  HLS tiles – how many?  Minimum size?  What band combinations?
* Sanity check – k-means clustering with scikit learn in 3 dimensions?
* Corrine user guide: [https://land.copernicus.eu/user-corner/technical-library/clc-product-user-manual](https://land.copernicus.eu/user-corner/technical-library/clc-product-user-manual)  
* 10 meter global classification: https://registry.opendata.aws/io-lulc/
* Classifier example: https://ml-gis-service.com/index.php/2020/10/14/data-science-unsupervised-classification-of-satellite-images-with-k-means-algorithm/


## Sonia

* Title: ocean chlorophyll regional/temporal variability?
* Dataset: [https://registry.opendata.aws/sentinel-3/](https://registry.opendata.aws/sentinel-3/) – olci instrument for ocean color
* tutorials: 
  * https://carpentries-incubator.github.io/geospatial-python/05-access-data/
  * https://github.com/Sentinel-5P/data-on-s3/blob/master/notebooks/Sentinel3_Tutorial.ipynb
  * https://www.matecdev.com/posts/landsat-sentinel-aws-s3-python.html
* Compare sentinel and modis/aqua retrievals?
* NASA modis
  * https://oceancolor.gsfc.nasa.gov/data/download_methods/
  * https://github.com/OceanOptics/getOC
* Algorithms: 
  * Modis: https://oceancolor.gsfc.nasa.gov/atbd/
  * Sentinel: https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-3-olci/level-2/ocean-processing
  



## Nick


## Grace

* Analyze vertical structure of hurricanes and how it relates to their intensity and other meteorological variables using Cloudsat
* Cloudsat data archive: [https://cloudsat.atmos.colostate.edu/data](https://cloudsat.atmos.colostate.edu/data)
    * Extract the vertical profiles of radar reflectivity, cloud top height, and precipitation rate from the CloudSat granules using Python
    * Analyze relationship between the cloud and precipitation profiles and other meteorological variables such as sea surface temperature, wind shear, atmospheric stability, and humidity.
    * Use statistical techniques (correlation analysis, regression analysis?) to identify the most important variables that influence the hurricane intensity and track.
* [https://journals.ametsoc.org/view/journals/bams/96/4/bams-d-13-00282.1.xml](https://journals.ametsoc.org/view/journals/bams/96/4/bams-d-13-00282.1.xml)
