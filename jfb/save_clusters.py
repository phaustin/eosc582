# Try different sizes, bands, dates, number of clusters, and save the results to cluster_data.json

from compare_clusters import plot_side_by_side, save_cluster
from cluster_sat_data import fetch_data
import rioxarray
import numpy as np
import pylab
import plotly.express as px
from copern_data import plot_classified_data, get_copern_xarray, get_feature_dict
from plotly.subplots import make_subplots

# test different locations
"""
- band numbers and combinations
- test different algorithms
    - knn: diff n_clusters
    - dbscan: epsilon
- diff cluster amounts
- diff sizes
- over the course of a year (try four seasons?), should be 2019!
"""

def get_combinations():
    band_combinations = ["all", "false_colour"]
    #algorithm_combinations = ["knn", "dbscan"] #agglomerative clustering
    algorithm_combinations = ["knn"]
    cluster_combinations = np.arange(3, 10)
    size_combinations = [0.01, 0.05, 0.10, 0.25]
    # jan, apr, july, oct
    date_combinations = ["2019-01-01/2019-01-31", "2019-04-01/2019-04-30", "2019-07-01/2019-07-31", "2019-10-01/2019-10-31"]

    combinations = []
    for bands in band_combinations:
        for algorithm in algorithm_combinations:
            for size in size_combinations:
                for date in date_combinations:
                    if algorithm == "knn":
                        for n_clusters in cluster_combinations:
                            combinations.append(
                                {
                                    "bands": bands,
                                    "algorithm": algorithm,
                                    "n_clusters": n_clusters,
                                    "size": size,
                                    "date": date,
                                }
                                )
                    elif algorithm == "dbscan":
                        combinations.append(
                                {
                                    "bands": bands,
                                    "algorithm": algorithm,
                                    "n_clusters": None,
                                    "size": size,
                                    "date": date,
                                }
                                )
    
    return combinations

combinations = get_combinations()

# 111
for i in range(111, len(combinations)):
    comb_dict = combinations[i]
    #save_cluster(lat=49.2827, lon=-123.120, date="2015-06-01/2015-06-30", size=0.10, algorithm="knn", n_clusters=3, bands="all")
    n_clusters = comb_dict["n_clusters"]
    if n_clusters is not None:
        n_clusters = n_clusters.item()

    save_cluster(lat=49.2827, lon=-123.120, date=comb_dict["date"], size=comb_dict["size"], algorithm=comb_dict["algorithm"], n_clusters=n_clusters, bands=comb_dict["bands"])
    print(i)
