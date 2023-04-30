# Plot the information from the cluster_data.json file. See how many good clusters (80% correct and not built-up)
# there are for different sizes, dates, number of clusters, and band combinations.

import json
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

def plot_saved_data():
    f = open('cluster_data.json')
    data = json.load(f)

    size_labels = np.array([0.01, 0.05, 0.10, 0.25])
    # jan, apr, july, oct
    date_labels = np.array(["2019-01-01/2019-01-31", "2019-04-01/2019-04-30", "2019-07-01/2019-07-31", "2019-10-01/2019-10-31"])
    cluster_labels = np.arange(3, 10)
    band_labels = np.array(["all", "false_colour"])

    size_data = np.zeros((5, len(size_labels)))
    date_data = np.zeros((5, len(date_labels)))
    cluster_data = np.zeros((5, len(cluster_labels)))
    band_data = np.zeros((5, len(band_labels)))

    x = ["size", "date", "n_clusters", "bands"]
    x_labels = [size_labels, date_labels, cluster_labels, band_labels]
    x_data = [size_data, date_data, cluster_data, band_data]

    for i in range(len(x)):
        plt.subplot(2, 2, i+1)
        plt.xlabel(x[i])
        for data_dict in data['data']:
            comp_dict = data_dict["composition_dict"]
            good_clusters = 0
            for cluster, cluster_dict in comp_dict.items():
                for material, percent in cluster_dict.items():
                    if percent > 0.8 and material != "built-up":
                        good_clusters += 1
            #plt.scatter(data_dict[x[i]], good_clusters)
            ind = np.where(x_labels[i] == data_dict[x[i]])
            x_data[i][good_clusters, ind] += 1
    
    #plt.show()
    for i in range(len(x_data)):
        fig=go.Figure(data=go.Heatmap(
            x=x_labels[i],
            z=x_data[i],
            type = 'heatmap',
            colorscale = 'Viridis'
            )
        )
        fig.update_layout(title=x[i])
        fig.show()

        #plt.subplot(2, 2, i+1)
        #plt.imshow(x_data[i])
        #plt.colorbar()
    
    #plt.tight_layout()
    #plt.show()

    f.close()