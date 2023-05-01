from compare_clusters import plot_side_by_side
from compare_saved_clusters import plot_saved_data

# This is the main file for running clustering. CNN work can be run by the cnn_notebook.ipynb file.

# Example figures from running this code are under the figures folder.

# Plot clustered data beside classified data, along with pie plots showing cluster composition.
plot_side_by_side(lat=49.2827, lon=-123.120, date="2019-07-01/2019-07-31", size=0.25, algorithm="knn", n_clusters=3, bands="all")

# Plot data collected in cluster_data.json. Histograms showing the effects of different clustering
# input variables
plot_saved_data()

# There's not much to get out of these plots, especially because they are all for Vancouver
# and they don't account for the size of the clusters. It looks like false colour maybe
# performs a little better than using all the bands as input. Out of the months I tried, 
# July and January perform best, and April performs worst.

# Comments on CNN figures:
# I only ended up running the CNN on a small amount of data. The CNN figures in the figures
# folder show the results from running the CNN notebook. The majority of the data is only 3
# classes, and the CNN usually only has categories for a few of classes. There's not much 
# more to say without running on a larger data set.
