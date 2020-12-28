import matplotlib.pyplot as plt
import numpy as np

throughput_clutering = [1997.510760723532, 3697.0485165085242, 4116.416717151468, 4402.071188544323, 4385.325111397377,
                        4424.299101631065, 4481.648049825185, 4509.947720387431, 4391.749176363464, 4691.14934384838]
RMSE_clustering = [0.9562829060081101, 0.9762161787341792, 1.000261226515741, 1.0281929073916665, 1.032699316864204,
                   1.0271989237241848, 1.0285947637067343, 1.0283681137179892, 1.0459787470963162, 1.0714727720582307]
MAE_clustering = [0.7615892008659068, 0.7658893984034355, 0.7967548339123608, 0.8162719815308184, 0.8189308769015148,
                  0.8145725425971369, 0.8151898002965804, 0.8146410301230511, 0.8284236524393833, 0.8516709890483503]

throughput = [1973.4215923873198, 1957.9326819216553, 1944.122073698611, 1847.0446060246452, 1803.8297293801008,
              1851.8745764931202, 1937.787651686609, 1944.7521701898693, 1945.3021969258962, 1775.972806350836]
RMSE = [0.95503864527689, 0.9561851212238764, 0.9562829060081101, 0.9562829060081101, 0.9562829060081101,
        0.9562829060081101, 0.9562829060081101, 0.9562829060081101, 0.9562829060081101, 0.9562829060081101]
MAE = [0.7579700385007065, 0.7613461080543161, 0.7615892008659069, 0.7615892008659069, 0.7615892008659069,
       0.7615892008659069, 0.7615892008659069, 0.7615892008659069, 0.7615892008659069, 0.7615892008659069]
k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


# Note that even in the OO-style, we use `.pyplot.figure` to create the figure.
fig1, ax1 = plt.subplots()  # Create a figure and an axes.
fig2, ax2 = plt.subplots()  # Create a figure and an axes.
fig3, ax3 = plt.subplots()  # Create a figure and an axes.

ax1.plot(k, MAE, 's-', label='CF')
ax1.plot(k, MAE_clustering, 's-', label='Clustering')
ax1.set_xlabel('No. of clusters')  # Add an x-label to the axes.
ax1.set_ylabel('MAE')  # Add a y-label to the axes.
ax1.set_title("Prediction quality: Clustering vs. CF")  # Add a title to the axes.
ax1.legend()  # Add a legend.

ax2.plot(k, RMSE, 's-', label='CF')
ax2.plot(k, RMSE_clustering, 's-', label='Clustering')
ax2.set_xlabel('No. of clusters')  # Add an x-label to the axes.
ax2.set_ylabel('RMSE')  # Add a y-label to the axes.
ax2.set_title("Prediction quality: Clustering vs. CF")  # Add a title to the axes.
ax2.legend()  # Add a legend.

ax3.plot(k, throughput, 's-', label='CF')
ax3.plot(k, throughput_clutering, 's-', label='Clustering')
ax3.set_xlabel('No. of clusters')  # Add an x-label to the axes.
ax3.set_ylabel('Throughput (Recs./sec')  # Add a y-label to the axes.
ax3.set_title("Throughput: Clustering vs. CF")  # Add a title to the axes.
ax3.legend()  # Add a legend.

plt.show()
