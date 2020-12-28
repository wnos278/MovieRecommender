import time

import pandas as pd
import numpy as np
from CF import CF
from clustering import bisecting_kmeans



ratings_base = pd.read_csv('ml-1m/train.csv')
ratings_test = pd.read_csv('ml-1m/test.csv')


rate_train = ratings_base.values
rate_test = ratings_test.values


# indices start from 0
rate_train[:, :2] -= 1
rate_test[:, :2] -= 1

rs = CF(rate_train, None, k=30, uuCF=1)

rs.normalize_Y()
rs.similarity()


K_CLUSTERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
throughput_clustering = []
RMSE_clustering = []
MAE_clustering = []


for k in K_CLUSTERS:
    print('k = ', k)
    clusters, user_mapping = bisecting_kmeans(rs.Ybar.transpose().tocsr(), k=1)
    cf_clusters = {}
    for i in clusters:
        row, col = clusters[i].nonzero()
        data = np.array(clusters[i][row, col]).flatten()
        pd_data = pd.DataFrame()
        pd_data['userId'] = row
        pd_data['movieId'] = col
        pd_data['rating'] = data
        cf_clusters[i] = CF(pd_data.values, clusters[i].transpose())
        cf_clusters[i].similarity()

    n_tests = rate_test.shape[0]
    SE = 0 # squared error
    AE = 0 # absolute error

    total_time = 0
    for n in range(n_tests):
        cf_cluster = cf_clusters[user_mapping[rate_test[n, 0]][0]]
        user_cluster_id = user_mapping[rate_test[n, 0]][1]
        start = time.time()
        pred = cf_cluster.pred(user_cluster_id, rate_test[n, 1], normalized=1) + rs.mu[rate_test[n, 0]]
        total_time += time.time() - start
        # print(pred, rate_test[n, 2])
        SE += (pred - rate_test[n, 2])**2
        AE += np.abs(pred - rate_test[n, 2])

    throughput = n_tests / total_time
    RMSE = np.sqrt(SE/n_tests)
    MAE = AE / n_tests
    throughput_clustering.append(throughput)
    RMSE_clustering.append((RMSE))
    MAE_clustering.append((MAE))
    print('throughput:', throughput)
    print('RMSE:', RMSE)
    print('MAE:', MAE)


print('throughput', throughput_clustering)
print('RMSE: ', RMSE_clustering)
print('MAE: ', MAE_clustering)

# throughput = [1997.510760723532, 3697.0485165085242, 4116.416717151468, 4402.071188544323, 4385.325111397377, 4424.299101631065, 4481.648049825185, 4509.947720387431, 4391.749176363464, 4691.14934384838]
# RMSE =  [0.9562829060081101, 0.9762161787341792, 1.000261226515741, 1.0281929073916665, 1.032699316864204, 1.0271989237241848, 1.0285947637067343, 1.0283681137179892, 1.0459787470963162, 1.0714727720582307]
# MAE =  [0.7615892008659068, 0.7658893984034355, 0.7967548339123608, 0.8162719815308184, 0.8189308769015148, 0.8145725425971369, 0.8151898002965804, 0.8146410301230511, 0.8284236524393833, 0.8516709890483503]


# throughput = [1973.4215923873198, 1957.9326819216553, 1944.122073698611, 1847.0446060246452, 1803.8297293801008, 1851.8745764931202, 1937.787651686609, 1944.7521701898693, 1945.3021969258962, 1775.972806350836]
# RMSE =  [0.95503864527689, 0.9561851212238764, 0.9562829060081101, 0.9562829060081101, 0.9562829060081101, 0.9562829060081101, 0.9562829060081101, 0.9562829060081101, 0.9562829060081101, 0.9562829060081101]
# MAE =  [0.7579700385007065, 0.7613461080543161, 0.7615892008659069, 0.7615892008659069, 0.7615892008659069, 0.7615892008659069, 0.7615892008659069, 0.7615892008659069, 0.7615892008659069, 0.7615892008659069]
