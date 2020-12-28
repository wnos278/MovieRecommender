import time

import pandas as pd
import numpy as np
from CF import CF
from clustering import bisecting_kmeans


K_CLUSTERS = 4

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

print('clustering...')
clusters, user_mapping = bisecting_kmeans(rs.Ybar.transpose().tocsr(), k=K_CLUSTERS)
print('clustering done')
cf_clusters = {}
print(clusters)
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

print('evaluate test set')
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

print('Throughput = ', n_tests / total_time)
RMSE = np.sqrt(SE/n_tests)
MAE = AE / n_tests
print('RMSE =', RMSE)
print('MAE =', MAE)

