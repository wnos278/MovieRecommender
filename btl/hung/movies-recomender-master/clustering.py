"""Implementation of k-means clustering algorithm.
These functions are designed to work with cartesian data points
"""
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from scipy import sparse
from sklearn.cluster import KMeans


def convert_to_2d_array(points):
    """
    Converts `points` to a 2-D numpy array.
    """
    points = np.array(points)
    if len(points.shape) == 1:
        points = np.expand_dims(points, -1)
    return points


def visualize_clusters(clusters):
    """
    Visualizes the first 2 dimensions of the data as a 2-D scatter plot.
    """
    plt.figure()
    for cluster in clusters:
        points = convert_to_2d_array(cluster)
        if points.shape[1] < 2:
            points = np.hstack([points, np.zeros_like(points)])
        plt.plot(points[:, 0], points[:, 1], 'o')
    plt.show()


def SSE(points):
    """
    Calculates the sum of squared errors for the given list of data points.
    """
    points = convert_to_2d_array(points)
    centroid = np.mean(points, 0)
    errors = np.linalg.norm(points - centroid, ord=2, axis=1)
    return np.sum(errors)


def bisecting_kmeans(points: sparse.csr_matrix, k=2):
    user_to_cluster = {}
    cluster_to_user = {}
    clusters = {0: points}
    for i in range(points.get_shape()[0]):
        user_to_cluster[i] = (0, i)
        cluster_to_user[(0, i)] = i
    while len(clusters) < k:
        biggest_cluster_key = list(clusters.keys())[0]
        for i in clusters:
            if clusters[i].get_shape()[0] > clusters[biggest_cluster_key].get_shape()[0]:
                biggest_cluster_key = i
        biggest_cluster = clusters[biggest_cluster_key]
        # remove cluster from dict
        del clusters[biggest_cluster_key]
        kmeans = KMeans(n_clusters=2, max_iter=100).fit(biggest_cluster)

        key1 = random.randint(1, 1000000)
        key2 = random.randint(1, 1000000)
        id1 = 0
        id2 = 0
        clusters_data1 = [[], [], []]
        clusters_data2 = [[], [], []]

        for i in range(len(kmeans.labels_)):
            row, col = biggest_cluster.getrow(i).nonzero()
            data = np.array(biggest_cluster.getrow(i)[row, col]).flatten()
            if kmeans.labels_[i] == 0:
                row = np.ones(len(col), dtype=int) * id1
                for j in range(len(col)):
                    clusters_data1[0].append(row[j])
                    clusters_data1[1].append(col[j])
                    clusters_data1[2].append(data[j])
                    # update mapping
                user_id = cluster_to_user[(key1, id1)] = cluster_to_user[(biggest_cluster_key, i)]
                user_to_cluster[user_id] = (key1, id1)
                id1 += 1

            else:
                row = np.ones(len(col), dtype=int) * id2
                for j in range(len(col)):
                    clusters_data2[0].append(row[j])
                    clusters_data2[1].append(col[j])
                    clusters_data2[2].append(data[j])
                    # update mapping
                user_id = cluster_to_user[(key2, id2)] = cluster_to_user[(biggest_cluster_key, i)]
                user_to_cluster[user_id] = (key2, 2)
                id2 += 1
            del cluster_to_user[biggest_cluster_key, i]

        clusters[key1] = (sparse.csr_matrix((clusters_data1[2], (clusters_data1[0], clusters_data1[1])),
                                            (id1, biggest_cluster.get_shape()[1])))
        clusters[key2] = (sparse.csr_matrix((clusters_data2[2], (clusters_data2[0], clusters_data2[1])),
                                            (id2, biggest_cluster.get_shape()[1])))

    return clusters, user_to_cluster