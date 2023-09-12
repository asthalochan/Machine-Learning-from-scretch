import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KMeans:

    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []

    def fit(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape
        self._initialize_centroids()

        for _ in range(self.max_iters):
            self.clusters = self._create_clusters()
            new_centroids = self._calculate_new_centroids()

            if self._converged(new_centroids):
                break

            self.centroids = new_centroids

        return self._get_cluster_labels()

    def _initialize_centroids(self):
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

    def _create_clusters(self):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample):
        distances = [euclidean_distance(sample, point) for point in self.centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _calculate_new_centroids(self):
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(self.clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _converged(self, new_centroids):
        distances = [euclidean_distance(new_centroids[i], self.centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    def _get_cluster_labels(self):
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(self.clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

