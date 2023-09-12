import matplotlib.pyplot as plt  
from sklearn.datasets import make_blobs
from kmeans import KMeans  

# Generate synthetic data
X, y = make_blobs(
    n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40
)

kmeans = KMeans(K=2)
labels = kmeans.fit(X)

# Plot the clusters and centroids
fig, ax = plt.subplots(figsize=(12, 8))
for i, index in enumerate(kmeans.clusters):
    point = X[index].T
    ax.scatter(*point)

for point in kmeans.centroids:
    ax.scatter(*point, marker="x", color="black", linewidth=2)

plt.show()
