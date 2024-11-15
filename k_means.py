import numpy as np
import matplotlib.pyplot as plt

def initialize_centroids(X, k):
    """Randomly choose k points as initial centroids."""
    np.random.seed(0)
    random_indices = np.random.choice(len(X), k, replace=False)
    return X[random_indices]

def calculate_distance(point, centroid):
    """Calculate the Euclidean distance between a point and a centroid."""
    return np.sqrt(np.sum((point - centroid) ** 2))

def assign_clusters(X, centroids):
    """Assign each point to the nearest centroid."""
    labels = []
    for point in X:
        distances = [calculate_distance(point, centroid) for centroid in centroids]
        labels.append(np.argmin(distances))
    return np.array(labels)

def update_centroids(X, labels, k):
    """Calculate new centroids as the mean of points in each cluster."""
    print('labels',labels)
    new_centroids = []
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            new_centroids.append(cluster_points.mean(axis=0))
        else:
            # If a cluster has no points, keep the centroid unchanged
            new_centroids.append(centroids[i])
    return np.array(new_centroids)

def kmeans(X, k, max_iters=1):
    """Perform K-means clustering."""
    centroids = initialize_centroids(X, k)
    for i in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids, labels

# Example usage with random data
X = np.random.rand(100, 2)  # Generating random data for demonstration
k = 3  # Number of clusters
centroids, labels = kmeans(X, k)

# Plotting results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
plt.title("K-means Clustering")
plt.show()
