import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    """Compute the Euclidean distance between two points."""
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        """Fit the KNN model with training data."""
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """Predict labels for a set of test data points."""
        predictions = [self._predict(x) for x in X_test]
        return predictions

    def _predict(self, x):
        """Predict the label for a single data point."""
        # Compute distances between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # Get indices of k-nearest examples
        k_indices = np.argsort(distances)[:self.k]

        # Get the labels of the k-nearest examples
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Find the most common class label among the k-nearest neighbors
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
