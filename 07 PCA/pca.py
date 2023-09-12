import numpy as np

class PCA:

    def __init__(self, n_components):
        """
        Initialize the PCA model with a specified number of components.

        Parameters:
        n_components (int): Number of principal components to keep.
        """
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        """
        Fit the PCA model to the input data.

        Parameters:
        X (numpy.ndarray): Input data with shape (n_samples, n_features).
        """
        # Mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # Compute the covariance matrix (assuming samples as columns)
        cov = np.cov(X.T)

        # Compute eigenvectors and eigenvalues
        eigenvectors, eigenvalues = np.linalg.eig(cov)

        # Transpose eigenvectors for easier calculations
        eigenvectors = eigenvectors.T

        # Sort eigenvectors by descending eigenvalues
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        self.components = eigenvectors[:self.n_components]

    def transform(self, X):
        """
        Transform the input data using the fitted PCA model.

        Parameters:
        X (numpy.ndarray): Input data with shape (n_samples, n_features).

        Returns:
        numpy.ndarray: Transformed data with reduced dimensions (n_samples, n_components).
        """
        # Project data onto the principal components
        X = X - self.mean
        return np.dot(X, self.components.T)


