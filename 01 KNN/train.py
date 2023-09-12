from KNN import KNN
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
  

# Define a colormap for plotting
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Load the Iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Plot the Iris dataset
plt.figure()
plt.scatter(X[:, 2], X[:, 3], c=y, cmap=cmap, edgecolor='k', s=20)
plt.show()

# Create and fit a KNN classifier with k=5
clf = KNN(k=5)
clf.fit(X_train, y_train)

# Make predictions on the test data
predictions = clf.predict(X_test)

# Print the predicted labels
print(predictions)

# Calculate and print the accuracy of the classifier
acc = np.sum(predictions == y_test) / len(y_test)
print(f"Accuracy: {acc * 100:.2f}%")
