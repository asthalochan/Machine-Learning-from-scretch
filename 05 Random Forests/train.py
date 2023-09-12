from random import random
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from RandomForest import RandomForest

# Load the Iris dataset
data = datasets.load_iris()
X = data.data
y = data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

# Define the accuracy function
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

# Create and fit the Random Forest classifier
clf = RandomForest(n_trees=20)
clf.fit(X_train, y_train)

# Make predictions and calculate accuracy
predictions = clf.predict(X_test)
acc = accuracy(y_test, predictions)

print(f"Accuracy: {acc:.2f}")
