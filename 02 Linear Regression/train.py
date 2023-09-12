from LinearRegression import LinearRegression  
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

# Generate synthetic data
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Plot the data
fig = plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], y, color="b", marker="o", s=30)
plt.show()

# Create and train the LinearRegression model
reg = LinearRegression(learning_rate=0.01, n_iterations=1000)  # Use the new parameter names
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)

# Define a function to calculate Mean Squared Error (MSE)
def mse(y_test, predictions):
    return np.mean((y_test - predictions) ** 2)

mse_value = mse(y_test, predictions)
print("Mean Squared Error:", mse_value)

# Plot the training and test data along with the regression line
y_pred_line = reg.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')
plt.show()
