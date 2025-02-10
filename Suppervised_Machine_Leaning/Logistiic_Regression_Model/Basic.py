import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function
def compute_cost(X, y, weights):
    m = len(y)
    h = sigmoid(np.dot(X, weights))
    cost = (-1/m) * np.sum(y*np.log(h) + (1-y)*np.log(1-h))
    return cost

# Gradient Descent
def gradient_descent(X, y, weights, alpha, epochs):
    m = len(y)
    cost_history = []
    
    for _ in range(epochs):
        h = sigmoid(np.dot(X, weights))
        gradient = (1/m) * np.dot(X.T, (h - y))
        weights -= alpha * gradient
        cost_history.append(compute_cost(X, y, weights))
    
    return weights, cost_history

# Initialize weights and add bias term
X_train_bias = np.c_[np.ones((X_train.shape[0], 1)), X_train]
weights = np.zeros(X_train_bias.shape[1])

# Train model
alpha = 0.01
epochs = 1000
weights, cost_history = gradient_descent(X_train_bias, y_train, weights, alpha, epochs)

# Plot cost function convergence
plt.plot(range(epochs), cost_history)
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.title("Cost Function Convergence")
plt.show()
