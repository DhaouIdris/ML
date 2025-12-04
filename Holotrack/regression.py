import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def standardize(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std


class Regression:
    def __init__(self, learning_rate=0.01, convergence= 1e-6):
        self.W = None
        self.b = None
        self.learning_rate = learning_rate
        self.convergence = convergence

    def initialize_parameters(self, n_features):
        self.W = np.random.randn(n_features)
        self.b = 0

    def forward(self, X):
        return np.dot(X, self.W) + self.b
    
    def compute_loss(self, predictions):
        m = len(predictions)
        return np.sum(np.square((self.y - predictions))/(2*m))
    
    def backward(self, predictions):
        m = len(predictions)
        self.dW = np.dot(predictions - self.y, self.X) / m
        self.db = np.sum(predictions - self.y) / m
    
    def fit(self, X, y, iterations):
        self.X = X
        self.y = y
        self.initialize_parameters(X.shape[1])
        costs = []

        for i in range(iterations):

            predictions = self.forward(X=X)
            loss = self.compute_loss(predictions)
            costs.append(loss)
            self.backward(predictions)
            self.W -= self.learning_rate * self.dW
            self.b -= self.learning_rate * self.db

            if i % 100 == 0:
                print(f'Iterations{i} with cost: {costs[i]}')

            if abs(costs[-1] - costs[-2]) < self.convergence:
                print(f'Training ended early at iteration {i}')
                break


if __name__ == "__main__":
    # Load the training and test datasets
    train_data = pd.read_csv('/kaggle/input/random-linear-regression/train.csv')
    test_data = pd.read_csv('/kaggle/input/random-linear-regression/test.csv')

    # Remove rows with missing values
    train_data = train_data.dropna()
    test_data = test_data.dropna()

    X_train = train_data['x'].values
    y_train = train_data['y'].values
    X_test = test_data['x'].values
    y_test = test_data['y'].values

    # Standardize the features
    X_train = standardize(X_train)
    X_test = standardize(X_test)

    # Train the regression model
    model = Regression(learning_rate=0.01, convergence=1e-6)
    model.fit(X_train.reshape(-1, 1), y_train, iterations=1000)
    # Make predictions on the test set
    predictions = model.forward(X_test.reshape(-1, 1))

    #Plot loss curve
    plt.plot(predictions, label='Predictions', color='blue')
    