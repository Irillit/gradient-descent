import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


class Network:
    parameters = {}

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-1 * z))

    def tanh(self, z):
        num = np.exp(z) - np.exp(-1 * z)
        den = np.exp(z) + np.exp(-1 * z)
        return num / den

    def relu(self, z):
        return z * (z > 0)

    def init_weights(self):
        self.parameters["W1"] = np.random.randn(7, 4) * 0.01
        self.parameters["b1"] = np.zeros([7, 1])

        self.parameters["W2"] = np.random.randn(1, 7) * 0.01
        self.parameters["b2"] = np.zeros([1,1])

    def forward_propagation(self, X_train):
        W1 = self.parameters["W1"]
        b1 = self.parameters["b1"]
        W2 = self.parameters["W2"]
        b2 = self.parameters["b2"]

        Z1 = np.dot(W1, X_train) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = self.sigmoid(Z2)

        return A1, A2

    def cost_function(self, A2, y_train):
        m = y_train.shape[1]
        logprobs = np.multiply(np.log(A2), y_train) + np.multiply(np.log(1 - A2), (1 - y_train))
        cost = -1 / m * np.sum(logprobs)
        return np.squeeze(cost)

    def backward_propagation(self, A1, A2, X_train, y_train):
        m = X_train.shape[1]
        dZ2 = A2 - y_train

        W1 = self.parameters["W1"]
        W2 = self.parameters["W2"]
        dw2 = (1./m) * np.dot(dZ2, A1.T)
        db2 = (1./m) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
        dw1 = (1./m) * np.dot(dZ1, X_train.T)
        db1 = (1./m) * np.sum(dZ1, axis=1, keepdims=True)

        self.parameters["W1"] = W1 - dw1 * self.learning_rate
        self.parameters["W2"] = W2 - dw2 * self.learning_rate
        self.parameters["b1"] = self.parameters["b1"] - db1 * self.learning_rate
        self.parameters["b2"] = self.parameters["b2"] - db2 * self.learning_rate


if __name__ == "__main__":
    X, Y = make_classification(n_features=4, n_redundant=0, n_informative=4, n_clusters_per_class=1)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    network = Network(0.7)
    network.init_weights()
    X_train = np.reshape(X_train, (4, X_train.shape[0]))
    y_train = np.reshape(y_train, (1, y_train.shape[0]))

    for i in range(1000):
        A1, A2 = network.forward_propagation(X_train)
        cost = network.cost_function(A2, y_train)
        if i % 10 == 0:
            print(f"#{i} cost {cost}")
        network.backward_propagation(A1, A2, X_train, y_train)
    
    print(network.parameters["W1"])

