import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from KNN.utils import get_data, get_random_perceptron_data, get_XOR
from datetime import datetime

class Perceptron:
    def __init__(self):
        pass
    
    def fit(self, X, y, learning_rate=1.0, epochs=1000):
        D = X.shape[1]
        self.w = np.random.randn(D)
        self.b = 0

        N = len(y)
        costs = []
        for epoch in range(epochs):
            y_hat = self.predict(X)
            incorrect = np.nonzero(y != y_hat)[0]
            if len(incorrect) == 0:
                print("Training complete!")
                break
        
            # choose a random incorrect sample
            random_sample = np.random.choice(incorrect)
            self.w += learning_rate * y[random_sample] * X[random_sample]
            self.b += learning_rate * y[random_sample]

            cost = len(incorrect) / float(N)
            costs.append(cost)

        print("Final w: {} | Final b: {} | epochs: {}/{}".format(self.w, self.b, epoch+1, epochs))
        plt.plot(costs)
        plt.show()

    def predict(self, X):
        return np.sign(X.dot(self.w) + self.b)

    def score(self, X, y):
        prediction = self.predict(X)
        return np.mean(prediction==y)

if __name__ == '__main__':
    print("Random Perceptron Data")
    print("-----------------------")
    X, y = get_random_perceptron_data()
    plt.figure()
    plt.scatter(X[:,0], X[:,1], c=y, s=100, alpha=0.5)
    plt.show()
    N_train = len(y) // 4 * 3
    X_train, y_train = X[:N_train], y[:N_train]
    X_test, y_test = X[N_train:], y[N_train:]
    
    model = Perceptron()
    t0 = datetime.now()
    model.fit(X_train, y_train)
    print("Training time:", (datetime.now() - t0))

    t0 = datetime.now()
    print("Train accuracy:", model.score(X_train, y_train))
    print("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(y_train))

    t0 = datetime.now()
    print("Test accuracy:", model.score(X_test, y_test))
    print("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(y_test))


    print("MNIST Data")
    print("-----------------------")
    X, y = get_data()
    idx = np.logical_or(y==0, y==1)
    X = X[idx]
    y = y[idx]
    y[y==0] = -1

    N_train = len(y) // 4 * 3
    X_train, y_train = X[:N_train], y[:N_train]
    X_test, y_test = X[N_train:], y[N_train:]
    
    model = Perceptron()
    t0 = datetime.now()
    model.fit(X_train, y_train)
    print("Training time:", (datetime.now() - t0))

    t0 = datetime.now()
    print("Train accuracy:", model.score(X_train, y_train))
    print("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(y_train))

    t0 = datetime.now()
    print("Test accuracy:", model.score(X_test, y_test))
    print("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(y_test))


    print("XOR Data")
    print("-----------------------")
    X, y = get_XOR()
    y[y==0] = -1
    
    N_train = len(y) // 4 * 3
    X_train, y_train = X[:N_train], y[:N_train]
    X_test, y_test = X[N_train:], y[N_train:]
    
    model = Perceptron()
    t0 = datetime.now()
    model.fit(X_train, y_train)
    print("Training time:", (datetime.now() - t0))

    t0 = datetime.now()
    print("Train accuracy:", model.score(X_train, y_train))
    print("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(y_train))

    t0 = datetime.now()
    print("Test accuracy:", model.score(X_test, y_test))
    print("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(y_test))