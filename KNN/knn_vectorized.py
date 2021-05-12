import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from datetime import datetime
from KNN.utils import get_data
from sklearn.metrics.pairwise import pairwise_distances

class KNN(object):
    def __init__(self, K):
        self.K = K

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X_test):
        y_hat = np.zeros(len(X_test)) # These are the test points; we need a prediction for every input

        # We first return the distances in a matrix of shape (N_test, N_train)
        distances = pairwise_distances(X_test, self.X)

        # We then gen the minimum K elements' indexes
        indexes = distances.argsort(axis=1)[:, :self.K]

        # We then determine the winning votes.
        # Each row of indexes contains the indexes between [0, N_train] correponding to indexes of the closes samples from the training set.
        votes = self.y[indexes]

        # Y contains the classes in each row
        for i in range(len(X_test)):
            y_hat[i] = np.bincount(votes[i]).argmax()

        return y_hat

    def score(self, X, y):
        prediction = self.predict(X)
        return np.mean(prediction == y)

if __name__ == '__main__':
    X, y = get_data(limit=2000)
    train_points = 1000
    X_train, y_train = X[:train_points], y[:train_points]
    X_test, y_test = X[train_points:], y[train_points:]
    del X, y
    train_scores = []
    test_scores = []
    Ks = (1,2,3,4,5)
    for K in Ks:
        print('\nK = ', K)
        knn_model = KNN(K)
        t0 = datetime.now()
        knn_model.fit(X_train, y_train)
        print("Training time: ", datetime.now() - t0)

        t0 = datetime.now()
        train_score = knn_model.score(X_train, y_train)
        train_scores.append(train_score)
        print("Train accuracy: ", train_score)
        print("Time spent computing train accuracy: {} ; Train size: {}".format(datetime.now()-t0, len(y_train)))

        t0 = datetime.now()
        test_score = knn_model.score(X_test, y_test)
        test_scores.append(test_score)
        print("Train accuracy: ", test_score)
        print("Time spent computing test accuracy: {} ; Test size: {}".format(datetime.now()-t0, len(y_test)))

    plt.figure()
    plt.plot(Ks, train_scores, label="Train Scores")
    plt.plot(Ks, test_scores, label='Test Scores')
    plt.legend()
    plt.show()