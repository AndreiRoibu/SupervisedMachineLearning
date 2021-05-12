import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sortedcontainers import SortedList
from datetime import datetime
from KNN.utils import get_data

class KNN(object):
    def __init__(self, K):
        self.K = K

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X_test):
        y_hat = np.zeros(len(X_test)) # These are the test points; we need a prediction for every input
        for i, x in enumerate(X_test): # Enumerate through the test points
            sorted_list = SortedList() # Sorted list that stores the (distance, class) tuples
            for j, x_train in enumerate(self.X): # Enumerate through the train points
                difference = x - x_train
                distance = difference.dot(difference) # Squared distance
                if len(sorted_list) < self.K:
                    sorted_list.add( (distance, self.y[j]) )
                else:
                    if distance < sorted_list[-1][0]: # The list being sorted, the final value is the largest
                        del sorted_list[-1]
                        sorted_list.add( (distance, self.y[j]) )

            # Now we collect all the votes:
            votes = {}
            for _, value in sorted_list:
                votes[value] = votes.get(value, 0) + 1
            max_votes = 0
            max_votes_class = -1
            for value, count in votes.items():
                if count > max_votes:
                    max_votes = count
                    max_votes_class = value
            y_hat[i] = max_votes_class
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