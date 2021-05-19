import numpy as np
from datetime import datetime
from KNN.utils import get_data
from scipy.stats import multivariate_normal as mvn

class Bayes(object):
    def fit(self, X, Y, smoothing=1e-2):
        _, D = X.shape
        self.gaussians = dict()
        self.priors = dict()
        labels = set(Y)
        for label in labels:
            x = X[Y==label]
            self.gaussians[label] = {
                'mean': x.mean(axis=0),
                'cov': np.cov(x.T) + smoothing * np.eye(D)
            }
            self.priors[label] = float(len(Y[Y == label])) / len(Y)

    def score(self, X, Y):
        prediction = self.predict(X)
        return np.mean(prediction == Y)

    def predict(self, X):
        N, _ = X.shape
        K = len(self.gaussians)
        P = np.zeros((N, K)) # For each N samples we calculate K probabilities
        for label, gaussian in self.gaussians.items():
            mean, cov = gaussian['mean'], gaussian['cov']
            P[:,label] = mvn.logpdf(X, mean=mean, cov=cov) + np.log(self.priors[label])
        return np.argmax(P, axis=1)

if __name__ == '__main__':
    X, Y = get_data()
    N_train = len(Y) // 4 * 3
    X_train, Y_train = X[:N_train], Y[:N_train]
    X_test, Y_test = X[N_train:], Y[N_train:]

    model = Bayes()
    t0 = datetime.now()
    model.fit(X_train, Y_train)
    print("Trainin time: {}".format(datetime.now() - t0))

    t0 = datetime.now()
    print("Train accuracy: {}".format(model.score(X_train, Y_train)))
    print("Train accuracy time: {} | Test size: {}".format(datetime.now() - t0, len(Y_train)))

    t0 = datetime.now()
    print("Test accuracy: {}".format(model.score(X_test, Y_test)))
    print("Test accuracy time: {} | Test size: {}".format(datetime.now() - t0, len(Y_test)))