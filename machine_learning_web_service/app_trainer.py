import pickle
import numpy as np
from KNN.utils import get_data
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    X, y = get_data()
    N_train = len(y) // 4
    X_train, y_train = X[:N_train], y[:N_train]
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    X_test, y_test = X[N_train:], y[N_train:]
    print("Test Accuracy: {}".format(model.score(X_test, y_test)))

    with open('my_model.pkl', 'wb') as file:
        pickle.dump(model, file)