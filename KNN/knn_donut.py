import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from KNN.utils import get_donut
from KNN.knn import KNN

if __name__ == '__main__':
    X, y = get_donut()
    plt.figure()
    plt.scatter(X[:,0], X[:, 1], s=100, c=y)
    plt.show()

    model = KNN(K=3)
    model.fit(X, y)
    print("Model accuracy: {}".format(model.score(X,y)))