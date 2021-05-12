import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from KNN.knn import KNN

def get_grid_data():
    width = 8
    height = 8
    number_of_points = width * height
    X = np.zeros((number_of_points,2))
    y = np.zeros(number_of_points)
    n = 0
    start_t = 0 
    for i in range(width):
        t = start_t
        for j in range(height):
            X[n] = [i, j]
            y[n] = t
            n+=1
            t = (t+1) % 2 # alternates between {0,1}
        start_t = (start_t + 1) % 2
    return X, y

if __name__ == '__main__':
    X,y = get_grid_data()

    plt.figure()
    plt.scatter(X[:,0], X[:,1], s=100, c=y)
    plt.show()

    model = KNN(K=3)
    model.fit(X,y)
    print("Train Accuracy: {}".format(model.score(X,y)))