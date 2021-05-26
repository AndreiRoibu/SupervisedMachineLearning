import requests
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from KNN.utils import get_data

X, y = get_data()
N = len(y)

while True:
    i = np.random.choice(N)
    r = requests.post("http://localhost:8888/predict", data={'input': X[i]})
    print("Response:")
    print(r.content)
    j = r.json() # Parsing the result into a dictionary
    print(j)
    print("target:", y[i])

    plt.figure()
    plt.imshow(X[i].reshape(28, 28), cmap='gray')
    plt.title("Target: %d, Prediction: %d" % (y[i], j['prediction']))
    plt.show()

    response = input('Continue? [y]/n \n')
    if response in ('n', 'N'):
        break