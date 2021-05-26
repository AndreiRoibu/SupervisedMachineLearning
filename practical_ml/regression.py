import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

N = 200
X = np.linspace(0,10,N).reshape(N,1)
y = np.sin(X)

N_train = 20
idx = np.random.choice(N, N_train)
X_train = X[idx]
y_train = y[idx]

knn = KNeighborsRegressor(n_neighbors=2)
knn.fit(X_train, y_train)
y_knn = knn.predict(X)

knn2 = KNeighborsRegressor(n_neighbors=2, weights='distance')
knn2.fit(X_train, y_train)
y_knn2 = knn2.predict(X)

dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
y_dt = dt.predict(X)

plt.figure()
plt.scatter(X_train, y_train, label='train data')
plt.plot(X, y, label='original data')
plt.plot(X, y_knn, label='knn')
plt.plot(X, y_knn2, label='knn distance')
plt.plot(X, y_dt, label='Decision Tree')
plt.legend()
plt.show()