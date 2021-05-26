import numpy as np
import pandas as pd
import os

def get_data(limit=None):

    print("Reading and transforming data...")
    data_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    data_path = data_path.replace('SupervisedMachineLearning/KNN', '') 
    data_path = data_path + 'large_files/train.csv'

    data = pd.read_csv(data_path).values
    X = data[:, 1:] / 255.0
    y = data[:, 0]

    if limit is not None:
        X, y = X[:limit], y[:limit]

    print("Data read and transformed successfully!")

    return X, y

def get_XOR():
    X = np.zeros((200, 2))
    X[:50] = np.random.random((50,2)) / 2 + 0.5 # (0.5-1, 0.5-1)
    X[50:100] = np.random.random((50,2)) / 2 # (0-0.5, 0-0.5)
    X[100:150] = np.random.random((50,2)) / 2 + np.array([[0, 0.5]]) # (0-0.5, 0.5-1)
    X[150:] = np.random.random((50, 2)) / 2 + np.array([[0.5, 0]]) # (0.5-1, 0-0.5)
    y = np.array([0]*100 + [1]*100)
    return X,y

def get_donut():
    number_of_points = 200
    inner_R = 5
    outer_R = 10
    R1 = np.random.randn(number_of_points//2) + inner_R
    R2 = np.random.randn(number_of_points//2) + outer_R
    theta1 = 2 * np.pi * np.random.random(number_of_points//2)
    theta2 = 2 * np.pi * np.random.random(number_of_points//2)
    X_inner = np.concatenate( [ [R1 * np.cos(theta1)], [R1*np.sin(theta1)] ] ).T
    X_outer = np.concatenate( [ [R2 * np.cos(theta2)], [R2*np.sin(theta2)] ] ).T

    X = np.concatenate([ X_inner, X_outer ])
    y = np.array([0] * (number_of_points//2) + [1] * (number_of_points//2))

    return X, y

def get_random_perceptron_data():
    w = np.array([-0.5, 0.5])
    b = 0.1
    X = np.random.random((300, 2)) * 2.0 - 1.0
    y = np.sign(X.dot(w) + b)
    return X, y

if __name__ == "__main__":
    get_data()