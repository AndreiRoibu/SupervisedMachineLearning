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

if __name__ == "__main__":
    get_data()