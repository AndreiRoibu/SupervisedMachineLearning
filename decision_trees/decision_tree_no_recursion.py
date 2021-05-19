import numpy as np
from datetime import datetime
from KNN.utils import get_data, get_XOR, get_donut
from sklearn.utils import shuffle

def entropy(y):
    # This assumes that y is binary
    N = len(y)
    s1 = (y == 1).sum()
    if 0 == s1 or N == s1:
        return 0 # if the number of ones is 0 or N, the entropy will be 0 == shortcut
    p1 = float(s1) / N 
    p0 = 1 - p1

    return 1 - p0*p0 - p1*p1

class TreeNode:
    def __init__(self, depth=1, max_depth=None):

        self.max_depth = max_depth
        self.root = {}
        # each node has the following k-v pairs atributes: col, split, left, right, prediction

    def fit(self, X, Y):
        # First, we go through the base case where we only have 1 sample -- only receives examples from 1 class, so we can't make a split
        
        current_node = self.root
        depth = 0
        queue = []

        while True:
            
            if len(Y) == 1 or len(set(Y)) == 1:
                current_node['col'] = None
                current_node['split'] = None
                current_node['left'] = None
                current_node['right'] = None
                current_node['prediction'] = Y[0]
            else:
                D = X.shape[1]
                cols = range(D)

                max_ig = 0 # Max information gain
                best_col = None
                best_split = None

                for col in cols:
                    ig, split = self.find_split(X, Y, col)
                    if ig > max_ig:
                        max_ig = ig
                        best_col = col
                        best_split = split

                if max_ig == 0: # Another base case which means we can't do more splits here
                    current_node['col'] = None
                    current_node['split'] = None
                    current_node['left'] = None
                    current_node['right'] = None
                    current_node['prediction'] = np.round(Y.mean())
                else:
                    current_node['col'] = best_col
                    current_node['split'] = best_split

                    if depth == self.max_depth:
                        current_node['left'] = None
                        current_node['right'] = None
                        current_node['prediction'] = [
                            np.round(Y[X[:,best_col] < self.split].mean()),
                            np.round(Y[X[:,best_col] >= self.split].mean()),
                        ]             
                    else:
                        # Not a base case, so recursion is needed
                        left_idx = (X[:, best_col] < best_split)
                        X_left = X[left_idx]
                        Y_left = Y[left_idx]

                        new_node = {}
                        current_node['left'] = new_node
                        left_data = {
                            'node': new_node,
                            'X': X_left,
                            'Y': Y_left,
                        }
                        queue.insert(0, left_data)

                        right_idx = (X[:, best_col] >= best_split)
                        X_right = X[right_idx]
                        Y_right = Y[right_idx]

                        new_node = {}
                        current_node['right'] = new_node
                        right_data = {
                            'node': new_node,
                            'X': X_right,
                            'Y': Y_right,
                        }
                        queue.insert(0, right_data)

            if len(queue) == 0:
                break

            next_data = queue.pop()
            current_node = next_data['node']
            X = next_data['X']
            Y = next_data['Y']

    def find_split(self, X, Y, col):
        x = X[:, col]
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y = Y[sort_idx]

        # The optimal split is the midpoint between 2 points, and only on the boundaries between 2 classes 

        boundaries = np.nonzero(y[:-1] != y[1:])[0] 
        # if boundaries is true, then y[i]!=y[i+1], allowing nonzoer to return the indices where arg is true
        best_split = None
        max_ig = 0
        last_ig = 0
        for boundary in boundaries:
            split = (x[boundary] + x[boundary+1]) / 2
            ig = self.information_gain(x, y, split)

            if ig < last_ig:
                break
            last_ig = ig

            if ig > max_ig:
                max_ig = ig
                best_split = split

        return max_ig, best_split

    def information_gain(self, x, y, split):

        y0 = y[x < split]
        y1 = y[x >= split]
        N = len(y)
        y0_len = len(y0)
        if y0_len == 0 or y0_len == N:
            return 0
        p0 = float(len(y0)) / N
        p1 = 1 - p0
        return entropy(y) - p0 * entropy(y0) - p1*entropy(y1)

    def predict_one(self, x):

        p = None
        current_node = self.root
        while True:           
            if current_node['col'] is not None and current_node['split'] is not None:
                feature = x[current_node['col']]
                if feature < current_node['split']:
                    if current_node['left']:
                        current_node = current_node['left']
                    else:
                        p = current_node['prediction'][0]
                        break
                else:
                    if current_node['right']:
                        current_node = current_node['right']
                    else:
                        p = current_node['prediction'][1]
                        break
            else:
                # We only have 1 prediction
                p = current_node['prediction']
                break

        return p

    def predict(self, X):
        N = len(X)
        P = np.zeros(N)
        for i in range(N):
            P[i] = self.predict_one(X[i])
        return P

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
    
    def fit(self, X, Y):
        self.root = TreeNode(max_depth=self.max_depth)
        self.root.fit(X, Y)

    def predict(self, X):
        return self.root.predict(X)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)

if __name__ == '__main__':
        
        X, Y = get_data()

        # X, Y = get_xor()
        # # X, Y = get_donut()
        # X, Y = shuffle(X, Y)

        # only take 0s and 1s since we're doing binary classification
        idx = np.logical_or(Y == 0, Y == 1)
        X = X[idx]
        Y = Y[idx]

        N_train = len(Y) // 4 * 3
        X_train, Y_train = X[:N_train], Y[:N_train]
        X_test, Y_test = X[N_train:], Y[N_train:]
               
        model = DecisionTree()
        # model = DecisionTree(max_depth=7)
        t0 = datetime.now()
        model.fit(X_train, Y_train)
        print("Trainin time: {}".format(datetime.now() - t0))

        t0 = datetime.now()
        print("Train accuracy: {}".format(model.score(X_train, Y_train)))
        print("Train accuracy time: {} | Test size: {}".format(datetime.now() - t0, len(Y_train)))

        t0 = datetime.now()
        print("Test accuracy: {}".format(model.score(X_test, Y_test)))
        print("Test accuracy time: {} | Test size: {}".format(datetime.now() - t0, len(Y_test)))