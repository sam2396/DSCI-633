import pandas as pd
import numpy as np
from collections import Counter
from math import sqrt


class my_KNN:

    def __init__(self, n_neighbors=5, metric="minkowski", p=2):
        # metric = {"minkowski", "euclidean", "manhattan", "cosine"}
        # p value only matters when metric = "minkowski"
        # notice that for "cosine", 1 is closest and -1 is furthest
        # therefore usually cosine_dist = 1- cosine(x,y)
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.p = p

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str
        self.classes_ = list(set(list(y)))
        self.X = X
        self.y = y
        return

    def dist(self,x):
        # Calculate distances of training data to a single input data point (np.array)
        if self.metric == "minkowski":
            mainArray = np.empty
            c = np.array(self.X)
            for i in range(len(c)):
                dist = (sum((abs(c[i]-x))**self.p))**(1/self.p)
                mainArray = np.append(mainArray,dist)
            mainArray = np.delete(mainArray,[0])
            distances = mainArray
            return distances
        elif self.metric == "euclidean":
            mainArray = np.empty
            c = np.array(self.X)
            for i in range(len(c)):
                dist = (sum((abs(c[i]-x))**2))**(1/2)
                mainArray = np.append(mainArray,dist)
            mainArray = np.delete(mainArray,[0])
            distances = mainArray
            return distances
        elif self.metric == "manhattan":
            mainArray = np.empty
            c = np.array(self.X)
            for i in range(len(c)):
                dist = (sum((abs(c[i]-x))))
                mainArray = np.append(mainArray,dist)
            mainArray = np.delete(mainArray,[0])
            distances = mainArray
            return distances
        elif self.metric == "cosine":
            mainArray = np.empty
            c = np.array(self.X)
            for i in range(len(c)):
                dist = (sum(c[i]*x))/(sqrt(sum(c[i]**2))*(sqrt(sum(x**2))))
                mainArray = np.append(mainArray,1-dist)
            mainArray = np.delete(mainArray,[0])    
            distances = mainArray
            return distances
        else:
            raise Exception("Unknown criterion.")
        return distances

    def k_neighbors(self,x):
        # Return the stats of the labels of k nearest neighbors to a single input data point (np.array)
        # Output: Counter(labels of the self.n_neighbors nearest neighbors)
        distances = self.dist(x)
        sortedDistances = np.argsort(distances)
        BestDistances = sortedDistances[:self.n_neighbors]
        BestDistancesLabels = []
        for label in BestDistances:
            BestDistancesLabels.append(self.y[label])
        output = Counter(BestDistancesLabels)
        return output

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        probs = self.predict_proba(X)
        predictions = [self.classes_[np.argmax(prob)] for prob in probs.to_numpy()]
        return predictions

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # prob is a dict of prediction probabilities belonging to each categories
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        probs = []
        try:
            X_feature = X[self.X.columns]
        except:
            raise Exception("Input data mismatch.")
        
        for x in X_feature.to_numpy():
            neighbors = self.k_neighbors(x)
            probs.append({key: neighbors[key] / float(self.n_neighbors) for key in self.classes_})
        print(probs)
        probs = pd.DataFrame(probs, columns=self.classes_)
        print(probs)
        return probs