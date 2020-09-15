import pandas as pd
import numpy as np
from collections import Counter

class my_KNN_hint:

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
            finaldist1 = np.empty
            c = np.array(self.X)
            for i in range(len(c)):
                dist = (sum((abs(c[i]-x))**self.p))**(1/self.p)
                finaldist1 = np.append(finaldist1,dist)
                #Final minkowski distance  
            finaldist1 = np.delete(finaldist1,[0])
            distances = finaldist1
            return distances


        elif self.metric == "euclidean":
            distances = "write your own code"


        elif self.metric == "manhattan":
            distances = "write your own code"


        elif self.metric == "cosine":
            distances = "write your own code"


        else:
            raise Exception("Unknown criterion.")
        return distances

    def k_neighbors(self,x):
        # Return the stats of the labels of k nearest neighbors to a single input data point (np.array)
        # Output: Counter(labels of the self.n_neighbors nearest neighbors)
        distances = self.dist(x)
        
        #distances2 = np.sort(distances)
        #distances3 = distances2[:5]
        p1 = np.argsort(distances)
        p2 = p1[:5]
        p3 = []
        for i in p2:
            p3.append(self.y[i])
        output1 = Counter(p3)
        #output = distances3



        return output1

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
        probs = pd.DataFrame(probs, columns=self.classes_)
        return probs



