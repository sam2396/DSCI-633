import pandas as pd
import numpy as np
from copy import deepcopy
from pdb import set_trace
import math

class my_AdaBoost:

    def __init__(self, base_estimator = None, n_estimators = 50):
        # base_estimator: the base classifier class, e.g. my_DT
        # n_estimators: # of base_estimator rounds
        self.base_estimator = base_estimator
        self.n_estimators = int(n_estimators)
        self.estimators = [deepcopy(self.base_estimator) for i in range(self.n_estimators)]

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str

        self.classes_ = list(set(list(y)))
        k = len(self.classes_)
        n = len(y)
        w = np.array([1.0 / n] * n)
        labels = np.array(y)
        self.alpha = []
        for i in range(self.n_estimators):
            # Sample with replacement from X, with probability w
            sample = np.random.choice(n, n, p=w)
            # Train base classifier with sampled training data
            sampled = X.iloc[sample]
            sampled.index = range(len(sample))
            self.estimators[i].fit(sampled, labels[sample])
            predictions = self.estimators[i].predict(X)
            diffs = np.array(predictions) != y
            # Compute error rate and alpha for estimator i
            error = np.sum(diffs * w)
            while error >= (1 - 1.0 / k):
                w = np.array([1.0 / n] * n)
                sample = np.random.choice(n, n, p=w)
                # Train base classifier with sampled training data
                sampled = X.iloc[sample]
                sampled.index = range(len(sample))
                self.estimators[i].fit(sampled, labels[sample])
                predictions = self.estimators[i].predict(X)
                diffs = np.array(predictions) != y
                # Compute error rate and alpha for estimator i
                error = np.sum(diffs * w)
            # Compute alpha for estimator i
            al = (math.log((1 - error) / error))+(math.log(k-1))
            #print(al)
            self.alpha.append(al)
            #print(self.alpha)
            # Update wi
            wrongweight = math.exp(al)  
            for j in range(len(w)):
                if(diffs[j] == True):
                    w[j] = w[j]*math.exp(al)
                else:
                    w[j] = w[j]
            #print(w)
            #print(np.sum(w))
            w = w/np.sum(w)
            
            #print(np.sum(w))
            #print("----------------")
            #print(w)
        # Normalize alpha
        #print(self.alpha)
        self.alpha = self.alpha / np.sum(self.alpha)
        #print(self.alpha)
        return

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        probs = self.predict_proba(X)
        predictions = [self.classes_[np.argmax(prob)] for prob in probs.to_numpy()]
        return predictions

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # prob: what percentage of the base estimators predict input as class C
        # prob(x)[C] = sum(alpha[j] * (base_model[j].predict(x) == C))
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        # write your code below
        probs = {}
        #prob = 
        #for x in range(len(X)):
        for label in self.classes_:
            a = []
            for i in range(len(self.alpha)):
                a.append(self.alpha[i] * (self.estimators[i].predict(X)==label))
                probs[label] = np.sum(a,axis=0)
                #print(probs)
            
            
           # for i in range(self.n_estimators):
            #    probs(X)[label] = sum(self.alpha[i]*(self.estimators[i].predict(X)==label))
        probs = pd.DataFrame(probs, columns=self.classes_)
        return probs










