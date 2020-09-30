import pandas as pd
import numpy as np
from pdb import set_trace

class my_KMeans:

    def __init__(self, n_clusters=8, init = "k-means++", n_init = 10, max_iter=300, tol=1e-4):
        # init = {"k-means++", "random"}
        # stop when either # iteration is greater than max_iter or the delta of self.inertia_ is smaller than tol.
        # repeat n_init times and keep the best run (cluster_centers_, inertia_) with the lowest inertia_.
        self.n_clusters = int(n_clusters)
        self.init = init
        self.n_init = n_init
        self.max_iter = int(max_iter)
        self.tol = tol

        self.classes_ = range(n_clusters)
        # Centroids
        self.cluster_centers_ = None
        # Sum of squared distances of samples to their closest cluster center.
        self.inertia_ = None

    def dist(self, a, b):
        # Compute Euclidean distance between a and b
        return np.sum((np.array(a)-np.array(b))**2)**(0.5)

    def initiate(self, X):
        # Initiate cluster centers
        # Input X is numpy.array
        # Output cluster_centers (list)

        if self.init == "random":
            b = np.random.choice(X.shape[0],self.n_clusters,replace=False)
            c = X[b]
            randDist = []
            for i in range(len(b)):
                randDist.append(c[i])
            cluster_centers = randDist
            #print(cluster_centers)

        elif self.init == "k-means++":
            finalClusterdata = []
            b = np.random.choice(X.shape[0],1,replace=False)
            c = X[b]
            for i in range(len(b)):
                finalClusterdata.append(c[i])
            minIndice = 0
            kdistIndice = list()
            for o in range(self.n_clusters-1):
                kdist = [np.min([self.dist(x,j) for j in finalClusterdata])**2 for x in X]
                min1 = kdist/np.sum(kdist)
                minIndice = np.random.choice(X.shape[0],1,p=min1)   
                kdistIndice.append(minIndice)
            for i in kdistIndice:    
                finalClusterdata.extend(list(X[i]))   
            cluster_centers = finalClusterdata

        else:
            raise Exception("Unknown value of self.init.")
        return cluster_centers

    def fit_once(self, X):
        # Fit once
        # Input X is numpy.array
        # Output: cluster_centers (list), inertia

        # Initiate cluster centers
        cluster_centers = self.initiate(X)
        last_inertia = None
        # Iterate
        for i in range(self.max_iter+1):
            # Assign each training data point to its nearest cluster_centers
            clusters = [[] for i in range(self.n_clusters)]
            inertia = 0
            for x in X:
                # calculate distances between x and each cluster center
                dists = [self.dist(x, center) for center in cluster_centers]
                # calculate inertia
                inertia += min(dists)*min(dists)
                # find the cluster that x belongs to
                cluster_id = dists.index(min(dists))
                # add x to that cluster
                clusters[cluster_id].append(x)

            if (last_inertia and last_inertia - inertia < self.tol) or i==self.max_iter:
                break
            # Update cluster centers
            v = []
            for i in clusters:
                r = list((np.mean(i,axis=0)))
                v.append(np.array(r))
            cluster_centers = v


            last_inertia = inertia

        return cluster_centers, inertia


    def fit(self, X):
        # X: pd.DataFrame, independent variables, float
        # repeat self.n_init times and keep the best run
            # (self.cluster_centers_, self.inertia_) with the lowest self.inertia_.
        X_feature = X.to_numpy()
        for i in range(self.n_init):
            cluster_centers, inertia = self.fit_once(X_feature)
            if self.inertia_==None or inertia < self.inertia_:
                self.inertia_ = inertia
                self.cluster_centers_ = cluster_centers
        return


    def transform(self, X):
        # Transform to cluster-distance space
        # X: pd.DataFrame, independent variables, float
        # return dists = list of [dist to centroid 1, dist to centroid 2, ...]
        dists = [[self.dist(x,centroid) for centroid in self.cluster_centers_] for x in X.to_numpy()]
        return dists

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        predictions = [np.argmax(dist) for dist in self.transform(X)]
        return predictions


    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

