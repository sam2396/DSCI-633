import numpy as np
from scipy.linalg import svd
from copy import deepcopy
from collections import Counter
from pdb import set_trace

def pca(X, n_components = 5):
    #  Use svd to perform PCA on X
    #  Inputs:
    #     X: input matrix
    #     n_components: number of principal components to keep
    #  Output:
    #     principal_components: the top n_components principal_components
    #     X_pca = X.dot(principal_components)

    U, s, Vh = svd(X)
    V = Vh.transpose()
    principal_components = V[:,:n_components]
    return principal_components

def vector_norm(x, norm="Min-Max"):
    # Calculate the normalized vector
    if norm == "Min-Max":
        x_norm = (x-np.min(x))/(np.max(x)-np.min(x))
    elif norm == "L1":
        x_norm = x/(np.sum(abs(x)))
    elif norm == "L2":
        x_norm = x/np.sqrt(np.sum(x**2))
    elif norm == "Standard_Score":
        x_norm = ((x-np.mean(x))/np.std(x))
    else:
        raise Exception("Unknown normlization.")
    return x_norm

def normalize(X, norm="Standard_Score", axis = 1):
    #  Inputs:
    #     X: input matrix
    #     norm = {"L1", "L2", "Min-Max", "Standard_Score"}
    #     axis = 0: normalize rows
    #     axis = 1: normalize columns
    #  Output:
    #     X_norm: normalized matrix (numpy.array)

    X_norm = deepcopy(np.asarray(X))
    m, n = X_norm.shape
    if axis == 1:
        for col in range(n):
            X_norm[:,col] = vector_norm(X_norm[:,col], norm=norm)
    elif axis == 0:
        X_norm = np.array([vector_norm(X_norm[i], norm=norm) for i in range(m)])
    else:
        raise Exception("Unknown axis.")
    return X_norm

def stratified_sampling(y, ratio, replace = True):
    #  Inputs:
    #     y: class labels
    #     0 < ratio < 1: number of samples = len(y) * ratio
    #     replace = True: sample with replacement
    #     replace = False: sample without replacement
    #  Output:
    #     sample: indices of stratified sampled points
    #             (ratio is the same across each class,
    #             samples for each class = int(np.ceil(ratio * # data in each class)) )

    if ratio<=0 or ratio>=1:
        raise Exception("ratio must be 0 < ratio < 1.")
    y_array = np.asarray(y)
    dist = []
    dist = list(set(y_array))
    clusters = [[] for i in range(len(dist))]
    for x in y_array:
        for d in dist:
            if(x == d):
                clusters[dist.index(d)] = np.where(y_array==x)
    fin = []
    for n in range(len(clusters)):
        tot = np.ceil(len(clusters[n][0])*ratio)
        fin.append(np.random.choice(clusters[n][0],int(tot),replace=replace))
    fin = np.concatenate((fin),axis=0)
    sample = fin

    return sample.astype(int)
