from scipy import io
import math
import numpy as np
from sklearn.cluster import KMeans

def gestureRecognition():

    #instantiate variables
    pi = 0

    #number of hidden states N
    N = 10

    #number of observation types M
    M = 8;

    #A and B matrix
    A = np.random.rand(N,N)
    A = A / A.sum(axis=1)[:, None]

    B = np.random.rand(N,M)
    B = B / B.sum(axis=1)[:, None]

