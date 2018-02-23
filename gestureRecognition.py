from scipy import io
import math
import numpy as np
import pickle

def gestureRecognition():
    with open('beat3Obs.pickle', 'rb') as handle:
        beat3Obs = pickle.load(handle)
    with open('beat4Obs.pickle', 'rb') as handle:
        beat4Obs = pickle.load(handle)
    with open('circleObs.pickle', 'rb') as handle:
        circleObs = pickle.load(handle)
    with open('eightObs.pickle', 'rb') as handle:
        eightObs = pickle.load(handle)
    with open('eightObs.pickle', 'rb') as handle:
        eightObs = pickle.load(handle)
    with open('waveObs.pickle', 'rb') as handle:
        waveObs = pickle.load(handle)

    #number of hidden states N
    N = 10

    #number of observation types M
    M = 100

    #Create an HMM for each of the different gestures

    #instantiate variables
    pi = (1.0 / N) * np.ones((N, 1))

    #A and B matrix
    A = np.random.rand(N,N)
    A = A / A.sum(axis=1)[:, None]

    B = np.random.rand(M,N)
    B = B / B.sum(axis=1)[:, None]

