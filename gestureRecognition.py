from scipy import io
import math
import numpy as np
import pickle
from HMM import HMM
from preprocessTrainingData import preprocessTrainingData

def gestureRecognition():

    #if want to pre-process data
    #beat3Obs, beat4Obs, circleObs, eightObs, infObs, waveObs = preprocessTrainingData()

    #else simply load saved pre-process data
    with open('beat3Obs.pickle', 'rb') as handle:
        beat3Obs = pickle.load(handle)
    # with open('beat4Obs.pickle', 'rb') as handle:
    #     beat4Obs = pickle.load(handle)
    # with open('circleObs.pickle', 'rb') as handle:
    #     circleObs = pickle.load(handle)
    # with open('eightObs.pickle', 'rb') as handle:
    #     eightObs = pickle.load(handle)
    # with open('infObs.pickle', 'rb') as handle:
    #     infObs = pickle.load(handle)
    # with open('waveObs.pickle', 'rb') as handle:
    #     waveObs = pickle.load(handle)

    #number of hidden states N
    n_states = 10

    #number of observation types M
    n_obs = 30

    #Create an HMM for each of the different gestures

    #instantiate variables
    pi = (1.0 / n_states) * np.ones((n_states, 1))

    #A and B matrix
    A = np.random.rand(n_states,n_states)
    A = A / A.sum(axis=1)[:, None]

    B = np.random.rand(n_obs,n_states)
    B = B / B.sum(axis=1)[:, None]

    #test the forward backwards algorithm on one of the datasets
    hmm = HMM(n_states, n_obs, pi, A, B)
    testSequence = np.array(beat3Obs[0])
    probObservations = hmm.log_backward(testSequence)
    print probObservations



if __name__ == "__main__":
    gestureRecognition()

