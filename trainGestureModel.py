from scipy import io
import math
import numpy as np
import pickle
from HMM import HMM
from preprocessTrainingData import preprocessTrainingData

#returns a 6x1 np.array of trained HMM Objects
def trainGestureModel():

    #if want to pre-process data
    #beat3Obs, beat4Obs, circleObs, eightObs, infObs, waveObs = preprocessTrainingData()

    #number of hidden states N
    n_states = 10

    #number of observation types M
    n_obs = 30

    #instantiate variables
    pi = (1.0 / n_states) * np.ones((n_states, 1))

    #A and B matrix
    A = np.random.rand(n_states,n_states)
    A = A / A.sum(axis=1)[:, None]

    B = np.random.rand(n_obs,n_states)
    B = B / B.sum(axis=1)[:, None]

    #Get the probability of observations
    gestureNames = np.array(['beat3','beat4','circle','eight','inf','wave'],dtype='object')
    HMMModels = np.empty((6,7),dtype='object')
    #iterate through the list of gestures
    for gesture in range(0,gestureNames.shape[0]):
        gestureName = gestureNames[gesture]
        #load the data for the type of gesture
        observationDataFileName = "".join((gestureName,"Obs.pickle"))
        with open(observationDataFileName, 'rb') as handle:
            observationSequences = pickle.load(handle)
        #Generate the trained HMM model for the correct gesture
        for j in range(0,len(observationSequences)):
            hmmModelOfGesture = HMM(n_states, n_obs, pi, A, B)
            observationSequence = observationSequences[j]
            hmmModelOfGesture.baum_welch(observationSequence, max_iter=3)
            #Add the model to the list of models
            HMMModels[gesture,j] = hmmModelOfGesture

    with open('HMMModels.pickle', 'wb') as handle:
        pickle.dump(HMMModels, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return HMMModels


if __name__ == "__main__":
    trainGestureModel()

