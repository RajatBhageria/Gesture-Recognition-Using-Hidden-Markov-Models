import numpy as np
import pickle
from HMM import HMM
from trainGestureModel import trainGestureModel

#returns a striong of the predicted gesture from a list of 'beat3','beat4','circle','eight','inf','wave'
def predictTestGesture(fileName = None):
    #filename = 'test_data/file.txt'
    filename = 'train_data/eight07.txt'
    IMU = np.loadtxt(filename)
    allData = IMU[:,1:6]

    #predict the clusters using k-means and generate an observation sequence
    KMeansModel = pickle.load(open('kmeans_model.pickle', 'rb'))
    observationSequence = KMeansModel.predict(allData)

    #Generate HMM models for all the gestures
    gestureNames = np.array(['beat3','beat4','circle','eight','inf','wave'],dtype='object')

    #Run if you want to train new models
    #HMMModels = trainGestureModel()
    #Else just load the pre-trained models
    HMMModels = pickle.load(open('HMMModels.pickle', 'rb'))

    #Predict the model that generated the sequence of observations using argmax
    maxProability = -float("inf")
    predictedGestureName = ""
    for i in range(0,len(HMMModels)):
        gestureName = gestureNames[i]
        model = HMMModels[i,0]
        #Use the forward algorithm to find the probaility that the model predicts the sequence
        [logProbabilityOfObs,_] = model.log_forward(observationSequence)
        print gestureName+" "+str(logProbabilityOfObs)
        #Check if this model has a higher probaility than the higest so far
        if logProbabilityOfObs > maxProability:
            maxProability = logProbabilityOfObs
            predictedGestureName = gestureName

    #return the name of that gesture
    print predictedGestureName
    return predictedGestureName

if __name__ == "__main__":
    predictTestGesture()