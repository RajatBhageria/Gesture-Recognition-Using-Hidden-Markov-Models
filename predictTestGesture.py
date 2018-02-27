import numpy as np
import pickle
import glob
from HMM import HMM
from trainGestureModel import trainGestureModel

#returns a striong of the predicted gesture from a list of 'beat3','beat4','circle','eight','inf','wave'
def predictTestGesture(folder = 'train_data/*.txt'):
    # import all the IMU data
    file_list = glob.glob(folder)
    allData = np.empty((0, 5))

    for i in range(0, len(file_list)):
        fileName = file_list[i]
        IMU = np.loadtxt(fileName)
        allData = IMU[:, 1:6]

        #predict the clusters using k-means and generate an observation sequence
        KMeansModel = pickle.load(open('kmeans_model.pickle', 'rb'))
        observationSequence = KMeansModel.predict(allData)

        #Generate HMM models for all the gestures
        gestureNames = np.array(['beat3','beat4','circle','eight','inf','wave'],dtype='object')

        #Run if you want to train new models
        #HMMModels = trainGestureModel()
        #Else just load the pre-trained models
        HMMModels = pickle.load(open('HMMModels.pickle', 'rb'))
        [_,maxNumModels] = HMMModels.shape
        #Print which model we're running
        print "The test file we're running is: " + fileName
        #Predict the model that generated the sequence of observations using argmax
        maxProability = -float("inf")
        #set an inital guess of the name
        predictedGestureName = "beat3"
        for i in range(0,len(HMMModels)):
            gestureName = gestureNames[i]
            for j in range(0,maxNumModels):
                model = HMMModels[i,j]
                if model is not None:
                    #Use the forward algorithm to find the probaility that the model predicts the sequence
                    [logProbabilityOfObs,_] = model.log_forward(observationSequence)
                    #print "The log probability for " + gestureName+" is: "+str(logProbabilityOfObs)
                    #Check if this model has a higher probaility than the higest so far
                    if logProbabilityOfObs > maxProability:
                        maxProability = logProbabilityOfObs
                        predictedGestureName = gestureName

        #return the name of that gesture
        print "The predicted gesture for filename " + fileName + " is: " + predictedGestureName

if __name__ == "__main__":
    predictTestGesture()