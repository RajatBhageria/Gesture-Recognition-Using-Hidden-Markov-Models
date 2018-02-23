import numpy as np
import pickle

#returns a striong of the predicted gesture from a list of 'beat3','beat4','circle','eight','inf','wave'
def classifyTestData(fileName = None):
    filename = 'test_data/file.txt'
    IMU = np.loadtxt(filename)
    allData = IMU[:,1:6]

    #predict the clusters using k-means and generate an observation sequence
    KMeansModel = pickle.load(open('kmeans_model.pickle', 'rb'))
    observationSequence = KMeansModel.predict(allData)

    #use Baum-Welch to find which gesture has the highest probability
    gestureNames = np.array(['beat3','beat4','circle','eight','inf','wave'],dtype='object')
    probabilities = np.empty((6,1))
    for gesture in range(0,6):
        probabilities[gesture] = 0

    #take the argmax of the matrix to find which gesture has the highest probability
    indexOfName = np.argmax(probabilities)

    #return the name of that gesture
    predictedGesture = gestureNames[indexOfName]
    print predictedGesture
    return predictedGesture

if __name__ == "__main__":
    classifyTestData()