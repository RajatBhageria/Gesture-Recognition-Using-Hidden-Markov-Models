from scipy import io
import math
import numpy as np
from sklearn.cluster import KMeans
import glob
import pickle

def preprocessTrainingData():
    #import all the IMU data
    folder = 'train_data/*.txt'
    file_list = glob.glob(folder)
    allData = np.empty((0,5))

    #combine the data to do k-means
    gestures = np.empty((0,1),dtype='object')
    indices = np.empty((1,),dtype='int')
    totalVals = 0
    for i in range(0,len(file_list)):
        fileName = file_list[i]
        IMU = np.loadtxt(fileName)
        dataSansTime = IMU[:,1:6]
        numValsInSequence = dataSansTime.shape[0]
        allData = np.vstack((allData,dataSansTime)) #everything except time
        gestures = np.vstack((gestures,fileName))
        totalVals = totalVals + numValsInSequence + 1
        indices = np.hstack((indices,totalVals))

    #run k-means and crate 100 clusters
    k = 30
    kmeans = KMeans(n_clusters=k, random_state=0).fit(allData)
    labels = kmeans.labels_
    pickle.dump(kmeans, open('kmeans_model.pickle', 'wb'))

    #split the results back into the sequences
    result = np.array(np.split(labels,indices))
    result = result[1:len(result)-1]

    #create objects to actually hold the observations
    beat3Obs = []
    beat4Obs = []
    circleObs = []
    eightObs = []
    infObs = []
    waveObs = []

    #find max number of observations

    for i in range (0,len(result)):
        #find the name of the particular gesture
        gestureI = str(gestures[i])
        dataSequence = result[i].T
        dataSequence = np.ndarray.tolist(dataSequence)

        if (gestureI.__contains__("beat3")):
            beat3Obs.append(dataSequence)
        elif (gestureI.__contains__("beat4")):
            beat4Obs.append(dataSequence)
        elif (gestureI.__contains__("circle")):
            circleObs.append(dataSequence)
        elif (gestureI.__contains__("eight")):
            eightObs.append(dataSequence)
        elif (gestureI.__contains__("inf")):
            infObs.append(dataSequence)
        elif (gestureI.__contains__("wave")):
            waveObs.append(dataSequence)

    beat3Obs = np.array(beat3Obs)
    beat4Obs = np.array(beat4Obs)
    circleObs = np.array(circleObs)
    eightObs = np.array(eightObs)
    infObs = np.array(infObs)
    waveObs = np.array(waveObs)

    with open('beat3Obs.pickle', 'wb') as handle:
        pickle.dump(beat3Obs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('beat4Obs.pickle', 'wb') as handle:
        pickle.dump(beat4Obs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('circleObs.pickle', 'wb') as handle:
        pickle.dump(circleObs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('eightObs.pickle', 'wb') as handle:
        pickle.dump(eightObs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('infObs.pickle', 'wb') as handle:
        pickle.dump(infObs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('waveObs.pickle', 'wb') as handle:
        pickle.dump(waveObs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return beat3Obs, beat4Obs, circleObs, eightObs, infObs, waveObs

if __name__ == "__main__":
    preprocessTrainingData()

