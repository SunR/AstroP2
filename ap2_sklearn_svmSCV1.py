from __future__ import division
from math import *
import random
import numpy as np
from sklearn import svm
import astroML
from astroML.datasets import fetch_sdss_specgals
import time


#get just spirals and ellipticals in 1 array, shuffle them, then extract the label column
data = np.loadtxt("crossmatched1_imagingdata_srane.txt", delimiter = ",", skiprows = 1, usecols = (2, 11, 13, 15, 17, 19, 23, 24, 25))
labels = np.loadtxt("crossmatched1_imagingdata_srane.txt", delimiter = ",", skiprows = 1, usecols = (14,)) #only gets binary label col of elliptical

isSpiral = data[:,6]
isElliptical = data[:,7] #corresponds to col 24 of real data file, which is the elliptical bool value
isUncertain = data[:,8]

ellipticals = data[isElliptical == 1]

spirals = data[isSpiral == 1]
uncertains = data[isUncertain == 1]

trainingSetEllipticals = ellipticals[:50000] #check whether these numbers are inclusive
trainingSetSpirals = spirals[:50000] #extracting first 5000 spiral and elliptical to train model, excluding last 3 cols (labels)

trainingSet = np.vstack((trainingSetEllipticals, trainingSetSpirals))  #using only elliptical and spiral for training
np.random.shuffle(trainingSet)
trainingSetLabels = trainingSet[:,7]  #putting labels in separate array
trainingSetLabels[trainingSetLabels == 0] = -1 #replacing all 0 with -1 to match sklearn format
trainingSet = trainingSet[:, 1:6] #removing label cols from actual inputs
startTime = time.time()
print "Time before training = ", startTime

clf = svm.SVC() 
clf_info = clf.fit(trainingSet, trainingSetLabels)
print clf_info

print "Done training! Time = ", time.time() - startTime, "seconds"
#Training accuracy

#prediction = clf.predict(trainingSetEllipticals)

trainingAccuracy = clf.score(trainingSet, trainingSetLabels)
    
print "Training accuracy = ", trainingAccuracy
print "Time = ", time.time() - startTime, "seconds"
print

#Testing accuracy
testingSetEllipticals = ellipticals[50000:100000]
testingSetSpirals = spirals[50000:100000]

testingSet = np.vstack((testingSetEllipticals, testingSetSpirals))  #using only elliptical and spiral for training
np.random.shuffle(testingSet)
testingSetLabels = testingSet[:,7]  #putting labels in separate array
testingSetLabels[testingSetLabels == 0] = -1 #replacing all 0 with -1 to match sklearn format
testingSet = testingSet[:, 1:6] #removing label cols from actual inputs

testingAccuracy = clf.score(testingSet, testingSetLabels)
print "Testing accuracy = ", testingAccuracy
print "Time = ", time.time() - startTime, "seconds"

classificationSet = uncertains[:, 1:6]
predictions = clf.predict(classificationSet) #use model to classify uncertains - SEE IF YOU CAN FIND CONFIDENCE INTERVAL FOR THIS PREDICTION AFTERWARD, THAT WILL BE PROGRESS

uncertainsPredictions = np.hstack(uncertains[:, 0:6], predictions) #check to make sure this is aligning inputs to predictions properly

f = open('uncertainsPredictionsOutput1.txt', 'w')


