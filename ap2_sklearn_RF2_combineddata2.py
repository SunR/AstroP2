from __future__ import division
from math import *
import random
import numpy as np
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from astroML.datasets import fetch_sdss_specgals
from astroML.density_estimation import KNeighborsDensity
import time

#Photometric and spectral data, random forest classifer

print "Random Forest Classifier"
data = np.loadtxt("crossmatched3_combineddata1_srane.txt", delimiter = ",", skiprows = 1, usecols = (37,)) #after row 35 creates the problems..

print data[:5]

#get just spirals and ellipticals in 1 array, shuffle them, then extract the label column
data = np.loadtxt("combined2_w_velDisp_srane2.txt", delimiter = ",", skiprows = 1, usecols = (1, 10, 11, 12, 13, 14, 35, 37, 39, 41, 43, 49, 50, 51)) #58, 

isSpiral = data[:,12]
isElliptical = data[:,13] #corresponds to the elliptical bool value

isUncertain = data[:,14]

ellipticals = data[isElliptical == 1]

spirals = data[isSpiral == 1]
uncertains = data[isUncertain == 1]

trainingSetEllipticals = ellipticals #use all of set b/c cross validation will split into training and testing sets
trainingSetSpirals = spirals

trainingSet = np.vstack((trainingSetEllipticals, trainingSetSpirals))  #using only elliptical and spiral for training
np.random.shuffle(trainingSet)
trainingSetLabels = trainingSet[:,13]  #putting labels in separate array

trainingSetLabels[trainingSetLabels == 0] = -1 #replacing all 0 with -1 to match sklearn format

trainingSet = trainingSet[:, 1:12] #removing label cols from actual inputs

trainingSet, testingSet, trainingSetLabels, testingSetLabels = train_test_split(trainingSet, trainingSetLabels, test_size = 0.6, random_state = 0) #fixes random_state so results reproducible

startTime = time.time()
print "Time before training = ", startTime

clf = RandomForestClassifier() #No max depth initial, tweak as necessary later
clf = clf.fit(trainingSet, trainingSetLabels)

print "Params after training:"
print clf.get_params()

trainingAccuracy = clf.score(trainingSet, trainingSetLabels)

print "Training accuracy = ", trainingAccuracy

testingAccuracy = clf.score(testingSet, testingSetLabels)

print "Testing accuracy = ", testingAccuracy

print "Done training and testing! Time = ", time.time() - startTime, "seconds"

