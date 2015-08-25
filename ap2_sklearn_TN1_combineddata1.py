from __future__ import division
from math import *
import random
import numpy as np
import climate
import theano
import theanets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import time

#Photometric and spectral data, neural network in theanet

#NOTE: pip install version of theanets had issues -- clone from github is working :)

climate.enable_default_logging()

print "Theanets Neural Network Classifier" #try Radius Neighbor Classifer!! This will give better estimate of density...(?)
#get just spirals and ellipticals in 1 array, shuffle them, then extract the label column
data = np.loadtxt("crossmatched3_combineddata1_srane.txt", delimiter = ",", skiprows = 1, usecols = (1, 10, 11, 12, 13, 14, 35, 37, 39, 41, 43, 49, 50, 51)) #dr7objid, petromag_u, petromag_g, petromag_r, pertomag_i, petromag_z, z(redshift),h alpha ew, h beta ew, OII ew, h delta ew, spiral, elliptical, uncertain

isSpiral = data[:,11]
isElliptical = data[:,12] #corresponds to the elliptical bool value

isUncertain = data[:,13]

ellipticals = data[isElliptical == 1]

spirals = data[isSpiral == 1]
uncertains = data[isUncertain == 1]

trainingSetEllipticals = ellipticals #use all of set b/c cross validation will split into training and testing sets
trainingSetSpirals = spirals

trainingSet = np.vstack((trainingSetEllipticals, trainingSetSpirals))  #using only elliptical and spiral for training
np.random.shuffle(trainingSet)
trainingSetLabels = trainingSet[:,12]  #putting labels in separate array

trainingSetLabels[trainingSetLabels == 0] = -1 #replacing all 0 with -1 to match sklearn format

trainingSet = trainingSet[:, 1:11] #removing label cols from actual inputs

trainingSet, testingSet, trainingSetLabels, testingSetLabels = train_test_split(trainingSet, trainingSetLabels, test_size = 0.6, random_state = 0) #fixes random_state so results reproducible

startTime = time.time()
print "Time before training = ", startTime

clf = theanets.Classifier((10, 5, 5, 5, 2)) #dummy values for layers for now, 1 hidden layer -- 10 inputs mapped to a binary classification output (2 choices)
clf.itertrain([trainingSet, trainingSetLabels], [testingSet, testingSetLabels], algo = 'sgd',  #theanets uses training/validation split to mean training/testing split, methinks
                learning_rate = 0.1, momentum = 0.9, hidden_l1 = 0.000001, #sparse regularizer
                input_noise = 0.1, hidden_noise = 0.1, input_dropout = 0.3, hidden_dropout = 0.3, #Dropout and Noise regularizer to prevent overfitting
                save_progress = "TN1_model_save.txt", save_every = 1000, dtype=theano.config.floatX, allow_input_downcast=True)

#print "Params after training:"
#print clf.find('layer_name', 'variable') #print params later

#Visualize weights!!! 

trainingAccuracy = clf.score(trainingSet, trainingSetLabels)

print "Training accuracy = ", trainingAccuracy

testingAccuracy = clf.score(testingSet, testingSetLabels)

print "Testing accuracy = ", testingAccuracy

print "Done training and testing! Time = ", time.time() - startTime, "seconds"

