from __future__ import division
from math import *
import random
import numpy as np
from sklearn import svm
import astroML
from astroML.datasets import fetch_sdss_specgals
from astroML.density_estimation import KNeighborsDensity
import time
import matplotlib.pyplot as plt

#Trying to plot u-r vs. density for ellipticals
#Question = Do bluer ellipticals have lower environment densities? Does this mean their star death was slower, b/c no sudden loss of AGN heat jet?
#Higher u-r = more blue


#get just spirals and ellipticals in 1 array, shuffle them, then extract the label column
data = np.loadtxt("crossmatched3_combineddata1_srane.txt", delimiter = ",", skiprows = 1, usecols = (1, 2, 3, 10, 11, 12, 13, 14, 35, 37, 39, 41, 43, 49, 50, 51)) #dr7objid, petromag_u, petromag_g, petromag_r, pertomag_i, petromag_z, z(redshift),h alpha ew, h beta ew, OII ew, h delta ew, spiral, elliptical, uncertain

#umr = data[:, 3] - data[:, 5] #u-r color

#data = np.column_stack((data, umr))

coords = data[:, 1:3]

print coords.shape

print "started knd"
knd = KNeighborsDensity("bayesian", 10) #try using something other than 10 for n_neighbors values, to experiment + optimize
knd.fit(coords)
density = knd.eval(coords)

data[:, 1] = density #!!!!! CHECK, is this still in order??


data = np.delete(data, 2, 1) #(col# 2 , 0/1 for row/col)
print "finished knd"

isSpiral = data[:,12]#with one col removed!!
#print data[:5, :]
isElliptical = data[:,13] #corresponds to the elliptical bool value

isUncertain = data[:,14]

ellipticals = data[isElliptical == 1]
spirals = data[isSpiral == 1]
uncertains = data[isUncertain == 1]

hAlpha = spirals[:, 8] #halpa equivalent width, keep in mind 1 col of array has been removed

density = spirals[:, 1]
plt.scatter(hAlpha, density)
plt.xlabel("Equivalent width of H-alpha emission line")
plt.ylabel("Environment density")
plt.title("Spiral Galaxies as identified by Galaxy Zoo Volunteers")
plt.show()



##trainingSetEllipticals = ellipticals[:50000] #check whether these numbers are inclusive
##trainingSetSpirals = spirals[:50000] #extracting first 5000 spiral and elliptical to train model, excluding last 3 cols (labels)
##
##trainingSet = np.vstack((trainingSetEllipticals, trainingSetSpirals))  #using only elliptical and spiral for training
##np.random.shuffle(trainingSet)
##trainingSetLabels = trainingSet[:,13]  #putting labels in separate array
##
##trainingSetLabels[trainingSetLabels == 0] = -1 #replacing all 0 with -1 to match sklearn format
##counter = 0
##for i in trainingSetLabels:
##    if i == -1:
##        counter +=1
##print counter
##trainingSet = trainingSet[:, 1:12] #removing label cols from actual inputs
##startTime = time.time()
##print "Time before training = ", startTime
##
##clf = svm.SVC() 
##clf_info = clf.fit(trainingSet, trainingSetLabels)
##print clf_info
##
##print "Done training! Time = ", time.time() - startTime, "seconds"
###Training accuracy
##
###prediction = clf.predict(trainingSetEllipticals)
##
##trainingAccuracy = clf.score(trainingSet, trainingSetLabels)
##    
##print "Training accuracy = ", trainingAccuracy
##print "Time = ", time.time() - startTime, "seconds"
##print
##
###Testing accuracy
##testingSetEllipticals = ellipticals[50000:100000]
##testingSetSpirals = spirals[50000:100000]
##
##testingSet = np.vstack((testingSetEllipticals, testingSetSpirals))  #using only elliptical and spiral for training
##np.random.shuffle(testingSet)
##testingSetLabels = testingSet[:,13]  #putting labels in separate array
##testingSetLabels[testingSetLabels == 0] = -1 #replacing all 0 with -1 to match sklearn format
##testingSet = testingSet[:, 1:12] #removing label cols from actual inputs
##
##testingAccuracy = clf.score(testingSet, testingSetLabels)
##print "Testing accuracy = ", testingAccuracy
##print "Time = ", time.time() - startTime, "seconds"
##
##classificationSet = uncertains[:, 1:12]
##predictions = clf.predict(classificationSet) #use model to classify uncertains - SEE IF YOU CAN FIND CONFIDENCE INTERVAL FOR THIS PREDICTION AFTERWARD, THAT WILL BE PROGRESS
##
##uncertainsPredictions = np.column_stack((uncertains[:, 0:12], predictions)) #hstack doesn't work here, b/c multidimensional array?
##
##f = open('uncertainsPredictionsOutputDensity2.txt', 'w') 
##
##uncertainsPredictions.tofile(f, sep=",", format="%f") #export array to file in floating point values, comma separated
##print "Finished writing predictions to file!"

#FORMAT OF OUTPUT FILE: 
