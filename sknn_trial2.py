
#Trying old dummy model with SDSS inputs

# some utilities for command line interfaces
import climate
# deep neural networks on top of Theano
import theanets
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import time

from sknn.mlp import Classifier, Layer



#---------------

climate.enable_default_logging()

startTime = time.time()

print "Neural Network Classifer"
print "Using combined1_half1_test3.txt and combined1_half2_test3.txt"

#get just spirals and ellipticals in 1 array, shuffle them, then extract the label column
data = np.loadtxt("combined1_half1_test3.txt", delimiter = ",", skiprows = 1, usecols = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36)) #58, 

data2 = np.loadtxt("combined1_half2_test3.txt", delimiter = ",", skiprows = 1, usecols = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36)) #58, 

data = np.vstack((data, data2)) #combined two split csv files of data

print "stacked files!"

isSpiral = data[:,30]
isElliptical = data[:,31] #corresponds to the elliptical bool value

isUncertain = data[:,32]

ellipticals = data[isElliptical == 1]

spirals = data[isSpiral == 1]
uncertains = data[isUncertain == 1]

trainingSetEllipticals = ellipticals #use all of set b/c cross validation will split into training and testing sets
trainingSetSpirals = spirals

trainingSet = np.vstack((trainingSetEllipticals, trainingSetSpirals))  #using only elliptical and spiral for training
np.random.shuffle(trainingSet)
trainingSetLabels = trainingSet[:,31]  #putting labels in separate array

trainingSetLabels[trainingSetLabels == 0] = -1 #replacing all 0 with -1 to match sklearn format

trainingSet = trainingSet[:, 1:30] #removing label cols from actual inputs

X = trainingSet
y = trainingSetLabels

X_train = X[:100000]
y_train = y[:100000]
X_valid = X[100000:150000]
y_valid = y[100000:150000]
X_test = X[150000:]
y_test = y[150000:]

nn = Classifier(
    layers=[
        Layer("Sigmoid", units=100),
        Layer("Softmax")],
    learning_rate=0.00018,  #valid_set = ((X_valid, y_valid))
    n_iter=3000,
    valid_set = (X_valid, y_valid))
print "Neural network specifications:"
print nn

nn.fit(X_train, y_train)

y_valid = nn.predict(X_valid)  #OHHHH so the predict functions are always for validation!! (?) ... *facepalm*

score1 = nn.score(X_train, y_train)

score2 = nn.score(X_valid, y_valid)

score3 = nn.score(X_test, y_test)

print "Training accuracy = ", score1

print "Validation accuracy = ", score2

print "Testing accuracy = ", score3

print "Time = ", time.time() - startTime, "seconds"
