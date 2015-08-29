
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
    learning_rate=0.000002,
    valid_set = ((X_valid, y_valid)),
    n_iter=5000)
print "Neural network specifications:"
print nn
nn.fit(X_train, y_train)

y_valid = nn.predict(X_valid)  #OHHHH so the predict functions are always for validation!! (?) ... *facepalm*

score = nn.score(X_test, y_test)

print "Score = ", score

print "Time = ", time.time() - startTime, "seconds"
