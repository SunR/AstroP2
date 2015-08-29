
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

climate.enable_default_logging()

startTime = time.time()

data = np.loadtxt("crossmatched3_combineddata1_srane.txt", delimiter = ",", skiprows = 1, usecols = (1, 10, 11, 12, 13, 14, 35, 37, 39, 41, 43, 49, 50, 51)) #dr7objid, petromag_u, petromag_g, petromag_r, pertomag_i, petromag_z, z(redshift),h alpha ew, h beta ew, OII ew, h delta ew, spiral, elliptical, uncertain

isSpiral = data[:,11]
isElliptical = data[:,12] #corresponds to the elliptical bool value

isUncertain = data[:,13]

ellipticals = data[isElliptical == 1]

spirals = data[isSpiral == 1]
uncertains = data[isUncertain == 1]

trainingSetEllipticals = ellipticals[:5000] #use all of set b/c cross validation will split into training and testing sets
trainingSetSpirals = spirals[:5000]

trainingSet = np.vstack((trainingSetEllipticals, trainingSetSpirals))  #using only elliptical and spiral for training
np.random.shuffle(trainingSet)
trainingSetLabels = trainingSet[:,12]  #putting labels in separate array

trainingSetLabels[trainingSetLabels == 0] = -1 #replacing all 0 with -1 to match sklearn format

trainingSet = trainingSet[:, 1:11] #removing label cols from actual inputs

X = trainingSet
y = trainingSetLabels

# centers - number of classes
# n_features - dimension of the data

# convert the features and targets to the 32-bit format suitable for the model
X = X.astype(np.float32)
y = y.astype(np.int32)

# -- split the data into training, validation and test sets --

def split_data(X, y, slices):

    #Splits the data into training, validation and test sets.

    datasets = {}
    starts = np.floor(np.cumsum(len(X) * np.hstack([0, slices[:-1]])))
    slices = {
        'training': slice(starts[0], starts[1]),
        'validation': slice(starts[1], starts[2]),
        'test': slice(starts[2], None)}
    data = X, y
    def slice_data(data, sli):
        return tuple(d[sli] for d in data)
    for label in slices:
        datasets[label] = slice_data(data, slices[label])
    return datasets

datasets = {}
datasets['training'] = (X[:6000], y[:6000])
datasets['validation'] = (X[6000:8000], y[6000:8000])
datasets['test'] = (X[8000:10000], y[8000:10000])

print datasets['training'][0].shape, datasets['validation'][0].shape

# plain neural network with a single hidden layer
exp = theanets.Experiment(
    theanets.Classifier,
    # (input dimension, hidden layer size, output dimension = number of classes)
    layers=(10, 10, 10, 5, 5, 5, 2))

# train the network - stochastic gradient descent
exp.train(
    datasets['training'],
    datasets['validation'],
    optimize='sgd',
    min_improvement = 0.005,
    learning_rate=0.01,
    momentum=0.5,
    hidden_l1=0.5)

print "Done training! Time = ", time.time()-startTime, "seconds"
#evaluate the model on test data

X_test, y_test = datasets['test']
y_pred = exp.network.classify(X_test)

print('classification_report:\n', classification_report(y_test, y_pred))
print('confusion_matrix:\n', confusion_matrix(y_test, y_pred))
