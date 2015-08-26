'''
Theanets "Hello, world!" - train a simple neural network for classifying
simple data and evaluate the results.
[Theanets](https://github.com/lmjohns3/theanets) allows to build and train
neural networks on top of the [Theano](https://github.com/Theano/Theano)
compiler.
The goal is to get familiar with theanets on some simple example. You can
modify this example bit by bit to work on more complex data and models.
In this example we generate some synthetic data (via scikit-learn) - two 2D
blobs with Gaussian distribution which are in addition linearly separable.
Thus any classification model should have no problem with such data.
We create a neural network with three layers - input, hidden and output - each
with two dimensions (2D featues, two classes). The input and hidden layer has
by default sigmoid activation, the output clasification layer has softmax
actiovation by default. The model is trained via the stochastic gradient descent
algorithm.
Finally the model is evaluated by functions provided by scikit-learn.
'''

#Trying old dummy model with SDSS inputs

# some utilities for command line interfaces
import climate
# deep neural networks on top of Theano
import theanets
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

climate.enable_default_logging()

# -- generate some data --

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

# very simple data - two linearly separable 2D blobs
n_samples = 1000
# centers - number of classes
# n_features - dimension of the data

# convert the features and targets to the 32-bit format suitable for the model
X = X.astype(np.float32)
y = y.astype(np.int32)

# -- visualize the data for better understanding --

def plot_2d_blobs(dataset):
    X, y = dataset
    plt.axis('equal')
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.1, edgecolors='none')
    plt.show()

# plot the blobs in interactive mode
#plot_2d_blobs((X, y))

# -- split the data into training, validation and test sets --

def split_data(X, y, slices):
    '''
    Splits the data into training, validation and test sets.
    slices - relative sizes of each set (training, validation, test)
        test - provide None, since it is computed automatically
    '''
    datasets = {}
    starts = np.floor(np.cumsum(len(X) * np.hstack([0, slices[:-1]])))
    slices = {
        'training': slice(starts[0], starts[1]),
        'validation': slice(starts[1], starts[2]),
        'test': slice(starts[2], None)}
    data = X, y
    def slice_data(data, sl):
        return tuple(d[sl] for d in data)
    for label in slices:
        datasets[label] = slice_data(data, slices[label])
    return datasets

datasets = split_data(X, y, (0.6, 0.2, None))

# -- create and train the model --

# plain neural network with a single hidden layer
exp = theanets.Experiment(
    theanets.Classifier,
    # (input dimension, hidden layer size, output dimension = number of classes)
    layers=(10, 2, 2))

# train the network via stochastic gradient descent
exp.train(
    datasets['training'],
    datasets['validation'],
    optimize='sgd',
    learning_rate=0.01,
    momentum=0.5,
    hidden_l1=0.1)

# -- evaluate the model on test data --

X_test, y_test = datasets['test']
y_pred = exp.network.classify(X_test)

print('classification_report:\n', classification_report(y_test, y_pred))
print('confusion_matrix:\n', confusion_matrix(y_test, y_pred))
