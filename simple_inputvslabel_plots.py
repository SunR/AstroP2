

# some utilities for command line interfaces
import climate
# deep neural networks on top of Theano
import theanets
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.grid_search import GridSearchCV
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from astroML.density_estimation import KNeighborsDensity
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import time

from sknn.mlp import Classifier, Layer

#-----------------

climate.enable_default_logging()

startTime = time.time()

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
