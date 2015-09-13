
#Trying old dummy model with SDSS inputs

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

#---------------

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

trainingSet = np.vstack((trainingSetEllipticals, trainingSetSpirals))  #using only elliptical and spiral for training
np.random.shuffle(trainingSet)
trainingSetLabels = trainingSet[:,31]  #putting labels in separate array

trainingSetLabels[trainingSetLabels == 0] = -1 #replacing all 0 with -1 to match sklearn format

trainingSet = trainingSet[:, 1:30] #removing label cols from actual inputs

trainingSet, testingSet, trainingSetLabels, testingSetLabels = train_test_split(trainingSet, trainingSetLabels, test_size = 0.25, random_state = 0)

print "Time = ", time.time() - startTime, "seconds"

startTime = time.time()

print
print

#-----------------------RF---------------------------
print "Random Forest Classifier"
clf = RandomForestClassifier() #No max depth initial, tweak as necessary later
clf = clf.fit(trainingSet, trainingSetLabels)

print "Params after training:"
print clf.get_params()

score1 = clf.score(trainingSet, trainingSetLabels)

score3 = clf.score(testingSet, testingSetLabels)

print "Training accuracy = ", score1

print "Testing accuracy = ", score3

probRF = clf.predict_proba(testingSet)
tprRF, fprRF, threshRF = metrics.roc_curve(testingSetLabels, probRF[:, 0]) #true positive rate, false positive rate (ROC curve)

print "Time = ", time.time() - startTime, "seconds"

startTime = time.time()

print
print

#------------------------NN---------------------------
print "Neural Network Classifer"

nn = Classifier(
    layers=[
        Layer("Sigmoid", units=100),
        Layer("Softmax")],
    learning_rate=0.00018,  #valid_set = ((X_valid, y_valid))
    n_iter=1000)
print "Neural network specifications:"
print nn

nn.fit(trainingSet, trainingSetLabels)

score1 = nn.score(trainingSet, trainingSetLabels)

score3 = nn.score(testingSet, testingSetLabels)

print "Training accuracy = ", score1

print "Testing accuracy = ", score3

probNN = nn.predict_proba(testingSet)
tprNN, fprNN, threshNN = metrics.roc_curve(testingSetLabels, probNN[:, 0]) #true positive rate, false positive rate (ROC curve)

print "Time = ", time.time() - startTime, "seconds"

startTime = time.time()

print
print


#------------------------DT-----------------------------
print "Decision Tree Classifier"

clf = tree.DecisionTreeClassifier(max_depth = 10)
clf = clf.fit(trainingSet, trainingSetLabels)

print "Params after training:"
print clf.get_params()  
tree.export_graphviz(clf, out_file="tree2.dot")

score1 = clf.score(trainingSet, trainingSetLabels)

score3 = clf.score(testingSet, testingSetLabels)

print "Training accuracy = ", score1

print "Testing accuracy = ", score3

probDT = clf.predict_proba(testingSet)
tprDT, fprDT, threshDT = metrics.roc_curve(testingSetLabels, probDT[:, 0]) #true positive rate, false positive rate (ROC curve)

print "Time = ", time.time() - startTime, "seconds"

startTime = time.time()

print
print

#------------------------SVM----------------------------
print "Support Vector Machine Classifier"

clf = svm.SVC(C = 100, gamma = 1.0) 
clf_info = clf.fit(trainingSet, trainingSetLabels)
print clf_info

score1 = clf.score(trainingSet, trainingSetLabels)

score3 = clf.score(testingSet, testingSetLabels)

print "Training accuracy = ", score1

print "Testing accuracy = ", score3

probSVM = clf.predict_proba(testingSet)
tprSVM, fprSVM, threshSVM = metrics.roc_curve(testingSetLabels, probSVM[:, 0]) #true positive rate, false positive rate (ROC curve)

print "Time = ", time.time() - startTime, "seconds"

startTime = time.time()

print
print

#------------------------KNN----------------------------
print "K-Neighbors Classifier"
clf = KNeighborsClassifier(n_neighbors = 5) #starting off with 5 neighbors for now
clf = clf.fit(trainingSet, trainingSetLabels)

print "Params after training:"
print clf.get_params()

score1 = clf.score(trainingSet, trainingSetLabels)

score3 = clf.score(testingSet, testingSetLabels)

print "Training accuracy = ", score1

print "Testing accuracy = ", score3

probKNN = clf.predict_proba(testingSet)
tprKNN, fprKNN, threshKNN = metrics.roc_curve(testingSetLabels, probKNN[:, 0]) #true positive rate, false positive rate (ROC curve)

print "Time = ", time.time() - startTime, "seconds"

startTime = time.time()

print
print

#------------------------NB-----------------------------
print "Gaussian Naive - Bayes Classifier"

clf = GaussianNB()
clf = clf.fit(trainingSet, trainingSetLabels)

print "Params after training:"
print clf.get_params()

score1 = clf.score(trainingSet, trainingSetLabels)

score3 = clf.score(testingSet, testingSetLabels)

print "Training accuracy = ", score1

print "Testing accuracy = ", score3

probNB = clf.predict_proba(testingSet)
tprNB, fprNB, threshNB = metrics.roc_curve(testingSetLabels, probNB[:, 0]) #true positive rate, false positive rate (ROC curve)

print "Time = ", time.time() - startTime, "seconds"

startTime = time.time()

print
print
#----------------------ROC Curve Plot-------------------
plt.plot(fprRF, tprRF)
plt.plot(fprNN, tprNN)
plt.plot(fprDT, tprDT)
plt.plot(fprSVM, tprSVM)
plt.plot(fprKNN, tprKNN)
plt.plot(fprNB, tprNB)

plt.legend([RF, NN, DT, SVM, KNN, NB], loc = "bottom right")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC Curve for Various Classifiers")
plt.show()

