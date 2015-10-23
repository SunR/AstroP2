Python 2.7.10 (default, May 23 2015, 09:44:00) [MSC v.1500 64 bit (AMD64)] on win32
Type "copyright", "credits" or "license()" for more information.
>>> ================================ RESTART ================================
>>> 
Neural Network Classifer
Using combined1_half1_test3.txt and combined1_half2_test3.txt
stacked files!
Neural network specifications:
Classifier(batch_size=1, debug=False, dropout_rate=None, f_stable=0.001,
      hidden0=<sknn.nn.Layer `Sigmoid`: name=u'hidden0', units=100>,
      layers=[<sknn.nn.Layer `Sigmoid`: name=u'hidden0', units=100>, <sknn.nn.Layer `Softmax`: name=u'output'>],
      learning_momentum=0.9, learning_rate=0.00018, learning_rule=u'sgd',
      loss_type=u'mse', mutator=None, n_iter=3000, n_stable=50,
      output=<sknn.nn.Layer `Softmax`: name=u'output'>, random_state=None,
      regularize=None,
      valid_set=(array([[ 2.31849,  2.22621, ...,  1.02229, -2.01987],
       [ 5.05157,  4.26333, ...,  0.68753, -1.68628],
       ...,
       [ 2.91442,  2.94063, ..., -1.61419, -2.5228 ],
       [ 2.07025,  2.09384, ...,  1.57891, -5.00864]]), array([-1., -1., ..., -1., -1.])),
      valid_size=0.0, verbose=None, weight_decay=None)
I 2015-09-12 01:42:52 sknn:198 Initializing neural network with 2 layers, 29 inputs and 2 outputs.
I 2015-09-12 01:42:53 sknn:129 Training on dataset of 100,000 samples with 3,100,000 total size.
I 2015-09-12 03:33:52 sknn:478 Early termination condition fired at 186 iterations.
Training accuracy =  0.9242
Validation accuracy =  1.0
Testing accuracy =  0.922858836313
Time =  6684.41599989 seconds
>>> prob = nn.predict_proba(X_test)
>>> tpr, fpr, thresh = metrics.roc_curve(y, prob[:, 0])

Traceback (most recent call last):
  File "<pyshell#1>", line 1, in <module>
    tpr, fpr, thresh = metrics.roc_curve(y, prob[:, 0])
NameError: name 'metrics' is not defined
>>> from sklearn import metrics
>>> tpr, fpr, thresh = metrics.roc_curve(y, prob[:, 0])

Traceback (most recent call last):
  File "<pyshell#3>", line 1, in <module>
    tpr, fpr, thresh = metrics.roc_curve(y, prob[:, 0])
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\sklearn\metrics\ranking.py", line 477, in roc_curve
    y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\sklearn\metrics\ranking.py", line 283, in _binary_clf_curve
    check_consistent_length(y_true, y_score)
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\sklearn\utils\validation.py", line 174, in check_consistent_length
    "%s" % str(uniques))
ValueError: Found arrays with inconsistent numbers of samples: [101230 251230]
>>> y.shape
(251230L,)
>>> prob.shape
(101230L, 2L)
>>> y_test.shape
(101230L,)
>>> tpr, fpr, thresh = metrics.roc_curve(y_test, prob[:, 0])
>>> import matplotlib.pyplot as plt
>>> plt.plot(tpr, fpr)
[<matplotlib.lines.Line2D object at 0x0000000016C6D0F0>]
>>> plt.show()
>>> plt.plot(fpr, tpr)
[<matplotlib.lines.Line2D object at 0x0000000025710EF0>]
>>> plt.plot(fpr, tpr, x_label = "False positive rate", y_label = "True positive rate", title = "Neural Network Classifier")

Traceback (most recent call last):
  File "<pyshell#12>", line 1, in <module>
    plt.plot(fpr, tpr, x_label = "False positive rate", y_label = "True positive rate", title = "Neural Network Classifier")
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\matplotlib\pyplot.py", line 3099, in plot
    ret = ax.plot(*args, **kwargs)
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\matplotlib\axes\_axes.py", line 1373, in plot
    for line in self._get_lines(*args, **kwargs):
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\matplotlib\axes\_base.py", line 304, in _grab_next_args
    for seg in self._plot_args(remaining, kwargs):
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\matplotlib\axes\_base.py", line 292, in _plot_args
    seg = func(x[:, j % ncx], y[:, j % ncy], kw, kwargs)
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\matplotlib\axes\_base.py", line 244, in _makeline
    self.set_lineprops(seg, **kwargs)
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\matplotlib\axes\_base.py", line 184, in set_lineprops
    raise TypeError('There is no line property "%s"' % key)
TypeError: There is no line property "x_label"
>>> plt.xlabel("False positive rate")
<matplotlib.text.Text object at 0x0000000017300A90>
>>> plt.ylabels("True positive rate")

Traceback (most recent call last):
  File "<pyshell#14>", line 1, in <module>
    plt.ylabels("True positive rate")
AttributeError: 'module' object has no attribute 'ylabels'
>>> plt.ylabel("True positive rate")
<matplotlib.text.Text object at 0x000000002563F0F0>
>>> plt.title("ROC: Neural Network Classifier")
<matplotlib.text.Text object at 0x00000000256E5518>
>>> plt.show()
>>> plt.title("ROC Curve: Neural Network Classifier")
<matplotlib.text.Text object at 0x0000000031F7A400>
>>> plt.show()
>>> plt.plot(tpr, fpr)
[<matplotlib.lines.Line2D object at 0x0000000015D8CEF0>]
>>> plt.xlabel("False positive rate")
<matplotlib.text.Text object at 0x0000000031F58908>
>>> plt.ylabel("True positive rate")
<matplotlib.text.Text object at 0x00000000316C2518>
>>> plt.title("ROC Curve: Neural Network Classifier")
<matplotlib.text.Text object at 0x0000000015D5F438>
>>> plt.show()
>>> plt.plot(fpr, tpr)
[<matplotlib.lines.Line2D object at 0x00000000318E41D0>]
>>> plt.xlabel("False positive rate")
<matplotlib.text.Text object at 0x000000002565A550>
>>> plt.ylabel("True positive rate")
<matplotlib.text.Text object at 0x000000001A4127F0>
>>> plt.title("ROC Curve: Neural Network Classifier")
<matplotlib.text.Text object at 0x000000001A4456D8>
>>> plt.show()
>>> proba

Traceback (most recent call last):
  File "<pyshell#30>", line 1, in <module>
    proba
NameError: name 'proba' is not defined
>>> prob
array([[  9.99751213e-01,   2.48787208e-04],
       [  2.31510335e-01,   7.68489665e-01],
       [  8.54931525e-03,   9.91450685e-01],
       ..., 
       [  9.99660131e-01,   3.39869171e-04],
       [  9.88884244e-01,   1.11157564e-02],
       [  8.39936860e-01,   1.60063140e-01]])
>>> ================================ RESTART ================================
>>> 
Using combined1_half1_test3.txt and combined1_half2_test3.txt
stacked files!
Time =  14.0709998608 seconds
Random Forest Classifier
Params after training:
{'warm_start': False, 'oob_score': False, 'n_jobs': 1, 'verbose': 0, 'max_leaf_nodes': None, 'bootstrap': True, 'min_samples_leaf': 1, 'n_estimators': 10, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'criterion': 'gini', 'random_state': None, 'max_features': 'auto', 'max_depth': None, 'class_weight': None}
Training accuracy =  0.997935485241
Testing accuracy =  0.963062030315

Traceback (most recent call last):
  File "C:/Users/S/Documents/Sonu/SciComp/Research/AstroP2/allclassifiers1.py", line 81, in <module>
    tprRF, fprRF, threshRF = metrics.roc_curve(testingSetLabels, probRF[:, 0]) #true positive rate, false positive rate (ROC curve)
NameError: name 'metrics' is not defined
>>> ================================ RESTART ================================
>>> 
Using combined1_half1_test3.txt and combined1_half2_test3.txt
stacked files!
Time =  13.8819999695 seconds
Random Forest Classifier
Params after training:
{'warm_start': False, 'oob_score': False, 'n_jobs': 1, 'verbose': 0, 'max_leaf_nodes': None, 'bootstrap': True, 'min_samples_leaf': 1, 'n_estimators': 10, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'criterion': 'gini', 'random_state': None, 'max_features': 'auto', 'max_depth': None, 'class_weight': None}
Training accuracy =  0.997898334589
Testing accuracy =  0.964033244173

Traceback (most recent call last):
  File "C:/Users/S/Documents/Sonu/SciComp/Research/AstroP2/allclassifiers1.py", line 84, in <module>
    tprRF, fprRF, threshRF = metrics.roc_curve(testingSetLabels, probRF[:, 0]) #true positive rate, false positive rate (ROC curve)
NameError: name 'metrics' is not defined
>>> ================================ RESTART ================================
>>> 
Using combined1_half1_test3.txt and combined1_half2_test3.txt
stacked files!
Time =  13.9330000877 seconds
Random Forest Classifier
Params after training:
{'warm_start': False, 'oob_score': False, 'n_jobs': 1, 'verbose': 0, 'max_leaf_nodes': None, 'bootstrap': True, 'min_samples_leaf': 1, 'n_estimators': 10, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'criterion': 'gini', 'random_state': None, 'max_features': 'auto', 'max_depth': None, 'class_weight': None}
Training accuracy =  0.997824033287
Testing accuracy =  0.964049165711
Time =  14.2849998474 seconds
Neural Network Classifer
Neural network specifications:
Classifier(batch_size=1, debug=False, dropout_rate=None, f_stable=0.001,
      hidden0=<sknn.nn.Layer `Sigmoid`: name=u'hidden0', units=100>,
      layers=[<sknn.nn.Layer `Sigmoid`: name=u'hidden0', units=100>, <sknn.nn.Layer `Softmax`: name=u'output'>],
      learning_momentum=0.9, learning_rate=0.00018, learning_rule=u'sgd',
      loss_type=u'mse', mutator=None, n_iter=1000, n_stable=50,
      output=<sknn.nn.Layer `Softmax`: name=u'output'>, random_state=None,
      regularize=None, valid_set=None, valid_size=0.0, verbose=None,
      weight_decay=None)
I 2015-09-13 09:35:57 sknn:198 Initializing neural network with 2 layers, 29 inputs and 2 outputs.
I 2015-09-13 09:35:57 sknn:129 Training on dataset of 188,422 samples with 5,841,082 total size.

>>> ================================ RESTART ================================
>>> 
Using combined1_half1_test3.txt and combined1_half2_test3.txt
stacked files!
Time =  14.7960000038 seconds


Random Forest Classifier
Params after training:
{'warm_start': False, 'oob_score': False, 'n_jobs': 1, 'verbose': 0, 'max_leaf_nodes': None, 'bootstrap': True, 'min_samples_leaf': 1, 'n_estimators': 10, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'criterion': 'gini', 'random_state': None, 'max_features': 'auto', 'max_depth': None, 'class_weight': None}
Training accuracy =  0.997919563533
Testing accuracy =  0.963014265699
Time =  14.6009998322 seconds


Neural Network Classifer
Neural network specifications:
Classifier(batch_size=1, debug=False, dropout_rate=None, f_stable=0.001,
      hidden0=<sknn.nn.Layer `Sigmoid`: name=u'hidden0', units=100>,
      layers=[<sknn.nn.Layer `Sigmoid`: name=u'hidden0', units=100>, <sknn.nn.Layer `Softmax`: name=u'output'>],
      learning_momentum=0.9, learning_rate=0.00018, learning_rule=u'sgd',
      loss_type=u'mse', mutator=None, n_iter=1000, n_stable=50,
      output=<sknn.nn.Layer `Softmax`: name=u'output'>, random_state=None,
      regularize=None, valid_set=None, valid_size=0.0, verbose=None,
      weight_decay=None)
I 2015-09-13 09:38:11 sknn:198 Initializing neural network with 2 layers, 29 inputs and 2 outputs.
I 2015-09-13 09:38:12 sknn:129 Training on dataset of 188,422 samples with 5,841,082 total size.
I 2015-09-13 20:40:21 sknn:482 Terminating after specified 1000 total iterations.
Training accuracy =  0.945935188035
Testing accuracy =  0.941345051586
Time =  39738.4619999 seconds


Decision Tree Classifier
Params after training:
{'splitter': 'best', 'max_leaf_nodes': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'criterion': 'gini', 'random_state': None, 'max_features': None, 'max_depth': 10, 'class_weight': None}
Training accuracy =  0.953805818853
Testing accuracy =  0.943940262387
Time =  8.11099982262 seconds


Support Vector Machine Classifier
SVC(C=100, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=1.0,
  kernel='rbf', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False)
Training accuracy =  1.0
Testing accuracy =  0.83349254872

Traceback (most recent call last):
  File "C:/Users/S/Documents/Sonu/SciComp/Research/AstroP2/allclassifiers1.py", line 173, in <module>
    probSVM = clf.predict_proba(testingSet)
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\sklearn\svm\base.py", line 542, in predict_proba
    self._check_proba()
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\sklearn\svm\base.py", line 510, in _check_proba
    " probability=%r" % self.probability)
AttributeError: predict_proba is not available when probability=False
>>> ================================ RESTART ================================
>>> 
Using combined1_half1_test3.txt and combined1_half2_test3.txt
stacked files!
Time =  14.5440001488 seconds


Random Forest Classifier
Params after training:
{'warm_start': False, 'oob_score': False, 'n_jobs': 1, 'verbose': 0, 'max_leaf_nodes': None, 'bootstrap': True, 'min_samples_leaf': 1, 'n_estimators': 10, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'criterion': 'gini', 'random_state': None, 'max_features': 'auto', 'max_depth': None, 'class_weight': None}
Training accuracy =  0.997935485241
Testing accuracy =  0.964526811871
Time =  14.0770001411 seconds


Decision Tree Classifier
Params after training:
{'splitter': 'best', 'max_leaf_nodes': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'criterion': 'gini', 'random_state': None, 'max_features': None, 'max_depth': 10, 'class_weight': None}
Training accuracy =  0.952770907856
Testing accuracy =  0.944386065469
Time =  7.60600018501 seconds


Support Vector Machine Classifier
SVC(C=100, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=1.0,
  kernel='rbf', max_iter=-1, probability=True, random_state=None,
  shrinking=True, tol=0.001, verbose=False)
Training accuracy =  1.0
Testing accuracy =  0.832123296395
Time =  114051.42 seconds


K-Neighbors Classifier
Params after training:
{'n_neighbors': 5, 'algorithm': 'auto', 'metric': 'minkowski', 'metric_params': None, 'p': 2, 'weights': 'uniform', 'leaf_size': 30}
Training accuracy =  0.93175425375
Testing accuracy =  0.903547318813
Time =  275.842999935 seconds


Gaussian Naive - Bayes Classifier
Params after training:
{}
Training accuracy =  0.833506703039
Testing accuracy =  0.830371927143
Time =  0.97000002861 seconds


Neural Network Classifer
Neural network specifications:
Classifier(batch_size=1, debug=False, dropout_rate=None, f_stable=0.001,
      hidden0=<sknn.nn.Layer `Sigmoid`: name=u'hidden0', units=100>,
      layers=[<sknn.nn.Layer `Sigmoid`: name=u'hidden0', units=100>, <sknn.nn.Layer `Softmax`: name=u'output'>],
      learning_momentum=0.9, learning_rate=0.00018, learning_rule=u'sgd',
      loss_type=u'mse', mutator=None, n_iter=1000, n_stable=50,
      output=<sknn.nn.Layer `Softmax`: name=u'output'>, random_state=None,
      regularize=None, valid_set=None, valid_size=0.0, verbose=None,
      weight_decay=None)
I 2015-09-15 09:36:06 sknn:198 Initializing neural network with 2 layers, 29 inputs and 2 outputs.
I 2015-09-15 09:36:06 sknn:129 Training on dataset of 188,422 samples with 5,841,082 total size.
I 2015-09-15 20:35:07 sknn:482 Terminating after specified 1000 total iterations.
Training accuracy =  0.939778794408
Testing accuracy =  0.938351802318
Time =  39549.506 seconds



Traceback (most recent call last):
  File "C:/Users/S/Documents/Sonu/SciComp/Research/AstroP2/allclassifiers1.py", line 244, in <module>
    plt.xlabel("False positive rate")
NameError: name 'RF' is not defined
>>> plt.legend(["RF", "NN", "DT", "SVM", "KNN", "NB"], loc = "bottom right")

Warning (from warnings module):
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\matplotlib\legend.py", line 319
    % (loc, '\n\t'.join(six.iterkeys(self.codes))))
UserWarning: Unrecognized location "bottom right". Falling back on "best"; valid locations are
	right
	center left
	upper right
	lower right
	best
	center
	lower left
	center right
	upper left
	upper center
	lower center

<matplotlib.legend.Legend object at 0x00000000525D0EB8>
>>> plt.legend(["RF", "NN", "DT", "SVM", "KNN", "NB"], loc = "lower right")
<matplotlib.legend.Legend object at 0x000000005C77C8D0>
>>> plt.xlabel("False positive rate")
<matplotlib.text.Text object at 0x0000000001E33FD0>
>>> plt.ylabel("True positive rate")
<matplotlib.text.Text object at 0x0000000054561B00>
>>> plt.title("ROC Curve for Various Classifiers")
<matplotlib.text.Text object at 0x000000005257DC18>
>>> plt.show()
>>> clf_info = SVC(C=100, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=1.0,
  kernel='rbf', max_iter=-1, probability=True, random_state=None,
  shrinking=True, tol=0.001, verbose=False)

Traceback (most recent call last):
  File "<pyshell#38>", line 1, in <module>
    clf_info = SVC(C=100, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=1.0,
NameError: name 'SVC' is not defined
>>> clf.get_params()
{}
>>> clf
GaussianNB()
>>> clf_info
SVC(C=100, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=1.0,
  kernel='rbf', max_iter=-1, probability=True, random_state=None,
  shrinking=True, tol=0.001, verbose=False)
>>> ================================ RESTART ================================
>>> 
Using combined1_half1_test3.txt and combined1_half2_test3.txt
stacked files!
Time =  14.7880001068 seconds


Random Forest Classifier
Params after training:
{'warm_start': False, 'oob_score': False, 'n_jobs': 1, 'verbose': 0, 'max_leaf_nodes': None, 'bootstrap': True, 'min_samples_leaf': 1, 'n_estimators': 10, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'criterion': 'gini', 'random_state': None, 'max_features': 'auto', 'max_depth': None, 'class_weight': None}
Training accuracy =  0.997866491174
Testing accuracy =  0.964701948796
Time =  15.1920001507 seconds


Decision Tree Classifier
Params after training:
{'splitter': 'best', 'max_leaf_nodes': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'criterion': 'gini', 'random_state': None, 'max_features': None, 'max_depth': 10, 'class_weight': None}
Training accuracy =  0.95255861842
Testing accuracy =  0.942905362374
Time =  8.21099996567 seconds


K-Neighbors Classifier
Params after training:
{'n_neighbors': 5, 'algorithm': 'auto', 'metric': 'minkowski', 'metric_params': None, 'p': 2, 'weights': 'uniform', 'leaf_size': 30}
Training accuracy =  0.931557886022
Testing accuracy =  0.902878614189
Time =  279.154999971 seconds


Gaussian Naive - Bayes Classifier
Params after training:
{}
Training accuracy =  0.698729447729
Testing accuracy =  0.699082919373
Time =  0.486999988556 seconds


Neural Network Classifer
Neural network specifications:
Classifier(batch_size=1, debug=False, dropout_rate=None, f_stable=0.001,
      hidden0=<sknn.nn.Layer `Sigmoid`: name=u'hidden0', units=100>,
      layers=[<sknn.nn.Layer `Sigmoid`: name=u'hidden0', units=100>, <sknn.nn.Layer `Softmax`: name=u'output'>],
      learning_momentum=0.9, learning_rate=0.00018, learning_rule=u'sgd',
      loss_type=u'mse', mutator=None, n_iter=1000, n_stable=50,
      output=<sknn.nn.Layer `Softmax`: name=u'output'>, random_state=None,
      regularize=None, valid_set=None, valid_size=0.0, verbose=None,
      weight_decay=None)
I 2015-09-15 21:18:57 sknn:198 Initializing neural network with 2 layers, 29 inputs and 2 outputs.
I 2015-09-15 21:18:58 sknn:129 Training on dataset of 188,422 samples with 5,841,082 total size.
I 2015-09-16 08:23:39 sknn:482 Terminating after specified 1000 total iterations.
Training accuracy =  0.922519663309
Testing accuracy =  0.923067125207
Time =  39892.494 seconds


Support Vector Machine Classifier
SVC(C=100, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.1,
  kernel='rbf', max_iter=-1, probability=True, random_state=None,
  shrinking=True, tol=0.001, verbose=False)
Training accuracy =  0.99490505355
Testing accuracy =  0.934212202267
Time =  78273.7289999 seconds


>>> 