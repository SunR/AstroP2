Python 2.7.10 (default, May 23 2015, 09:44:00) [MSC v.1500 64 bit (AMD64)] on win32
Type "copyright", "credits" or "license()" for more information.
>>> ================================ RESTART ================================
>>> 

Traceback (most recent call last):
  File "C:\Users\S\Documents\Sonu\SciComp\Research\AstroP2\ap2_sklearn_svmSCV13_combineddata1.py", line 8, in <module>
    from astroML.metrics import classification_report
ImportError: No module named metrics
>>> ================================ RESTART ================================
>>> 

Traceback (most recent call last):
  File "C:\Users\S\Documents\Sonu\SciComp\Research\AstroP2\ap2_sklearn_svmSCV13_combineddata1.py", line 43, in <module>
    tuned_parameters = [{'kernel':['rbf'], 'gamma':[0, 1e-10, 1e-5, 0.1, 1], 'C':[1, 10, 100, 1000, 10000]}, {'kernel':[linear], 'C':[1, 10, 100, 1000, 10000]}] #try 2 diff kernels, w/ many diff params, and optimize
NameError: name 'linear' is not defined
>>> ================================ RESTART ================================
>>> 
Tuning hyper-parameters for precision


Traceback (most recent call last):
  File "C:\Users\S\Documents\Sonu\SciComp\Research\AstroP2\ap2_sklearn_svmSCV13_combineddata1.py", line 50, in <module>
    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv = 10, scoring = score) #cv = #of folds of cross-validation
NameError: name 'SVC' is not defined
>>> ================================ RESTART ================================
>>> 
Tuning hyper-parameters for precision


Traceback (most recent call last):
  File "C:\Users\S\Documents\Sonu\SciComp\Research\AstroP2\ap2_sklearn_svmSCV13_combineddata1.py", line 51, in <module>
    clf.fit(trainingSet, trainingSetLabels)
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\sklearn\grid_search.py", line 732, in fit
    return self._fit(X, y, ParameterGrid(self.param_grid))
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\sklearn\grid_search.py", line 477, in _fit
    X, y = indexable(X, y)
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\sklearn\utils\validation.py", line 199, in indexable
    check_consistent_length(*result)
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\sklearn\utils\validation.py", line 174, in check_consistent_length
    "%s" % str(uniques))
ValueError: Found arrays with inconsistent numbers of samples: [100482 150724]
>>> len(trainingSet)
100482
>>> len(trainingSetLabels)
150724
>>> ================================ RESTART ================================
>>> 
251206 251206
Tuning hyper-parameters for precision


Traceback (most recent call last):
  File "C:\Users\S\Documents\Sonu\SciComp\Research\AstroP2\ap2_sklearn_svmSCV13_combineddata1.py", line 53, in <module>
    clf.fit(trainingSet, trainingSetLabels)
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\sklearn\grid_search.py", line 732, in fit
    return self._fit(X, y, ParameterGrid(self.param_grid))
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\sklearn\grid_search.py", line 477, in _fit
    X, y = indexable(X, y)
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\sklearn\utils\validation.py", line 199, in indexable
    check_consistent_length(*result)
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\sklearn\utils\validation.py", line 174, in check_consistent_length
    "%s" % str(uniques))
ValueError: Found arrays with inconsistent numbers of samples: [100482 150724]
>>> 251206*.6
150723.6
>>> ================================ RESTART ================================
>>> 
251206 251206
251206
Tuning hyper-parameters for precision


Traceback (most recent call last):
  File "C:\Users\S\Documents\Sonu\SciComp\Research\AstroP2\ap2_sklearn_svmSCV13_combineddata1.py", line 55, in <module>
    clf.fit(trainingSet, trainingSetLabels)
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\sklearn\grid_search.py", line 732, in fit
    return self._fit(X, y, ParameterGrid(self.param_grid))
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\sklearn\grid_search.py", line 477, in _fit
    X, y = indexable(X, y)
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\sklearn\utils\validation.py", line 199, in indexable
    check_consistent_length(*result)
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\sklearn\utils\validation.py", line 174, in check_consistent_length
    "%s" % str(uniques))
ValueError: Found arrays with inconsistent numbers of samples: [100482 150724]
>>> ================================ RESTART ================================
>>> 
251206 251206
251206
Tuning hyper-parameters for precision


>>> ================================ RESTART ================================
>>> 
251206 251206
251206
Time before training =  1440298483.69
Tuning hyper-parameters for precision


Warning (from warnings module):
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\sklearn\metrics\classification.py", line 958
    'precision', 'predicted', average, warn_for)
UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples.

>>> ================================ RESTART ================================
>>> 
251206 251206
251206
Time before training =  1440341317.1
Tuning hyper-parameters for precision


Warning (from warnings module):
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\sklearn\metrics\classification.py", line 958
    'precision', 'predicted', average, warn_for)
UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples.

>>> ================================ RESTART ================================
>>> 
Decision Tree Classifier
Time before training =  1440387512.56

Traceback (most recent call last):
  File "C:\Users\S\Documents\Sonu\SciComp\Research\AstroP2\ap2_sklearn_DT1_combineddata1.py", line 47, in <module>
    clf = tree.DecisionTreeClassifer(max_depth = 3)
AttributeError: 'module' object has no attribute 'DecisionTreeClassifer'
>>> ================================ RESTART ================================
>>> 
Decision Tree Classifier
Time before training =  1440387606.92
Params after training:
{'splitter': 'best', 'max_leaf_nodes': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'criterion': 'gini', 'random_state': None, 'max_features': None, 'max_depth': 3, 'class_weight': None}
Training accuracy =  0.898051392289
Testing accuracy =  0.898105145829
Done training and testing! Time =  0.644000053406 seconds
>>> ================================ RESTART ================================
>>> 
Decision Tree Classifier
Time before training =  1440388058.42
Params after training:
{'splitter': 'best', 'max_leaf_nodes': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'criterion': 'gini', 'random_state': None, 'max_features': None, 'max_depth': 5, 'class_weight': None}
Training accuracy =  0.909436515993
Testing accuracy =  0.907645763117
Done training and testing! Time =  1.06599998474 seconds
>>> ================================ RESTART ================================
>>> 
Decision Tree Classifier
Time before training =  1440388078.66
Params after training:
{'splitter': 'best', 'max_leaf_nodes': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'criterion': 'gini', 'random_state': None, 'max_features': None, 'max_depth': 7, 'class_weight': None}
Training accuracy =  0.926663482017
Testing accuracy =  0.922149093708
Done training and testing! Time =  2.18700003624 seconds
>>> ================================ RESTART ================================
>>> 
Decision Tree Classifier
Time before training =  1440391489.18
Params after training:
{'splitter': 'best', 'max_leaf_nodes': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'criterion': 'gini', 'random_state': None, 'max_features': None, 'max_depth': 10, 'class_weight': None}
Training accuracy =  0.945821142095
Testing accuracy =  0.92939412436
Done training and testing! Time =  1.59099984169 seconds
>>> ================================ RESTART ================================
>>> 
Decision Tree Classifier
Time before training =  1440391516.18
Params after training:
{'splitter': 'best', 'max_leaf_nodes': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'criterion': 'gini', 'random_state': None, 'max_features': None, 'max_depth': 13, 'class_weight': None}
Training accuracy =  0.963336717024
Testing accuracy =  0.926275841936
Done training and testing! Time =  1.98399996758 seconds
>>> ================================ RESTART ================================
>>> 
Decision Tree Classifier
Time before training =  1440391539.58
Params after training:
{'splitter': 'best', 'max_leaf_nodes': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'criterion': 'gini', 'random_state': None, 'max_features': None, 'max_depth': 12, 'class_weight': None}
Training accuracy =  0.957664059234
Testing accuracy =  0.928359119981
Done training and testing! Time =  1.8069999218 seconds
>>> ================================ RESTART ================================
>>> 
Decision Tree Classifier
Time before training =  1440391561.43
Params after training:
{'splitter': 'best', 'max_leaf_nodes': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'criterion': 'gini', 'random_state': None, 'max_features': None, 'max_depth': 11, 'class_weight': None}
Training accuracy =  0.950528452857
Testing accuracy =  0.929142007908
Done training and testing! Time =  1.47599983215 seconds
>>> ================================ RESTART ================================
>>> 
Decision Tree Classifier
Time before training =  1440391593.84
Params after training:
{'splitter': 'best', 'max_leaf_nodes': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'criterion': 'gini', 'random_state': None, 'max_features': None, 'max_depth': 8, 'class_weight': None}
Training accuracy =  0.934694771203
Testing accuracy =  0.925771609034
Done training and testing! Time =  1.10800004005 seconds
>>> ================================ RESTART ================================
>>> 
Decision Tree Classifier
Time before training =  1440391615.89
Params after training:
{'splitter': 'best', 'max_leaf_nodes': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'criterion': 'gini', 'random_state': None, 'max_features': None, 'max_depth': 9, 'class_weight': None}
Training accuracy =  0.939640930714
Testing accuracy =  0.927961041374
Done training and testing! Time =  1.2539999485 seconds
>>> ================================ RESTART ================================
>>> 
Decision Tree Classifier
Time before training =  1440391638.51
Params after training:
{'splitter': 'best', 'max_leaf_nodes': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'criterion': 'gini', 'random_state': None, 'max_features': None, 'max_depth': 10, 'class_weight': None}
Training accuracy =  0.946209271312
Testing accuracy =  0.929977972984
Done training and testing! Time =  1.33899998665 seconds
>>> 
