Python 2.7.10 (default, May 23 2015, 09:44:00) [MSC v.1500 64 bit (AMD64)] on win32
Type "copyright", "credits" or "license()" for more information.
>>> ================================ RESTART ================================
>>> 
251206 251206
251206
Time before training =  1440500851.6
Tuning hyper-parameters for accuracy


>>> ================================ RESTART ================================
>>> 
251206 251206
251206
Time before training =  1440582317.75
Tuning hyper-parameters for accuracy

Best parameters found of development set:
{'kernel': 'rbf', 'C': 10, 'gamma': 0}
Time: 19396.6059999
Grid scores on development set:
[mean: 0.91077, std: 0.00332, params: {'kernel': 'rbf', 'C': 1, 'gamma': 0}, mean: 0.91077, std: 0.00332, params: {'kernel': 'rbf', 'C': 1, 'gamma': 0.1}, mean: 0.92027, std: 0.00265, params: {'kernel': 'rbf', 'C': 10, 'gamma': 0}, mean: 0.92027, std: 0.00265, params: {'kernel': 'rbf', 'C': 10, 'gamma': 0.1}]

Detailed classification report:
Model is trained on full development set.
Scores are computed on the full evaluation set.
Training set classification report:
             precision    recall  f1-score   support

       -1.0       0.96      0.97      0.96     75566
        1.0       0.90      0.87      0.89     24916

avg / total       0.94      0.95      0.94    100482

Testing set classification report:
             precision    recall  f1-score   support

       -1.0       0.94      0.96      0.95    113801
        1.0       0.86      0.81      0.84     36923

avg / total       0.92      0.92      0.92    150724

Tuning hyper-parameters for recall

Best parameters found of development set:
{'kernel': 'rbf', 'C': 10, 'gamma': 0}
Time: 39004.4660001
Grid scores on development set:
[mean: 0.78207, std: 0.00802, params: {'kernel': 'rbf', 'C': 1, 'gamma': 0}, mean: 0.78207, std: 0.00802, params: {'kernel': 'rbf', 'C': 1, 'gamma': 0.1}, mean: 0.81036, std: 0.00707, params: {'kernel': 'rbf', 'C': 10, 'gamma': 0}, mean: 0.81036, std: 0.00707, params: {'kernel': 'rbf', 'C': 10, 'gamma': 0.1}]

Detailed classification report:
Model is trained on full development set.
Scores are computed on the full evaluation set.
Training set classification report:
             precision    recall  f1-score   support

       -1.0       0.96      0.97      0.96     75566
        1.0       0.90      0.87      0.89     24916

avg / total       0.94      0.95      0.94    100482

Testing set classification report:
             precision    recall  f1-score   support

       -1.0       0.94      0.96      0.95    113801
        1.0       0.86      0.81      0.84     36923

avg / total       0.92      0.92      0.92    150724

Done training and testing! Time =  39196.477 seconds
>>> 
