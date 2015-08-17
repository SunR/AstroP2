Python 2.7.10 (default, May 23 2015, 09:44:00) [MSC v.1500 64 bit (AMD64)] on win32
Type "copyright", "credits" or "license()" for more information.
>>> ================================ RESTART ================================
>>> 
Time before training =  1439769917.26
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
  kernel='rbf', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False)
Done training! Time =  597.164999962 seconds
Training accuracy =  0.86787
Time =  694.163999796 seconds

Testing accuracy =  0.83773993758
Time =  754.287999868 seconds
Finished writing predictions to file!
>>> ================================ RESTART ================================
>>> 
Time before training =  1439774576.45
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
  kernel='rbf', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False)
Done training! Time =  491.720999956 seconds
Training accuracy =  0.90094
Time =  575.304999828 seconds

Testing accuracy =  0.883018806902
Time =  628.135999918 seconds

Traceback (most recent call last):
  File "C:/Users/S/Documents/Sonu/SciComp/Research/AstroP2/ap2_sklearn_svmSCV5_combineddata1.py", line 64, in <module>
    predictions = clf.predict(classificationSet) #use model to classify uncertains - SEE IF YOU CAN FIND CONFIDENCE INTERVAL FOR THIS PREDICTION AFTERWARD, THAT WILL BE PROGRESS
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\sklearn\svm\base.py", line 500, in predict
    y = super(BaseSVC, self).predict(X)
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\sklearn\svm\base.py", line 290, in predict
    X = self._validate_for_predict(X)
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\sklearn\svm\base.py", line 443, in _validate_for_predict
    (n_features, self.shape_fit_[1]))
ValueError: X.shape[1] = 11 should be equal to 10, the number of features at training time
>>> ================================ RESTART ================================
>>> 
Time before training =  1439775874.33
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
  kernel='rbf', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False)
Done training! Time =  497.773999929 seconds
Training accuracy =  0.90092
Time =  585.801000118 seconds

Testing accuracy =  0.883002635877
Time =  641.088000059 seconds
Finished writing predictions to file!
>>> 
