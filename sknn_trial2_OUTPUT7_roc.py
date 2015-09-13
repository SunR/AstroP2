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
>>> 
