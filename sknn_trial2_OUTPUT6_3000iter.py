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
>>> 
