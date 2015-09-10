Python 2.7.10 (default, May 23 2015, 09:44:00) [MSC v.1500 64 bit (AMD64)] on win32
Type "copyright", "credits" or "license()" for more information.
>>> ================================ RESTART ================================
>>> 

Traceback (most recent call last):
  File "C:\Users\S\Documents\Sonu\SciComp\Research\AstroIm\centroid_program1.py", line 14, in <module>
    imgData= pyfits.getdata("sampleimage.fits") #extracts array of pixel values
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\pyfits\convenience.py", line 182, in getdata
    hdulist, extidx = _getext(filename, mode, *args, **kwargs)
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\pyfits\convenience.py", line 742, in _getext
    hdulist = fitsopen(filename, mode=mode, **kwargs)
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\pyfits\hdu\hdulist.py", line 109, in fitsopen
    return HDUList.fromfile(name, mode, memmap, save_backup, **kwargs)
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\pyfits\hdu\hdulist.py", line 241, in fromfile
    save_backup=save_backup, **kwargs)
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\pyfits\hdu\hdulist.py", line 758, in _readfrom
    ffo = _File(fileobj, mode=mode, memmap=memmap)
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\pyfits\file.py", line 121, in __init__
    raise IOError('File does not exist: %r' % fileobj)
IOError: File does not exist: 'sampleimage.fits'
>>> from numpy import array
>>> from pydl.photoop.photoobj import unwrap_objid

Traceback (most recent call last):
  File "<pyshell#1>", line 1, in <module>
    from pydl.photoop.photoobj import unwrap_objid
ImportError: No module named pydl.photoop.photoobj
>>> from pydl.photoop.photoobj import unwrap_objid
>>> unwrap_objid(587722952230174000)

Traceback (most recent call last):
  File "<pyshell#3>", line 1, in <module>
    unwrap_objid(587722952230174000)
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\pydl\photoop\photoobj\unwrap_objid.py", line 39, in unwrap_objid
    if objid.dtype.type is string_ or objid.dtype.type is unicode_:
AttributeError: 'long' object has no attribute 'dtype'
>>> unwrap_objid([587722952230174000])

Traceback (most recent call last):
  File "<pyshell#4>", line 1, in <module>
    unwrap_objid([587722952230174000])
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\pydl\photoop\photoobj\unwrap_objid.py", line 39, in unwrap_objid
    if objid.dtype.type is string_ or objid.dtype.type is unicode_:
AttributeError: 'list' object has no attribute 'dtype'
>>> unwrap_objid(array[587722952230174000])

Traceback (most recent call last):
  File "<pyshell#5>", line 1, in <module>
    unwrap_objid(array[587722952230174000])
TypeError: 'builtin_function_or_method' object has no attribute '__getitem__'
>>> unwrap_objid(array([587722952230174000]))
rec.array([(1, 40, 745, 2, 517, 64816)], 
      dtype=[('skyversion', '<i4'), ('rerun', '<i4'), ('run', '<i4'), ('camcol', '<i4'), ('frame', '<i4'), ('id', '<i4')])
>>> 587722952230174996
587722952230174996L
>>> unwrap_objid(array([587722952230174996]))
rec.array([(1, 40, 745, 2, 518, 276)], 
      dtype=[('skyversion', '<i4'), ('rerun', '<i4'), ('run', '<i4'), ('camcol', '<i4'), ('frame', '<i4'), ('id', '<i4')])
>>> a = unwrap_objid(array([587722952230174000]))
>>> a
rec.array([(1, 40, 745, 2, 517, 64816)], 
      dtype=[('skyversion', '<i4'), ('rerun', '<i4'), ('run', '<i4'), ('camcol', '<i4'), ('frame', '<i4'), ('id', '<i4')])
>>> a[0]
(1, 40, 745, 2, 517, 64816)
>>> a[1]

Traceback (most recent call last):
  File "<pyshell#12>", line 1, in <module>
    a[1]
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\numpy\core\records.py", line 458, in __getitem__
    obj = ndarray.__getitem__(self, indx)
IndexError: index 1 is out of bounds for axis 0 with size 1
>>> a['dtype']

Traceback (most recent call last):
  File "<pyshell#13>", line 1, in <module>
    a['dtype']
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\numpy\core\records.py", line 458, in __getitem__
    obj = ndarray.__getitem__(self, indx)
ValueError: field named dtype not found
>>> a[0][0]
1
>>> ================================ RESTART ================================
>>> 

Warning (from warnings module):
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\pyfits\card.py", line 979
    self._image)
UserWarning: The following header keyword is invalid or follows an unrecognized non-standard convention:
STRIP   ='N          '         / Strip in the stripe being tracked.             

Warning (from warnings module):
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\pyfits\card.py", line 979
    self._image)
UserWarning: The following header keyword is invalid or follows an unrecognized non-standard convention:
FLAVOR  ='science    '         / Flavor of this run                             

Warning (from warnings module):
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\pyfits\card.py", line 979
    self._image)
UserWarning: The following header keyword is invalid or follows an unrecognized non-standard convention:
SYS_SCN ='mean       '         / System of the scan great circle (e.g., mean)   

Warning (from warnings module):
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\pyfits\card.py", line 979
    self._image)
UserWarning: The following header keyword is invalid or follows an unrecognized non-standard convention:
OBJECT  ='100 N      '         / e.g., 'stripe 50.6 degrees, north strip'       

Warning (from warnings module):
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\pyfits\card.py", line 979
    self._image)
UserWarning: The following header keyword is invalid or follows an unrecognized non-standard convention:
EXPTIME ='53.886976'           / Exposure time (seconds)                        

Warning (from warnings module):
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\pyfits\card.py", line 979
    self._image)
UserWarning: The following header keyword is invalid or follows an unrecognized non-standard convention:
DAVERS  ='v12_6   '            / Version of DA software                         
Please enter center X coordinate: 5
Please enter center Y coordinate: 5

Warning (from warnings module):
  File "C:\Users\S\Documents\Sonu\SciComp\Research\AstroIm\centroid_program1.py", line 55
    centroidX = sumX/sumIntensities #Divides sum of x values by sum of intensities to get weighted mean for X coordinates
RuntimeWarning: invalid value encountered in double_scalars

Warning (from warnings module):
  File "C:\Users\S\Documents\Sonu\SciComp\Research\AstroIm\centroid_program1.py", line 56
    centroidY = sumY/sumIntensities ##Divides sum of y values by sum of intensities to get weighted mean for Y coordinates
RuntimeWarning: invalid value encountered in double_scalars
Centroid: ( nan , nan )

Traceback (most recent call last):
  File "C:\Users\S\Documents\Sonu\SciComp\Research\AstroIm\centroid_program1.py", line 62, in <module>
    calculateCentroid(getImageData())
  File "C:\Users\S\Documents\Sonu\SciComp\Research\AstroIm\centroid_program1.py", line 60, in calculateCentroid
    getUncertainty(actualX, actualY, selectedData, sumIntensities) #calculates uncertainties for centroid calculations
NameError: global name 'getUncertainty' is not defined
>>> ================================ RESTART ================================
>>> 

Warning (from warnings module):
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\pyfits\card.py", line 979
    self._image)
UserWarning: The following header keyword is invalid or follows an unrecognized non-standard convention:
STRIP   ='N          '         / Strip in the stripe being tracked.             

Warning (from warnings module):
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\pyfits\card.py", line 979
    self._image)
UserWarning: The following header keyword is invalid or follows an unrecognized non-standard convention:
FLAVOR  ='science    '         / Flavor of this run                             

Warning (from warnings module):
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\pyfits\card.py", line 979
    self._image)
UserWarning: The following header keyword is invalid or follows an unrecognized non-standard convention:
SYS_SCN ='mean       '         / System of the scan great circle (e.g., mean)   

Warning (from warnings module):
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\pyfits\card.py", line 979
    self._image)
UserWarning: The following header keyword is invalid or follows an unrecognized non-standard convention:
OBJECT  ='100 N      '         / e.g., 'stripe 50.6 degrees, north strip'       

Warning (from warnings module):
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\pyfits\card.py", line 979
    self._image)
UserWarning: The following header keyword is invalid or follows an unrecognized non-standard convention:
EXPTIME ='53.886976'           / Exposure time (seconds)                        

Warning (from warnings module):
  File "C:\WinPython-64bit-2.7.10.1\python-2.7.10.amd64\lib\site-packages\pyfits\card.py", line 979
    self._image)
UserWarning: The following header keyword is invalid or follows an unrecognized non-standard convention:
DAVERS  ='v12_6   '            / Version of DA software                         
Please enter center X coordinate: 5
Please enter center Y coordinate: 5

Warning (from warnings module):
  File "C:\Users\S\Documents\Sonu\SciComp\Research\AstroIm\centroid_program1.py", line 55
    centroidX = sumX/sumIntensities #Divides sum of x values by sum of intensities to get weighted mean for X coordinates
RuntimeWarning: invalid value encountered in double_scalars

Warning (from warnings module):
  File "C:\Users\S\Documents\Sonu\SciComp\Research\AstroIm\centroid_program1.py", line 56
    centroidY = sumY/sumIntensities ##Divides sum of y values by sum of intensities to get weighted mean for Y coordinates
RuntimeWarning: invalid value encountered in double_scalars
Centroid: ( nan , nan )
>>> ================================ RESTART ================================
>>> 
Neural Network Classifer
Using combined1_half1_test3.txt and combined1_half2_test3.txt
stacked files!
Neural network specifications:
Classifier(batch_size=1, debug=False, dropout_rate=None, f_stable=0.001,
      hidden0=<sknn.nn.Layer `Sigmoid`: name=u'hidden0', units=100>,
      layers=[<sknn.nn.Layer `Sigmoid`: name=u'hidden0', units=100>, <sknn.nn.Layer `Softmax`: name=u'output'>],
      learning_momentum=0.9, learning_rate=7e-07, learning_rule=u'sgd',
      loss_type=u'mse', mutator=None, n_iter=1000, n_stable=50,
      output=<sknn.nn.Layer `Softmax`: name=u'output'>, random_state=None,
      regularize=None, valid_set=None, valid_size=0.0, verbose=None,
      weight_decay=None)
I 2015-09-06 02:06:04 sknn:198 Initializing neural network with 2 layers, 29 inputs and 2 outputs.
I 2015-09-06 02:06:04 sknn:129 Training on dataset of 100,000 samples with 3,100,000 total size.
I 2015-09-06 08:00:17 sknn:482 Terminating after specified 1000 total iterations.
Training accuracy =  0.88682
Validation accuracy =  1.0
Testing accuracy =  0.88445124963
Time =  21278.8969998 seconds
>>> ================================ RESTART ================================
>>> 
Neural Network Classifer
Using combined1_half1_test3.txt and combined1_half2_test3.txt
stacked files!
Neural network specifications:
Classifier(batch_size=1, debug=False, dropout_rate=None, f_stable=0.001,
      hidden0=<sknn.nn.Layer `Sigmoid`: name=u'hidden0', units=100>,
      layers=[<sknn.nn.Layer `Sigmoid`: name=u'hidden0', units=100>, <sknn.nn.Layer `Softmax`: name=u'output'>],
      learning_momentum=0.9, learning_rate=6e-05, learning_rule=u'sgd',
      loss_type=u'mse', mutator=None, n_iter=1000, n_stable=50,
      output=<sknn.nn.Layer `Softmax`: name=u'output'>, random_state=None,
      regularize=None, valid_set=None, valid_size=0.0, verbose=None,
      weight_decay=None)
I 2015-09-06 17:10:32 sknn:198 Initializing neural network with 2 layers, 29 inputs and 2 outputs.
I 2015-09-06 17:10:33 sknn:129 Training on dataset of 100,000 samples with 3,100,000 total size.
I 2015-09-06 23:14:23 sknn:482 Terminating after specified 1000 total iterations.
Training accuracy =  0.92529
Validation accuracy =  1.0
Testing accuracy =  0.926019954559
Time =  21856.1370001 seconds
>>> ================================ RESTART ================================
>>> 
Neural Network Classifer
Using combined1_half1_test3.txt and combined1_half2_test3.txt
stacked files!
Neural network specifications:
Classifier(batch_size=1, debug=False, dropout_rate=None, f_stable=0.001,
      hidden0=<sknn.nn.Layer `Sigmoid`: name=u'hidden0', units=100>,
      layers=[<sknn.nn.Layer `Sigmoid`: name=u'hidden0', units=100>, <sknn.nn.Layer `Softmax`: name=u'output'>],
      learning_momentum=0.9, learning_rate=0.0006, learning_rule=u'sgd',
      loss_type=u'mse', mutator=None, n_iter=1000, n_stable=50,
      output=<sknn.nn.Layer `Softmax`: name=u'output'>, random_state=None,
      regularize=None, valid_set=None, valid_size=0.0, verbose=None,
      weight_decay=None)
I 2015-09-07 03:11:18 sknn:198 Initializing neural network with 2 layers, 29 inputs and 2 outputs.
I 2015-09-07 03:11:19 sknn:129 Training on dataset of 100,000 samples with 3,100,000 total size.
I 2015-09-07 09:06:17 sknn:482 Terminating after specified 1000 total iterations.
Training accuracy =  0.91555
Validation accuracy =  1.0
Testing accuracy =  0.911715894498
Time =  21329.8199999 seconds
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
      loss_type=u'mse', mutator=None, n_iter=1000, n_stable=50,
      output=<sknn.nn.Layer `Softmax`: name=u'output'>, random_state=None,
      regularize=None,
      valid_set=(array([[  5.65133e+00,   4.89288e+00, ...,   7.32027e-01,  -5.18000e-01],
       [  2.20939e+00,   1.89130e+00, ...,  -8.26506e+02,  -1.99161e+00],
       ...,
       [  5.88102e+00,   4.73546e+00, ...,   2.54482e+00,  -1.79068e+00],
       [  2.90614e+00,   2.45517e+00, ...,   8.62723e+00,  -4.50604e+00]]), array([-1.,  1., ..., -1., -1.])),
      valid_size=0.0, verbose=None, weight_decay=None)
I 2015-09-07 16:10:54 sknn:198 Initializing neural network with 2 layers, 29 inputs and 2 outputs.
I 2015-09-07 16:10:55 sknn:129 Training on dataset of 100,000 samples with 3,100,000 total size.
I 2015-09-07 20:32:30 sknn:478 Early termination condition fired at 425 iterations.
Training accuracy =  0.92841
Validation accuracy =  1.0
Testing accuracy =  0.928459942705
Time =  15722.75 seconds
>>> 
