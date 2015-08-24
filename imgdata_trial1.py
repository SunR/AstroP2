from pylab import imshow, figure, zeros, plot
from scipy.misc import imread
from scipy.ndimage.interpolation import rotate
from numpy import savetxt

#Extracting ra dec pairs

data = np.loadtxt("crossmatched3_combineddata1_srane.txt", delimiter = ",", skiprows = 1, usecols = (1, 2, 3, 10, 11, 12, 13, 14, 35, 37, 39, 41, 43, 49, 50, 51)) #dr7objid, petromag_u, petromag_g, petromag_r, pertomag_i, petromag_z, z(redshift),h alpha ew, h beta ew, OII ew, h delta ew, spiral, elliptical, uncertain

isSpiral = data[:,13]
isElliptical = data[:,14] #corresponds to the elliptical bool value

isUncertain = data[:,15]

ellipticals = data[isElliptical == 1]

spirals = data[isSpiral == 1]
uncertains = data[isUncertain == 1]

trainingSetEllipticals = ellipticals[:500] #use all of set b/c cross validation will split into training and testing sets
trainingSetSpirals = spirals[:500]

trainingSet = np.vstack((trainingSetEllipticals, trainingSetSpirals))

coords = trainingSet[:, 2:4]

