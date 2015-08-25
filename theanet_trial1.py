import climate
import theanets
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix

climate.enable_default_logging()

# Create a classification dataset.
X, y = make_classification(
    n_samples=3000, n_features=100, n_classes=10, n_informative=10)
X = X.astype('f')
y = y.astype('i')
cut = int(len(X) * 0.8)  # training / validation split
train = X[:cut], y[:cut]
valid = X[cut:], y[cut:]

# Build a classifier model with 100 inputs and 10 outputs.
net = theanets.Classifier(layers=[100, 10])

# Train the model using SGD with momentum.
net.train(train, valid, algo='sgd', learning_rate=1e-4, momentum=0.9)

# Show confusion matrices on the training/validation splits.
for label, (X, y) in (('training:', train), ('validation:', valid)):
    print(label)
    print(confusion_matrix(y, net.predict(X)))
