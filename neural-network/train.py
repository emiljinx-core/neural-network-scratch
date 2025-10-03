import numpy as np
import pickle
from sklearn.datasets import fetch_openml
from nn_model import NN, one_hot_encode

# Load MNIST
mnist = fetch_openml(name="mnist_784")
with open("mnist.pickle", 'wb') as f:
    pickle.dump(mnist, f)

# Prepare data
with open("mnist.pickle", 'rb') as f:
    mnist = pickle.load(f)
    
data = mnist.data
labels = mnist.target
train_test_split_no = 60000

X_train = (data.values[:train_test_split_no] / 255.0).T
y_train = labels[:train_test_split_no].values.astype(int)
y_train = one_hot_encode(y_train, 10).T

X_test = (data.values[train_test_split_no:] / 255.0).T
y_test = labels[train_test_split_no:].values.astype(int)
y_test = one_hot_encode(y_test, 10).T

# Train
PARAMS = [X_train, y_train, X_test, y_test, "relu", 10, [512, 256]]
nn_relu = NN(*PARAMS, optimizer="adam", l2_lambda=0.0001)
nn_relu.fit(lr=0.001, epochs=100, batch_size=128)
nn_relu.pickle_model("relu_optimized")