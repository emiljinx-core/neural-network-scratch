import numpy as np
import matplotlib.pyplot as plt
from typing import List
import pickle
from tqdm import tqdm

from sklearn.datasets import fetch_openml
mnist = fetch_openml(name="mnist_784")

with open("mnist.pickle", 'wb') as f:
    pickle.dump(mnist, f)
with open("mnist.pickle", 'rb') as f:
    mnist = pickle.load(f)

data = mnist.data
labels = mnist.target

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)

def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)

def leaky_relu(z: np.ndarray) -> np.ndarray:
    return np.where(z > 0, z, z * 0.01)

def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - np.max(z, axis=0, keepdims=True)) 
    return e / np.sum(e, axis=0, keepdims=True)

def normalize(x: np.ndarray) -> np.ndarray:
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)

def one_hot_encode(x: np.ndarray, num_labels: int) -> np.ndarray:
    return np.eye(num_labels)[x]

def derivative(function_name: str, z: np.ndarray) -> np.ndarray:
    if function_name == "sigmoid":
        s = sigmoid(z)
        return s * (1 - s)
    if function_name == "tanh":
        t = tanh(z)
        return 1 - np.square(t)
    if function_name == "relu":
        return (z > 0).astype(np.float32)
    if function_name == "leaky_relu":
        return np.where(z > 0, 1, 0.01)
    raise ValueError(f"No such activation: {function_name}")


class NN(object):
    def __init__(self, X: np.ndarray, y: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, activation: str, num_labels: int, architecture: List[int], optimizer="adam", l2_lambda=0.0001):
        self.X = X.copy() 
        self.X_test = X_test.copy()
        self.y = y.copy()
        self.y_test = y_test.copy()
        
        self.layers = {}
        self.architecture = architecture
        self.activation = activation
        assert self.activation in ["relu", "tanh", "sigmoid", "leaky_relu"]
        
        self.parameters = {}
        self.num_labels = num_labels
        self.m = X.shape[1]
        self.architecture.append(self.num_labels)
        self.num_input_features = X.shape[0]
        self.architecture.insert(0, self.num_input_features)
        self.L = len(architecture)
        
        # Optimizer settings
        self.optimizer = optimizer
        self.l2_lambda = l2_lambda
        self.velocity = {}
        self.s = {}
        self.t = 0
        
        assert self.X.shape == (self.num_input_features, self.m)
        assert self.y.shape == (self.num_labels, self.m)
        
    def initialize_parameters(self):
        for i in range(1, self.L):
            print(f"Initializing parameters for layer: {i}.")
            # He initialization for better convergence with ReLU
            self.parameters["w"+str(i)] = np.random.randn(
                self.architecture[i], self.architecture[i-1]) * np.sqrt(2.0 / self.architecture[i-1])   
            self.parameters["b"+str(i)] = np.zeros((self.architecture[i], 1))
            
            # Initialize optimizer parameters
            if self.optimizer in ["momentum", "adam"]:
                self.velocity["dW"+str(i)] = np.zeros((self.architecture[i], self.architecture[i-1]))
                self.velocity["db"+str(i)] = np.zeros((self.architecture[i], 1))
            if self.optimizer == "adam":
                self.s["dW"+str(i)] = np.zeros((self.architecture[i], self.architecture[i-1]))
                self.s["db"+str(i)] = np.zeros((self.architecture[i], 1))
    
    def forward(self):
        params = self.parameters
        self.layers["a0"] = self.X
        for l in range(1, self.L-1):
            self.layers["z" + str(l)] = np.dot(params["w" + str(l)], 
            self.layers["a"+str(l-1)]) + params["b"+str(l)]
            self.layers["a" + str(l)] = eval(self.activation)(self.layers["z"+str(l)])
            assert self.layers["a"+str(l)].shape == (self.architecture[l], self.m)
        
        self.layers["z" + str(self.L-1)] = np.dot(params["w" + str(self.L-1)],
        self.layers["a"+str(self.L-2)]) + params["b"+str(self.L-1)]
        self.layers["a"+str(self.L-1)] = softmax(self.layers["z"+str(self.L-1)])
        self.output = self.layers["a"+str(self.L-1)]
        
        assert self.output.shape == (self.num_labels, self.m)
        assert np.allclose(np.sum(self.output, axis=0), 1.0)
        
        # Cross-entropy cost with L2 regularization
        cost = -np.sum(self.y * np.log(self.output + 1e-8)) / self.m
        
        if self.l2_lambda > 0:
            l2_cost = 0
            for l in range(1, self.L):
                l2_cost += np.sum(np.square(params["w" + str(l)]))
            cost += (self.l2_lambda / (2 * self.m)) * l2_cost

        return cost, self.layers

    def backpropagate(self):
        derivatives = {}
        dZ = self.output - self.y
        assert dZ.shape == (self.num_labels, self.m)
        
        dW = np.dot(dZ, self.layers["a" + str(self.L-2)].T) / self.m
        if self.l2_lambda > 0:
            dW += (self.l2_lambda / self.m) * self.parameters["w" + str(self.L-1)]
        
        db = np.sum(dZ, axis=1, keepdims=True) / self.m
        dAPrev = np.dot(self.parameters["w" + str(self.L-1)].T, dZ)
        derivatives["dW" + str(self.L-1)] = dW
        derivatives["db" + str(self.L-1)] = db
        
        for l in range(self.L-2, 0, -1):
            dZ = dAPrev * derivative(self.activation, self.layers["z" + str(l)])  
            dW = np.dot(dZ, self.layers["a" + str(l-1)].T) / self.m
            
            if self.l2_lambda > 0:
                dW += (self.l2_lambda / self.m) * self.parameters["w" + str(l)]
            
            db = np.sum(dZ, axis=1, keepdims=True) / self.m
            if l > 1:
                dAPrev = np.dot(self.parameters["w" + str(l)].T, dZ)
            derivatives["dW" + str(l)] = dW
            derivatives["db" + str(l)] = db
        
        self.derivatives = derivatives
        return self.derivatives

    def update_parameters(self, lr, derivatives):
        """Update parameters using specified optimizer"""
        if self.optimizer == "sgd":
            for layer in range(1, self.L):
                self.parameters["w"+str(layer)] -= lr * derivatives["dW" + str(layer)]
                self.parameters["b"+str(layer)] -= lr * derivatives["db" + str(layer)]
        
        elif self.optimizer == "momentum":
            beta = 0.9
            for layer in range(1, self.L):
                self.velocity["dW"+str(layer)] = beta * self.velocity["dW"+str(layer)] + (1-beta) * derivatives["dW"+str(layer)]
                self.velocity["db"+str(layer)] = beta * self.velocity["db"+str(layer)] + (1-beta) * derivatives["db"+str(layer)]
                
                self.parameters["w"+str(layer)] -= lr * self.velocity["dW"+str(layer)]
                self.parameters["b"+str(layer)] -= lr * self.velocity["db"+str(layer)]
        
        elif self.optimizer == "adam":
            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-8
            self.t += 1
            
            for layer in range(1, self.L):
                # Update biased first moment estimate
                self.velocity["dW"+str(layer)] = beta1 * self.velocity["dW"+str(layer)] + (1-beta1) * derivatives["dW"+str(layer)]
                self.velocity["db"+str(layer)] = beta1 * self.velocity["db"+str(layer)] + (1-beta1) * derivatives["db"+str(layer)]
                
                # Update biased second raw moment estimate
                self.s["dW"+str(layer)] = beta2 * self.s["dW"+str(layer)] + (1-beta2) * np.square(derivatives["dW"+str(layer)])
                self.s["db"+str(layer)] = beta2 * self.s["db"+str(layer)] + (1-beta2) * np.square(derivatives["db"+str(layer)])
                
                # Compute bias-corrected estimates
                v_corrected_dW = self.velocity["dW"+str(layer)] / (1 - beta1**self.t)
                v_corrected_db = self.velocity["db"+str(layer)] / (1 - beta1**self.t)
                s_corrected_dW = self.s["dW"+str(layer)] / (1 - beta2**self.t)
                s_corrected_db = self.s["db"+str(layer)] / (1 - beta2**self.t)
                
                # Update parameters
                self.parameters["w"+str(layer)] -= lr * v_corrected_dW / (np.sqrt(s_corrected_dW) + epsilon)
                self.parameters["b"+str(layer)] -= lr * v_corrected_db / (np.sqrt(s_corrected_db) + epsilon)

    def fit(self, lr=0.001, epochs=1000, batch_size=128):
        self.costs = [] 
        self.initialize_parameters()
        self.accuracies = {"train": [], "test": []}
        
        num_batches = self.m // batch_size
        
        for epoch in tqdm(range(epochs), colour="BLUE"):
            # Shuffle training data
            permutation = np.random.permutation(self.m)
            X_shuffled = self.X[:, permutation]
            y_shuffled = self.y[:, permutation]
            
            epoch_cost = 0
            
            for i in range(num_batches):
                start = i * batch_size
                end = start + batch_size
                
                # Mini-batch
                X_batch = X_shuffled[:, start:end]
                y_batch = y_shuffled[:, start:end]
                
                # Temporarily set batch data
                original_X, original_y, original_m = self.X, self.y, self.m
                self.X, self.y, self.m = X_batch, y_batch, batch_size
                
                # Forward and backward pass
                cost, cache = self.forward()
                epoch_cost += cost
                derivatives = self.backpropagate()
                
                # Update parameters
                self.update_parameters(lr, derivatives)
                
                # Restore original data
                self.X, self.y, self.m = original_X, original_y, original_m
            
            avg_cost = epoch_cost / num_batches
            self.costs.append(avg_cost)
            
            train_accuracy = self.accuracy(self.X, self.y)
            test_accuracy = self.accuracy(self.X_test, self.y_test)
            
            if epoch % 10 == 0:
                print(f"Epoch: {epoch:3d} | Cost: {avg_cost:.3f} | Train: {train_accuracy:.2f}% | Test: {test_accuracy:.2f}%")
            
            self.accuracies["train"].append(train_accuracy)
            self.accuracies["test"].append(test_accuracy)
        
        print("Training terminated")

    def predict(self, x):
        params = self.parameters
        n_layers = self.L - 1
        values = [x]
        for l in range(1, n_layers):
            z = np.dot(params["w" + str(l)], values[l-1]) + params["b" + str(l)]
            a = eval(self.activation)(z)
            values.append(a)
        z = np.dot(params["w"+str(n_layers)], values[n_layers-1]) + params["b"+str(n_layers)]
        a = softmax(z)
        if x.shape[1] > 1:
            ans = np.argmax(a, axis=0)
        else:
            ans = np.argmax(a)
        return ans
    
    def accuracy(self, X, y):
        P = self.predict(X)
        return sum(np.equal(P, np.argmax(y, axis=0))) / y.shape[1] * 100
    
    def pickle_model(self, name: str):
        with open("fitted_model_"+ name + ".pickle", "wb") as modelFile:
            pickle.dump(self, modelFile)
    
    def plot_counts(self):
        counts = np.unique(np.argmax(self.output, axis=0), return_counts=True)
        plt.bar(counts[0], counts[1], color="navy")
        plt.ylabel("Counts")
        plt.xlabel("y_hat")
        plt.title("Distribution of predictions")
        plt.show()
    
    def plot_cost(self, lr):
        plt.figure(figsize=(8, 4))
        plt.plot(np.arange(0, len(self.costs)), self.costs, lw=1, color="orange")
        plt.title(f"Learning rate: {lr}\nFinal Cost: {self.costs[-1]:.5f}", fontdict={
            "family":"sans-serif", 
            "size": "12"})
        plt.xlabel("Epoch")
        plt.ylabel("Cost")
        plt.show()
    
    def plot_accuracies(self, lr):
        acc = self.accuracies
        fig = plt.figure(figsize=(6,4))
        ax = fig.add_subplot(111)
        ax.plot(acc["train"], label="train")
        ax.plot(acc["test"], label="test")
        plt.legend(loc="lower right")
        ax.set_title("Accuracy")
        ax.annotate(f"Train: {acc['train'][-1]:.2f}", (len(acc["train"])+4, acc["train"][-1]+2), color="blue")
        ax.annotate(f"Test: {acc['test'][-1]:.2f}", (len(acc["test"])+4, acc["test"][-1]-2), color="orange")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.show()
    
    def __str__(self):                
        return str(self.architecture)

