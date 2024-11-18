import numpy as np

def relu(z):
	return np.maximum(0, z)

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def tanh(z):
	return np.tanh(z)

def softmax(z):
	exps = np.exp(z)
	return exps / np.sum(exps)

def relu_derivative(z):
	return np.where(z <= 0, 0, 1)

def sigmoid_derivative(z):
	return np.exp(-z) / (1 + np.exp(-z))**2

def tanh_derivative(z):
	return 1 - np.tanh(z)**2
