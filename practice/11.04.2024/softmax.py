import numpy as np

def relu(x):
  return np.max(0,x)

def relu_derivative(x):
  return (x>0).astype(float)


def sigmoid(x):
  return 1/ 1+ np.exp(-x)

def sigmoid_derivative(x):
  return x* (1-x)

def tanh(x):
  return np.exp(x) - np.exp(-x) / np.exp(x) + np.exp(x)

def tanh_derivative(x):
  return 1- tanh(x)**2


def softmax(x):
  exp_x = np.exp(x - np.max(x))
  return exp_x/ exp_x.sum()

