import numpy as np

def relu(x):
  return max(0,x)

def relu_derivative(x):
  return (x>0).astype(float)

def sigmoid(x):
  return 1/ (1 + np.exp(-x))

def sigmoid_derivative(x):
  return x* (1-x)

def tanh(x):
  return np.exp(x) - np.exp(-x) / np.exp(x) + np.exp(x)

def tanh(x):
  return 1 - tanh(x)**2

def softmax(x):
  exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
  return exp_x / exp_x.sum(axis=0, keepdims=True)

























# import numpy as np

# def relu(x):
#   return np.max(0,x)

# def relu_derivative(x):
#   """
#   x = np.array([2, 3, 0, -1, -5])
#   relu_derivative(x)

#   output: array([1., 1., 0., 0., 0.])

#   the x>0 would return a boolean either 0 or 1

#   """
#   return (x > 0).astype(float)

# def sigmoid(x):
#   return 1 / (1 + np.exp(-x)) 

# def sigmoid_derivative(x):
#   s = sigmoid(x)
#   return s * (1 - s)

# def tanh(x):
#   return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

# def tanh_derivative(x):
#   return 1 - tanh(x)**2

# def softmax(x):
#   """
#   axis =0 column wise operation
#   axis =1 row wise operation
#   """
#   exp_x = np.exp(x - np.max(x))
#   return exp_x / np.sum(exp_x)