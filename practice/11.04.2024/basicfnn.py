import numpy as np

# Initialize parameters
def initialize_parameters(input_dim, hidden_dim, output_dim):
  """
  W1 = hidden, input
  b1 = hidden 
  W2 = output, hidden
  b2 = output 
  """
  np.random.seed(42)  # For reproducibility
  params = {
      'W1': np.random.randn(hidden_dim, input_dim) * 0.01, # randn gives normalized randomized values
      'b1': np.zeros((hidden_dim, 1)),
      'W2': np.random.randn(output_dim, hidden_dim) * 0.01,
      'b2': np.zeros((output_dim, 1)),
  }
  return params


# Forward pass
def forward_pass(X, params):
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']

    # np.dot ( dot product for two vectors are communicative, however tdot product of matrices are nto )
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)  # Activation for hidden layer
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)  # Activation for output layer

    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
    return A2, cache


# Compute loss (Cross-Entropy)
# A2 : final output of the feedforward NN
def compute_loss(Y, A2):
    m = Y.shape[1] # Y ( class, m )
    # loss = -np.sum(Y * np.log(A2 + 1e-9)) / m
    loss = - np.sum(Y * np.log(A2 + 1e-9)) / m
    return loss