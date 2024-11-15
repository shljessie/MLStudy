import numpy as np

"""
Overview of all activation functions
1. first write down all the activation functions you will probably use
2. initialize paramenters to pass in
"""

def relu(x):
  return np.maximum(0,x)

def relu_derivative(x):
  """
  x = np.array([2, 3, 0, -1, -5])
  relu_derivative(x)

  output: array([1., 1., 0., 0., 0.])

  the x>0 would return a boolean either 0 or 1

  """
  return (x > 0).astype(float)

def sigmoid(x):
  """
  values between 0,1 --> classification
  """
  return 1 / (1 + np.exp(-x)) 

def sigmoid_derivative(x):
  s = sigmoid(x)
  return s * (1 - s)

def tanh(x):
  """
  faster convergence centers around zero
  """
  return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def tanh_derivative(x):
  return 1 - tanh(x)**2

# softmax
def softmax(x):
  # x  (c,m)  - want to normalize over classes 
  exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
  return exp_x / exp_x.sum(axis=0, keepdims=True)

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
      'W1': np.random.randn(hidden_dim, input_dim) * 0.01,
      'b1': np.zeros((hidden_dim, 1)),
      'W2': np.random.randn(output_dim, hidden_dim) * 0.01,
      'b2': np.zeros((output_dim, 1)),
  }
  return params

# Forward pass
def forward_pass(X, params):
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']

    Z1 = np.dot(W1, X) + b1  # (hidden_dim, m) 
    A1 = relu(Z1)  # Activation for hidden layer #(hidden_dim, m)
    Z2 = np.dot(W2, A1) + b2 # (output_dim, m)
    A2 = softmax(Z2)  # Activation for output layer # (output_dim, m)

    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
    return A2, cache


# Compute loss (Cross-Entropy)
def compute_loss(Y, A2):
    m = Y.shape[1] # Y ( class, m )
    # loss = -np.sum(Y * np.log(A2 + 1e-9)) / m
    loss = - np.sum(Y * np.log(A2 + 1e-9)) / m
    return loss



# Backward pass ! Good let's repeat this a few times until it becomes easier for you
def backward_pass(X, Y, cache, params):
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    A1, A2 = cache['A1'], cache['A2']
    m = X.shape[1]

    dZ2 = A2- Y # (c,m)  #  # (output_dim, m)
    dW2 = np.dot(dZ2, A1.T) / m  # a1 ( n,m ) a1.t (m,n)  # (output_dim, m) (m,hidden) # (output,hidden)
    db2 = np.sum(dZ2, axis=1, keepdims=True) /m # we want to sum for each element 

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1  * relu_derivative(cache['Z1'])
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    return grads

# Update parameters
def update_parameters(params, grads, learning_rate):
    params['W1'] -= learning_rate * grads['dW1']
    params['b1'] -= learning_rate * grads['db1']
    params['W2'] -= learning_rate * grads['dW2']
    params['b2'] -= learning_rate * grads['db2']
    return params

# Simple feedforward network training
def train(X, Y, input_dim, hidden_dim, output_dim, epochs=1000, learning_rate=0.01):
    params = initialize_parameters(input_dim, hidden_dim, output_dim)

    for epoch in range(epochs):
        # Forward pass
        A2, cache = forward_pass(X, params)

        # Compute loss
        loss = compute_loss(Y, A2)

        # Backward pass
        grads = backward_pass(X, Y, cache, params)

        # Update parameters
        params = update_parameters(params, grads, learning_rate)

        # Print loss every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

    return params



## with the adam optimizer 
def initialize_adam(params):
    v = {}
    s = {}
    for key in params:
        v[key] = np.zeros_like(params[key])
        s[key] = np.zeros_like(params[key])
    return v, s

def update_parameters_with_adam(params, grads, v, s, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Update parameters using Adam optimization algorithm.
    
    Arguments:
    params -- dictionary containing your parameters 
    grads -- dictionary containing your gradients, output of backward propagation
    v -- Adam variable, moving average of the first gradient
    s -- Adam variable, moving average of the squared gradient
    t -- current iteration number
    learning_rate -- the learning rate, scalar
    beta1 -- Exponential decay hyperparameter for the first moment estimates 
    beta2 -- Exponential decay hyperparameter for the second moment estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    params -- dictionary containing your updated parameters 
    v -- Adam variable, moving average of the first gradient
    s -- Adam variable, moving average of the squared gradient
    """
    for key in params:
        # Moving average of the gradients
        v[key] = beta1 * v[key] + (1 - beta1) * grads['d' + key]
        # Moving average of the squared gradients
        s[key] = beta2 * s[key] + (1 - beta2) * (grads['d' + key] ** 2)
        
        # Bias correction
        v_corrected = v[key] / (1 - beta1 ** t)
        s_corrected = s[key] / (1 - beta2 ** t)
        
        # Update parameters
        params[key] -= learning_rate * v_corrected / (np.sqrt(s_corrected) + epsilon)
        
    return params, v, s




"""
Now... if we wanted to incorporate attention 

"""

def attention(Q, K, V):
    """
    Compute scaled dot-product attention.
    Q, K, V all have shape (m, dk), where
    hidden_dim = number of hidden neurons
    m = number of examples (batch size)
    """
    # Step 1: Calculate attention scores
    scores = np.dot(Q, K.T) / np.sqrt(Q.shape[0])  # Shape: (m,m)

    # Step 2: Apply softmax to get attention weights
    attn_weights = softmax(scores)  # Shape: (m, m)

    # Step 3: Compute weighted sum of values
    attn_output = np.dot(attn_weights, V)  # Shape: (m, dk)

    return attn_output

# Modified forward pass to include attention
def forward_pass_with_attention(X, params):
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']

    # Step 1: Hidden layer activation
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)  # Activation for hidden layer

    # Step 2: Attention layer
    Q, K, V = A1, A1, A1  # Using A1 as Q, K, V for simplicity
    A1_attn = attention(Q, K, V)  # Apply attention to A1

    # Step 3: Output layer activation
    Z2 = np.dot(W2, A1_attn) + b2
    A2 = softmax(Z2)  # Activation for output layer

    cache = {'Z1': Z1, 'A1': A1, 'A1_attn': A1_attn, 'Z2': Z2, 'A2': A2}
    return A2, cache


def multi_head_attention(Q, K, V, num_heads=4):
    """
    Multi-head attention implementation.

    Q, K, V have shape (hidden_dim, m).
    num_heads: Number of attention heads.

    In the multi-head attention mechanism, the mathematical reshaping 
    and splitting serve a very specific purpose:
    to enable each head to focus on different aspects of the input,
    such as semantics, grammar, or other features. 
    Let’s break down how this is achieved mathematically and why it work
    """
    hidden_dim, m = Q.shape

    # Ensure hidden_dim is divisible by num_heads
    assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
    depth = hidden_dim // num_heads # how many features should be allocated to each head

    # Split Q, K, V into multiple heads
    Q_split = Q.reshape(num_heads, depth, m)
    K_split = K.reshape(num_heads, depth, m)
    V_split = V.reshape(num_heads, depth, m)

    # Calculate attention for each head
    heads = []
    for i in range(num_heads):
        head_output, _ = attention(Q_split[i], K_split[i], V_split[i])
        heads.append(head_output)

    # Concatenate the heads
    concatenated_heads = np.concatenate(heads, axis=0)  # Shape: (hidden_dim, m)

    # Optional: Apply a final linear transformation (W_O)
    # W_O can be implemented as a linear transformation here

    return concatenated_heads

def forward_pass_with_complex_attention(X, params):
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']

    # Step 1: Hidden layer activation
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)  # Activation for hidden layer

    # Step 2: Multi-head attention layer
    Q, K, V = A1, A1, A1  # Using A1 as Q, K, V for simplicity
    A1_attn = multi_head_attention(Q, K, V, num_heads=4)  # Multi-head attention

    # Step 3: Output layer activation
    Z2 = np.dot(W2, A1_attn) + b2
    A2 = softmax(Z2)  # Activation for output layer

    cache = {'Z1': Z1, 'A1': A1, 'A1_attn': A1_attn, 'Z2': Z2, 'A2': A2}
    return A2, cache

