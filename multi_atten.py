
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