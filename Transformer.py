import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

class ScaledDotProductAttention:
    def __call__(self, Q, K, V):
        dk = Q.shape[-1]
        scores = np.dot(Q, K.T) / np.sqrt(dk)
        attn_weights = softmax(scores, axis=-1)
        return np.dot(attn_weights, V), attn_weights

class MultiHeadAttention:
    def __init__(self, dk, dv, num_heads):
        self.num_heads = num_heads
        self.dk = dk
        self.dv = dv
        self.Wq = np.random.rand(num_heads, dk, dk)
        self.Wk = np.random.rand(num_heads, dk, dk)
        self.Wv = np.random.rand(num_heads, dv, dv)
        self.Wo = np.random.rand(num_heads * dv, dv)

    def split_heads(self, x):
        batch_size, seq_len, d_model = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, d_model // self.num_heads)
        return np.transpose(x, (0, 2, 1, 3))

    def __call__(self, Q, K, V):
        batch_size, seq_len, _ = Q.shape
        Q_heads = self.split_heads(np.dot(Q, self.Wq))
        K_heads = self.split_heads(np.dot(K, self.Wk))
        V_heads = self.split_heads(np.dot(V, self.Wv))

        attn_output, _ = ScaledDotProductAttention()(Q_heads, K_heads, V_heads)
        attn_output = np.transpose(attn_output, (0, 2, 1, 3)).reshape(batch_size, seq_len, -1)
        return np.dot(attn_output, self.Wo)

class FeedForwardLayer:
    def __init__(self, d_model, dff):
        self.W1 = np.random.rand(d_model, dff)
        self.W2 = np.random.rand(dff, d_model)

    def __call__(self, x):
        return np.maximum(0, np.dot(x, self.W1)).dot(self.W2)  # ReLU activation

class TransformerBlock:
    def __init__(self, dk, dv, d_model, num_heads, dff):
        self.mha = MultiHeadAttention(dk, dv, num_heads)
        self.ffn = FeedForwardLayer(d_model, dff)
        self.layer_norm1 = lambda x: (x - np.mean(x)) / (np.std(x) + 1e-6)
        self.layer_norm2 = lambda x: (x - np.mean(x)) / (np.std(x) + 1e-6)

    def __call__(self, x):
        attn_output = self.mha(x, x, x)
        out1 = self.layer_norm1(x + attn_output)
        ffn_output = self.ffn(out1)
        return self.layer_norm2(out1 + ffn_output)

class Transformer:
    def __init__(self, num_blocks, dk, dv, d_model, num_heads, dff):
        self.blocks = [TransformerBlock(dk, dv, d_model, num_heads, dff) for _ in range(num_blocks)]

    def __call__(self, x):
        for block in self.blocks:
            x = block(x)
        return x

# Example usage
num_blocks = 2
dk = 64  # Dimension of queries and keys
dv = 64  # Dimension of values
d_model = 128  # Dimension of the model
num_heads = 8
dff = 256  # Dimension of feed-forward layer

# Random input: batch of 10 sequences of 20 tokens each with dimension 128
x = np.random.rand(10, 20, d_model)

transformer = Transformer(num_blocks, dk, dv, d_model, num_heads, dff)
output = transformer(x)
print("Transformer output shape:", output.shape)
