import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    def forward(self, Q, K, V):
        dk = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dk)
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, V), attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dk = d_model // num_heads

        # Linear layers for Q, K, V and the final output projection
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        # Split the last dimension into (num_heads, dk)
        x = x.view(batch_size, -1, self.num_heads, self.dk)
        return x.transpose(1, 2)  # (batch_size, num_heads, seq_len, dk)

    def forward(self, Q, K, V):
        batch_size = Q.size(0)

        # Apply linear transformations and split into heads
        Q = self.split_heads(self.Wq(Q), batch_size)
        K = self.split_heads(self.Wk(K), batch_size)
        V = self.split_heads(self.Wv(V), batch_size)

        # Scaled dot-product attention
        attn_output, _ = ScaledDotProductAttention()(Q, K, V)

        # Concatenate heads and apply final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.dk)
        return self.Wo(attn_output)

class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, dff):
        super(FeedForwardLayer, self).__init__()
        self.linear1 = nn.Linear(d_model, dff)
        self.linear2 = nn.Linear(dff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dff):
        super(TransformerBlock, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardLayer(d_model, dff)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
