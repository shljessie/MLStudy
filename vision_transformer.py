import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # Scaling factor for dot product attention

        # Linear layers for queries, keys, and values
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape  # Batch size, number of tokens, embedding dimension
        H = self.heads

        # Compute queries, keys, and values
        q = self.to_q(x).view(B, N, H, D // H).transpose(1, 2)  # (B, H, N, D // H)
        k = self.to_k(x).view(B, N, H, D // H).transpose(1, 2)  # (B, H, N, D // H)
        v = self.to_v(x).view(B, N, H, D // H).transpose(1, 2)  # (B, H, N, D // H)

        # Calculate scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn_weights = attn_scores.softmax(dim=-1)  # Softmax to get attention weights

        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)  # (B, H, N, D // H)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, D)  # Combine heads

        # Final linear layer
        return self.to_out(attn_output)  # (B, N, D)


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, dim=768, depth=12, heads=12, mlp_dim=3072):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.num_patches = (img_size // patch_size) ** 2

        # Linear projection of flattened patches
        self.to_patch_embedding = nn.Linear(patch_size * patch_size * 3, dim)

        # Positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))

        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Self-attention layers
        self.layers = nn.ModuleList([
            nn.ModuleList([
                SelfAttention(dim, heads=heads),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, mlp_dim),
                    nn.GELU(),
                    nn.Linear(mlp_dim, dim),
                ),
                nn.LayerNorm(dim)
            ])
            for _ in range(depth)
        ])

        # MLP head for classification
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # Reshape image into patches and embed them
        x = x.view(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(B, -1, self.patch_size * self.patch_size * C)
        x = self.to_patch_embedding(x)

        # Concatenate class token and add positional embeddings
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding

        # Pass through transformer layers
        for attn, norm1, mlp, norm2 in self.layers:
            x = attn(norm1(x)) + x  # Apply attention with residual connection
            x = mlp(norm2(x)) + x  # Apply MLP with residual connection

        # Classification token output
        x = x[:, 0]  # Get the class token embedding
        return self.mlp_head(x)

# Testing the Vision Transformer model
model = VisionTransformer()
x = torch.randn(1, 3, 224, 224)  # Example input (batch of images)
output = model(x)
print(output.shape)  # Expected output: (1, num_classes)
