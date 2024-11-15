import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, dim=768, depth=12, heads=12, mlp_dim=3072, dropout=0.1):
        super(VisionTransformer, self).__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"

        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = (3 * patch_size * patch_size)  # Assuming 3-channel input

        # Linear projection of patches
        self.to_patch_embedding = nn.Linear(self.patch_dim, dim)

        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)

        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout),
            num_layers=depth
        )

        # MLP head for classification
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # Reshape image into patches
        batch_size, channels, height, width = img.shape
        assert height == width == self.patch_size * (height // self.patch_size), "Image dimensions must be divisible by patch size"
        patches = img.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(batch_size, -1, self.patch_dim)

        # Patch embeddings
        patch_embeddings = self.to_patch_embedding(patches)

        # Add class token and positional embeddings
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, dim)
        x = torch.cat((cls_tokens, patch_embeddings), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)

        # Transformer
        x = self.transformer(x)

        # Classification output
        cls_output = x[:, 0]  # Extract the class token
        out = self.mlp_head(cls_output)

        return out

# Example Usage
if __name__ == "__main__":
    model = VisionTransformer(img_size=224, patch_size=16, num_classes=10, dim=768, depth=12, heads=12, mlp_dim=3072)
    sample_image = torch.randn(1, 3, 224, 224)  # Batch size of 1, RGB image
    output = model(sample_image)
    print("Output shape:", output.shape)  # Expected: [1, 10]
