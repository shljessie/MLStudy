import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=4):
        super().__init__()
        self.original_layer = original_layer
        d_model, d_k = original_layer.weight.size()
        
        # Initialize low-rank matrices
        self.A = nn.Parameter(torch.randn(d_model, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, d_k) * 0.01)
        
    def forward(self, x):
        # Compute low-rank update
        delta = torch.matmul(x, torch.matmul(self.A, self.B))
        # Add the low-rank update to the frozen layer output
        return self.original_layer(x) + delta

# Example usage
original_layer = nn.Linear(768, 768)  # Assume 768-dim transformer layer
lora_layer = LoRALayer(original_layer, rank=4)

# Forward pass
x = torch.randn(1, 768)  # Example input
output = lora_layer(x)
