import torch
import torch.nn as nn
import torch.optim as optim


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=1.0):
        super(LoRALinear, self).__init__()
        
        # Original weight matrix is frozen
        self.weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)
        
        # Low-rank components
        self.r = r
        self.alpha = alpha
        self.A = nn.Parameter(torch.randn(r, in_features))  # Low-rank A
        self.B = nn.Parameter(torch.randn(out_features, r))  # Low-rank B

    def forward(self, x):
        # Compute LoRA output: W*x + alpha * (B @ (A @ x))
        original = torch.matmul(x, self.weight.T)
        lora_update = self.alpha * torch.matmul(self.B, torch.matmul(self.A, x.T)).T
        return original + lora_update


def apply_lora(model, r=8, alpha=1.0):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            in_features, out_features = module.in_features, module.out_features
            lora_layer = LoRALinear(in_features, out_features, r, alpha)
            setattr(model, name, lora_layer)
    return model


# Example model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

# Initialize and modify model with LoRA
model = SimpleModel()
model = apply_lora(model, r=8, alpha=1.0)

# Set optimizer for LoRA parameters only
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# Training loop (example)
for epoch in range(10):
    optimizer.zero_grad()
    # Forward pass
    x = torch.randn(32, 512)  # Example input
    y = model(x)
    # Example loss
    loss = torch.mean(y)
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item()}")
