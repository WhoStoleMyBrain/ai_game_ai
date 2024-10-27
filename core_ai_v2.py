import torch
import torch.nn as nn
import torch.onnx
import onnx

class CoreAIModule(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CoreAIModule, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Define model parameters
input_size = 10   # Example input feature size
hidden_size = 20
output_size = 5   # Number of possible actions

# Initialize the model
model = CoreAIModule(input_size, hidden_size, output_size)

# Dummy input for tracing
dummy_input = torch.randn(1, input_size)

# Export the model to ONNX format
model.eval()
torch.onnx.export(
    model, 
    dummy_input, 
    "core_ai_model.onnx", 
    export_params=True,
    do_constant_folding=True,
    input_names=['input'], 
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print("Model exported to core_ai_model.onnx")
