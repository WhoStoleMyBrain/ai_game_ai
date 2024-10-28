import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from environment_simulation_v1 import TaskType, run_simulation

training_data = run_simulation()

train_summary = {event_type: len([training_dat for training_dat in training_data if training_dat[1] == event_type]) for event_type in TaskType}
print(f'training summary:')
for key in train_summary:
    print(f'{key}: {train_summary[key]} events')
    
# task_label_mapping = {
#     TaskType.EVALUATE_RESOURCES.value: 0,
#     TaskType.MANAGE_SETTLEMENTS.value: 1,
#     TaskType.UPDATE_WEATHER.value: 2,
#     TaskType.EVALUATE_SOCIAL_STABILITY.value: 3,
#     TaskType.NO_TASK.value: 4  # "No Event" is added as a valid label
# }

task_label_mapping = {task.value: i for i, task in enumerate(TaskType)}
print(f'task label mapping: {task_label_mapping}')


class CoreAIModel(nn.Module):
    def __init__(self):
        super(CoreAIModel, self).__init__()
        self.fc1 = nn.Linear(5, 128)  # Input size 5 (world state)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, len(task_label_mapping))   # Output size 4 (war, volcanic eruption, political unrest, no_event)
        self.softmax = nn.Softmax(dim=1)  # Softmax layer for probabilities

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)  # Apply softmax to get probabilities
        return x


# Assuming you have `training_data` from your simulation
def train_model(model, training_data: tuple[torch.Tensor, TaskType], epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()  # For classification

    # Prepare training data
    states, labels = zip(*training_data)
    labels_encoded = [task_label_mapping[label.value] for label in labels]  # Convert labels to integers
    dataset = TensorDataset(torch.stack(states), torch.tensor(labels_encoded, dtype=torch.long))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Training loop
    for epoch in range(epochs):
        for batch_states, batch_labels in dataloader:
            optimizer.zero_grad()
            output = model(batch_states)
            loss = loss_fn(output, batch_labels)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Initialize and train the model
model = CoreAIModel()
train_model(model, training_data)

input_size = 5   # Example input feature size
dummy_input = torch.randn(1, input_size)
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