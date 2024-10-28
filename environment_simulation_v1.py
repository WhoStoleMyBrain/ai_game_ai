import random
import time
import torch
from enum import Enum

class TaskType(Enum):
    EVALUATE_RESOURCES = "Evaluate Ressources"
    MANAGE_SETTLEMENTS = "Manage Settlements"
    UPDATE_WEATHER = "Update Weather"
    EVALUATE_SOCIAL_STABILITY = "Evaluate Societal Stability"
    NO_TASK = "No Task"  # Added "No Event"

# Define the world state
class WorldState:
    def __init__(self):
        self.time = 0  # Starting timestamp

    def to_tensor(self):
        # Convert time step to tensor for model input; other inputs are placeholders for simplicity
        return torch.tensor([self.time, 0, 0, 0, 0], dtype=torch.float32)

# Event Generator
class TaskGenerator:
    def __init__(self):
        self.events = []
    def check_tasks(self):
        # Assign probabilities to tasks for a controlled distribution
        random_value = random.random()
        if random_value < 0.05:  # 5% chance to generate a resource check task
            return TaskType.EVALUATE_RESOURCES
        elif random_value < 0.10:  # Next 5% for settlement management
            return TaskType.MANAGE_SETTLEMENTS
        elif random_value < 0.15:  # 5% for weather update
            return TaskType.UPDATE_WEATHER
        elif random_value < 0.20:  # 5% for social stability evaluation
            return TaskType.EVALUATE_SOCIAL_STABILITY
        return TaskType.NO_TASK  # No task most of the time


# Simulation Loop
def run_simulation(steps=10000) -> tuple[torch.Tensor, TaskType]:
    training_data = []
    world_state = WorldState()
    event_generator = TaskGenerator()
    for _ in range(steps):
        # Check for new events
        task = event_generator.check_tasks()
        training_data.append((world_state.to_tensor(), task))
    return training_data