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
        self.population = 1000  # Initial population
        self.resources = 500     # Resource level
        self.political_tension = 0.1  # Scale of 0 to 1
        self.volcanic_activity = 0.0  # Scale of 0 to 1
        self.time = 0  # Starting timestamp
    
    def update(self):
        # Simulate the gradual increase or decrease of key factors
        self.population += random.randint(-5, 10)
        self.resources += random.uniform(-10 - self.population/25000., 10 - self.population/25000.)
        self.political_tension += random.uniform(-0.01, 0.013)
        self.volcanic_activity += random.uniform(-0.01, 0.011)
        self.time += 1  # Simulate time passing
    
    def to_tensor(self):
        # Convert state to tensor for AI model input
        return torch.tensor([self.population, self.resources, self.political_tension, self.volcanic_activity, self.time], dtype=torch.float32)

# Event Generator
class TaskGenerator:
    def __init__(self):
        self.events = []

    def check_tasks(self, world_state: WorldState):
        # Conditions to trigger tasks based on world state
        if world_state.resources < 200:
            return TaskType.EVALUATE_RESOURCES
        elif world_state.population < 900:
            return TaskType.MANAGE_SETTLEMENTS
        elif world_state.volcanic_activity > 0.75:
            return TaskType.UPDATE_WEATHER
        elif world_state.political_tension > 0.85:
            return TaskType.EVALUATE_SOCIAL_STABILITY
        return TaskType.NO_TASK


# Simulation Loop
def run_simulation(steps=10000) -> tuple[torch.Tensor, TaskType]:
    training_data = []
    world_state = WorldState()
    event_generator = TaskGenerator()
    

    for _ in range(steps):
        world_state.update()

        # Check for new events
        task = event_generator.check_tasks(world_state)
        
        
        training_data.append((world_state.to_tensor(), task))

    return training_data