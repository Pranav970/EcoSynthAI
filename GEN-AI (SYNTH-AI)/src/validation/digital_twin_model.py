import numpy as np
import torch
import torch.nn as nn

class EcosystemDigitalTwin(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_features)
        )
    
    def simulate_ecosystem_evolution(self, initial_conditions, timesteps=10):
        current_state = initial_conditions
        ecosystem_trajectory = [current_state]
        
        for _ in range(timesteps):
            predicted_next_state = self.network(current_state)
            ecosystem_trajectory.append(predicted_next_state)
            current_state = predicted_next_state
        
        return torch.stack(ecosystem_trajectory)