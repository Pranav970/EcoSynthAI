import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeciesGenerator(nn.Module):
    def __init__(self, input_dim=100, trait_dimensions=50):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, trait_dimensions),
            nn.Tanh()
        )
    
    def forward(self, environmental_conditions):
        return self.generator(environmental_conditions)

class SpeciesDiscriminator(nn.Module):
    def __init__(self, trait_dimensions=50):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(trait_dimensions, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, species_traits):
        return self.discriminator(species_traits)