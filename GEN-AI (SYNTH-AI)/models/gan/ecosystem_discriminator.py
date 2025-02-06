class EcosystemDiscriminator(nn.Module):
    def __init__(
        self,
        input_dim: int = 50,
        hidden_dims: List[int] = [1024, 512, 256],
    ):
        super().__init__()
        
        # Build discriminator layers
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            current_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(current_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, species_data: torch.Tensor) -> torch.Tensor:
        return self.model(species_data)
