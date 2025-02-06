class SymbiosisPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int = 100,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.species_embedding = nn.Linear(input_dim, input_dim)
        
        encoder_layer = TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.symbiosis_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 3)  # 3 classes: beneficial, neutral, harmful
        )
        
    def forward(self, species_pair: torch.Tensor) -> torch.Tensor:
        # species_pair shape: (batch_size, 2, input_dim)
        embedded = self.species_embedding(species_pair)
        transformed = self.transformer(embedded)
        # Pool the transformer outputs
        pooled = torch.mean(transformed, dim=1)
        symbiosis_scores = self.symbiosis_head(pooled)
        return F.softmax(symbiosis_scores, dim=-1)