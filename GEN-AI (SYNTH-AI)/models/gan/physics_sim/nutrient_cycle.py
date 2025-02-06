class NutrientCycleSimulator(nn.Module):
    def __init__(
        self,
        num_nutrients: int = 5,
        hidden_dim: int = 256,
        time_steps: int = 100
    ):
        super().__init__()
        
        self.num_nutrients = num_nutrients
        self.time_steps = time_steps
        
        # Neural ODE for nutrient dynamics
        self.dynamics_net = nn.Sequential(
            nn.Linear(num_nutrients, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_nutrients)
        )
        
    def forward(self, initial_state: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
        batch_size = initial_state.shape[0]
        timeline = torch.linspace(0, self.time_steps * dt, self.time_steps)
        
        # Runge-Kutta 4th order integration
        states = [initial_state]
        current_state = initial_state
        
        for t in timeline[1:]:
            k1 = self.dynamics_net(current_state)
            k2 = self.dynamics_net(current_state + dt * k1 / 2)
            k3 = self.dynamics_net(current_state + dt * k2 / 2)
            k4 = self.dynamics_net(current_state + dt * k3)
            
            current_state = current_state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
            states.append(current_state)
        
        return torch.stack(states, dim=1)