import torch
from torch import nn

class PPOPolicy(nn.Module):
    def __init__(self, input_dim,hidden_dim=64):
        super().__init__()
      
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, 1)  # mean output for steering
        )
        self.log_std = nn.Parameter(torch.zeros(1))  # trainable log std

        self.critic = nn.Sequential(
            nn.Linear(input_dim, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        
          # last time step's output
        mean = self.actor(x)
        std = self.log_std.exp()
        value = self.critic(x)
        return mean, std, value  