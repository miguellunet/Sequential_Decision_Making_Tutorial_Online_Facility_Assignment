import torch.nn as nn

# Define the neural network architecture for imitation learning

class ImitationLearningNet(nn.Module):
 
    def __init__(self, num_warehouses):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * num_warehouses, 64), nn.ReLU(),
            nn.Linear(64, 16), nn.ReLU(),
            nn.Linear(16, num_warehouses),
        )

    def forward(self, x):
        return self.net(x)