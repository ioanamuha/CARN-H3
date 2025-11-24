import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self, num_classes, input_channels=3, input_size=32):
        super().__init__()
        flat_dim = input_channels * input_size * input_size

        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(flat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)
