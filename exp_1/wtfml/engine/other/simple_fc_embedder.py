import torch.nn as nn
import torch.nn.functional as F


class MetaDataNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=50):
        super(MetaDataNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
