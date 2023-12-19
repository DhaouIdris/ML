import torch
import torch.nn as nn
from functools import reduce
import torch.nn.functional as F
import operator




def linear(config, input_size, output_size):
    """
    congig      -- Configuration parameters from the yaml file
    input_size  -- tuple
    output_size -- int

    """

    layers = [
        nn.Flatten(start_dim=1),
        nn.Linear(reduce(operator.mul, input_size, 1), output_size),
    ]
    return nn.Sequential(*layers)

###  test wether the model is working or not
def test_linear():
    cfg = {"class": "Linear"}
    batch_size = 64
    input_size = (3, 32, 32)
    num_classes = 18
    model = linear(cfg, input_size, num_classes)

    input_tensor = torch.randn(batch_size, *input_size)
    output = model(input_tensor)
    print(f"Output tensor of size : {output.shape}")

class LinearNet(nn.Module):
    def __init__(self, cfg, input_size, num_classes):
        super(LinearNet, self).__init__()
        self.input_size = reduce(operator.mul, input_size, 1)
        self.classifier = nn.Linear(self.input_size, num_classes)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        y = self.classifier(x)
        return y


class ConvNet(nn.Module):
    def __init__(self, config, input_size, num_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
