import torch
import torch.nn as nn
from functools import reduce
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


