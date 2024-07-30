from collections import *

from torch import nn


class MLP(nn.Module):
    def __init__(self, ECFP_dim, hidden_dim, output_dim):
        super().__init__()
        self.ECFP_dim = ECFP_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        layer_dim = [self.ECFP_dim, *self.hidden_dim]

        layers = OrderedDict()
        for layer_idx, (in_dim, out_dim) in enumerate(zip(layer_dim[:-1], layer_dim[1:])):
            layers[f"fc layer {layer_idx}"] = nn.Linear(in_dim, out_dim)
            layers[f"relu {layer_idx}"] = nn.ReLU()
        self.represent_layers = nn.Sequential(layers)
        self.fc_layers = nn.Linear(layer_dim[-1], self.output_dim)

    def represent(self, x):
        return self.represent_layers(x)

    def forward(self, x):
        x = self.represent(x)
        return self.fc_layers(x)
