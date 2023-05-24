import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers, dropout_rate):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layers = layers
        self.dropout_rate = dropout_rate

        self.layers = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim), nn.ReLU(),
             nn.Dropout(dropout_rate)]
        )
        for _ in range(layers - 1):
            self.layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
        self.layers.extend([nn.Linear(hidden_dim, output_dim)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
