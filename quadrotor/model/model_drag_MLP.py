import torch
import torch.nn as nn

class Drag_MLP(nn.Module):
    def __init__(self, input_size=6, hidden_size=36, output_size=3):
        super(Drag_MLP, self).__init__()
        self.input_layer = nn.Linear(
            input_size, hidden_size, bias=True, dtype=torch.float32
        )
        self.hidden_layer1 = nn.Linear(
            hidden_size, hidden_size, bias=True, dtype=torch.float32
        )
        self.hidden_layer2 = nn.Linear(
            hidden_size, hidden_size, bias=True, dtype=torch.float32
        )
        self.output_layer = nn.Linear(
            hidden_size, output_size, bias=False, dtype=torch.float32
        )
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.hidden_layer1(x)
        x = self.activation(x)
        x = self.hidden_layer2(x)
        x = self.activation(x)
        x = self.output_layer(x)

        return x