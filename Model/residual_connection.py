import torch.nn
from Model.layer_norm import AddNorm


class ResidualC(torch.nn.Module):

    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.addn = AddNorm()

    def forward(self, x, prev_layer):
        out = self.addn(x)
        out = prev_layer(out)
        out = self.dropout(out)

        return x + out

