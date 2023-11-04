import torch.nn
from layer_normalization import AddNorm


class SkipConnection(torch.nn.Module):

    def __init__(self, dropout):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.add_norm = AddNorm()

    def forward(self, inp, layer):
        out = self.add_norm(inp)
        out = layer(out)
        out = self.dropout(out)
        return inp + out
