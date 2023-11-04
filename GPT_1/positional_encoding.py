import torch.nn
import math


class PosEncoding(torch.nn.Module):

    def __init__(self, sequence_length: int, width: int, dropout: float):
        super().__init__()
        self.width = width
        self.sequence_length = sequence_length
        self.dropout = torch.nn.Dropout(dropout)

        encoding = torch.ones(sequence_length, width)  # matrix of size (sequence_length,b size, width)
        pos = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)  # vector of shape (sequence_length, 1)
        div = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000.0) / width))

        encoding[:, 0::2] = torch.sin(pos * div)
        encoding[:, 1::2] = torch.cos(pos * div)

        encoding = encoding.unsqueeze(0)  # (1, sequence_length, width)
        self.register_buffer('encoding', encoding)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:

        inp = inp + (self.encoding[:, :inp.shape[1], :]).requires_grad_(False)
        return self.dropout(inp)
