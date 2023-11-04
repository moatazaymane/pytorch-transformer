import torch.nn


class FeedF(torch.nn.Module):

    def __init__(self, hidden: int, width: int, dropout: float):
        super().__init__()
        self.linear1 = torch.nn.Linear(width, hidden)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(hidden, width)
        self.Gelu = torch.nn.GELU()

    def forward(self, inp):
        out = self.linear1(inp)
        out = self.Gelu(out)
        out = self.dropout(out)
        return self.linear2(out)
