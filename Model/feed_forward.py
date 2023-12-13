import torch.nn


class FeedF(torch.nn.Module):

    def __init__(self, dff: int, dmodel: int, dropout: float):
        super().__init__()
        self.linear1 = torch.nn.Linear(dmodel, dff)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dff, dmodel)

    def forward(self, inp):
        out = self.linear1(inp)
        out = torch.relu(out)
        out = self.dropout(out)
        return self.linear2(out)