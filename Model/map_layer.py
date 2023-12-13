import torch.nn


class MapLayer(torch.nn.Module):

    def __init__(self, dmodel, vocab_size):
        super().__init__()
        self.map = torch.nn.Linear(dmodel, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.map(x), dim=-1)
