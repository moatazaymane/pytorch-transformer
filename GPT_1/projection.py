import torch.nn


class ProjectionLayer(torch.nn.Module):

    def __init__(self, width, vocab_size):
        super().__init__()
        self.map = torch.nn.Linear(width, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.map(x), dim=-1)
