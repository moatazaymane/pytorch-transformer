import torch.nn


class AddNorm(torch.nn.Module):

    def __init__(self, epsilon: float = 10 ** -6):
        super().__init__()
        self.beta = torch.nn.Parameter(torch.ones(1))
        self.epsilon = epsilon
        self.gamma = torch.nn.Parameter(torch.zeros(1))

    def forward(self, inp) -> torch.nn.Parameter:
        mean = inp.mean(dim=(0, 2), keepdim=True)
        std = inp.std(dim=(0, 2), keepdim=True)
        return self.gamma * (inp - mean) / (std + self.epsilon) + self.beta
