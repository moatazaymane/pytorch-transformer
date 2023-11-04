import torch.nn


class Embedding(torch.nn.Module):

    def __init__(self, vocab_size, width):
        super().__init__()
        self.width = width
        self.vocab_size = vocab_size
        self.embedding_layer = torch.nn.Embedding(vocab_size, self.width)

    def forward(self, inp):
        out = self.embedding_layer(inp)
        return out
