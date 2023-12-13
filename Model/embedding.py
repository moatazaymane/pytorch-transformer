import torch.nn
import math


class InputEmbedding(torch.nn.Module):

    def __init__(self, vocab: int, dmodel: int):
        super().__init__()
        self.dmodel = dmodel
        self.vocab = vocab
        self.embedding = torch.nn.Embedding(vocab, dmodel)

    def forward(self, input_sentence) -> torch.nn.Embedding:
        return self.embedding(input_sentence) * math.sqrt(self.dmodel)
