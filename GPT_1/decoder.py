import torch.nn
from masked_attention import MaskedAttention
from feed_forward import FeedF
from skip_connection import SkipConnection
from layer_normalization import AddNorm


class DecoderModule(torch.nn.Module):

    def __init__(self, attention_layer: MaskedAttention, feed_forward_layer: FeedF, dropout: float):
        super().__init__()
        self.attention_layer = attention_layer
        self.feed_forward_layer = feed_forward_layer
        self.dropout = dropout
        self.skip_connection = torch.nn.ModuleList([SkipConnection(dropout) for _ in range(2)])

    def forward(self, inp, mask):

        out = self.skip_connection[0](inp, lambda y: self.attention_layer(y, mask))
        out = self.skip_connection[1](out, self.feed_forward_layer)

        return out


class Decoder(torch.nn.Module):

    def __init__(self, decoders: torch.nn.ModuleList):
        super().__init__()
        self.decoders = decoders
        self.add_norm = AddNorm()


    def forward(self, inp, mask):
        for decoder in self.decoders:
            inp = decoder(inp, mask)
        return self.add_norm(inp)
