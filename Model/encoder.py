import torch.nn
from Model.multi_head_attention import MultiHeadA
from Model.layer_norm import AddNorm
from Model.residual_connection import ResidualC
from Model.feed_forward import FeedF


class EncoderModule(torch.nn.Module):

    def __init__(self, attention_layer: MultiHeadA, feed_forward: FeedF, dropout: float):
        super().__init__()
        self.multi_h_attention_layer = attention_layer
        self.feed_forward = feed_forward
        self.skip_connection = torch.nn.ModuleList([ResidualC(dropout) for _ in range(2)])

    def forward(self, x, input_mask):
        x = self.skip_connection[0](x, lambda y: self.multi_h_attention_layer(y, y, y, input_mask))
        x = self.skip_connection[1](x, self.feed_forward)
        return x


class Encoder(torch.nn.Module):

    def __init__(self, layers: torch.nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.add_norm = AddNorm()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.add_norm(x)
