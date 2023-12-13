import torch.nn
from Model.layer_norm import AddNorm
from Model.multi_head_attention import MultiHeadA
from Model.feed_forward import FeedF
from Model.residual_connection import ResidualC


class DecoderModule(torch.nn.Module):

    def __init__(self, self_attention: MultiHeadA, cross_attention: MultiHeadA, feed_forward: FeedF, dropout: float):
        super().__init__()
        self.dropout = dropout
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.skip_connection = torch.nn.ModuleList([ResidualC(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, input_mask, out_mask):

        x = self.skip_connection[0](x, lambda y: self.self_attention(y, y, y, out_mask))
        x = self.skip_connection[1](x, lambda y: self.cross_attention(y, encoder_output, encoder_output,
                                                                      input_mask))
        x = self.skip_connection[2](x, self.feed_forward)

        return x


class Decoder(torch.nn.Module):

    def __init__(self, layers: torch.nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.add_norm = AddNorm()

    def forward(self, x, encoder_output, input_mask, out_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, input_mask, out_mask)
        return self.add_norm(x)
