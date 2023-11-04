import torch.nn
from positional_encoding import PosEncoding
from embedding import Embedding
from projection import ProjectionLayer
from decoder import Decoder, DecoderModule
from masked_attention import MaskedAttention
from feed_forward import FeedF


class GPT(torch.nn.Module):

    def __init__(self, embedding_layer: Embedding, positional_encoding_layer: PosEncoding,
                 projection_layer: ProjectionLayer, decoder: Decoder):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.positional_encoding_layer = positional_encoding_layer
        self.projection_layer = projection_layer
        self.decoder = decoder

    def forward(self, inp, mask):
        inp = self.embedding_layer(inp)
        inp = self.positional_encoding_layer(inp)
        inp = self.decoder(inp, mask)
        out = self.projection_layer(inp)
        return out


def Instantiate_GPT(width, context_size, vocab_size, h, hidden, Nx, dropout: float) -> GPT:

    embedding_layer = Embedding(vocab_size, width)
    poe_layer = PosEncoding(context_size, width, dropout)
    decoders = torch.nn.ModuleList([])
    for _ in range(Nx):
        attention_layer = MaskedAttention(width, h, dropout)
        feed_forward_layer = FeedF(hidden, width, dropout)
        decoder = DecoderModule(attention_layer, feed_forward_layer, dropout)
        decoders.append(decoder)

    proj = ProjectionLayer(width, vocab_size)
    decoder = Decoder(decoders)

    gpt = GPT(embedding_layer, poe_layer, proj, decoder)

    for param in gpt.parameters():
        torch.nn.init.normal_(param, 0, 0.02)  # following the gpt1 paper

    return gpt
