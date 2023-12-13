import torch.nn
from Model.pos_encoding import PosEncoding
from Model.embedding import InputEmbedding
from Model.encoder import Encoder
from Model.decoder import Decoder
from Model.map_layer import MapLayer
from Model.encoder import EncoderModule
from Model.decoder import DecoderModule
from Model.feed_forward import FeedF
from Model.multi_head_attention import MultiHeadA


class Net(torch.nn.Module):

    def __init__(self, input_embedding: InputEmbedding, out_embedding: InputEmbedding, input_pos: PosEncoding,
                 out_pos: PosEncoding, encoder: Encoder, decoder: Decoder, map_layer: MapLayer):
        super().__init__()
        self.input_embedding = input_embedding
        self.input_pos = input_pos
        self.out_embedding = out_embedding
        self.out_pos = out_pos
        self.encoder = encoder
        self.decoder = decoder
        self.map_layer = map_layer

    def forward_e(self, inp, input_mask):
        out = self.input_embedding(inp)
        out = self.input_pos(out)
        out = self.encoder(out, input_mask)
        return out

    def forward_d(self, out, input_mask, out_mask, out_encoder):
        res = self.out_embedding(out)
        res = self.input_pos(res)
        res = self.decoder(res, out_encoder, input_mask, out_mask)
        return res

    def map_to_vocab(self, x):
        res = self.map_layer(x)
        return res


def instance_Transformer(vocab_size, out_vocab_size, seq_length, out_seq_length, dmodel, dff, dropout, Nx, h) -> Net:
    input_embedding = InputEmbedding(vocab_size, dmodel)
    out_embedding = InputEmbedding(out_vocab_size, dmodel)
    input_pos = PosEncoding(seq_length, dmodel, dropout)
    out_pos = PosEncoding(out_seq_length, dmodel, dropout)

    layers_encoder, layers_decoder = torch.nn.ModuleList([]), torch.nn.ModuleList([])

    for _ in range(Nx):
        multi_head, masked_multi_head, cross_attention = (
            MultiHeadA(dmodel, h, dropout), MultiHeadA(dmodel, h, dropout),
            MultiHeadA(dmodel, h, dropout))
        feed_forward_encoder, feed_forward_decoder = FeedF(dff, dmodel, dropout), FeedF(dff, dmodel, dropout)
        layers_encoder.append(EncoderModule(multi_head, feed_forward_encoder, dropout))
        layers_decoder.append(DecoderModule(masked_multi_head, cross_attention, feed_forward_decoder, dropout))

    encoder, decoder = Encoder(layers_encoder), Decoder(layers_decoder)

    map_layer = MapLayer(dmodel, out_vocab_size)

    # Transformer:
    net = Net(input_embedding, out_embedding, input_pos, out_pos, encoder, decoder, map_layer)

    for param in net.parameters():
        torch.nn.init.xavier_uniform(param) if param.dim() > 1 else torch.nn.init.uniform_(param, 0, 1)
    return net
