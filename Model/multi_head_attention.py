import torch.nn
import math
from loguru import logger


def attention(query, key, value, dropout: torch.nn.Dropout, mask=None):
    dk = query.shape[-1]
    scores = (query @ key.transpose(-2, -1)) / math.sqrt(dk)  # (batch, h, seq_len, dk) = > (Batch, h, seq_len, seq_len)
    if mask is not None:
        logger.debug(f"Dk is {(dk, query.shape, key.shape, key.transpose(-2, -1).shape, scores.shape)}")
        logger.debug(f"this is the mask {mask}")
        logger.debug(f"this is the shape of the mask {mask.shape}")
        logger.debug(f"this is the shape of the scores {scores.shape}")


        scores.masked_fill_(mask == 0, -1e9)
    scores = scores.softmax(dim=-1)  # (batch, h, seq_len, seq_len)
    if dropout is not None:
        scores = dropout(scores)

    return (scores @ value), scores


class MultiHeadA(torch.nn.Module):

    def __init__(self, dmodel: int, h: int, dropout: float):
        super().__init__()
        self.scores = None
        self.dmodel = dmodel
        self.h = h
        self.dropout = torch.nn.Dropout(dropout)

        self.dk = self.dmodel // h
        self.WQ = torch.nn.Linear(dmodel, dmodel)
        self.WK = torch.nn.Linear(dmodel, dmodel)
        self.WV = torch.nn.Linear(dmodel, dmodel)
        self.WO = torch.nn.Linear(dmodel, dmodel)

    def forward(self, q, k, v, mask=None):
        query, key, value = self.WQ(q), self.WK(k), self.WV(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.dk).transpose(1, 2)

        key = key.view(key.shape[0], key.shape[1], self.h, self.dk).transpose(1, 2)

        value = value.view(value.shape[0], value.shape[1], self.h, self.dk).transpose(1, 2)

        logger.debug(f"shape of query, key and value is {(query.shape, key.shape, value.shape)}")

        x, self.scores = attention(query, key, value, self.dropout, mask)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.dk)
        return self.WO(x)
