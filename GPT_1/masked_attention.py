import torch.nn
import math
from loguru import logger


def attention(q, k, v, dropout: torch.nn.Dropout, mask):
    qk = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    qk.masked_fill_(mask == 0, -1e9)
    qk = qk.softmax(dim=1)
    qk = dropout(qk)
    return qk @ v, qk


class MaskedAttention(torch.nn.Module):

    def __init__(self, width, h, dropout):
        super().__init__()
        self.scores = None
        self.width = width
        self.h = h
        self.dropout = torch.nn.Dropout(dropout)
        self.WQ = torch.nn.Linear(width, width)
        self.WK = torch.nn.Linear(width, width)
        self.WV = torch.nn.Linear(width, width)
        self.WO = torch.nn.Linear(width, width)

    def forward(self, inp, mask):
        q, k, v = inp, inp, inp
        q, k, v = self.WQ(q), self.WK(k), self.WV(v)
        q = q.view(q.shape[0], q.shape[1], self.h, self.width // self.h).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.h, self.width // self.h).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], self.h, self.width // self.h).transpose(1, 2)
        logger.debug(f"shape of query, key and value is {(q.shape, k.shape, v.shape)}")

        x, self.scores = attention(q, k, v, self.dropout, mask)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.width)
        logger.debug(f"shape of softmax(QK(transpose))@val {x.shape}")

        return self.WO(x)
