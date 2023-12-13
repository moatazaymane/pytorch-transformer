import torch
from loguru import logger

t1 = torch.tensor([119, 417, 29, 5, 767, 7, 5, 149, 192, 4, 30, 5, 3970, 16, 24, 70, 14, 10, 385, 29, 282
, 295, 16, 41, 4779, 5, 96, 3730, 7, 1194, 3299, 9, 3802, 6])
t2 = torch.tensor([1])
logger.debug(t1 != t2)
