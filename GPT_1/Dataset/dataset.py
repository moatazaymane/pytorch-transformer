from torch.utils.data import Dataset
import torch


class GPTDataset(Dataset):

    def __init__(self, tokenizer, ds, context_size):
        self.ds = ds
        self.tokenizer = tokenizer
        self.context_size = context_size
        self.pad = torch.tensor([tokenizer.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, item_idx):
        item_list = self.ds[item_idx]
        text = " ".join(item_list)
        tokens = self.tokenizer.encode(text).ids
        # truncate sentence to fin in the context size
        if self.context_size - len(tokens) + 1 < 0:
            tokens = tokens[0:self.context_size + 1]

        inputs = torch.cat(
            [torch.tensor(tokens[:-1], dtype=torch.int64),
             torch.tensor([self.pad]*(self.context_size - len(tokens) + 1))]
        )
        target = torch.cat(
            [torch.tensor(tokens[1:], dtype=torch.int64),
             torch.tensor([self.pad]*(self.context_size - len(tokens) + 1))]
        )

        input_mask = torch.triu(torch.ones(1, inputs.size(0), inputs.size(0)), diagonal=1)

        return {
            "input": inputs,
            "target": target,
            "mask": input_mask,
            "next_token": tokens[-1]
        }
