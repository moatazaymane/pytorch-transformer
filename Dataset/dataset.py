import torch.nn
from torch.utils.data import Dataset
from loguru import logger

class TrDataset(Dataset):

    def __init__(self, inp_tokenizer, out_tokenizer, ds, sequence_length, inp_lang, out_lang):
        super().__init__()
        self.inp_tokenizer = inp_tokenizer
        self.out_tokenizer = out_tokenizer
        self.ds = ds
        self.sequence_length = sequence_length
        self.inp_lang = inp_lang
        self.out_lang = out_lang

        self.sos = torch.tensor([inp_tokenizer.token_to_id('[SOS]')], dtype=torch.int64)  # int64 because ....
        self.eos = torch.tensor([inp_tokenizer.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad = torch.tensor([inp_tokenizer.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        mapping_inp_out = self.ds[index]
        inp_text = mapping_inp_out['translation'][self.inp_lang]
        out_text = mapping_inp_out['translation'][self.out_lang] # ytrue

        input_tokens_encoder = self.inp_tokenizer.encode(inp_text).ids
        logger.debug(f"input tokens {input_tokens_encoder }")
        logger.debug(f"pad {self.pad}")
        logger.debug(f"input tokens {input_tokens_encoder }")
        logger.debug(f"bool tensor shape {input_tokens_encoder != torch.tensor(self.pad)}")
        input_tokens_decoder = self.out_tokenizer.encode(out_text).ids

        # truncate the input sentence to fit int the context window
        if self.sequence_length - len(input_tokens_encoder) - 2 < 0:
            input_tokens_encoder = input_tokens_encoder[0:self.sequence_length - 2]
        if self.sequence_length - len(input_tokens_decoder) - 1 < 0:
            input_tokens_decoder = input_tokens_decoder[0:self.sequence_length - 1]

        num_padding_encoder, num_padding_decoder = (self.sequence_length - len(input_tokens_encoder) - 2,
                                                    self.sequence_length - len(input_tokens_decoder) - 1)

        encoder_input = torch.cat(
            [self.sos, torch.tensor(input_tokens_encoder, dtype=torch.int64), self.eos,
             torch.tensor([self.pad] * num_padding_encoder)]
        )

        decoder_input = torch.cat(
            [self.sos, torch.tensor(input_tokens_decoder, dtype=torch.int64),
             torch.tensor([self.pad]*num_padding_decoder)]
        )

        target = torch.cat([
            torch.tensor(input_tokens_decoder, dtype=torch.int64),
            self.eos,
            torch.tensor([self.pad]*num_padding_decoder, dtype=torch.int64)
        ])

        input_masks_encoder = ((encoder_input != self.pad).unsqueeze(0).unsqueeze(0).int())   # adding batch dim (1, 1, sequence_length)
        input_masks_decoder = torch.triu(torch.ones(1, decoder_input.size(0), decoder_input.size(0)), diagonal=1)
        logger.info(f"input mask encoder {input_masks_encoder.shape} and input mask decoder {input_masks_decoder.shape} ")


        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": input_masks_encoder,
            "decoder_mask": input_masks_decoder,
            "target": target,
            "inp_text": inp_text,
            "out_text": out_text
        }
