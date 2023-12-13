import torch.nn
from torchtext.data.metrics import bleu_score
from tqdm import tqdm
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
import os
from torch.utils.data import DataLoader, random_split
from Dataset.dataset import TrDataset
from Model.Net import instance_Transformer
from utils.config import (inp_lang, out_lang, train_batch_size, val_batch_size, sequence_length, dmodel,
                          train_size, tokenizer_name, version_exp, learning_rate, num_epochs, preload, model_name, Nx, dff, h)
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
logger.remove()


def write_msg(msg, iterator):
    iterator.write(msg)
    return

def greedy_outputs(model, inp, inp_mask, inp_tokenizer, out_tokenizer, sequence_length, device):
    i_sos, i_eos = inp_tokenizer.token_to_id('[SOS]'), out_tokenizer.token_to_id('[EOS]')

    # Encoder output is precomputed and reused

    encoder_output = model.forward_e(inp, inp_mask)  # (bsize, sequence_length, dmodel)

    # initialize the decoder input with the sos token
    inp_decoder = torch.empty(1, 1).fill_(i_sos).type_as(inp).to(device)

    while True:
        if inp_decoder.size(1) == sequence_length:
            break

        input_masks_decoder = torch.t(torch.triu(torch.ones(1, inp_decoder.size(0), inp_decoder.size(0)), diagonal=0))
        decoder_output = model.forward_d(inp_decoder, inp_mask, input_masks_decoder,
                                         encoder_output)

        next_token_prob = model.map_to_vocab(decoder_output[:, -1])
        _, next_token = torch.max(next_token_prob, dim=1)
        decoder_input = torch.cat([inp_decoder, torch.empty(1, 1).type_as(inp).fill_(next_token.item()).to(device)])

        if next_token == i_eos:
            break

    return inp_decoder.squeeze(0)


def infer(model, val_ds, inp_tokenizer, out_tokenizer, sequence_length, device, write, iterator, num_examples=4):
    model.eval()
    count = 0

    inp_texts, ytrue, yhat = [], [], []

    console_width = 80

    with torch.no_grad():
        for batch in val_ds:
            count+=1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            model_out = greedy_outputs(model, encoder_input, encoder_mask, inp_tokenizer, out_tokenizer, sequence_length, device)

            inp_txt = batch['inp_text'][0]
            target_txt = batch['target'][0]
            model_out_text = out_tokenizer.decode(model_out.detach().cpu().numpy())

            inp_texts.append(inp_txt)
            ytrue.append(target_txt)
            yhat.append(model_out_text)


            write('--'*console_width, iterator=iterator)
            write(f"Source Text {inp_txt}", iterator=iterator)
            write(f"Label Text {target_txt}", iterator=iterator)
            write(f"Predicted {model_out_text}", iterator=iterator)

            if count == num_examples:
                break
    '''
    if summary_writer:
        # TODO : Calculate the Bleu metric
        
    '''


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_ds = load_dataset("opus_books", f"{inp_lang}-{out_lang}")
    all_ds = all_ds['train']

    # encoder / decoder tokenizers

    langs = []
    if not os.path.isfile(f"{tokenizer_name}{inp_lang}"):
        langs.append(inp_lang)
    if not os.path.isfile(f"{tokenizer_name}{out_lang}"):
        langs.append(out_lang)

    for lang in langs:

        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        trainer = WordLevelTrainer(show_progress = True, special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.pre_tokenizer = Whitespace()
        list_els = []
        for el in all_ds:
            list_els.append(el['translation'][lang])

        logger.info(f"Saving tokenizer to path: {tokenizer_name}{lang}")
        tokenizer.train_from_iterator(list_els, trainer=trainer)
        tokenizer.save(f"{tokenizer_name}{lang}")

    logger.info(f"Loading tokenizers from paths: {tokenizer_name}{inp_lang} and {tokenizer_name}{out_lang}")
    inp_tokenizer = Tokenizer.from_file(f"{tokenizer_name}{inp_lang}")
    out_tokenizer = Tokenizer.from_file(f"{tokenizer_name}{out_lang}")

    # Dataset split

    train_ds_, val_ds_ = random_split(all_ds, [int(train_size*len(all_ds)), len(all_ds) - int(train_size*len(all_ds))])

    train_ds = TrDataset(inp_tokenizer, out_tokenizer, train_ds_, sequence_length=sequence_length,
                         inp_lang=inp_lang, out_lang=out_lang)
    val_ds = TrDataset(inp_tokenizer, out_tokenizer, val_ds_, sequence_length=sequence_length,
                       inp_lang=inp_lang, out_lang=out_lang)
    dataloader_train = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
    dataloader_val = DataLoader(val_ds, batch_size=val_batch_size)
    model = instance_Transformer(vocab_size=inp_tokenizer.get_vocab_size(), out_vocab_size=out_tokenizer.get_vocab_size(), seq_length=sequence_length,
                                 dmodel=dmodel, Nx=Nx, h=h, dropout=0.1, out_seq_length=sequence_length, dff=dff)
    summary_w = SummaryWriter(version_exp)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-9)
    objective_function = torch.nn.CrossEntropyLoss(ignore_index=out_tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    start, global_step = 0, 0

    if preload:
        state = torch.load(model_name)
        start = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    for epoch in range(start, num_epochs):
        model.train()
        iterator = tqdm(dataloader_train, desc = f"epoch {epoch:02d}")
        for batch in iterator:
            model.train()
            encoder_input = batch["encoder_input"].to(device)  # (bsize, sequence_length)
            decoder_inp = batch["decoder_input"].to(device)  # (bsize, sequence_length)
            encoder_mask = batch["encoder_mask"].to(device)  # (bsize, 1, 1, sequence_length)
            decoder_mask = batch['decoder_mask'].to(device) # (bsize, 1, sequence_length, sequence_length)

            # forward ass through the transformer model
            encoder_output = model.forward_e(encoder_input, encoder_mask)  # (bsize, sequence_length, dmodel)
            decoder_output = model.forward_d(decoder_inp, encoder_mask, decoder_mask,
                                             encoder_output)   # (bsize, sequence_length, sequence_length)

            final_output = model.map_to_vocab(decoder_output)  # (bsize, sequence_length, out_vocab_size)
            target = batch['target'].to(device) # (bsize, sequence_length)

            loss = objective_function(final_output.view(-1, out_tokenizer.get_vocab_size()), target.view(-1))
            iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            summary_w.add_scalar("train loss", loss.item(), global_step=global_step)
            summary_w.flush()

            # backprop
            loss.backward()

            # Optimizing steps
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        # val loop
        infer(model, dataloader_val, inp_tokenizer, out_tokenizer, sequence_length, device, write_msg, global_step,
              summary_w, num_examples=1)

        # Save the model
        model_filename = model_name
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step
            }, model_filename
        )


if __name__ == '__main__':
    train()
