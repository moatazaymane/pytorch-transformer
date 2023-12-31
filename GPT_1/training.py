import torch.optim
import re
from datasets import load_dataset
from tqdm import tqdm
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torcheval.metrics import Perplexity
from random import randint
import os
from loguru import logger
from utils.config import (tokenizer_path,context_size, tokenizer_name, train_size, train_batch_size, val_batch_size,
                          width, hidden, h, num_epochs, Nx, learning_rate, model_name, load_pretrained, model_path)
from Dataset.dataset import GPTDataset
from torch.utils.data import DataLoader
from GPT import Instantiate_GPT


def write_msg(msg, iterator):
    iterator.write(msg)
    return


def get_next_token_greedy(gpt_model, inp, mask, device):

    inp = inp.to(dtype=torch.int64)
    mask = mask.to(dtype=torch.int64)
    next_token_probabilities = gpt_model.forward(inp.to(device), mask.to(device))
    _, next_token = torch.max(next_token_probabilities, dim=2)
    return next_token


def validate(gpt_model, dataloader_val, tokenizer, write, iterator, device):

    gpt_model.eval()
    y_true, yhat = [], []

    console_width = 70

    with torch.no_grad():
        for batch in dataloader_val:
            inp = batch['input'].to(device)
            mask = batch['mask'].to(device)
            yt = batch['next_token']
            target_token = yt
            yh = get_next_token_greedy(gpt_model, inp, mask, device)
            target_text = tokenizer.decode([yt])
            predicted_text = tokenizer.decode(yh[0].detach().cpu().tolist())
            decoded_inp = tokenizer.decode(inp[0].detach().cpu().tolist())

            y_true.append(target_text)
            yhhat = predicted_text.split(' ')[-1]

            yhat.append(yhhat)

            logger.debug(f"this is ytrue {yt}")

            write('--' * console_width, iterator=iterator)
            write(f"Target text {decoded_inp} {target_text}", iterator=iterator)
            write(f"Predicted text {predicted_text}", iterator=iterator)

        correct_predictions = 0
        logger.debug(f"yhat is {yhat}")
        logger.debug(f"ytrue is {target_text}")
        for i in range(len(y_true)):
            if yhat[i] == y_true[i]:
                correct_predictions += 1
        write(f"Next token prediction accuracy {correct_predictions/len(y_true):02f}", iterator=iterator)


import torch.optim
import re
from datasets import load_dataset
from tqdm import tqdm
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torcheval.metrics import Perplexity
from random import randint
import os
from loguru import logger
from utils.config import (tokenizer_path,context_size, tokenizer_name, train_size, train_batch_size, val_batch_size,
                          width, hidden, h, num_epochs, Nx, learning_rate, model_name, load_pretrained, model_path)
from Dataset.dataset import GPTDataset
from torch.utils.data import DataLoader
from GPT import Instantiate_GPT


def write_msg(msg, iterator):
    iterator.write(msg)
    return


def get_next_token_greedy(gpt_model, inp, mask, device):

    inp = inp.to(dtype=torch.int64)
    mask = mask.to(dtype=torch.int64)
    next_token_probabilities = gpt_model.forward(inp.to(device), mask.to(device))
    _, next_token = torch.max(next_token_probabilities, dim=2)
    return next_token


def validate(gpt_model, dataloader_val, tokenizer, write, iterator, device):

    gpt_model.eval()
    y_true, yhat = [], []

    console_width = 70

    with torch.no_grad():
        for batch in dataloader_val:
            inp = batch['input'].to(device)
            mask = batch['mask'].to(device)
            yt = batch['next_token']
            target_token = yt
            yh = get_next_token_greedy(gpt_model, inp, mask, device)
            target_text = tokenizer.decode([yt])
            predicted_text = tokenizer.decode(yh[0].detach().cpu().tolist())
            decoded_inp = tokenizer.decode(inp[0].detach().cpu().tolist())

            y_true.append(target_text)
            yhhat = predicted_text.split(' ')[-1]

            yhat.append(yhhat)

            logger.debug(f"this is ytrue {yt}")

            write('--' * console_width, iterator=iterator)
            write(f"Target text {decoded_inp} {target_text}", iterator=iterator)
            write(f"Predicted text {predicted_text}", iterator=iterator)

        correct_predictions = 0
        logger.debug(f"yhat is {yhat}")
        logger.debug(f"ytrue is {target_text}")
        for i in range(len(y_true)):
            if yhat[i] == y_true[i]:
                correct_predictions += 1
        write(f"Next token prediction accuracy {correct_predictions/len(y_true):02f}", iterator=iterator)



def get_data_translation_nxt_token():

    sentences_ds = df_.en.tolist()

    # training Tokenizer from available sentences
    if not os.path.isfile("tokenizer/gpt_tokenizer_1"):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        trainer = WordLevelTrainer(show_progress=True, special_tokens=["[UNK]", "[PAD]"],
                                   min_frequency=2)
        tokenizer.pre_tokenizer = Whitespace()
        logger.info(f"Training tokenizer: {tokenizer_name}")
        tokenizer.train_from_iterator(sentences_ds, trainer=trainer)
        logger.info(f"Saving tokenizer to path: {tokenizer_path}")
        tokenizer.save("tokenizer/gpt_tokenizer_1")

    logger.info(f"Loading tokenizers from path: {tokenizer_path}")
    tokenizer = Tokenizer.from_file("tokenizer/gpt_tokenizer_1")

    max_len = 0
    ds = []
    for sentence in sentences_ds:
      sentence = sentence.strip()
      enc = tokenizer.encode(sentence).ids

      ds.append(sentence.strip())

    train_ds_, val_ds_ = ds[:int(train_size*len(sentences_ds))], sentences_ds[int(train_size*len(sentences_ds)):]

    return train_ds_, val_ds_, tokenizer


def get_data_opus():

    all_ds = load_dataset("opus_books", f"en-fr")
    all_ds = all_ds['train']


    sentences_ds = []
    for el in all_ds:
      if len(el['translation']["en"].split(" ")) > 5:
        sentences_ds.append(el['translation']["en"])

    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~1234567890<>>»—«'''

    for i in range(len(sentences_ds)):

        phrase = sentences_ds[i]

        for cr in punc:
          if cr in phrase:
              phrase = phrase.replace(cr,"")
              phrase = re.sub('\s+',' ', phrase)
        phrase = re.sub("—"," ", phrase)
        sentences_ds[i] = phrase.strip()


    # training Tokenizer from available sentences
    if not os.path.isfile("tokenizer/gpt_tokenizer_1"):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        trainer = WordLevelTrainer(show_progress=True, special_tokens=["[UNK]", "[PAD]"],
                                   min_frequency=2)
        tokenizer.pre_tokenizer = Whitespace()
        logger.info(f"Training tokenizer: {tokenizer_name}")
        tokenizer.train_from_iterator(sentences_ds, trainer=trainer)
        logger.info(f"Saving tokenizer to path: {tokenizer_path}")
        tokenizer.save("tokenizer/gpt_tokenizer_1")

    logger.info(f"Loading tokenizers from path: {tokenizer_path}")
    tokenizer = Tokenizer.from_file("tokenizer/gpt_tokenizer_1")

    max_len = 0
    ds = []
    for sentence in sentences_ds:
      sentence = sentence.strip()
      enc = tokenizer.encode(sentence).ids

      ds.append(sentence.strip())

    train_ds_, val_ds_ = ds[:int(train_size*len(sentences_ds))], sentences_ds[int(train_size*len(sentences_ds)):]

    return train_ds_, val_ds_, tokenizer

def get_data_books():


    if not os.path.isfile('books/clean_gpt_training.txt'):

      #  Preparing the dataset (text file with books)
      with open('books/gpt_training.txt', 'r', encoding="utf8") as file:
          # Read all lines to list
          lines = file.readlines()
          # Join the lines
          content = ' '.join(lines)
          # Strip the new line
          clean_content = content.replace('/\s\s+/g', ' ')
          clean_content = clean_content.replace('\n', ' ')

      punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~1234567890'''

      for cr in content:
          if cr in punc:
              clean_content = clean_content.replace(cr,"")

      f = open('books/clean_gpt_training.txt', "w")
      f.write(clean_content)
      f.close()


    f = open('books/clean_gpt_training.txt', "r")
    clean_content = f.read()
    f.close()

    sentences_ds = []
    L = clean_content.split(" ")
    k = randint(5,context_size)
    l =[]
    for word in tqdm(L):
        if len(word) > 0 and len(l) < k:
            l.append(word)
        elif len(word) > 0 and len(l) == k:
            sentences_ds.append(l)
            l = []
            k = randint(5, context_size)

    # training Tokenizer from available sentences
    if not os.path.isfile("tokenizer/gpt_tokenizer_1"):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        trainer = WordLevelTrainer(show_progress=True, special_tokens=["[UNK]", "[PAD]"],
                                   min_frequency=2)
        tokenizer.pre_tokenizer = Whitespace()
        logger.info(f"Training tokenizer: {tokenizer_name}")
        tokenizer.train_from_iterator(sentences_ds, trainer=trainer)
        logger.info(f"Saving tokenizer to path: {tokenizer_path}")
        tokenizer.save("tokenizer/gpt_tokenizer_1")

    logger.info(f"Loading tokenizers from path: {tokenizer_path}")
    tokenizer = Tokenizer.from_file("tokenizer/gpt_tokenizer_1")


    train_ds_, val_ds_ = sentences_ds[:int(train_size*len(sentences_ds))], sentences_ds[int(train_size*len(sentences_ds)):]

    return train_ds_, val_ds_, tokenizer


def train():


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = GPTDataset(tokenizer, train_ds_, context_size)
    val_ds = GPTDataset(tokenizer, val_ds_, context_size)

    train_dataloader = DataLoader(batch_size=train_batch_size, dataset=train_ds, shuffle=True)
    val_dataloader = DataLoader(batch_size=val_batch_size, dataset=val_ds)

    gpt_model = Instantiate_GPT(width=width, context_size=context_size, vocab_size=tokenizer.get_vocab_size(), h=h,
                                hidden=hidden, Nx=Nx, dropout=0.1).to(device)

    print(gpt_model)


    optimizer_gpt = torch.optim.Adam(gpt_model.parameters(), lr=learning_rate)
    objective_function = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]")).to(device)

    start, step = 0, 0

    if load_pretrained:
        state = torch.load(model_name)
        start = state['epoch'] + 1
        optimizer_gpt.load_state_dict(state['optimizer_state_dict'])
        step = state['global_step']

    for epoch in range(start, num_epochs):
        gpt_model.train()
        iterator = tqdm(train_dataloader, desc=f"epoch {epoch:02d}")
        for batch in iterator:
            gpt_model.train()

            inp, mask, target = batch["input"].to(device), batch["mask"].to(device), batch["target"].to(device)
            inp = inp.to(dtype=torch.int64)
            mask = mask.to(dtype=torch.int64)
            target = target.to(dtype=torch.int64)
            output_gpt = gpt_model.forward(inp.cuda(), mask.cuda())  # (bsize, context_size, vocab_size)
            logger.debug(f"Output of gpt model is of size {output_gpt.shape}")

            loss = objective_function(output_gpt.view(-1, tokenizer.get_vocab_size()), target.view(-1))
            iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # backprop
            loss.backward()

            # Optimizing steps
            optimizer_gpt.step()
            optimizer_gpt.zero_grad()
            step += 1

            validate(gpt_model, val_dataloader, tokenizer, write_msg, iterator, device)

        model_filename = model_name

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": gpt_model.state_dict(),
                "optimizer_state_dict": optimizer_gpt.state_dict(),
                "global_step": step
            }, model_filename
        )