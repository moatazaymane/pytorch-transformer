train_batch_size = 64
version_exp = 1
val_batch_size = 1
num_epochs = 16
learning_rate = 2.5e-4
context_size = 10
k = context_size
width = 512
inp_lang = "en"
out_lang = "fr"
model_folder = "files"
model_name=f"transformer_{version_exp}"
model_path = f"weights/transformer_{version_exp}"
tokenizer_name = f"gpt_tokenizer_{version_exp}"
tokenizer_path = f"GPT_1/tokenizer/gpt_tokenizer_{version_exp}"
hidden = 1000
Nx = 4
train_size = 0.95
h = 4
load_pretrained = False
