import time
time0 = time.time()

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import pickle

BATCH_SIZE = 16

time1 = time.time()
print("Time taken for imports:", time1 - time0)
time0 = time1

# Dataset preparation (we use a subset of WMT14 in this example)
# Change cache-dir to desired directory (ours is specific to the server we used)
SIZE = "small"
dataset = load_dataset("wmt14", "de-en", split={"train": "train[:6400]", "val": "validation[:640]", "test": "test[:640]"}, cache_dir="/w/331/noeartru/.cache")
time1 = time.time()
print("Time taken to load dataset:", time1 - time0)
time0 = time1

# Tokenizer initialization
de_tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased", use_fast=True)
en_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
special_tokens = ['<unk>', '<pad>', '<bos>', '<eos>']
de_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
en_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
time1 = time.time()
print("Time taken to create tokenizers:", time1 - time0)
time0 = time1


def tokenize_and_build_vocab(dataset, tokenizer, language_key):
    tokenized_texts = tokenizer.batch_encode_plus(
        [example['translation'][language_key] for example in dataset],
        truncation=True,
        max_length=512,  
        padding=True,    
        add_special_tokens=True
    )['input_ids']
    vocab = tokenizer.get_vocab()
    return vocab, tokenized_texts

de_vocab, de_tokenized_texts = tokenize_and_build_vocab(dataset["train"], de_tokenizer, "de")
en_vocab, en_tokenized_texts = tokenize_and_build_vocab(dataset["train"], en_tokenizer, "en")
time1 = time.time()
print("Time taken to initialize tokenizers on dataset:", time1 - time0)
time0 = time1

def data_process(dataset, de_tokenizer, en_tokenizer):
    """
    Processes the dataset by tokenizing and converting to tensors.
    """
    data = []
    for example in dataset:
        de_tensor_ = torch.tensor(
            de_tokenizer.encode(
                example['translation']["de"],
                truncation=False,
                max_length=512,
                add_special_tokens=False
            ),
            dtype=torch.long
        )
        en_tensor_ = torch.tensor(
            en_tokenizer.encode(
                example['translation']["en"],
                truncation=False,
                max_length=512,
                add_special_tokens=False
            ),
            dtype=torch.long
        )
        data.append((de_tensor_, en_tensor_))
    return data

# Processing train, validation, and test datasets
train_data = data_process(dataset["train"], de_tokenizer, en_tokenizer)
val_data = data_process(dataset["val"], de_tokenizer, en_tokenizer)
test_data = data_process(dataset["test"], de_tokenizer, en_tokenizer)

time1 = time.time()
print("Time taken to create train,test,val data:", time1 - time0)
time0 = time1

# Creating Dataloaders
PAD_IDX_de = de_vocab['<pad>']
BOS_IDX_de = de_vocab['<bos>']
EOS_IDX_de = de_vocab['<eos>']

PAD_IDX_en = en_vocab['<pad>']
BOS_IDX_en = en_vocab['<bos>']
EOS_IDX_en = en_vocab['<eos>']

def generate_batch(data_batch):
  de_batch, en_batch = [], []
  for (de_item, en_item) in data_batch:
    de_batch.append(torch.cat([torch.tensor([BOS_IDX_de]), de_item, torch.tensor([EOS_IDX_de])], dim=0))
    en_batch.append(torch.cat([torch.tensor([BOS_IDX_en]), en_item, torch.tensor([EOS_IDX_en])], dim=0))
  
  de_batch = pad_sequence(de_batch, padding_value=PAD_IDX_de).to(dtype=torch.long)
  en_batch = pad_sequence(en_batch, padding_value=PAD_IDX_en).to(dtype=torch.long)
  
  de_batch = de_batch.transpose(0, 1)
  en_batch = en_batch.transpose(0, 1)
  return de_batch, en_batch

train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
            shuffle=True, collate_fn=generate_batch, drop_last=True)
valid_iter = DataLoader(val_data, batch_size=BATCH_SIZE,
            shuffle=True, collate_fn=generate_batch, drop_last=True)
test_iter = DataLoader(test_data, batch_size=BATCH_SIZE,
             shuffle=True, collate_fn=generate_batch, drop_last=True)

time1 = time.time()
print("Time taken to create dataloaders:", time1 - time0)
time0 = time1


# Saving data for training pipeline and stats (iterators, tokenizers and data parameters)
save_data = {
    "train_iter": train_iter,
    "valid_iter": valid_iter,
    "test_iter": test_iter,
    "de_vocab_size": len(de_vocab),
    "en_vocab_size": len(en_vocab),
    "batch_size": BATCH_SIZE,
    "en_pad_idx": PAD_IDX_en,
    "de_pad_idx": PAD_IDX_de,
    "en_bos_idx": BOS_IDX_en,
    "de_bos_idx": BOS_IDX_de,
    "en_eos_idx": EOS_IDX_en,
    "de_eos_idx": EOS_IDX_de,
}

# Change to desired directory (ours is specific to the server we used)
de_tokenizer.save_pretrained("/w/331/noeartru/ToyModel/Data/de_tokenizer_"+SIZE)
en_tokenizer.save_pretrained("/w/331/noeartru/ToyModel/Data/en_tokenizer_"+SIZE)

with open("/w/331/noeartru/ToyModel/Data/" + SIZE + "EnDe.pkl", "wb") as f:
    pickle.dump(save_data, f)

print("Data pipeline components saved successfully.")