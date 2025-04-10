import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import math
from TransformerGQA import GQATransformerNextWord
from TransformerCHA import CHATransformerNextWord
import pickle
import time

N_EPOCHS = 10 # 10 recommended for small dataset, 3 for mid dataset
DATASET_SIZE = "small"
model_params = {
    "Small": { # Mid-sized model
        "EMB_DIM": 64,
        "ENC_LAYERS": 3,
        "DEC_LAYERS": 3,
        "ENC_DROPOUT": 0.1,
        "DEC_DROPOUT": 0.1,
        "FF_DIM": 16,
    },
    "Mid": { # Mid-sized model
        "EMB_DIM": 64,
        "ENC_LAYERS": 3,
        "DEC_LAYERS": 3,
        "ENC_DROPOUT": 0.1,
        "DEC_DROPOUT": 0.1,
        "FF_DIM": 128,
    }
    }

QKV_HEAD_COMBINATIONS = [
    (2, 8),
    (4, 8),
    (2, 16),
    (4, 16),
    (8, 16),
    (2, 32),
    (4, 32),
    (8, 32),
    (16, 32),
    (2, 64),
    (4, 64),
    (8, 64),
    (16, 64),
    (32, 64)
]

print("test")

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

# Change to desired directory (ours is specific to the server we used)
with open("/w/331/noeartru/ToyModel/Data/" + DATASET_SIZE + "EnDe.pkl", "rb") as f:
    loaded_data = pickle.load(f)

INPUT_DIM = loaded_data["de_vocab_size"]
OUTPUT_DIM = loaded_data["en_vocab_size"]
BATCH_SIZE = loaded_data["batch_size"]
PAD_IDX_en = loaded_data["en_pad_idx"]
PAD_IDX_de = loaded_data["de_pad_idx"]
BOS_IDX_en = loaded_data["en_bos_idx"]
BOS_IDX_de = loaded_data["de_bos_idx"]
EOS_IDX_en = loaded_data["en_eos_idx"]
EOS_IDX_de = loaded_data["de_eos_idx"]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", DEVICE)
DTYPE = torch.float32

train_iter = loaded_data["train_iter"]
valid_iter = loaded_data["valid_iter"]
test_iter = loaded_data["test_iter"]

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX_en)

def train(model: nn.Module,
          iterator: torch.utils.data.DataLoader,
          optimizer: optim.Optimizer,
          criterion: nn.Module):

    model.train()

    epoch_loss = 0
    time0 = time.time()
    print("Number of batches:", len(iterator))
    for i, (src, trg) in enumerate(iterator):
        src, trg = src.to(DEVICE), trg.to(DEVICE)

        optimizer.zero_grad()
        output = model(src, trg)

        output = output[:, 1:]
        trg = trg[:, 1:]
        output = output.transpose(1, 2)

        loss = criterion(output, trg)

        loss.backward()



        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        optimizer.step()

        epoch_loss += loss.item()

    print("Total time taken for training epoch: ", time.time() - time0)
    return epoch_loss / len(iterator)


def evaluate(model: nn.Module,
             iterator: torch.utils.data.DataLoader,
             criterion: nn.Module):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for i, (src, trg) in enumerate(iterator):
            src, trg = src.to(DEVICE), trg.to(DEVICE)

            output = model(src, trg)

            output = output[:, 1:]
            trg = trg[:, 1:]
            output = output.transpose(1, 2)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def count_parameters(model: nn.Module):
  return sum(p.numel() for p in model.transformer.encoder.parameters() if p.requires_grad)

def training_regiment(model, optimizer, save_name, hyperparameters):
    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss = train(model, train_iter, optimizer, criterion)
        valid_loss = evaluate(model, valid_iter, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

        # if(epoch%500 == 0):
        #     print(torch.cuda.memory_summary(device=DEVICE))

    test_loss = evaluate(model, test_iter, criterion)



    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

    # Save the model
    # Change to desired directory (ours is specific to the server we used)
    torch.save(model.state_dict(), "/w/331/noeartru/ToyModel/Data/" + save_name + ".pt")
    

    with open("/w/331/noeartru/ToyModel/Data/" + save_name + "_hyperparameters.txt", 'w') as f:
        for key, value in hyperparameters.items():
            f.write(f"{key}: {value}\n")

    print(f"Model and hyperparameters saved successfully for {save_name}.")

# Main training loop
for model_size, params in model_params.items():
    for kvh, qh in QKV_HEAD_COMBINATIONS:
        for model_type in ["GQA", "CHA"]:
            save_name = f"{model_size}_{model_type}_model_{qh}qh_{kvh}kvh"

            hyperparameters = {
                "INPUT_DIM": INPUT_DIM,
                "OUTPUT_DIM": OUTPUT_DIM,
                "EMB_DIM": params["EMB_DIM"],
                "QUERY_HEADS": qh,
                "KV_HEADS": kvh,
                "ENC_LAYERS": params["ENC_LAYERS"],
                "DEC_LAYERS": params["DEC_LAYERS"],
                "ENC_DROPOUT": params["ENC_DROPOUT"],
                "DEC_DROPOUT": params["DEC_DROPOUT"],
                "FF_DIM": params["FF_DIM"],
                "BATCH_SIZE": BATCH_SIZE,
                "N_EPOCHS": N_EPOCHS,
                "DEVICE": str(DEVICE),
                "DTYPE": str(DTYPE)
            }

            if model_type == "GQA":
                model = GQATransformerNextWord(num_tokens_enc=INPUT_DIM, num_tokens_dec=OUTPUT_DIM, embed_dim=hyperparameters["EMB_DIM"], query_heads=hyperparameters["QUERY_HEADS"], kv_heads=hyperparameters["KV_HEADS"], num_encoder_layers=hyperparameters["ENC_LAYERS"], num_decoder_layers=hyperparameters["DEC_LAYERS"], feedforward_dim=hyperparameters["FF_DIM"], device=DEVICE, dtype=DTYPE).to(DEVICE)
            else:
                model = CHATransformerNextWord(num_tokens_enc=INPUT_DIM, num_tokens_dec=OUTPUT_DIM, embed_dim=hyperparameters["EMB_DIM"], query_heads=hyperparameters["QUERY_HEADS"], kv_heads=hyperparameters["KV_HEADS"], num_encoder_layers=hyperparameters["ENC_LAYERS"], num_decoder_layers=hyperparameters["DEC_LAYERS"], feedforward_dim=hyperparameters["FF_DIM"], device=DEVICE, dtype=DTYPE).to(DEVICE)
            optimizer = optim.Adam(model.parameters())
            print(f'The model {save_name} has {count_parameters(model):,} trainable parameters')
            training_regiment(model, optimizer, save_name, hyperparameters)
            