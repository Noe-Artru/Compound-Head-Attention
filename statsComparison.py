import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
import pickle
import math
from TransformerGQA import GQATransformerNextWord
from TransformerCHA import CHATransformerNextWord
import numpy as np



DATASET_SIZE = "mid"
model_params = {
    # "Small": { # Smaller model with lower FF block parameters
    #     "EMB_DIM": 64,
    #     "ENC_LAYERS": 3,
    #     "DEC_LAYERS": 3,
    #     "ENC_DROPOUT": 0.1,
    #     "DEC_DROPOUT": 0.1,
    #     "FF_DIM": 16,
    # },
    "Mid": { # Mid-sized model with larger FF block parameters
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

test_iter = loaded_data["test_iter"]

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX_en)

# Change to desired directory (ours is specific to the server we used)
loaded_de_tokenizer = AutoTokenizer.from_pretrained("/w/331/noeartru/ToyModel/Data/de_tokenizer_" + DATASET_SIZE)
loaded_en_tokenizer = AutoTokenizer.from_pretrained("/w/331/noeartru/ToyModel/Data/en_tokenizer_" + DATASET_SIZE)

def count_parameters(model: nn.Module, model_type):
    if model_type == "GQA":
        module = model.transformer.encoder.layers[0].GQA_attention
    else:
        module = model.transformer.encoder.layers[0].CHA_attention
    # For counting only QKV head parameters, take into account the following:
    # fc_params = sum(p.numel() for p in module.FC.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters() if p.requires_grad) #- fc_params

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

def show_examples(model, iterator, de_tokenizer, en_tokenizer, num_examples=1): # num_examples < BATCH_SIZE
    model.eval()
    examples = []

    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            if i == 1:
                break

            src, trg = src.to(DEVICE), trg.to(DEVICE)
            output = model(src, trg)
            output = output[:, 1:]
            output = output.argmax(dim=2) 

            for j in range(num_examples):
                src_sentence = de_tokenizer.decode(src[j].tolist(), skip_special_tokens=True)
                trg_sentence = en_tokenizer.decode(trg[j].tolist(), skip_special_tokens=True)
                output_tokens = [token for token in output[j].tolist() if token not in {BOS_IDX_en, EOS_IDX_en, PAD_IDX_en}]
                output_sentence = en_tokenizer.decode(output_tokens, skip_special_tokens=True)
                examples.append((src_sentence, trg_sentence, output_sentence))
    
    examples_string = ""

    for idx, (src_sentence, trg_sentence, output_sentence) in enumerate(examples):
        examples_string += f"Example {idx + 1}:\n"
        examples_string += f"  Source: {src_sentence}\n"
        examples_string += f"  Target: {trg_sentence}\n"
        examples_string += f"  Output: {output_sentence}\n"

# Change to desired directory (ours is specific to the server we used)
    with open(f"/w/331/noeartru/ToyModel/examples/{save_name}_examples.txt", "w", encoding="utf-8") as file:
        file.write(examples_string)

# Main loop for testing/analysing models
stats = {"CHA": {}, "GQA": {}}
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
                "DEVICE": str(DEVICE),
                "DTYPE": str(DTYPE)
            }

            if model_type == "GQA":
                model = GQATransformerNextWord(num_tokens_enc=INPUT_DIM, num_tokens_dec=OUTPUT_DIM, embed_dim=hyperparameters["EMB_DIM"], query_heads=hyperparameters["QUERY_HEADS"], kv_heads=hyperparameters["KV_HEADS"], num_encoder_layers=hyperparameters["ENC_LAYERS"], num_decoder_layers=hyperparameters["DEC_LAYERS"], feedforward_dim=hyperparameters["FF_DIM"], device=DEVICE, dtype=DTYPE).to(DEVICE)
            else:
                model = CHATransformerNextWord(num_tokens_enc=INPUT_DIM, num_tokens_dec=OUTPUT_DIM, embed_dim=hyperparameters["EMB_DIM"], query_heads=hyperparameters["QUERY_HEADS"], kv_heads=hyperparameters["KV_HEADS"], num_encoder_layers=hyperparameters["ENC_LAYERS"], num_decoder_layers=hyperparameters["DEC_LAYERS"], feedforward_dim=hyperparameters["FF_DIM"], device=DEVICE, dtype=DTYPE).to(DEVICE)

            # Change to desired directory (ours is specific to the server we used)
            model.load_state_dict(torch.load(f"/w/331/noeartru/ToyModel/Data/{save_name}.pt", map_location=DEVICE))

            num_parameters = count_parameters(model, model_type)
            print(f'The model {save_name} has {num_parameters:,} trainable parameters')
            test_loss = evaluate(model, test_iter, criterion)
            print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

            num_examples = 8
            show_examples(model, test_iter, loaded_de_tokenizer, loaded_en_tokenizer, num_examples)

            stats[model_type][(kvh, qh)] = (num_parameters, test_loss)


# Plotting graphs

# 1st plot for test loss
kvh_qh_combinations = list(stats["GQA"].keys())
x_labels = [f"({kvh}, {qh})" for kvh, qh in kvh_qh_combinations]
x_indices = np.arange(len(kvh_qh_combinations))

gqa_losses = [stats["GQA"][(kvh, qh)][1] for kvh, qh in kvh_qh_combinations]
cha_losses = [stats["CHA"][(kvh, qh)][1] for kvh, qh in kvh_qh_combinations]

bar_width = 0.4
opacity = 0.7

plt.figure(figsize=(12, 6))
plt.bar(x_indices - bar_width / 2, gqa_losses, bar_width, alpha=opacity, label="GQA")
plt.bar(x_indices + bar_width / 2, cha_losses, bar_width, alpha=opacity, label="CHA")

plt.xlabel('(KVH, QH) Combinations')
plt.ylabel('Test Loss')
plt.title('Test Loss for Each Model Type and (KVH, QH) Combination')
plt.xticks(x_indices, x_labels, rotation=45, ha='right')
plt.legend()
plt.tight_layout()

# Change to desired directory (ours is specific to the server we used)
bar_plot_path = "/w/331/noeartru/ToyModel/plots/test_loss_bar_plot.png"
plt.savefig(bar_plot_path)
print(f"Bar plot saved to {bar_plot_path}")

# 2nd plot for model sizes

gqa_sizes = [stats["GQA"][(kvh, qh)][0] for kvh, qh in kvh_qh_combinations]
cha_sizes = [stats["CHA"][(kvh, qh)][0] for kvh, qh in kvh_qh_combinations]
plt.figure(figsize=(12, 6))

plt.bar(x_indices - bar_width / 2, gqa_sizes, bar_width, alpha=opacity, label="GQA")
plt.bar(x_indices + bar_width / 2, cha_sizes, bar_width, alpha=opacity, label="CHA")

plt.xlabel('(KVH, QH) Combinations')
plt.ylabel('Model Size (Attention Module + Linear Layer)')
plt.title('Number of Parameters for Each Model Type and (KVH, QH) Combination')
plt.xticks(x_indices, x_labels, rotation=45, ha='right')
plt.legend()
plt.tight_layout()

# Change to desired directory (ours is specific to the server we used)
bar_plot_path = "/w/331/noeartru/ToyModel/plots/sizes_bar_plot.png"
plt.savefig(bar_plot_path)
print(f"Bar plot saved to {bar_plot_path}")