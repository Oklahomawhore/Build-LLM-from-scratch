from gpt_model import GPTModel, GPT_CONFIG_124M
import torch
from torch.utils.data import DataLoader
from gpt_download import download_and_load_gpt2
settings, params = download_and_load_gpt2(
    model_size="124M", models_dir="gpt2"
)
from gpt_model import generate, text_to_token_ids, token_ids_to_text
from train import evaluate_model, get_textdata, create_dataloader_v1, calc_loss_loader
import tiktoken
import requests
import os
import json
from tqdm import tqdm
from datasets import load_dataset



tokenizer = tiktoken.get_encoding("gpt2")
print("Settings :", settings)
print("parameter keys: ", params.keys())

device = 'cuda'


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))



model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


import numpy as np

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

def get_tokens(tokenizer, text, strict=True, device=None):
    if not strict:
        tokens = text_to_token_ids(text, tokenizer)
        tokens = tokens.to(device)
        return tokens[:, :-1], tokens[:, -1]
    last_token = text.split()[-1]
    start_idx = text.rfind(last_token)
    beginning_tokens = text_to_token_ids(text[:start_idx], tokenizer)
    all_tokens = text_to_token_ids(text, tokenizer)
    last_token = all_tokens[:,beginning_tokens.shape[-1]:]
    beginning_tokens = beginning_tokens.to(device)
    last_token = last_token.to(device)
    return beginning_tokens, last_token

def evaluate_lambada(model, device, tokenizer, test_file_path, max_samples=None):
    """Evaluate model on LAMBADA dataset from plain text file."""
    model.eval()
    correct = 0
    total = 0
    
    
    tokenized_data = []
    tokenized_label = []
    # Read examples from plain text file
    with open(test_file_path, "r") as f:
        for line in f.readlines():
            text = line.strip('\n')
            tokens, labels = get_tokens(tokenizer, text, strict=False, device=device)
            tokenized_data.append(tokens)
            tokenized_label.append(labels)
    
    if max_samples is not None:
        tokenized_data = tokenized_data[:max_samples]
        tokenized_label = tokenized_label[:max_samples]
    
    print("Number of examples in LAMBADA test: ",len(tokenized_label))
    with torch.no_grad():
        for context, label in tqdm(zip(tokenized_data, tokenized_label), desc="Evaluating LAMBADA"):
            
            # Get model prediction
            outputs = model(context)
            next_token_logits = outputs[:, -1, :]
            pred_id = torch.argmax(next_token_logits, dim=-1)
            
            # Check accuracy
            if pred_id.item() == label.flatten().item():
                correct += 1
            total += 1
    print(total)
    accuracy = (correct / total) * 100
    # Calculate perplexity: exp(average negative log likelihood)
    
    
    return accuracy

if __name__ == '__main__':
    # Copy the base configuration and update with specific model settings
    model_name = "gpt2-small (124M)"  # Example model name
    NEW_CONFIG = GPT_CONFIG_124M.copy()
    NEW_CONFIG.update(model_configs[model_name])
    NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})

    gpt = GPTModel(NEW_CONFIG)
    gpt.eval();
    load_weights_into_gpt(gpt, params)
    gpt.to(device);

    text_data = get_textdata()

    train_ratio = 0.9
    split_idx = int(len(text_data) * train_ratio)
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    train_loader = create_dataloader_v1(train_data, batch_size=2, max_length=GPT_CONFIG_124M["context_length"], drop_last=True, shuffle=True, num_workers=0)
    val_loader = create_dataloader_v1(val_data, batch_size=2, max_length=GPT_CONFIG_124M["context_length"], drop_last=True, shuffle=True, num_workers=0)

    train_loss, val_loss = evaluate_model(gpt, train_loader, val_loader, device, 5)
    print("Training loss", train_loss)
    print("Validation loss", val_loss)

    torch.manual_seed(123)

    print("\nEvaluating on LAMBADA dataset...")
    # Load LAMBADA test set
    
    lambada_test_path = 'lambada/lambada_test_plain_text.txt'
    lambada_accuracy = evaluate_lambada(gpt, device, tokenizer, lambada_test_path)
    print(f"LAMBADA Results:")
    print(f"  Accuracy: {lambada_accuracy:.2f}%")

    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
        max_new_tokens=25,
        context_size=NEW_CONFIG["context_length"],
        top_k=50,
        temperature=0.7,
        eos_id=text_to_token_ids("<|endoftext|>", tokenizer)[0].to(device)
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))