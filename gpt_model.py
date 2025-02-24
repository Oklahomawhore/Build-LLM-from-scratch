import torch
import torch.nn as nn
import tiktoken
import matplotlib.pyplot as plt
from attention import MultiheadAttention
from utility import text_to_token_ids, token_ids_to_text

torch.set_printoptions(sci_mode=False)

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads" : 12,
    "n_layers" : 12,
    "drop_rate" : 0.1,
    "qkv_bias" : False,
}

GPT_CONFIG_M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1024,
    "n_heads" : 16,
    "n_layers" : 24,
    "drop_rate" : 0.1,
    "qkv_bias" : False,
}

GPT_CONFIG_L = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1280,
    "n_heads" : 20,
    "n_layers" : 36,
    "drop_rate" : 0.1,
    "qkv_bias" : False,
}

GPT_CONFIG_XL = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1600,
    "n_heads" : 25,
    "n_layers" : 48,
    "drop_rate" : 0.1,
    "qkv_bias" : False,
}

class GPTModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_emb = self.tok_emb(in_idx)
        pos_emb = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_emb + pos_emb
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        # inlude learnable shift and scale parameters to let model decide whether to return to the original distribution
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.eps = 1e-5

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.scale + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 0.5 *(1.0 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], cfg["emb_dim"] * 4),
            GELU(),
            nn.Linear(cfg["emb_dim"] * 4, cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiheadAttention(cfg["emb_dim"], cfg["emb_dim"], cfg["n_heads"], cfg["context_length"], cfg["drop_rate"], cfg["qkv_bias"])
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.ff = FeedForward(cfg)
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:] # only get the last context tokens
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :] # only get the last token
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=-1)
    return idx # return token ids


if __name__ == '__main__':

    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    print(batch)

    # ln = LayerNorm(4)
    # x = torch.randn(2, 4)
    # x = ln(x)
    # mean = x.mean(dim=-1, keepdim=True)
    # var = x.var(dim=-1, keepdim=True, unbiased=False)
    # print(mean)
    # print(var)

    # x = torch.rand(2, 4, 768)
    # block = TransformerBlock(GPT_CONFIG_124M)
    # output = block(x)

    # print(f"input shape: {x.shape}")
    # print(f"output shape: {output.shape}")

    # cfgs = [GPT_CONFIG_124M, GPT_CONFIG_M, GPT_CONFIG_L, GPT_CONFIG_XL]
    cfgs = [GPT_CONFIG_124M]


    device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"

    for cfg in cfgs:
        model = GPTModel(cfg).to(device)
        batch = batch.to(device)
        out = model(batch)
        print(cfg)
        print(f"input batch:\n {batch}")
        print(f"\noutput shape: {out.shape}")
        print(out)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {total_params:,}")
        print("Token embedding layer shape:", model.tok_emb.weight.shape)
        print("Output layer shape:", model.out_head.weight.shape)

        bytes_in_param = 4
        total_bytes = total_params * bytes_in_param
        print(f"Total memory size: {total_bytes / (1024 * 1024):.2f} MB")



    prompt = "Hello, I am"
    encoded_tensor = text_to_token_ids(prompt, tokenizer)
    model.eval()

    out = generate_text_simple(model, encoded_tensor, 6, GPT_CONFIG_124M["context_length"])
    print(f"output: {out}")
    print("Output length:", len(out[0]))


    decoded_text = token_ids_to_text(out, tokenizer)
    print(decoded_text)









