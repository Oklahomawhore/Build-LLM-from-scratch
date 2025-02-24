import torch
import torch.nn as nn

# Note:
# After attention, the output is a NxN matrix, where N is the sequence length.
# After the computation of attention weights and context vector computation, the result is a dxd matrix, where d is the latent embedding dimension.
# Summarizing the whole process, for simplified self-attention it's:
# N (length of input ids) -> Nxd (embeddings) -> NxN (attention scores) -> Nxd context vectors
# The complexity of the whole process is O(Nd^2), because first the attention scores computing is two Nxd matrix multiplication which is O(Nd), then we need all context vectors for every input,
# thus the complexity becomes O(N^2d).

# scaled dot-product attention:
# for better distinguishing between dimensions, we use dxd_k for all weight parameters.
# We have W_k, W_q, W_v, each have dimension dxd_k.
# the corresponding query, key and value matrices are Nxd_k, 
# the attention weigths matrix is NxN , and the context vectors matrix is Nxd_k.


# *Attention_v2*:
# class Attention_v2(nn.Module):
#     def __init__(self, d_in, d_out, qkv_bias=False):
#         self.M_q = nn.Linear(d_in, d_out, bias=qkv_bias)
#         self.M_k = nn.Linear(d_in, d_out, bias=qkv_bias)
#         self.M_v = nn.Linear(d_in, d_out, bias=qkv_bias)

#     def forward_pass(self, x):
#         q = self.M_q(x)
#         k = self.M_k(x)
#         v = self.M_v(x)

#         attention_score = q @ k.T
#         attention_weights = attention_score / k.shape[-1] ** 0.5
#         attention_weights = attention_weights.softmax(dim=-1)
#         context_vectors = attention_weights @ v
#         return context_vectors
import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")

class MultiheadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, drop_out, qkv_bias=False):

        super().__init__()
        self.d_out = d_out
        self.M_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.M_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.M_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.context_length = context_length
        self.drop_out_layer = nn.Dropout(drop_out)
        self.num_heads = num_heads
        self.out_proj = nn.Linear(d_out, d_out) # optional linear projection layer

        assert(d_out % num_heads == 0) # ensure d_out is divisible by num_heads
        self.head_dim = d_out // num_heads
        
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        # v2 handling batched inputs
        b, num_token, d_in = x.shape

        # split outputs to head representations
        q = self.M_q(x)
        k = self.M_k(x)
        v = self.M_v(x)

        q = q.view(b, num_token, self.num_heads, self.d_out // self.num_heads).transpose(1, 2)  # b, num_heads, num_token, head_dim
        k = k.view(b, num_token, self.num_heads, self.d_out // self.num_heads).transpose(1, 2)  # b, num_heads, num_token, head_dim
        v = v.view(b, num_token, self.num_heads, self.d_out // self.num_heads).transpose(1, 2)  # b, num_heads, num_token, head_dim

        attention_score = q @ k.transpose(2,3) # result is bxnum_headsxnum_tokenxnum_token
        # print("attention score: ")
        # print(attention_score)
        
        # context_length = attention_score.shape[0]
        # # Apply attention mask before softmax and normalization.
        # mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        # print("-inf mask: ")
        # print(mask)
        
        
        # attention_score = attention_score.masked_fill(mask.bool(), -torch.inf)

        attention_score.masked_fill_(self.mask.bool()[:num_token, :num_token], - torch.inf)
        # print("attention score after mask: ")
        # print(attention_score)

        attention_weights = torch.softmax(attention_score / k.shape[-1] ** 0.5, dim=-1)
        # print("attention weights: ")
        # print(attention_weights)
        attention_weights = self.drop_out_layer(attention_weights)

        context_vectors = attention_weights @ v
        context_vectors = context_vectors.contiguous().view(b, num_token, self.d_out)

        # out projection

        context_vectors = self.out_proj(context_vectors)
        return context_vectors


if __name__ == '__main__':
    # Test the attention mechanism.
    torch.manual_seed(123)

    batch_size = 4
    context_length = 1024
    embedding_dim = 768
    d_model = 768
    num_heads = 12
    # attention = Casual_Attention(embedding_dim, d_model, 128, 0.1)
    attention = MultiheadAttention(embedding_dim, d_model, num_heads, context_length, 0.1)
    x = torch.rand(batch_size, context_length,  embedding_dim)
    
    x_out = attention(x)
    print(x)
    print(x_out)