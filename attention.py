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

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head
        
        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) 
        
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec


if __name__ == '__main__':
    # Test the attention mechanism.
    torch.manual_seed(123)

    batch_size = 4
    context_length = 1024
    embedding_dim = 768
    d_model = 768
    num_heads = 12
    # attention = Casual_Attention(embedding_dim, d_model, 128, 0.1)
    attention = MultiHeadAttention(embedding_dim, d_model, context_length, 0.1, num_heads)
    x = torch.rand(batch_size, context_length,  embedding_dim)
    
    x_out = attention(x)
    print(x)
    print(x_out)