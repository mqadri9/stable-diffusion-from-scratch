import torch
from torch import nn
import math 
from torch.nn import functional as F
# self-attention block
# given a sequence (seq, d) where each element in the sequence has embedding of size d ,
# we construction Q, K and V matrices of size (d,d) where in the self-attention all are the same and equal to the input 
# multiple each by Wq, Wk and Wv which are parameter matrices to get Q', K' and V'
# and then we split them along the d dimension into h heads and then we compute the attention score as 
# Attention(Q', K', V') = softmax(Q'K'^T/sqrt(d/h))V' with Attention(QW_i^q, KW_i^k, VW_i^v) 
# and then we concat the h heads and multiply by W^o to get the output of the self-attention block

def selfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super(selfAttention, self).__init__()
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

    def forward(self, x: torch.Tensor, causal_mask=False):
        # x: (batch_size, seq_len, dim)

        input_shape = x.shape 
        batch_size, sequence_length, d_embed = input_shape

        intermim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # (batch_size, seq_len, dim) -> (batch_size, seq_len, 3*dim) -> 3 tensors of shape (batch_size, seq_len, dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        
        # (batch_size, seq_len, dim) -> (batch_size, seq_len, n_heads, d_head) -> (batch_size, n_heads, seq_len, d_head)
        q = q.view(intermim_shape).transpose(1, 2)
        k = k.view(intermim_shape).transpose(1, 2)
        v = v.view(intermim_shape).transpose(1, 2)

        # (batch_size, n_heads, seq_len, seq_len)
        weight = q @ k.transpose(-2, -1) / (self.d_head ** 0.5)

        if causal_mask:
            # mask where the upper triangle (above the principal diagonal) is made of ones and the rest is zeros
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight = weight.masked_fill(mask, float('-inf'))
        
        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)
        
        # (batch_size, n_heads, seq_len, seq_len) @ (batch_size, n_heads, seq_len, d_head) -> (batch_size, n_heads, seq_len, d_head)
        output = weight @ v
        
        # (batch_size, n_heads, seq_len, d_head) -> (batch_size, seq_len, n_heads, d_head)
        output = output.transpose(1, 2)

        output = output.reshape(input_shape)

        output = self.out_proj(output)

        return output 


