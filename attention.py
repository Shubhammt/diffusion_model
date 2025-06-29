import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True):
        super().__init__()
        self.q_proj   = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj   = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.v_proj   = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.n_heads = n_heads
        self.device = 'cpu'
        self.multihead_attn = nn.MultiheadAttention(d_embed, n_heads, bias = in_proj_bias)

    def forward(self, x, causal_mask=False):
        if self.device!=x.device:
            self.device = x.device
            self.multihead_attn.to(x.device)
        sequence_length = x.shape[1]
        x = x.transpose(0, 1)
        if causal_mask:
            mask = torch.ones((sequence_length, sequence_length), dtype=torch.bool, device = x.device).triu(1)
            output = self.multihead_attn(x, x, x, attn_mask = mask)
        else:
            output = self.multihead_attn(x, x, x)
        return output[0].transpose(0, 1)

class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True):
        super().__init__()
        self.q_proj   = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    
    def forward(self, x, y):
        input_shape = x.shape
        batch_size = input_shape[0]
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2) 
        
        weight = q @ k.transpose(-1, -2)
        
        weight /= math.sqrt(self.d_head)
        
        weight = F.softmax(weight, dim=-1)
        
        output = weight @ v
        
        output = output.transpose(1, 2).contiguous()
        
        output = output.view(input_shape)
        
        output = self.out_proj(output)

        return output
    
if __name__ == "__main__":
    generator = torch.Generator(device='cuda')
    x = torch.randn((4, 77, 768), generator=generator, device='cuda')
    y = torch.randn((4, 31, 45), generator=generator, device='cuda')
    model = SelfAttention(8, 768, in_proj_bias=True).to('cuda')
    output = model(x, causal_mask = True)
    model = CrossAttention(8, 768, 45, in_proj_bias=True).to('cuda')
    output = model(x, y)
    print(output.shape)