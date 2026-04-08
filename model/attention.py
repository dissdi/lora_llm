import torch
import torch.nn as nn
import math
from .lora import LoRALinear

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        
        # Q, K, V: adjust LoRA
        self.q_proj = LoRALinear(config.d_model, config.d_model,
                                 r=config.lora_r, alpha=config.lora_alpha,
                                 dropout=config.lora_dropout)
        self.k_proj = LoRALinear(config.d_model, config.d_model,
                                 r=config.lora_r, alpha=config.lora_alpha,
                                 dropout=config.lora_dropout)
        self.v_proj = LoRALinear(config.d_model, config.d_model,
                                 r=config.lora_r, alpha=config.lora_alpha,
                                 dropout=config.lora_dropout)
        # out_proj: normal linear (without LoRA)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, attention_mask=None):
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        # causal attention
        scale = math.sqrt(self.d_head)
        attn = (q @ k.transpose(-2, -1)) / scale # (B, H, d_head, T)
        
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).unfreeze(0).unsqueeze(0)
        attn = attn.masked_fill(causal_mask == 0, float('-inf'))
        
        if attention_mask is not None:
            attn = attn + attention_mask[:, None, None, :]
            
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)
        
        
        
        
        
        
        
        
        
        
        
        
