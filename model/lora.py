import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r, alpha, dropout=0.0, bias=False):
        super().__init__()
        self.r = r
        self.scaling = alpha / r
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_param = nn.Parameter(torch.zeros(out_features)) if bias else None
        
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.lora_dropout = nn.Dropout(dropout)
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
    def forward(self, x):
        base_out = nn.functional.linear(x, self.weight, self.bias_param)
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B
        return base_out + lora_out * self.scaling