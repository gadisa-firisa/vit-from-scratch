import torch 
import torch.nn as nn 
from ..utils import ViTConfig, GELU

class VitMLP(nn.Module): 
    def __init__(self, config: ViTConfig): 
        super().__init__() 
        self.fc1 = nn.Linear(config.hidden_size, int(config.hidden_size * config.mlp_ratio)) 
        self.act = GELU(config, approx='none') 
        self.fc2 = nn.Linear(int(config.hidden_size * config.mlp_ratio), config.hidden_size) 
        self.drop = nn.Dropout(config.drop_rate) 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.fc1(x) 
        x = self.act(x) 
        x = self.drop(x) 
        x = self.fc2(x) 
        x = self.drop(x) 
        return x

class VitAttention(nn.Module): 
    def __init__(self, config: ViTConfig): 
        super().__init__() 
        self.config = config 
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=config.qkv_bias) 
        self.attn_drop = nn.Dropout(config.attn_drop_rate) 
        self.proj = nn.Linear(config.hidden_size, config.hidden_size) 
        self.proj_drop = nn.Dropout(config.drop_rate) 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        B, N, C = x.shape 
        qkv = self.qkv(x).reshape(B, N, 3, self.config.num_heads, C // self.config.num_heads).permute(2, 0, 3, 1, 4) 
        q, k, v = qkv.unbind(0) 
        attn = (q @ k.transpose(-2, -1)) * self.config.inv_sqrt_dk 
        attn = attn.softmax(dim=-1) 
        attn = self.attn_drop(attn) 
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) 
        x = self.proj(x) 
        x = self.proj_drop(x) 
        return x

class EncoderLayer(nn.Module): 
    def __init__(self, config: ViTConfig): 
        super().__init__() 
        self.attention = VitAttention(config) 
        self.mlp = VitMLP(config) 
        self.ln1 = nn.LayerNorm(config.hidden_size) 
        self.ln2 = nn.LayerNorm(config.hidden_size) 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = x + self.attention(self.ln1(x)) 
        x = x + self.mlp(self.ln2(x)) 
        return x    

class Encoder(nn.Module): 
    def __init__(self, config: ViTConfig): 
        super().__init__() 
        self.config = config 
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_layers)]) 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        for layer in self.layers: 
            x = layer(x) 
        return x

    