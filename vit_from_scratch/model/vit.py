import torch 
import torch.nn as nn 
from ..config import ViTConfig
from .encoder import Encoder
from ..utils import LayerNorm

class VitEmbeddings(nn.Module): 
    def __init__(self, config: ViTConfig): 
        super().__init__() 
        self.config = config 
        self.patch_embed = nn.Conv2d(config.num_channels, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size) 
        self.position_embeddings = nn.Parameter(torch.zeros(1, config.num_patches + 1, config.hidden_size)) 
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) 
        self.dropout = nn.Dropout(config.embed_drop_rate) 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.patch_embed(x) 
        x = x.flatten(2).transpose(1, 2) 
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1) 
        x = x + self.position_embeddings 
        x = self.dropout(x) 
        return x
    
class VitBlock(nn.Module): 
    def __init__(self, config: ViTConfig): 
        super().__init__() 
        self.config = config 
        self.embeddings = VitEmbeddings(config) 
        self.encoder = Encoder(config) 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.embeddings(x) 
        x = self.encoder(x) 
        return x    

class Vit(nn.Module): 
    def __init__(self, config: ViTConfig): 
        super().__init__() 
        self.config = config 
        self.block = VitBlock(config) 
        self.norm = LayerNorm(config)
        self.head = nn.Linear(config.hidden_size, config.num_classes)
        
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x) 
        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.extract_features(x)
        x = self.head(x[:, 0])
        return x

    
    
    
    

    