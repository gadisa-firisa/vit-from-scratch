import torch 
import torch.nn as nn 
from dataclasses import dataclass 
from typing import List, Optional, Tuple
from vit_from_scratch.config import ViTConfig

class LayerNorm(nn.Module): 
    def __init__(self, config: ViTConfig): 
        super().__init__() 
        self.config = config 
        self.gamma = nn.Parameter(torch.ones(config.hidden_size)) 
        self.beta = nn.Parameter(torch.zeros(config.hidden_size)) 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        inv_std = torch.rsqrt(var + self.config.epsilon)
        return (x - mean) * inv_std * self.gamma + self.beta

class GELU(nn.Module): 
    def __init__(self, config: ViTConfig, approx: str = 'none'): 
        super().__init__() 
        self.config = config 
        self.approx = approx 
        assert approx in ['none', 'tanh'] 

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        if self.approx == 'tanh': 
            return 0.5 * x * (1.0 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3.0)))) 
        else: 
            return x * 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0)))) 

class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 0: {betas[0]}')
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 1: {betas[1]}')
        if eps < 0.0:
            raise ValueError(f'Invalid epsilon value: {eps}')
        
        self.defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.param_groups = []
        
        if isinstance(params, torch.Tensor):
            params = [params]
        params = list(params)
        
        if len(params) > 0 and isinstance(params[0], dict):
            for group in params:
                self.add_param_group(group)
        else:
            self.add_param_group({'params': params})
        
        self.state = {}

    def add_param_group(self, param_group):
        params = param_group['params']
        if isinstance(params, torch.Tensor):
            param_group['params'] = [params]
        
        for key, value in self.defaults.items():
            param_group.setdefault(key, value)
        
        self.param_groups.append(param_group)

    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                if p not in self.state:
                    self.state[p] = {
                        'step': 0,
                        'exp_avg': torch.zeros_like(p),
                        'exp_avg_sq': torch.zeros_like(p)
                    }
                
                state = self.state[p]
                state['step'] += 1
                
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                step = state['step']
                
                p.data.mul_(1 - lr * weight_decay)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
                
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
