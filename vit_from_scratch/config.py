from dataclasses import dataclass

@dataclass 
class ViTConfig: 
    image_size: int = 224 
    patch_size: int = 16 
    num_channels: int = 3 
    num_classes: int = 1000 
    hidden_size: int = 768 
    num_heads: int = 12 
    num_layers: int = 12 
    mlp_ratio: float = 4.0 
    qkv_bias: bool = True 
    qk_scale: Optional[float] = None 
    drop_rate: float = 0.0 
    attn_drop_rate: float = 0.0 
    drop_path_rate: float = 0.0     
    embed_drop_rate: float = 0.0 
    use_abs_pos: bool = True 
    use_rel_pos: bool = False 
    rel_pos_zero_init: bool = True 
    window_size: int = 14 
    global_attn_indexes: Optional[List[int]] = None 
    patch_norm: bool = True 
    use_checkpoint: bool = False 
    use_auth: bool = False 
    learning_rate: float = 1e-4 
    betas: Tuple[float, float] = (0.9, 0.999) 
    epsilon: float = 1e-8 

    @property
    def num_patches(self) -> int:
        assert self.image_size % self.patch_size == 0, "image_size must be divisible by patch_size"
        return (self.image_size // self.patch_size) ** 2

    @property
    def inv_sqrt_dk(self) -> float:
        head_dim = self.hidden_size // self.num_heads
        return head_dim ** -0.5


@dataclass
class TrainingConfig:
    batch_size: int = 64 
    num_epochs: int = 100 
    learning_rate: float = 1e-4 
    betas: Tuple[float, float] = (0.9, 0.999) 
    epsilon: float = 1e-8 
    weight_decay: float = 0.01 
    dataset_name: str = "AI-Lab-Makerere/beans" 
    dataset_image_col: str = "image" 
    dataset_label_col: str = "labels" 
    train_split: str = "train" 
    val_split: str = "validation" 
    test_split: str = "test" 
    save_dir: str = "checkpoints"
    max_checkpoints: int = 5
    seed: int = 42
