# Vision Transformer from Scratch

A minimal implementation of a Vision Transformer (ViT) from scratch in PyTorch.

## Repository Structure
```bash 
   vit-from-scratch/
   ├── vit_from_scratch/
   │   ├── model/
   │   │   ├── encoder.py
   │   │   └── vit.py
   │   ├── dataset.py
   │   └── utils.py
   ├── pyproject.toml
   ├── README.md
   ├── requirements.txt
   ├── test.py
   ├── train.py
   └── uv.lock

```

## Features

- **Model**: Full ViT implementation including Patch Embeddings, Multi-Head Self-Attention, MLP blocks, and Transformer Encoder layers.
- **Training**: Complete training loop with checkpointing, logging, and evaluation.
- **Dataset**: Hugging Face `datasets`, default set to `Beans`.
- **Configuration**: Can be  configured using `ViTConfig` dataclass.

## Installation & Usage

Can be run using either `uv` or regular Python `venv`.

### Using `uv` (Recommended)

Install **uv**: https://docs.astral.sh/uv/getting-started/installation/.

```bash
# run training directly
uv run train.py --dataset_name beans --epochs 5

# or sync dependencies first
uv sync
```

### Using standard `venv`

   ```bash
   # create and activate a virtual environment
   python3 -m venv venv
   source venv/bin/activate
   ```

   ```bash
   # install dependencies
   pip install -r requirements.txt
   ```

   ```bash
   # run training
   python3 train.py --dataset_name beans --epochs 5
   ```

## Args

- `--dataset_name`: Hugging Face dataset name (default: `beans`).
- `--batch_size`: Train batch size (default: `32`).
- `--epochs`: Number of training epochs (default: `10`).
- `--lr`: Learning rate (default: `1e-3`).
- `--save_dir`: Directory to save checkpoints to (default: `./checkpoints`).

## Testing

Evaluate a trained checkpoint on a test dataset:

```bash
# using uv
uv run test.py --checkpoint checkpoints/vit_epoch_5.pth --dataset_name beans

# using python
python3 test.py --checkpoint checkpoints/vit_epoch_5.pth --dataset_name beans
```
