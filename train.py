import argparse
import torch 
import torch.nn as nn
import os 
import math 
import random 
from tqdm import tqdm 
from vit_from_scratch.model.vit import Vit   
from vit_from_scratch.dataset import get_dataset 
from vit_from_scratch.utils import ViTConfig, AdamW
from torchvision import transforms
from torch.utils.data import DataLoader
from dotenv import load_dotenv
import wandb
from datetime import datetime
load_dotenv()

WANDB_API_KEY = os.getenv('WANDB_API_KEY', None)
WANDB_PROJECT_NAME = os.getenv('WANDB_PROJECT_NAME', 'vit-from-scratch')


def parse_args():
    parser = argparse.ArgumentParser(description='Train ViT from scratch')
    parser.add_argument('--dataset_name', type=str, default='beans', help='Name of the Hugging Face dataset (e.g. beans, cifar10)')
    parser.add_argument('--split', type=str, default='train', help='Split to evaluate on')
    parser.add_argument('--val_split', type=str, default='validation', help='Split to validate on')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--max_checkpoints', type=int, default=5, help='Maximum number of checkpoints to keep')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.save_dir = os.path.join(args.save_dir, run_id)
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Saving checkpoints to: {args.save_dir}")
    
    config = ViTConfig()
   
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    

    print(f"Loading dataset: {args.dataset_name} ({args.split} split)...")
    
    mode = "online" if WANDB_API_KEY else "disabled"
    print(f"Initializing WandB in {mode} mode")
    
    wandb.init(
        project=WANDB_PROJECT_NAME,
        mode=mode,
        config={
            "learning_rate": args.lr,
            "architecture": "ViT",
            "dataset": args.dataset_name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "image_size": config.image_size,
            "patch_size": config.patch_size,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
        }
    )
    
    dataset = get_dataset(args.dataset_name, split=args.split, transform=transform)
    val_dataset = get_dataset(args.dataset_name, split=args.val_split, transform=transform)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0) 
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    

    model = Vit(config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting training...")
    saved_checkpoints = []
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in progress_bar:
            images = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            wandb.log({"train_loss": loss.item(), "epoch": epoch})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
        
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Validation {epoch+1}/{args.epochs}"):
                images = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_dataloader)
        val_accuracy = 100 * correct / total
        print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
        
        wandb.log({
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
            "epoch": epoch
        })

        checkpoint_path = os.path.join(args.save_dir, f'vit_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state if hasattr(optimizer, 'state') else {}, 
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        saved_checkpoints.append(checkpoint_path)
        if len(saved_checkpoints) > args.max_checkpoints:
            oldest_checkpoint = saved_checkpoints.pop(0)
            if os.path.exists(oldest_checkpoint):
                os.remove(oldest_checkpoint)
                print(f"Removed old checkpoint: {oldest_checkpoint}")

if __name__ == '__main__':
    main()
