import argparse
import torch 
import torch.nn as nn
import os 
from tqdm import tqdm 
from vit_from_scratch.model.vit import Vit   
from vit_from_scratch.dataset import get_dataset 
from vit_from_scratch.utils import ViTConfig
from torchvision import transforms
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate ViT')
    parser.add_argument('--dataset_name', type=str, default='beans', help='Name of the Hugging Face dataset')
    parser.add_argument('--split', type=str, default='test', help='Split to evaluate on')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    config = ViTConfig()
    
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    print(f"Loading dataset: {args.dataset_name} ({args.split} split)...")
    dataset = get_dataset(args.dataset_name, split=args.split, transform=transform)
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = Vit(config).to(device)
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file {args.checkpoint} not found.")
        return
        
    print(f"Loading checkpoint {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    correct = 0
    total = 0
    
    print("Starting evaluation...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print(f'Accuracy on {args.dataset_name}: {accuracy:.2f}%')

if __name__ == '__main__':
    main()

