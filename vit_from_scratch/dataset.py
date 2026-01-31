from datasets import load_dataset
from torchvision import transforms
from typing import Optional, Dict

def get_dataset(dataset_name: str, split: str = "train", image_col: str = "image", label_col: str = "labels", transform: Optional[transforms.Compose] = None, sample_size: Optional[int] = None):
    
    dataset = load_dataset(dataset_name, split=split)
    
    def apply_transform(batch):
        if image_col in batch:
            if transform:
                batch['pixel_values'] = [transform(img.convert("RGB")) for img in batch[image_col]]
            else:
                batch['pixel_values'] = batch[image_col]
        else:
            raise ValueError(f"Image column {image_col} not found in dataset")
        
        if label_col in batch:
            batch['labels'] = batch[label_col]
        else:
            raise ValueError(f"Label column {label_col} not found in dataset")
            
        return {
            'pixel_values': batch['pixel_values'],
            'labels': batch['labels']
        }
    
    if sample_size:
        dataset = dataset.select(range(sample_size))
   
    dataset.set_transform(apply_transform)
    
    return dataset
