from datasets import load_dataset
from torchvision import transforms
from typing import Optional, Dict

def get_dataset(dataset_name: str, split: str = "train", transform: Optional[transforms.Compose] = None):
    
    dataset = load_dataset(dataset_name, split=split).select(range(10))
    
    def apply_transform(batch):
        if transform:
            if 'image' in batch:
                batch['pixel_values'] = [transform(img.convert("RGB")) for img in batch['image']]
            elif 'img' in batch:
                 batch['pixel_values'] = [transform(img.convert("RGB")) for img in batch['img']]
            else:
                 keys = [k for k in batch.keys() if 'img' in k or 'image' in k]
                 if keys:
                     batch['pixel_values'] = [transform(img.convert("RGB")) for img in batch[keys[0]]]
        
        if 'label' in batch:
            batch['labels'] = batch['label']
        elif 'labels' in batch:
            pass
            
        return {
            'pixel_values': batch['pixel_values'],
            'labels': batch['labels'] if 'labels' in batch else batch.get('label')
        }

    if transform:
        dataset.set_transform(apply_transform)
        
    return dataset
