import torch
from torch.utils.data import Dataset
import os
import cv2 

def collate_fn(samples: list[dict]) -> dict:
    #images = [sample['image'].permute(1, 2, -1).unsqueeze(0) for sample in samples] 
    images = [sample['image'] for sample in samples]
    labels = [sample['label'] for sample in samples] 

    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels)
    
    return {
        'image': images,
        'label': labels
    }

class VinaFood(Dataset):
    #def __init__(self, path: str):
    def __init__(self, path: str, image_size: tuple[int]):
        super().__init__()
    
        self.image_size = image_size
        self.label2idx = {}
        self.idx2label = {}
        self.data: list[dict] = self.load_data(path)
        
    def load_data(self, path):
        data = []
        label_id = 0
        print(f"Loading data from: {path}")
        for folder in os.listdir(path):
            label = folder
            if label not in self.label2idx:
                self.label2idx[label] = label_id
                label_id += 1
            folder_path = os.path.join(path, folder)
            print(f"Processing folder: {folder} (label_id: {self.label2idx[label]})")
            
            for image_file in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_file)
                image = cv2.imread(image_path)
                data.append({
                    'image': image,
                    'label': label
                })

        self.idx2label = {id: label for label, id in self.label2idx.items()}
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        
        image = item['image']
        label = item['label']
        
        # image = cv2.resize(image, (224, 224))
        # label_id = self.label2idx[label]
        
        image = cv2.resize(image, self.image_size)
        # Convert to RGB if needed (OpenCV loads in BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert to tensor once
        image = torch.tensor(image, dtype=torch.float32).permute(2,0,1) / 255.0
        return {
            'image': image,
            'label': self.label2idx[label]
        }
    
    
    