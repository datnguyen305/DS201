import numpy as np
import torch
from torch.utils.data import Dataset
import idx2numpy

def collate_fn(items: list[dict]) -> dict[torch.Tensor]:

    data = {
        "image": np.stack([item["image"] for item in items], axis = 0),
        "label": np.stack([item["label"] for item in items], axis = 0)
    }

    data = {
        "image": torch.tensor(data["image"], dtype=torch.float32),
        "label": torch.tensor(data["label"], dtype=torch.float32)
    }
    # data: ("image": Tensor(batch_size, 28, 28), "label": Tensor(batch_size))
    return data

class MNISTDataset(Dataset):
    def __init__(self, image_path: str, label_path: str): 
        self.images = idx2numpy.convert_from_file(image_path)
        self.labels = idx2numpy.convert_from_file(label_path)
        self.data = [
            {
                "image": image,
                "label": label
            }
            for image, label in zip(self.images, self.labels)
        ]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]