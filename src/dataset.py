from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

REPO_ROOT = Path(__file__).resolve().parent.parent

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(split: str = "train"):
    if split == "train":
        return T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class DRDataset(Dataset):
    def __init__(self, csv_path, split: str = "train", root_dir: Path = None):
        self.csv_path = Path(csv_path)
        self.split = split
        self.root_dir = Path(root_dir) if root_dir is not None else REPO_ROOT
        self.df = pd.read_csv(self.csv_path)
        self.transform = get_transforms(split)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_rel_path = row["image_path"]
        img_full_path = self.root_dir / img_rel_path
        
        image = Image.open(img_full_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
            
        label = int(row["severity"])
        return image, label

