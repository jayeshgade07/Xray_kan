import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import glob

NIH_DISEASES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 
    'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

class NIHChestXrayDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        if not os.path.exists(csv_file):
            print(f"[Warning] CSV not found at {csv_file}. Using dummy 10-item empty dataset.")
            self.data = pd.DataFrame(columns=['Image Index'] + NIH_DISEASES)
            self.is_dummy = True
            self.image_paths = {}
        else:
            self.data = pd.read_csv(csv_file)
            self.is_dummy = False
            
            # Dynamically map all image paths handles images_001/images/*.png or images/*.png
            self.image_paths = {}
            print(f"Mapping image paths in {data_dir}... (this may take a few seconds)")
            for file_path in glob.glob(os.path.join(data_dir, 'images_*', '**', '*.png'), recursive=True):
                self.image_paths[os.path.basename(file_path)] = file_path
            for file_path in glob.glob(os.path.join(data_dir, 'images', '*.png')):
                self.image_paths[os.path.basename(file_path)] = file_path
            
    def __len__(self):
        return 10 if self.is_dummy else len(self.data)

    def __getitem__(self, idx):
        if self.is_dummy:
            # Return blank image tensor for dry-run testing
            img = Image.new('RGB', (224, 224), color='black')
            labels = torch.zeros(14, dtype=torch.float32)
            if self.transform: img = self.transform(img)
            return img, labels

        row = self.data.iloc[idx]
        img_name = row['Image Index']
        full_img_path = self.image_paths.get(img_name)
        
        # Robust loading just in case image is missing during download phase
        if full_img_path is None or not os.path.exists(full_img_path):
            image = Image.new('RGB', (224, 224), color='black')
        else:
            try:
                image = Image.open(full_img_path).convert('RGB')
            except Exception:
                image = Image.new('RGB', (224, 224), color='black')
            
        labels = torch.tensor(row[NIH_DISEASES].values.astype(np.float32), dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, labels

def get_nih_dataloaders(data_dir="D:/Xray_Dataset/data", batch_size=32, num_workers=4):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = NIHChestXrayDataset(os.path.join(data_dir, 'train.csv'), data_dir, transform=train_transform)
    val_dataset = NIHChestXrayDataset(os.path.join(data_dir, 'val.csv'), data_dir, transform=test_transform)
    test_dataset = NIHChestXrayDataset(os.path.join(data_dir, 'test.csv'), data_dir, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
